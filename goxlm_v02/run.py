"""Versioned training, deterministic holdout scoring and checkpoint validation."""
from dataclasses import asdict, dataclass
import math
import os
import platform
from pathlib import Path
import random
import re
import resource
import subprocess
import time

import torch
import torch.nn.functional as F

from .common import ROOT, canonical, digest, publish, verify_baseline, write_json
from .data import affected_documents, catalog_lock, load_release, revoked
from .model import MODEL, MODEL_VERSION, TOKENIZER, new_model, windows

CHECKPOINT = "goxlm-checkpoint-0.2.1"


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Invalid seed")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Invalid batch size")
        if (not all(math.isfinite(value) for value in
                    (self.learning_rate, self.weight_decay, self.grad_clip))
                or min(self.learning_rate, self.grad_clip) <= 0 or self.weight_decay < 0):
            raise ValueError("Invalid optimizer configuration")


def revision():
    paths = [ROOT / "model" / "transformer.py", *sorted(Path(__file__).parent.glob("*.py"))]
    return digest(canonical({path.relative_to(ROOT).as_posix(): digest(path.read_bytes())
                             for path in paths}))


def environment(device):
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
              "device": str(device), "cpu_threads": torch.get_num_threads(),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(gpu=props.name, gpu_total_bytes=props.total_memory,
                      gpu_compute_capability=[props.major, props.minor])
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=3)
            report["gpu_driver_versions"] = sorted(set(result.stdout.splitlines()))
        except (OSError, subprocess.SubprocessError):
            report["gpu_driver_versions"] = None
    return report


def make_batch(pairs, device):
    x, y = zip(*pairs)
    return torch.tensor(x, device=device), torch.tensor(y, device=device)


@torch.inference_mode()
def evaluate(model, rows, device, batch_size=8):
    previous = model.training
    model.eval()
    by_domain = {}
    try:
        for row in rows:
            total, count = by_domain.get(row["domain"], (0.0, 0))
            pairs = list(windows(row["text"]))
            for start in range(0, len(pairs), batch_size):
                x, y = make_batch(pairs[start:start + batch_size], device)
                logits, _ = model(x)
                total += F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                         reduction="sum").item()
                count += (y != -100).sum().item()
            by_domain[row["domain"]] = (total, count)
    finally:
        model.train(previous)

    def score(total, count):
        if count == 0 or not math.isfinite(total):
            raise ValueError("Cannot score an empty or nonfinite evaluation")
        return {"bytes": count, "loss": total / count, "bits_per_byte": total / count / math.log(2)}

    return {"overall": score(sum(v[0] for v in by_domain.values()),
                             sum(v[1] for v in by_domain.values())),
            "domains": {key: score(*value) for key, value in sorted(by_domain.items())}}


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if (checkpoint.get("schema") != CHECKPOINT
            or checkpoint.get("model_version") != MODEL_VERSION
            or checkpoint.get("model_config") != MODEL
            or checkpoint.get("tokenizer") != TOKENIZER):
        raise ValueError("Unsupported checkpoint, model or tokenizer configuration")
    steps, history = checkpoint["step"], checkpoint["history"]
    if type(steps) is not int or steps < 1 or [r["step"] for r in history] != list(
            range(1, steps + 1)):
        raise ValueError("Checkpoint step and history disagree")
    RunConfig(**checkpoint["run_config"])
    for field in ("dataset_version", "code_revision"):
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint[field]):
            raise ValueError(f"Missing or invalid {field}")
    with torch.random.fork_rng(devices=[]):
        model = new_model()
    state = checkpoint["model"]
    if not torch.equal(state["lm_head.weight"], state["token_embedding.weight"]):
        raise ValueError("Checkpoint has inconsistent tied weights")
    if any(tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()
           for tensor in state.values()):
        raise ValueError("Checkpoint model tensors must be finite float32")
    model.load_state_dict(state, strict=True)
    for field in ("optimizer", "rng", "environment", "hardware"):
        if not isinstance(checkpoint[field], dict):
            raise ValueError(f"Invalid checkpoint {field}")
    rng = checkpoint["rng"]
    torch.Generator().set_state(rng["torch"])
    torch.Generator().set_state(rng["sampler"])
    random.Random().setstate(rng["python"])
    return model.to(device).eval(), checkpoint


def train(catalog, release, output, steps, device, config, resume=None, checkpoint_every=1):
    verify_baseline()
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    dataset = load_release(catalog, release)
    device = torch.device(device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.get_device_properties(device)
    torch.use_deterministic_algorithms(True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    sampler = torch.Generator().manual_seed(config.seed)
    env = environment(device)
    identity = {"schema": CHECKPOINT, "model_version": MODEL_VERSION, "model_config": MODEL,
                "run_config": asdict(config), "dataset_version": release.stem,
                "tokenizer": TOKENIZER, "code_revision": revision(), "environment": env}
    start, history = 0, []
    if resume is None:
        model = new_model().to(device)
    else:
        model, saved = load_checkpoint(resume, device)
        if any(saved[key] != value for key, value in identity.items()):
            raise ValueError("Resume requires matching data, configuration, code and environment")
        start, history = saved["step"], saved["history"]
    if steps <= start:
        raise ValueError("Target step must be greater than saved step")
    train_windows = [pair for row in dataset["documents"] if row["split"] == "train"
                     for pair in windows(row["text"])]
    validation = [row for row in dataset["documents"] if row["split"] == "validation"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, foreach=False)
    if resume is not None:
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["rng"]["python"])
        torch.set_rng_state(saved["rng"]["torch"])
        sampler.set_state(saved["rng"]["sampler"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])

    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.0.2.json", {**identity, "target_step": steps,
                                       "resumed_from_step": start, "checkpoint_every": checkpoint_every})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    started = time.perf_counter()
    cpu_start = time.process_time()
    trained_bytes = 0
    for step in range(start + 1, steps + 1):
        step_start = time.perf_counter()
        indices = torch.randint(len(train_windows), (config.batch_size,), generator=sampler)
        x, y = make_batch([train_windows[i] for i in indices.tolist()], device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                              error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_start
        count = (y != -100).sum().item()
        trained_bytes += count
        record = {"step": step, "train_loss": loss.item(), "gradient_norm": norm.item(),
                  "train_bytes": count, "train_seconds": elapsed, "bytes_per_second": count / elapsed}
        history.append(record)
        if step % checkpoint_every and step != steps:
            continue
        record["validation"] = evaluate(model, validation, device, config.batch_size)
        elapsed_run = time.perf_counter() - started
        hardware = {"segment_wall_seconds": elapsed_run,
                    "segment_process_cpu_seconds": time.process_time() - cpu_start,
                    "segment_training_target_bytes": trained_bytes,
                    "segment_end_to_end_bytes_per_second": trained_bytes / elapsed_run,
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        if device.type == "cuda":
            hardware.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        checkpoint = {**identity, "step": step, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "history": history,
                      "hardware": hardware, "rng": {"python": random.getstate(),
                      "torch": torch.get_rng_state(), "sampler": sampler.get_state(),
                      "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}}
        path = output / f"goxlm-0.2-step-{step:08d}.pt"
        with catalog_lock(catalog):
            if affected_documents(dataset["documents"], revoked(catalog)):
                raise ValueError("Rights revoked during run. Checkpoint publication blocked.")
            publish(path, lambda stream: torch.save(checkpoint, stream))
    return path
