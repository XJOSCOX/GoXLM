"""Versioned training, deterministic holdout scoring and checkpoint validation."""
from contextlib import nullcontext
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

from .common import ROOT, git_provenance, publish, source_fingerprint, verify_baseline, write_json
from .data import affected_documents, catalog_lock, load_release, revoked
from .model import (EXPERIMENT_VERSION, MODEL, MODEL_VERSION, TOKENIZER, experiment_config,
                    new_model, windows)

LEGACY_CHECKPOINT = "goxlm-checkpoint-0.2.1"
CHECKPOINT = "goxlm-checkpoint-0.2.2"
EXPERIMENT_CHECKPOINT = "goxlm-checkpoint-0.4.0-experimental"
CHECKPOINT_SCHEMAS = (LEGACY_CHECKPOINT, CHECKPOINT, EXPERIMENT_CHECKPOINT)
PRECISIONS = ("fp32", "bf16")


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
    return source_fingerprint(ROOT, paths)


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
            pairs = list(windows(row["text"], model.config.context_length))
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


def check_precision(precision, device):
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported training precision: {precision}")
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("BF16 training requires native CUDA BF16 support")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("BF16 training requires native CUDA BF16 support")


def checkpoint_precision(checkpoint):
    precision = checkpoint.get("precision", "fp32" if checkpoint["schema"] == LEGACY_CHECKPOINT else None)
    if precision not in PRECISIONS or (checkpoint["schema"] == LEGACY_CHECKPOINT and precision != "fp32"):
        raise ValueError("Invalid checkpoint precision")
    return precision


def verify_float32_state(tensors, label):
    by_device = {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError(f"Checkpoint {label} tensors must be finite float32")
        by_device.setdefault(tensor.device, []).append(tensor)
    for group in by_device.values():
        if not torch.stack([tensor.isfinite().all() for tensor in group]).all().item():
            raise ValueError(f"Checkpoint {label} tensors must be finite float32")


def verify_adamw(optimizer, updates):
    parameters = [key for group in optimizer["param_groups"] for key in group["params"]]
    state = optimizer["state"]
    if not parameters or len(parameters) != len(state) or set(parameters) != set(state):
        raise ValueError("Checkpoint AdamW state does not match its parameter groups")
    moments = []
    for entry in state.values():
        step = entry["step"]
        if (not isinstance(step, torch.Tensor) or step.dtype != torch.float32
                or step.ndim != 0 or step.item() != updates):
            raise ValueError("Checkpoint AdamW state and optimizer update count disagree")
        moments.extend([entry["exp_avg"], entry["exp_avg_sq"]])
    verify_float32_state(moments, "optimizer")


def load_checkpoint(path, device="cpu", *, allowed_schemas=CHECKPOINT_SCHEMAS):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    schema = checkpoint.get("schema")
    if schema not in allowed_schemas:
        raise ValueError("Unsupported checkpoint schema for this reader")
    architecture = checkpoint.get("architecture")
    model_config = checkpoint.get("model_config")
    if schema == EXPERIMENT_CHECKPOINT and isinstance(architecture, str) and isinstance(model_config, dict):
        expected_model = experiment_config(architecture, model_config.get("context_length"))
        expected_version = EXPERIMENT_VERSION
    elif schema in (LEGACY_CHECKPOINT, CHECKPOINT) and architecture is None:
        expected_model = MODEL
        expected_version = MODEL_VERSION
    else:
        raise ValueError("Unsupported checkpoint schema or experimental architecture")
    if (checkpoint.get("model_version") != expected_version or model_config != expected_model
            or checkpoint.get("tokenizer") != TOKENIZER):
        raise ValueError("Unsupported checkpoint, model or tokenizer configuration")
    steps, history = checkpoint["step"], checkpoint["history"]
    if type(steps) is not int or steps < 1 or [r["step"] for r in history] != list(
            range(1, steps + 1)):
        raise ValueError("Checkpoint step and history disagree")
    checkpoint_precision(checkpoint)
    RunConfig(**checkpoint["run_config"])
    for field in ("dataset_version", "code_revision"):
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint[field]):
            raise ValueError(f"Missing or invalid {field}")
    with torch.random.fork_rng(devices=[]):
        model = new_model(expected_model)
    state = checkpoint["model"]
    if not torch.equal(state["lm_head.weight"], state["token_embedding.weight"]):
        raise ValueError("Checkpoint has inconsistent tied weights")
    verify_float32_state(state.values(), "model")
    model.load_state_dict(state, strict=True)
    for field in ("optimizer", "rng", "environment", "hardware"):
        if not isinstance(checkpoint[field], dict):
            raise ValueError(f"Invalid checkpoint {field}")
    if schema != LEGACY_CHECKPOINT:
        updates = checkpoint.get("optimizer_updates")
        if type(updates) is not int or updates != steps:
            raise ValueError("Checkpoint step and optimizer update count disagree")
        verify_adamw(checkpoint["optimizer"], updates)
    rng = checkpoint["rng"]
    torch.Generator().set_state(rng["torch"])
    torch.Generator().set_state(rng["sampler"])
    random.Random().setstate(rng["python"])
    return model.to(device).eval(), checkpoint


def train(catalog, release, output, steps, device, config, resume=None, checkpoint_every=1,
          *, precision="fp32", architecture=None, context_length=None):
    device = torch.device(device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    check_precision(precision, device)
    if architecture is None and context_length is not None:
        raise ValueError("A custom context length requires an experimental architecture")
    model_config = (MODEL if architecture is None else
                    experiment_config(architecture, 128 if context_length is None else context_length))
    experimental = architecture is not None
    artifact_version = "0.4" if experimental else "0.2"
    verify_baseline()
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    dataset = load_release(catalog, release)
    torch.use_deterministic_algorithms(True)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    sampler = torch.Generator().manual_seed(config.seed)
    env = environment(device)
    identity = {"schema": EXPERIMENT_CHECKPOINT if experimental else CHECKPOINT,
                "model_version": EXPERIMENT_VERSION if experimental else MODEL_VERSION,
                "model_config": model_config,
                "run_config": asdict(config), "dataset_version": release.stem,
                "tokenizer": TOKENIZER, "code_revision": revision(), "environment": env,
                "precision": precision}
    if experimental:
        identity["architecture"] = architecture
    start, history = 0, []
    if resume is None:
        model = new_model(model_config).to(device)
    else:
        model, saved = load_checkpoint(resume, device)
        if checkpoint_precision(saved) != precision:
            raise ValueError("Cannot change training precision when resuming a checkpoint")
        previous_identity = dict(identity, schema=saved["schema"])
        if any(saved.get(key) != value for key, value in previous_identity.items() if key != "precision"):
            raise ValueError("Resume requires matching data, configuration, code and environment")
        start, history = saved["step"], saved["history"]
    if steps <= start:
        raise ValueError("Target step must be greater than saved step")
    train_windows = [pair for row in dataset["documents"] if row["split"] == "train"
                     for pair in windows(row["text"], model_config["context_length"])]
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

    provenance = {"git": git_provenance()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / f"run.{artifact_version}.json", {**identity, **provenance, "target_step": steps,
                                       "resumed_from_step": start, "checkpoint_every": checkpoint_every})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    updates = start
    train_seconds = validation_seconds = checkpoint_seconds = 0.0
    started = time.perf_counter()
    cpu_start = time.process_time()
    trained_bytes = 0
    for step in range(start + 1, steps + 1):
        step_start = time.perf_counter()
        indices = torch.randint(len(train_windows), (config.batch_size,), generator=sampler)
        x, y = make_batch([train_windows[i] for i in indices.tolist()], device)
        optimizer.zero_grad(set_to_none=True)
        with (torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()):
            _, loss = model(x, y)
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                              error_if_nonfinite=True)
        optimizer.step()
        updates += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_start
        train_seconds += elapsed
        count = (y != -100).sum().item()
        trained_bytes += count
        record = {"step": step, "train_loss": loss.item(), "gradient_norm": norm.item(),
                  "train_bytes": count, "train_seconds": elapsed, "bytes_per_second": count / elapsed}
        history.append(record)
        if step % checkpoint_every and step != steps:
            continue
        validation_start = time.perf_counter()
        record["validation"] = evaluate(model, validation, device, config.batch_size)
        validation_seconds += time.perf_counter() - validation_start
        checkpoint_start = time.perf_counter()
        elapsed_run = time.perf_counter() - started
        hardware = {"segment_wall_seconds": elapsed_run,
                    "segment_process_cpu_seconds": time.process_time() - cpu_start,
                    "segment_training_target_bytes": trained_bytes,
                    "segment_end_to_end_bytes_per_second": trained_bytes / elapsed_run,
                    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        if device.type == "cuda":
            hardware.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        state, adamw = model.state_dict(), optimizer.state_dict()
        verify_float32_state(state.values(), "model")
        verify_adamw(adamw, updates)
        checkpoint = {**identity, **provenance, "step": step, "optimizer_updates": updates,
                      "model": state, "optimizer": adamw, "history": history,
                      "hardware": hardware, "rng": {"python": random.getstate(),
                      "torch": torch.get_rng_state(), "sampler": sampler.get_state(),
                      "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}}
        path = output / f"goxlm-{artifact_version}-step-{step:08d}.pt"
        with catalog_lock(catalog):
            if affected_documents(dataset["documents"], revoked(catalog)):
                raise ValueError("Rights revoked during run. Checkpoint publication blocked.")
            publish(path, lambda stream: torch.save(checkpoint, stream))
        checkpoint_seconds += time.perf_counter() - checkpoint_start
    write_json(output / f"metrics.{artifact_version}.json", {
        "schema": "goxlm-segment-metrics-0.4.0" if experimental else "goxlm-segment-metrics-0.2.2", "precision": precision,
        "first_step": start + 1, "last_step": steps, "optimizer_updates": updates - start,
        "training_target_bytes": trained_bytes, "training_seconds": train_seconds,
        "validation_seconds": validation_seconds, "checkpoint_seconds": checkpoint_seconds,
        "wall_seconds": time.perf_counter() - started,
        "hardware": hardware,
    })
    return path
