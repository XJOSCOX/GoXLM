"""Run bounded FP32/BF16 CUDA training and fresh-process resume on synthetic data."""
import json
import math
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys

import torch

from goxlm_v02 import test_v02
from goxlm_v02.model import MODEL


STEPS = 256
MIDPOINT = 128
SEED = 42


def run_segment(fixture, release, batch_size, precision, name, steps=STEPS, resume=None):
    output = fixture.root / name
    config = output.with_suffix(".config.json")
    config.write_text(json.dumps({"batch_size": batch_size, "seed": SEED}), encoding="utf-8")
    args = [sys.executable, "-B", "-m", "goxlm_v02", "train", str(fixture.catalog),
            str(release), str(output), "--steps", str(steps), "--checkpoint-every",
            str(MIDPOINT), "--precision", precision, "--device", "cuda", "--config", str(config)]
    if resume is not None:
        args.extend(["--resume", str(resume)])
    result = subprocess.run(args, check=True, stdout=subprocess.PIPE, text=True, timeout=300,
                            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    return Path(result.stdout.strip())


def read_checkpoint(path):
    from goxlm_v02.run import load_checkpoint

    _, checkpoint = load_checkpoint(path)
    return checkpoint


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def numerical_history(checkpoint):
    return [{key: value for key, value in row.items()
             if key not in {"train_seconds", "bytes_per_second"}} for row in checkpoint["history"]]


def resume_report(direct, resumed):
    checks = {
        "model": exact(direct["model"], resumed["model"]),
        "optimizer": exact(direct["optimizer"], resumed["optimizer"]),
        "python_rng": exact(direct["rng"]["python"], resumed["rng"]["python"]),
        "cpu_rng": exact(direct["rng"]["torch"], resumed["rng"]["torch"]),
        "cuda_rng": exact(direct["rng"]["cuda"], resumed["rng"]["cuda"]),
        "sampler": exact(direct["rng"]["sampler"], resumed["rng"]["sampler"]),
        "numerical_history": exact(numerical_history(direct), numerical_history(resumed)),
        "optimizer_updates": direct["optimizer_updates"] == resumed["optimizer_updates"] == STEPS,
    }
    delta = max((value - resumed["model"][name]).abs().max().item()
                for name, value in direct["model"].items())
    optimizer_delta = max(
        (value - resumed["optimizer"]["state"][key][field]).abs().max().item()
        for key, entry in direct["optimizer"]["state"].items() for field, value in entry.items()
    )
    return {"checks": checks, "max_model_abs_delta": delta,
            "max_optimizer_abs_delta": optimizer_delta,
            "max_train_loss_delta": max(abs(a["train_loss"] - b["train_loss"])
                                       for a, b in zip(direct["history"], resumed["history"])),
            "optimizer_updates": resumed["optimizer_updates"],
            "exact": all(checks.values())}


def performance(path, checkpoint):
    metrics = json.loads((path.parent / "metrics.0.2.json").read_text())
    history = checkpoint["history"]
    latencies = sorted(row["train_seconds"] * 1000 for row in history)
    stable = (checkpoint["optimizer_updates"] == STEPS
              and metrics["optimizer_updates"] == STEPS
              and all(math.isfinite(row["train_loss"]) and math.isfinite(row["gradient_norm"])
                      for row in history))
    return {
        "tokens_per_second": metrics["training_target_bytes"] / metrics["training_seconds"],
        "step_ms_p50": median(latencies),
        "step_ms_p95": latencies[math.ceil(len(latencies) * 0.95) - 1],
        "peak_allocated_bytes": metrics["hardware"]["gpu_peak_allocated_bytes"],
        "peak_reserved_bytes": metrics["hardware"]["gpu_peak_reserved_bytes"],
        "validation_seconds": metrics["validation_seconds"],
        "checkpoint_seconds": metrics["checkpoint_seconds"],
        "elapsed_seconds": metrics["wall_seconds"],
        "optimizer_updates": checkpoint["optimizer_updates"],
        "stable": stable,
        "training_loss": [row["train_loss"] for row in history],
        "gradient_norm": [row["gradient_norm"] for row in history],
        "validation_loss": {row["step"]: row["validation"]["overall"]["loss"]
                            for row in history if "validation" in row},
    }


def validate():
    fixture = test_v02.PipelineTests()
    fixture.setUp()
    try:
        rng = random.Random(SEED)
        alphabet = "abcdefghijklmnopqrstuvwxyz012345 "
        for i in range(24):
            text = "".join(rng.choices(alphabet, k=4 * MODEL["context_length"]))
            fixture.submit(f"doc{i}", text)
        release = test_v02.build(fixture.catalog)
        profiles = {}
        direct = None
        for batch_size in (8, 32):
            # Alternate the order of the paired cases.
            modes = ("fp32", "bf16") if batch_size == 8 else ("bf16", "fp32")
            profiles[batch_size] = {}
            for precision in modes:
                print(f"Synthetic CUDA check: batch {batch_size}, {precision}, {STEPS} steps",
                      file=sys.stderr, flush=True)
                path = run_segment(fixture, release, batch_size, precision, f"b{batch_size}-{precision}")
                checkpoint = read_checkpoint(path)
                profiles[batch_size][precision] = performance(path, checkpoint)
                if batch_size == 32 and precision == "bf16":
                    direct = checkpoint
        print("Synthetic CUDA check: fresh-process BF16 checkpoint resume",
              file=sys.stderr, flush=True)
        half = run_segment(fixture, release, 32, "bf16", "bf16-first-half", steps=MIDPOINT)
        resumed_path = run_segment(fixture, release, 32, "bf16", "bf16-second-half", resume=half)
        resumed = read_checkpoint(resumed_path)
        return {"schema": "goxlm-bf16-validation-1", "steps": STEPS, "resume_step": MIDPOINT,
                "seed": SEED, "context_length": MODEL["context_length"],
                "dataset_version": release.stem, "environment": direct["environment"],
                "git": direct["git"], "code_revision": direct["code_revision"],
                "profiles": profiles, "resume_batch32_bf16": resume_report(direct, resumed)}
    finally:
        fixture.doCleanups()


def main():
    report = validate()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    stable = all(result["stable"] for modes in report["profiles"].values() for result in modes.values())
    return 0 if stable and report["resume_batch32_bf16"]["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
