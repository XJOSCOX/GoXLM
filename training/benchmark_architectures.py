"""Measure the experimental GoXLM trainer using bounded, generated fixtures only."""
import json
import math
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
import time

import torch

from goxlm_v02.run import EXPERIMENT_CHECKPOINT, load_checkpoint
from goxlm_v02.test_v02 import PipelineTests, build

from .bf16_validation import exact, numerical_history

STEPS = 24
WARMUP = 4
CONTEXT = 128


def segment(fixture, release, architecture, precision, name, steps=STEPS, resume=None):
    output = fixture.root / name
    args = [sys.executable, "-B", "-m", "goxlm_v02", "train", str(fixture.catalog),
            str(release), str(output), "--steps", str(steps), "--checkpoint-every", str(STEPS),
            "--device", "cuda", "--architecture", architecture, "--context-length", str(CONTEXT),
            "--precision", precision]
    if resume:
        args.extend(["--resume", str(resume)])
    # Separate processes isolate CUDA allocator peaks and exercise actual checkpoint loading.
    result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=180,
                            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    return Path(result.stdout.strip())


def read_run(path):
    started = time.perf_counter()
    model, state = load_checkpoint(path)
    load_seconds = time.perf_counter() - started
    metrics = json.loads((path.parent / "metrics.0.4.json").read_text())
    assert state["schema"] == EXPERIMENT_CHECKPOINT
    assert model.config.context_length == CONTEXT
    history = state["history"]
    assert state["optimizer_updates"] == len(history)
    assert all(math.isfinite(row[field]) for row in history
               for field in ("train_loss", "gradient_norm"))
    with torch.inference_mode():
        logits, _ = model(torch.zeros((1, CONTEXT), dtype=torch.long))
    assert torch.isfinite(logits).all().item()
    return model, state, metrics, load_seconds


def main():
    fixture = PipelineTests()
    fixture.setUp()
    try:
        rng = random.Random(42)
        for i in range(24):
            fixture.submit(f"doc{i}", "".join(rng.choices("abcdefghijklmnopqrstuvwxyz012345 ", k=512)))
        release = build(fixture.catalog)
        profiles = []
        resume_reference = None
        environment = None
        provenance = None
        for architecture in ("baseline", "10m", "25m"):
            for precision in ("fp32", "bf16"):
                path = segment(fixture, release, architecture, precision, f"{architecture}-{precision}")
                model, state, metrics, load_seconds = read_run(path)
                history = state["history"]
                steps = history[WARMUP:]
                latencies = sorted(row["train_seconds"] * 1000 for row in steps)
                profiles.append({
                    "architecture": architecture, "precision": precision,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "updates": state["optimizer_updates"],
                    "tokens_per_second": sum(row["train_bytes"] for row in steps)
                    / sum(row["train_seconds"] for row in steps),
                    "p50_ms": median(latencies), "p95_ms": latencies[math.ceil(len(latencies) * .95) - 1],
                    "allocated_bytes": metrics["hardware"]["gpu_peak_allocated_bytes"],
                    "reserved_bytes": metrics["hardware"]["gpu_peak_reserved_bytes"],
                    "gradient_norm_min": min(row["gradient_norm"] for row in history),
                    "gradient_norm_max": max(row["gradient_norm"] for row in history),
                    "loss_first": history[0]["train_loss"], "loss_last": history[-1]["train_loss"],
                    "validation_seconds": metrics["validation_seconds"],
                    "checkpoint_save_seconds": metrics["checkpoint_seconds"],
                    "checkpoint_load_seconds": load_seconds, "checkpoint_bytes": path.stat().st_size,
                })
                if architecture == "10m" and precision == "bf16":
                    resume_reference = state
                environment, provenance = state["environment"], state["git"]
                del model, state

        first = segment(fixture, release, "10m", "bf16", "resume-first", steps=STEPS // 2)
        final = segment(fixture, release, "10m", "bf16", "resume-final", resume=first)
        _, resumed, _, _ = read_run(final)
        resume_checks = {key: exact(resume_reference[key], resumed[key])
                         for key in ("model", "optimizer", "rng", "optimizer_updates")}
        # The midpoint save adds a validation record. It is not a training update.
        def history_without_validation(state):
            return [{k: v for k, v in row.items() if k != "validation"} for row in numerical_history(state)]

        resume_checks["training_history"] = exact(history_without_validation(resume_reference),
                                                  history_without_validation(resumed))
        resume_checks["final_validation"] = (resume_reference["history"][-1]["validation"]
                                             == resumed["history"][-1]["validation"])
        print(json.dumps({"schema": "goxlm-architecture-smoke-1", "steps": STEPS, "warmup": WARMUP,
                          "context_length": CONTEXT, "batch_size": 8, "profiles": profiles,
                          "environment": environment, "git": provenance, "resume": resume_checks},
                         indent=2, sort_keys=True, allow_nan=False))
        return 0 if all(resume_checks.values()) else 1
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    raise SystemExit(main())
