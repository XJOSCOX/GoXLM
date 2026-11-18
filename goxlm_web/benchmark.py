"""Bounded paired CUDA benchmark for the hash-pinned GoXLM-0.1 checkpoint."""
import json
import math
from pathlib import Path
from statistics import median
import time

import torch

from .engine import Completion, Settings, load_local, resolve_device


def trial(model, device, tokens, steps, strategy):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    run = Completion(model, tokens, device, 0.8, 1, strategy)
    times = []
    for _ in range(steps):
        run.step()
        times.append(time.perf_counter())
    intervals = sorted((b - a) * 1000 for a, b in zip(times, times[1:]))
    report = {"first_token_ms": (times[0] - start) * 1000,
              "sustained_tokens_per_second": (steps - 1) / (times[-1] - times[0]),
              "p95_token_ms": intervals[math.ceil(.95 * len(intervals)) - 1],
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "cache_rebuilds": run.cache.rebuilds if run.cache else 0}
    generated = run.tokens[len(tokens):]
    run.close()
    return report, generated


def main():
    torch.set_num_threads(1)
    device = resolve_device("cuda")
    entry = Settings.model_validate_json(Path(__file__).with_name("models.example.json").read_bytes()).models[0]
    started = time.perf_counter()
    model = load_local(entry, device.torch_device)
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - started) * 1000
    prompt = ("The properties of a material depend on its composition and structure. " * 3).encode()
    profiles = []
    for prompt_length, steps in ((16, 64), (64, 64), (120, 8), (120, 64)):
        tokens = list(prompt[:prompt_length])
        # Warm the shape-specific kernels outside the five paired measurements.
        for strategy in ("full", "cached"):
            trial(model, device.torch_device, tokens, steps, strategy)
        measurements = {"full": [], "cached": []}
        for i in range(5):
            generated = {}
            order = ("full", "cached") if i % 2 == 0 else ("cached", "full")
            for strategy in order:
                report, generated[strategy] = trial(model, device.torch_device, tokens, steps, strategy)
                measurements[strategy].append(report)
            if generated["full"] != generated["cached"]:
                raise RuntimeError("Cached greedy completions differ from full forward")
        profiles.append({"prompt_tokens": prompt_length, "new_tokens": steps,
                         "greedy_exact": True, "measurements": measurements,
                         "median": {key: {k: median(r[k] for r in rows) for k in rows[0]}
                                    for key, rows in measurements.items()}})
    print(json.dumps({"gpu": device.label, "torch": str(torch.__version__),
                      "checkpoint_sha256": entry.sha256, "model_load_ms": model_load_ms,
                      "profiles": profiles}, indent=2))


if __name__ == "__main__":
    main()
