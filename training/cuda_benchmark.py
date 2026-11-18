"""Bounded synthetic CUDA training benchmark for the GoXLM baseline."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from statistics import median
import time

import torch

from goxlm_v02.common import digest, git_provenance
from goxlm_v02.model import MODEL, new_model
from goxlm_v02.run import environment, revision

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
FP16_INITIAL_SCALE = 1024.0
HELDOUT_LOSS_MAX_DELTA = 0.05


def supported_precisions(device):
    with torch.cuda.device(device):
        modes = {"fp32": None, "fp16": torch.float16}
        if torch.cuda.is_bf16_supported(including_emulation=False):
            modes["bf16"] = torch.bfloat16
        return modes


def autocast(dtype):
    return nullcontext() if dtype is None else torch.autocast("cuda", dtype=dtype)


def percentile(values, fraction):
    """Nearest-rank percentile."""
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def benchmark_trial(device, dtype, batch_size, warmup, steps, seed):
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    model = new_model().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY, foreach=False)
    scaler = (torch.amp.GradScaler("cuda", init_scale=FP16_INITIAL_SCALE)
              if dtype == torch.float16 else None)
    reference = model.token_embedding.weight
    initial_weights = reference.detach().clone()
    context = model.config.context_length
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    tokens = torch.randint(model.config.vocab_size, (warmup + steps + 1,
                           batch_size, context + 1), device=device, generator=generator)
    train_batches = tokens[:-1]
    heldout = tokens[-1]

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype):
            _, loss = model(batch[:, :-1], batch[:, 1:])
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        return loss.detach(), norm.detach()

    def optimizer_steps():
        state = optimizer.state.get(reference)
        return int(state["step"].item()) if state else 0

    @torch.no_grad()
    def heldout_loss():
        model.eval()
        _, loss = model(heldout[:, :-1], heldout[:, 1:])
        model.train()
        return loss.item()

    initial_loss = heldout_loss()
    metrics = [step(batch) for batch in train_batches[:warmup]]
    torch.cuda.synchronize(device)
    warmup_updates = optimizer_steps()
    allocated_start = torch.cuda.memory_allocated(device)
    reserved_start = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    # Prime CUDA events before timing to exclude event initialization.
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(steps)]
    for start_event, end_event in events:
        start_event.record()
        end_event.record()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for batch, (start_event, end_event) in zip(train_batches[warmup:], events):
        start_event.record()
        metrics.append(step(batch))
        end_event.record()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    latencies = [start_event.elapsed_time(end_event) for start_event, end_event in events]
    updates = optimizer_steps()
    final_loss = heldout_loss()
    weights_changed = not torch.equal(initial_weights, reference.detach())
    weights_finite = bool(torch.stack([p.detach().isfinite().all() for p in model.parameters()]).all())
    loss_values, norms = torch.stack([torch.stack(pair) for pair in metrics]).T.cpu().tolist()
    numbers_finite = all(math.isfinite(v) for v in [initial_loss, final_loss, *loss_values, *norms])
    return {
        "tokens_per_second": batch_size * context * steps / elapsed,
        "wall_ms_per_step": 1000 * elapsed / steps,
        "cuda_step_ms_p50": median(latencies),
        "cuda_step_ms_p95": percentile(latencies, 0.95),
        "device_used_bytes_before_trial": total - free,
        "allocated_bytes_at_timing_start": allocated_start,
        "reserved_bytes_at_timing_start": reserved_start,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "warmup_optimizer_updates": warmup_updates,
        "timed_optimizer_updates": updates - warmup_updates,
        "skipped_optimizer_updates": warmup + steps - updates,
        "initial_heldout_fp32_loss": initial_loss if math.isfinite(initial_loss) else None,
        "final_heldout_fp32_loss": final_loss if math.isfinite(final_loss) else None,
        "max_grad_norm": max(norms) if numbers_finite else None,
        "min_train_loss": min(loss_values) if numbers_finite else None,
        "max_train_loss": max(loss_values) if numbers_finite else None,
        "weights_changed": weights_changed,
        "scaler_final_scale": scaler.get_scale() if scaler is not None else None,
        "stable": numbers_finite and weights_finite and weights_changed and updates == warmup + steps,
    }


def benchmark(device, batch_size, warmup, steps, repeats, seed):
    modes = supported_precisions(device)
    results = {mode: [] for mode in modes}
    # Rotate the starting precision across repeats to limit order effects.
    order = list(modes)
    for repeat in range(repeats):
        offset = repeat % len(order)
        for mode in order[offset:] + order[:offset]:
            results[mode].append(benchmark_trial(device, modes[mode], batch_size,
                                                  warmup, steps, seed + 2 * repeat))

    summary = {}
    for mode, trials in results.items():
        deltas = [abs(trial["final_heldout_fp32_loss"] - reference["final_heldout_fp32_loss"])
                  if trial["final_heldout_fp32_loss"] is not None
                  and reference["final_heldout_fp32_loss"] is not None else None
                  for trial, reference in zip(trials, results["fp32"])]
        max_delta = max(deltas) if all(d is not None for d in deltas) else None
        summary[mode] = {
            "median_tokens_per_second": median(t["tokens_per_second"] for t in trials),
            "median_wall_ms_per_step": median(t["wall_ms_per_step"] for t in trials),
            "median_cuda_step_ms_p50": median(t["cuda_step_ms_p50"] for t in trials),
            "median_cuda_step_ms_p95": median(t["cuda_step_ms_p95"] for t in trials),
            "max_peak_allocated_bytes": max(t["peak_allocated_bytes"] for t in trials),
            "max_peak_reserved_bytes": max(t["peak_reserved_bytes"] for t in trials),
            "max_final_fp32_loss_delta": max_delta,
            "stable": all(t["stable"] for t in trials) and max_delta is not None
                      and max_delta <= HELDOUT_LOSS_MAX_DELTA,
        }
    return {"schema": "goxlm-synthetic-cuda-benchmark-1",
            "environment": environment(device), "git": git_provenance(),
            "training_source_fingerprint": revision(),
            "benchmark_source_sha256": digest(Path(__file__).read_bytes()),
            "config": {"model": MODEL, "batch_size": batch_size, "warmup_steps": warmup,
                       "timed_steps": steps, "repeats": repeats, "seed": seed,
                       "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                       "grad_clip": GRAD_CLIP, "fp16_initial_scale": FP16_INITIAL_SCALE,
                       "tokens_per_step": batch_size * MODEL["context_length"],
                       "heldout_loss_max_delta": HELDOUT_LOSS_MAX_DELTA},
            "unavailable": [] if "bf16" in modes else ["bf16 (requires native CUDA support)"],
            "summary": summary, "trials": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0, help="visible CUDA device index")
    parser.add_argument("--batch-size", type=int, default=8, metavar="1..32")
    parser.add_argument("--warmup", type=int, default=8, metavar="1..16")
    parser.add_argument("--steps", type=int, default=30, metavar="2..64")
    parser.add_argument("--repeats", type=int, default=3, metavar="1..3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name, low, high in [("batch_size", 1, 32), ("warmup", 1, 16),
                            ("steps", 2, 64), ("repeats", 1, 3), ("seed", 0, 2**32 - 8)]:
        if not low <= getattr(args, name) <= high:
            parser.error(f"--{name.replace('_', '-')} must be in {low}..{high}")
    if not torch.cuda.is_available() or not 0 <= args.device_index < torch.cuda.device_count():
        parser.error("A visible CUDA device is required")

    device = torch.device("cuda", args.device_index)
    with torch.cuda.device(device):
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        # Measure the normal eager CUDA path. The separate suite checks deterministic resume.
        torch.use_deterministic_algorithms(False)
        report = benchmark(device, args.batch_size, args.warmup, args.steps, args.repeats, args.seed)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0 if all(result["stable"] for result in report["summary"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
