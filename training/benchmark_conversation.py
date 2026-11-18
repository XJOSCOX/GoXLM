"""Bounded CUDA exercise of the governed conversation-base path with generated fixtures."""
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

from goxlm_v02.common import canonical, digest
from goxlm_v02.data import build
from goxlm_v02.run import CONVERSATION_CHECKPOINT, load_checkpoint
from goxlm_v02.test_v02 import PipelineTests
from goxlm_v02.tokenizer import MAX_VOCAB, fit_catalog, load_tokenizer

from .bf16_validation import exact

STEPS = 80
WARMUP = 16
BATCH = 4
LIMIT = 750_000


def numerical_history(checkpoint):
    return [{k: v for k, v in row.items() if k not in ("train_seconds", "tokens_per_second", "validation")}
            for row in checkpoint["history"]]


def causal(model, device):
    model.to(device)
    with torch.inference_mode():
        x = torch.arange(32, device=device).view(1, -1)
        y = x.clone()
        y[:, 16:] += 35
        a, _ = model(x)
        b, _ = model(y)
        torch.testing.assert_close(a[:, :16], b[:, :16], rtol=0, atol=0)
        assert not torch.equal(a[:, -1], b[:, -1])
    return True


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("The benchmark requires CUDA")
    fixture = PipelineTests()
    fixture.setUp()
    try:
        rng = random.Random(42)
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        key = next(f"tokenizer-fixture-{i}" for i in range(100)
                   if int(digest(canonical([42, [f"tokenizer-fixture-{i}"]]))[:16], 16) % 100 < 80)
        words = ["".join(rng.choices(alphabet, k=12)) for _ in range(2600)] * 3
        rng.shuffle(words)
        fixture.submit(key, " ".join(words))
        for i in range(32):
            fixture.submit(f"short-fixture-{i}", digest(f"conversation foundation fixture {i}".encode()))
        release = build(fixture.catalog)
        start = time.perf_counter()
        first = fit_catalog(fixture.catalog, release, fixture.root / "tokens-1")
        fit_seconds = time.perf_counter() - start
        second = fit_catalog(fixture.catalog, release, fixture.root / "tokens-2")
        assert first.read_bytes() == second.read_bytes()
        tokenizer = load_tokenizer(first)
        assert tokenizer.vocab_size == MAX_VOCAB
        text = "π² 👩‍🔬\n\tdef f():\r\n    return 42 "
        assert tokenizer.decode(tokenizer.encode(text)) == text
        opts = fixture.root / "bench-run.json"
        opts.write_text(json.dumps({"batch_size": BATCH}), encoding="utf-8")
        used = 0

        def segment(context, precision, label, steps=STEPS, resume=None, initial=0):
            nonlocal used
            used += (steps - initial) * BATCH * context
            if used > LIMIT:
                raise ValueError("Synthetic optimizer-token-slot budget exceeded")
            path = fixture.root / label
            command = [sys.executable, "-B", "-m", "goxlm_v02", "train",
                       str(fixture.catalog), str(release), str(path), "--device", "cuda",
                       "--architecture", "conversation", "--tokenizer", str(first), "--config", str(opts),
                       "--context-length", str(context), "--precision", precision,
                       "--steps", str(steps), "--checkpoint-every", str(STEPS)]
            if resume is not None:
                command.extend(("--resume", str(resume)))
            out = subprocess.run(command, capture_output=True, text=True, timeout=240, check=True,
                                 env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
            return Path(out.stdout.strip())

        profiles = []
        for context in (256, 512):
            for precision in ("fp32", "bf16"):
                name = f"ctx{context}-{precision}"
                checkpoint_path = segment(context, precision, name)
                started = time.perf_counter()
                model, checkpoint = load_checkpoint(checkpoint_path)
                load_seconds = time.perf_counter() - started
                assert checkpoint["schema"] == CONVERSATION_CHECKPOINT
                assert checkpoint["optimizer_updates"] == STEPS
                assert checkpoint["tokenizer"] == tokenizer.identity()
                history = checkpoint["history"]
                assert all(math.isfinite(row[k]) for row in history for k in ("train_loss", "gradient_norm"))
                metrics = json.loads((checkpoint_path.parent / "metrics.0.5.json").read_text())
                sampled = history[WARMUP:]
                latencies = sorted(1000 * row["train_seconds"] for row in sampled)
                measured = sum(row["train_seconds"] for row in sampled)
                details = {"context": context, "precision": precision, "updates": STEPS,
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "validation": history[-1]["validation"]["overall"],
                           "causal": causal(model, torch.device("cuda:0")),
                           "tokens_per_second": sum(row["train_tokens"] for row in sampled) / measured,
                           "slots_per_second": len(sampled) * BATCH * context / measured,
                           "median_ms": median(latencies), "p95_ms": latencies[math.ceil(.95 * len(latencies)) - 1],
                           "loss_first": history[0]["train_loss"], "loss_last": history[-1]["train_loss"],
                           "max_gradient_norm": max(r["gradient_norm"] for r in history),
                           "peak_allocated": metrics["hardware"]["gpu_peak_allocated_bytes"],
                           "peak_reserved": metrics["hardware"]["gpu_peak_reserved_bytes"],
                           "save_seconds": metrics["checkpoint_seconds"], "load_seconds": load_seconds}
                del model
                torch.cuda.empty_cache()
                if context == 256:
                    first_half = segment(context, precision, name+"-half", steps=STEPS // 2)
                    final = segment(context, precision, name+"-resume", resume=first_half, initial=STEPS // 2)
                    _, resumed = load_checkpoint(final)
                    parity = {k: exact(checkpoint[k], resumed[k]) for k in
                              ("model", "optimizer", "rng", "optimizer_updates")}
                    parity["history"] = exact(numerical_history(checkpoint), numerical_history(resumed))
                    parity["validation"] = checkpoint["history"][-1]["validation"] == resumed["history"][-1]["validation"]
                    details["resume_exact"] = parity
                    if not all(parity.values()):
                        raise RuntimeError(f"Unexpected {precision} resume difference: {parity}")
                    del resumed
                profiles.append(details)
                print(f"Validated {name}", file=sys.stderr, flush=True)
                del checkpoint
        return {"device": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "batch": BATCH,
                "token_slots": used, "fit_seconds": fit_seconds, "tokenizer": tokenizer.identity(), "profiles": profiles}
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2, allow_nan=False))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr or str(exc)) from None
