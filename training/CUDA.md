# Local CUDA validation and benchmarking

GitHub Actions runs the synthetic CPU suite. Run the CUDA suite separately on
the RTX 3090 in Ubuntu WSL2 with a CUDA-enabled PyTorch installation.

```bash
cd /path/to/goxlm
PY=../.venv/bin/python  # Set this to your CUDA-enabled Python.
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 "$PY" -B -m unittest -v training.test_cuda
```

The suite checks CUDA forward/backward, finite gradients, optimizer updates,
and causal attention in FP32, FP16, and native BF16 where available. The
checkpoint test uses the existing versioned trainer and temporary synthetic
fixtures. It verifies bitwise FP32 resume against an uninterrupted run. A
CUDA-unavailable test run is skipped.

## Compare precision modes on synthetic inputs

```bash
mkdir -p runs
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 "$PY" -B -m training.cuda_benchmark \
  --batch-size 8 --warmup 8 --steps 30 --repeats 3 --seed 42 > runs/cuda-batch8.json
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 "$PY" -B -m training.cuda_benchmark \
  --batch-size 32 --warmup 8 --steps 30 --repeats 3 --seed 42 > runs/cuda-batch32.json
```

Use a quiet GPU and retain the JSON reports locally. Use the same seed,
PyTorch version, driver, batch size, and context when comparing reruns.
Within each repeat the modes start from the same random weights and synthetic
batches, and the first mode rotates between repeats. The model, parameters,
FP32 optimizer state and context match the 3.26M GoXLM baseline. FP32 uses
the highest matmul precision with TF32 disabled. BF16 uses native autocast
with FP32 parameters and no scaler. FP16 uses autocast and a dynamic
`GradScaler` starting at 1024, with unscaling before gradient clipping.

- **Throughput** measures timed target slots per synchronized wall second.
  Synthetic batches are preloaded on the GPU and contain no padding.
- **Latency** reports median and nearest-rank p95 CUDA event durations per
  trial, plus mean synchronized wall milliseconds per step. Warmup,
  evaluation, checkpointing, and dataset preparation are outside timed work.
- **Memory** reports live PyTorch allocation and reservation at timing start,
  peak allocated and peak allocator-reserved bytes over timed steps, and
  device-wide used bytes before each trial. Allocator figures exclude GPU
  memory held by other processes and non-PyTorch CUDA allocations.
- **Stability** requires finite losses and gradient norms, finite final
  parameters, changed weights, no skipped optimizer updates, and a final
  held-out FP32 loss within 0.05 of paired FP32. The synthetic loss is a
  numerical comparison, with long-run convergence still unmeasured.
- **Bounds** are at most 32 sequences of context 128, 16 warmup steps, 64
  timed steps, and three repeats per available mode. Both commands use
  synthetic data and fresh weights and never fetch assets. The benchmark
  emits JSON and exits nonzero for an unstable mode.

Keep FP32 for deterministic baseline reproduction. Compare native BF16 for
future throughput-oriented pilots, especially at higher batch sizes. Record
batch size and precision together, since a throughput advantage can reverse
at smaller batches. The benchmark is eager and does not use deterministic
algorithms. Mixed-precision training and checkpoint resume remain separate
work from this local benchmark.
