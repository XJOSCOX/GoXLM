# GoXLM Web 0.1

GoXLM Web is a local **experimental text completion** workspace. The supplied
GoXLM-0.1 model is a 3.26M-parameter byte model trained from scratch with a
128-token context. It continues the current prompt and can generate incoherent
or inaccurate text. The sidebar saves conversation history in this browser.
Each generation sends just the current draft. Clear this site's browser
storage to erase saved history and preferences.

## Start on Windows with Ubuntu WSL2

Run both launchers inside Ubuntu. Use Node.js 24 LTS and the existing
CUDA-enabled PyTorch virtual environment. WSL's native localhost forwarding
lets a Windows browser reach the two IPv4 loopback listeners. The frontend
uses port 13000 and the backend uses port 18000. These ports avoid Windows
port reservations observed with the conventional defaults.

In the backend terminal:

```bash
cd /home/${USER}/ai-research/goxlm
../.venv/bin/python -m pip install -r goxlm_web/requirements.txt
OMP_NUM_THREADS=1 ../.venv/bin/python -B -m goxlm_web
```

The example allowlist pins the original GoXLM-0.1 checkpoint by SHA-256.
The default device policy selects CUDA when available and reports a CPU
fallback in the UI. To require a GPU, copy the example into the ignored
`private/` directory and set `"device": "cuda"`.

In the frontend terminal:

```bash
cd /home/${USER}/ai-research/goxlm/web
export NEXT_TELEMETRY_DISABLED=1
npm ci
npm run build
npm run start
```

Open [http://localhost:13000](http://localhost:13000) in Windows. If the app
or backend is unreachable, compare the WSL and Windows localhost responses
and check WSL localhost forwarding and Windows excluded port ranges. Both
official launchers bind only to `127.0.0.1`. Keep the services on loopback.

## Add a versioned checkpoint

Copy `goxlm_web/models.example.json` to `private/goxlm-web-models.json` and
add an object to `models` containing a unique public `id`, display `label`,
`"format": "0.2"`, the trusted local `path`, and its exact `sha256` from
`sha256sum`. Paths in this operator-controlled file are resolved relative to
the project root unless absolute. Restart the backend with:

```bash
OMP_NUM_THREADS=1 ../.venv/bin/python -B -m goxlm_web --models private/goxlm-web-models.json
```

The app recognizes the 0.2.1 and 0.2.2 schemas supported by the existing
versioned loader. The loader applies its FP32 weight, tokenizer, model, and
checkpoint validation rules even for a BF16 training checkpoint. Approved
local files should stay immutable while served. Changing the approved bytes
requires updating the digest and restarting.

## Architecture

- **Frontend** uses the Next.js App Router, TypeScript, and Tailwind CSS.
  It renders the local workspace, settings, model information and status.
  It consumes NDJSON incrementally and stores history and theme in localStorage.
  Graphite and True Black use locally bundled Geist Variable, licensed under
  the [SIL Open Font License](https://github.com/vercel/geist-font).
- **Backend** uses FastAPI and the local PyTorch Transformer. The model
  registry resolves IDs against the operator's allowlist, hashes the open
  checkpoint before weights-only loading, and moves a single selected model
  to the inference device. Format adapters isolate validation, tokenizer
  prefix, and model metadata so later architectures can be added explicitly.
- **Requests** use `GET /api/models`, `POST /api/generate` and
  `POST /api/generations/{uuid}/cancel`. The stream emits start, delta, and a
  terminal done or error event. Backpressure limits work to one active
  generation, including model load, with a 409 response for competing work.
  Cancellation or disconnect releases the slot after the in-flight token step.
- **Context** counts UTF-8 bytes and includes the versioned model's prefix.
  Oversized prompts are rejected. Generation uses a rolling window at or
  below the model's 128-token context, with at most 256 new byte tokens.
  Streaming uses incremental UTF-8 decoding, so one token can emit empty
  text while a multi-byte character is still incomplete.
- **Privacy** stays browser- and machine-local. The service accepts only the
  two loopback frontend origins and known loopback Host headers. Requests
  address model IDs and fixed API operations. The launcher disables access
  logs, and expected stream errors omit exception details, paths and text.

## Tests

From the project root with CPU forced:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ../.venv/bin/python -B -m unittest -v goxlm_web.test_web
cd web
npm test
npm run typecheck
npm run build
```

Backend tests use synthetic checkpoints and verify hash gating, format and
device selection, UTF-8 streaming, cancellation and busy-slot behavior,
prompt and context bounds, and API validation. The browser's token rate
counts emitted byte tokens during decoding and excludes the cold model load.
When adding a larger checkpoint format, validate its tokenizer, context,
memory profile, and cancellation behavior before listing it in the web allowlist.
Live inference and performance do not require an external model service or dataset.


## Web 0.2 inference

Full forward is the default. Select **KV cache** in generation settings
to try request-local cached decoding. It reuses attention keys and values
until the context fills. Each subsequent step rebuilds the retained window
with positions starting at zero, matching the existing learned position
embeddings and rolling-context semantics. The cache is freed when the
request ends. Token IDs match the reference in greedy regression tests.
Small floating-point differences can affect sampled continuations.

The model list distinguishes configured metadata from metadata verified
after a successful checkpoint load. In generation settings, select a
second allowlisted model to compare it with the primary model. Both receive
the same prompt, sampling settings, decode strategy and seed. Completions
appear side by side on wide screens. Models run sequentially with one
resident checkpoint. Stopping cancels the active stream and skips any
remaining comparison. A disconnect holds the generation slot until the
in-flight operation finishes. Deserialization and GPU kernels can only be
interrupted between operations.

From the project root, run local CUDA parity and benchmark the allowlisted
GoXLM-0.1 checkpoint. These commands leave the checkpoint read-only. Keep
results in the ignored local runs directory.

```bash
GOXLM_WEB_TEST_CUDA=1 OMP_NUM_THREADS=1 ../.venv/bin/python -B -m unittest -v goxlm_web.test_web.CacheTests
mkdir -p runs/web-cache-local
OMP_NUM_THREADS=1 ../.venv/bin/python -B -m goxlm_web.benchmark > runs/web-cache-local/cache-benchmark.json
```

The benchmark warms both paths, alternates their trial order, and reports
synchronized first-token latency, sustained byte tokens per second, p95
inter-token latency, and PyTorch peak allocation and reservation. The
initial checkpoint load is measured separately. Browser transport and
rendering are outside these timings.
