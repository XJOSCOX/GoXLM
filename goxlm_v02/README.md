# GoXLM 0.2

Run commands from the repository root with a Python environment that includes PyTorch. Replace the example interpreter path with your environment. The 0.2 entry points verify the original baseline before live training.

## Run synthetic tests and verify the local baseline

```sh
../.venv/bin/python -B -m unittest -v goxlm_v02.test_v02 goxlm_v02.test_governance
# In the private research workspace with the original artifacts:
../.venv/bin/python -B -m goxlm_v02 verify-baseline
GOXLM_TEST_DEVICE=cuda ../.venv/bin/python -B -m unittest -v goxlm_v02.test_v02.PipelineTests.test_reproducible_resume_and_evaluation
```

## Legacy GoXLM-0.1 entry point

Importing `training.train` performs no training. In the private workspace,
the explicit historical entry point is:

```sh
python -m training.train --output checkpoints/goxlm_v01-reproduction.pt
```

This runs the original 300-step experiment on the local GoXLM-0.1 corpus. The
output must be a new file. Checkpoint publication refuses an existing path,
including a file created by a competing process before the save finishes. The
original checkpoint layout and inference loader remain compatible.

## Build a catalog from authorized local files

Use a new catalog directory on the Linux filesystem. Keep write access to approval records and the test freeze with authorized operators. Back up that directory with the release, as the training loader rechecks the original evidence and decisions. The tool captures evidence and verifies integrity. The designated reviewer verifies its authenticity, authority, applicability to the exact document, and commercial AI training scope. Record rejection if the evidence is inadequate. Releases contain document text and review metadata, so keep catalogs and releases private unless separately authorized.

Intake metadata is a JSON object with these nonempty strings: `document_id`, `source_id`, `family_id`, `title`, `domain`, `rights_holder`, `source_reference`, `rights_basis`, and `evidence_reference`. Valid domains are `science`, `technology`, `math`, and `programming`. Give all related versions the same family ID even when their origins differ. Use the same source ID for documents from a common source. Supply a local copy of license text or permission evidence that can be checked against its reference. Intake prints the fingerprint to use in the decision.

A separate decision JSON requires `submission_sha256`, `decision` (`approved` or `rejected`), `reviewer`, `review_date` (YYYY-MM-DD), `rationale`, and `permissions_scope` (`commercial_ai_training` for approval). Document any applicable conditions in the rationale and clear them before approval. A document and its decision are immutable under the CLI. To correct or replace either, use a new document ID and preserve the related family ID. Use the governance commands below to record rights withdrawals and locate affected artifacts.

```sh
../.venv/bin/python -B -m goxlm_v02.data CATALOG intake metadata.json document.txt evidence.txt
../.venv/bin/python -B -m goxlm_v02.data CATALOG review DOCUMENT_ID decision.json
../.venv/bin/python -B -m goxlm_v02.data CATALOG build --seed 42
```

The first successful build assigns related groups to splits by seeded hash, targeting 80/10/10 by group in expectation. All three splits must be present. The catalog freezes the first test membership and submission hashes. Later builds use the same seed and keep test membership fixed. A related addition to test blocks the build and requires operator review. The content-addressed release includes the complete split manifest and documents. Similarity detection uses normalized exact matches, character similarity and containment, plus declared source and family IDs. It is heuristic and quadratic. Inspect split counts, topic balance, and similarity before a real run.

## Run and evaluate an approved experiment

```sh
../.venv/bin/python -B -m goxlm_v02 train CATALOG RELEASE NEW_RUN --steps N --checkpoint-every K --device cuda
../.venv/bin/python -B -m goxlm_v02 evaluate CATALOG RELEASE CHECKPOINT --split validation
../.venv/bin/python -B -m goxlm_v02 generate --catalog CATALOG CHECKPOINT 'The derivative of' --steps 20
```

### Optional CUDA BF16

The default is deterministic FP32. Set `--device cuda --precision bf16` to
enable BF16 on a device with native support. The device check runs before
the baseline and catalog are opened or output is created. Forward and loss
use CUDA autocast. Model parameters, AdamW moments, and validation stay in
FP32. Backward, gradient clipping, and optimizer updates run outside autocast.
BF16 uses no gradient scaler. Nonfinite losses or gradients stop the run.
Checkpoint publication also checks finite FP32 parameters and AdamW state
and verifies each AdamW step against the completed update count.

New checkpoints use `goxlm-checkpoint-0.2.2`. They retain dataset, tokenizer,
configuration, source and environment identity, and save precision, completed
optimizer updates, and Python, CPU, CUDA and sampler RNG states. Precision
must match on resume. The reader and revocation audit accept both 0.2.1 and
0.2.2. A 0.2.1 checkpoint is interpreted as FP32 and can be evaluated with
the new reader. Exact resume still requires the source fingerprint and
environment recorded by the saved run. For a checkpoint created by an older
source revision, resume from that revision. All new checkpoints and segment
reports are written to a fresh run directory.

The `metrics.0.2.json` segment report measures training, validation, and
checkpoint time separately. Training step time includes batch sampling,
transfer, forward/backward, clipping, and the optimizer step through CUDA
synchronization. Checkpoint time includes state validation, the rights
check, and atomic serialization. Checkpoint hardware metrics report peak
allocator usage through its validation boundary.

The trainer accepts a JSON snapshot via `--config` with `seed`, `batch_size`, `learning_rate`, `weight_decay`, and `grad_clip` (omitted fields use defaults). For resume, pass a 0.2 checkpoint via `--resume` and a fresh output directory. `--steps` is the absolute stopping step. A checkpoint is always emitted at the final step. Finalize selection on validation before requesting `--split test`. The test option is explicit but operator discipline controls its frequency. Evaluation reports loss and bits per UTF-8 byte, overall and by topic. It scores every byte with a fixed 128-byte window scheme. It is a language modeling metric and does not measure task accuracy.

The byte tokenizer has no trained vocabulary. The 3,257,856 parameter model starts from random weights for every fresh run. Resume is limited to a matching 0.2 run identity. The checkpoint and run manifest retain the source-tree SHA-256 fingerprint
(`code_revision`) and record a separate `git` object with the checkout's
`commit_sha` and `dirty` status at run start. Dirty includes staged changes,
tracked edits, or non-ignored untracked files. Ignored private artifacts are
outside that scope. Missing Git or a source export yields null fields. An
unborn repository has a null commit and a measured dirty status. Older
checkpoints remain loadable without Git metadata. Strict resume still requires
the matching data, configuration, source fingerprint and environment recorded
by the original run. A source change can require resuming from the original
checkout. Use trusted local checkpoint files. Deterministic resume is checked within the same environment. Timing, CPU time, memory and throughput values describe each invocation and naturally vary. GPU telemetry measures PyTorch allocation and reservation. Board power and energy are not recorded.

## Corpus governance for GoXLM 0.3 preparation

Ordinary builds create preview releases and retain the legacy preview freeze. To create the separate permanent test set, run `data CATALOG build --permanent --seed SEED` only after the corpus and holdout plan have been reviewed. The permanent build requires 40 independent groups and 256 KiB after conservative group counting, including three groups and 16 KiB in each holdout. It rejects baseline demonstration overlap and cross-split matches from the independent audit. Group thresholds are minimums and a seeded assignment can fail on an imbalanced corpus. A revoked frozen test source blocks future builds for that test lineage. Keep the historical freeze and have a reviewer authorize any successor catalog.

Replace `PYTHON` with the path to the project virtual environment's Python interpreter.

```sh
PYTHON -B -m goxlm_v02.audit_tool CATALOG integrity
PYTHON -B -m goxlm_v02.audit_tool CATALOG overlap RELEASE [--against OTHER_RELEASE]
PYTHON -B -m goxlm_v02.audit_tool CATALOG revoke DOCUMENT_ID revocation.json
PYTHON -B -m goxlm_v02.audit_tool CATALOG impact TRUSTED_CHECKPOINT_DIRECTORY
```

A revocation JSON object requires `submission_sha256`, `reviewer`, `revoked_on` in YYYY-MM-DD form, `reason`, and `notice_reference`. Keep the underlying withdrawal evidence with the project records. The command publishes an immutable event through the CLI and blocks the document and normalized exact copies. Review affected families and near copies through the overlap audit and record further revocations where applicable. Always inventory the authoritative catalog and pass all relevant checkpoint locations. The impact report includes affected document IDs and splits, and reports unknown lineage separately. Treat affected checkpoint files as quarantined until reviewed. The tool preserves historical files and cannot recall copies. Older GoXLM 0.1 is outside this lineage system and retains its own hash manifest.

The audit compares normalized exact text, overlapping five-token passages and declared source and family IDs. It can compare releases in the same catalog. The audit is pairwise and heuristic. Restrict catalog write access and keep independently protected backups or signed copies of ledger events and freezes. The local files provide no protection against an administrator rewriting or deleting the complete history.


## Experimental GoXLM 0.4 architectures

The trainer selects the legacy 0.2 baseline by default. Explicitly select
`--architecture baseline`, `--architecture 10m`, or `--architecture 25m`
to opt into the experimental schema. Set `--context-length` with a preset
to choose a length from 32 to 1024. Experimental runs default to 128.
These are validated configurations of the existing decoder, with tied
embeddings, causal attention, random initialization, and the existing
UTF-8 byte tokenizer.

| Preset | Layers | Attention heads | Width | Parameters at context 128 |
|---|---:|---:|---:|---:|
| `baseline` | 4 | 4 | 256 | 3,257,856 |
| `10m` | 6 | 6 | 384 | 10,795,008 |
| `25m` | 8 | 8 | 512 | 25,416,704 |

Training and validation use the selected context. Experimental outputs
use `goxlm-checkpoint-0.4.0-experimental`, include the preset and exact
configuration, and still pass through the same catalog, approval, and
revocation checks. Resume requires the same architecture, context,
precision, release, code, and environment. Existing 0.2 checkpoints remain
readable. Strict resume of older runs uses the original code revision.
Web 0.2 continues to use its existing adapters. A future adapter must
explicitly opt into the experimental schema and validate its limits.

Run the local CUDA check from the repository root in the environment
with native BF16 support. It uses a temporary generated corpus and
temporary checkpoints. It runs 24 steps per preset and precision at
microbatch 8 and context 128, then checks a 12+12-step BF16 resume for
the 10M preset in fresh processes. Standard CI remains CPU-only.

```sh
OMP_NUM_THREADS=1 ../.venv/bin/python -B -m training.benchmark_architectures
```

The report excludes the first four updates from throughput and step
latency. Peak allocator statistics cover the segment through validation,
while validation, save, and verified CPU load durations are separate.
Keep any saved report in an ignored local runs directory.
