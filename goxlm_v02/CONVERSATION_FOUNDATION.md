# Conversation model foundation

## Tokenizer and governed training

The local byte BPE tokenizer starts with all 256 byte IDs. It uses ordinary
ASCII word runs, whitespace runs, and other byte runs, split into at most
64-byte chunks. Encoding and fitting use the same boundaries. Training
chooses the most frequent adjacent pair with at least two occurrences.
Ties use lexicographic token-ID order. Merges stay within chunks and
documents. The original UTF-8 bytes drive fitting and reconstruction.
Unicode normalization, capitalization changes, and whitespace collapsing
are outside this format.

The final three IDs are reserved for user, assistant, and end of turn.
For a full 4,096-ID artifact they are 4093, 4094, and 4095. A smaller
fixture can end at a smaller vocabulary. The checkpoint and tokenizer
identify the exact sizes and IDs. The content encoder cannot emit any
of the reserved IDs.

From the project root, `python -B -m goxlm_v02 fit-tokenizer CATALOG RELEASE
PRIVATE_OUTPUT_DIRECTORY` validates the approved release and fits its train
split. The canonical JSON contains the release hash, a fingerprint of the
sorted train document IDs and text hashes, and the merge list. The file name
is its SHA-256. The historical demonstration corpus is excluded. Fit and
publication recheck release rights. Keep tokenizer artifacts in an ignored
private directory since the merge table comes from corpus text.

After separate training authorization use `--architecture conversation
--tokenizer PRIVATE_ARTIFACT` with the existing governed trainer. It checks
tokenizer lineage against the same release and emits checkpoint schema
`goxlm-checkpoint-0.5.0-conversation-base`. With a full vocabulary, the
configuration has 27,448,320 parameters at context 256 and 27,579,392 at
context 512. It supports deterministic FP32 and native CUDA BF16 with
FP32 parameters and AdamW state. BPE evaluation needs the identical
tokenizer artifact and reports loss per target token and bits per UTF-8
byte of canonical tokenization. A change to model context or tokenizer
requires a new run identity.

Conversation-base checkpoints store the formatter version and role map
and have the explicit training stage `base_pretraining`. Existing Web
adapters accept their original schemas and still serve experimental text
completion. Strict legacy resume continues to require the original recorded
source and environment.

## Future conversation adapter contract

An authorized instruction-training stage must bind the eligible checkpoint
to its tokenizer digest, formatter version, role IDs, context length, and
instruction release, and pass independent evaluations. Flatten complete
stored user/assistant pairs into the formatter and append the newest user
message. Reserve response tokens and surface the formatter's dropped-turn
count. At generation time allow ordinary token IDs plus end-of-turn, stop
at the reserve or end-of-turn, and decode stable UTF-8 incrementally.
The current base checkpoint reader validates architecture and lineage and
provides no chat-serving path.

## Private dialogue review pilot

Use a dedicated ignored dialogue review store. The schema is available via
`python -B -m goxlm_review.dialogue schema`. The module also applies
cross-field origin, requested-use, and digest checks during intake.

- Use `seal DRAFT_INPUT SEALED_PATH` to capture the hash of exact UTF-8
  turns and origins in a new draft. Each pair contains a user message and
  assistant reply. Supply opaque contributor, scenario and evidence refs.
- Use `audit SEALED_PATH...` for context fit, normalized exact matches,
  near-duplicate shingles, and likely private-data patterns. Pass
  `--against PRIOR_DRAFT...` to compare with other batches or held-out
  fixtures. The report emits draft IDs and flag categories without the
  matched text. These checks miss some paraphrases and personal data.
  Separately review source, author and scenario links before splitting.
- Use `quality-schema` to prepare an operator's private assessment. The
  review records the exact sealed file SHA-256, reviewer ref, ISO date,
  accepted/hold/rejected decision, all six rubric scores, and explicit
  notes for naturalness, accuracy, follow-up consistency, privacy and
  duplicates. Basic STEM acceptance requires a fact-check ref bound to
  the draft. Other declared fact-check refs must also match the draft.
- Use `record-quality STORE SEALED_PATH ASSESSMENT_JSON RECORDED_QUALITY`
  in an interactive WSL terminal. An allowlisted operator confirms the
  exact draft SHA-256. The CLI saves a hash-verified quality envelope
  separately from the rights ledger. Accepted assessments require at
  least 10/12, no zeros, full accuracy and context marks, and no
  unresolved structured privacy flags. Redactions require a
  new draft and a new exact review.
- Use `stage STORE SEALED_PATH RECORDED_QUALITY EVIDENCE_INDEX INPUT_ROOT`
  for accepted human-written training candidates. The evidence index
  maps opaque refs to relative files under the input root. Stage checks
  exact evidence hashes, authorship and scope evidence, independent
  quality acceptance and full 256-token context fit. With no tokenizer
  argument, the context check uses the conservative byte fixture.
- Staging creates an undecided case. A separate authorized human uses
  the protected rights-review CLI to confirm the exact extract hash and
  commercial-training permission. Keep this store isolated from the
  pretraining catalog. Synthetic records are plumbing fixtures.
  AI-assisted drafts remain quarantined for scope review.

## Validation

Run the bounded RTX 3090 harness with `python -B -m training.benchmark_conversation`.
It creates its own temporary synthetic catalog and independent tokenizer
builds. The 512 profiles run after the 256 tests. The harness refuses more
than 750,000 optimizer token slots. Keep the output report private. The
benchmark measures authenticated trainer behavior and model mechanics with
synthetic text. It provides no language-quality evidence.

Before enabling chat, author and hold out separate groups for greetings,
ordinary replies, factual explanations, corrections, follow-ups, and
instruction following. Use 100 blind base continuations first. Target at
least 80% coherent and relevant continuations and at most 5% repetition
loops. Compare held-out bits per byte with a matched baseline. After
instruction tuning use at least 300 scenario-held-out cases, including
context-removal controls, with at least 85% appropriate everyday replies,
80% follow-up/correction accuracy, and 80% simple instruction adherence.
Score factual correctness and uncertainty separately. Keep these datasets
and derived material away from tokenizer fitting, pretraining and SFT.
