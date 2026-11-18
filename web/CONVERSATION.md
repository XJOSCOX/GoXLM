# GoXLM conversation format and development plan

## Conversation format v1

`goxlm_web.conversation` produces model input IDs for a future conversation-capable
checkpoint. The existing Web 0.2 generation routes continue to accept
text-completion prompts. GoXLM-0.1 remains an experimental completion model.

A history is a sequence of complete user and assistant pairs followed by the
latest user message. The formatter rejects other roles and requires nonempty
message text. It wraps each message in a role start token and a shared
end-of-turn token. The prompt ends with an open assistant marker.

```text
USER_START text END_TURN ASSISTANT_START text END_TURN
USER_START latest_text END_TURN ASSISTANT_START
```

The checkpoint adapter supplies an ordinary-text encoder, text vocabulary
boundary, total vocabulary size, and three distinct control IDs at or above the
text vocabulary boundary. The formatter adds those IDs itself and rejects
any content encoding outside the ordinary range, including unassigned
control IDs. A string resembling a role marker is therefore ordinary text
or an encoding error. A future API must derive assistant history from
trusted server-side turns and must not map user-submitted role fields to
privileged messages. Token isolation prevents structural role spoofing.
Model behavior still requires separate prompt-injection evaluation.

The adapter passes the checkpoint context length and the requested generation
reserve. The formatter counts encoded text and all framing tokens, including
the final assistant marker. If the latest user message cannot fit, return a
budget error so the user can shorten it or reduce the reserve. Keep it intact.
Starting from the newest historical pair, retain a contiguous suffix of
complete turns until the next pair would exceed the budget. Do not split
pairs or skip an intervening pair. The result includes the format version,
input IDs, reserved generation tokens, and the count of dropped turns. A future
chat client can display “Older conversation turns omitted” when that count is
positive. The serving adapter must stop generation at or before the reserved
token count and must not roll the framed prompt past the model context.

A conversation-trained checkpoint should declare the format version, tokenizer
version and control token mapping. Validate those fields against the serving
adapter before exposing chat. Train the format and stopping behavior on
approved data before enabling chat for any checkpoint.

## Controlled data acquisition

Prioritize modern explanatory English and everyday dialogue. Propose a small
batch of original authored conversations, follow-ups, corrections and plain
explanations under written terms covering commercial training, evaluation and
intended distribution. Aim for 500–1,000 pilot examples, then measure review
time and quality before scaling. Seek item-verified public-domain general
prose and documented, commercially licensed dialogue and Q&A to add breadth.
Treat historical fiction as a limited source of dialogue. Use source-specific
rights evidence for each exact version and exclude borrowed passages unless the
rights cover them. Ask for explicit retrieval authorization before acquiring
candidate material.

Retain the approved STEM material with its original provenance and conditions
as a supporting part of a broader mix. Keep the existing checkpoint,
architecture presets, catalog, decision ledger and evidence history. Review
rights and quality independently, track deduplication and related versions,
and split by source, author and conversation scenario. Keep test material
outside acquisition and training.

A first conversational base target is 30–50 million deduplicated subword
tokens. Count optimizer exposures separately from unique retained text. Train
a small byte-fallback tokenizer on approved training text and verify exact
Unicode and whitespace round trips. Evaluate coherent continuation before
instruction tuning. Test a fresh 25M-class decoder at context 256, then 512
if validation warrants it. Reserve instruction examples for a reviewed
supervised phase with diverse greetings, everyday Q&A and actual multi-turn
history. Use data, evaluation and measured hardware gates before authorization
of sustained training.

## Future private memory and consent

- Store local conversation history and explicit memories separately. A memory
  stores its text, version, timestamp and optional source turn. Expose list,
  inspect, edit and delete controls. Use only relevant memories within a small
  tokenizer-counted budget. Deleting an entry invalidates retrieval caches and
  derived summaries that could reintroduce it. Test local isolation and
  deletion before enabling persistence.
- Keep the default conversation history private and outside model improvement.
  Offer an explicit opt-in for selected turns and clearly described purposes.
  Treat a rating as feedback and a correction as a proposed record. Neither
  grants permission to train on the conversation.
- Isolate contributed content. Minimize retention, remove sensitive details,
  check rights and provenance, and screen coordinated duplicates, attacks and
  misleading corrections. A reviewer must approve the exact redacted record.
  Keep versioned release manifests with consent, record, run and checkpoint
  lineage. Store private artifacts outside the public repository.
- Withdrawal blocks future inclusion and drives deletion of retained and
  derived records under the retention policy. Trace any affected releases and
  checkpoints for review, quarantine or retraining from a clean ancestor.
  Deleting raw records from storage does not remove learned influence from
  existing weights.
- Improve models through separately authorized offline runs. Compare held-out
  everyday replies, instruction following, follow-up accuracy, clarification,
  correction handling, repetition and factual answers against a matched base.
  Test remembered context with an ablation that removes the relevant history.
  Check privacy and consent lifecycle tests before enabling any contribution
  feature.

## Future instruction adapter acceptance

The future versioned instruction adapter must check the checkpoint schema
and training stage, exact tokenizer artifact SHA-256, role IDs, formatter
version and context. Its allowlist entry should bind an independent
conversation evaluation result to that exact checkpoint hash. Existing
completion endpoints keep their existing model eligibility. Prototype
tests should cover full-turn trimming, cancellation, end-of-turn stop,
partial UTF-8, and synthetic rejection of base checkpoints before a chat
route is authorized.
