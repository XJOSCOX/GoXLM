# GoXLM human rights review

Use a review store in the private research workspace. Evidence, source bytes,
candidate extracts, and human decisions belong in that store. The review queue
and training catalog are separate.

```sh
STORE=/path/to/private/review-store
review() { python -B -m goxlm_review "$STORE" "$@"; }
review list
review verify
review show "$ID"
review show "$ID" --text
review evidence "$ID" "$NEW_EXPORT_DIRECTORY" --include-source
```

Set the variables using the selected case and an unused private destination for
the export. Inspect the exact extract, source, and evidence. Confirm rights
holder authority, the scope for commercial AI training, and applicable
attribution and exclusions. An authorized operator enters the full digest in
the interactive prompt and records their decision.

```sh
review decide "$ID" approve --extract-sha256 "$SHA" --reviewer "$NAME" \
    --date "$DATE" --jurisdiction "$JURISDICTION" --rationale "$RATIONALE" \
    --commercial-ai-training --acknowledge-findings
review decide "$ID" hold --extract-sha256 "$SHA" --reviewer "$NAME" \
    --date "$DATE" --category rights --rationale "$RATIONALE"
review decide "$ID" reject --extract-sha256 "$SHA" --reviewer "$NAME" \
    --date "$DATE" --category rights --rationale "$RATIONALE"
```

Run only the command corresponding to the human decision. Archive the output
of `review head` independently. Use `review verify --expected-head "$SAVED_HEAD"`
to detect changes through that point. Chain integrity is distinct from an
electronic signature or a verification of legal authority.

Use `review template` with one of `us_public_domain_expired`,
`us_government_work`, `cc0`, or `explicit_commercial_ai` to prepare batch
metadata. `review batch MANIFEST --input-root LOCAL_INPUTS` checks supplied
hashes and stages local bytes for review. Supply source and evidence references
as provenance metadata. Staging and automated findings do not authorize use.
Run `python -B -m goxlm_review "$STORE" --help` for the interface.


### Read-only case browser

Create a private queue outside the repository's tracked files. The JSON array has one entry per case, with `id`, the complete `case_sha256` from the case envelope, and a two-element `minutes` array for the estimated review range.

Run `python -B -m goxlm_review.browse STORE QUEUE.json` from an interactive local terminal. Use `n` and `p` to change cases, `j N` to jump, `>` and `<` to page through the exact extract, and `e N` to inspect an archived evidence item. The browser verifies staged objects and the decision chain, then references shared rights notices by digest and displays each shared notice once per session. Evidence and originals are read locally without opening links.

For optional elapsed-time tracking, pass `--effort-dir PRIVATE_DIR`, using a mode-0700 directory outside the protected store. After measuring time spent on the displayed exact extract, enter `t SECONDS`. Each entry records case and extract hashes and the elapsed time entered by the reviewer. It does not infer consent or a decision. Use the existing interactive protected workflow to record any separate rights decision.


### Private transcript-only intake preview

Run the synthetic suite without provider data.

```sh
python -B -m unittest -v goxlm_review.test_transcript_intake
```

Before preparing real inputs, an authorized operator must verify the executed data-use agreement, named recipient, active provider approval, exact delivery, commercial text-training scope, and the provider's age eligibility evidence. The program checks the operator's independent receipt digest and the exact file digests. It cannot authenticate an unsigned assertion, infer provider approval, or determine the meaning of a license. Keep agreement versions, source manifests and external approval evidence in protected storage.

Use a real, current-user-owned directory with mode 0700, outside the project checkout, on a filesystem directly backed by Linux dm-crypt. This deliberately refuses other storage layouts. Arrange access controls, encrypted backups and applicable incident and deletion procedures separately. In that directory place real files with mode 0600.

- agreement.txt and provider-approval.txt hold the controlling evidence for this delivery. Do not store portal credentials in either.
- transcripts.jsonl contains only normalized transcript records. Preserve separate private provenance mapping to the publisher's exact source bytes. Use stable, opaque IDs matching c_, r_ or s_ followed by 12–64 lowercase hexadecimal characters. Use provider pseudonyms or a consistent private mapping of those pseudonyms. Do not infer names or create links to external identities.
- Each record has id, recording_id, speakers (exactly two IDs), participant_status, language, segment_start_ms, segment_end_ms and turns. A turn has speaker (0 or 1), start_ms, end_ms and text. Use source-recording-relative times. Set participant_status to all_adult_verified only from appropriate provider evidence about both participants at recording time. Otherwise use contains_minor or unknown. The script excludes every segment of a recording if any segment is flagged. Never include birth dates, ages, demographics, audio, or participant identities as model inputs. Keep any withheld content out of model or application logs.
- heldout.json is an array of nonempty protected evaluation strings from a separate evaluation set.
- scope.json has provider set to Open Yap 1K, the exact approved recipient, scope set to approved_transcript_only_commercial_training, status set to active, an opaque delivery_id (d_ plus 12–64 lowercase hexadecimal characters), and files with SHA-256 values under agreement, provider_approval, delivery and heldout. Use an independently retained anchor for these bytes. Re-check that the authorization and rights are still active before each use.

```sh
python -B -m goxlm_review.transcript_intake \
  "$PRIVATE_ROOT" --scope-sha256 "$TRUSTED_RECEIPT_SHA256" --recipient "$APPROVED_RECIPIENT"
```

The script screens malformed or low-signal turns, sensitive-data patterns, quotation cues, held-out overlap, and exact or near-duplicate content. It never searches external services. All repeated speakers and recording segments stay within a single connected component, including links seen in records screened out later. The deterministic preview aims for an 80/10/10 word split and reports actual weights and component concentration. Fewer than three components block a three-way preview. A giant component or weak holdout can still make an otherwise valid preview unsuitable for training.

The hash-only preview is written once inside the encrypted root. It omits raw turns and speaker IDs, and stdout contains aggregate counts only. Every surviving record remains in quarantine pending item-level privacy, incorporated-material, provenance, quality and rights checks. Pattern filters cannot establish anonymization or rule out every borrowed passage. Do not stage a training release or publish transcripts, manifests that reconstruct transcripts, or raw examples. Preserve the prior rights ledger and review any provider-required withdrawal through the applicable contract and existing governance workflow.
