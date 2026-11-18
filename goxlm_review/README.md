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
