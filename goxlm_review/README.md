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

#### Storage on WSL2

The ordinary WSL distribution filesystem and Windows shared paths do not pass the intake check. Use a dedicated Linux filesystem directly on a dm-crypt LUKS2 mapping. The program checks the running kernel's device identity, rejects Linux swap, requires an operator-owned mode-0700 root outside the checkout, and accepts only operator-owned mode-0600 regular files on the same volume. The CLI disables Linux process dumps before reading private inputs.

Use a new, dedicated LUKS2 container or volume. The following is an example for a **fresh blank image** created specifically for this purpose. Size the image for the agreed transcript delivery and necessary evidence. The operator must confirm local storage policy and provision `cryptsetup`, its kernel module, and `gpgv`. Use interactive key entry and keep credentials separate.

```sh
sudo modprobe dm_crypt
umask 077
IMAGE=$(mktemp "$HOME/goxlm-open-yap.XXXXXXXX.luks")
fallocate -l 8G "$IMAGE"
sudo cryptsetup luksFormat --type luks2 "$IMAGE"
sudo cryptsetup open "$IMAGE" goxlm-open-yap
sudo mkfs.ext4 /dev/mapper/goxlm-open-yap
sudo install -d -m 0700 /mnt/goxlm-open-yap
sudo mount -o nodev,nosuid,noexec /dev/mapper/goxlm-open-yap /mnt/goxlm-open-yap
sudo install -d -m 0700 -o "$(id -u)" -g "$(id -g)" /mnt/goxlm-open-yap/intake
sudo cryptsetup status goxlm-open-yap
findmnt -T /mnt/goxlm-open-yap/intake
swapon --show
```

Set `swap=0` in the `[wsl2]` section of the Windows user's `.wslconfig`. Apply it with `wsl --shutdown` after saving active work and before reopening WSL. Confirm that `swapon --show` returns no devices. Close the mount and mapping after use. Verify backup and recovery procedures without using real transcripts.

The storage check assumes a trusted host, WSL kernel and administrators, and that root provisions the LUKS2 mapping with cryptsetup. The volume protects files when locked or offline. An unlocked host, privileged software, host page/hibernation/crash files, and live VM snapshots can expose memory or plaintext. Before handling provider data, require host-level controls for these paths, including encryption of any host paging and hibernation storage or disabling it, preventing plaintext dumps and snapshots, and encrypting authorized backups. The WSL process cannot attest to these Windows controls. Keep transcripts out of synced paths, Web services, telemetry, logs and Git. Do not continue when storage or host controls remain unverified. [Microsoft WSL configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)

#### Independent authorization

An independent authorized reviewer must confirm the executed provider agreement, named recipient, exact delivery and normalized export, commercial text-training scope, age eligibility evidence at recording time, and active grant. Check the controlling provider evidence and any withdrawal before signing. The reviewer keeps their private signing key separate from the intake operator. An administrator independently verifies the reviewer key fingerprint, installs only current authorized reviewer public keys in the root-owned `/etc/goxlm/open-yap-reviewers.gpg` keyring, and removes keys when trust changes. No key supplied with a delivery can establish trust. Protect the keyring and its parent directories from non-root writes.

Keep these real files at mode 0600 on the private volume.

- `agreement.txt` and `provider-approval.txt` hold the controlling agreement and grant evidence. Preserve original signatures and metadata in the private evidence archive and bind exact evidence exports to the receipt. Do not put portal credentials in these files.
- `source-manifest.json` and `age-evidence.json` preserve the provider's exact delivery mapping and age-eligibility evidence or verified evidence references. Use opaque IDs and keep raw identity information outside model inputs.
- `transcripts.jsonl` contains the normalized delivery. IDs are stable and opaque, using `c_`, `r_` or `s_` plus 12–64 lowercase hexadecimal characters with no external identity linkage. Each record has `id`, `recording_id`, `speakers` (two IDs), `participant_status`, `language`, `segment_start_ms`, `segment_end_ms` and `turns`. A turn has `speaker` (0 or 1), `start_ms`, `end_ms` and `text`. Use recording-relative times. Assign `all_adult_verified` only when provider evidence covers both participants at recording time. Otherwise use `contains_minor` or `unknown`. Any flagged segment excludes the entire recording.
- `heldout.json` is an array of nonempty independently protected evaluation strings.
- `scope.json` has `provider` set to `Open Yap 1K`, the exact approved `recipient`, `scope` set to `approved_transcript_only_commercial_training`, `status` set to `active`, and a `delivery_id` using `d_` plus 12–64 lowercase hexadecimal characters. Its `files` object binds SHA-256 hashes under `agreement`, `provider_approval`, `source_manifest`, `age_evidence`, `heldout` and `delivery`. Its `review` object contains UTC dates `verified_on` and `valid_through`, and `findings` with `signed_provider_agreement`, `approved_recipient`, `authorized_delivery`, `commercial_text_training_scope`, `participant_age_evidence` and `active_provider_grant`, each set to `verified` only after the reviewer checks the finding.
- `scope.json.sig` is the independent reviewer's binary OpenPGP detached signature of those exact receipt bytes, created with `gpg --local-user "$REVIEWER_KEY_FINGERPRINT" --output scope.json.sig --detach-sign scope.json` on the reviewer-controlled system. Retain the receipt SHA-256 independently. Use the authorization validity period and recheck withdrawal before each run.

```sh
python -B -m goxlm_review.transcript_intake \
  "$PRIVATE_ROOT" --scope-sha256 "$TRUSTED_RECEIPT_SHA256" --recipient "$APPROVED_RECIPIENT"
```

The independent signature authenticates the reviewer's assertion. SHA-256 proves byte integrity. The operator and reviewer establish whether underlying provider evidence grants access because the program cannot interpret arbitrary provider signature formats or verify provider revocations offline.

The script screens malformed or low-signal turns, sensitive-data patterns, quotation cues, held-out overlap, and exact or near-duplicate content. All repeated speakers and recording segments stay in one connected component, including links seen in excluded records. The deterministic preview aims for an 80/10/10 word split and reports actual component concentration. Fewer than three components block a three-way preview.

The hash-only preview is written once inside the encrypted root. It omits raw turns and speaker IDs, and stdout contains aggregate counts only. Every surviving record remains quarantined pending item-level privacy, incorporated-material, provenance, quality and rights checks. Preserve provider confidentiality and non-redistribution, prohibit re-identification, and apply provider-required withdrawal and deletion through the existing governance workflow. A successful run grants no training approval.
