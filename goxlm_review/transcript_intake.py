"""Offline, private screening of an authorized transcript-only delivery."""

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path

from goxlm_v02.common import ROOT, canonical, digest, write_json
from goxlm_v02.overlap import terms

POLICY = "goxlm-private-transcripts-v1"
FILES = {
    "agreement": ("agreement.txt", 2 << 20),
    "provider_approval": ("provider-approval.txt", 2 << 20),
    "delivery": ("transcripts.jsonl", 512 << 20),
    "heldout": ("heldout.json", 16 << 20),
}
ID = re.compile(r"[a-z]_[0-9a-f]{12,64}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
WORDS = re.compile(r"\w", re.UNICODE)
PRIVACY = re.compile(
    r"https?://|www\.|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|"
    r"\b(?:\d{3}-\d{2}-\d{4}|(?:\+1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4})\b|"
    r"\b(?:my (?:full |real )?name is|i live at|my (?:home )?address is|"
    r"email me at|call me at|my (?:password|diagnosis|passport number|birthday) is|"
    r"social security number|bank account number|credit card number)\b|"
    r"\b\d{1,6}\s+\S+(?:\s+\S+){0,2}\s+(?:street|road|avenue|lane|drive|boulevard)\b",
    re.I,
)
INCORPORATED = re.compile(r"\"|[“”«»]|\b(?:quoting|the lyrics|excerpt from|according to the article|copyright)\b", re.I)
FILLERS = {"um", "uh", "erm", "hmm"}


class Components:
    def __init__(self):
        self.parent = {}

    def root(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, first, second):
        self.parent[self.root(second)] = self.root(first)


def transcript(row):
    if not isinstance(row, dict) or set(row) != {
        "id", "recording_id", "speakers", "participant_status", "language",
        "segment_start_ms", "segment_end_ms", "turns",
    }:
        raise ValueError("Unexpected transcript record schema.")
    if not all(isinstance(row[k], str) and ID.fullmatch(row[k]) and row[k].startswith(prefix)
               for k, prefix in (("id", "c_"), ("recording_id", "r_"))):
        raise ValueError("Invalid opaque transcript or recording ID.")
    spe = row["speakers"]
    if (not isinstance(spe, list) or len(spe) != 2
            or not all(isinstance(k, str) and k.startswith("s_") and ID.fullmatch(k) for k in spe)
            or spe[0] == spe[1]):
        raise ValueError("Use two distinct stable opaque speaker IDs.")
    if row["participant_status"] not in ("all_adult_verified", "contains_minor", "unknown"):
        raise ValueError("Invalid adult eligibility marker.")
    if not isinstance(row["language"], str):
        raise ValueError("Invalid language marker.")
    start, end = row["segment_start_ms"], row["segment_end_ms"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("Invalid segment bounds.")
    if not isinstance(row["turns"], list) or not 1 <= len(row["turns"]) <= 100_000:
        raise ValueError("Invalid transcript turns.")
    latest = start
    last_end = [start, start]
    for t in row["turns"]:
        if not isinstance(t, dict) or set(t) != {"speaker", "start_ms", "end_ms", "text"}:
            raise ValueError("Unexpected turn fields.")
        s, a, b = t["speaker"], t["start_ms"], t["end_ms"]
        if (type(s) is not int or s not in (0, 1) or type(a) is not int or type(b) is not int
                or not latest <= a < b <= end or a < last_end[s]):
            raise ValueError("Turn timing is unordered or outside the recording segment.")
        if not isinstance(t["text"], str) or not t["text"].strip() or len(t["text"]) > 20_000:
            raise ValueError("Invalid turn text.")
        latest, last_end[s] = a, b
    return row


def words(row):
    return [t for turn in row["turns"] for t in terms(turn["text"]) if WORDS.search(t)]


def quality_flags(row, tokens):
    text = "\n".join(turn["text"] for turn in row["turns"])
    if PRIVACY.search(text):
        return "possible_private_information"
    if INCORPORATED.search(text):
        return "possible_incorporated_text"
    if "�" in text or any(not (c.isprintable() or c in "\t\n") for c in text):
        return "invalid_or_corrupted_text"
    if (row["language"] != "en" or len(tokens) < 16
            or len({t["speaker"] for t in row["turns"]}) < 2):
        return "language_or_low_dialogue_volume"
    if sum(w in FILLERS for w in tokens) / len(tokens) > .4 or max(Counter(tokens).values()) / len(tokens) > .4:
        return "filler_or_repetition"
    return None


def ngrams(tokens, length):
    return zip(*(tokens[i:] for i in range(length)))


def word_hash(span):
    return hashlib.blake2b(" ".join(span).encode(), digest_size=12).digest()


def split_groups(groups, seed):
    if len(groups) < 3:
        return None
    weights = {key: sum(count for _, count in pack) for key, pack in groups.items()}
    ordered = sorted(groups, key=lambda k: (-weights[k], digest(canonical([seed, k]))))
    assignment = {}
    totals = dict.fromkeys(("train", "validation", "test"), 0)
    goals = {key: sum(weights.values()) * part
             for key, part in zip(totals, (.8, .1, .1))}
    for i, group in enumerate(ordered):
        if len(ordered) - i == sum(n == 0 for n in totals.values()):
            choices = [key for key in totals if totals[key] == 0]
        else:
            choices = list(totals)
        split = max(choices, key=lambda k: goals[k] - totals[k])
        totals[split] += weights[group]
        assignment[group] = split
    return assignment


def screen(records, heldout, seed=42):
    """Return a hash-only preview. Every retained record needs a rights/privacy/quality decision."""
    if not isinstance(heldout, list) or not heldout or not all(isinstance(t, str) and t.strip() for t in heldout):
        raise ValueError("Provide a nonempty list of protected evaluation texts.")
    frozen = [[t for t in terms(text) if WORDS.search(t)] for text in heldout]
    exact_eval = {word_hash(text) for text in frozen}
    protected = {word_hash(s) for text in frozen for s in ngrams(text, 12)}
    components = Components()
    ids, recordings, bad_recordings = set(), {}, set()
    selected = []
    for row in records:
        row = transcript(row)
        key = row["id"]
        if key in ids:
            raise ValueError("Repeated opaque conversation ID.")
        ids.add(key)
        recording = row["recording_id"]
        a, b = row["speakers"]
        pair = frozenset((a, b))
        if recordings.setdefault(recording, pair) != pair:
            raise ValueError("Source recording has inconsistent speakers.")
        if row["participant_status"] != "all_adult_verified":
            bad_recordings.add(recording)
        components.join(recording, a)
        components.join(recording, b)
        selected.append(row)
    drops = Counter()
    exact_seen, passage_seen, sketches = set(), set(), []
    retained = []
    for row in sorted(selected, key=lambda r: r["id"]):
        if row["recording_id"] in bad_recordings:
            drops["minor_or_unverified_age_recording"] += 1
            continue
        tokens = words(row)
        issue = quality_flags(row, tokens)
        if issue:
            drops[issue] += 1
            continue
        exact = word_hash(tokens)
        if exact in exact_eval or any(word_hash(span) in protected for span in ngrams(tokens, 12)):
            drops["heldout_overlap"] += 1
            continue
        spans = {word_hash(span) for span in ngrams(tokens, 24)}
        # Min-hash approximation for modified copies, plus exact contiguous spans
        # for overlapping clips. Neither can certify absence of paraphrasing.
        sketch = frozenset(sorted({word_hash(span) for span in ngrams(tokens, 5)})[:128])
        if exact in exact_seen or spans & passage_seen or any(
            len(sketch & other) >= 8 and len(sketch & other) * 2 >= len(sketch | other)
            for other in sketches
        ):
            drops["exact_near_or_passage_duplicate"] += 1
            continue
        exact_seen.add(exact)
        passage_seen.update(spans)
        sketches.append(sketch)
        retained.append((row, len(tokens)))
    groups = defaultdict(list)
    for row, count in retained:
        groups[components.root(row["recording_id"])].append((row, count))
    # Hash component members rather than exposing speaker identities in output.
    groups = {digest(canonical(sorted(r["id"] for r, _ in pack))): pack for pack in groups.values()}
    assignment = split_groups(groups, seed)
    split_counts, split_words, split_components = Counter(), Counter(), Counter()
    manifest = []
    for group, pack in sorted(groups.items()):
        split = assignment[group] if assignment else None
        if split:
            split_components[split] += 1
        for r, count in pack:
            if split:
                split_counts[split] += 1
                split_words[split] += count
            manifest.append({"record_id": r["id"], "record_sha256": digest(canonical(r)),
                             "component_sha256": group, "words": count, "split_preview": split,
                             "status": "quarantined_pending_item_review"})
    return {"policy": POLICY, "training_authorized": False, "input_records": len(ids),
            "screened_records": len(manifest), "excluded_counts": dict(drops),
            "speaker_connected_components": len(groups), "split_preview_counts": dict(split_counts),
            "split_preview_words": dict(split_words), "split_preview_components": dict(split_components),
            "split_preview_available": assignment is not None,
            "largest_component_records": max(map(len, groups.values()), default=0),
            "largest_component_word_share": max(
                (sum(count for _, count in pack) for pack in groups.values()), default=0
            ) / max(1, sum(r["words"] for r in manifest)), "records": manifest}


def encrypted_private_root(path):
    path = path.absolute()
    if path.is_relative_to(ROOT.resolve()):
        raise ValueError("Keep the private intake root outside the project checkout.")
    info = path.lstat()
    if (path.resolve(strict=True) != path or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("Use a real mode-0700 root owned by the current operator.")
    block = Path(f"/sys/dev/block/{os.major(info.st_dev)}:{os.minor(info.st_dev)}/dm/uuid")
    try:
        encrypted = block.read_text(encoding="ascii").startswith("CRYPT-")
    except OSError:
        encrypted = False
    if not encrypted:
        raise ValueError("Use a filesystem directly on a verified Linux dm-crypt device.")
    return path


def private_file(root, filename, cap):
    with os.fdopen(os.open(root / filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        mode = os.fstat(stream.fileno())
        if not stat.S_ISREG(mode.st_mode) or mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) != 0o600:
            raise ValueError("Input files must be real mode-0600 files owned by the operator.")
        raw = stream.read(cap + 1)
        if not raw or len(raw) > cap:
            raise ValueError("Input file is empty or exceeds the stated limit.")
        return raw


def intake(root, anchor, recipient):
    root = encrypted_private_root(root)
    receipt = private_file(root, "scope.json", 64 << 10)
    if not SHA.fullmatch(anchor) or digest(receipt) != anchor:
        raise ValueError("Authorization receipt does not match its trusted SHA-256 anchor.")
    scope = json.loads(receipt)
    if not isinstance(scope, dict) or set(scope) != {"provider", "recipient", "scope", "status", "files", "delivery_id"}:
        raise ValueError("Missing or unexpected fields in the authorization receipt.")
    if (scope["provider"] != "Open Yap 1K" or scope["recipient"] != recipient or not recipient.strip()
            or scope["scope"] != "approved_transcript_only_commercial_training"
            or scope["status"] != "active" or not isinstance(scope["delivery_id"], str)
            or not ID.fullmatch(scope["delivery_id"]) or not scope["delivery_id"].startswith("d_")):
        raise ValueError("Recipient or authorized delivery scope mismatch.")
    if not isinstance(scope["files"], dict) or set(scope["files"]) != set(FILES):
        raise ValueError("Bind agreement, approval, delivery and held-out evidence.")
    for key, wanted in scope["files"].items():
        if not isinstance(wanted, str) or not SHA.fullmatch(wanted):
            raise ValueError("Invalid bound file digest.")
    payloads = {}
    for key, (name, cap) in FILES.items():
        value = private_file(root, name, cap)
        if digest(value) != scope["files"][key]:
            raise ValueError("A bound input file differs from the authorization receipt.")
        if key in {"delivery","heldout"}:
            payloads[key] = value
    records = (json.loads(line) for line in payloads["delivery"].splitlines() if line.strip())
    report = screen(records, json.loads(payloads["heldout"]))
    report.update(scope_receipt_sha256=anchor, bound_files=scope["files"], delivery_id=scope["delivery_id"])
    write_json(root / "screened-preview.json", report)
    return {key: val for key, val in report.items() if key not in {"records", "delivery_id", "bound_files"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_root", type=Path)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--recipient", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(intake(args.private_root, args.scope_sha256, args.recipient), sort_keys=True))
    except (OSError, TypeError, ValueError):
        # Diagnostics must not echo a transcript, speaker ID or provider credential.
        parser.exit(2, "Transcript screening stopped. Check the private intake requirements.\n")


if __name__ == "__main__":
    main()
