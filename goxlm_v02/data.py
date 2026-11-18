"""Local intake, recorded human decisions, and reproducible dataset releases."""
import argparse
import base64
import fcntl
from contextlib import contextmanager
import re
import unicodedata
from datetime import date
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
import json

from .common import canonical, digest, write_json
from .overlap import demo_hits, findings, fingerprint

SCHEMA = "goxlm-dataset-0.2.1"
DOMAINS = {"science", "technology", "math", "programming"}
FIELDS = ("document_id", "source_id", "family_id", "title", "domain", "rights_holder",
          "source_reference", "rights_basis", "evidence_reference")


def read(path):
    return json.loads(path.read_bytes())


def required(record, field):
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {field}")
    return value.strip()


def document_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value):
        raise ValueError("Invalid document ID")
    return value


def normalize(raw):
    # Preserve indentation and mathematical notation.
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise ValueError("Empty UTF-8 document")
    return text


def validate_submission(submission):
    metadata = submission["metadata"]
    for field in FIELDS:
        required(metadata, field)
    document_id(metadata["document_id"])
    if metadata["domain"] not in DOMAINS:
        raise ValueError("Unsupported subject domain")
    for kind in ("document", "evidence"):
        raw = base64.b64decode(submission[kind], validate=True)
        if not raw.strip() or digest(raw) != submission[kind + "_sha256"]:
            raise ValueError(f"Empty or altered {kind}")
    return normalize(base64.b64decode(submission["document"]))


def intake(catalog, metadata, document, evidence):
    meta = {field: required(metadata, field) for field in FIELDS}
    submission = {"metadata": meta}
    for kind, path in (("document", document), ("evidence", evidence)):
        raw = path.read_bytes()
        submission[kind] = base64.b64encode(raw).decode("ascii")
        submission[kind + "_sha256"] = digest(raw)
    validate_submission(submission)
    write_json(catalog / "intake" / (meta["document_id"] + ".json"), submission)


def review(catalog, doc_id, decision):
    doc_id = document_id(doc_id)
    submission = read(catalog / "intake" / (doc_id + ".json"))
    validate_submission(submission)
    if submission["metadata"]["document_id"] != doc_id:
        raise ValueError("Intake document ID mismatch")
    fields = ("submission_sha256", "decision", "reviewer", "review_date",
              "rationale", "permissions_scope")
    decision = {field: required(decision, field) for field in fields}
    validate_review(submission, decision)
    write_json(catalog / "reviews" / (doc_id + ".json"), decision)


def validate_review(submission, decision):
    if required(decision, "submission_sha256") != digest(canonical(submission)):
        raise ValueError("Decision is not bound to this submission and evidence")
    if required(decision, "decision") not in {"approved", "rejected"}:
        raise ValueError("Invalid reviewer decision")
    required(decision, "reviewer")
    required(decision, "rationale")
    parsed = date.fromisoformat(required(decision, "review_date"))
    if parsed.isoformat() != decision["review_date"]:
        raise ValueError("Use YYYY-MM-DD review date")
    scope = required(decision, "permissions_scope")
    if decision["decision"] == "approved" and scope != "commercial_ai_training":
        raise ValueError("Approval must explicitly cover commercial AI training")


def comparable(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def clusters(rows, extra_links=()):
    parents = list(range(len(rows)))
    texts = [comparable(row["text"]) for row in rows]

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, j in combinations(range(len(rows)), 2):
        left, right = rows[i], rows[j]
        same_family = left["family_id"] == right["family_id"]
        same_source = left["source_id"] == right["source_id"]
        exact = left["document_sha256"] == right["document_sha256"] or texts[i] == texts[j]
        near = False
        if not (same_family or same_source or exact):
            matcher = SequenceMatcher(None, texts[i], texts[j], autojunk=False)
            near = matcher.quick_ratio() >= 0.85 and matcher.ratio() >= 0.85
            if not near:
                short, long = sorted((texts[i], texts[j]), key=len)
                near = len(short) >= 64 and short in long
        if same_family or same_source or exact or near:
            parents[root(j)] = root(i)
    by_id = {row["document_id"]: i for i, row in enumerate(rows)}
    for link in extra_links:
        parents[root(by_id[link["right"]])] = root(by_id[link["left"]])
    grouped = {}
    for i, row in enumerate(rows):
        grouped.setdefault(root(i), []).append(row)
    return list(grouped.values())


def assign(rows, seed, frozen):
    held_out = {} if frozen is None else frozen
    by_id = {row["document_id"]: row for row in rows}
    for key, fingerprint in held_out.items():
        if key not in by_id or by_id[key]["submission_sha256"] != fingerprint:
            raise ValueError(f"Frozen test document missing or changed: {key}")
    for group in clusters(rows):
        ids = {row["document_id"] for row in group}
        if ids.intersection(held_out):
            if not ids.issubset(held_out):
                raise ValueError("New document overlaps a frozen test group")
            split = "test"
        else:
            bucket = int(digest(canonical([seed, sorted(ids)]))[:16], 16) % 100
            split = ("train" if bucket < 80 else "validation" if bucket < 90 else "test"
                     ) if frozen is None else ("train" if bucket < 80 or bucket >= 90 else "validation")
        for row in group:
            row["split"] = split
    if {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError("Need independent approved groups in all three splits")


def freeze_path(catalog, permanent):
    return catalog / ("test-freeze.permanent.0.3.json" if permanent else "test-freeze.0.2.json")


def permanent_gate(rows):
    hits = demo_hits(rows)
    if hits:
        raise ValueError(f"GoXLM-0.1 demonstration overlap: {', '.join(hits)}")
    overlaps = findings(rows)
    if any(item["left_split"] != item["right_split"] for item in overlaps):
        raise ValueError("Independent overlap audit found cross-split contamination")
    groups = clusters(rows, overlaps)
    sizes = {"train": [], "validation": [], "test": []}
    for group in groups:
        # Count at most one document per related group toward the minimum.
        sizes[group[0]["split"]].append(max(len(row["text"].encode()) for row in group))
    if len(groups) < 40 or sum(map(sum, sizes.values())) < 256 * 1024:
        raise ValueError("Permanent test freeze requires 40 independent groups and 256 KiB")
    for split in ("validation", "test"):
        if len(sizes[split]) < 3 or sum(sizes[split]) < 16 * 1024:
            raise ValueError(f"Permanent {split} requires three groups and 16 KiB")


def read_approved(catalog, key):
    submission = read(catalog / "intake" / (key + ".json"))
    decision = read(catalog / "reviews" / (key + ".json"))
    text = validate_submission(submission)
    validate_review(submission, decision)
    if submission["metadata"]["document_id"] != key or decision["decision"] != "approved":
        raise ValueError(f"No matching approval for {key}")
    return submission, text


def revoked(catalog):
    result = {}
    for path in sorted((catalog / "revocations").glob("*.json")):
        record = read(path)
        key = document_id(required(record, "document_id"))
        if path.stem != key or record.get("schema") != "goxlm-revocation-0.3.1":
            raise ValueError(f"Invalid revocation record {path.name}")
        for field in ("reviewer", "reason", "notice_reference"):
            required(record, field)
        when = required(record, "revoked_on")
        if date.fromisoformat(when).isoformat() != when:
            raise ValueError("Invalid revocation date")
        submission, text = read_approved(catalog, key)
        if (record["submission_sha256"] != digest(canonical(submission))
                or record["text_fingerprint"] != fingerprint(text)):
            raise ValueError(f"Revocation fingerprint mismatch: {key}")
        result[key] = record
    return result


def affected_documents(rows, records):
    blocked = {record["text_fingerprint"] for record in records.values()}
    return sorted(row["document_id"] for row in rows
                  if row["document_id"] in records or fingerprint(row["text"]) in blocked)


def revoke(catalog, key, decision):
    key = document_id(key)
    with catalog_lock(catalog):
        revoked(catalog)
        submission, text = read_approved(catalog, key)
        fields = ("submission_sha256", "reviewer", "revoked_on", "reason", "notice_reference")
        record = {name: required(decision, name) for name in fields}
        if record["submission_sha256"] != digest(canonical(submission)):
            raise ValueError("Revocation must reference the approved submission hash")
        when = record["revoked_on"]
        if date.fromisoformat(when).isoformat() != when:
            raise ValueError("Invalid revocation date")
        record.update(schema="goxlm-revocation-0.3.1", document_id=key,
                      text_fingerprint=fingerprint(text))
        write_json(catalog / "revocations" / (key + ".json"), record)


@contextmanager
def catalog_lock(catalog):
    catalog.mkdir(parents=True, exist_ok=True)
    with (catalog / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def build(catalog, seed=42, *, permanent=False):
    with catalog_lock(catalog):
        return _build_locked(catalog, seed, permanent)


def _build_locked(catalog, seed, permanent):
    rows = []
    for path in sorted((catalog / "intake").glob("*.json")):
        review_path = catalog / "reviews" / path.name
        if not review_path.exists():
            continue
        submission, decision = read(path), read(review_path)
        text = validate_submission(submission)
        validate_review(submission, decision)
        metadata = submission["metadata"]
        if metadata["document_id"] != path.stem:
            raise ValueError("Intake document ID mismatch")
        if decision["decision"] == "approved":
            rows.append({**metadata, "text": text, "review": decision,
                         "document_sha256": submission["document_sha256"],
                         "evidence_sha256": submission["evidence_sha256"],
                         "submission_sha256": digest(canonical(submission))})
    excluded = set(affected_documents(rows, revoked(catalog)))
    rows = [row for row in rows if row["document_id"] not in excluded]
    freeze = freeze_path(catalog, permanent)
    frozen = read(freeze) if freeze.exists() else None
    if frozen is None and any(
            read(path).get("permanent", False) == permanent
            for path in (catalog / "releases").glob("*.json")):
        raise ValueError("Release history exists but the test freeze is missing")
    if frozen is not None and (frozen["schema"] != SCHEMA or frozen["seed"] != seed):
        raise ValueError("Frozen catalog uses a different split version or seed")
    if permanent and demo_hits(rows):
        raise ValueError("GoXLM-0.1 demonstration overlap in permanent corpus")
    assign(rows, seed, None if frozen is None else frozen["documents"])
    if permanent:
        permanent_gate(rows)
    if frozen is None:
        frozen = {"schema": SCHEMA, "seed": seed, "documents": {
            row["document_id"]: row["submission_sha256"] for row in rows if row["split"] == "test"
        }}
        write_json(freeze, frozen)
    # Retain provenance in the release and exact bytes/evidence in the catalog.
    payload = {"schema": SCHEMA, "seed": seed, "test_freeze": frozen, "documents": rows}
    if permanent:
        payload["permanent"] = True
    version = digest(canonical(payload))
    output = catalog / "releases" / (version + ".json")
    if output.exists():
        if output.read_bytes() != canonical(payload):
            raise ValueError("Existing release does not match its content hash")
    else:
        write_json(output, payload)
    return output


def load_release(catalog, path, *, historical=False):
    payload = read(path)
    if payload["schema"] != SCHEMA or path.stem != digest(canonical(payload)):
        raise ValueError("Invalid release version or digest")
    permanent = payload.get("permanent", False)
    if type(permanent) is not bool:
        raise ValueError("Invalid release kind")
    if payload["test_freeze"] != read(freeze_path(catalog, permanent)):
        raise ValueError("Release test set differs from catalog freeze")
    rows = payload["documents"]
    if len({row["document_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated document ID")
    for row in rows:
        key = document_id(row["document_id"]) + ".json"
        submission, decision = read(catalog / "intake" / key), read(catalog / "reviews" / key)
        text = validate_submission(submission)
        validate_review(submission, decision)
        expected = {**submission["metadata"], "text": text, "review": decision,
                    "document_sha256": submission["document_sha256"],
                    "evidence_sha256": submission["evidence_sha256"],
                    "submission_sha256": digest(canonical(submission))}
        if decision["decision"] != "approved" or expected != {
                key: value for key, value in row.items() if key != "split"}:
            raise ValueError("Release does not match approved evidence")
    if {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError("Invalid split coverage")
    if {row["document_id"]: row["submission_sha256"] for row in rows if row["split"] == "test"
        } != payload["test_freeze"]["documents"]:
        raise ValueError("Invalid frozen test assignments")
    expected_rows = [dict(row) for row in rows]
    assign(expected_rows, payload["seed"], payload["test_freeze"]["documents"])
    if expected_rows != rows:
        raise ValueError("Release assignments differ from the split algorithm")
    if permanent:
        permanent_gate(rows)
    affected = affected_documents(rows, revoked(catalog))
    if affected and not historical:
        raise ValueError(f"Release contains revoked documents: {', '.join(affected)}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    incoming = commands.add_parser("intake")
    for name in ("metadata", "document", "evidence"):
        incoming.add_argument(name, type=Path)
    approval = commands.add_parser("review")
    approval.add_argument("document_id")
    approval.add_argument("decision_file", type=Path)
    release = commands.add_parser("build")
    release.add_argument("--seed", type=int, default=42)
    release.add_argument("--permanent", action="store_true")
    args = parser.parse_args()
    if args.command == "intake":
        intake(args.catalog, read(args.metadata), args.document, args.evidence)
        print(digest(canonical(read(args.catalog / "intake" /
                                   (read(args.metadata)["document_id"].strip() + ".json")))))
    elif args.command == "review":
        review(args.catalog, args.document_id, read(args.decision_file))
    else:
        print(build(args.catalog, args.seed, permanent=args.permanent))


if __name__ == "__main__":
    main()
