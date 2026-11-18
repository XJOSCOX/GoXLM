"""Local, deterministic acquisition records. This store has no catalog writer."""
import re
import time
from pathlib import Path
from uuid import uuid4

from goxlm_v02.common import canonical, digest, publish, write_json
from goxlm_v02.data import catalog_lock, document_id, read, required
from goxlm_v02.overlap import findings

SCHEMA = "goxlm-review-store-0.3.1"
DOMAINS = {"science", "math", "programming", "technology", "general"}
TEMPLATES = {
    "us_public_domain_expired": {
        "evidence_roles": ["rights_notice", "edition_provenance"],
        "checks": "Confirm exact printed edition, U.S. copyright status and later editorial or format additions.",
    },
    "us_government_work": {
        "evidence_roles": ["rights_notice", "official_duties"],
        "checks": "Confirm federal official duties, identify contractors and separate third-party components.",
    },
    "cc0": {
        "evidence_roles": ["rights_notice", "depositor_authority"],
        "checks": "Confirm exact version and file, scope of CC0 and evidence of depositor authority.",
    },
    "voluntary_public_domain": {
        "evidence_roles": ["dedication_notice", "document_provenance"],
        "checks": "Verify the dedication covers the exact document revision and review any item-specific license exceptions.",
    },
    "original_dialogue": {
        "evidence_roles": ["authorship_declaration", "signature_or_terms", "quality_review"],
        "checks": "Verify exact human authorship, commercial-training scope, consent and withdrawal terms.",
    },
    "explicit_commercial_ai": {
        "evidence_roles": ["training_grant", "pinned_provenance"],
        "checks": "Confirm the licensor, pinned files, training scope, attribution and redistribution terms.",
    },
}
PATTERNS = {
    "third_party_credit": r"\b(?:used by permission|reproduced with permission|courtesy of|adapted from)\b",
    "editorial_addition": r"\b(?:transcriber.s note|editor.s note|modern introduction)\b",
    "restrictive_terms": r"\b(?:all rights reserved|non.?commercial only|research use only|no derivatives)\b",
    "external_reference": r"https?://|\.(?:csv|parquet|zip|png)\b",
}


def content_hash(value):
    return digest(canonical(value))


def init(root, operators):
    if not operators or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name)
                            for name in operators):
        raise ValueError("Specify permitted WSL login names")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_json(root / "store.json", {"schema": SCHEMA, "operators": sorted(set(operators))})


def config(root):
    info = read(root / "store.json")
    if info.get("schema") != SCHEMA or not info.get("operators"):
        raise ValueError("Not a GoXLM review store")
    return info


def read_verified(path, expected):
    raw = path.read_bytes()
    if not re.fullmatch(r"[a-f0-9]{64}", expected) or digest(raw) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return raw


def object_bytes(root, sha):
    if not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("Invalid object SHA-256")
    return read_verified(root / "objects" / sha, sha)


def case(root, key):
    key = document_id(key)
    envelope = read(root / "cases" / (key + ".json"))
    row = envelope["record"]
    if row["id"] != key or envelope["case_sha256"] != content_hash(row):
        raise ValueError(f"Case digest mismatch: {key}")
    for ref in [row["source"], row["extract"], *row["evidence"]]:
        if len(object_bytes(root, ref["sha256"])) != ref["bytes"]:
            raise ValueError(f"Case object size mismatch: {key}")
    return envelope


def relative_file(base, name):
    p = (base / name).resolve()
    if not p.is_relative_to(base) or not p.is_file():
        raise ValueError(f"Input must be a file inside the declared input root: {name}")
    return p


def exception_scan(raw, notes):
    result = [{"kind": "declared_document_exception", "detail": note} for note in notes]
    try:
        value = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "invalid_utf8", result
    if not value.strip():
        return "empty_extract", result
    if any((ord(c) < 32 and c not in "\t\r\n") or c == "\ufffd" for c in value):
        return "corrupt_or_control_characters", result
    for kind, pattern in PATTERNS.items():
        lines = [i for i, line in enumerate(value.splitlines(), 1) if re.search(pattern, line, re.I)]
        if lines:
            result.append({"kind": kind, "extract_lines": lines})
    return "usable", result


def prepare_case(item, base):
    key = document_id(required(item, "id"))
    template = required(item, "template")
    policy = TEMPLATES.get(template)
    if policy is None:
        raise ValueError(f"Unknown rights template: {template}")
    domain = required(item, "domain")
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")
    objects = {}
    def capture(entry):
        name = required(entry, "file")
        expected = required(entry, "sha256")
        path = relative_file(base, name)
        raw = read_verified(path, expected)
        objects[expected] = path
        ref = {"name": Path(name).name, "sha256": expected, "bytes": len(raw)}
        if entry.get("url"):
            ref["url"] = required(entry, "url")
        return ref, raw

    source, source_raw = capture(item["source"])
    built_in_metrics = {}
    requested_extract = item["extract"]
    method = required(item, "extraction_method")
    if requested_extract.get("mode") == "utf8_lf_v1":
        if set(requested_extract) != {"mode"} or method != "utf8_lf_v1":
            raise ValueError("Built-in extraction requires an explicit utf8_lf_v1 recipe")
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        try:
            value = source_raw.decode("utf-8-sig")
            raw = value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        except UnicodeDecodeError:
            # Keep the bytes accessible for inspection. Quality screening rejects them.
            raw = source_raw
        built_in_metrics = {"extract_wall_ns": time.perf_counter_ns() - wall,
                            "extract_cpu_ns": time.process_time_ns() - cpu}
        sha = digest(raw)
        extract = {"name": key + ".txt", "sha256": sha, "bytes": len(raw)}
        if sha not in objects:
            objects[sha] = raw
    else:
        extract, raw = capture(requested_extract)
    evidence = []
    for entry in item["evidence"]:
        ref, content = capture(entry)
        if not content.strip():
            raise ValueError(f"Empty rights evidence: {key}")
        ref["role"] = required(entry, "role")
        ref["reference"] = required(entry, "reference")
        evidence.append(ref)
    if not set(policy["evidence_roles"]).issubset({ref["role"] for ref in evidence}):
        raise ValueError(f"Missing template evidence roles: {key}")
    notes = item.get("exceptions", [])
    if not isinstance(notes, list) or any(not isinstance(n, str) or not n.strip() for n in notes):
        raise ValueError("Exceptions must be a list of text notes")
    quality, flagged = exception_scan(raw, notes)
    provenance = item.get("extraction_cost", {})
    if not isinstance(provenance, dict) or provenance.keys() - {"cpu_ms", "wall_ms", "peak_rss_bytes"}:
        raise ValueError("Invalid upstream extraction metrics")
    for value in provenance.values():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Invalid upstream extraction metric value")
    row = {"id": key, "domain": domain, "family_id": required(item, "family_id"),
           "title": required(item, "title"), "template": template,
           "template_checks": policy["checks"], "rights_summary": required(item, "rights_summary"),
           "license_conditions": required(item, "license_conditions"),
           "attribution": required(item, "attribution"), "extraction_method": method,
           "source": source, "extract": extract,
           "evidence": sorted(evidence, key=lambda r: (r["role"], r["sha256"], r["name"], r["reference"], r.get("url", ""))),
           "upstream_extraction_cost": provenance, "technical_status": quality, "findings": flagged}
    return {"record": row, "case_sha256": content_hash(row)}, objects, built_in_metrics


def batch(root, items, input_root):
    config(root)
    base = input_root.resolve(strict=True)
    prepared, costs = {}, {}
    started = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    if not isinstance(items, list) or not items:
        raise ValueError("Batch needs a nonempty list of items")
    for item in items:
        start, cpu = time.perf_counter_ns(), time.process_time_ns()
        envelope, objects, metrics = prepare_case(item, base)
        key = envelope["record"]["id"]
        if key in prepared:
            raise ValueError(f"Repeated candidate ID in batch: {key}")
        prepared[key] = envelope, objects
        costs[key] = {"capture_wall_ns": time.perf_counter_ns() - start,
                      "capture_cpu_ns": time.process_time_ns() - cpu, **metrics}
    members = {key: pair[0]["case_sha256"] for key, pair in sorted(prepared.items())}
    version = content_hash(members)
    with catalog_lock(root):
        # Validate conflicts for the entire batch before writing any objects.
        for key in members:
            path = root / "cases" / (key + ".json")
            if path.exists() and case(root, key)["case_sha256"] != members[key]:
                raise ValueError(f"Candidate ID already binds different bytes or evidence: {key}")
        for key, (envelope, objects) in sorted(prepared.items()):
            new_bytes = 0
            for sha, staged in objects.items():
                path = root / "objects" / sha
                if path.exists():
                    object_bytes(root, sha)
                else:
                    raw = staged if isinstance(staged, bytes) else read_verified(staged, sha)
                    publish(path, lambda stream: stream.write(raw))
                    new_bytes += len(raw)
            costs[key]["new_blob_bytes_written"] = new_bytes
            path = root / "cases" / (key + ".json")
            if not path.exists():
                write_json(path, envelope)
        saved = root / "batches" / (version + ".json")
        if saved.exists():
            if read(saved) != members:
                raise ValueError("Batch manifest mismatch")
        else:
            write_json(saved, members)
        run = root / "runs" / (uuid4().hex + ".json")
        write_json(run, {"batch_sha256": version, "items": costs,
                         "wall_ns": time.perf_counter_ns() - started,
                         "cpu_ns": time.process_time_ns() - cpu_start})
    return {"batch_sha256": version, "run_report": str(run), "documents": len(members)}


def overlap_report(root, envelopes):
    rows = []
    for envelope in envelopes:
        row = envelope["record"]
        if row["technical_status"] == "usable":
            rows.append({"document_id": row["id"], "family_id": row["family_id"],
                         "text": object_bytes(root, row["extract"]["sha256"]).decode()})
    return findings(rows)
