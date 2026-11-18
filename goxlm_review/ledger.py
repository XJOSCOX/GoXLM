"""Reviewer decisions and portable hash-chain verification."""
import os
import pwd
import re
from datetime import date

from goxlm_v02.common import write_json
from goxlm_v02.data import catalog_lock, read, required

from .staging import case, config, content_hash, overlap_report

GENESIS = "0" * 64
REASONS = {"rights", "technical", "relevance"}
DECISIONS = {"approve", "hold", "reject", "revoke"}


def local_operator(root):
    operator = pwd.getpwuid(os.getuid()).pw_name
    if operator not in config(root)["operators"]:
        raise PermissionError(f"WSL login is not an authorized pilot operator: {operator}")
    return operator


def validate_action(row, decision, previous):
    action = required(decision, "decision")
    if action not in DECISIONS or required(decision, "category") not in REASONS:
        raise ValueError("Unknown decision or reason category")
    for key in ("reviewer", "actor", "rationale"):
        required(decision, key)
    when = required(decision, "decision_date")
    if date.fromisoformat(when).isoformat() != when or (previous and when < previous["decision_date"]):
        raise ValueError("Invalid or out-of-order review date")
    if decision["extract_sha256"] != row["extract"]["sha256"]:
        raise ValueError("Decision is not bound to the exact extract SHA-256")
    if action == "approve":
        if row["technical_status"] != "usable":
            raise ValueError("Technically unusable text cannot be approved")
        if decision["commercial_ai_training"] is not True:
            raise ValueError("Approval requires explicit commercial AI-training authorization")
        required(decision, "jurisdiction")
        if decision["findings_acknowledged"] is not True:
            raise ValueError("Approval requires acknowledging the case findings")
    elif decision["commercial_ai_training"] is not False:
        raise ValueError("Only an approval may authorize commercial training")
    prior = previous["decision"] if previous else None
    if prior == "revoke" or prior == "reject":
        raise ValueError("Closed decision. Create a new case with newly reviewed evidence.")
    if prior == "approve" and action != "revoke":
        raise ValueError("Withdraw an approval with an explicit revocation")
    if action == "revoke" and prior != "approve":
        raise ValueError("Revocation requires an active approval")
    if action == "revoke":
        if decision["category"] != "rights":
            raise ValueError("Rights revocation requires the rights category")
        required(decision, "notice_reference")


def load_history(root, envelopes, expected=None):
    cfg = config(root)
    paths = sorted((root / "decisions").glob("*.json"))
    previous, events, state = GENESIS, [], {}
    found = expected in (None, GENESIS)
    for index, path in enumerate(paths, 1):
        if path.name != f"{index:08d}.json":
            raise ValueError("Decision history has a missing or unexpected sequence")
        envelope = read(path)
        event = envelope["event"]
        head = content_hash(event)
        if (envelope["event_sha256"] != head or event["sequence"] != index
                or event["previous_sha256"] != previous
                or event["store_config_sha256"] != content_hash(cfg)):
            raise ValueError(f"Decision chain integrity failure at sequence {index}")
        key = event["candidate_id"]
        item = envelopes.get(key)
        if item is None or event["case_sha256"] != item["case_sha256"]:
            raise ValueError(f"Decision references missing or changed case: {key}")
        if event["actor"] not in cfg["operators"]:
            raise ValueError("Decision from unauthorized operator")
        validate_action(item["record"], event, state.get(key))
        state[key] = event
        events.append(envelope)
        previous = head
        found = found or head == expected
    if not found:
        raise ValueError("Decision chain does not contain the independently saved head")
    return previous, events, state


def all_cases(root):
    return {path.stem: case(root, path.stem) for path in sorted((root / "cases").glob("*.json"))}


def record(root, key, decision, confirmed_hash):
    actor = local_operator(root)
    with catalog_lock(root):
        cases = all_cases(root)
        item = cases[key]
        head, events, state = load_history(root, cases)
        if decision["decision"] in {"approve", "revoke"} and confirmed_hash != decision["extract_sha256"]:
            raise ValueError("Interactive confirmation does not match the exact SHA-256")
        # Bind to the case envelope, which includes original, extract, evidence and scope.
        payload = {**decision, "actor": actor, "sequence": len(events) + 1,
                   "previous_sha256": head, "store_config_sha256": content_hash(config(root)),
                   "case_sha256": item["case_sha256"], "candidate_id": key}
        validate_action(item["record"], payload, state.get(key))
        event = {"event": payload, "event_sha256": content_hash(payload)}
        write_json(root / "decisions" / f"{payload['sequence']:08d}.json", event)
    return event


def status(root, expected=None):
    if expected is not None and not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("Invalid saved head SHA-256")
    config(root)
    with catalog_lock(root):
        candidates = all_cases(root)
        head, events, state = load_history(root, candidates, expected)
        batches = 0
        for path in sorted((root / "batches").glob("*.json")):
            members = read(path)
            if path.stem != content_hash(members) or any(
                    key not in candidates or candidates[key]["case_sha256"] != sha
                    for key, sha in members.items()):
                raise ValueError(f"Batch manifest corrupted: {path.name}")
            batches += 1
    return candidates, head, events, state, batches


def report(root):
    cases, head, events, history, batches = status(root)
    rows = []
    for key, item in cases.items():
        case_row = item["record"]
        event = history.get(key, {})
        original = case_row["source"]["bytes"]
        extracted = case_row["extract"]["bytes"]
        usable = case_row["technical_status"] == "usable"
        rows.append({"id": key, "subject": case_row["domain"], "family_id": case_row["family_id"],
                     "case_sha256": item["case_sha256"], "source_sha256": case_row["source"]["sha256"],
                     "extract_sha256": case_row["extract"]["sha256"], "source_bytes": original,
                     "extract_bytes": extracted, "extract_to_source_ratio": extracted / original if original else None,
                     "proposed_byte_tokens": extracted if usable else 0,
                     "technical_status": case_row["technical_status"],
                     "review": event.get("decision", "pending"), "category": event.get("category"),
                     "legally_rejected": event.get("decision") == "reject" and event["category"] == "rights",
                     "rights_revoked": event.get("decision") == "revoke",
                     "extraction_cost": case_row["upstream_extraction_cost"], "findings": case_row["findings"]})
    return {"head": head, "decisions": len(events), "batches": batches,
            "legally_rejected": sum(row["legally_rejected"] for row in rows),
            "rights_revoked": sum(row["rights_revoked"] for row in rows),
            "technically_unusable": sum(row["technical_status"] != "usable" for row in rows),
            "documents": rows, "overlap": overlap_report(root, cases.values())}
