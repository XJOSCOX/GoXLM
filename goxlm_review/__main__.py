"""CLI for the isolated GoXLM acquisition review pilot."""
import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

from goxlm_v02.common import canonical, publish, write_json
from goxlm_v02.data import read

from .ledger import REASONS, local_operator, record, report, status
from .staging import TEMPLATES, batch, case, config, init, object_bytes


def show_json(value):
    print(canonical(value).decode(), end="")


def export_evidence(root, key, out, include_source):
    envelope = case(root, key)
    row = envelope["record"]
    refs = row["evidence"] + ([row["source"]] if include_source else [])
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_json(out / "case.json", envelope)
    for index, ref in enumerate(refs, 1):
        raw = object_bytes(root, ref["sha256"])
        publish(out / f"{index:02d}-{Path(ref['name']).name}", lambda stream: stream.write(raw))


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    main.add_argument("store", type=Path)
    commands = main.add_subparsers(dest="command", required=True)
    start = commands.add_parser("init")
    start.add_argument("--operator", action="append", required=True)
    if find_spec(".dossiers", package=__package__) is not None:
        stage = commands.add_parser("stage-dossiers")
        stage.add_argument("package", type=Path)
        stage.add_argument("--package-sha256", required=True)
        stage.add_argument("--dossier-index", type=Path, required=True)
    generic = commands.add_parser("batch")
    generic.add_argument("manifest", type=Path)
    generic.add_argument("--input-root", type=Path, required=True)
    template = commands.add_parser("template")
    template.add_argument("category", choices=TEMPLATES)
    commands.add_parser("list")
    commands.add_parser("report")
    commands.add_parser("head")
    verify = commands.add_parser("verify")
    verify.add_argument("--expected-head")
    display = commands.add_parser("show")
    display.add_argument("id")
    display.add_argument("--text", action="store_true")
    export = commands.add_parser("evidence")
    export.add_argument("id")
    export.add_argument("directory", type=Path)
    export.add_argument("--include-source", action="store_true")
    decide = commands.add_parser("decide")
    decide.add_argument("id")
    decide.add_argument("decision", choices=("approve", "hold", "reject", "revoke"))
    decide.add_argument("--extract-sha256", required=True)
    decide.add_argument("--reviewer", required=True)
    decide.add_argument("--date", required=True, dest="decision_date")
    decide.add_argument("--category", choices=sorted(REASONS), default="rights")
    decide.add_argument("--rationale", required=True)
    decide.add_argument("--jurisdiction")
    decide.add_argument("--notice-reference")
    decide.add_argument("--commercial-ai-training", action="store_true")
    decide.add_argument("--acknowledge-findings", action="store_true")
    return main


def main():
    command = parser()
    a = command.parse_args()
    root = a.store
    try:
        if a.command == "init":
            init(root, a.operator)
            return
        if a.command == "template":
            policy = TEMPLATES[a.category]
            show_json({"template": a.category, **policy, "batch_example": {"items": [{
                "id": "", "title": "", "domain": "", "family_id": "", "template": a.category,
                "attribution": "", "rights_summary": "", "license_conditions": "",
                "source": {"file": "", "sha256": "", "url": ""},
                "extract": {"file": "", "sha256": ""}, "extraction_method": "",
                "extraction_cost": {"cpu_ms": None, "wall_ms": None, "peak_rss_bytes": None},
                "evidence": [{"role": role, "file": "", "sha256": "", "reference": "", "url": ""}
                             for role in policy["evidence_roles"]], "exceptions": [],
            }]}})
            return
        config(root)
        if a.command == "stage-dossiers":
            from .dossiers import existing_batch

            items, base = existing_batch(a.package, a.package_sha256, a.dossier_index)
            show_json(batch(root, items, base))
        elif a.command == "batch":
            show_json(batch(root, read(a.manifest)["items"], a.input_root))
        elif a.command == "evidence":
            export_evidence(root, a.id, a.directory, a.include_source)
        elif a.command == "show":
            row = case(root, a.id)
            if a.text:
                sys.stdout.buffer.write(object_bytes(root, row["record"]["extract"]["sha256"]))
            else:
                show_json(row)
        elif a.command == "report":
            show_json(report(root))
        elif a.command == "decide":
            local_operator(root)
            if not sys.stdin.isatty():
                raise ValueError("Reviewer decisions require an interactive WSL terminal")
            item = case(root, a.id)
            if a.extract_sha256 != item["record"]["extract"]["sha256"]:
                raise ValueError("Review the correct extract SHA-256")
            print(f"{a.decision} {a.id} for case {item['case_sha256']}", file=sys.stderr)
            confirm = input("Type the full extract SHA-256 to record this decision: ").strip()
            if confirm != a.extract_sha256:
                raise ValueError("Confirmation does not match")
            decision = {"decision": a.decision, "reviewer": a.reviewer,
                        "decision_date": a.decision_date, "category": a.category,
                        "rationale": a.rationale, "commercial_ai_training": a.commercial_ai_training,
                        "jurisdiction": a.jurisdiction, "notice_reference": a.notice_reference,
                        "extract_sha256": a.extract_sha256,
                        "findings_acknowledged": a.acknowledge_findings}
            show_json(record(root, a.id, decision, confirm))
        else:
            cases, head, events, state, count = status(root, getattr(a, "expected_head", None))
            if a.command == "head":
                print(head)
            elif a.command == "verify":
                show_json({"verified": True, "head": head, "cases": len(cases), "batches": count,
                           "decisions": len(events)})
            else:
                for key, item in cases.items():
                    event = state.get(key, {})
                    print(key, event.get("decision", "pending"), item["record"]["technical_status"],
                          item["record"]["extract"]["sha256"], sep="\t")
    except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError) as exc:
        command.exit(2, f"review: {exc}\n")


if __name__ == "__main__":
    main()
