"""Audit catalog integrity, corpus overlap, and checkpoint lineage."""
import argparse
from pathlib import Path
import re

from .common import canonical
from .data import (affected_documents, document_id, freeze_path, load_release,
                   read, revoke, revoked, validate_review, validate_submission)
from .overlap import demo_hits, findings


def inventory(catalog):
    records = revoked(catalog)
    errors, releases = [], {}
    intake = {path.stem: path for path in (catalog / "intake").glob("*.json")}
    reviews = {path.stem: path for path in (catalog / "reviews").glob("*.json")}
    errors.extend(f"Orphan review: {key}" for key in sorted(reviews.keys() - intake.keys()))
    for key, path in sorted(intake.items()):
        try:
            submission = read(path)
            validate_submission(submission)
            if document_id(submission["metadata"]["document_id"]) != key:
                raise ValueError("Intake document ID mismatch")
            if key in reviews:
                validate_review(submission, read(reviews[key]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path}: {exc}")
    for path in sorted((catalog / "releases").glob("*.json")):
        try:
            payload = load_release(catalog, path, historical=True)
            rows = payload["documents"]
            blocked = set(affected_documents(rows, records))
            releases[path.stem] = {"permanent": payload.get("permanent", False),
                                   "affected": {row["document_id"]: row["split"]
                                                for row in rows if row["document_id"] in blocked}}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path}: {exc}")
    for permanent in (False, True):
        if freeze_path(catalog, permanent).exists() and not any(
                release["permanent"] == permanent for release in releases.values()):
            errors.append(f"No valid release for {freeze_path(catalog, permanent).name}")
    return {"revocations": sorted(records), "releases": releases, "errors": errors}


def checkpoint_impact(inventory_report, roots):
    import torch

    from .run import CHECKPOINT_SCHEMAS

    paths = set()
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_dir():
            paths.update(path.resolve() for path in root.rglob("*.pt"))
        else:
            paths.add(root.resolve())
    result = []
    for path in sorted(paths):
        row = {"path": str(path)}
        try:
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            version = checkpoint.get("dataset_version")
            if checkpoint.get("schema") not in CHECKPOINT_SCHEMAS or not isinstance(
                    version, str) or not re.fullmatch(r"[0-9a-f]{64}", version):
                row["status"] = "untracked_format"
            else:
                release = inventory_report["releases"].get(version)
                row["dataset_version"] = version
                row["status"] = ("unresolved_release" if release is None else
                                 "affected" if release["affected"] else "no_recorded_revocation")
                if release is not None:
                    row["affected"] = release["affected"]
        except Exception as exc:
            # A bad file must not hide results for other checkpoints.
            row.update(status="unreadable", error=str(exc))
        result.append(row)
    return result


def overlap_report(catalog, release, against=None):
    rows = load_release(catalog, release, historical=True)["documents"]
    pairs = findings(rows)
    return {"release": release.stem,
            "cross_split": [pair for pair in pairs if pair["left_split"] != pair["right_split"]],
            "within_split": [pair for pair in pairs if pair["left_split"] == pair["right_split"]],
            "legacy_demo_overlap": demo_hits(rows),
            "against_release": against.stem if against else None,
            "between_releases": findings(rows, load_release(
                catalog, against, historical=True)["documents"]) if against else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("integrity")
    record = sub.add_parser("revoke")
    record.add_argument("document_id")
    record.add_argument("decision", type=Path)
    impact = sub.add_parser("impact")
    impact.add_argument("checkpoints", type=Path, nargs="+", help="Trusted local files or directories")
    overlap = sub.add_parser("overlap")
    overlap.add_argument("release", type=Path)
    overlap.add_argument("--against", type=Path)
    args = parser.parse_args()
    if not args.catalog.is_dir():
        parser.error("Catalog directory is missing")
    if args.command == "revoke":
        revoke(args.catalog, args.document_id, read(args.decision))
        report = inventory(args.catalog)
    elif args.command == "overlap":
        report = overlap_report(args.catalog, args.release, args.against)
    else:
        report = inventory(args.catalog)
        if args.command == "impact":
            report["checkpoints"] = checkpoint_impact(report, args.checkpoints)
    print(canonical(report).decode(), end="")
    if (report.get("errors") or report.get("cross_split") or report.get("legacy_demo_overlap")
            or any(item["status"] in {"unreadable", "unresolved_release"}
                   for item in report.get("checkpoints", []))):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
