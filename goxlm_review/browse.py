"""Browse hash-verified rights-review cases and optionally record elapsed effort."""

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from uuid import uuid4

from goxlm_v02.common import write_json

from .ledger import all_cases, load_history
from .staging import object_bytes


PAGE_LINES = 30


def token_proxy(size):
    return [round(size / 5), round(size / 3)]


def verified_queue(items, cases):
    if not isinstance(items, list) or not items:
        raise ValueError("Queue must be a non-empty JSON array.")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"id", "case_sha256", "minutes"}:
            raise ValueError("Queue entries require id, case_sha256 and minutes.")
        key, minutes = item["id"], item["minutes"]
        if not isinstance(key, str) or key in seen or key not in cases:
            raise ValueError("Unknown or repeated queue case.")
        if item["case_sha256"] != cases[key]["case_sha256"]:
            raise ValueError(f"Queue case fingerprint changed: {key}")
        if (not isinstance(minutes, list) or len(minutes) != 2
                or any(type(n) is not int or n <= 0 for n in minutes)
                or minutes[1] < minutes[0]):
            raise ValueError(f"Invalid review estimate: {key}")
        seen.add(key)
    return items


def display_text(text):
    # Keep archived control codes from acting as terminal commands.
    return "".join(c if c.isprintable() or c in "\t\n" else ascii(c)[1:-1] for c in text)


class ArticlePreview(HTMLParser):
    """Produce a readable preview of archived HTML publisher evidence."""

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.depth += 1
        if self.depth and tag in {"p", "li", "h1", "h2", "h3", "h4", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "article" and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def notice_preview(raw):
    if raw.startswith(b"%PDF") or b"\0" in raw:
        return "Binary evidence. Inspect its verified local object."
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("<"):
        parser = ArticlePreview()
        parser.feed(text)
        preview = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
        if preview:
            return "Archived HTML article preview (see the exact object for the full notice)\n" + display_text(preview)
    return display_text(text)


def pages(text):
    lines = display_text(text).split("\n")
    return ["\n".join(lines[i:i + PAGE_LINES]) for i in range(0, len(lines), PAGE_LINES)]


def shared_notices(queue, cases):
    return Counter(
        sha
        for entry in queue
        for sha in {ref["sha256"] for ref in cases[entry["id"]]["record"]["evidence"]
                    if ref["role"] == "rights_notice"}
    )


def effort_directory(store, path):
    path = path.resolve()
    store = store.resolve()
    if path.is_relative_to(store) or store.is_relative_to(path):
        raise ValueError("Use a dedicated private effort directory separate from the protected store.")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir() or path.stat().st_mode & 0o077:
        raise ValueError("Effort directory must be private (mode 0700).")
    return path


def log_effort(path, envelope, seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Enter a positive elapsed duration in seconds.")
    row = envelope["record"]
    write_json(path / f"{uuid4().hex}.json", {
        "schema": "goxlm-review-effort-v1",
        "id": row["id"],
        "case_sha256": envelope["case_sha256"],
        "extract_sha256": row["extract"]["sha256"],
        "elapsed_seconds": seconds,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
    })


def session(store, queue, cases, decisions, effort, read=input, emit=print):
    total = len(queue)
    notices = shared_notices(queue, cases)
    seen_notices = set()
    index = 0
    while 0 <= index < total:
        entry = queue[index]
        key = entry["id"]
        envelope = cases[key]
        row = envelope["record"]
        source, extract = row["source"], row["extract"]
        evidence = row["evidence"]
        siblings, duplicates = [], []
        for other, candidate in cases.items():
            if other == key:
                continue
            related = candidate["record"]
            if related["family_id"] == row["family_id"]:
                siblings.append(other)
            if related["extract"]["sha256"] == extract["sha256"]:
                duplicates.append(other)
        pending_state = decisions.get(key, {}).get("decision", "pending")
        duration = "–".join(map(str, entry["minutes"]))
        emit(display_text(
            f"\n[{index + 1}/{total}] {key} | {pending_state} | {row['title']}\n"
            f"Case SHA-256 {envelope['case_sha256']}\nExtract SHA-256 {extract['sha256']}\n"
            f"Source SHA-256 {source['sha256']} | {source['name']} | {source.get('url', '')}\n"
            f"{extract['bytes']:,} bytes | {token_proxy(extract['bytes'])} estimated subwords "
            f"(3–5 bytes per token) | {duration} estimated review minutes\n"
            f"Family {row['family_id']} | Exact duplicate IDs {', '.join(duplicates) or 'none'}\n"
            f"Same-family IDs {', '.join(siblings) or 'none'}\n"
            f"Attribution {row['attribution']}\nRights basis {row['rights_summary']}\n"
            f"Conditions {row['license_conditions']}\nExtraction {row['extraction_method']}\n"
            f"Item findings {json.dumps(row['findings'], ensure_ascii=False)}"
        ))
        for number, ref in enumerate(evidence, 1):
            sha = ref["sha256"]
            emit(display_text(f"  {number}. {ref['role']} | {ref['name']} | {sha}\n"
                              f"     {ref['reference']} | {ref.get('url', '')}"))
            if notices[sha] > 1 and sha not in seen_notices:
                seen_notices.add(sha)
                emit(f"\nShared publisher evidence ({notices[sha]} queue cases), SHA-256 {sha}")
                emit(str((store / "objects" / sha).resolve()))
                emit(notice_preview(object_bytes(store, sha)))
        emit(f"Local archived original {(store / 'objects' / source['sha256']).resolve()}")
        extract_pages = pages(object_bytes(store, extract["sha256"]).decode("utf-8"))
        part = 0
        emit(f"\nExact extract, page 1/{len(extract_pages)}\n{extract_pages[0]}")
        while True:
            try:
                command = read("n/p case | j N jump | >/< text page | e N evidence | t seconds | q quit: ").split()
            except EOFError:
                return
            if not command:
                continue
            action, *args = command
            try:
                if action == "q" and not args:
                    return
                if action in {"n", "p"} and not args:
                    index = max(0, index - 1) if action == "p" else index + 1
                    break
                if action == "j" and len(args) == 1 and 1 <= int(args[0]) <= total:
                    index = int(args[0]) - 1
                    break
                if action in {">", "<"} and not args:
                    part = max(0, min(part + (1 if action == ">" else -1), len(extract_pages) - 1))
                    emit(f"\nExact extract, page {part + 1}/{len(extract_pages)}\n{extract_pages[part]}")
                elif action == "e" and len(args) == 1 and 1 <= int(args[0]) <= len(evidence):
                    ref = evidence[int(args[0]) - 1]
                    raw = object_bytes(store, ref["sha256"])
                    path = (store / "objects" / ref["sha256"]).resolve()
                    emit(f"Verified local evidence {path}")
                    if not raw.startswith(b"%PDF") and b"\0" not in raw:
                        emit(display_text(raw.decode("utf-8")))
                elif action == "t" and len(args) == 1 and effort is not None:
                    log_effort(effort, envelope, float(args[0]))
                    emit("Elapsed effort logged separately. Make any rights decision in the protected workflow.")
                else:
                    emit("Unknown command or unavailable effort logging.")
            except (ValueError, OSError) as exc:
                emit(f"Unable to complete command: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("queue", type=Path, help="Local JSON list of pinned case IDs and review estimates")
    parser.add_argument("--effort-dir", type=Path, help="Optional private directory for explicit elapsed-time entries")
    args = parser.parse_args()
    try:
        cases = all_cases(args.store)
        _, _, decisions = load_history(args.store, cases)
        queue = verified_queue(json.loads(args.queue.read_text(encoding="utf-8")), cases)
        if not sys.stdin.isatty():
            raise ValueError("The case browser requires an interactive terminal.")
        effort = effort_directory(args.store, args.effort_dir) if args.effort_dir else None
        session(args.store, queue, cases, decisions, effort)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Review browser: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nReview interrupted.\n")


if __name__ == "__main__":
    main()
