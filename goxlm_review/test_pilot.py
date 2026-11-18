"""Synthetic regression tests for staging and reviewer decisions."""
import io
import os
import pwd
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from goxlm_v02.common import canonical, digest
from goxlm_v02.data import read

from .__main__ import main
from .ledger import GENESIS, record, report, status
from .staging import batch, case, init


class PilotTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.store = self.path / "pilot"
        self.login = pwd.getpwuid(os.getuid()).pw_name
        init(self.store, [self.login])

    def item(self, key, text, family=None):
        def entry(name, raw):
            path = self.path / name
            path.write_bytes(raw)
            return {"file": name, "sha256": digest(raw)}

        rights = {**entry("rights.txt", b"Synthetic evidence; no actual license."),
                  "role": "rights_notice", "reference": "Fixture"}
        authority = {**entry("authority.txt", b"Synthetic depositor record."),
                     "role": "depositor_authority", "reference": "Fixture"}
        return {"id": key, "domain": "science", "title": key, "family_id": family or key,
                "template": "cc0", "rights_summary": "Fixture", "attribution": "Fixture submitter",
                "license_conditions": "Synthetic only", "source": entry(key + ".source", b"raw copy"),
                "extract": entry(key + ".extract", text), "evidence": [rights, authority],
                "extraction_method": "Fixture", "extraction_cost": {"cpu_ms": 7},
                "exceptions": ["Check external credits"]}

    def decision(self, key, action="approve", **changes):
        sha = case(self.store, key)["record"]["extract"]["sha256"]
        return {"decision": action, "category": "rights", "reviewer": "Fixture Reviewer",
                "decision_date": "2026-11-18", "rationale": "Exercise decision flow",
                "extract_sha256": sha, "commercial_ai_training": action == "approve",
                "findings_acknowledged": True, "jurisdiction": "US", **changes}

    def test_reproducible_batch_costs_and_overlap(self):
        a = self.item("left", b"The rotating frame introduces an apparent force in the equations of motion.")
        b = self.item("right", b"The rotating frame introduces an apparent force in the equations of motion.")
        c = self.item("near", b"Preliminary notes: the rotating frame introduces an apparent force in the equations of motion.")
        result = batch(self.store, [a, b, c], self.path)
        repeated = batch(self.store, [b, c, a], self.path)
        self.assertEqual(result["batch_sha256"], repeated["batch_sha256"])
        second = read(Path(repeated["run_report"]))
        self.assertEqual(0, sum(c["new_blob_bytes_written"] for c in second["items"].values()))
        self.assertGreater(second["cpu_ns"], 0)
        view = report(self.store)
        self.assertEqual(GENESIS, view["head"])
        self.assertEqual(len((self.path / a["extract"]["file"]).read_bytes()),
                         view["documents"][0]["proposed_byte_tokens"])
        exact = next(p for p in view["overlap"] if p["left"] == "left" and p["right"] == "right")
        self.assertEqual({"normalized_exact", "shingle_overlap"}, set(exact["reasons"]))
        self.assertTrue(any("shingle_overlap" in p["reasons"] for p in view["overlap"]))
        self.assertEqual(7, view["documents"][0]["extraction_cost"]["cpu_ms"])

    def test_built_in_text_extraction_is_deterministic(self):
        item = self.item("plain", b"unused fixture")
        src = b"\xef\xbb\xbfOne technical paragraph.\r\n    preserve indentation\r\n"
        (self.path / item["source"]["file"]).write_bytes(src)
        item["source"]["sha256"] = digest(src)
        item["extract"] = {"mode": "utf8_lf_v1"}
        item["extraction_method"] = "utf8_lf_v1"
        staged = batch(self.store, [item], self.path)
        item_report = read(Path(staged["run_report"]))["items"]["plain"]
        self.assertGreater(item_report["extract_cpu_ns"], 0)
        expected = b"One technical paragraph.\n    preserve indentation\n"
        self.assertEqual(digest(expected), case(self.store, "plain")["record"]["extract"]["sha256"])
        self.assertEqual(staged["batch_sha256"], batch(self.store, [item], self.path)["batch_sha256"])

    def test_hash_mismatch_does_not_stage_a_batch(self):
        good = self.item("good", b"A technical document.")
        bad = self.item("bad", b"Second technical document.")
        bad["evidence"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            batch(self.store, [good, bad], self.path)
        self.assertEqual({}, status(self.store)[0])
        self.assertFalse((self.store / "objects").exists())

    def test_case_conflict_and_capture_tampering(self):
        item = self.item("doc", b"A document originally entered.")
        batch(self.store, [item], self.path)
        changed = self.item("doc", b"Changed text.")
        with self.assertRaisesRegex(ValueError, "already binds different"):
            batch(self.store, [changed], self.path)
        sha = item["evidence"][0]["sha256"]
        (self.store / "objects" / sha).write_bytes(b"Edited evidence")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            status(self.store)

    def test_approval_integrity_revoke_and_anchored_truncation(self):
        item = self.item("doc", b"Text for human review.")
        batch(self.store, [item], self.path)
        good = self.decision("doc")
        sha = good["extract_sha256"]
        for changes, needle in [({"commercial_ai_training": False}, "explicit commercial"),
                                ({"findings_acknowledged": False}, "acknowledging"),
                                ({"extract_sha256": "0" * 64}, "exact extract")]:
            with self.assertRaisesRegex(ValueError, needle):
                payload = {**good, **changes}
                record(self.store, "doc", payload, payload["extract_sha256"])
        with self.assertRaisesRegex(ValueError, "Interactive confirmation"):
            record(self.store, "doc", good, "0" * 64)
        with patch("goxlm_review.ledger.pwd.getpwuid",
                   return_value=type("User", (), {"pw_name": "unauthorized"})()):
            with self.assertRaises(PermissionError):
                record(self.store, "doc", good, sha)
        first = record(self.store, "doc", good, sha)
        withdrawal = self.decision("doc", "revoke", notice_reference="Fixture withdrawal")
        second = record(self.store, "doc", withdrawal, sha)
        self.assertEqual(first["event_sha256"], second["event"]["previous_sha256"])
        self.assertEqual(1, report(self.store)["rights_revoked"])
        self.assertEqual(second["event_sha256"], status(self.store, first["event_sha256"])[1])
        last = self.store / "decisions/00000002.json"
        last.unlink()
        with self.assertRaisesRegex(ValueError, "independently saved head"):
            status(self.store, second["event_sha256"])
        first_path = self.store / "decisions/00000001.json"
        first["event"]["rationale"] = "Tampered"
        first_path.write_bytes(canonical(first))
        with self.assertRaisesRegex(ValueError, "chain integrity"):
            status(self.store)

    def test_rights_rejections_and_technical_failures_are_distinct(self):
        a = self.item("legal", b"Legible document.")
        b = self.item("broken", b"\xff\xfe")
        batch(self.store, [a, b], self.path)
        rejection = self.decision("legal", "reject")
        hold = self.decision("broken", "hold", category="technical")
        record(self.store, "legal", rejection, rejection["extract_sha256"])
        record(self.store, "broken", hold, hold["extract_sha256"])
        view = report(self.store)
        self.assertEqual(1, view["legally_rejected"])
        self.assertEqual(1, view["technically_unusable"])
        self.assertEqual(0, view["documents"][0]["proposed_byte_tokens"])
        bad = self.decision("broken")
        with self.assertRaisesRegex(ValueError, "Technically unusable"):
            record(self.store, "broken", bad, bad["extract_sha256"])

    def test_cli_blocks_noninteractive_review(self):
        batch(self.store, [self.item("doc", b"Reviewed document.")], self.path)
        sha = case(self.store, "doc")["record"]["extract"]["sha256"]
        args = ["review", str(self.store), "decide", "doc", "approve",
                "--extract-sha256", sha, "--reviewer", "Fixture", "--date", "2026-11-18",
                "--rationale", "Fixture", "--commercial-ai-training", "--jurisdiction", "US",
                "--acknowledge-findings"]
        with (patch("sys.argv", args), patch("sys.stdin", io.StringIO(sha + "\n")),
              redirect_stderr(io.StringIO()) as errors, redirect_stdout(io.StringIO())):
            with self.assertRaises(SystemExit) as exc:
                main()
        self.assertEqual(2, exc.exception.code)
        self.assertIn("interactive WSL terminal", errors.getvalue())
        self.assertEqual([], status(self.store)[2])
