"""Synthetic-only tests for private conversational corpora."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from goxlm_review.transcript_intake import (
    FILES, REVIEW_FINDINGS, encrypted_private_root, intake, private_file, screen,
    trusted_keyring, verify_reviewer_signature,
)
from goxlm_v02.common import canonical, digest

HELDOUT = ["The empty paper carton stayed on the bench while the neighbor washed several garden tools nearby."]
TOPICS = (
    ("I was planning to bring a warm lunch for everyone after the morning walk.",
     "We could fill a large food container with rice and share it using bowls."),
    ("The flowers seem to dry out fastest when they stand in the afternoon sun.",
     "Move the pots nearer the shaded wall and check their soil before watering."),
    ("The old soccer ball rolled behind the fence while we were playing outside.",
     "We can use the spare ball until the other field opens later this evening."),
    ("Our books have been scattered across the apartment since the shelf arrived.",
     "Sort them into stacks on a table while we finish clearing space upstairs."),
    ("The mountain path turns west after the picnic shelter beside the creek.",
     "Bring a map and leave a note about when we expect to be back."),
)


def ident(prefix, n):
    return f"{prefix}_{n:012x}"


def turn(speaker, start, text):
    return {"speaker": speaker, "start_ms": start, "end_ms": start + 500, "text": text}


def row(index, recording, speakers, topic=0, adulthood="all_adult_verified"):
    a, b = TOPICS[topic]
    return {"id": ident("c", index), "recording_id": ident("r", recording),
            "speakers": [ident("s", i) for i in speakers], "participant_status": adulthood,
            "language": "en", "segment_start_ms": 0, "segment_end_ms": 2000,
            "turns": [turn(0, 0, a), turn(1, 500, b)]}


class TranscriptIntakeTests(unittest.TestCase):
    def test_transitive_connection_and_split_stability(self):
        rows = [row(1, 1, (1, 2)), row(2, 2, (2, 3), 1), row(3, 3, (3, 4), 2),
                row(4, 4, (5, 6), 3), row(5, 5, (7, 8), 4)]
        result = screen(rows, HELDOUT)
        self.assertEqual(result, screen(reversed(rows), HELDOUT))
        self.assertEqual(3, result["speaker_connected_components"])
        self.assertEqual(3, len(result["split_preview_counts"]))
        records = {r["record_id"]: r for r in result["records"]}
        self.assertEqual(1, len({records[ident("c", i)]["split_preview"] for i in (1, 2, 3)}))
        self.assertEqual(1, len({records[ident("c", i)]["component_sha256"] for i in (1, 2, 3)}))
        self.assertFalse(result["training_authorized"])
        self.assertNotIn(TOPICS[0][0], json.dumps(result))
        self.assertNotIn(ident("s", 1), json.dumps(result))

    def test_age_scope_applies_to_recording_and_unknown_fails_closed(self):
        rows = [row(1, 3, (1, 2)), row(2, 3, (1, 2), 1, "contains_minor"),
                row(3, 4, (3, 4), 2, "unknown")]
        result = screen(rows, HELDOUT)
        self.assertEqual({"minor_or_unverified_age_recording": 3}, result["excluded_counts"])
        self.assertFalse(result["split_preview_available"])

    def test_privacy_quality_duplicate_and_eval_contamination(self):
        rows = [row(1, 1, (1, 2)), row(2, 2, (3, 4)), row(3, 3, (5, 6), 2),
                row(4, 4, (7, 8), 3), row(5, 5, (9, 10), 4), row(6, 6, (11, 12), 1)]
        rows[2]["turns"][0]["text"] = "Call me at the usual time. My full name is not needed for this exercise."
        rows[3]["turns"][0]["text"] = HELDOUT[0]
        rows[4]["turns"][0]["text"] = "um " * 100
        rows[5]["turns"][0]["text"] = "I heard a song and want to give an excerpt from the lyrics."
        result = screen(rows, HELDOUT)
        self.assertEqual(1, result["screened_records"])
        self.assertEqual({"exact_near_or_passage_duplicate": 1, "possible_private_information": 1,
                          "heldout_overlap": 1, "filler_or_repetition": 1, "possible_incorporated_text": 1},
                         result["excluded_counts"])

    def test_opaque_ids_identical_speaker_reject_and_allow_cross_speaker_overlap(self):
        item = row(1, 1, (1, 2))
        item["turns"][1]["start_ms"] = 250
        self.assertEqual(1, screen([item], HELDOUT)["screened_records"])
        item["speakers"][0] = item["speakers"][1]
        with self.assertRaisesRegex(ValueError, "distinct"):
            screen([item], HELDOUT)
        item["speakers"][0] = ident("s", 1)
        item["turns"][1]["start_ms"] = 2000
        with self.assertRaisesRegex(ValueError, "timing"):
            screen([item], HELDOUT)

    def test_near_duplicates_and_recording_segments(self):
        x, y = row(1, 4, (1, 2)), row(2, 4, (1, 2), 1)
        y["segment_start_ms"], y["segment_end_ms"] = 2000, 4000
        for t in y["turns"]:
            t["start_ms"] += 2000
            t["end_ms"] += 2000
        near = row(3, 7, (3, 4))
        near["turns"][0]["text"] = near["turns"][0]["text"].replace("planning", "hoping")
        report = screen([x, y, near], HELDOUT)
        self.assertTrue(report["excluded_counts"]["exact_near_or_passage_duplicate"])
        self.assertEqual(1, report["speaker_connected_components"])

    def test_bound_rights_scope_private_files_and_no_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = {"agreement": b"synthetic scope only", "provider_approval": b"fake receipt",
                       "source_manifest": b"synthetic source", "age_evidence": b"synthetic adults",
                       "delivery": canonical(row(1, 1, (1, 2))), "heldout": canonical(HELDOUT)}
            def put(name, raw, mode=0o600):
                p = root / name
                p.write_bytes(raw)
                p.chmod(mode)
                return p
            for key, (name, _) in FILES.items():
                put(name, payload[key])
            today = datetime.now(timezone.utc).date().isoformat()
            statement = {"provider": "Open Yap 1K", "recipient": "Synthetic fixture",
                               "scope": "approved_transcript_only_commercial_training", "status": "active",
                               "delivery_id": ident("d", 1),
                               "review": {"verified_on": today, "valid_through": today,
                                          "findings": dict.fromkeys(REVIEW_FINDINGS, "verified")},
                               "files": {k: digest(v) for k, v in payload.items()}}
            scope = canonical(statement)
            put("scope.json", scope)
            with (patch("goxlm_review.transcript_intake.encrypted_private_root", return_value=root),
                  patch("goxlm_review.transcript_intake.verify_reviewer_signature") as verify):
                with self.assertRaises(FileNotFoundError):
                    intake(root, digest(scope), "Synthetic fixture")
                put("scope.json.sig", b"synthetic fixture signature")
                verify.side_effect = ValueError("untrusted signature")
                with self.assertRaisesRegex(ValueError, "untrusted signature"):
                    intake(root, digest(scope), "Synthetic fixture")
                verify.side_effect = None
                statement["review"]["findings"]["participant_age_evidence"] = "unknown"
                unverified = canonical(statement)
                put("scope.json", unverified)
                with self.assertRaisesRegex(ValueError, "every required"):
                    intake(root, digest(unverified), "Synthetic fixture")
                statement["review"]["findings"]["participant_age_evidence"] = "verified"
                statement["review"]["valid_through"] = "2000-01-01"
                expired = canonical(statement)
                put("scope.json", expired)
                with self.assertRaisesRegex(ValueError, "validity period"):
                    intake(root, digest(expired), "Synthetic fixture")
                put("scope.json", scope)
                with self.assertRaisesRegex(ValueError, "anchor"):
                    intake(root, "0" * 64, "Synthetic fixture")
                with self.assertRaisesRegex(ValueError, "scope mismatch"):
                    intake(root, digest(scope), "Other")
                put("provider-approval.txt", b"tampered")
                with self.assertRaisesRegex(ValueError, "bound input"):
                    intake(root, digest(scope), "Synthetic fixture")
                put("provider-approval.txt", payload["provider_approval"])
                summary = intake(root, digest(scope), "Synthetic fixture")
                self.assertNotIn("records", summary)
                self.assertFalse(summary["training_authorized"])
                with self.assertRaises(FileExistsError):
                    intake(root, digest(scope), "Synthetic fixture")
            self.assertEqual(0o600, (root / "screened-preview.json").stat().st_mode & 0o777)
            put("unsafe", b"synthetic", 0o644)
            with self.assertRaisesRegex(ValueError, "0600"):
                private_file(root, "unsafe", 1024)
            os.symlink("agreement.txt", root / "link")
            with self.assertRaises(OSError):
                private_file(root, "link", 1024)

    def test_unencrypted_storage_stops_before_reading_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
                with self.assertRaisesRegex(ValueError, "dm-crypt"):
                    intake(Path(td), "0" * 64, "Synthetic fixture")

    def test_storage_rejects_active_swap_and_non_luks_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            luks_uuid = "CRYPT-LUKS2-" + "a" * 32 + "-goxlm"
            header = "Filename\tType\tSize\tUsed\tPriority\n"

            def read(path, **kwargs):
                return header if path == Path("/proc/swaps") else luks_uuid

            with patch("pathlib.Path.read_text", autospec=True, side_effect=read):
                self.assertEqual(root, encrypted_private_root(root))
            with patch("pathlib.Path.read_text", side_effect=[luks_uuid, header + "/dev/sdc partition 8 0 -2\n"]):
                with self.assertRaisesRegex(ValueError, "swap"):
                    encrypted_private_root(root)
            with patch("pathlib.Path.read_text", return_value="CRYPT-PLAIN-name"):
                with self.assertRaisesRegex(ValueError, "LUKS2"):
                    encrypted_private_root(root)
            root.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "0700"):
                encrypted_private_root(root)

    @unittest.skipUnless(shutil.which("gpg") and Path("/usr/bin/gpgv").is_file(), "GnuPG tools unavailable")
    def test_trusted_signature_verifies_exact_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            keyring, data, sig = (home / name for name in ("reviewers.gpg", "receipt", "receipt.sig"))
            gpg = ["gpg", "--homedir", td, "--batch", "--pinentry-mode", "loopback", "--passphrase", ""]
            def run(*args):
                return subprocess.run([*gpg, *args], capture_output=True, check=True, timeout=30).stdout

            run("--quick-generate-key", "Synthetic Reviewer <fixture@example.invalid>", "ed25519", "sign", "1d")
            keyring.write_bytes(run("--export"))
            data.write_bytes(b"exact synthetic receipt")
            run("--output", str(sig), "--detach-sign", str(data))
            with patch("goxlm_review.transcript_intake.trusted_keyring", return_value=keyring):
                verify_reviewer_signature(data.read_bytes(), sig.read_bytes())
                with self.assertRaisesRegex(ValueError, "trusted reviewer signature"):
                    verify_reviewer_signature(b"tampered receipt with matching replacement hash", sig.read_bytes())
                with self.assertRaisesRegex(ValueError, "trusted reviewer signature"):
                    verify_reviewer_signature(data.read_bytes(), b"synthetic fake signature")
            with patch("goxlm_review.transcript_intake.TRUSTED_KEYRING", keyring):
                home.chmod(0o777)
                with self.assertRaisesRegex(ValueError, "administered by root"):
                    trusted_keyring()


if __name__ == "__main__":
    unittest.main()
