"""Regression tests for the 0.3 corpus preparation safeguards."""
import unittest
from unittest.mock import patch

import torch

from .audit_tool import checkpoint_impact, inventory, overlap_report
from .common import canonical, digest
from .data import build, freeze_path, load_release, permanent_gate, read, revoke
from .run import CHECKPOINT

from . import test_v02


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        fixture = test_v02.PipelineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.catalog = fixture.catalog
        self.demo_text = "A synthetic demonstration paragraph about coupled pendulums."
        path = fixture.root / "data/approved/science_technology_v01.txt"
        path.parent.mkdir(parents=True)
        path.write_text(self.demo_text + "\n", encoding="utf-8")
        self.enterContext(patch("goxlm_v02.overlap.ROOT", fixture.root))

    def test_revocation_releases_and_checkpoint_lineage(self):
        release = self.fixture.release()
        old = load_release(self.catalog, release)
        target = next(row for row in old["documents"] if row["split"] == "train")
        text = target["text"]
        key, sha = target["document_id"], target["submission_sha256"]
        self.fixture.submit("mirror", "  " + text.upper() + "\n")
        newer = build(self.catalog)
        decision = {"submission_sha256": sha, "reviewer": "Synthetic reviewer",
                    "revoked_on": "2026-11-18", "reason": "Synthetic removal",
                    "notice_reference": "Synthetic notice"}
        with self.assertRaisesRegex(ValueError, "submission hash"):
            revoke(self.catalog, key, {**decision, "submission_sha256": "wrong"})
        revoke(self.catalog, key, decision)
        with self.assertRaises(FileExistsError):
            revoke(self.catalog, key, decision)
        for blocked in (release, newer):
            with self.assertRaisesRegex(ValueError, "revoked documents"):
                load_release(self.catalog, blocked)
        snapshot = inventory(self.catalog)
        self.assertEqual([], snapshot["errors"])
        self.assertEqual({key, "mirror"}, set(snapshot["releases"][newer.stem]["affected"]))
        path = self.fixture.root / "synthetic.pt"
        torch.save({"schema": CHECKPOINT, "dataset_version": newer.stem}, path)
        self.assertEqual("affected", checkpoint_impact(snapshot, [path])[0]["status"])
        self.assertNotIn(key, {row["document_id"] for row in
                               load_release(self.catalog, build(self.catalog))["documents"]})

    def test_catalog_integrity_detects_orphan_and_release_tamper(self):
        release = self.fixture.release()
        record = read(release)
        record["seed"] += 1
        release.write_bytes(canonical(record))
        orphan = self.catalog / "reviews/orphan.json"
        orphan.write_text("{}")
        errors = inventory(self.catalog)["errors"]
        self.assertTrue(any("Orphan review" in error for error in errors))
        self.assertTrue(any("Invalid release version or digest" in error for error in errors))
        self.assertTrue(any("No valid release" in error for error in errors))

    def test_permanent_freeze_rejects_demo_and_tiny_preview(self):
        release = self.fixture.release()
        self.assertEqual([], overlap_report(self.catalog, release)["legacy_demo_overlap"])
        with self.assertRaisesRegex(ValueError, "40 independent groups"):
            build(self.catalog, permanent=True)
        self.assertFalse(freeze_path(self.catalog, True).exists())
        self.fixture.submit("repackaged-demo", f"A new introduction. {self.demo_text}")
        with self.assertRaisesRegex(ValueError, "demonstration overlap"):
            build(self.catalog, permanent=True)
        self.assertFalse(freeze_path(self.catalog, True).exists())

    def test_independent_overlap_flags_shingles_and_related_families(self):
        common = "A rotating frame introduces apparent forces in the equations of motion."
        a = {"document_id": "left", "text": "Preliminary derivation. " + common,
             "family_id": "edition", "source_id": "source-a", "split": "train"}
        b = {"document_id": "right", "text": common + " See the appendix for details.",
             "family_id": "edition", "source_id": "source-b", "split": "test"}
        with self.assertRaisesRegex(ValueError, "cross-split contamination"):
            permanent_gate([a, b])
        from .overlap import findings
        self.assertEqual(["family_id", "shingle_overlap"], findings([a, b])[0]["reasons"])

    def test_permanent_gate_accepts_and_freezes_independent_corpus(self):
        for i in range(90):
            # Distinct alphabets keep the synthetic documents independent and cheap to compare.
            char = chr(0x4E00 + i)
            self.fixture.submit(f"permanent{i}", char * 2500)
        first = build(self.catalog, permanent=True)
        self.assertEqual(first, build(self.catalog, permanent=True))
        self.assertTrue(load_release(self.catalog, first)["permanent"])
        target = next(row for row in read(first)["documents"] if row["split"] == "test")
        self.fixture.submit("descendant", "Distinct added text.", family_id=target["family_id"])
        with self.assertRaisesRegex(ValueError, "frozen test group"):
            build(self.catalog, permanent=True)
