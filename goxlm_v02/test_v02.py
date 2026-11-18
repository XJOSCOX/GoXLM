import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .common import canonical, digest, verify_baseline
from .data import build, intake, load_release, read, review
from .model import decode, encode, generate, new_model, windows
from .run import RunConfig, evaluate, load_checkpoint, train

torch.set_num_threads(1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "catalog"

    def submit(self, key, text, approved=True, **overrides):
        metadata = {k: key for k in ("document_id", "source_id", "family_id", "title")}
        metadata.update(domain="science", rights_holder="Synthetic test fixture",
                        source_reference="Local test fixture", rights_basis="Test only",
                        evidence_reference="Paired synthetic evidence")
        metadata.update(overrides)
        document, evidence = self.root / "input.txt", self.root / "evidence.txt"
        document.write_text(text, encoding="utf-8")
        evidence.write_text("Synthetic fixture. This is not a real rights grant.")
        intake(self.catalog, metadata, document, evidence)
        submission = read(self.catalog / "intake" / (key + ".json"))
        decision = {"submission_sha256": digest(canonical(submission)),
                    "decision": "approved", "reviewer": "Synthetic test reviewer",
                    "review_date": "2026-11-18", "rationale": "Exercise the approval pipeline",
                    "permissions_scope": "commercial_ai_training"}
        if approved:
            review(self.catalog, key, decision)
        return decision

    def release(self):
        for i in range(24):
            content = digest(f"original fixture {i}".encode())
            self.submit(f"doc{i}", content)
        return build(self.catalog)

    def test_rights_gate_and_tampering(self):
        path = self.release()
        decision = self.submit("pending", "Pending test content.", approved=False)
        self.assertEqual(path, build(self.catalog))
        for bad in ({"submission_sha256": "wrong"}, {"permissions_scope": "research"},
                    {"reviewer": ""}):
            with self.assertRaises(ValueError):
                review(self.catalog, "pending", {**decision, **bad})
        review(self.catalog, "pending", {**decision, "decision": "rejected"})
        self.assertEqual(path, build(self.catalog))
        source = self.catalog / "intake" / "doc0.json"
        record = read(source)
        record["evidence"] = "dGFtcGVyZWQ="
        source.write_bytes(canonical(record))
        with self.assertRaises(ValueError):
            load_release(self.catalog, path)

    def test_freeze_and_related_documents(self):
        content = "The period of a pendulum depends on its length and local gravity."
        self.release_groups_with_duplicates(content)
        path = build(self.catalog)
        self.assertEqual(path, build(self.catalog))
        release = load_release(self.catalog, path)
        rows = {row["document_id"]: row for row in release["documents"]}
        self.assertEqual(1, len({rows[key]["split"] for key in ("a", "b", "c", "d")}))
        frozen = next(row for row in rows.values() if row["split"] == "test")
        self.submit("new_exact", frozen["text"])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            build(self.catalog)

    def release_groups_with_duplicates(self, content):
        for i in range(24):
            self.submit(f"doc{i}", digest(str(i).encode()))
        self.submit("a", content)
        self.submit("b", content)
        self.submit("c", content.replace("local", "nearby"))
        self.submit("d", "An unrelated string in the same document family.", family_id="c")

    def test_context_tokenizer_causality_and_generation(self):
        text = "π = 3.14\n    print('λ')\n" * 10
        self.assertEqual(text, decode(encode(text)))
        self.assertEqual(encode(text), [t for _, y in windows(text) for t in y if t != -100])
        model = new_model().eval()
        self.assertEqual(3_257_856, sum(p.numel() for p in model.parameters()))
        length = model.config.context_length
        left = torch.zeros((1, length), dtype=torch.long)
        right = left.clone()
        right[:, length // 2:] = 1
        with torch.inference_mode():
            first, _ = model(left)
            other, _ = model(right)
            torch.testing.assert_close(first[:, :length // 2], other[:, :length // 2])
            self.assertFalse(torch.equal(first[:, -1], other[:, -1]))
            with self.assertRaisesRegex(ValueError, "context length"):
                model(torch.zeros((1, length + 1), dtype=torch.long))
        self.assertEqual(generate(model, text, steps=3), generate(model, text, steps=3))
        self.assertIsInstance(generate(model, "", steps=1), str)
        for bad in ({"top_k": 257}, {"temperature": float("nan")}, {"steps": -1}):
            with self.assertRaises(ValueError):
                generate(model, text, **({"steps": 1} | bad))

    def test_reproducible_resume_and_evaluation(self):
        baseline = self.root / "synthetic-baseline.txt"
        baseline.write_bytes(b"Synthetic baseline fixture.\n")
        expected = {baseline.name: digest(baseline.read_bytes())}
        self.enterContext(patch("goxlm_v02.common.ROOT", self.root))
        self.enterContext(patch("goxlm_v02.common.BASELINE", expected))
        self.assertEqual(expected, verify_baseline())
        release = self.release()
        config = RunConfig(batch_size=2)
        device = os.environ.get("GOXLM_TEST_DEVICE", "cpu")
        one = train(self.catalog, release, self.root / "segment1", 1, device, config)
        two = train(self.catalog, release, self.root / "segment2", 2, device, config, resume=one)
        direct = train(self.catalog, release, self.root / "direct", 2, device, config)
        model, resumed = load_checkpoint(two, device)
        _, uninterrupted = load_checkpoint(direct, device)
        for name, value in resumed["model"].items():
            torch.testing.assert_close(value, uninterrupted["model"][name], rtol=0, atol=0)
        self.assertEqual(resumed["history"][-1]["validation"],
                         uninterrupted["history"][-1]["validation"])
        self.assertEqual(str(device), resumed["environment"]["device"])
        self.assertGreater(resumed["hardware"]["process_peak_rss_bytes"], 0)
        if device == "cuda":
            self.assertGreater(resumed["hardware"]["gpu_peak_allocated_bytes"], 0)
            self.assertIn("3090", resumed["environment"]["gpu"])
        with self.assertRaises(FileExistsError):
            train(self.catalog, release, self.root / "segment1", 1, device, config)
        with self.assertRaisesRegex(ValueError, "matching data"):
            train(self.catalog, release, self.root / "mismatch", 2, device,
                  RunConfig(batch_size=3), resume=one)
        rows = [row for row in load_release(self.catalog, release)["documents"]
                if row["split"] == "test"]
        result = evaluate(model, rows, torch.device(device))
        self.assertEqual(sum(len(row["text"].encode()) for row in rows), result["overall"]["bytes"])
        corrupt = self.root / "bad.pt"
        torch.save({**resumed, "tokenizer": {"version": "other"}}, corrupt)
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
            load_checkpoint(corrupt)
