from contextlib import nullcontext
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .common import canonical, digest, git_provenance, verify_baseline
from .data import build, intake, load_release, read, review
from .model import decode, encode, generate, new_model, windows
from .run import CHECKPOINT, LEGACY_CHECKPOINT, RunConfig, evaluate, load_checkpoint, train

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

    @unittest.skipUnless(shutil.which("git"), "Git is required for this test")
    def test_git_provenance_tracks_only_visible_changes(self):
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.root), *args],
                                           stderr=subprocess.DEVNULL, text=True).strip()

        git("init", "--quiet")
        ignore = self.root / ".gitignore"
        ignore.write_text("research/\n")
        git("add", ".gitignore")
        git("-c", "commit.gpgSign=false", "-c", "user.name=Synthetic Fixture",
            "-c", "user.email=fixture@example.invalid",
            "commit", "--quiet", "-m", "Synthetic fixture")
        commit = git("rev-parse", "HEAD")
        self.assertEqual({"commit_sha": commit, "dirty": False}, git_provenance(self.root))
        self.assertEqual({"commit_sha": None, "dirty": None}, git_provenance(self.root / "absent"))
        private = self.root / "research"
        private.mkdir()
        (private / "ignored.txt").write_text("Synthetic fixture")
        self.assertEqual({"commit_sha": commit, "dirty": False}, git_provenance(self.root))
        self.assertEqual({"commit_sha": None, "dirty": None}, git_provenance(private))
        visible = self.root / "code.py"
        visible.write_text("# Synthetic fixture\n")
        self.assertEqual({"commit_sha": commit, "dirty": True}, git_provenance(self.root))
        git("add", "code.py")
        self.assertEqual({"commit_sha": commit, "dirty": True}, git_provenance(self.root))
        git("-c", "commit.gpgSign=false", "-c", "user.name=Synthetic Fixture",
            "-c", "user.email=fixture@example.invalid",
            "commit", "--quiet", "-m", "Track fixture")
        visible.write_text("# Modified fixture\n")
        self.assertEqual({"commit_sha": git("rev-parse", "HEAD"), "dirty": True},
                         git_provenance(self.root))

    def test_bf16_requires_native_cuda_before_creating_output(self):
        output = self.root / "blocked"
        for platform in ("cpu", "unavailable", "emulated"):
            with self.subTest(platform=platform), patch("goxlm_v02.run.verify_baseline") as baseline, \
                    patch("torch.cuda.is_available", return_value=platform == "emulated"), \
                    patch("torch.cuda.device", return_value=nullcontext()), \
                    patch("torch.cuda.is_bf16_supported", return_value=False) as supported:
                device = "cpu" if platform == "cpu" else "cuda"
                with self.assertRaisesRegex(ValueError, "native CUDA BF16 support"):
                    train(self.catalog, self.root / "missing", output, 1, device, RunConfig(),
                          precision="bf16")
                baseline.assert_not_called()
                if platform == "emulated":
                    supported.assert_called_once_with(including_emulation=False)
                self.assertFalse(output.exists())

    def test_reproducible_resume_and_evaluation(self):
        self.check_checkpoint_resume("fp32")

    def check_checkpoint_resume(self, precision):
        baseline = self.root / "synthetic-baseline.txt"
        baseline.write_bytes(b"Synthetic baseline fixture.\n")
        expected = {baseline.name: digest(baseline.read_bytes())}
        self.enterContext(patch("goxlm_v02.common.ROOT", self.root))
        self.enterContext(patch("goxlm_v02.common.BASELINE", expected))
        self.assertEqual(expected, verify_baseline())
        release = self.release()
        config = RunConfig(batch_size=2)
        device = os.environ.get("GOXLM_TEST_DEVICE", "cpu")
        provenance = {"commit_sha": "a" * 40, "dirty": True}
        with patch("goxlm_v02.run.git_provenance", return_value=provenance):
            one = train(self.catalog, release, self.root / "segment1", 1, device, config,
                        **({"precision": precision} if precision == "bf16" else {}))
        _, saved = load_checkpoint(one)
        self.assertEqual(provenance, saved["git"])
        self.assertEqual(CHECKPOINT, saved["schema"])
        self.assertEqual(precision, saved["precision"])
        self.assertEqual(1, saved["optimizer_updates"])
        run = read(one.parent / "run.0.2.json")
        self.assertEqual(provenance, run["git"])
        self.assertEqual(saved["code_revision"], run["code_revision"])
        if precision == "fp32":
            old_checkpoint = self.root / "legacy-fp32.pt"
            legacy = {key: value for key, value in saved.items()
                      if key not in {"git", "precision", "optimizer_updates"}}
            torch.save(dict(legacy, schema=LEGACY_CHECKPOINT), old_checkpoint)
            resume = old_checkpoint
        else:
            resume = one
            with self.assertRaisesRegex(ValueError, "Cannot change training precision"):
                train(self.catalog, release, self.root / "wrong-precision", 2, device, config,
                      resume=one, precision="fp32")
            self.assertFalse((self.root / "wrong-precision").exists())
        two = train(self.catalog, release, self.root / "segment2", 2, device, config,
                    resume=resume, precision=precision)
        direct = train(self.catalog, release, self.root / "direct", 2, device, config,
                       precision=precision)
        model, resumed = load_checkpoint(two, device)
        _, uninterrupted = load_checkpoint(direct, device)
        for name, value in resumed["model"].items():
            torch.testing.assert_close(value, uninterrupted["model"][name], rtol=0, atol=0)
        self.assertEqual(resumed["history"][-1]["validation"],
                         uninterrupted["history"][-1]["validation"])
        self.assertEqual(2, resumed["optimizer_updates"])
        self.assertEqual(precision, resumed["precision"])
        for key, state in resumed["optimizer"]["state"].items():
            for field, tensor in state.items():
                self.assertEqual(torch.float32, tensor.dtype)
                torch.testing.assert_close(tensor, uninterrupted["optimizer"]["state"][key][field],
                                           rtol=0, atol=0)
        metrics = read(two.parent / "metrics.0.2.json")
        self.assertEqual(1, metrics["optimizer_updates"])
        self.assertEqual(2, metrics["first_step"])
        for field in ("training_seconds", "validation_seconds", "checkpoint_seconds"):
            self.assertGreater(metrics[field], 0)
        self.assertEqual(str(device), resumed["environment"]["device"])
        self.assertGreater(resumed["hardware"]["process_peak_rss_bytes"], 0)
        if device == "cuda":
            self.assertGreater(resumed["hardware"]["gpu_peak_allocated_bytes"], 0)
            self.assertIn("3090", resumed["environment"]["gpu"])
        with self.assertRaises(FileExistsError):
            train(self.catalog, release, self.root / "segment1", 1, device, config,
                  precision=precision)
        with self.assertRaisesRegex(ValueError, "matching data"):
            train(self.catalog, release, self.root / "mismatch", 2, device,
                  RunConfig(batch_size=3), resume=one, precision=precision)
        rows = [row for row in load_release(self.catalog, release)["documents"]
                if row["split"] == "test"]
        result = evaluate(model, rows, torch.device(device))
        self.assertEqual(sum(len(row["text"].encode()) for row in rows), result["overall"]["bytes"])
        corrupt = self.root / "bad.pt"
        torch.save({**resumed, "tokenizer": {"version": "other"}}, corrupt)
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
            load_checkpoint(corrupt)
        wrong_count = self.root / "wrong-count.pt"
        torch.save(dict(resumed, optimizer_updates=0), wrong_count)
        with self.assertRaisesRegex(ValueError, "step and optimizer update count"):
            load_checkpoint(wrong_count)
