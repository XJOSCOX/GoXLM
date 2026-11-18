"""Regressions for explicit, non-overwriting legacy training."""
import contextlib
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from model.transformer import GoXLM, GoXLMConfig
from training import train


class LegacyTrainingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.env = dict(os.environ, PYTHONPATH=str(Path(train.__file__).resolve().parents[1]))

    def invoke(self, *args):
        return subprocess.run([sys.executable, "-B", *args], cwd=self.root, env=self.env,
                              capture_output=True, text=True, timeout=15)

    def test_import_has_no_training_or_file_side_effects(self):
        data = self.root / "data/approved/science_technology_v01.txt"
        checkpoint = self.root / "checkpoints/goxlm_v01.pt"
        for path in (data, checkpoint):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"synthetic sentinel")

        def snapshot():
            return {path.relative_to(self.root): None if path.is_dir() else path.read_bytes()
                    for path in self.root.rglob("*")}

        before = snapshot()
        # Any top-level Torch import or execution would fail, even with valid local paths.
        result = self.invoke("-c", "import sys; sys.modules['torch'] = None; import training.train")
        self.assertEqual((0, "", ""), (result.returncode, result.stdout, result.stderr))
        self.assertEqual(before, snapshot())

    def test_requires_explicit_output(self):
        result = self.invoke("-m", "training.train")
        self.assertEqual(2, result.returncode)
        self.assertIn("--output", result.stderr)
        self.assertEqual([], list(self.root.iterdir()))

    def test_refuses_existing_checkpoint_and_dangling_symlink(self):
        existing = self.root / "existing.pt"
        existing.write_bytes(b"keep checkpoint")
        dangling = self.root / "dangling.pt"
        target = self.root / "missing.pt"
        dangling.symlink_to(target)
        for path in (existing, dangling):
            with self.subTest(path=path):
                # The rejection must occur before the absent dataset is read.
                result = self.invoke("-m", "training.train", "--output", str(path))
                self.assertEqual(1, result.returncode)
                self.assertIn("Checkpoint path already exists", result.stderr)
        self.assertEqual(b"keep checkpoint", existing.read_bytes())
        self.assertEqual(target, dangling.readlink())
        self.assertFalse(target.exists())

    def test_checkpoint_keeps_legacy_fields_and_compatible_weights(self):
        raw = b"A synthetic note on geometry.\n" * 32
        data = self.root / "synthetic.txt"
        data.write_bytes(raw)
        output = self.root / "new-run" / "legacy.pt"
        provenance = {"commit_sha": "a" * 40, "dirty": True}
        with patch.object(train, "DATA_PATH", data), patch.object(train, "MAX_STEPS", 1), \
                patch("torch.cuda.is_available", return_value=False), \
                patch.object(train, "git_provenance", return_value=provenance), \
                contextlib.redirect_stdout(io.StringIO()):
            train.train(output)

        checkpoint = torch.load(output, weights_only=True, map_location="cpu")
        self.assertEqual(1, checkpoint["step"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), checkpoint["data_sha256"])
        self.assertIn("param_groups", checkpoint["optimizer"])
        model = GoXLM(GoXLMConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        self.assertEqual(3_257_856, sum(p.numel() for p in model.parameters()))
        self.assertEqual(provenance, checkpoint["git"])
        self.assertRegex(checkpoint["code_revision"], r"^[0-9a-f]{64}$")
