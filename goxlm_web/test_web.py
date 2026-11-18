"""CPU inference fixtures and local API tests; no training or private files."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import torch

from goxlm_v02.model import MODEL, MODEL_VERSION, TOKENIZER, new_model
from goxlm_v02.run import CHECKPOINT, LEGACY_CHECKPOINT, RunConfig
from dataclasses import asdict

from .api import create_app
from .engine import CheckpointError, ModelEntry, Settings, load_local, resolve_device


def entry(**overrides):
    return ModelEntry(**({"id": "local-test", "label": "Local test", "format": "0.1",
                          "path": "test.pt", "sha256": "0" * 64} | overrides))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def test_hash_checked_before_deserialization(self):
        (self.root / "test.pt").write_bytes(b"untrusted")
        with patch("goxlm_web.engine.torch.load") as loader:
            with self.assertRaisesRegex(CheckpointError, "hash"):
                load_local(entry(), "cpu", self.root)
        loader.assert_not_called()

    def test_both_checkpoint_formats_load_without_writes(self):
        original = new_model()
        weights = original.state_dict()
        optimizer = {"param_groups": [{"params": list(range(len(list(original.parameters()))))}],
                     "state": {
                         i: {"step": torch.tensor(1.0), "exp_avg": torch.zeros_like(p),
                             "exp_avg_sq": torch.zeros_like(p)}
                         for i, p in enumerate(original.parameters())
                     }}
        legacy = {"config": MODEL, "step": 1, "data_sha256": "a" * 64, "model": weights}
        versioned = {"model_version": MODEL_VERSION, "model_config": MODEL,
                     "tokenizer": TOKENIZER, "model": weights,
                     "dataset_version": "b" * 64, "code_revision": "c" * 64,
                     "step": 1, "history": [{"step": 1}], "run_config": asdict(RunConfig()),
                     "rng": {"torch": torch.Generator().get_state(), "python": random.Random(0).getstate(),
                             "sampler": torch.Generator().get_state()},
                     "optimizer": optimizer, "environment": {}, "hardware": {}}
        for fmt, checkpoint in (
            ("0.1", legacy),
            ("0.2", {**versioned, "schema": LEGACY_CHECKPOINT}),
            ("0.2", {**versioned, "schema": CHECKPOINT, "precision": "bf16", "optimizer_updates": 1}),
        ):
            with self.subTest(schema=checkpoint.get("schema", fmt)):
                path = self.root / "test.pt"
                torch.save(checkpoint, path)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                model = load_local(entry(format=fmt, sha256=sha), "cpu", self.root)
                self.assertFalse(model.training)
                self.assertEqual(128, model.config.context_length)
                torch.testing.assert_close(model.token_embedding.weight, original.token_embedding.weight)
                self.assertEqual(sha, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_corrupt_metadata_and_malformed_checkpoints(self):
        path = self.root / "test.pt"
        torch.save({"config": {**MODEL, "context_length": 1024}}, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(CheckpointError, "metadata"):
            load_local(entry(sha256=sha), "cpu", self.root)
        path.write_bytes(b"bad checkpoint")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(CheckpointError, "could not be validated"):
            load_local(entry(sha256=sha), "cpu", self.root)

    def test_cuda_selection_and_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertTrue(resolve_device("auto").fallback)
            self.assertFalse(resolve_device("cpu").fallback)
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                resolve_device("cuda")
        with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3090"):
            self.assertEqual("cuda:0", str(resolve_device("auto").torch_device))
        with patch("torch.cuda.is_available", return_value=True), patch(
                "torch.cuda.get_device_name", side_effect=RuntimeError):
            self.assertTrue(resolve_device("auto").fallback)


class ScriptedModel:
    config = SimpleNamespace(context_length=128)

    def __init__(self, tokens=b"A", entered=None, unblock=None):
        self.tokens = tokens
        self.windows = []
        self.entered = entered
        self.unblock = unblock

    def __call__(self, inputs):
        i = len(self.windows)
        self.windows.append(inputs[0].tolist())
        if i == 0 and self.entered is not None:
            self.entered.set()
            if not self.unblock.wait(5):
                raise TimeoutError("test worker did not unblock")
        logits = torch.full((1, inputs.shape[1], 256), -100.0)
        logits[0, -1, self.tokens[i % len(self.tokens)]] = 100.0
        return logits, None


class ApiTests(unittest.TestCase):
    def client(self, model, fmt="0.1"):
        settings = Settings(device="cpu", models=[entry(format=fmt)])
        app = create_app(settings, registry_factory=lambda _settings, _device: SimpleNamespace(
            entries={"local-test": settings.models[0]}, get=lambda _entry: model))
        return TestClient(app, base_url="http://localhost")

    def payload(self, **values):
        return {"generation_id": str(uuid4()), "model_id": "local-test", "prompt": "Example ",
                "top_k": 1, "max_new_tokens": 3, **values}

    def test_models_and_streamed_utf8(self):
        model = ScriptedModel("µA".encode())
        with self.client(model) as client:
            meta = client.get("/api/models").json()["models"][0]
            self.assertEqual(3_257_856, meta["parameter_count"])
            self.assertEqual(128, meta["context_length"])
            self.assertNotIn("path", meta)
            with client.stream("POST", "/api/generate", json=self.payload()) as response:
                self.assertEqual(200, response.status_code)
                self.assertIn("application/x-ndjson", response.headers["content-type"])
                events = [json.loads(line) for line in response.iter_lines()]
            self.assertEqual(["start", "delta", "delta", "delta", "done"], [e["type"] for e in events])
            self.assertEqual("µA", "".join(e.get("text", "") for e in events))
            self.assertEqual(3, events[-1]["generated_tokens"])
            self.assertGreater(events[-1]["tokens_per_second"], 0)

    def test_prompt_budget_and_rolling_window(self):
        legacy = ScriptedModel()
        with self.client(legacy) as client:
            too_long = client.post("/api/generate", json=self.payload(prompt="é" * 65))
            self.assertEqual(422, too_long.status_code)
        versioned = ScriptedModel()
        with self.client(versioned, "0.2") as client:
            self.assertEqual(422, client.post("/api/generate", json=self.payload(prompt="a" * 128)).status_code)
            good = client.post("/api/generate", json=self.payload(prompt="a" * 127))
            self.assertEqual(200, good.status_code)
            self.assertEqual([128, 128, 128], list(map(len, versioned.windows)))
            self.assertEqual(10, versioned.windows[0][0])
            self.assertEqual(ord("a"), versioned.windows[1][0])

    def test_invalid_requests_and_host_origin(self):
        with self.client(ScriptedModel()) as client:
            for bad in ({"temperature": 0}, {"temperature": "0.8"}, {"top_k": 0},
                        {"top_k": 257}, {"top_k": True}, {"max_new_tokens": 257},
                        {"generation_id": "bad"}, {"prompt": ""}, {"prompt": " "},
                        {"checkpoint_path": "/private/file.pt"}):
                with self.subTest(bad=bad):
                    self.assertEqual(422, client.post("/api/generate", json=self.payload(**bad)).status_code)
            self.assertEqual(404, client.post("/api/generate", json=self.payload(model_id="other")).status_code)
            self.assertEqual(415, client.post("/api/generate", data={"prompt": "x"}).status_code)
            self.assertEqual(413, client.post("/api/generate", json={"prompt": "x" * 5000}).status_code)
            self.assertEqual(403, client.get("/api/models", headers={"Origin": "https://example.org"}).status_code)
            self.assertEqual(400, client.get("/api/models", headers={"Host": "example.org"}).status_code)
            cors = client.options("/api/generate", headers={"Origin": "http://localhost:13000",
                                  "Access-Control-Request-Method": "POST",
                                  "Access-Control-Request-Headers": "content-type"})
            self.assertEqual(200, cors.status_code)
            self.assertEqual("http://localhost:13000", cors.headers["access-control-allow-origin"])

    def test_cancel_and_busy_slot_release(self):
        entered, unblock = Event(), Event()
        model = ScriptedModel(entered=entered, unblock=unblock)
        request = self.payload(max_new_tokens=16)
        with self.client(model) as client, ThreadPoolExecutor(max_workers=1) as worker:
            future = worker.submit(client.post, "/api/generate", json=request)
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(409, client.post("/api/generate", json=self.payload()).status_code)
                endpoint = f"/api/generations/{request['generation_id']}/cancel"
                self.assertEqual(200, client.post(endpoint).status_code)
            finally:
                unblock.set()
            events = [json.loads(line) for line in future.result(timeout=5).text.splitlines()]
            self.assertTrue(events[-1]["cancelled"])
            self.assertEqual(1, events[-1]["generated_tokens"])
            self.assertEqual(404, client.post(endpoint).status_code)
            self.assertEqual(200, client.post("/api/generate", json=self.payload()).status_code)

    def test_safe_stream_error_releases_slot(self):
        class BrokenModel:
            config = SimpleNamespace(context_length=128)

            def __call__(self, _inputs):
                raise RuntimeError("private prompt and absolute path")

        with self.client(BrokenModel()) as client:
            for _ in range(2):
                response = client.post("/api/generate", json=self.payload())
                self.assertEqual(200, response.status_code)
                self.assertEqual("error", json.loads(response.text.splitlines()[-1])["type"])
                self.assertNotIn("private prompt", response.text)
