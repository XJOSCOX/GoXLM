"""Opt-in CUDA checks for GoXLM, using synthetic data and temporary checkpoints."""
import os
import unittest
from unittest.mock import patch

import torch

from goxlm_v02.model import new_model
from goxlm_v02 import test_v02
from training.cuda_benchmark import FP16_INITIAL_SCALE, autocast, supported_precisions


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(42)

    def test_forward_backward_optimizer_in_supported_precisions(self):
        for mode, dtype in supported_precisions(self.device).items():
            with self.subTest(precision=mode):
                model = new_model().to(self.device).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False)
                inputs = torch.randint(256, (2, model.config.context_length), device=self.device)
                targets = inputs.roll(-1, dims=1)
                before = model.token_embedding.weight.detach().clone()
                with autocast(dtype):
                    logits, loss = model(inputs, targets)
                self.assertEqual((2, model.config.context_length, 256), logits.shape)
                self.assertEqual(dtype or torch.float32, logits.dtype)
                self.assertEqual(torch.float32, loss.dtype)
                self.assertTrue(bool(loss.isfinite()))
                scaler = (torch.amp.GradScaler("cuda", init_scale=FP16_INITIAL_SCALE)
                          if dtype == torch.float16 else None)
                if scaler is None:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(parameter.grad.isfinite().all()))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.assertGreater(norm.item(), 0)
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    self.assertEqual(FP16_INITIAL_SCALE, scaler.get_scale())
                self.assertFalse(torch.equal(before, model.token_embedding.weight))
                self.assertTrue(bool(model.token_embedding.weight.isfinite().all()))
                self.assertEqual(torch.float32, model.token_embedding.weight.dtype)

    def test_causal_attention_in_supported_precisions(self):
        model = new_model().to(self.device).eval()
        length = model.config.context_length
        source = torch.randint(256, (2, length), device=self.device)
        altered = source.clone()
        altered[:, length // 2:] = (altered[:, length // 2:] + 1) % 256
        for mode, dtype in supported_precisions(self.device).items():
            with self.subTest(precision=mode), torch.inference_mode(), autocast(dtype):
                first, _ = model(source)
                second, _ = model(altered)
                torch.testing.assert_close(first[:, :length // 2], second[:, :length // 2],
                                           rtol=0, atol=0)
                self.assertFalse(torch.equal(first[:, -1], second[:, -1]))

    def test_checkpoint_save_and_bitwise_resume(self):
        self.check_resume("fp32")

    def test_bf16_checkpoint_save_and_bitwise_resume(self):
        if "bf16" not in supported_precisions(self.device):
            self.skipTest("Native CUDA BF16 is unavailable")
        self.check_resume("bf16")

    def check_resume(self, precision):
        # Exercise the versioned checkpoint and deterministic RNG path.
        fixture = test_v02.PipelineTests("test_reproducible_resume_and_evaluation")
        self.addCleanup(fixture.doCleanups)
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        with patch.dict(os.environ, {"GOXLM_TEST_DEVICE": "cuda"}):
            fixture.setUp()
            fixture.check_checkpoint_resume(precision)
