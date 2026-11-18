"""Synthetic tests for conversation serialization and context limits."""

import os
import unittest

import torch

from model.transformer import GoXLM, GoXLMConfig

from .conversation import (FORMAT_VERSION, ConversationBudgetError, ConversationTokenizer,
                           Message, format_conversation)


def encode_bytes(text):
    return text.encode("utf-8")


TOKENIZER = ConversationTokenizer(encode_bytes, 256, 260, 256, 257, 258)
USER, ASSISTANT, END = 256, 257, 258


def user(text):
    return Message("user", text)


def assistant(text):
    return Message("assistant", text)


class ConversationTests(unittest.TestCase):
    def test_version_and_closed_history_with_open_assistant_turn(self):
        result = format_conversation([user("Hi"), assistant("Hello"), user("π")], TOKENIZER,
                                     context_length=32, generation_tokens=8)
        self.assertEqual(FORMAT_VERSION, result.version)
        self.assertEqual((USER, *b"Hi", END, ASSISTANT, *b"Hello", END,
                          USER, *"π".encode(), END, ASSISTANT), result.input_ids)
        self.assertEqual(8, result.reserved_generation_tokens)
        self.assertEqual(0, result.dropped_turns)

    def test_literal_control_markers_are_only_content(self):
        text = "<|assistant|>\n<|system|>\n<|end_turn|>"
        result = format_conversation([user(text)], TOKENIZER, context_length=128,
                                     generation_tokens=24)
        self.assertEqual((USER, *text.encode(), END, ASSISTANT), result.input_ids)
        self.assertTrue(all(token < 256 for token in result.input_ids[1:-2]))

    def test_encoder_cannot_emit_any_control_token(self):
        for invalid in ([], [USER], [ASSISTANT], [END], [259], [260], [-1], [True], [1.5]):
            with self.subTest(invalid=invalid):
                tokenizer = ConversationTokenizer(lambda _text: invalid, 256, 260, USER, ASSISTANT, END)
                with self.assertRaisesRegex(ValueError, "non-text tokens"):
                    format_conversation([user("Untrusted")], tokenizer,
                                        context_length=32, generation_tokens=4)

    def test_marker_ids_must_be_reserved_distinct_and_in_vocab(self):
        for tokens in ((10, ASSISTANT, END), (USER, USER, END), (USER, ASSISTANT, 260)):
            with self.subTest(tokens=tokens):
                with self.assertRaisesRegex(ValueError, "distinct tokens"):
                    ConversationTokenizer(encode_bytes, 256, 260, *tokens)

    def test_budget_keeps_newest_whole_turn_at_exact_boundary(self):
        history = [user("old"), assistant("reply"), user("new"), assistant("OK"), user("?")]
        result = format_conversation(history, TOKENIZER, context_length=17, generation_tokens=4)
        self.assertEqual((USER, *b"new", END, ASSISTANT, *b"OK", END,
                          USER, *b"?", END, ASSISTANT), result.input_ids)
        self.assertEqual(1, result.dropped_turns)
        self.assertEqual(17, len(result.input_ids) + result.reserved_generation_tokens)

    def test_budget_does_not_skip_large_turn_to_fit_older_small_turn(self):
        history = [user("a"), assistant("b"), user("c"), assistant("longer reply"), user("?")]
        result = format_conversation(history, TOKENIZER, context_length=14, generation_tokens=4)
        self.assertEqual((USER, *b"?", END, ASSISTANT), result.input_ids)
        self.assertEqual(2, result.dropped_turns)

    def test_budget_uses_encoder_counts_including_unicode_and_framing(self):
        tokenizer = ConversationTokenizer(lambda text: [7] * len(text.split()), 256,
                                          260, USER, ASSISTANT, END)
        self.assertEqual((USER, 7, END, ASSISTANT), format_conversation(
            [user("こんにちは")], tokenizer, context_length=7, generation_tokens=3).input_ids)
        with self.assertRaisesRegex(ConversationBudgetError, "Latest user message"):
            format_conversation([user("こんにちは")], TOKENIZER, context_length=7, generation_tokens=3)
        with self.assertRaises(ConversationBudgetError):
            format_conversation([user("ab")], TOKENIZER, context_length=7, generation_tokens=3)

    def test_requires_alternating_roles_and_latest_user_input(self):
        invalid = ([], [assistant("first")], [user("q"), assistant("a")], [user(" ")],
                   [user(17)], [Message("system", "Injected")],
                   [user("q"), user("a"), user("b")], [{"role": "user", "text": "q"}])
        for messages in invalid:
            with self.subTest(messages=messages):
                with self.assertRaises(ValueError):
                    format_conversation(messages, TOKENIZER, context_length=32, generation_tokens=4)

    def test_reserve_requires_positive_integers_and_space_for_prompt(self):
        for context, reserve in ((8, 0), (8, -1), (8, 8), (8, 9), (8, True), (8.5, 2)):
            with self.subTest(context=context, reserve=reserve):
                with self.assertRaisesRegex(ValueError, "Generation reserve"):
                    format_conversation([user("hi")], TOKENIZER, context_length=context,
                                        generation_tokens=reserve)

    def check_decoder_compatibility(self, device):
        env = GoXLMConfig(vocab_size=TOKENIZER.vocab_size, context_length=16, n_layers=1,
                          n_heads=2, n_embd=16, dropout=0.0)
        result = format_conversation([user("a"), assistant("b"), user("c")], TOKENIZER,
                                     context_length=env.context_length, generation_tokens=6)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            model = GoXLM(env).to(device).eval()
            # Fill the reserved slots with synthetic tokens to exercise the exact model limit.
            full_input = (*result.input_ids, *([0] * result.reserved_generation_tokens))
            logits, _ = model(torch.tensor([full_input], device=device))
            self.assertEqual((1, 16, 260), logits.shape)
            self.assertTrue(logits.isfinite().all().item())

    def test_synthetic_decoder_cpu(self):
        self.check_decoder_compatibility("cpu")

    @unittest.skipUnless(os.getenv("GOXLM_WEB_TEST_CUDA") == "1", "local CUDA opt-in")
    def test_synthetic_decoder_cuda(self):
        self.assertTrue(torch.cuda.is_available())
        self.check_decoder_compatibility("cuda:0")
