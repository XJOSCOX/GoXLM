"""Synthetic coverage for deterministic artifacts, compatibility, and dialogue staging."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from goxlm_review.dialogue import (Draft, PILOT_SCHEMA, SCORE_NAMES, audit, record_quality, seal, stage)
from goxlm_review.ledger import status
from goxlm_review.staging import init
from goxlm_v02 import test_v02 as fixtures
from goxlm_v02.common import canonical, digest
from goxlm_v02.data import load_release, revoke
from goxlm_v02.run import CHECKPOINT, CONVERSATION_CHECKPOINT, RunConfig, load_checkpoint, train
from goxlm_v02.tokenizer import ByteBPE, _fit_bpe, fit_catalog, load_tokenizer, training_fingerprint, training_rows
from goxlm_web.conversation import Message, format_conversation

torch.set_num_threads(1)


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PipelineTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_byte_round_trip_recursive_merges_and_role_ids(self):
        texts = ["é é π² 👩‍🔬  😀\tdef f():\r\n    return 42 \n"] * 8
        merges = _fit_bpe(texts, 310)
        self.assertEqual(merges, _fit_bpe(reversed(texts), 310))
        tokenizer = ByteBPE("1" * 64, "2" * 64, 310, merges)
        for text in texts + ["▁\r\n  \x00\u200d字 ", "x" * 205, "<|assistant|>"]:
            self.assertEqual(text.encode(), tokenizer.decode(tokenizer.encode(text)).encode())
        text = "<|assistant|>"
        result = format_conversation([Message("user", text)], tokenizer.chat_tokenizer(),
                                     context_length=256, generation_tokens=20)
        self.assertEqual(text, tokenizer.decode(result.input_ids[1:-2]))
        self.assertEqual(len(set(tokenizer.format_identity().values())), 4)
        with self.assertRaises(ValueError):
            tokenizer.decode([tokenizer.text_vocab_size])

    def test_artifact_byte_identity_and_holdout_provenance(self):
        fixture = self.fixture
        release = fixture.release()
        snapshot = load_release(fixture.catalog, release)
        first = fit_catalog(fixture.catalog, release, fixture.root / "first", 276)
        again = fit_catalog(fixture.catalog, release, fixture.root / "again", 276)
        artifact = load_tokenizer(first)
        self.assertEqual(first.read_bytes(), again.read_bytes())
        self.assertEqual(first.stem, artifact.identity()["sha256"])
        self.assertEqual(training_fingerprint(training_rows(snapshot)), artifact.training_fingerprint)
        self.assertEqual(first, fit_catalog(fixture.catalog, release, fixture.root / "first", 276))

        # Changing held-out prose cannot change learned pieces or the train fingerprint.
        held_out = {**snapshot, "documents": [
            dict(row, text="漢字📌 " * 80, document_sha256="5" * 64) if row["split"] != "train" else row
            for row in snapshot["documents"]]}
        with patch("goxlm_v02.tokenizer.load_release", return_value=held_out):
            isolated = fit_catalog(fixture.catalog, release, fixture.root / "isolated", 276)
        self.assertEqual(first.read_bytes(), isolated.read_bytes())

        first.write_bytes(first.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "digest"):
            load_tokenizer(first)
        with self.assertRaisesRegex(ValueError, "chunk width"):
            ByteBPE("1"*64, "2"*64, 300, ((0, 0), *((i, i) for i in range(256, 262))))

        row = training_rows(snapshot)[0]
        revoke(fixture.catalog, row["document_id"], {"submission_sha256": row["submission_sha256"],
               "reviewer": "Synthetic", "revoked_on": "2026-11-18", "reason": "Test",
               "notice_reference": "Synthetic revocation"})
        blocked = fixture.root / "blocked"
        with self.assertRaisesRegex(ValueError, "revoked"):
            fit_catalog(fixture.catalog, release, blocked)
        self.assertFalse(blocked.exists())

    def test_conversation_checkpoint_metadata_and_adapter_boundary(self):
        fixture = self.fixture
        release = fixture.release()
        artifact = fit_catalog(fixture.catalog, release, fixture.root / "tokens", 276)
        with patch("goxlm_v02.run.verify_baseline"):
            path = train(fixture.catalog, release, fixture.root / "run", 1, "cpu",
                         RunConfig(batch_size=1), architecture="conversation", tokenizer_path=artifact)
        model, saved = load_checkpoint(path)
        self.assertEqual(CONVERSATION_CHECKPOINT, saved["schema"])
        self.assertEqual("base_pretraining", saved["training_stage"])
        self.assertEqual(load_tokenizer(artifact).identity(), saved["tokenizer"])
        self.assertEqual(256, model.config.context_length)
        with self.assertRaisesRegex(ValueError, "schema for this reader"):
            load_checkpoint(path, allowed_schemas=(CHECKPOINT,))
        wrong = fixture.root / "invalid.pt"
        torch.save(dict(saved, conversation_format={"version": "untrusted"}), wrong)
        with self.assertRaisesRegex(ValueError, "format or stage"):
            load_checkpoint(wrong)
        with patch("goxlm_v02.run.verify_baseline"):
            other = fit_catalog(fixture.catalog, release, fixture.root / "other", 278)
            with self.assertRaisesRegex(ValueError, "matching data"):
                train(fixture.catalog, release, fixture.root / "resume", 2, "cpu", RunConfig(batch_size=1),
                      resume=path, architecture="conversation", tokenizer_path=other)
        self.assertFalse((fixture.root / "resume").exists())


def draft_value(origin):
    text = "Demo text"
    value = {"schema_version": PILOT_SCHEMA, "record_id": "test-record", "revision": 1,
             "language": "en", "task_class": "everyday_qa", "scenario_family_ref": "test-family",
             "source_family_refs": [], "authoring_origin": origin,
             "requested_use": {"human_written": "train_candidate", "synthetic": "plumbing_only",
                               "ai_assisted": "scope_review"}[origin],
             "contributor_ref": "fixture-author", "declaration_ref": "declaration",
             "evidence_refs": [{"ref": ref, "sha256": digest(ref.encode()), "kind": kind}
                               for ref, kind in (("declaration", "authorship_declaration"),
                                                 ("terms", "signature_or_terms"))],
             "turns": [{"user": {"text": text, "origin": origin},
                        "assistant": {"text": "Demo answer", "origin": origin}}]}
    if origin != "human_written":
        value["tool_provenance"] = {"tool_ref": "fixture", "version_ref": "v1",
                                    "purpose": "test", "scope_description": "Synthetic intent"}
    return value


def quality_value(raw, **overrides):
    return {"draft_sha256": digest(raw), "reviewer_ref": "fixture-review", "review_date": "2026-11-18",
            "decision": "accepted", "scores": {key: 2 for key in SCORE_NAMES},
            "naturalness_note": "Synthetic test rubric.", "accuracy_note": "Fixture has no factual claim.",
            "fact_check_refs": [], "followup_note": "Single-turn fixture.",
            "privacy_note": "No identifying facts in fixture.", "duplicate_note": "Independent fixture.",
            **overrides}


class DialogueTests(unittest.TestCase):
    def test_origin_hash_context_and_duplicate_checks(self):
        tokenizer = ByteBPE("0" * 64, "0" * 64, 259, ())
        one = seal(draft_value("synthetic"))
        self.assertEqual("plumbing_only", one.requested_use)
        two = seal({**draft_value("synthetic"), "record_id": "copy"})
        stats = audit([one, two], tokenizer)
        self.assertEqual(2, stats["context_compatible"])
        self.assertEqual(1, len(stats["overlap_findings"]))
        self.assertEqual(0, audit([one], tokenizer, context=9)["context_compatible"])
        with self.assertRaisesRegex(ValueError, "requested use"):
            seal({**draft_value("synthetic"), "requested_use": "train_candidate"})
        value = one.model_dump()
        value["turns"][0]["assistant"]["text"] += " changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            Draft.model_validate(value)

    def test_stage_human_draft_without_any_decision_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "reviews"
            init(store, ["fixture"])
            record, decision, review, index = (root / name for name in
                                                ("draft.json", "decision.json", "quality.json", "index.json"))
            for origin in ("synthetic", "ai_assisted"):
                record.write_bytes(canonical(seal(draft_value(origin)).model_dump()))
                with self.assertRaisesRegex(ValueError, "human-written"):
                    stage(store, record, review, index, root)
            for key in ("declaration", "terms"):
                (root / key).write_bytes(key.encode())
            index.write_text(json.dumps({k: k for k in ("declaration", "terms")}))
            record.write_bytes(canonical(seal(draft_value("human_written")).model_dump()))
            quality = quality_value(record.read_bytes())
            decision.write_bytes(canonical(quality))
            with patch("goxlm_review.dialogue.local_operator", return_value="fixture"), \
                    patch("goxlm_review.dialogue.sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", return_value=quality["draft_sha256"]):
                record_quality(store, record, decision, review)
            envelope = json.loads(review.read_bytes())
            bad_review = root / "corrupt-quality.json"
            envelope["quality"]["review"]["privacy_note"] = "Tampered"
            bad_review.write_bytes(canonical(envelope))
            with self.assertRaisesRegex(ValueError, "integrity"):
                stage(store, record, bad_review, index, root)
            with self.assertRaisesRegex(ValueError, "recorded quality"):
                stage(store, record, decision, index, root)
            (root / "terms").write_bytes(b"mismatch")
            with self.assertRaisesRegex(ValueError, "Evidence hash mismatch"):
                stage(store, record, review, index, root)
            (root / "terms").write_bytes(b"terms")
            result = stage(store, record, review, index, root)
            cases, _, events, state, _ = status(store)
            self.assertIn(result["case_id"], cases)
            self.assertEqual([], events)
            self.assertEqual({}, state)


    def test_privacy_and_cross_batch_overlap_report_do_not_echo_content(self):
        tokenizer = ByteBPE("0" * 64, "0" * 64, 259, ())
        value = draft_value("synthetic")
        value["turns"][0]["assistant"]["text"] = "Try placeholder@example.invalid for this fixture."
        flagged = seal(value)
        counterpart = seal({**value, "record_id": "previous"})
        stats = audit([flagged], tokenizer, against=[counterpart])
        self.assertEqual({"test-record.r1": ["email"]}, stats["privacy_flags"])
        self.assertEqual(1, len(stats["overlap_findings"]))
        self.assertNotIn("placeholder", str(stats))

    def test_quality_acceptance_requires_interactive_hash_and_bound_facts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            draft, decision, output = (root / name for name in ("draft.json", "decision.json", "quality.json"))
            value = dict(draft_value("human_written"), task_class="basic_stem")
            draft.write_bytes(canonical(seal(value).model_dump()))
            grade = quality_value(draft.read_bytes())
            decision.write_bytes(canonical(grade))
            with patch("goxlm_review.dialogue.local_operator", return_value="fixture"):
                with patch("goxlm_review.dialogue.sys.stdin.isatty", return_value=False):
                    with self.assertRaisesRegex(ValueError, "interactive"):
                        record_quality(root, draft, decision, output)
                with patch("goxlm_review.dialogue.sys.stdin.isatty", return_value=True):
                    with self.assertRaisesRegex(ValueError, "Fact-check"):
                        record_quality(root, draft, decision, output)
                    decision.write_bytes(canonical(dict(grade, decision="hold")))
                    with patch("builtins.input", return_value="a" * 64):
                        with self.assertRaisesRegex(ValueError, "did not match"):
                            record_quality(root, draft, decision, output)
            self.assertFalse(output.exists())
