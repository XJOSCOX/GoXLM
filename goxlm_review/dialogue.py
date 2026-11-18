"""Private dialogue intake, audit, and review staging. This module never approves records."""

import argparse
from collections import Counter
from datetime import date
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from goxlm_v02.common import canonical, digest, write_json
from goxlm_v02.overlap import terms
from goxlm_v02.tokenizer import ByteBPE, load_tokenizer
from goxlm_web.conversation import ConversationBudgetError, Message, format_conversation

from .staging import batch, relative_file

Origin = Literal["human_written", "ai_assisted", "synthetic"]
Ref = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9._:-]{1,128}$")]
SHA = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
PILOT_SCHEMA = "goxlm-instruction-pilot-draft-v1"
SCORE_NAMES = {"naturalness", "intent", "accuracy", "context", "completeness", "format"}


class StrictRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Evidence(StrictRecord):
    ref: Ref
    sha256: SHA
    kind: Literal["signature_or_terms", "authorship_declaration", "third_party_source",
                  "fact_check", "fixture_provenance", "tool_terms", "other"]


class Tool(StrictRecord):
    tool_ref: Ref
    version_ref: Ref
    purpose: str = Field(min_length=1, max_length=500)
    scope_description: str = Field(min_length=1, max_length=1000)


class Utterance(StrictRecord):
    text: Annotated[str, StringConstraints(pattern=r"\S", max_length=12000)]
    origin: Origin


class Turn(StrictRecord):
    user: Utterance
    assistant: Utterance


class Draft(StrictRecord):
    schema_version: Literal["goxlm-instruction-pilot-draft-v1"]
    record_id: Ref
    revision: int = Field(ge=1)
    supersedes_record_sha256: SHA | None = None
    language: Literal["en"]
    task_class: Literal["everyday_dialogue", "everyday_qa", "followup_correction",
                        "clarification", "general_explanation", "basic_stem"]
    scenario_family_ref: Ref
    source_family_refs: list[Ref]
    authoring_origin: Origin
    requested_use: Literal["train_candidate", "evaluation_candidate", "plumbing_only", "scope_review"]
    contributor_ref: Ref
    declaration_ref: Ref
    evidence_refs: list[Evidence] = Field(min_length=1)
    tool_provenance: Tool | None = None
    turns: list[Turn] = Field(min_length=1, max_length=6)
    turns_sha256: SHA

    @model_validator(mode="after")
    def validate_provenance(self):
        permitted = {"human_written": {"train_candidate", "evaluation_candidate"},
                     "ai_assisted": {"scope_review"}, "synthetic": {"plumbing_only"}}
        origins = {getattr(t, role).origin for t in self.turns for role in ("user", "assistant")}
        allowed = {"human_written", "ai_assisted"} if self.authoring_origin == "ai_assisted" else {
            self.authoring_origin}
        if (self.requested_use not in permitted[self.authoring_origin]
                or self.authoring_origin not in origins or not origins <= allowed
                or (self.tool_provenance is None) != (self.authoring_origin == "human_written")):
            raise ValueError("Origin, affected turns, tool provenance, and requested use disagree")
        if (self.revision == 1) != (self.supersedes_record_sha256 is None):
            raise ValueError("Revised drafts must reference the superseded record")
        if len({item.ref for item in self.evidence_refs}) != len(self.evidence_refs):
            raise ValueError("Evidence references must be unique")
        if self.turns_sha256 != digest(canonical(self.model_dump()["turns"])):
            raise ValueError("Dialogue text and origin hash mismatch")
        return self

    def messages(self):
        return [Message(role, getattr(pair, role).text)
                for pair in self.turns for role in ("user", "assistant")]


class QualityReview(StrictRecord):
    draft_sha256: SHA
    reviewer_ref: Ref
    review_date: str
    scores: dict[str, Annotated[int, Field(ge=0, le=2)]]

    def verify(self, draft_sha):
        if (date.fromisoformat(self.review_date).isoformat() != self.review_date
                or self.draft_sha256 != draft_sha or self.scores.keys() != SCORE_NAMES
                or min(self.scores.values()) < 1 or sum(self.scores.values()) < 10
                or self.scores["accuracy"] != 2 or self.scores["context"] != 2):
            raise ValueError("Quality review does not meet the rubric for the exact draft")


def seal(value):
    return Draft.model_validate({**value, "turns_sha256": digest(canonical(value["turns"]))})


def inspect_context(draft, tokenizer, context):
    messages = draft.messages()
    for index in range(1, len(messages), 2):
        response = messages[index]
        reserve = len(tokenizer.encode(response.text)) + 1
        if reserve >= context:
            return False
        try:
            result = format_conversation(messages[:index], tokenizer.chat_tokenizer(),
                                         context_length=context, generation_tokens=reserve)
        except ConversationBudgetError:
            return False
        if result.dropped_turns:
            return False
    return True


def audit(drafts, tokenizer, context=256):
    if type(context) is not int or context < 4:
        raise ValueError("Invalid pilot context length")
    rows = []
    keys = set()
    for draft in drafts:
        key = (draft.record_id, draft.revision)
        if key in keys:
            raise ValueError("Repeated draft ID and revision")
        keys.add(key)
        messages = draft.messages()
        words = tuple(word for message in messages for word in terms(message.text))
        grams = set(zip(*(words[offset:] for offset in range(5))))
        rows.append((draft, key, words, grams))
    findings = []
    for (_, left, words, grams), (_, right, other, rest) in combinations(rows, 2):
        shared = len(grams & rest)
        if words and words == other:
            findings.append([left, right, "normalized_exact"])
        elif shared >= 12 and (shared / len(grams | rest) >= .80 or shared / min(len(grams), len(rest)) >= .90):
            findings.append([left, right, "near_duplicate"])
    return {"drafts": len(rows),
            "origins": dict(Counter(d.authoring_origin for d, *_ in rows)),
            "categories": dict(Counter(d.task_class for d, *_ in rows)),
            "multi_turn": sum(len(d.turns) >= 2 for d, *_ in rows),
            "context": context,
            "context_compatible": sum(inspect_context(d, tokenizer, context) for d, *_ in rows),
            "overlap_findings": findings}


def stage(store, draft_path, quality_path, evidence_index, input_root, tokenizer_path=None):
    raw = Path(draft_path).read_bytes()
    draft = Draft.model_validate_json(raw)
    if draft.authoring_origin != "human_written" or draft.requested_use != "train_candidate":
        raise ValueError("Only human-written training candidates can enter this review stage")
    if not {"authorship_declaration", "signature_or_terms"} <= {e.kind for e in draft.evidence_refs}:
        raise ValueError("The draft needs authorship and rights evidence")
    tokenizer = (load_tokenizer(tokenizer_path) if tokenizer_path is not None
                 else ByteBPE("0" * 64, "0" * 64, 259, ()))
    if not inspect_context(draft, tokenizer, 256):
        raise ValueError("The complete draft does not fit the 256-token instruction context")
    index = json.loads(Path(evidence_index).read_bytes())
    quality = Path(quality_path).read_bytes()
    grade = QualityReview.model_validate_json(quality)
    grade.verify(digest(raw))
    if grade.reviewer_ref == draft.contributor_ref:
        raise ValueError("Quality reviewer must differ from the contributor")
    with TemporaryDirectory(prefix="goxlm-dialogue-bridge-") as directory:
        root = Path(directory)

        def capture(name, contents):
            (root / name).write_bytes(contents)
            return {"file": name, "sha256": digest(contents)}

        refs = []
        for i, evidence in enumerate(draft.evidence_refs):
            data = relative_file(Path(input_root).resolve(strict=True), index[evidence.ref]).read_bytes()
            if digest(data) != evidence.sha256:
                raise ValueError(f"Evidence hash mismatch: {evidence.ref}")
            refs.append({**capture(f"evidence-{i}", data), "role": evidence.kind, "reference": evidence.ref})
        refs.append({**capture("quality.json", quality), "role": "quality_review", "reference": "Exact draft rubric"})
        content = canonical([{"role": m.role, "text": m.text} for m in draft.messages()])
        case_id = "dialogue-" + digest(canonical([draft.record_id, draft.revision]))[:40]
        item = {"id": case_id, "title": "Original dialogue draft",
                "template": "original_dialogue", "domain": "general", "family_id": draft.scenario_family_ref,
                "source": capture("draft.json", raw), "extract": capture("turns.json", content),
                "evidence": refs, "attribution": draft.contributor_ref,
                "rights_summary": "Author declaration and license require independent verification.",
                "license_conditions": "Confirm commercial training, scope, attribution and withdrawal with the exact evidence.",
                "extraction_method": "Canonical role messages v1",
                "exceptions": ["Instruction-only review store. Never use as pretraining text."]}
        return {**batch(store, [item], root), "case_id": case_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("schema")
    seal_cmd = commands.add_parser("seal")
    seal_cmd.add_argument("input", type=Path)
    seal_cmd.add_argument("output", type=Path)
    check = commands.add_parser("audit")
    check.add_argument("input", type=Path, nargs="+")
    check.add_argument("--tokenizer", type=Path)
    check.add_argument("--context", type=int, default=256)
    stage_cmd = commands.add_parser("stage")
    for name in ("store", "draft", "quality", "evidence_index", "input_root"):
        stage_cmd.add_argument(name, type=Path)
    stage_cmd.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if args.action == "schema":
        result = Draft.model_json_schema()
    elif args.action == "seal":
        value = seal(json.loads(args.input.read_bytes()))
        write_json(args.output, value.model_dump(exclude_none=True))
        return
    elif args.action == "audit":
        tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else ByteBPE("0"*64, "0"*64, 259, ())
        result = audit([Draft.model_validate_json(path.read_bytes()) for path in args.input], tokenizer, args.context)
    else:
        result = stage(args.store, args.draft, args.quality, args.evidence_index, args.input_root, args.tokenizer)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
