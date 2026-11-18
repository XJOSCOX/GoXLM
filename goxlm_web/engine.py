"""Local model allowlist, checkpoint adapters, and bounded byte completion."""
import codecs
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Annotated, BinaryIO, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
import torch

from goxlm_v02.common import ROOT
from goxlm_v02.model import MODEL, TOKENIZER, new_model
from goxlm_v02.run import CHECKPOINT, LEGACY_CHECKPOINT, load_checkpoint

from .cache import CachedDecoder

PARAMETERS = 3_257_856
MAX_NEW_TOKENS = 256
ModelId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9.-]{0,63}$")]


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: ModelId
    label: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    format: Literal["0.1", "0.2"]
    path: Path
    sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["auto", "cuda", "cpu"] = "auto"
    models: list[ModelEntry] = Field(min_length=1, max_length=32)

    @field_validator("models")
    @classmethod
    def unique_ids(cls, models):
        if len({entry.id for entry in models}) != len(models):
            raise ValueError("Model IDs must be unique")
        return models


class CheckpointError(ValueError):
    pass


class GenerationCancelled(Exception):
    pass


def check_cancelled(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise GenerationCancelled


@dataclass(frozen=True)
class Device:
    torch_device: torch.device
    label: str
    fallback: bool = False


def resolve_device(requested):
    if requested == "cpu":
        return Device(torch.device("cpu"), "CPU")
    if torch.cuda.is_available():
        try:
            return Device(torch.device("cuda:0"), torch.cuda.get_device_name(0))
        except RuntimeError:
            if requested == "cuda":
                raise RuntimeError("CUDA is unavailable for local inference") from None
    elif requested == "cuda":
        raise RuntimeError("CUDA is unavailable for local inference")
    return Device(torch.device("cpu"), "CPU", fallback=True)


def legacy_model(stream):
    checkpoint = torch.load(stream, map_location="cpu", weights_only=True, mmap=False)
    step = checkpoint.get("step")
    data_digest = checkpoint.get("data_sha256")
    if (checkpoint.get("config") != MODEL or type(step) is not int or step < 1
            or not isinstance(data_digest, str) or not re.fullmatch(r"[a-f0-9]{64}", data_digest)):
        raise CheckpointError("Unsupported GoXLM-0.1 checkpoint metadata")
    weights = checkpoint["model"]
    if not torch.equal(weights["lm_head.weight"], weights["token_embedding.weight"]):
        raise CheckpointError("Invalid tied checkpoint weights")
    if any(tensor.dtype != torch.float32 or not tensor.isfinite().all().item()
           for tensor in weights.values()):
        raise CheckpointError("Checkpoint weights must be finite FP32")
    with torch.random.fork_rng(devices=[]):
        model = new_model()
    model.load_state_dict(weights, strict=True)
    return model, {"checkpoint_schema": "goxlm-0.1", "training_precision": "fp32",
                   "checkpoint_step": step}


def versioned_model(stream):
    model, saved = load_checkpoint(stream, "cpu", allowed_schemas=(LEGACY_CHECKPOINT, CHECKPOINT))
    return model, {"checkpoint_schema": saved["schema"],
                   "training_precision": saved.get("precision", "fp32"),
                   "checkpoint_step": saved["step"]}


@dataclass(frozen=True)
class ModelFormat:
    load: Callable[[BinaryIO], tuple[torch.nn.Module, dict]]
    prefix: int | None
    tokenizer_version: str
    context_length: int = MODEL["context_length"]
    vocab_size: int = MODEL["vocab_size"]
    parameter_count: int = PARAMETERS


FORMATS = {"0.1": ModelFormat(legacy_model, None, "utf8-byte-0.1"),
           "0.2": ModelFormat(versioned_model, TOKENIZER["document_prefix"], TOKENIZER["version"])}


def model_info(entry, verified=None, resident=False):
    spec = FORMATS[entry.format]
    return {"id": entry.id, "label": entry.label, "format": entry.format,
            "parameter_count": spec.parameter_count, "context_length": spec.context_length,
            "vocab_size": spec.vocab_size, "tokenizer": "utf8-byte",
            "prefix_tokens": int(spec.prefix is not None), "max_new_tokens": MAX_NEW_TOKENS,
            "mode": "experimental-text-completion", "tokenizer_version": spec.tokenizer_version,
            "status": "loaded" if resident else "validated" if verified else "unverified",
            **(verified or {})}


def prompt_tokens(entry, text):
    tokens = list(text.encode("utf-8"))
    spec = FORMATS[entry.format]
    if spec.prefix is not None:
        tokens.insert(0, spec.prefix)
    if len(tokens) > spec.context_length:
        raise ValueError("Input exceeds the model context. Shorten the prompt.")
    return tokens


def load_record(entry, device, root=ROOT, cancelled=None):
    # Hash and deserialize through the same open file. The operator owns the allowlist.
    path = root / entry.path
    try:
        check_cancelled(cancelled)
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                check_cancelled(cancelled)
                digest.update(chunk)
            if digest.hexdigest() != entry.sha256:
                raise CheckpointError("Checkpoint hash does not match the allowlist")
            check_cancelled(cancelled)
            stream.seek(0)
            model, metadata = FORMATS[entry.format].load(stream)
        spec = FORMATS[entry.format]
        if (model.config.context_length != spec.context_length
                or model.config.vocab_size != spec.vocab_size
                or sum(p.numel() for p in model.parameters()) != spec.parameter_count):
            raise CheckpointError("Checkpoint dimensions do not match the supported format")
    except (CheckpointError, GenerationCancelled):
        raise
    except OSError:
        raise CheckpointError("Configured checkpoint is unavailable") from None
    except Exception:
        raise CheckpointError("Checkpoint could not be validated for the configured format") from None
    check_cancelled(cancelled)
    model = model.eval().to(device)
    check_cancelled(cancelled)
    return model, metadata


def load_local(entry, device, root=ROOT):
    return load_record(entry, device, root)[0]


class ModelRegistry:
    """The broker provides exclusive use while loading or generating."""

    def __init__(self, settings, device):
        self.entries = {entry.id: entry for entry in settings.models}
        self.device = device.torch_device
        self.current_id = None
        self.current = None
        self.verified = {}

    def info(self, entry):
        return model_info(entry, self.verified.get(entry.id), self.current_id == entry.id)

    def get(self, entry, cancelled=None):
        check_cancelled(cancelled)
        if entry.id != self.current_id:
            self.current_id = None
            self.current = None
            self.verified.pop(entry.id, None)
            model, metadata = load_record(entry, self.device, cancelled=cancelled)
            self.current = model
            self.verified[entry.id] = metadata
            self.current_id = entry.id
        return self.current


class Completion:
    def __init__(self, model, tokens, device, temperature, top_k, strategy="full", seed=None):
        self.model = model
        self.tokens = list(tokens)
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.cache = CachedDecoder(model, device) if strategy == "cached" else None

    @torch.inference_mode()
    def step(self):
        if self.cache is None:
            window = self.tokens[-self.model.config.context_length:]
            inputs = torch.tensor([window], dtype=torch.long, device=self.device)
            logits, _ = self.model(inputs)
            last = logits[0, -1]
        else:
            last = self.cache.logits(self.tokens)
        scores, indices = torch.topk(last.float() / self.temperature, self.top_k)
        token = indices[0].item() if self.top_k == 1 else indices[torch.multinomial(
            scores.softmax(dim=-1), 1, generator=self.generator)].item()
        self.tokens.append(token)
        return self.decoder.decode(bytes([token]))

    def finish(self):
        return self.decoder.decode(b"", final=True)

    def close(self):
        if self.cache is not None:
            self.cache.close()
