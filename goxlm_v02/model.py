"""GoXLM 0.2 retains the original decoder architecture and byte vocabulary."""
from dataclasses import asdict
import math

import torch

from model.transformer import GoXLM, GoXLMConfig

TOKENIZER = {"version": "utf8-byte-0.2.1", "vocab_size": 256, "document_prefix": 10}
MODEL = asdict(GoXLMConfig(vocab_size=256))
MODEL_VERSION = "goxlm-0.2-baseline"
EXPERIMENT_VERSION = "goxlm-0.4-experimental"
ARCHITECTURES = {
    "baseline": (4, 4, 256),
    "10m": (6, 6, 384),
    "25m": (8, 8, 512),
}


def experiment_config(architecture, context_length=128):
    if architecture not in ARCHITECTURES:
        raise ValueError("Unsupported experimental architecture")
    if type(context_length) is not int or not 32 <= context_length <= 1024:
        raise ValueError("Experimental context length must be an integer from 32 to 1024")
    layers, heads, width = ARCHITECTURES[architecture]
    return {**MODEL, "context_length": context_length, "n_layers": layers,
            "n_heads": heads, "n_embd": width}


def conversation_config(vocab_size, context_length=256):
    if type(context_length) is not int or context_length not in (256, 512):
        raise ValueError("Conversation context length must be 256 or 512")
    return {**experiment_config("25m", context_length), "vocab_size": vocab_size}


def encode(text):
    return list(text.encode("utf-8"))


def decode(tokens, *, errors="strict"):
    return bytes(tokens).decode("utf-8", errors=errors)


def new_model(config=None):
    return GoXLM(GoXLMConfig(**(MODEL if config is None else config)))


def windows(text, context=MODEL["context_length"]):
    """Each target byte occurs once. Windows never cross documents."""
    yield from token_windows([TOKENIZER["document_prefix"], *encode(text)], context)


def token_windows(tokens, context):
    for start in range(0, len(tokens) - 1, context):
        part = tokens[start:start + context + 1]
        width = len(part) - 1
        yield part[:-1] + [0] * (context - width), part[1:] + [-100] * (context - width)


@torch.inference_mode()
def generate(model, prompt, *, steps, temperature=1.0, top_k=40, seed=42):
    if model.config.vocab_size != TOKENIZER["vocab_size"]:
        raise ValueError("Checkpoint vocabulary does not match byte tokenizer")
    if type(steps) is not int or not 0 <= steps <= 4096:
        raise ValueError("steps must be an integer from 0 through 4096")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if type(top_k) is not int or not 1 <= top_k <= TOKENIZER["vocab_size"]:
        raise ValueError("top_k must be between 1 and 256")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    context = [TOKENIZER["document_prefix"], *encode(prompt)]
    produced = []
    model.eval()
    for _ in range(steps):
        idx = torch.tensor([context[-model.config.context_length:]], device=device)
        logits, _ = model(idx)
        values, indices = (logits[0, -1].float() / temperature).topk(top_k)
        sample = torch.multinomial(values.softmax(dim=-1), 1, generator=rng)
        token = indices[sample].item()
        context.append(token)
        produced.append(token)
    return decode(produced, errors="replace")
