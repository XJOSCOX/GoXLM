"""Token-level chat formatting for future conversation-capable checkpoints."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

FORMAT_VERSION = "goxlm-conversation-v1"


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class ConversationTokenizer:
    """Checkpoint-specific text encoder and the disjoint control token IDs."""
    encode_text: Callable[[str], Iterable[int]]
    text_vocab_size: int
    vocab_size: int
    user_start: int
    assistant_start: int
    end_turn: int

    def __post_init__(self):
        sizes = self.text_vocab_size, self.vocab_size
        if any(type(size) is not int for size in sizes) or not 0 < sizes[0] < sizes[1]:
            raise ValueError("Invalid conversation vocabulary sizes")
        tokens = self.user_start, self.assistant_start, self.end_turn
        if (any(type(token) is not int or not self.text_vocab_size <= token < self.vocab_size
                for token in tokens) or len(set(tokens)) != len(tokens)):
            raise ValueError("Conversation markers must be distinct tokens outside the text vocabulary")

    def _encode_message(self, message: Message) -> tuple[int, ...]:
        content = tuple(self.encode_text(message.text))
        if not content or any(type(token) is not int or not 0 <= token < self.text_vocab_size
                              for token in content):
            raise ValueError("Conversation text encoder emitted empty or non-text tokens")
        start = self.user_start if message.role == "user" else self.assistant_start
        return (start, *content, self.end_turn)


@dataclass(frozen=True)
class ConversationPrompt:
    version: str
    input_ids: tuple[int, ...]
    reserved_generation_tokens: int
    dropped_turns: int


class ConversationBudgetError(ValueError):
    """The latest user message and its framing cannot fit the prompt budget."""


def format_conversation(messages: Sequence[Message], tokenizer: ConversationTokenizer, *,
                        context_length: int, generation_tokens: int) -> ConversationPrompt:
    """Keep the latest user message and the newest contiguous suffix of complete turns."""
    if (type(context_length) is not int or type(generation_tokens) is not int
            or not 0 < generation_tokens < context_length):
        raise ValueError("Generation reserve must be positive and smaller than the context length")

    messages = tuple(messages)
    if not messages or len(messages) % 2 == 0:
        raise ValueError("Conversation must end with a user message")
    for i, message in enumerate(messages):
        expected = "assistant" if i % 2 else "user"
        if not isinstance(message, Message) or message.role != expected:
            raise ValueError("Conversation must alternate user and assistant roles")
        if not isinstance(message.text, str) or not message.text.strip():
            raise ValueError("Conversation messages must contain nonempty text")

    budget = context_length - generation_tokens
    latest = (*tokenizer._encode_message(messages[-1]), tokenizer.assistant_start)
    remaining = budget - len(latest)
    if remaining < 0:
        raise ConversationBudgetError("Latest user message cannot fit with the requested generation reserve")

    turns = []
    for i in range(len(messages) - 3, -1, -2):
        turn = (*tokenizer._encode_message(messages[i]), *tokenizer._encode_message(messages[i + 1]))
        if len(turn) > remaining:
            break
        turns.append(turn)
        remaining -= len(turn)

    prefix = [token for turn in reversed(turns) for token in turn]
    return ConversationPrompt(FORMAT_VERSION, (*prefix, *latest), generation_tokens,
                              len(messages) // 2 - len(turns))
