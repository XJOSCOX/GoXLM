"""Deterministic byte BPE and training-only catalog artifacts."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
import heapq
import json
from pathlib import Path
import re

from goxlm_web.conversation import FORMAT_VERSION, ConversationTokenizer

from .common import BASELINE, canonical, digest, write_json
from .data import catalog_lock, load_release

SCHEMA = "goxlm-byte-bpe-v1"
MAX_VOCAB = 4096
MIN_VOCAB = 259
CHUNK_SIZE = 64
_SEGMENTS = re.compile(rb"[A-Za-z0-9_]+|\s+|[^A-Za-z0-9_\s]+")


def _chunks(text):
    for match in _SEGMENTS.finditer(text.encode("utf-8")):
        value = match.group()
        for offset in range(0, len(value), CHUNK_SIZE):
            yield value[offset:offset + CHUNK_SIZE]


def _pairs(tokens):
    return Counter(zip(tokens, tokens[1:]))


def _replace(tokens, pair, symbol):
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(symbol)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return tuple(result)


def _fit_bpe(texts, target_vocab):
    if type(target_vocab) is not int or not MIN_VOCAB <= target_vocab <= MAX_VOCAB:
        raise ValueError("Invalid target vocabulary size")
    counts = Counter(chunk for text in texts for chunk in _chunks(text))
    weights = list(counts.values())
    chunks = [tuple(chunk) for chunk in counts]
    totals = Counter()
    members = defaultdict(set)
    for index, tokens in enumerate(chunks):
        for pair, count in _pairs(tokens).items():
            totals[pair] += count * weights[index]
            members[pair].add(index)
    queue = [(-count, pair) for pair, count in totals.items()]
    heapq.heapify(queue)
    merges = []
    while queue and len(merges) < target_vocab - MIN_VOCAB:
        negative_count, pair = heapq.heappop(queue)
        if -negative_count < 2:
            break
        if totals[pair] != -negative_count:
            continue
        changed = set()
        symbol = 256 + len(merges)
        for i in sorted(members[pair]):
            old = _pairs(chunks[i])
            new = _replace(chunks[i], pair, symbol)
            current = _pairs(new)
            chunks[i] = new
            for retired in old.keys() - current.keys():
                members[retired].remove(i)
            for added in current.keys() - old.keys():
                members[added].add(i)
            for other in old.keys() | current.keys():
                totals[other] += (current[other] - old[other]) * weights[i]
                changed.add(other)
        for other in changed:
            if totals[other] >= 2:
                heapq.heappush(queue, (-totals[other], other))
        merges.append(pair)
    return tuple(merges)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class ByteBPE:
    dataset_version: str
    training_fingerprint: str
    target_vocab: int
    merges: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if (not _sha(self.dataset_version) or not _sha(self.training_fingerprint)
                or type(self.target_vocab) is not int
                or not MIN_VOCAB <= self.vocab_size <= self.target_vocab <= MAX_VOCAB):
            raise ValueError("Invalid tokenizer artifact metadata")
        seen = set()
        lengths = [1] * 256
        for symbol, pair in enumerate(self.merges, 256):
            if (len(pair) != 2 or any(type(part) is not int or not 0 <= part < symbol for part in pair)
                    or pair in seen):
                raise ValueError("Invalid tokenizer merge")
            seen.add(pair)
            length = lengths[pair[0]] + lengths[pair[1]]
            if length > CHUNK_SIZE:
                raise ValueError("Tokenizer merge exceeds the chunk width")
            lengths.append(length)

    @property
    def text_vocab_size(self):
        return 256 + len(self.merges)

    @property
    def vocab_size(self):
        return self.text_vocab_size + 3

    @cached_property
    def pieces(self):
        pieces = [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            pieces.append(pieces[left] + pieces[right])
        return tuple(pieces)

    @cached_property
    def ranks(self):
        return {pair: index for index, pair in enumerate(self.merges, 256)}

    def encode(self, text):
        result = []
        for chunk in _chunks(text):
            tokens = tuple(chunk)
            while len(tokens) > 1:
                best = min((self.ranks.get(pair, self.vocab_size)
                            for pair in zip(tokens, tokens[1:])), default=self.vocab_size)
                if best == self.vocab_size:
                    break
                tokens = _replace(tokens, self.merges[best - 256], best)
            result.extend(tokens)
        return result

    def decode(self, tokens):
        pieces = []
        for token in tokens:
            if type(token) is not int or not 0 <= token < self.text_vocab_size:
                raise ValueError("Cannot decode a conversation marker as ordinary text")
            pieces.append(self.pieces[token])
        return b"".join(pieces).decode("utf-8")

    def payload(self):
        return {"schema": SCHEMA, "dataset_version": self.dataset_version,
                "training_fingerprint": self.training_fingerprint, "target_vocab": self.target_vocab,
                "merges": self.merges}

    def identity(self):
        return {"schema": SCHEMA, "sha256": digest(canonical(self.payload())),
                "dataset_version": self.dataset_version, "training_fingerprint": self.training_fingerprint,
                "vocab_size": self.vocab_size, "text_vocab_size": self.text_vocab_size}

    def chat_tokenizer(self):
        return ConversationTokenizer(self.encode, self.text_vocab_size, self.vocab_size,
                                     *range(self.text_vocab_size, self.vocab_size))

    def format_identity(self):
        return format_identity(self.vocab_size)


def format_identity(vocab_size):
    return {"version": FORMAT_VERSION, "user_start": vocab_size - 3,
            "assistant_start": vocab_size - 2, "end_turn": vocab_size - 1}


def validate_identity(identity, release_version):
    required = {"schema", "sha256", "dataset_version", "training_fingerprint",
                "vocab_size", "text_vocab_size"}
    if (not isinstance(identity, dict) or set(identity) != required or identity["schema"] != SCHEMA
            or identity["dataset_version"] != release_version
            or not _sha(identity["sha256"]) or not _sha(identity["training_fingerprint"])
            or type(identity["vocab_size"]) is not int or type(identity["text_vocab_size"]) is not int
            or not MIN_VOCAB <= identity["vocab_size"] <= MAX_VOCAB
            or identity["text_vocab_size"] != identity["vocab_size"] - 3):
        raise ValueError("Invalid checkpoint tokenizer identity")
    return identity["vocab_size"]


def load_tokenizer(path):
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or set(payload) != {"schema", "dataset_version",
                                                         "training_fingerprint", "target_vocab", "merges"}
            or payload["schema"] != SCHEMA
            or raw != canonical(payload) or path.stem != digest(raw)
            or not isinstance(payload["merges"], list)
            or any(not isinstance(pair, list) for pair in payload["merges"])):
        raise ValueError("Invalid tokenizer artifact or digest")
    return ByteBPE(payload["dataset_version"], payload["training_fingerprint"],
                   payload["target_vocab"], tuple(map(tuple, payload["merges"])))


def training_rows(dataset):
    return sorted((row for row in dataset["documents"] if row["split"] == "train"),
                  key=lambda row: row["document_id"])


def training_fingerprint(rows):
    return digest(canonical([(row["document_id"], row["document_sha256"]) for row in rows]))


def fit_catalog(catalog, release, directory, target_vocab=MAX_VOCAB):
    release = Path(release)
    rows = training_rows(load_release(catalog, release))
    if any(row["document_sha256"] == BASELINE["data/approved/science_technology_v01.txt"]
           for row in rows):
        raise ValueError("The historical demonstration corpus is excluded from tokenizer fitting")
    tokenizer = ByteBPE(release.stem, training_fingerprint(rows), target_vocab,
                        _fit_bpe((row["text"] for row in rows), target_vocab))
    path = Path(directory) / (tokenizer.identity()["sha256"] + ".json")
    with catalog_lock(catalog):
        load_release(catalog, release)
        try:
            write_json(path, tokenizer.payload())
        except FileExistsError:
            if load_tokenizer(path) != tokenizer:
                raise ValueError("Existing tokenizer artifact differs from fitted tokenizer") from None
    return path
