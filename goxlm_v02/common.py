"""Artifact IO and baseline verification."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = {
    "checkpoints/goxlm_v01.pt": "94fd77371db338df7e158d9d284e4b103fd9d108cf3614aa16336eb71932150b",
    "data/approved/science_technology_v01.txt": "fe32a2c7ed6035142346a7b2e626fec435b95bdae87bdc68f96122d32296e452",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def publish(path, write):
    """Publish a complete file atomically, refusing to replace any destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def write_json(path, value):
    publish(path, lambda stream: stream.write(canonical(value)))


def verify_baseline():
    for name, expected in BASELINE.items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"GoXLM-0.1 artifact hash mismatch: {name}")
    return BASELINE
