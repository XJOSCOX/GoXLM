"""Artifact IO and baseline verification."""
import hashlib
import json
import os
import re
import subprocess
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


def source_fingerprint(root, paths):
    return digest(canonical({path.relative_to(root).as_posix(): digest(path.read_bytes())
                             for path in paths}))


def git_provenance(root=None):
    """Snapshot the source checkout, with null values when Git is unavailable."""
    root = ROOT if root is None else Path(root)
    unavailable = {"commit_sha": None, "dirty": None}
    # An archive inside an unrelated repository has no GoXLM Git provenance.
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL, env=env, timeout=5,
        )
        if Path(os.fsdecode(top.rstrip(b"\n"))).resolve() != root.resolve():
            return unavailable
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v2", "--branch",
             "--untracked-files=normal", "--ignore-submodules=none"],
            stderr=subprocess.DEVNULL, env=env, timeout=5,
        ).splitlines()
    except (OSError, subprocess.SubprocessError):
        return unavailable

    prefix = b"# branch.oid "
    oid = next((line[len(prefix):] for line in status if line.startswith(prefix)), None)
    if oid == b"(initial)":
        commit = None
    elif oid is not None and re.fullmatch(rb"[0-9a-f]{40}(?:[0-9a-f]{24})?", oid):
        commit = oid.decode("ascii")
    else:
        return unavailable
    return {"commit_sha": commit, "dirty": any(not line.startswith(b"# ") for line in status)}


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
