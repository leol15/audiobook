"""Content-hash keyed artifact store.

Every stage output is addressed by a hash of the inputs that produced it, so
re-running skips finished work and changing one thing invalidates only what
depends on it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def content_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (dict, list, tuple)):
            p = json.dumps(p, sort_keys=True, ensure_ascii=False)
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def render_key(backend_name: str, voice: str, params: dict, text: str) -> str:
    return content_hash("render", backend_name, voice, params, text)


def stamp_path(artifact: Path) -> Path:
    return artifact.with_suffix(artifact.suffix + ".inputs")


def is_fresh(artifact: Path, inputs_hash: str) -> bool:
    """True if artifact exists and was built from inputs with this hash."""
    stamp = stamp_path(artifact)
    return artifact.exists() and stamp.exists() and stamp.read_text().strip() == inputs_hash


def mark_fresh(artifact: Path, inputs_hash: str) -> None:
    stamp_path(artifact).write_text(inputs_hash)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:24]
