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


def is_fresh(artifact: Path, inputs: str | dict) -> bool:
    """True if artifact exists and was built from these inputs."""
    return artifact.exists() and read_stamp(artifact) == _norm(inputs)


def mark_fresh(artifact: Path, inputs: str | dict) -> None:
    """Record what the artifact was built from. A dict of named components is
    stored as JSON so `ab status` can explain which input changed."""
    stamp_path(artifact).write_text(json.dumps(_norm(inputs), sort_keys=True))


def read_stamp(artifact: Path) -> dict | str | None:
    stamp = stamp_path(artifact)
    if not stamp.exists():
        return None
    raw = stamp.read_text().strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _norm(inputs):
    return inputs if isinstance(inputs, dict) else str(inputs)


def stale_reasons(artifact: Path, inputs: dict) -> list[str]:
    """Which named inputs differ from the stamp (empty list == fresh)."""
    if not artifact.exists():
        return ["missing"]
    old = read_stamp(artifact)
    if not isinstance(old, dict):
        return ["no stamp"]
    return [k for k in inputs if old.get(k) != inputs[k]] or (["unknown"] if old != inputs else [])


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:24]
