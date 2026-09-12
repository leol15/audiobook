"""TTS backends. One protocol, many implementations, selected by name."""

from __future__ import annotations

import importlib

from ab.tts.base import TTSBackend

_REGISTRY = {
    "kokoro": "ab.tts.kokoro:KokoroBackend",
    "chatterbox": "ab.tts.chatterbox:ChatterboxBackend",
    "subprocess": "ab.tts.subprocess:SubprocessBackend",
    "tone": "ab.tts.tone:ToneBackend",
}


def load_backend(name: str, params: dict | None = None) -> TTSBackend:
    if name not in _REGISTRY:
        raise SystemExit(f"unknown tts backend {name!r}; known: {', '.join(_REGISTRY)}")
    mod, cls = _REGISTRY[name].split(":")
    try:
        backend_cls = getattr(importlib.import_module(mod), cls)
    except ImportError as e:
        raise SystemExit(f"backend {name!r} not installed: {e}. Try: uv sync --extra {name}") from e
    return backend_cls(**(params or {}))
