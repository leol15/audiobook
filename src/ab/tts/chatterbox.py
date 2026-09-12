"""Chatterbox backend (milestone 4). Reference-audio voices, exaggeration knob."""

from __future__ import annotations

from typing import ClassVar


class ChatterboxBackend:
    name = "chatterbox-multilingual"
    max_chars = 300
    sample_rate = 24000
    languages: ClassVar[set[str]] = {"en", "zh"}

    def __init__(self, **params):
        raise NotImplementedError("Chatterbox backend arrives in milestone 4")
