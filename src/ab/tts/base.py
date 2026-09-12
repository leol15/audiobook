from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TTSBackend(Protocol):
    name: str            # includes version; part of every cache key
    max_chars: int       # segmenter splits utterances above this
    sample_rate: int
    languages: set[str]  # e.g. {"en", "zh"}

    def voices(self) -> list[str]: ...

    def synthesize(self, text: str, voice: str, *, lang: str, **params) -> np.ndarray:
        """Return mono float32 audio at self.sample_rate."""
        ...
