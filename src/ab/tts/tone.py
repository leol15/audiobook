"""Test backend: emits a short tone per chunk, no model needed.

Lets the render/build path (cache, chunking, pauses, ffmpeg) be exercised in
tests and on machines without a GPU. Duration scales with text length.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class ToneBackend:
    name = "tone-test"
    max_chars = 200
    sample_rate = 24000
    languages: ClassVar[set[str]] = {"en", "zh"}

    def __init__(self, chars_per_second: float = 15.0, **_):
        self.cps = chars_per_second

    def voices(self) -> list[str]:
        return ["a220", "a330", "a440"]

    def synthesize(self, text: str, voice: str, *, lang: str, **_) -> np.ndarray:
        freq = float(voice.lstrip("a")) if voice.lstrip("a").isdigit() else 440.0
        n = max(1, int(self.sample_rate * len(text) / self.cps))
        t = np.arange(n) / self.sample_rate
        env = np.minimum(1.0, np.minimum(t, t[::-1]) * 50)  # 20 ms fade in/out
        return (0.2 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)
