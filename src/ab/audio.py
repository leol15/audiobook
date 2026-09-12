"""Small numpy audio helpers. Loudness and encoding are delegated to ffmpeg in build."""

from __future__ import annotations

import numpy as np


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def crossfade_concat(pieces: list[np.ndarray], sr: int, fade_ms: float = 10.0) -> np.ndarray:
    pieces = [np.asarray(p, dtype=np.float32).reshape(-1) for p in pieces if len(p)]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    n = int(sr * fade_ms / 1000)
    out = pieces[0]
    for p in pieces[1:]:
        k = min(n, len(out), len(p))
        if k == 0:
            out = np.concatenate([out, p])
            continue
        ramp = np.linspace(0, 1, k, dtype=np.float32)
        mixed = out[-k:] * (1 - ramp) + p[:k] * ramp
        out = np.concatenate([out[:-k], mixed, p[k:]])
    return out


def trim_silence(audio: np.ndarray, sr: int, threshold: float = 1e-3, pad_ms: float = 50.0) -> np.ndarray:
    idx = np.flatnonzero(np.abs(audio) > threshold)
    if len(idx) == 0:
        return audio
    pad = int(sr * pad_ms / 1000)
    return audio[max(0, idx[0] - pad) : min(len(audio), idx[-1] + pad)]
