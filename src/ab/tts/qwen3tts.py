"""Qwen3-TTS (Apache 2.0) via the subprocess backend.

Lives in backends/qwen3tts/.venv. Set up with:  cd backends/qwen3tts && uv sync

Voices: a preset speaker (Vivian, Serena, Uncle_Fu, Dylan, Eric for Mandarin;
Ryan, Aiden for English) or a path to a reference wav for cloning. Params:
size ("1.7B" | "0.6B"), instruct (style text for presets), seed.
Autoregressive: keep verify on.
"""

from __future__ import annotations

from ab.tts.subprocess import SubprocessBackend

PRESETS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]


class Qwen3TTSBackend(SubprocessBackend):
    def __init__(self, size: str = "1.7B", **params):
        super().__init__(
            python="backends/qwen3tts/.venv/bin/python",
            worker="backends/qwen3tts/worker.py",
            name=f"qwen3tts-{size}",
            max_chars=300,
            size=size,
            **params,
        )

    def voices(self) -> list[str]:
        return PRESETS + ["<path to reference wav>"]
