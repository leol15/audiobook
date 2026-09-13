"""Qwen3-TTS (Apache 2.0) via the subprocess backend.

Lives in backends/qwen3tts/.venv. Set up with:  cd backends/qwen3tts && uv sync

Voices: a preset speaker (Vivian, Serena, Uncle_Fu, Dylan, Eric for Mandarin;
Ryan, Aiden for English) or a path to a reference wav for cloning. Params:
size ("1.7B" | "0.6B"), attn ("sdpa" | "flash_attention_2"), instruct (style
text for presets), seed. Autoregressive: keep verify on.

Batches: the worker generates a whole batch of lines in one call and the call
takes about as long as a single line (the decode loop is launch-bound), so
`tts.batch` in book.yaml is the main speed knob. See DESIGN.md.
"""

from __future__ import annotations

from ab.tts.subprocess import SubprocessBackend

PRESETS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]


class Qwen3TTSBackend(SubprocessBackend):
    batch_size = 24  # line-count ceiling; tts.batch_tokens (memory) usually cuts first. See DESIGN.md

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

    def design(self, text: str, *, lang: str, instruct: str, out: str) -> str:
        """Create a reference clip for a voice described in words (VoiceDesign model)."""
        self.request({"kind": "design", "text": text, "lang": lang, "instruct": instruct,
                      "out": out, "params": dict(self.params)})
        return out
