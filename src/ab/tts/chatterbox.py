"""Chatterbox Multilingual via the subprocess backend.

Lives in backends/chatterbox/.venv because chatterbox-tts pins torch 2.6.
Set up with:  cd backends/chatterbox && uv sync

Voices: "default" or a path to a 5-15 s reference wav. Params: exaggeration,
cfg_weight, temperature. Roughly realtime on the GPU.
"""

from __future__ import annotations

from ab.tts.subprocess import SubprocessBackend


class ChatterboxBackend(SubprocessBackend):
    def __init__(self, **params):
        super().__init__(
            python="backends/chatterbox/.venv/bin/python",
            worker="backends/chatterbox/worker.py",
            name="chatterbox-mtl",
            max_chars=300,
            **params,
        )
