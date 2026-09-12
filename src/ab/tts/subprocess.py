"""Runs any backend in a separate venv and talks JSON over stdin/stdout.

Escape hatch for TTS packages whose torch pins cannot coexist in one venv.
Protocol: one JSON request per line {text, voice, lang, params}; one JSON
response per line {path} pointing to a wav the child wrote. (Milestone 4.)
"""

from __future__ import annotations


class SubprocessBackend:
    def __init__(self, **params):
        raise NotImplementedError("subprocess backend arrives in milestone 4")
