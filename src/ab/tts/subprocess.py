"""Runs a TTS model in a separate venv and talks JSON lines over stdin/stdout.

Escape hatch for TTS packages whose torch pins cannot coexist with the main
venv (Chatterbox is the first). See backends/chatterbox/worker.py for the
protocol. The worker loads its model once and stays alive for the whole run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]


class SubprocessBackend:
    name = "subprocess"
    max_chars = 300
    sample_rate = 24000
    languages: ClassVar[set[str]] = {"en", "zh"}

    def __init__(self, python: str, worker: str, name: str | None = None,
                 max_chars: int | None = None, **params):
        self.python = str(ROOT / python) if not Path(python).is_absolute() else python
        self.worker = str(ROOT / worker) if not Path(worker).is_absolute() else worker
        if name:
            self.name = name
        if max_chars:
            self.max_chars = max_chars
        self.params = params
        self._proc: subprocess.Popen | None = None
        self._tmp = Path(tempfile.mkdtemp(prefix="ab-tts-"))

    def _start(self) -> None:
        if self._proc is not None:
            return
        if not Path(self.python).exists():
            raise SystemExit(f"worker venv not found at {self.python}; see backends/*/pyproject.toml")
        env = {**os.environ, "TQDM_DISABLE": "1", "TRANSFORMERS_VERBOSITY": "error",
               "PYTHONWARNINGS": "ignore"}
        self._proc = subprocess.Popen(
            [self.python, self.worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1, env=env,
        )
        hello = self._read()
        if not hello.get("ready"):
            raise SystemExit(f"worker failed to start: {hello}")
        self.sample_rate = int(hello.get("sr", self.sample_rate))

    def _read(self) -> dict:
        """Next JSON line from the worker; anything else is forwarded to stderr."""
        assert self._proc and self._proc.stdout
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise SystemExit(f"worker exited (code {self._proc.poll()})")
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
            sys.stderr.write(f"[worker] {line}")

    def voices(self) -> list[str]:
        return ["default", "<path to reference wav>"]

    def synthesize(self, text: str, voice: str, *, lang: str, **params) -> np.ndarray:
        self._start()
        assert self._proc and self._proc.stdin
        out = self._tmp / "chunk.wav"
        req = {"text": text, "voice": voice, "lang": lang, "out": str(out),
               "params": {**self.params, **params}}
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        res = self._read()
        if not res.get("ok"):
            raise RuntimeError(f"worker error: {res.get('error')}")
        audio, sr = sf.read(res["path"], dtype="float32")
        if sr != self.sample_rate:
            raise RuntimeError(f"worker sample rate {sr} != {self.sample_rate}")
        return audio.reshape(-1)

    def close(self) -> None:
        if self._proc and self._proc.stdin:
            self._proc.stdin.close()
            self._proc.wait(timeout=30)
            self._proc = None

    def __del__(self):
        import contextlib

        with contextlib.suppress(Exception):
            self.close()
