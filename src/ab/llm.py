"""Thin Ollama client. Ollama runs on Windows; reached over HTTP (see DESIGN.md)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")


def _load_dotenv() -> None:
    """Read OLLAMA_* from a .env in the cwd or its parents, without overriding the environment."""
    for d in [Path.cwd(), *Path.cwd().parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return


_load_dotenv()


class Ollama:
    def __init__(self, url: str | None = None, model: str = DEFAULT_MODEL, timeout: float = 300,
                 log: Path | None = None):
        self.url = (url or os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.model = model
        self.client = httpx.Client(base_url=self.url, timeout=timeout)
        self.log = log  # every exchange appended as one JSON line, for debugging

    def ping(self) -> list[str]:
        r = self.client.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def json(self, prompt: str, schema: dict, system: str = "", retries: int = 2) -> dict:
        """Generate a JSON object conforming to schema (Ollama structured output)."""
        body = {"model": self.model, "prompt": prompt, "system": system, "format": schema,
                "stream": False, "options": {"temperature": 0}}
        last: Exception | None = None
        for _ in range(retries + 1):
            r = self.client.post("/api/generate", json=body)
            r.raise_for_status()
            raw = r.json().get("response", "")
            if self.log:
                with self.log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"model": self.model, "prompt": prompt, "response": raw},
                                       ensure_ascii=False) + "\n")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                last = e
        raise RuntimeError(f"ollama returned non-JSON after retries: {last}")

    def unload(self) -> None:
        """Free VRAM before a large TTS model loads."""
        self.client.post("/api/generate", json={"model": self.model, "keep_alive": 0})
