"""Per-book configuration (book.yaml) and cast (cast.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Language = Literal["en", "zh"]


class Pauses(BaseModel):
    sentence: float = 0.35
    paragraph: float = 0.6
    speaker_change: float = 1.2
    chapter: float = 2.5


class TTSConfig(BaseModel):
    backend: str = "kokoro"
    params: dict = Field(default_factory=dict)  # backend-specific, part of cache key


class VerifyConfig(BaseModel):
    threshold: float = 0.15
    max_attempts: int = 3
    whisper_model: str | None = None  # default chosen by language


class BookConfig(BaseModel):
    title: str
    author: str = ""
    language: Language = "en"
    cover: str | None = None
    chapter_regex: str | None = None
    tts: TTSConfig = Field(default_factory=TTSConfig)
    # role -> voice id, where roles are "narrator", "_default", or a cast name.
    # Either one flat map, or one map per backend name:
    #   voices: {narrator: bm_george}                     # applies to any backend
    #   voices: {kokoro: {narrator: bm_george}, chatterbox: {narrator: default}}
    voices: dict[str, str | dict[str, str]] = Field(default_factory=dict)

    def voice_map(self, backend: str) -> dict[str, str]:
        nested = {k: v for k, v in self.voices.items() if isinstance(v, dict)}
        if nested:
            if backend not in nested:
                raise SystemExit(f"book.yaml voices has no map for backend {backend!r}")
            return nested[backend]
        return {k: v for k, v in self.voices.items() if isinstance(v, str)}
    pauses: Pauses = Field(default_factory=Pauses)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    loudness_lufs: float = -18.0


class Character(BaseModel):
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class Cast(BaseModel):
    characters: dict[str, Character] = Field(default_factory=dict)

    def resolve(self, name: str) -> str | None:
        """Map a name or alias to a canonical cast name.

        Tolerant of case, punctuation, and spacing ("mrs bennet" == "Mrs. Bennet"),
        since both regexes and the LLM produce slightly different surface forms.
        """
        key = _fold(name)
        if not key:
            return None
        for canon, ch in self.characters.items():
            if key == _fold(canon) or any(key == _fold(a) for a in ch.aliases):
                return canon
        return None


def _fold(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


class BookPaths:
    """Filesystem layout of one book directory."""

    def __init__(self, root: Path):
        self.root = root
        self.config = root / "book.yaml"
        self.cast = root / "cast.yaml"
        self.overrides = root / "overrides.yaml"
        self.work = root / "work"
        self.out = root / "out"
        self.chapters = self.work / "chapters.json"
        self.chapters_norm = self.work / "chapters.norm.json"
        self.lines = self.work / "lines.jsonl"
        self.audio = self.work / "audio"
        self.verify = self.work / "verify.jsonl"

    @property
    def source(self) -> Path:
        for name in ("source.md", "source.txt"):
            p = self.root / name
            if p.exists():
                return p
        raise FileNotFoundError(f"no source.md or source.txt in {self.root}")

    def load_config(self) -> BookConfig:
        return BookConfig.model_validate(yaml.safe_load(self.config.read_text(encoding="utf-8")))

    def load_cast(self) -> Cast:
        if not self.cast.exists():
            return Cast()
        return Cast.model_validate(yaml.safe_load(self.cast.read_text(encoding="utf-8")) or {})

    def load_overrides(self) -> dict[str, str]:
        if not self.overrides.exists():
            return {}
        return yaml.safe_load(self.overrides.read_text(encoding="utf-8")) or {}
