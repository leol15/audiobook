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
    # Lines per model call for backends that batch (qwen3tts). None = backend default.
    # Not part of the cache key: batching changes throughput, not the cache identity.
    batch: int | None = None


class LLMConfig(BaseModel):
    model: str | None = None   # default: OLLAMA_MODEL env or qwen3:14b
    # Context window requested from Ollama. Its default is 4096 and it truncates
    # silently; cast/attribute windows are sized from this so prompts always fit.
    num_ctx: int = 16384


class CastConfig(BaseModel):
    # Characters ranked highest by (rule-attributed) line count are the main
    # cast: listed in every attribute prompt and expected to have a voice.
    # The rest are listed only in chapters where they appear and fall back to
    # the _default voice.
    main_cap: int = 20
    # Chapters whose discovered characters are merged per LLM call, against
    # the running cast (also bounded by the context budget).
    merge_chapters: int = 5


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
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cast: CastConfig = Field(default_factory=CastConfig)
    # role -> voice id, where roles are "narrator", "_default", or a cast name.
    # Either one flat map, or one map per backend name:
    #   voices: {narrator: bm_george}                     # applies to any backend
    #   voices: {kokoro: {narrator: bm_george}, chatterbox: {narrator: default}}
    voices: dict[str, str | dict[str, str]] = Field(default_factory=dict)
    # Used by `ab voices-design` for the narrator (cast members use cast.yaml descriptions).
    narrator_description: str | None = None

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
    main: bool = True              # in every attribute prompt; give it a voice in book.yaml
    lines: int = 0                 # dialogue lines resolved by rules at cast time (ranking hint)
    chapters: list[int] = Field(default_factory=list)  # chapter indexes where the character speaks


class Cast(BaseModel):
    characters: dict[str, Character] = Field(default_factory=dict)

    def names_for_chapter(self, chapter: int) -> list[str]:
        """Main cast plus characters known to speak in this chapter, in cast order."""
        return [n for n, c in self.characters.items() if c.main or chapter in c.chapters]

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
    """Filesystem layout of one book directory. Work artifacts are numbered by stage."""

    def __init__(self, root: Path):
        self.root = root
        self.config = root / "book.yaml"
        self.cast = root / "cast.yaml"
        self.overrides = root / "overrides.yaml"
        self.voices_dir = root / "voices"
        self.work = root / "work"
        self.out = root / "out"
        w = self.work
        self.chapters = w / "01-chapters.json"
        self.chapters_norm = w / "02-chapters.norm.json"
        self.llm_log = w / "03-llm.jsonl"          # cast + attribute exchanges
        self.cast_work = w / "03-cast"             # per-chapter discovery results (cache)
        self.lines = w / "04-lines.jsonl"
        self.script = w / "04-script.md"           # human-readable view of lines
        self.attribute_review = w / "04-attribute.review.txt"
        self.audio = w / "05-audio"
        self.audio_by_line = w / "05-audio" / "by-line"
        self.verify = w / "06-verify.jsonl"
        self.verify_review = w / "06-verify.review.txt"
        self.chapter_audio = w / "07-chapters"
        self.ffmetadata = w / "07-ffmetadata.txt"
        self.run_log = w / "run.log"
        self.report = w / "REPORT.md"

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
