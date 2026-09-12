"""Artifact schemas shared between stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Chapter(BaseModel):
    index: int
    title: str
    paragraphs: list[str]


class ChapterList(BaseModel):
    language: str
    chapters: list[Chapter]


class Line(BaseModel):
    """One utterance: a narration or dialogue span to be rendered in one voice."""

    id: str
    chapter: int
    para: int
    kind: Literal["narration", "dialogue"]
    speaker: str  # "narrator" or a canonical cast name
    text: str
    lang: str = "en"
    style: str | None = None  # reserved for emotion/style tags
    confidence: float = 1.0
    locked: bool = False  # hand-edited; attribute stage must not overwrite

    # Filled by render/verify.
    audio: str | None = None
    backend: str | None = None
    attempts: int = 0
    error_rate: float | None = None


# Fields of Line that the attribute stage owns (04-lines/cNNN.jsonl).
ATTRIBUTION_FIELDS = ("id", "chapter", "para", "kind", "speaker", "text", "lang", "style",
                      "confidence", "locked")


class RenderState(BaseModel):
    """Render's per-line record (05-render/cNNN.jsonl). audio None = must (re)render."""

    id: str
    audio: str | None = None
    backend: str | None = None
    attempts: int = 0


class VerifyResult(BaseModel):
    id: str
    transcript: str
    error_rate: float
    edits: int = 0
    duration: float
    ok: bool
    audio: str | None = None  # the file that was checked; stale once render replaces it
