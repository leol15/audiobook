"""Stage 1: source.md / source.txt -> work/chapters.json"""

from __future__ import annotations

import re

from ab import cache
from ab.config import BookConfig, BookPaths
from ab.models import Chapter, ChapterList
from ab.text import canonicalize_quotes

_CHAPTER_RE = {
    "en": re.compile(r"^\s*(chapter|part|book)\s+([0-9]+|[ivxlc]+|\w+)\b.*$", re.IGNORECASE),
    "zh": re.compile(r"^\s*第[0-9一二三四五六七八九十百千零〇]+[章回节卷部].*$"),
}
_BARE_NUMERAL_RE = re.compile(r"^\s*([0-9]+|[IVXLC]+)\.?\s*$")
_MD_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*#*\s*$")
_GUTENBERG_START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)
_GUTENBERG_END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)

# Full-width punctuation that the segmenter and TTS handle better in canonical form.
_ZH_PUNCT = str.maketrans({"　": " ", "﹁": '"', "﹂": '"', "．": "."})  # full-width period -> ASCII


def run(paths: BookPaths, cfg: BookConfig, force: bool = False):
    src = paths.source
    inputs = cache.content_hash("ingest", cache.file_hash(src), cfg.language, cfg.chapter_regex)
    if not force and cache.is_fresh(paths.chapters, inputs):
        return paths.chapters
    text = src.read_text(encoding="utf-8")
    chapters = parse(text, cfg.language, cfg.chapter_regex, is_markdown=src.suffix == ".md")
    paths.work.mkdir(parents=True, exist_ok=True)
    paths.chapters.write_text(
        ChapterList(language=cfg.language, chapters=chapters).model_dump_json(indent=1),
        encoding="utf-8",
    )
    cache.mark_fresh(paths.chapters, inputs)
    return paths.chapters


def parse(text: str, lang: str, chapter_regex: str | None, is_markdown: bool) -> list[Chapter]:
    lines = _strip_gutenberg(text.replace("\r\n", "\n").split("\n"))
    heading_re = re.compile(chapter_regex, re.MULTILINE) if chapter_regex else _CHAPTER_RE[lang]

    sections: list[tuple[str, list[str]]] = []
    title, buf = "", []
    for line in lines:
        head = _heading(line, heading_re, is_markdown)
        if head is not None:
            if buf or title:
                sections.append((title, buf))
            title, buf = head, []
        else:
            buf.append(line)
    sections.append((title, buf))

    # Drop an empty pre-heading section (front matter before chapter 1).
    sections = [(t, b) for t, b in sections if t or any(l.strip() for l in b)]
    if not sections:
        return []

    chapters = []
    for i, (t, b) in enumerate(sections):
        paras = _paragraphs(b, lang)
        if not paras and not t:
            continue
        chapters.append(Chapter(index=len(chapters), title=t or f"Chapter {i + 1}", paragraphs=paras))
    return chapters


def _heading(line: str, heading_re: re.Pattern, is_markdown: bool) -> str | None:
    if is_markdown:
        m = _MD_HEADING_RE.match(line)
        if m:
            return m.group(2)
        return None
    s = line.strip()
    if not s:
        return None
    if heading_re.match(s) or _BARE_NUMERAL_RE.match(s):
        return s
    return None


def _paragraphs(lines: list[str], lang: str) -> list[str]:
    """Blank-line separated blocks; hard-wrapped lines inside a block are unwrapped."""
    paras: list[str] = []
    buf: list[str] = []
    joiner = "" if lang == "zh" else " "

    def flush():
        if buf:
            p = joiner.join(s.strip() for s in buf)
            p = re.sub(r"[ \t]+", " ", p).strip()
            if p:
                paras.append(canonicalize_quotes(p).translate(_ZH_PUNCT))
            buf.clear()

    for line in lines:
        if line.strip():
            buf.append(line)
        else:
            flush()
    flush()
    return paras


def _strip_gutenberg(lines: list[str]) -> list[str]:
    start = next((i + 1 for i, l in enumerate(lines) if _GUTENBERG_START.match(l)), 0)
    end = next((i for i, l in enumerate(lines) if _GUTENBERG_END.match(l)), len(lines))
    return lines[start:end]
