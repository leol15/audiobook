"""`ab new`: scaffold a book directory with a fully commented book.yaml, then
run ingest so the chapter count is visible before anything else is done."""

from __future__ import annotations

import shutil
from pathlib import Path

from rich import print

from ab.config import BookPaths, Language

_NARRATOR = {  # sensible narrator defaults per language and backend
    ("kokoro", "en"): "bm_george", ("kokoro", "zh"): "zm_yunyang",
    ("qwen3tts", "en"): "Ryan", ("qwen3tts", "zh"): "Uncle_Fu",
}

TEMPLATE = """\
# book.yaml — one book. Every key is listed; keys with defaults may be deleted.
# Docs: README.md ("Inspecting a book"), DESIGN.md. Validate with `ab check`.

title: {title}
author: {author}
language: {language}          # en | zh
# cover: cover.jpg           # optional, relative to this directory
{chapter_regex_line}

tts:
  backend: {backend}          # kokoro (fast, built-in voices) | qwen3tts (best Mandarin, clips/presets) | chatterbox
  params: {{}}                # backend-specific, part of the cache key (kokoro: speed; qwen3tts: size, attn, instruct)
  # batch: 32                 # lines per model call for batching backends (qwen3tts)

# Voice per role, one map per backend. Roles: narrator, _default, and cast names
# exactly as they appear in cast.yaml (run `ab cast` first, then `ab voices-assign`
# to fill these from the cast descriptions, or `ab voices-design` for qwen3tts clips).
voices:
  kokoro:
    narrator: {kokoro_narrator}
    _default: {kokoro_narrator}
  qwen3tts:
    narrator: {qwen_narrator}   # preset name, or voices/<role>.wav from `ab voices-design`
    _default: {qwen_narrator}

# narrator_description: "..."  # used by `ab voices-design` for the narrator clip

llm:
  model: null                 # Ollama model; null = OLLAMA_MODEL env or qwen3:14b
  num_ctx: 16384              # context requested from Ollama; windows are sized to it

cast:
  main_cap: 20                # characters ranked by line count that get their own voice
  merge_chapters: 5           # chapters per alias-merge LLM call

pauses:                       # seconds
  sentence: 0.35
  paragraph: 0.6
  speaker_change: 1.2
  chapter: 2.5

verify:
  threshold: 0.15             # max error rate (WER en, pinyin CER zh) before re-render
  max_attempts: 3
  whisper_model: null         # null = small.en / small by language
  batch: 16

build:
  chapter_format: mp3         # per-chapter files as chapters finish: mp3 | m4b | none
loudness_lufs: -18.0
"""


def create(dest: Path, *, source: Path, language: Language, title: str | None = None,
           author: str = "", chapter_regex: str | None = None, backend: str = "kokoro") -> BookPaths:
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} exists and is not empty")
    if not source.is_file():
        raise SystemExit(f"source not found: {source}")
    dest.mkdir(parents=True, exist_ok=True)
    suffix = ".md" if source.suffix.lower() in (".md", ".markdown") else ".txt"
    shutil.copy(source, dest / f"source{suffix}")
    title = title or dest.name.replace("-", " ").replace("_", " ").title()
    regex_line = (f"chapter_regex: '{chapter_regex}'" if chapter_regex
                  else "# chapter_regex: '^Chapter \\d+'   # override when ingest finds the wrong chapters")
    text = TEMPLATE.format(
        title=_yaml_str(title), author=_yaml_str(author), language=language,
        chapter_regex_line=regex_line, backend=backend,
        kokoro_narrator=_NARRATOR[("kokoro", language)],
        qwen_narrator=_NARRATOR[("qwen3tts", language)],
    )
    paths = BookPaths(dest)
    paths.config.write_text(text, encoding="utf-8")
    paths.load_config()  # fail early if the template does not validate
    return paths


def ingest_preview(paths: BookPaths) -> None:
    """Run ingest and print what it found, with a hint when chaptering looks wrong."""
    from ab.models import ChapterList
    from ab.stages import s01_ingest as ingest

    cfg = paths.load_config()
    ingest.run(paths, cfg)
    book = ChapterList.model_validate_json(paths.chapters.read_text(encoding="utf-8"))
    n = len(book.chapters)
    chars = sum(len(p) for c in book.chapters for p in c.paragraphs)
    print(f"[green]new[/] {paths.root}: {n} chapter(s), {chars:,} characters")
    for c in book.chapters[:5]:
        print(f"  [dim]{c.index:3d}[/] {c.title[:60]}  ({len(c.paragraphs)} paragraphs)")
    if n > 5:
        print(f"  [dim]... {n - 5} more[/]")
    if n == 1 and chars > 20_000:
        print("[yellow]Only one chapter found in a long text. Check how the source marks chapters "
              "and set chapter_regex in book.yaml, then `ab ingest --force`.[/]")
    print("[dim]next: `ab cast` (needs Ollama), then `ab voices-assign`, then `ab run`[/]")


def _yaml_str(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
