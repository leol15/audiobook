"""Stage 5: 04-lines/cNNN.jsonl -> 05-audio/<hash>.wav (one per line) + 05-render/cNNN.jsonl.

Chapter-granular: a chapter's render file is stamped with the hash of its
attribute file, the backend, the voice map and the params, so a fix in one
chapter or a voice change re-plans only the chapters it touches. Planning is
still whole-book so batches fill by voice across chapters. The audio cache
is content-addressed (backend, voice, params incl. seed, text), so a
"re-render" of an unchanged line is a cache hit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.progress import track

from ab import cache
from ab.audio import crossfade_concat
from ab.config import BookConfig, BookPaths
from ab.lines import chapter_indexes, read_attribution, read_render, write_render, write_views
from ab.log import note
from ab.models import Line
from ab.text import chunk_text, sequence_cost
from ab.tts import load_backend

CHECKPOINT_SECONDS = 60  # render state is flushed at least this often during a long run


def run(paths: BookPaths, cfg: BookConfig, force: bool = False, chapters: set[int] | None = None):
    backend = load_backend(cfg.tts.backend, cfg.tts.params)
    try:
        return _run(paths, cfg, backend, force, chapters)
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()


def voice_fingerprint(paths: BookPaths, voices: dict[str, str]) -> dict[str, str]:
    """role -> voice id, with reference clips identified by content hash as well
    as path, so regenerating a clip re-renders the lines that used it."""
    out = {}
    for role, vid in voices.items():
        clip = paths.root / vid
        out[role] = f"{vid}#{cache.file_hash(clip)}" if clip.is_file() else vid
    return out


def chapter_inputs(paths: BookPaths, cfg: BookConfig, chapter: int, backend_name: str) -> dict:
    f = paths.chapter_lines(chapter)
    return {"lines": cache.file_hash(f) if f.exists() else "", "backend": backend_name,
            "voices": cache.content_hash(voice_fingerprint(paths, cfg.voice_map(cfg.tts.backend))),
            "params": cache.content_hash(cfg.tts.params)}


def chapter_states(paths: BookPaths, cfg: BookConfig, backend_name: str | None) -> list[tuple[int, list[str]]]:
    """(chapter, stale reasons) for every attributed chapter. A chapter whose
    stamp matches but has a line without audio (verify dropped it) is stale too."""
    out = []
    for i in chapter_indexes(paths):
        reasons = cache.stale_reasons(paths.chapter_render(i), chapter_inputs(paths, cfg, i, backend_name or ""))
        if not reasons:
            st = read_render(paths, i)
            missing = [ln.id for ln in read_attribution(paths, i)
                       if not (st.get(ln.id) and st[ln.id].audio and (paths.work / st[ln.id].audio).exists())]
            if missing:
                reasons = [f"{len(missing)} lines without audio"]
        out.append((i, reasons))
    return out


def _run(paths: BookPaths, cfg: BookConfig, backend, force: bool, chapters: set[int] | None):
    if cfg.language not in backend.languages:
        raise SystemExit(f"backend {backend.name} does not support language {cfg.language!r}")
    paths.audio.mkdir(parents=True, exist_ok=True)
    voices = cfg.voice_map(cfg.tts.backend)
    fingerprints = voice_fingerprint(paths, voices)
    batch_size = cfg.tts.batch or getattr(backend, "batch_size", 1)
    if batch_size > 1 and not hasattr(backend, "synthesize_batch"):
        batch_size = 1

    # Plan: every stale chapter's lines get their cache path; lines without
    # cached audio become (line, chunk) work items grouped by voice+params, so
    # one model call renders many lines in the same voice, across chapters.
    todo_chapters = [i for i, reasons in chapter_states(paths, cfg, backend.name)
                     if (chapters is None or i in chapters) and (force or reasons)]
    per_chapter: dict[int, list[Line]] = {}
    jobs: list[_Job] = []
    for ci in todo_chapters:
        lines = read_attribution(paths, ci)
        state = read_render(paths, ci)
        per_chapter[ci] = lines
        for ln in lines:
            st = state.get(ln.id)
            if st:
                ln.audio, ln.backend, ln.attempts = st.audio, st.backend, st.attempts
            voice = resolve_voice(voices, ln.speaker)
            voice_key = resolve_voice(fingerprints, ln.speaker)  # cache identity (clip hash aware)
            if (paths.root / voice).is_file():
                voice = str((paths.root / voice).resolve())  # reference clip, not a preset id
            out = _expected(paths, backend.name, voice_key, cfg.tts.params, ln)
            if ln.audio is not None and ln.audio != str(out.relative_to(paths.work)):
                ln.attempts = 0  # text, voice, or backend changed: start the seed sequence over
                out = _expected(paths, backend.name, voice_key, cfg.tts.params, ln)
            params = {**cfg.tts.params, "seed": ln.attempts}
            if not force and ln.audio == str(out.relative_to(paths.work)) and out.exists():
                jobs.append(_Job(ln, ci, voice, params, out, None))
                continue
            chunks = chunk_text(ln.text, ln.lang, backend.max_chars) if force or not out.exists() else None
            jobs.append(_Job(ln, ci, voice, params, out, chunks))

    todo = [j for j in jobs if j.chunks is not None]
    groups: dict[tuple, list[_Job]] = {}
    for j in todo:
        groups.setdefault((j.voice, json.dumps(j.params, sort_keys=True)), []).append(j)
    batches = list(_batches(groups, batch_size, cfg.tts.batch_tokens if batch_size > 1 else 0))
    for j in jobs:
        if j.chunks is None:
            _mark(paths, j, backend.name)

    dirty: set[int] = set(todo_chapters)
    last_flush = time.time()
    progress = track(batches, description=f"render[{backend.name}]", total=len(batches))
    for batch in progress:
        pieces = _synthesize(backend, batch, batch_size)
        for job, chunks in zip(dict.fromkeys(j for j, _, _ in batch), pieces):
            job.pieces.extend(chunks)
            if len(job.pieces) == len(job.chunks):
                audio = crossfade_concat(job.pieces, backend.sample_rate) if job.pieces \
                    else np.zeros(0, np.float32)
                sf.write(job.out, audio, backend.sample_rate)
                _mark(paths, job, backend.name)
                dirty.add(job.chapter)
        if time.time() - last_flush >= CHECKPOINT_SECONDS:
            _flush(paths, per_chapter, dirty)  # progress survives an interrupted run
            last_flush = time.time()
    _flush(paths, per_chapter, dirty)
    for ci, lines in per_chapter.items():
        if all(ln.audio for ln in lines):
            cache.mark_fresh(paths.chapter_render(ci), chapter_inputs(paths, cfg, ci, backend.name))
    write_views(paths)
    _link_by_line(paths)
    n_lines = sum(len(v) for v in per_chapter.values())
    note(paths, f"render[{backend.name}]: {len(todo_chapters)} chapters planned ({n_lines} lines): "
         f"{len(todo)} synthesized in {len(batches)} calls (batch {batch_size}), "
         f"{n_lines - len(todo)} from cache")
    return paths.render_dir


def _expected(paths: BookPaths, backend_name: str, voice: str, params: dict, ln: Line) -> Path:
    key = cache.render_key(backend_name, voice, {**params, "seed": ln.attempts}, ln.text)
    return paths.audio / f"{key}.wav"


def _flush(paths: BookPaths, per_chapter: dict[int, list[Line]], dirty: set[int]) -> None:
    for ci in sorted(dirty):
        write_render(paths, ci, per_chapter[ci])
    dirty.clear()


@dataclass(eq=False)
class _Job:
    line: Line
    chapter: int
    voice: str
    params: dict
    out: Path
    chunks: list[str] | None  # None = already cached
    pieces: list[np.ndarray] = field(default_factory=list)


def _batches(groups: dict[tuple, list[_Job]], batch_size: int, batch_tokens: int = 0):
    """Yield lists of (job, chunk index, chunk text), one voice each, cut at
    batch_size lines or batch_tokens expected audio tokens, whichever first.

    Within a voice, chunks are ordered by length so a batch finishes together:
    batched generation runs until its longest sequence stops. Multi-chunk
    lines keep their chunks in order; a line's chunks may span batches.
    The token budget is what bounds GPU memory: it scales with lines x
    length, so a line count alone spills on batches of long lines.
    """
    for jobs in groups.values():
        items = [(j, i, c) for j in jobs for i, c in enumerate(j.chunks)]
        items.sort(key=lambda it: (len(it[2]), it[0].line.id, it[1]))
        batch: list = []
        tokens = 0
        for it in items:
            t = sequence_cost(it[2], it[0].line.lang)
            if batch and (len(batch) >= batch_size or (batch_tokens and tokens + t > batch_tokens)):
                yield batch
                batch, tokens = [], 0
            batch.append(it)
            tokens += t
        if batch:
            yield batch


def _synthesize(backend, batch, batch_size: int) -> list[list[np.ndarray]]:
    """Render one batch; returns, per job in the batch, its new pieces in chunk order."""
    first = batch[0][0]
    voice, lang, params = first.voice, first.line.lang, first.params
    texts = [c for _, _, c in batch]
    if batch_size > 1 and len(texts) > 1:
        audios = backend.synthesize_batch(texts, voice, lang=lang, **params)
    else:
        audios = [backend.synthesize(c, voice, lang=lang, **params) for c in texts]
    got: dict[int, list] = {}
    order: list[_Job] = []
    for (j, i, _), a in zip(batch, audios):
        if id(j) not in got:
            order.append(j)
        got.setdefault(id(j), []).append((i, a))
    return [[a for _, a in sorted(got[id(j)], key=lambda t: t[0])] for j in order]


def _mark(paths: BookPaths, job: _Job, backend_name: str) -> None:
    ln = job.line
    ln.audio = str(job.out.relative_to(paths.work))
    ln.backend = backend_name
    ln.error_rate = None  # verify must look at the new audio


def _link_by_line(paths: BookPaths) -> None:
    """05-audio/by-line/<line id>.wav -> ../<hash>.wav, so a line is easy to find and play."""
    d = paths.audio_by_line
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.wav"):
        old.unlink()
    for ci in chapter_indexes(paths):
        for st in read_render(paths, ci).values():
            if st.audio:
                (d / f"{st.id}.wav").symlink_to(Path("..") / Path(st.audio).name)


def resolve_voice(voices: dict[str, str], speaker: str) -> str:
    for key in (speaker, "_default", "narrator"):
        if key in voices:
            return voices[key]
    raise SystemExit(f"no voice configured for {speaker!r}; set voices.narrator in book.yaml")
