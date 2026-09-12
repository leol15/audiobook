"""Stage 5: lines.jsonl -> work/audio/<hash>.wav, one file per line."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.progress import track

from ab import cache
from ab.audio import crossfade_concat
from ab.config import BookConfig, BookPaths
from ab.log import note
from ab.stages.s04_attribute import read_lines, write_lines
from ab.text import chunk_text
from ab.tts import load_backend

CHECKPOINT_EVERY = 10  # lines between lines.jsonl writes during a long render


def run(paths: BookPaths, cfg: BookConfig, force: bool = False):
    backend = load_backend(cfg.tts.backend, cfg.tts.params)
    try:
        return _run(paths, cfg, backend, force)
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()


def _run(paths: BookPaths, cfg: BookConfig, backend, force: bool):
    if cfg.language not in backend.languages:
        raise SystemExit(f"backend {backend.name} does not support language {cfg.language!r}")
    lines = read_lines(paths)
    paths.audio.mkdir(parents=True, exist_ok=True)
    missing = [l for l in lines
               if force or l.backend != backend.name
               or not (l.audio and (paths.work / l.audio).exists())]
    voices = cfg.voice_map(cfg.tts.backend)
    batch_size = cfg.tts.batch or getattr(backend, "batch_size", 1)
    if batch_size > 1 and not hasattr(backend, "synthesize_batch"):
        batch_size = 1

    # Plan: every missing line gets its cache path; lines without cached audio
    # become (line, chunk) work items grouped by voice+params, so one model
    # call renders many lines in the same voice.
    jobs: list[_Job] = []
    for ln in missing:
        voice = resolve_voice(voices, ln.speaker)
        if (paths.root / voice).is_file():
            voice = str((paths.root / voice).resolve())  # reference clip, not a preset id
        params = {**cfg.tts.params, "seed": ln.attempts}
        key = cache.render_key(backend.name, voice, params, ln.text)
        out = paths.audio / f"{key}.wav"
        chunks = chunk_text(ln.text, ln.lang, backend.max_chars) if force or not out.exists() else None
        jobs.append(_Job(ln, voice, params, out, chunks))

    todo = [j for j in jobs if j.chunks is not None]
    groups: dict[tuple, list[_Job]] = {}
    for j in todo:
        groups.setdefault((j.voice, json.dumps(j.params, sort_keys=True)), []).append(j)
    batches = list(_batches(groups, batch_size))

    finished = 0
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
                finished += 1
                if finished % CHECKPOINT_EVERY == 0:
                    write_lines(paths, lines)  # progress survives an interrupted run
    for j in jobs:
        if j.chunks is None:
            _mark(paths, j, backend.name)
    write_lines(paths, lines)
    _link_by_line(paths, lines)
    note(paths, f"render[{backend.name}]: {len(todo)} lines synthesized in {len(batches)} calls "
         f"(batch {batch_size}), {len(missing) - len(todo)} from cache, "
         f"{len(lines) - len(missing)} untouched")
    return paths.audio


@dataclass(eq=False)
class _Job:
    line: object
    voice: str
    params: dict
    out: Path
    chunks: list[str] | None  # None = already cached
    pieces: list[np.ndarray] = field(default_factory=list)


def _batches(groups: dict[tuple, list[_Job]], batch_size: int):
    """Yield lists of (job, chunk text) no longer than batch_size, one voice each.

    Within a voice, chunks are ordered by length so a batch finishes together:
    batched generation runs until its longest sequence stops. Multi-chunk
    lines keep their chunks in order; a line's chunks may span batches.
    """
    for jobs in groups.values():
        items = [(j, i, c) for j in jobs for i, c in enumerate(j.chunks)]
        items.sort(key=lambda it: (len(it[2]), it[0].line.id, it[1]))
        for k in range(0, len(items), batch_size):
            yield items[k:k + batch_size]


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


def _link_by_line(paths: BookPaths, lines) -> None:
    """05-audio/by-line/<line id>.wav -> ../<hash>.wav, so a line is easy to find and play."""
    d = paths.audio_by_line
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.wav"):
        old.unlink()
    for ln in lines:
        if ln.audio:
            (d / f"{ln.id}.wav").symlink_to(Path("..") / Path(ln.audio).name)


def resolve_voice(voices: dict[str, str], speaker: str) -> str:
    for key in (speaker, "_default", "narrator"):
        if key in voices:
            return voices[key]
    raise SystemExit(f"no voice configured for {speaker!r}; set voices.narrator in book.yaml")
