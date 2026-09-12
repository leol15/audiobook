"""Stage 5: lines.jsonl -> work/audio/<hash>.wav, one file per line."""

from __future__ import annotations

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
    synthesized = 0
    for done, ln in enumerate(track(missing, description=f"render[{backend.name}]"), 1):
        voice = resolve_voice(voices, ln.speaker)
        if (paths.root / voice).is_file():
            voice = str((paths.root / voice).resolve())  # reference clip, not a preset id
        params = {**cfg.tts.params, "seed": ln.attempts}
        key = cache.render_key(backend.name, voice, params, ln.text)
        out = paths.audio / f"{key}.wav"
        if force or not out.exists():
            synthesized += 1
            pieces = [backend.synthesize(chunk, voice, lang=ln.lang, **params)
                      for chunk in chunk_text(ln.text, ln.lang, backend.max_chars)]
            audio = crossfade_concat(pieces, backend.sample_rate) if pieces else np.zeros(0, np.float32)
            sf.write(out, audio, backend.sample_rate)
        ln.audio = str(out.relative_to(paths.work))
        ln.backend = backend.name
        ln.error_rate = None  # verify must look at the new audio
        if done % CHECKPOINT_EVERY == 0:
            write_lines(paths, lines)  # progress survives an interrupted run; `ab status` sees it
    write_lines(paths, lines)
    _link_by_line(paths, lines)
    note(paths, f"render[{backend.name}]: {synthesized} lines synthesized, "
         f"{len(missing) - synthesized} from cache, {len(lines) - len(missing)} untouched")
    return paths.audio


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
