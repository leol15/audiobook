"""Stage 5: lines.jsonl -> work/audio/<hash>.wav, one file per line."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from rich.progress import track

from ab import cache
from ab.audio import crossfade_concat
from ab.config import BookConfig, BookPaths
from ab.stages.attribute import read_lines, write_lines
from ab.text import chunk_text
from ab.tts import load_backend


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
    for ln in track(missing, description=f"render[{backend.name}]"):
        voice = resolve_voice(voices, ln.speaker)
        if (paths.root / voice).is_file():
            voice = str((paths.root / voice).resolve())  # reference clip, not a preset id
        params = {**cfg.tts.params, "seed": ln.attempts}
        key = cache.render_key(backend.name, voice, params, ln.text)
        out = paths.audio / f"{key}.wav"
        if force or not out.exists():
            pieces = [backend.synthesize(chunk, voice, lang=ln.lang, **params)
                      for chunk in chunk_text(ln.text, ln.lang, backend.max_chars)]
            audio = crossfade_concat(pieces, backend.sample_rate) if pieces else np.zeros(0, np.float32)
            sf.write(out, audio, backend.sample_rate)
        ln.audio = str(out.relative_to(paths.work))
        ln.backend = backend.name
        ln.error_rate = None  # verify must look at the new audio
    write_lines(paths, lines)
    return paths.audio


def resolve_voice(voices: dict[str, str], speaker: str) -> str:
    for key in (speaker, "_default", "narrator"):
        if key in voices:
            return voices[key]
    raise SystemExit(f"no voice configured for {speaker!r}; set voices.narrator in book.yaml")
