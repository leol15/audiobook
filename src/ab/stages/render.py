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
    if cfg.language not in backend.languages:
        raise SystemExit(f"backend {backend.name} does not support language {cfg.language!r}")
    lines = read_lines(paths)
    paths.audio.mkdir(parents=True, exist_ok=True)
    missing = [l for l in lines if force or not (l.audio and (paths.work / l.audio).exists())]
    for ln in track(missing, description=f"render[{backend.name}]"):
        voice = resolve_voice(cfg, ln.speaker)
        params = {**cfg.tts.params, "seed": ln.attempts}
        key = cache.render_key(backend.name, voice, params, ln.text)
        out = paths.audio / f"{key}.wav"
        if force or not out.exists():
            pieces = [backend.synthesize(chunk, voice, lang=ln.lang, **params)
                      for chunk in chunk_text(ln.text, ln.lang, backend.max_chars)]
            audio = crossfade_concat(pieces, backend.sample_rate) if pieces else np.zeros(0, np.float32)
            sf.write(out, audio, backend.sample_rate)
        ln.audio = str(out.relative_to(paths.work))
    write_lines(paths, lines)
    return paths.audio


def resolve_voice(cfg: BookConfig, speaker: str) -> str:
    if speaker in cfg.voices:
        return cfg.voices[speaker]
    if "_default" in cfg.voices:
        return cfg.voices["_default"]
    if "narrator" in cfg.voices:
        return cfg.voices["narrator"]
    raise SystemExit(f"no voice configured for {speaker!r}; set voices.narrator in book.yaml")
