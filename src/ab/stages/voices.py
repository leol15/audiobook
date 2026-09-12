"""`ab voices-design`: one reference clip per role from text descriptions.

Uses the Qwen3-TTS VoiceDesign model once per role, saves
voices/<role>.wav plus voices/<role>.txt (the spoken text, used as ref_text
when cloning), and writes the resulting map into book.yaml under
voices.qwen3tts. Rendering then clones each clip with the Base model, so a
character sounds the same across the whole book.
"""

from __future__ import annotations

import re

import yaml
from rich import print
from rich.progress import track

from ab.config import BookConfig, BookPaths
from ab.stages.attribute import read_lines
from ab.tts import load_backend

_NARRATOR = {
    "en": "Calm, warm middle-aged male narrator with clear diction and a steady, unhurried pace.",
    "zh": "沉稳温和的中年男性旁白，吐字清晰，语速平稳从容。",
}
# Fallback sample text when a role has no lines of its own.
_SAMPLE = {
    "en": "The evening settled quietly over the town, and one by one the windows began to glow.",
    "zh": "夜色悄悄笼罩了小镇，窗户里的灯光一盏一盏地亮了起来。",
}
_MAX_CHARS = {"en": 160, "zh": 60}


def run(paths: BookPaths, cfg: BookConfig, force: bool = False, backend_name: str = "qwen3tts"):
    cast = paths.load_cast()
    lang = cfg.language
    roles: dict[str, str] = {"narrator": cfg.narrator_description or _NARRATOR[lang]}
    for name, ch in cast.characters.items():
        roles[name] = ch.description or name
    if not roles:
        raise SystemExit("no roles: run `ab cast` first")

    samples = _sample_text(paths, lang)
    backend = load_backend(backend_name, cfg.tts.params if cfg.tts.backend == backend_name else {})
    if not hasattr(backend, "design"):
        raise SystemExit(f"backend {backend_name!r} has no voice design")
    paths.voices_dir.mkdir(exist_ok=True)
    voice_map: dict[str, str] = {}
    try:
        for role, description in track(list(roles.items()), description="voices-design"):
            slug = _slug(role)
            wav = paths.voices_dir / f"{slug}.wav"
            txt = wav.with_suffix(".txt")
            text = samples.get(role) or _SAMPLE[lang]
            if force or not wav.exists():
                backend.design(text, lang=lang, instruct=description, out=str(wav))
                txt.write_text(text, encoding="utf-8")
            voice_map[role] = f"voices/{slug}.wav"
    finally:
        backend.close()

    voice_map["_default"] = voice_map["narrator"]
    _write_voice_map(paths, backend_name, voice_map)
    print(f"voices-design: {len(voice_map) - 1} clips in {paths.voices_dir}; "
          f"book.yaml voices.{backend_name} updated")
    return paths.voices_dir


def _sample_text(paths: BookPaths, lang: str) -> dict[str, str]:
    """A short passage per role from the book itself, so the designed voice is
    heard saying words the character actually says."""
    if not paths.lines.exists():
        return {}
    out: dict[str, str] = {}
    joiner = "" if lang == "zh" else " "
    for ln in read_lines(paths):
        cur = out.get(ln.speaker, "")
        if len(cur) >= _MAX_CHARS[lang]:
            continue
        piece = ln.text.strip()
        if len(piece) < 4:
            continue
        cand = f"{cur}{joiner}{piece}" if cur else piece
        out[ln.speaker] = cand[: _MAX_CHARS[lang] * 2]
    return {k: v for k, v in out.items() if len(v) >= 12}


def _write_voice_map(paths: BookPaths, backend_name: str, voice_map: dict[str, str]) -> None:
    data = yaml.safe_load(paths.config.read_text(encoding="utf-8")) or {}
    voices = data.get("voices") or {}
    nested = {k: v for k, v in voices.items() if isinstance(v, dict)}
    if not nested and voices:
        # Flat map applied to whichever backend is current; keep it under that name.
        current = (data.get("tts") or {}).get("backend", "kokoro")
        voices = {current: voices}
    voices[backend_name] = voice_map
    data["voices"] = voices
    paths.config.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _slug(s: str) -> str:
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE).strip("_")
    return s or "role"
