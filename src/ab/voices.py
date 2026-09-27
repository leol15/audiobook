"""`ab voices-design`: reference clips per role (and per style) from text descriptions.

Uses the Qwen3-TTS VoiceDesign model, saves voices/<role>.wav plus
voices/<role>.txt (the spoken text, used as ref_text when cloning), and
writes the resulting map into book.yaml under voices.qwen3tts. Rendering
then clones each clip with the Base model, so a character sounds the same
across the whole book.

With book.yaml `styles`, each main character also gets one clip per style
(voices/<role>.<style>.wav, map key `<role>@<style>`): the same description
with the style phrase appended, so it is the same voice in a different mood
(measured speaker-embedding cosine 0.96-0.99 against the neutral clip,
where two different designed voices score 0.89-0.99). Render picks the
style clip for lines the attribute LLM labelled with that style. Clips are
designed `batch` at a time in one model call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml
from rich import print
from rich.progress import track

from ab.config import BookConfig, BookPaths
from ab.lines import read_lines
from ab.styles import phrase, voice_key
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


def run(paths: BookPaths, cfg: BookConfig, force: bool = False, backend_name: str = "qwen3tts",
        styles: bool = True, batch: int = 8):
    cast = paths.load_cast()
    lang = cfg.language
    roles: dict[str, str] = {"narrator": cfg.narrator_description or _NARRATOR[lang]}
    for name, ch in cast.characters.items():
        if ch.main:  # minor characters use _default, so they need no clip
            roles[name] = ch.description or name
    if not roles:
        raise SystemExit("no roles: run `ab cast` first")

    samples = _sample_text(paths, lang)
    jobs = plan(roles, samples, lang, cfg.styles if styles else [])
    voice_map = {j.key: j.rel for j in jobs}
    todo = [j for j in jobs if force or not (paths.root / j.rel).exists()]

    backend = load_backend(backend_name, cfg.tts.params if cfg.tts.backend == backend_name else {})
    if not hasattr(backend, "design"):
        raise SystemExit(f"backend {backend_name!r} has no voice design")
    paths.voices_dir.mkdir(exist_ok=True)
    try:
        batches = [todo[i:i + max(1, batch)] for i in range(0, len(todo), max(1, batch))]
        for group in track(batches, description="voices-design"):
            items = [{"text": j.text, "instruct": j.instruct, "out": str(paths.root / j.rel)} for j in group]
            if len(items) > 1 and hasattr(backend, "design_batch"):
                backend.design_batch(items, lang=lang)
            else:
                for it in items:
                    backend.design(it["text"], lang=lang, instruct=it["instruct"], out=it["out"])
            for j in group:
                (paths.root / j.rel).with_suffix(".txt").write_text(j.text, encoding="utf-8")
    finally:
        backend.close()

    voice_map["_default"] = voice_map["narrator"]
    _write_voice_map(paths, backend_name, voice_map)
    print(f"voices-design: {len(todo)} clips designed in {len(batches)} calls, "
          f"{len(voice_map) - 1} in {paths.voices_dir}; book.yaml voices.{backend_name} updated")
    return paths.voices_dir


@dataclass
class DesignJob:
    key: str        # book.yaml voice-map key: role or role@style
    rel: str        # clip path relative to the book dir
    text: str       # what the clip says (also its .txt transcript)
    instruct: str   # VoiceDesign instruction


def plan(roles: dict[str, str], samples: dict[str, str], lang: str, styles: list[str]) -> list[DesignJob]:
    """One neutral clip per role, plus one per style for every role but the
    narrator, who reads narration and stays neutral."""
    jobs: list[DesignJob] = []
    for role, description in roles.items():
        slug = _slug(role)
        text = samples.get(role) or _SAMPLE[lang]
        jobs.append(DesignJob(role, f"voices/{slug}.wav", text, description))
        if role == "narrator":
            continue
        for st in styles:
            jobs.append(DesignJob(voice_key(role, st), f"voices/{slug}.{st}.wav", text,
                                  f"{description}{'，' if lang == 'zh' else ', '}{phrase(st, lang)}"))
    return jobs


def _sample_text(paths: BookPaths, lang: str) -> dict[str, str]:
    """A short passage per role from the book itself, so the designed voice is
    heard saying words the character actually says. Kept to whole lines and
    under _MAX_CHARS: the Base model clones best from 5-15 s of reference
    audio, and a clip cut mid-sentence (or 30 s long) makes the clone loop."""
    out: dict[str, str] = {}
    joiner = "" if lang == "zh" else " "
    limit = _MAX_CHARS[lang]
    for ln in read_lines(paths):
        cur = out.get(ln.speaker, "")
        piece = ln.text.strip()
        if len(piece) < 4 or len(piece) > limit:
            continue
        if cur and len(cur) + len(joiner) + len(piece) > limit:
            continue
        out[ln.speaker] = f"{cur}{joiner}{piece}" if cur else piece
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
