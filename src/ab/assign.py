"""`ab voices-assign`: pick a built-in voice per main cast member from the
cast.yaml descriptions (gender, age), for backends with a voice pool
(Kokoro, Qwen3-TTS presets). The counterpart of `ab voices-design` for
backends that cannot design voices.

Rules: existing assignments in book.yaml are kept unless --force; the narrator
gets a narrator-suitable voice; each main character gets the best-matching
unused voice (gender match first, then age), and a voice is reused only when
the pool for that gender is exhausted, starting with the least-used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rich import print

from ab.config import BookConfig, BookPaths
from ab.voicepool import Voice, pool
from ab.voices import _write_voice_map

# Explicit gender words win; relational words (wife, his, 他) only break ties,
# since descriptions mention other characters ("Mr. Bennet's daughter").
_FEMALE_EXPLICIT = re.compile(r"\b(female|woman|girl|lady)\b|女性|女孩|女人|少女|姑娘", re.IGNORECASE)
_MALE_EXPLICIT = re.compile(r"\b(male|man|boy|gentleman)\b|男性|男孩|男人|少年", re.IGNORECASE)
_FEMALE = re.compile(r"\b(she|her|mother|wife|daughter|sister|aunt|queen|princess|mrs|miss|ms)\b|"
                     r"她|母亲|妻|姐|妹|夫人|小姐|婆|妈", re.IGNORECASE)
_MALE = re.compile(r"\b(he|his|him|father|husband|son|brother|uncle|king|prince|lord|mr|sir)\b|"
                   r"他|父亲|丈夫|兄|弟|叔|爷|老头|先生|哥", re.IGNORECASE)
_MIDDLE = re.compile(r"\b(middle[- ]aged|adult|mature|grown)\b|中年|成年|大叔|大婶", re.IGNORECASE)
_OLD = re.compile(r"\b(old|elderly|senior|grandfather|grandmother|grandpa|grandma|veteran)\b|"
                  r"老年|老人|年长|年迈|爷爷|奶奶|老头|老太|[五六七八九]十岁|[5-9]\d岁", re.IGNORECASE)
_YOUNG = re.compile(r"\b(young|youth|teen|teenage|child|kid|boy|girl|student|1?\d\s*(?:years?|yrs?))\b|"
                    r"年轻|少年|少女|孩子|小孩|学生|青年|十[一二三四五六七八九]?岁|[一二三四五六七八九]岁|1?\d岁|2\d岁", re.IGNORECASE)


@dataclass
class Profile:
    gender: str  # "f" | "m" | "?"
    age: str     # "young" | "adult" | "old" | "?"


def profile(description: str) -> Profile:
    d = description or ""
    f, m = bool(_FEMALE_EXPLICIT.search(d)), bool(_MALE_EXPLICIT.search(d))
    if f == m:  # neither or both explicit: fall back to relational words
        f, m = bool(_FEMALE.search(d)), bool(_MALE.search(d))
    gender = "f" if f and not m else "m" if m and not f else "?"
    if _MIDDLE.search(d):
        age = "adult"
    elif _OLD.search(d):
        age = "old"
    elif _YOUNG.search(d):
        age = "young"
    else:
        age = "?"
    return Profile(gender, age)


def _score(v: Voice, p: Profile, used: dict[str, int], narrator_id: str | None) -> tuple:
    gender = 2 if p.gender == "?" or v.gender == p.gender else 0
    age = 1 if p.age == "?" or v.age == p.age else 0
    not_narrator = 0 if v.id == narrator_id else 1
    return (gender, -used.get(v.id, 0), age, not_narrator)


def assign(cast_items: list[tuple[str, str]], voices: list[Voice], existing: dict[str, str],
           lang: str) -> dict[str, str]:
    """cast_items: (name, description) in priority order. Returns role -> voice id."""
    out = dict(existing)
    used: dict[str, int] = {}
    for vid in out.values():
        used[vid] = used.get(vid, 0) + 1
    if "narrator" not in out:
        narr = [v for v in voices if v.narrator] or voices
        out["narrator"] = narr[0].id
        used[out["narrator"]] = used.get(out["narrator"], 0) + 1
    for name, desc in cast_items:
        if name in out:
            continue
        p = profile(desc)
        best = max(voices, key=lambda v: _score(v, p, used, out["narrator"]))
        out[name] = best.id
        used[best.id] = used.get(best.id, 0) + 1
    out.setdefault("_default", out["narrator"])
    return out


def run(paths: BookPaths, cfg: BookConfig, backend: str | None = None, force: bool = False) -> dict[str, str]:
    backend = backend or cfg.tts.backend
    voices = pool(backend, cfg.language)
    if not voices:
        raise SystemExit(f"backend {backend!r} has no built-in voice pool for {cfg.language}; "
                         f"use `ab voices-design` (qwen3tts) or reference clips")
    cast = paths.load_cast()
    if not cast.characters:
        raise SystemExit("no cast.yaml: run `ab cast` first")
    items = [(n, c.description) for n, c in sorted(cast.characters.items(), key=lambda kv: -kv[1].lines)
             if c.main]
    existing = {} if force else _existing(cfg, backend)
    result = assign(items, voices, existing, cfg.language)
    _write_voice_map(paths, backend, result)
    for role, vid in result.items():
        tag = "" if role in existing else "  [green]new[/]"
        p = profile(dict(items).get(role, "")) if role not in ("narrator", "_default") else None
        hint = f"  [dim]({p.gender}/{p.age})[/]" if p else ""
        print(f"  {role}: {vid}{hint}{tag}")
    print(f"voices-assign: {len(result)} roles -> book.yaml voices.{backend}")
    return result


def _existing(cfg: BookConfig, backend: str) -> dict[str, str]:
    try:
        return dict(cfg.voice_map(backend))
    except SystemExit:
        return {}
