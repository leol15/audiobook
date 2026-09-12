"""`ab check`: book.yaml voices <-> cast.yaml <-> backend consistency.

Returns (errors, warnings). Errors stop a run (rendering would fail or
silently use the wrong voice); warnings are printed and the run continues.
"""

from __future__ import annotations

from ab.config import BookConfig, BookPaths
from ab.voicepool import CLIP_BACKENDS, POOLS, known_ids, pool

_ROLES = {"narrator", "_default"}


def check(paths: BookPaths, cfg: BookConfig, backend: str | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    backend = backend or cfg.tts.backend
    cast = paths.load_cast()

    try:
        voices = cfg.voice_map(backend)
    except SystemExit as e:
        return [str(e)], []
    if "narrator" not in voices:
        errors.append(f"voices.{backend}: no narrator voice")

    # Voice keys that match no cast member (typos go to _default silently).
    for role in voices:
        if role in _ROLES:
            continue
        canon = cast.resolve(role)
        if canon is None:
            hint = ""
            if cast.characters:
                close = _closest(role, list(cast.characters))
                hint = f" (did you mean {close!r}?)" if close else ""
            errors.append(f"voices.{backend}.{role}: not in cast.yaml{hint}")
        elif canon != role:
            warnings.append(f"voices.{backend}.{role}: matched cast member {canon!r} by alias; "
                            f"use the canonical name")

    # Main cast members with no voice of their own.
    unvoiced = [n for n, c in cast.characters.items() if c.main and n not in voices]
    if unvoiced:
        warnings.append(f"{len(unvoiced)} main cast member(s) use _default: {', '.join(unvoiced[:8])}"
                        + (" ..." if len(unvoiced) > 8 else "")
                        + "  (run `ab voices-assign` or `ab voices-design`)")

    # Voice ids: known preset, valid clip path, or in the backend's pool for this language.
    ids = known_ids(backend)
    lang_ids = {v.id for v in pool(backend, cfg.language)}
    for role, vid in voices.items():
        clip = paths.root / vid
        if backend in CLIP_BACKENDS and ("/" in vid or vid.endswith(".wav")):
            if not clip.is_file():
                errors.append(f"voices.{backend}.{role}: clip not found: {vid}")
            continue
        if backend in POOLS and vid not in ids:
            errors.append(f"voices.{backend}.{role}: unknown voice id {vid!r} (see `ab voices {backend}`)")
        elif backend in POOLS and vid in ids and vid not in lang_ids and vid in {v.id for v in POOLS[backend]}:
            errors.append(f"voices.{backend}.{role}: {vid!r} is not a {cfg.language} voice")

    # Duplicate voices within the main cast are legal but usually unintended.
    used: dict[str, list[str]] = {}
    for role, vid in voices.items():
        if role not in _ROLES:
            used.setdefault(vid, []).append(role)
    dups = {v: r for v, r in used.items() if len(r) > 1}
    for vid, roles in dups.items():
        warnings.append(f"voice {vid!r} shared by {', '.join(roles)}")
    return errors, warnings


def _closest(name: str, candidates: list[str]) -> str | None:
    import difflib

    m = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return m[0] if m else None
