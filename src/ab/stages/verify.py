"""Stage 6: transcribe rendered audio and re-render lines whose text drifted.

Needs the `verify` extra (faster-whisper). Off by default in `ab run`; enable
with --skip "" or run `ab verify` directly. Meant for autoregressive backends;
Kokoro rarely needs it.
"""

from __future__ import annotations

import json
import re
import unicodedata

import soundfile as sf
from rich import print
from rich.progress import track

from ab.config import BookConfig, BookPaths
from ab.models import VerifyResult
from ab.stages import render
from ab.stages.attribute import read_lines, write_lines

_WHISPER_DEFAULT = {"en": "small.en", "zh": "small"}


def run(paths: BookPaths, cfg: BookConfig, force: bool = False):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit("verify needs faster-whisper: uv sync --extra verify") from e

    model_name = cfg.verify.whisper_model or _WHISPER_DEFAULT[cfg.language]
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    results: dict[str, VerifyResult] = {}

    for attempt in range(cfg.verify.max_attempts):
        lines = read_lines(paths)
        todo = [ln for ln in lines if ln.audio and (force or ln.error_rate is None)]
        if not todo:
            break
        failed = 0
        for ln in track(todo, description=f"verify[{model_name}] pass {attempt + 1}"):
            res = _check(model, paths, ln, cfg)
            results[ln.id] = res
            ln.error_rate = res.error_rate
            if not res.ok and ln.attempts + 1 < cfg.verify.max_attempts:
                # Drop the audio; render will regenerate with a new seed.
                ln.audio, ln.error_rate, ln.attempts = None, None, ln.attempts + 1
                failed += 1
        write_lines(paths, lines)
        force = False
        if failed:
            print(f"verify: re-rendering {failed} lines (attempt {attempt + 2})")
            render.run(paths, cfg)
        else:
            break

    with paths.verify.open("w", encoding="utf-8") as f:
        for r in results.values():
            f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")
    lines = read_lines(paths)
    bad = [ln for ln in lines if ln.id in results and not results[ln.id].ok]
    review = paths.work / "verify.review.txt"
    with review.open("w", encoding="utf-8") as f:
        f.write("# Lines still above the error threshold after re-rendering. Listen and fix by hand.\n")
        for ln in bad:
            f.write(f"{ln.id}\t{ln.error_rate:.2f}\t{ln.text[:100]}\n")
    rates = [ln.error_rate for ln in lines if ln.error_rate is not None]
    mean = sum(rates) / len(rates) if rates else 0.0
    print(f"verify: {len(rates)} lines checked, mean error {mean:.3f}, {len(bad)} above threshold")
    return paths.verify


def _check(model, paths: BookPaths, ln, cfg: BookConfig) -> VerifyResult:
    path = paths.work / ln.audio
    segments, _ = model.transcribe(str(path), language=cfg.language, beam_size=1,
                                   condition_on_previous_text=False)
    transcript = "".join(s.text for s in segments) if cfg.language == "zh" else \
        " ".join(s.text.strip() for s in segments)
    rate, edits = error_rate(ln.text, transcript, cfg.language, with_edits=True)
    # A single misheard word on a 3-word line is a 33% "error"; tolerate one
    # edit on short lines so whisper's own mistakes do not trigger re-renders.
    ok = rate <= cfg.verify.threshold or edits <= 1
    return VerifyResult(id=ln.id, transcript=transcript, error_rate=rate, edits=edits,
                        duration=sf.info(path).duration, ok=ok)


def error_rate(reference: str, hypothesis: str, lang: str, with_edits: bool = False):
    """Word error rate (en) or character error rate (zh), on normalized text."""
    ref, hyp = _tokens(reference, lang), _tokens(hypothesis, lang)
    if not ref:
        edits = len(hyp)
        rate = 0.0 if not hyp else 1.0
    else:
        edits = _levenshtein(ref, hyp)
        rate = edits / len(ref)
    return (rate, edits) if with_edits else rate


def _tokens(s: str, lang: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    if lang == "zh":
        return [c for c in s if not c.isspace()]
    return s.split()


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
