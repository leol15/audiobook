"""Stage 6: transcribe rendered audio and re-render lines whose text drifted.

Needs the `verify` extra (faster-whisper). Off by default in `ab run`; enable
with --skip "" or run `ab verify` directly. Meant for autoregressive backends;
Kokoro rarely needs it.

Chapter-granular: 06-verify/cNNN.jsonl is stamped with the chapter's render
file, so only chapters whose audio changed are revisited, and within a chapter
only lines whose current audio has no result are transcribed.
"""

from __future__ import annotations

import re
import unicodedata

import soundfile as sf
from rich.progress import track

from ab import cache
from ab.config import BookConfig, BookPaths
from ab.lines import (
    chapter_indexes,
    read_attribution,
    read_render,
    read_verify,
    verify_failures,
    write_render,
    write_verify,
    write_views,
)
from ab.log import note
from ab.models import VerifyResult
from ab.stages import s05_render as render

_WHISPER_DEFAULT = {"en": "small.en", "zh": "small"}


def whisper_model_name(cfg: BookConfig) -> str:
    return cfg.verify.whisper_model or _WHISPER_DEFAULT[cfg.language]


def chapter_inputs(paths: BookPaths, cfg: BookConfig, chapter: int) -> dict:
    f = paths.chapter_render(chapter)
    return {"render": cache.file_hash(f) if f.exists() else "", "whisper": whisper_model_name(cfg),
            "threshold": cfg.verify.threshold}


def chapter_states(paths: BookPaths, cfg: BookConfig) -> list[tuple[int, list[str]]]:
    return [(i, cache.stale_reasons(paths.chapter_verify(i), chapter_inputs(paths, cfg, i)))
            for i in chapter_indexes(paths)]


def run(paths: BookPaths, cfg: BookConfig, force: bool = False, chapters: set[int] | None = None):
    model_name = whisper_model_name(cfg)
    model = None
    checked = 0
    wanted = [i for i in chapter_indexes(paths) if chapters is None or i in chapters]
    for attempt in range(cfg.verify.max_attempts):
        stale = [i for i, reasons in chapter_states(paths, cfg) if i in wanted and (force or reasons)]
        failed_chapters: set[int] = set()
        n_failed = 0
        for ci in stale:
            lines = read_attribution(paths, ci)
            state = read_render(paths, ci)
            results = {} if force else read_verify(paths, ci)
            todo = [ln for ln in lines if state.get(ln.id) and state[ln.id].audio
                    and (ln.id not in results or results[ln.id].audio != state[ln.id].audio)]
            if todo and model is None:
                model = _load(model_name)
            for ln in track(todo, description=f"verify[{model_name}] c{ci:03d} pass {attempt + 1}"):
                st = state[ln.id]
                res = _check(model, paths, ln.text, st.audio, cfg)
                res.id, res.audio = ln.id, st.audio
                results[ln.id] = res
                checked += 1
                if not res.ok and st.attempts + 1 < cfg.verify.max_attempts:
                    # Drop the audio; render will regenerate with a new seed.
                    st.audio, st.attempts = None, st.attempts + 1
                    failed_chapters.add(ci)
                    n_failed += 1
            write_verify(paths, ci, {k: v for k, v in results.items() if k in state})
            if ci in failed_chapters:
                for ln in lines:
                    if ln.id in state:
                        ln.audio, ln.backend, ln.attempts = state[ln.id].audio, state[ln.id].backend, state[ln.id].attempts
                write_render(paths, ci, lines)
                cache.stamp_path(paths.chapter_render(ci)).unlink(missing_ok=True)
            else:
                cache.mark_fresh(paths.chapter_verify(ci), chapter_inputs(paths, cfg, ci))
        force = False
        if not failed_chapters:
            break
        note(paths, f"verify: re-rendering {n_failed} lines in {len(failed_chapters)} chapters "
             f"(attempt {attempt + 2})")
        render.run(paths, cfg, chapters=failed_chapters)

    bad = verify_failures(paths)
    lines = {ln.id: ln for i in chapter_indexes(paths) for ln in read_attribution(paths, i)}
    with paths.verify_review.open("w", encoding="utf-8") as f:
        f.write("# Lines still above the error threshold after re-rendering. "
                "Listen (05-audio/by-line/<id>.wav) and fix by hand.\n")
        for lid, rate in bad.items():
            f.write(f"{lid}\t{rate:.2f}\t{lines[lid].text[:100] if lid in lines else ''}\n")
    rates = [vr.error_rate for i in chapter_indexes(paths) for vr in _current_results(paths, i).values()]
    mean = sum(rates) / len(rates) if rates else 0.0
    note(paths, f"verify[{model_name}]: {checked} lines transcribed this run, {len(rates)} with a "
         f"current result, mean error {mean:.3f}, {len(bad)} above threshold")
    write_views(paths)  # refresh the script with verify flags
    return paths.verify_dir


def _current_results(paths: BookPaths, chapter: int) -> dict[str, VerifyResult]:
    state = read_render(paths, chapter)
    return {k: v for k, v in read_verify(paths, chapter).items()
            if state.get(k) and state[k].audio == v.audio}


def _load(model_name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit("verify needs faster-whisper: uv sync --extra verify") from e
    return WhisperModel(model_name, device="cuda", compute_type="float16")


def _check(model, paths: BookPaths, text: str, audio: str, cfg: BookConfig) -> VerifyResult:
    path = paths.work / audio
    segments, _ = model.transcribe(str(path), language=cfg.language, beam_size=1,
                                   condition_on_previous_text=False)
    transcript = "".join(s.text for s in segments) if cfg.language == "zh" else \
        " ".join(s.text.strip() for s in segments)
    rate, edits = error_rate(text, transcript, cfg.language, with_edits=True)
    # A single misheard word on a 3-word line is a 33% "error"; tolerate one
    # edit on short lines so whisper's own mistakes do not trigger re-renders.
    ok = rate <= cfg.verify.threshold or edits <= 1
    return VerifyResult(id="", transcript=transcript, error_rate=rate, edits=edits,
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
    if lang == "zh":
        from ab.stages.s02_normalize import _zh_number

        # whisper writes ages and counts as digits; the source spells them out.
        # Do this before stripping punctuation so "15、6" stays two numbers.
        s = re.sub(r"\d+", lambda m: _zh_number(int(m.group(0))), s)
    s = re.sub(r"[^\w\s]", "", s)
    if lang == "zh":
        # Compare pronunciation, not characters: whisper freely emits
        # traditional script and homophones (林峰 for 林风), which are not
        # TTS errors. Toneless pinyin per character; non-CJK runs kept as words.
        from pypinyin import Style, lazy_pinyin

        out: list[str] = []
        for tok in re.findall(r"[一-鿿]|[^一-鿿\s]+", s):
            if re.match(r"[一-鿿]", tok):
                out.extend(lazy_pinyin(tok, style=Style.NORMAL))
            else:
                out.append(tok)
        return out
    return s.split()


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
