"""Stage 6: transcribe rendered audio and re-render lines whose text drifted.

Needs the `verify` extra (faster-whisper). Off by default in `ab run`; enable
with --skip "" or run `ab verify` directly. Meant for autoregressive backends;
Kokoro rarely needs it.

Chapter-granular: 06-verify/cNNN.jsonl is stamped with the chapter's render
file, so only chapters whose audio changed are revisited, and within a chapter
only lines whose current audio has no result are transcribed.

Batched: `verify.batch` lines are decoded to 16 kHz, laid end to end with a
gap, and sent as one call to faster-whisper's BatchedInferencePipeline with
one clip per line, so the encoder and decoder run on a batch instead of a
file at a time. Segments come back with the clip's start time, which maps
them to their line. A line longer than whisper's 30 s window is transcribed
alone the old way (the batched pipeline would truncate it).
"""

from __future__ import annotations

import bisect
import re
import unicodedata

import numpy as np
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
            batch = max(1, cfg.verify.batch)
            groups = [todo[k:k + batch] for k in range(0, len(todo), batch)]
            for group in track(groups, description=f"verify[{model_name}] c{ci:03d} pass {attempt + 1}"):
                items = [(ln.text, state[ln.id].audio) for ln in group]
                for ln, res in zip(group, _check_batch(model, paths, items, cfg)):
                    st = state[ln.id]
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
    # Per-line rates are clamped at 1.0 for the mean: whisper's hallucinated
    # repetitions on 2 s sound-effect clips score 36+ and would swamp the book.
    rates = [min(vr.error_rate, 1.0)
             for i in chapter_indexes(paths) for vr in _current_results(paths, i).values()]
    mean = sum(rates) / len(rates) if rates else 0.0
    note(paths, f"verify[{model_name}]: {checked} lines transcribed this run, {len(rates)} with a "
         f"current result, mean error {mean:.3f}, {len(bad)} above threshold")
    write_views(paths)  # refresh the script with verify flags
    return paths.verify_dir


def _current_results(paths: BookPaths, chapter: int) -> dict[str, VerifyResult]:
    state = read_render(paths, chapter)
    return {k: v for k, v in read_verify(paths, chapter).items()
            if state.get(k) and state[k].audio == v.audio}


WHISPER_SR = 16000
MAX_CLIP_SECONDS = 28.0  # whisper window is 30 s; longer lines go one at a time
GAP_SECONDS = 0.5


def _load(model_name: str):
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as e:
        raise SystemExit("verify needs faster-whisper: uv sync --extra verify") from e
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    return model, BatchedInferencePipeline(model)


def _check_batch(models, paths: BookPaths, items: list[tuple[str, str]], cfg: BookConfig) -> list[VerifyResult]:
    """Transcribe several lines in one batched call; returns one result per item, in order."""
    from faster_whisper.audio import decode_audio

    model, pipeline = models
    audios = [decode_audio(str(paths.work / a), sampling_rate=WHISPER_SR) for _, a in items]
    durations = [len(a) / WHISPER_SR for a in audios]
    transcripts: list[str | None] = [None] * len(items)
    short = [i for i, d in enumerate(durations) if d <= MAX_CLIP_SECONDS]
    if len(short) > 1 and cfg.verify.batch > 1:
        joined, clips = concat_clips([audios[i] for i in short], WHISPER_SR)
        segments, _ = pipeline.transcribe(joined, language=cfg.language, beam_size=1,
                                          clip_timestamps=clips, batch_size=len(clips),
                                          condition_on_previous_text=False, vad_filter=False)
        for i, text in zip(short, segments_to_clips(list(segments), clips, cfg.language)):
            transcripts[i] = text
    for i in [j for j in range(len(items)) if transcripts[j] is None]:  # long lines, or batch 1
        segments, _ = model.transcribe(audios[i], language=cfg.language, beam_size=1,
                                       condition_on_previous_text=False)
        transcripts[i] = _join(list(segments), cfg.language)
    out = []
    for (text, _), transcript, dur in zip(items, transcripts, durations):
        rate, edits = error_rate(text, transcript or "", cfg.language, with_edits=True)
        ok = judge(text, transcript or "", rate, edits, dur, cfg)
        out.append(VerifyResult(id="", transcript=transcript or "", error_rate=rate, edits=edits,
                                duration=dur, ok=ok))
    return out


_CHARS_PER_S = {"zh": 4.0, "en": 15.0}


def judge(text: str, transcript: str, rate: float, edits: int, dur: float, cfg: BookConfig) -> bool:
    """Is this line's audio acceptable?

    - A runaway (audio far longer than the text could take) always fails.
    - Very short texts (sound effects: 哒哒, 嘶, 嗷嗷) are judged by duration
      only: whisper hallucinates long repetitions on a correct 2 s clip of a
      repeated syllable, so its transcript says nothing about the audio.
    - Otherwise the phonetic error rate applies, tolerating one edit so a
      single misheard word on a short line does not trigger a re-render.
    """
    expected = len(text) / _CHARS_PER_S.get(cfg.language, 8.0)
    runaway = dur > 3 * expected + 3
    if runaway:
        return False
    toks = _tokens(text, cfg.language)
    if len(toks) <= 4 or len(set(toks)) <= 2:  # a few syllables, or one repeated (哒哒哒哒)
        return True
    return rate <= cfg.verify.threshold or edits <= 1


def concat_clips(audios: list[np.ndarray], sr: int) -> tuple[np.ndarray, list[dict]]:
    """Lay clips end to end with a gap; returns the array and [{start, end}] in seconds."""
    gap = np.zeros(int(GAP_SECONDS * sr), np.float32)
    parts, clips, t = [], [], 0.0
    for a in audios:
        a = a.astype(np.float32)
        parts += [a, gap]
        clips.append({"start": round(t, 3), "end": round(t + len(a) / sr, 3)})
        t += (len(a) + len(gap)) / sr
    return np.concatenate(parts) if parts else np.zeros(0, np.float32), clips


def segments_to_clips(segments, clips: list[dict], lang: str) -> list[str]:
    """Assign each segment to the clip it starts in (segments carry the clip's offset)."""
    starts = [c["start"] for c in clips]
    per_clip: list[list] = [[] for _ in clips]
    for seg in segments:
        i = max(0, bisect.bisect_right(starts, seg.start + 1e-3) - 1)
        per_clip[i].append(seg)
    return [_join(segs, lang) for segs in per_clip]


def _join(segments, lang: str) -> str:
    return "".join(s.text for s in segments) if lang == "zh" else " ".join(s.text.strip() for s in segments)


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
