"""Stage 7: per-line audio -> 07-chapters/cNNN.wav -> out/<book>.<backend>.m4b (+ per-chapter files).

Chapter-granular: a chapter wav is rebuilt only when its render file or the
pause settings changed, and a per-chapter MP3/M4B (`build.chapter_format`)
is written to out/<title>.<backend>/ as soon as that chapter is fully
rendered, so a book can be listened to while the rest renders. The
whole-book M4B with chapter markers is written once every chapter is ready.
"""

from __future__ import annotations

import re
import subprocess

import numpy as np
import soundfile as sf

from ab import cache
from ab.audio import silence
from ab.config import BookConfig, BookPaths
from ab.lines import chapter_indexes, read_chapter
from ab.log import note
from ab.models import ChapterList
from ab.stages import s05_render as render


def chapter_inputs(paths: BookPaths, cfg: BookConfig, chapter: int) -> dict:
    f = paths.chapter_render(chapter)
    return {"render": cache.file_hash(f) if f.exists() else "", "pauses": cfg.pauses.model_dump()}


def chapter_states(paths: BookPaths, cfg: BookConfig) -> list[tuple[int, list[str]]]:
    return [(i, cache.stale_reasons(paths.chapter_wav(i), chapter_inputs(paths, cfg, i)))
            for i in chapter_indexes(paths)]


def run(paths: BookPaths, cfg: BookConfig, force: bool = False, chapters: set[int] | None = None):
    book = ChapterList.model_validate_json(paths.chapters_norm.read_text(encoding="utf-8"))
    titles = {ch.index: ch.title for ch in book.chapters}
    from ab.report import _backend_name

    ready = {i for i, reasons in render.chapter_states(paths, cfg, _backend_name(cfg.tts.backend))
             if not reasons}
    wanted = [i for i in chapter_indexes(paths) if chapters is None or i in chapters]
    paths.out.mkdir(parents=True, exist_ok=True)
    paths.chapter_audio.mkdir(exist_ok=True)

    backend_tag, sr = None, None
    built = 0
    for ci in wanted:
        if ci not in ready:
            continue
        lines = read_chapter(paths, ci)
        backend_tag = backend_tag or next((ln.backend for ln in lines if ln.backend), None)
        wav = paths.chapter_wav(ci)
        inp = chapter_inputs(paths, cfg, ci)
        if force or not cache.is_fresh(wav, inp):
            sr = sr or _sample_rate(paths, lines)
            _write_chapter_wav(paths, cfg, lines, sr, wav)
            cache.mark_fresh(wav, inp)
            built += 1
        if cfg.build.chapter_format != "none":
            _encode_chapter(paths, cfg, ci, titles[ci], backend_tag, wav, inp, force)

    if len(ready) < len(book.chapters) or any(not cache.is_fresh(paths.chapter_wav(i), chapter_inputs(paths, cfg, i))
                                              for i in range(len(book.chapters))):
        note(paths, f"build: {len(ready)}/{len(book.chapters)} chapters ready, {built} chapter wavs "
             f"rebuilt; whole-book M4B waits for every chapter")
        return paths.out
    out = _write_book(paths, cfg, book, backend_tag or _backend_name(cfg.tts.backend) or "")
    note(paths, f"build: {out.name} ({len(book.chapters)} chapters, {built} chapter wavs rebuilt)")
    return out


def _write_chapter_wav(paths: BookPaths, cfg: BookConfig, lines, sr: int, wav) -> None:
    parts = [silence(cfg.pauses.chapter, sr)]
    prev_speaker = None
    for ln in lines:
        if not ln.audio:
            raise SystemExit(f"line {ln.id} has no audio; run render first")
        if prev_speaker is not None:
            gap = cfg.pauses.speaker_change if ln.speaker != prev_speaker else cfg.pauses.paragraph
            parts.append(silence(gap, sr))
        audio, file_sr = sf.read(paths.work / ln.audio, dtype="float32")
        assert file_sr == sr, f"{ln.id}: sample rate {file_sr} != {sr}"
        parts.append(audio)
        prev_speaker = ln.speaker
    parts.append(silence(cfg.pauses.paragraph, sr))
    sf.write(wav, np.concatenate(parts), sr)


def chapter_out_dir(paths: BookPaths, cfg: BookConfig, backend_tag: str | None):
    return paths.out / f"{_slug(cfg.title)}.{backend_tag}" if backend_tag else paths.out / _slug(cfg.title)


def _encode_chapter(paths, cfg, ci, title, backend_tag, wav, inp, force) -> None:
    d = chapter_out_dir(paths, cfg, backend_tag)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"c{ci:03d} {_slug(title)[:60]}.{cfg.build.chapter_format}"
    # The stamp lives in work/, so out/ holds nothing but audio.
    stamp_for = paths.chapter_audio / f"c{ci:03d}.{cfg.build.chapter_format}"
    if not force and out.exists() and cache.read_stamp(stamp_for) == inp:
        return
    cmd = ["ffmpeg", "-y", "-i", str(wav), "-af", f"loudnorm=I={cfg.loudness_lufs}:TP=-1.5:LRA=11",
           "-metadata", f"title={title}", "-metadata", f"album={cfg.title}",
           "-metadata", f"artist={cfg.author}", "-metadata", f"track={ci + 1}", "-ac", "1", "-ar", "44100"]
    if cfg.build.chapter_format == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "64k"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", "-f", "mp4"]
    cmd.append(str(out))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cache.mark_fresh(stamp_for, inp)


def _write_book(paths: BookPaths, cfg: BookConfig, book: ChapterList, backend_tag: str):
    markers, t, files = [], 0.0, []
    for ch in book.chapters:
        wav = paths.chapter_wav(ch.index)
        dur = sf.info(wav).duration
        markers.append((t, t + dur, ch.title))
        t += dur
        files.append(wav)
    concat_list = paths.chapter_audio / "concat.txt"
    concat_list.write_text("".join(f"file '{f.resolve()}'\n" for f in files))
    meta = paths.ffmetadata
    meta.write_text(_ffmetadata(cfg, markers), encoding="utf-8")
    tag = f".{backend_tag}" if backend_tag else ""
    out = paths.out / f"{_slug(cfg.title)}{tag}.m4b"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", str(meta)]
    if cfg.cover:
        cmd += ["-i", str(paths.root / cfg.cover)]
    cmd += ["-map_metadata", "1", "-map", "0:a"]
    if cfg.cover:
        cmd += ["-map", "2:v", "-c:v", "copy", "-disposition:v", "attached_pic"]
    cmd += ["-af", f"loudnorm=I={cfg.loudness_lufs}:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "44100", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def _sample_rate(paths: BookPaths, lines) -> int:
    for ln in lines:
        if ln.audio:
            return sf.info(paths.work / ln.audio).samplerate
    raise SystemExit("no rendered audio found")


def _ffmetadata(cfg: BookConfig, markers) -> str:
    def esc(s: str) -> str:
        return re.sub(r"([=;#\\\n])", r"\\\1", s)

    out = [";FFMETADATA1", f"title={esc(cfg.title)}", f"artist={esc(cfg.author)}",
           f"album={esc(cfg.title)}", "genre=Audiobook", ""]
    for start, end, title in markers:
        out += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(start * 1000)}",
                f"END={int(end * 1000)}", f"title={esc(title)}", ""]
    return "\n".join(out)


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().replace(" ", "_")
    return s or "book"
