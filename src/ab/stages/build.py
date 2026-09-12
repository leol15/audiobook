"""Stage 7: per-line audio -> out/<book>.m4b with chapter markers."""

from __future__ import annotations

import re
import subprocess

import numpy as np
import soundfile as sf

from ab.audio import silence
from ab.config import BookConfig, BookPaths
from ab.models import ChapterList
from ab.stages.attribute import read_lines


def run(paths: BookPaths, cfg: BookConfig, force: bool = False):
    lines = read_lines(paths)
    book = ChapterList.model_validate_json(paths.chapters_norm.read_text(encoding="utf-8"))
    sr = _sample_rate(paths, lines)
    paths.out.mkdir(parents=True, exist_ok=True)
    chapter_dir = paths.work / "chapters"
    chapter_dir.mkdir(exist_ok=True)

    by_chapter: dict[int, list] = {}
    for ln in lines:
        by_chapter.setdefault(ln.chapter, []).append(ln)

    chapter_files, markers, t = [], [], 0.0
    for ch in book.chapters:
        parts = [silence(cfg.pauses.chapter, sr)]
        prev_speaker = None
        for ln in by_chapter.get(ch.index, []):
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
        audio = np.concatenate(parts)
        f = chapter_dir / f"{ch.index:03d}.wav"
        sf.write(f, audio, sr)
        chapter_files.append(f)
        dur = len(audio) / sr
        markers.append((t, t + dur, ch.title))
        t += dur

    concat_list = chapter_dir / "concat.txt"
    concat_list.write_text("".join(f"file '{f.resolve()}'\n" for f in chapter_files))
    meta = paths.work / "ffmetadata.txt"
    meta.write_text(_ffmetadata(cfg, markers), encoding="utf-8")

    backends = {ln.backend for ln in lines if ln.backend}
    tag = f".{next(iter(backends))}" if len(backends) == 1 else ""
    out = paths.out / f"{_slug(cfg.title)}{tag}.m4b"
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", str(meta)]
    if cfg.cover:
        cmd += ["-i", str(paths.root / cfg.cover)]
    cmd += ["-map_metadata", "1", "-map", "0:a"]
    if cfg.cover:
        cmd += ["-map", "2:v", "-c:v", "copy", "-disposition:v", "attached_pic"]
    cmd += ["-af", f"loudnorm=I={cfg.loudness_lufs}:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "44100", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
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
