"""Chapter granularity: editing one chapter or fixing one line re-runs only that chapter."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from ab import cache
from ab.config import BookPaths
from ab.lines import read_chapter, read_lines, read_render, read_verify, write_verify
from ab.models import VerifyResult
from ab.report import chapter_status, fix_line, stage_status
from ab.stages import s01_ingest as ingest
from ab.stages import s02_normalize as normalize
from ab.stages import s04_attribute as attribute
from ab.stages import s05_render as render
from ab.stages import s06_verify as verify
from ab.stages import s07_build as build

SAMPLE = Path(__file__).parent.parent / "books" / "sample"
SRC = "CHAPTER I\n\nOne alpha.\n\nOne beta.\n\nCHAPTER II\n\nTwo alpha.\n\nCHAPTER III\n\nThree alpha.\n\nThree beta.\n"


@pytest.fixture
def book(tmp_path):
    b = tmp_path / "book"
    b.mkdir()
    (b / "source.txt").write_text(SRC)
    (b / "book.yaml").write_text(yaml.safe_dump({
        "title": "T", "language": "en", "tts": {"backend": "tone"},
        "voices": {"narrator": "a220", "_default": "a330"}}))
    return BookPaths(b)


def _grid(paths, cfg):
    return {r["chapter"]: (r["attribute"], r["render"], r["verify"], r["build"]) for r in chapter_status(paths, cfg)}


def _stamps(paths, which):
    return {i: cache.read_stamp(which(i)) for i in range(3)}


def test_edit_one_chapter_reruns_only_that_chapter(book):
    cfg = book.load_config()
    for st in (ingest, normalize, attribute, render):
        st.run(book, cfg)
    assert _grid(book, cfg) == {i: ("fresh", "fresh", "missing", "missing") for i in range(3)}
    assert [ln.id for ln in read_lines(book)] == ["c000p0000s00", "c000p0001s00", "c001p0000s00",
                                                  "c002p0000s00", "c002p0001s00"]
    attr_before, render_before = _stamps(book, book.chapter_lines), _stamps(book, book.chapter_render)
    audio_before = {ln.id: ln.audio for ln in read_lines(book)}

    (book.root / "source.txt").write_text(SRC.replace("Two alpha.", "Two alpha changed."))
    ingest.run(book, cfg)
    normalize.run(book, cfg)
    grid = _grid(book, cfg)
    assert grid[0][0] == "fresh" and grid[2][0] == "fresh" and grid[1][0] == "stale"
    row = next(r for r in chapter_status(book, cfg) if r["chapter"] == 1)
    assert "chapter" in row["attribute_detail"]

    attribute.run(book, cfg)
    assert _stamps(book, book.chapter_lines)[0] == attr_before[0]
    assert _stamps(book, book.chapter_lines)[1] != attr_before[1]
    grid = _grid(book, cfg)
    assert grid[0][1] == "fresh" and grid[1][1] == "stale" and grid[2][1] == "fresh"
    render.run(book, cfg)
    after = {ln.id: ln.audio for ln in read_lines(book)}
    changed = [k for k in after if after[k] != audio_before.get(k)]
    assert changed == ["c001p0000s00"]
    assert _stamps(book, book.chapter_render)[0] == render_before[0]
    assert _grid(book, cfg)[1][1] == "fresh"


def test_fix_locked_line_reruns_only_its_chapter_and_line(book):
    cfg = book.load_config()
    for st in (ingest, normalize, attribute, render):
        st.run(book, cfg)
    shutil.copy(SAMPLE / "cast.yaml", book.cast)  # so a speaker name is valid
    n_wavs = len(list(book.audio.glob("*.wav")))
    ln = fix_line(book, "c002p0001s00", speaker="Jane")
    assert ln.locked and ln.speaker == "Jane"
    grid = _grid(book, cfg)
    assert grid[0][1] == "fresh" and grid[1][1] == "fresh" and grid[2][1] == "stale"
    assert "c002p0001s00" not in read_render(book, 2)
    render.run(book, cfg)
    assert len(list(book.audio.glob("*.wav"))) == n_wavs + 1
    assert _grid(book, cfg)[2][1] == "fresh"
    kept = read_chapter(book, 2)
    assert kept[1].locked and kept[1].audio and kept[1].backend == "tone-test"
    # Re-attributing (cast changed) keeps the locked line and its render state.
    attribute.run(book, cfg)
    again = read_chapter(book, 2)
    assert again[1].locked and again[1].speaker == "Jane"
    render.run(book, cfg)
    assert len(list(book.audio.glob("*.wav"))) == n_wavs + 1  # nothing new to synthesize


def test_voice_change_is_detected_and_cached_lines_are_free(book):
    cfg = book.load_config()
    for st in (ingest, normalize, attribute, render):
        st.run(book, cfg)
    cfg.voices["narrator"] = "a440"
    assert all(r["render"] == "stale" and "voices" in r["render_detail"] for r in chapter_status(book, cfg))
    render.run(book, cfg)
    assert all(r["render"] == "fresh" for r in chapter_status(book, cfg))
    n = len(list(book.audio.glob("*.wav")))
    cfg.voices["narrator"] = "a220"  # back: every file is already cached
    render.run(book, cfg)
    assert len(list(book.audio.glob("*.wav"))) == n
    assert all(r["render"] == "fresh" for r in chapter_status(book, cfg))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_build_emits_ready_chapters_before_the_whole_book(book):
    cfg = book.load_config()
    for st in (ingest, normalize, attribute):
        st.run(book, cfg)
    render.run(book, cfg, chapters={0, 2})
    out = build.run(book, cfg)
    assert out == book.out  # whole book not written yet
    names = sorted(p.name[:4] for p in (book.out / "T.tone-test").glob("*.mp3"))
    assert names == ["c000", "c002"]
    assert not list(book.out.glob("*.m4b"))
    grid = _grid(book, cfg)
    assert grid[1] == ("fresh", "missing", "missing", "missing")

    render.run(book, cfg)
    wav1_mtime = book.chapter_wav(0).stat().st_mtime
    out = build.run(book, cfg)
    assert out.suffix == ".m4b" and out.exists()
    assert book.chapter_wav(0).stat().st_mtime == wav1_mtime  # untouched chapter not rebuilt
    assert sorted(p.name[:4] for p in (book.out / "T.tone-test").glob("*.mp3")) == ["c000", "c001", "c002"]
    assert all(r["build"] == "fresh" for r in chapter_status(book, cfg))
    states = {r["stage"]: r for r in stage_status(book, cfg)}
    assert states["build"]["state"] == "fresh" and "3/3 chapters" in states["build"]["detail"]


def test_verify_results_follow_their_audio(book):
    cfg = book.load_config()
    for st in (ingest, normalize, attribute, render):
        st.run(book, cfg)
    st = read_render(book, 0)
    write_verify(book, 0, {lid: VerifyResult(id=lid, transcript="x", error_rate=0.5, edits=3, duration=1.0,
                                             ok=False, audio=s.audio) for lid, s in st.items()})
    cache.mark_fresh(book.chapter_verify(0), verify.chapter_inputs(book, cfg, 0))
    lines = read_chapter(book, 0)
    assert all(ln.error_rate == 0.5 for ln in lines)
    assert _grid(book, cfg)[0][2] == "fresh"
    # A fix drops the line's result; the chapter's verify goes stale with the render file.
    shutil.copy(SAMPLE / "cast.yaml", book.cast)
    fix_line(book, "c000p0000s00", speaker="Jane")
    assert "c000p0000s00" not in read_verify(book, 0)
    render.run(book, cfg)
    assert read_chapter(book, 0)[0].error_rate is None
    assert read_chapter(book, 0)[1].error_rate == 0.5
    assert _grid(book, cfg)[0][2] == "stale"
