"""status / script / fix on a tone-rendered book (no models, needs ffmpeg)."""

import shutil
from pathlib import Path

import pytest
import yaml

from ab.config import BookPaths
from ab.report import fix_line, stage_status, write_report
from ab.stages import attribute, ingest, normalize, render

SAMPLE = Path(__file__).parent.parent / "books" / "sample"


@pytest.fixture
def book(tmp_path):
    b = tmp_path / "book"
    b.mkdir()
    shutil.copy(SAMPLE / "source.txt", b / "source.txt")
    shutil.copy(SAMPLE / "cast.yaml", b / "cast.yaml")
    (b / "book.yaml").write_text(yaml.safe_dump({
        "title": "T", "language": "en", "tts": {"backend": "tone"},
        "voices": {"narrator": "a220", "Mrs. Bennet": "a440"}}))
    return BookPaths(b)


def test_status_progression_and_stale_reason(book):
    cfg = book.load_config()
    states = {r["stage"]: r["state"] for r in stage_status(book, cfg)}
    assert states["ingest"] == "missing" and states["render"] == "missing"

    ingest.run(book, cfg)
    normalize.run(book, cfg)
    states = {r["stage"]: r["state"] for r in stage_status(book, cfg)}
    assert states["ingest"] == "fresh" and states["normalize"] == "fresh"
    assert book.chapters.name == "01-chapters.json"

    # Change an input: the stage reports stale and names the changed input.
    (book.root / "source.txt").write_text("CHAPTER I\n\nNew text.\n")
    row = next(r for r in stage_status(book, cfg) if r["stage"] == "ingest")
    assert row["state"] == "stale" and "source" in row["detail"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_script_fix_and_report(book, monkeypatch):
    cfg = book.load_config()
    ingest.run(book, cfg)
    normalize.run(book, cfg)
    # No Ollama in tests: attribute in narrator-only mode by hiding the cast.
    book.cast.unlink()
    attribute.run(book, cfg)
    render.run(book, cfg)

    assert book.script.exists() and "**narrator**" in book.script.read_text()
    assert (book.audio_by_line / "c000p0000s00.wav").is_symlink()

    # fix: speaker must be in cast; re-add cast then fix a line.
    shutil.copy(SAMPLE / "cast.yaml", book.cast)
    ln = fix_line(book, "c000p0002s00", speaker="Mrs. Bennet")
    assert ln.locked and ln.speaker == "Mrs. Bennet" and ln.audio is None
    assert "[lock]" in book.script.read_text()

    # render only redoes the fixed line
    before = len(list(book.audio.glob("*.wav")))
    render.run(book, cfg)
    assert len(list(book.audio.glob("*.wav"))) == before + 1

    rep = write_report(book, cfg)
    text = rep.read_text()
    assert "## Stages" in text and "Mrs. Bennet" in text
    assert book.run_log.exists()
