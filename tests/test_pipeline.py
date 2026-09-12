"""End-to-end through build with the tone backend (no model, needs ffmpeg)."""

import shutil
from pathlib import Path

import pytest
import yaml

from ab.config import BookPaths
from ab.stages import attribute, build, ingest, normalize, render

SAMPLE = Path(__file__).parent.parent / "books" / "sample"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_end_to_end_tone(tmp_path):
    book = tmp_path / "book"
    book.mkdir()
    shutil.copy(SAMPLE / "source.txt", book / "source.txt")
    (book / "book.yaml").write_text(yaml.safe_dump({
        "title": "T", "language": "en",
        "tts": {"backend": "tone"},
        "voices": {"narrator": "a220"},
    }))
    paths = BookPaths(book)
    cfg = paths.load_config()
    for stage in (ingest, normalize, attribute, render):
        stage.run(paths, cfg)
    out = build.run(paths, cfg)
    assert out.exists() and out.stat().st_size > 1000
    wavs = list(paths.audio.glob("*.wav"))
    n = len(wavs)
    assert n == 24  # one per paragraph, narrator-only mode (no cast.yaml)

    # Re-render is a no-op: same hashes, no new files.
    render.run(paths, cfg)
    assert len(list(paths.audio.glob("*.wav"))) == n

    # Changing the voice produces new files and leaves the old ones for A/B.
    cfg.voices["narrator"] = "a440"
    render.run(paths, cfg, force=True)
    assert len(list(paths.audio.glob("*.wav"))) == 2 * n
