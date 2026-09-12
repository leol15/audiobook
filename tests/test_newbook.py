from pathlib import Path

import yaml

from ab.config import BookConfig
from ab.newbook import create, ingest_preview

SAMPLE = Path(__file__).parent.parent / "books" / "sample" / "source.txt"
ZH = Path(__file__).parent / "fixtures" / "zh.txt"


def test_new_book_template_validates_and_ingests(tmp_path):
    paths = create(tmp_path / "my-novel", source=SAMPLE, language="en", author="Jane Austen")
    cfg = paths.load_config()
    assert isinstance(cfg, BookConfig)
    assert cfg.title == "My Novel" and cfg.author == "Jane Austen" and cfg.language == "en"
    assert cfg.voice_map("kokoro")["narrator"] == "bm_george"
    assert cfg.voice_map("qwen3tts")["narrator"] == "Ryan"
    # every top-level section is present in the written file (with comments intact)
    text = paths.config.read_text()
    for key in ("tts:", "voices:", "llm:", "cast:", "pauses:", "verify:", "build:", "loudness_lufs:"):
        assert key in text
    assert (paths.root / "source.txt").exists()
    ingest_preview(paths)
    assert paths.chapters.exists()


def test_new_book_zh_regex_and_refuses_nonempty(tmp_path):
    paths = create(tmp_path / "zh", source=ZH, language="zh", chapter_regex=r"^\d{3}\s+\S.*$",
                   backend="qwen3tts")
    data = yaml.safe_load(paths.config.read_text())
    assert data["chapter_regex"] == r"^\d{3}\s+\S.*$" and data["tts"]["backend"] == "qwen3tts"
    assert data["voices"]["kokoro"]["narrator"] == "zm_yunyang"
    import pytest
    with pytest.raises(SystemExit):
        create(tmp_path / "zh", source=ZH, language="zh")
