from pathlib import Path

import yaml

from ab.check import check
from ab.config import BookPaths

SAMPLE = Path(__file__).parent.parent / "books" / "sample"


def _book(tmp_path, voices, backend="kokoro", lang="en"):
    b = tmp_path / "b"
    b.mkdir()
    (b / "cast.yaml").write_text((SAMPLE / "cast.yaml").read_text())
    (b / "book.yaml").write_text(yaml.safe_dump({
        "title": "t", "language": lang, "tts": {"backend": backend}, "voices": voices}))
    return BookPaths(b)


def test_clean_config_has_no_errors(tmp_path):
    p = _book(tmp_path, {"kokoro": {"narrator": "bm_george", "Elizabeth": "af_heart", "Jane": "af_sarah",
                                    "Mr. Bennet": "bm_lewis", "Mrs. Bennet": "bf_emma", "Bingley": "am_echo"}})
    errors, warnings = check(p, p.load_config())
    assert errors == [] and warnings == []


def test_typo_unknown_id_wrong_language_and_missing_narrator(tmp_path):
    p = _book(tmp_path, {"kokoro": {"Elizabet": "af_heart", "Jane": "af_nope", "Mr. Bennet": "zm_yunxi"}})
    errors, warnings = check(p, p.load_config())
    joined = "\n".join(errors)
    assert "no narrator" in joined
    assert "Elizabet" in joined and "Elizabeth" in joined      # suggestion
    assert "unknown voice id 'af_nope'" in joined
    assert "'zm_yunxi' is not a en voice" in joined
    assert any("main cast member" in w for w in warnings)      # Mrs. Bennet etc. unvoiced


def test_alias_and_duplicate_are_warnings(tmp_path):
    p = _book(tmp_path, {"kokoro": {"narrator": "bm_george", "Lizzy": "af_heart", "Jane": "af_heart"}})
    errors, warnings = check(p, p.load_config())
    assert errors == []
    assert any("alias" in w for w in warnings) and any("shared by" in w for w in warnings)


def test_clip_paths_must_exist(tmp_path):
    p = _book(tmp_path, {"qwen3tts": {"narrator": "Uncle_Fu", "Jane": "voices/jane.wav"}}, backend="qwen3tts", lang="zh")
    errors, _ = check(p, p.load_config())
    assert any("clip not found" in e for e in errors)
    (p.root / "voices").mkdir()
    (p.root / "voices" / "jane.wav").write_bytes(b"")
    errors, _ = check(p, p.load_config())
    assert not any("clip not found" in e for e in errors)
