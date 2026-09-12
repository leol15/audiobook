import yaml

from ab.config import BookPaths
from ab.voices import _slug, _write_voice_map


def test_slug():
    assert _slug("Mr. Bennet") == "Mr_Bennet"
    assert _slug("林风") == "林风"
    assert _slug("未知角色1") == "未知角色1"


def test_write_voice_map_nests_flat_map(tmp_path):
    (tmp_path / "book.yaml").write_text(yaml.safe_dump({
        "title": "t", "tts": {"backend": "kokoro"}, "voices": {"narrator": "bm_george"}}))
    paths = BookPaths(tmp_path)
    _write_voice_map(paths, "qwen3tts", {"narrator": "voices/narrator.wav"})
    data = yaml.safe_load((tmp_path / "book.yaml").read_text())
    assert data["voices"] == {"kokoro": {"narrator": "bm_george"},
                              "qwen3tts": {"narrator": "voices/narrator.wav"}}
    assert paths.load_config().voice_map("qwen3tts") == {"narrator": "voices/narrator.wav"}
