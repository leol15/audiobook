import numpy as np

from ab.audio import crossfade_concat, silence
from ab.cache import render_key


def test_render_key_changes_with_every_input():
    base = render_key("kokoro", "af_sky", {"speed": 1.0}, "hi")
    assert base != render_key("other", "af_sky", {"speed": 1.0}, "hi")
    assert base != render_key("kokoro", "am_adam", {"speed": 1.0}, "hi")
    assert base != render_key("kokoro", "af_sky", {"speed": 1.1}, "hi")
    assert base != render_key("kokoro", "af_sky", {"speed": 1.0}, "hi ")
    assert base == render_key("kokoro", "af_sky", {"speed": 1.0}, "hi")


def test_crossfade_concat_length():
    sr = 1000
    a, b = np.ones(500, np.float32), np.ones(500, np.float32)
    out = crossfade_concat([a, b], sr, fade_ms=10)
    assert len(out) == 990
    assert np.allclose(out, 1.0)
    assert len(silence(0.5, sr)) == 500


def test_voice_map_flat_and_nested():
    from ab.config import BookConfig
    from ab.stages.s05_render import resolve_voice

    flat = BookConfig(title="t", voices={"narrator": "a", "Jane": "b"})
    assert flat.voice_map("kokoro") == {"narrator": "a", "Jane": "b"}
    assert resolve_voice(flat.voice_map("kokoro"), "Jane") == "b"
    assert resolve_voice(flat.voice_map("kokoro"), "Unknown") == "a"

    nested = BookConfig(title="t", voices={"kokoro": {"narrator": "a"}, "chatterbox": {"narrator": "default"}})
    assert nested.voice_map("chatterbox") == {"narrator": "default"}
