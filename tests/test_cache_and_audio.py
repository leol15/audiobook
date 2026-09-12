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
