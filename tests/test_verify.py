from ab.stages.verify import error_rate


def test_wer_en():
    assert error_rate("the cat sat on the mat", "the cat sat on the mat", "en") == 0.0
    assert error_rate("the cat sat on the mat", "The cat sat on the mat.", "en") == 0.0
    assert abs(error_rate("the cat sat on the mat", "the cat on the mat", "en") - 1 / 6) < 1e-9
    assert error_rate("hello", "", "en") == 1.0


def test_cer_zh():
    assert error_rate("你好，需要帮忙吗？", "你好需要帮忙吗", "zh") == 0.0
    assert abs(error_rate("你好需要帮忙吗", "你好需要帮吗", "zh") - 1 / 7) < 1e-9
