from ab.stages.verify import error_rate


def test_wer_en():
    assert error_rate("the cat sat on the mat", "the cat sat on the mat", "en") == 0.0
    assert error_rate("the cat sat on the mat", "The cat sat on the mat.", "en") == 0.0
    assert abs(error_rate("the cat sat on the mat", "the cat on the mat", "en") - 1 / 6) < 1e-9
    assert error_rate("hello", "", "en") == 1.0


def test_cer_zh_is_phonetic():
    assert error_rate("你好，需要帮忙吗？", "你好需要帮忙吗", "zh") == 0.0
    assert abs(error_rate("你好需要帮忙吗", "你好需要帮吗", "zh") - 1 / 7) < 1e-9
    # traditional script and homophones are not errors
    assert error_rate("一间面积不小的教室里", "一間面積不小的教室裡", "zh") == 0.0
    assert error_rate("林风的伴生鬼灵", "林峰的半生鬼灵", "zh") == 0.0
    # a real substitution still counts
    assert error_rate("柳城", "流程", "zh") == 0.0  # liu cheng == liu cheng: same sounds
    assert error_rate("是啊万一", "十二万一", "zh") > 0
    # digits in the transcript match spelled-out numerals in the source
    assert error_rate("一群十五、六岁的少年", "一群15、6歲的少年", "zh") == 0.0
