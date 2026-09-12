from ab.assign import assign, profile
from ab.voicepool import pool


def test_profile_en_and_zh():
    assert profile("Mr. Bennet's wife; female; middle-aged; excitable").gender == "f"
    assert profile("Mr. Bennet's wife; female; middle-aged; excitable").age == "adult"
    assert profile("Mr. Bennet's daughter; female; young; sharp-witted") == profile("a young girl")
    assert profile("Head of the family; male; middle-aged; dry").age == "adult"
    assert profile("Head of the family; male; old; dry").age == "old"
    assert profile("男性，16岁，说话沉稳") == profile("a young boy")
    assert profile("女性，年龄不详") .gender == "f"
    assert profile("无性别，年龄不详").gender == "?"


def test_assign_matches_gender_avoids_reuse_and_keeps_existing():
    voices = pool("kokoro", "en")
    cast = [("Mrs. Bennet", "female; middle-aged"), ("Elizabeth", "female; young"),
            ("Jane", "female; young"), ("Mr. Bennet", "male; old"), ("Bingley", "male; young")]
    out = assign(cast, voices, {"narrator": "bm_george"}, "en")
    by_id = {v.id: v for v in voices}
    assert out["narrator"] == "bm_george"
    for name, desc in cast:
        assert by_id[out[name]].gender == profile(desc).gender
    chars = [out[n] for n, _ in cast]
    assert len(set(chars)) == len(chars)                 # no reuse while the pool lasts
    assert "bm_george" not in chars                      # narrator voice left to the narrator
    assert out["_default"] == "bm_george"
    # existing assignment survives
    out2 = assign(cast, voices, {"narrator": "am_michael", "Jane": "af_sky"}, "en")
    assert out2["Jane"] == "af_sky" and out2["narrator"] == "am_michael"


def test_assign_reuses_least_used_when_pool_exhausted():
    voices = pool("kokoro", "zh")  # 4 female voices
    cast = [(f"女{i}", "女性，年轻") for i in range(6)]
    out = assign(cast, voices, {}, "zh")
    ids = [out[n] for n, _ in cast]
    counts = {v: ids.count(v) for v in set(ids)}
    assert max(counts.values()) == 2 and all(v.startswith("zf_") for v in ids)
