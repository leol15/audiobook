from ab.config import Cast, Character
from ab.quotes import extract

CAST = Cast(characters={
    "Mrs. Bennet": Character(aliases=["his wife", "his lady"]),
    "Mr. Bennet": Character(aliases=[]),
    "Elizabeth": Character(aliases=["Lizzy"]),
    "Jane": Character(),
})


def flat(paras):
    return [(s.kind, s.text, s.speaker) for ps in paras for s in ps.spans]


def test_split_and_after_tag():
    paras = extract(['"My dear Mr. Bennet," said his lady to him one day, "have you heard?"'], "en", CAST.resolve)
    assert flat(paras) == [
        ("dialogue", "My dear Mr. Bennet,", "Mrs. Bennet"),
        ("narration", "said his lady to him one day,", "narrator"),
        ("dialogue", "have you heard?", "Mrs. Bennet"),
    ]


def test_before_tag_and_name_first():
    paras = extract(['Jane said, "Hello." "Hi," Elizabeth replied.'], "en", CAST.resolve)
    d = [(t, s) for k, t, s in flat(paras) if k == "dialogue"]
    assert d == [("Hello.", "Jane"), ("Hi,", "Elizabeth")]


def test_unresolved_quote_left_for_llm():
    paras = extract(['"What is his name?"', '"Bingley."'], "en", CAST.resolve)
    assert all(s.speaker is None for ps in paras for s in ps.spans)
    assert [s.quote_id for ps in paras for s in ps.spans] == [1, 2]


def test_multi_paragraph_quote_continues_speaker():
    paras = extract([
        '"First paragraph of a long speech,' , '"and the second paragraph," said Jane.'
    ], "en", CAST.resolve)
    assert paras[0].open_end
    d = [(k, s) for k, t, s in flat(paras) if k == "dialogue"]
    assert d[1] == ("dialogue", "Jane")


def test_zh_tags():
    cast = Cast(characters={"小明": Character(), "店主": Character(aliases=["老板"])})
    paras = extract(['"你好，"店主说，"需要帮忙吗？"', '小明答道："不用。"'], "zh", cast.resolve)
    d = [(t, s) for k, t, s in flat(paras) if k == "dialogue"]
    assert d == [("你好，", "店主"), ("需要帮忙吗？", "店主"), ("不用。", "小明")]


def test_resolve_is_tolerant():
    assert CAST.resolve("mrs bennet") == "Mrs. Bennet"
    assert CAST.resolve("Mrs. Bennet.") == "Mrs. Bennet"
    assert CAST.resolve("LIZZY") == "Elizabeth"
    assert CAST.resolve("Bingley") is None
    assert CAST.resolve("") is None
