from pathlib import Path

from ab.stages.ingest import parse
from ab.stages.normalize import _zh_number, normalize
from ab.text import chunk_text, split_sentences

SAMPLE = Path(__file__).parent.parent / "books" / "sample" / "source.txt"
ZH = Path(__file__).parent / "fixtures" / "zh.txt"


def test_en_chapters_and_paragraphs():
    chs = parse(SAMPLE.read_text(), "en", None, is_markdown=False)
    assert [c.title for c in chs] == ["CHAPTER I", "CHAPTER II"]
    assert len(chs[0].paragraphs) == 18
    assert len(chs[1].paragraphs) == 6
    # hard-wrapped lines are unwrapped, quotes canonicalized
    assert chs[0].paragraphs[0].startswith("It is a truth universally acknowledged, that a single man")
    assert chs[0].paragraphs[2].startswith('"My dear Mr. Bennet,"')


def test_zh_chapters_and_quotes():
    chs = parse(ZH.read_text(encoding="utf-8"), "zh", None, is_markdown=False)
    assert len(chs) == 1 and chs[0].title.startswith("第一章")
    assert chs[0].paragraphs[1].startswith('"你好，"')


def test_markdown_headings():
    chs = parse("# One\n\nfoo\n\n## Two\n\nbar\n", "en", None, is_markdown=True)
    assert [(c.title, c.paragraphs) for c in chs] == [("One", ["foo"]), ("Two", ["bar"])]


def test_en_sentence_split_respects_abbreviations():
    s = split_sentences('"My dear Mr. Bennet," said his lady. Mr. Bennet replied. He was tired.', "en")
    assert s == ['"My dear Mr. Bennet," said his lady.', "Mr. Bennet replied.", "He was tired."]


def test_zh_sentence_split():
    assert split_sentences("你好。需要帮忙吗？不用。", "zh") == ["你好。", "需要帮忙吗？", "不用。"]


def test_chunk_text_respects_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    chunks = chunk_text(text, "en", 60)
    assert all(len(c) <= 60 for c in chunks)
    assert " ".join(chunks) == text


def test_normalize_en():
    assert normalize("Mr. Darcy met Dr. Jones — briefly, etc.", "en") == "Mr. Darcy met Dr. Jones, briefly, et cetera"


def test_normalize_zh_numbers():
    assert normalize("2024年有3个人和15只猫", "zh") == "二零二四年有三个人和十五只猫"
    assert _zh_number(105) == "一百零五"
    assert _zh_number(20000) == "二万"
    assert _zh_number(12345) == "一万二千三百四十五"


def test_zh_ellipsis_and_fullwidth_period():
    from ab.stages.ingest import parse
    chs = parse("001 标题\n\n“这还算好的．．．要是失控了。”\n\n“．．．．．”\n", "zh", r"^\d{3}\s+\S.*$", is_markdown=False)
    assert chs[0].title == "001 标题"
    assert normalize(chs[0].paragraphs[0], "zh") == '"这还算好的，要是失控了。"'
    assert normalize(chs[0].paragraphs[1], "zh") == '"，"'
