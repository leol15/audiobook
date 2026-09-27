"""Delivery styles: LLM labels per quote, per-style voice clips, style-aware voice resolution."""

from __future__ import annotations

import yaml

from ab.check import check
from ab.config import BookConfig, BookPaths, Cast, Character
from ab.models import Chapter
from ab.stages import s04_attribute as attribute
from ab.stages.s05_render import resolve_voice
from ab.styles import split_key, voice_key
from ab.voices import plan

CAST = Cast(characters={"Anna": Character(), "Ben": Character()})
CH = Chapter(index=0, title="One", paragraphs=[
    '"Get out!" said Anna.',          # rules resolve the speaker; LLM only adds the style
    '"Please, no."',                  # LLM resolves both
])


class StyleLLM:
    def __init__(self):
        self.schemas: list[dict] = []
        self.prompts: list[str] = []

    def window_budget(self, overhead):
        return 4000

    def json(self, prompt, schema, system=""):
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return {"speakers": [{"quote_id": 1, "speaker": "Anna", "confidence": 1, "style": "angry"},
                             {"quote_id": 2, "speaker": "Ben", "confidence": 0.7, "style": "soft"},
                             {"quote_id": 2, "speaker": "Ben", "confidence": 0.7, "style": "bogus"}]}


def test_llm_labels_every_quote_when_styles_on():
    llm = StyleLLM()
    stats = {"narration": 0, "rules": 0, "llm": 0, "unknown": 0, "locked": 0, "styled": 0}
    cfg = BookConfig(title="t", styles=["angry", "soft"])
    lines = attribute.attribute_chapter(CH, CAST, cfg, llm, stats, {})
    dialogue = [ln for ln in lines if ln.kind == "dialogue"]
    assert [(ln.speaker, ln.style) for ln in dialogue] == [("Anna", "angry"), ("Ben", "soft")]
    assert dialogue[0].confidence >= 0.85       # rules' speaker untouched by the LLM echo
    enum = llm.schemas[0]["properties"]["speakers"]["items"]["properties"]["style"]["enum"]
    assert enum == ["neutral", "angry", "soft"]
    assert "neutral" in llm.prompts[0] and "angry (angry or shouting)" in llm.prompts[0]
    assert stats["styled"] == 2 and stats["llm"] == 1


def test_styles_off_keeps_old_prompt_and_skips_resolved_windows():
    llm = StyleLLM()
    stats = {"narration": 0, "rules": 0, "llm": 0, "unknown": 0, "locked": 0, "styled": 0}
    cfg = BookConfig(title="t", styles=[])
    resolved = Chapter(index=0, title="One", paragraphs=['"Get out!" said Anna.'])
    assert all(ln.style is None for ln in attribute.attribute_chapter(resolved, CAST, cfg, llm, stats, {}))
    assert llm.prompts == []                     # nothing pending: no call at all
    attribute.attribute_chapter(CH, CAST, cfg, llm, stats, {})
    assert "style" not in llm.schemas[0]["properties"]["speakers"]["items"]["properties"]
    assert "neutral" not in llm.prompts[0]


def test_style_set_is_part_of_the_chapter_stamp():
    a = attribute.chapter_inputs(CH, BookConfig(title="t", styles=["angry"]), "k")
    b = attribute.chapter_inputs(CH, BookConfig(title="t", styles=[]), "k")
    assert a != b


def test_resolve_voice_prefers_style_clip_then_falls_back():
    voices = {"narrator": "n.wav", "_default": "d.wav", "_default@sad": "d.sad.wav",
              "Anna": "a.wav", "Anna@angry": "a.angry.wav"}
    assert resolve_voice(voices, "Anna", "angry") == "a.angry.wav"
    assert resolve_voice(voices, "Anna", "sad") == "a.wav"          # no such style clip for Anna
    assert resolve_voice(voices, "Anna", None) == "a.wav"
    assert resolve_voice(voices, "Ben", "sad") == "d.sad.wav"       # minor character, styled default
    assert resolve_voice(voices, "Ben", "angry") == "d.wav"
    assert resolve_voice({"narrator": "n.wav"}, "Ben", "angry") == "n.wav"


def test_voice_key_roundtrip():
    assert voice_key("Mr. Bennet", "angry") == "Mr. Bennet@angry"
    assert voice_key("Anna", None) == "Anna" and voice_key("Anna", "neutral") == "Anna"
    assert split_key("Mr. Bennet@angry") == ("Mr. Bennet", "angry")
    assert split_key("Anna") == ("Anna", None)


def test_plan_one_clip_per_role_per_style_narrator_neutral():
    jobs = plan({"narrator": "calm", "林风": "young man"}, {"林风": "这里就是鬼洞？"}, "zh", ["angry", "sad"])
    keys = [j.key for j in jobs]
    assert keys == ["narrator", "林风", "林风@angry", "林风@sad"]
    angry = next(j for j in jobs if j.key == "林风@angry")
    assert angry.rel == "voices/林风.angry.wav" and angry.text == "这里就是鬼洞？"
    assert angry.instruct.startswith("young man，") and "愤怒" in angry.instruct
    assert next(j for j in jobs if j.key == "narrator").instruct == "calm"


def test_check_accepts_style_keys_and_warns_on_unknown_style(tmp_path):
    b = tmp_path / "b"
    (b / "voices").mkdir(parents=True)
    for f in ("a.wav", "a.angry.wav", "a.bogus.wav"):
        (b / "voices" / f).write_bytes(b"")
    (b / "cast.yaml").write_text(yaml.safe_dump({"characters": {"Anna": {}}}))
    (b / "book.yaml").write_text(yaml.safe_dump({
        "title": "t", "language": "zh", "tts": {"backend": "qwen3tts"}, "styles": ["angry"],
        "voices": {"qwen3tts": {"narrator": "voices/a.wav", "Anna": "voices/a.wav",
                                "Anna@angry": "voices/a.angry.wav", "Anna@bogus": "voices/a.bogus.wav",
                                "Nobody@angry": "voices/a.angry.wav"}}}))
    p = BookPaths(b)
    errors, warnings = check(p, p.load_config())
    assert [e for e in errors if "Nobody@angry" in e and "not in cast" in e]
    assert not [e for e in errors if "Anna" in e]
    assert [w for w in warnings if "Anna@bogus" in w and "bogus" in w]
