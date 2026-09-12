"""LLM context budget: prompts are sized to num_ctx and never truncated silently."""

import json

import pytest

from ab.config import BookConfig, Cast, Character
from ab.llm import OUTPUT_RESERVE, Ollama, PromptTooLong, approx_tokens, pack
from ab.quotes import extract
from ab.stages import s04_attribute as attribute


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class FakeClient:
    """Stands in for httpx.Client; records bodies and answers with canned JSON."""

    def __init__(self, response=None, prompt_eval_count=None):
        self.bodies = []
        self.response = response if response is not None else {"speakers": []}
        self.prompt_eval_count = prompt_eval_count

    def post(self, path, json=None):
        self.bodies.append(json)
        data = {"response": __import__("json").dumps(self.response)}
        if self.prompt_eval_count is not None:
            data["prompt_eval_count"] = self.prompt_eval_count
        return FakeResponse(data)


def test_approx_tokens_is_conservative_per_script():
    assert approx_tokens("") == 0
    assert approx_tokens("林风的伴生鬼灵") == 7          # one token per CJK char
    assert approx_tokens("abcdef") == 2                  # three chars per token
    assert approx_tokens("说：你好") == 4 and approx_tokens("Mr. Bennet 说：你好") == 8


def test_pack_respects_budget_and_keeps_oversize_alone():
    items = ["a" * 30, "b" * 30, "c" * 300, "d" * 30]   # 10, 10, 100, 10 tokens
    groups = pack(items, budget=25, sep="")
    assert groups == [[items[0], items[1]], [items[2]], [items[3]]]
    assert pack(items, budget=1000, sep="", max_items=3) == [items[:3], items[3:]]
    assert pack([], 10) == []


def test_json_sends_num_ctx_and_rejects_oversized_prompt():
    llm = Ollama(url="http://x", model="m", num_ctx=4096)
    llm.client = FakeClient()
    assert llm.prompt_budget == 4096 - OUTPUT_RESERVE
    llm.json("hello", {"type": "object"}, system="sys")
    body = llm.client.bodies[0]
    assert body["options"]["num_ctx"] == 4096 and body["model"] == "m"

    with pytest.raises(PromptTooLong):
        llm.json("x" * (llm.prompt_budget * 3 + 10), {"type": "object"})
    assert len(llm.client.bodies) == 1  # rejected before any request


def test_json_fails_when_ollama_counts_more_than_the_budget():
    llm = Ollama(url="http://x", num_ctx=4096)
    llm.client = FakeClient(prompt_eval_count=4000)
    with pytest.raises(PromptTooLong):
        llm.json("short", {"type": "object"})
    llm.client = FakeClient(prompt_eval_count=100)
    assert llm.json("short", {"type": "object"}) == {"speakers": []}


def test_from_config_reads_llm_section():
    cfg = BookConfig(title="t", llm={"model": "qwen3:8b", "num_ctx": 8192})
    llm = Ollama.from_config(cfg)
    assert (llm.model, llm.num_ctx) == ("qwen3:8b", 8192)
    assert Ollama.from_config(BookConfig(title="t")).num_ctx == 16384


def test_attribute_windows_shrink_to_the_context_budget():
    cast = Cast(characters={"Ann": Character(), "Bob": Character()})
    paras = extract([f'"Line {i} of a long conversation, well over a few words." Then more.'
                     for i in range(40)], "en", cast.resolve)
    big = Ollama(url="http://x", num_ctx=16384)
    assert attribute.windows(paras, big, "") == [(0, 12), (12, 24), (24, 36), (36, 40)]
    small = Ollama(url="http://x", num_ctx=OUTPUT_RESERVE + 300)  # ~7 paragraphs after context
    ws = attribute.windows(paras, small, "")
    assert ws[0][0] == 0 and ws[-1][1] == 40
    assert all(e - s < 12 for s, e in ws) and len(ws) > 4
    assert all(ws[i][1] == ws[i + 1][0] for i in range(len(ws) - 1))  # contiguous

    with pytest.raises(PromptTooLong):
        attribute.windows(paras, small, "x" * 3000)  # fixed text alone fills the window


def test_attribute_llm_fill_uses_windows(monkeypatch):
    cast = Cast(characters={"Ann": Character(), "Bob": Character()})
    paras = extract(['"Who?"', '"Me."', '"Sure?"', '"Yes."'], "en", cast.resolve)
    llm = Ollama(url="http://x", num_ctx=16384)
    llm.client = FakeClient(response={"speakers": [
        {"quote_id": q, "speaker": "ann" if q % 2 else "Bob", "confidence": 0.9} for q in range(1, 5)]})
    stats = {"llm": 0, "unknown": 0}
    attribute._llm_fill(paras, cast, "en", llm, stats)
    assert [ps.spans[0].speaker for ps in paras] == ["Ann", "Bob", "Ann", "Bob"]
    assert stats == {"llm": 4, "unknown": 0}
    assert len(llm.client.bodies) == 1
    assert json.dumps(llm.client.bodies[0])  # serializable request
