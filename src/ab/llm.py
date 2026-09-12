"""Thin Ollama client. Ollama runs on Windows; reached over HTTP (see DESIGN.md).

Context window: Ollama's default `num_ctx` is 4096 and a prompt longer than
that is truncated silently (the model just never sees the start). Every call
therefore sends `options.num_ctx` and checks the prompt against a token budget
before and after the request, so an oversized prompt is an error, not a
quietly wrong answer. Stages size their windows from `Ollama.prompt_budget`.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import httpx

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
DEFAULT_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
OUTPUT_RESERVE = 2048  # tokens of the window kept free for the model's JSON answer

_CJK = re.compile(r"[　-鿿豈-﫿＀-￯]")


def _load_dotenv() -> None:
    """Read OLLAMA_* from a .env in the cwd or its parents, without overriding the environment."""
    for d in [Path.cwd(), *Path.cwd().parents]:
        f = d / ".env"
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return


_load_dotenv()


class PromptTooLong(RuntimeError):
    """The prompt does not fit the context window; the caller must shrink its window."""


def approx_tokens(text: str) -> int:
    """Upper estimate of Qwen tokens: one per CJK character, one per three other characters.

    Qwen's tokenizer averages ~1.5 characters per token on Chinese and ~4 on
    English, so this overestimates by roughly a third, which is the safety
    margin against truncation.
    """
    cjk = len(_CJK.findall(text))
    return cjk + math.ceil((len(text) - cjk) / 3)


def pack(items: list[str], budget: int, sep: str = "\n\n", max_items: int | None = None) -> list[list[str]]:
    """Group consecutive items so each group's text stays within `budget` tokens.

    An item that alone exceeds the budget still gets its own group; the check
    in `Ollama.json` then fails loudly rather than truncating.
    """
    groups: list[list[str]] = []
    cur: list[str] = []
    used = 0
    sep_t = approx_tokens(sep)
    for it in items:
        t = approx_tokens(it)
        if cur and (used + sep_t + t > budget or (max_items and len(cur) >= max_items)):
            groups.append(cur)
            cur, used = [], 0
        cur.append(it)
        used += t + (sep_t if len(cur) > 1 else 0)
    if cur:
        groups.append(cur)
    return groups


class Ollama:
    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = 300,
                 log: Path | None = None, num_ctx: int = DEFAULT_NUM_CTX):
        self.url = (url or os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.num_ctx = num_ctx
        self.client = httpx.Client(base_url=self.url, timeout=timeout)
        self.log = log  # every exchange appended as one JSON line, for debugging

    @classmethod
    def from_config(cls, cfg, log: Path | None = None) -> Ollama:
        """Build from book.yaml's `llm:` section (model, num_ctx)."""
        return cls(model=cfg.llm.model, num_ctx=cfg.llm.num_ctx, log=log)

    @property
    def prompt_budget(self) -> int:
        """Tokens a prompt (system + user) may use, leaving room for the answer."""
        return self.num_ctx - OUTPUT_RESERVE

    def window_budget(self, overhead: str) -> int:
        """Tokens left for variable content once a fixed prompt part is accounted for."""
        b = self.prompt_budget - approx_tokens(overhead)
        if b < 256:
            raise PromptTooLong(f"fixed prompt text ({approx_tokens(overhead)} tokens) leaves no room "
                                f"in a {self.num_ctx}-token window; raise llm.num_ctx")
        return b

    def check(self, prompt: str, system: str = "") -> int:
        """Estimated prompt tokens; raises PromptTooLong if over budget."""
        need = approx_tokens(system) + approx_tokens(prompt)
        if need > self.prompt_budget:
            raise PromptTooLong(f"prompt is ~{need} tokens but the budget is {self.prompt_budget} "
                                f"(num_ctx {self.num_ctx} minus {OUTPUT_RESERVE} for the answer); "
                                "raise llm.num_ctx in book.yaml or shrink the window")
        return need

    def ping(self) -> list[str]:
        r = self.client.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def json(self, prompt: str, schema: dict, system: str = "", retries: int = 2) -> dict:
        """Generate a JSON object conforming to schema (Ollama structured output)."""
        self.check(prompt, system)
        body = {"model": self.model, "prompt": prompt, "system": system, "format": schema,
                "stream": False, "options": {"temperature": 0, "num_ctx": self.num_ctx}}
        last: Exception | None = None
        for _ in range(retries + 1):
            r = self.client.post("/api/generate", json=body)
            r.raise_for_status()
            data = r.json()
            raw = data.get("response", "")
            used = data.get("prompt_eval_count")
            if self.log:
                with self.log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"model": self.model, "num_ctx": self.num_ctx,
                                        "prompt_tokens": used, "prompt": prompt, "response": raw},
                                       ensure_ascii=False) + "\n")
            # Ollama reports how many prompt tokens it evaluated (fewer when a
            # prefix was cached, never more): a count over the budget means the
            # estimate was wrong and the answer may have been squeezed or cut.
            if used is not None and used > self.prompt_budget:
                raise PromptTooLong(f"ollama evaluated {used} prompt tokens, over the budget of "
                                    f"{self.prompt_budget} (num_ctx {self.num_ctx})")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                last = e
        raise RuntimeError(f"ollama returned non-JSON after retries: {last}")

    def unload(self) -> None:
        """Free VRAM before a large TTS model loads."""
        self.client.post("/api/generate", json={"model": self.model, "keep_alive": 0})
