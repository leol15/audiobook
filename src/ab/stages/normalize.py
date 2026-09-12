"""Stage 2: chapters.json -> chapters.norm.json (text TTS models can read aloud)."""

from __future__ import annotations

import re

from ab import cache
from ab.config import BookConfig, BookPaths
from ab.models import ChapterList

# Titles (Mr., Mrs., Dr., St.) are left alone: Kokoro's G2P reads them correctly,
# and expanding them before attribution breaks name matching against the cast.
_EN_ABBREV = {
    r"\betc\.": "et cetera",
    r"\be\.g\.": "for example",
    r"\bi\.e\.": "that is",
}
_ZH_DIGITS = "零一二三四五六七八九"


def run(paths: BookPaths, cfg: BookConfig, force: bool = False):
    overrides = paths.load_overrides()
    inputs = cache.content_hash("normalize", cache.file_hash(paths.chapters), cfg.language, overrides)
    if not force and cache.is_fresh(paths.chapters_norm, inputs):
        return paths.chapters_norm
    book = ChapterList.model_validate_json(paths.chapters.read_text(encoding="utf-8"))
    for ch in book.chapters:
        ch.title = normalize(ch.title, cfg.language, overrides)
        ch.paragraphs = [normalize(p, cfg.language, overrides) for p in ch.paragraphs]
    paths.chapters_norm.write_text(book.model_dump_json(indent=1), encoding="utf-8")
    cache.mark_fresh(paths.chapters_norm, inputs)
    return paths.chapters_norm


def normalize(text: str, lang: str, overrides: dict[str, str] | None = None) -> str:
    for k, v in (overrides or {}).items():
        text = text.replace(k, v)
    return _normalize_zh(text) if lang == "zh" else _normalize_en(text)


def _normalize_en(text: str) -> str:
    for pat, rep in _EN_ABBREV.items():
        text = re.sub(pat, rep, text)
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    text = re.sub(r"\s*,\s*,", ",", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_zh(text: str) -> str:
    # Years: 2024年 -> 二零二四年 (digit by digit).
    text = re.sub(r"(\d{4})年", lambda m: "".join(_ZH_DIGITS[int(c)] for c in m.group(1)) + "年", text)
    # Other integers: read as a number. Small implementation, good enough for v1.
    text = re.sub(r"\d+", lambda m: _zh_number(int(m.group(0))), text)
    text = text.replace("——", "，").replace("…", "，")
    return text.strip()


def _zh_number(n: int) -> str:
    if n == 0:
        return "零"
    if n >= 100_000_000:
        return str(n)  # leave very large numbers alone
    units = ["", "十", "百", "千"]
    big = ["", "万"]
    out = ""
    group_i = 0
    while n > 0:
        group, n = n % 10000, n // 10000
        if group:
            s = ""
            zero_pending = False
            for i in range(3, -1, -1):
                d = (group // 10**i) % 10
                if d:
                    if zero_pending:
                        s += "零"
                        zero_pending = False
                    s += _ZH_DIGITS[d] + units[i]
                elif s:
                    zero_pending = True
            if s.startswith("一十"):
                s = s[1:]
            out = s + big[group_i] + out
        group_i += 1
    return out
