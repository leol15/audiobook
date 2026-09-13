"""Language-aware text utilities: quote canonicalization, sentence splitting."""

from __future__ import annotations

import re

# Every opening/closing quote style we accept, mapped to straight ASCII.
_QUOTE_MAP = str.maketrans(
    {
        "“": '"',  # “
        "”": '"',  # ”
        "「": '"',  # 「
        "」": '"',  # 」
        "『": '"',  # 『
        "』": '"',  # 』
        "‘": "'",  # ‘
        "’": "'",  # ’
    }
)

_EN_ABBREV = {
    "mr", "mrs", "ms", "dr", "st", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "no", "col", "gen", "lt", "capt", "sgt", "rev", "hon", "mt", "ft",
}

# Boundary = terminator (+ optional closing quote/bracket), then whitespace,
# then something that looks like a sentence start.
_EN_BOUNDARY_RE = re.compile(r'[.!?]["\')\]]?\s+(?=["\'(\[]?[A-Z0-9])')
_ZH_BOUNDARY_RE = re.compile(r'[。！？；…]["”」』]?')


def canonicalize_quotes(text: str) -> str:
    return text.translate(_QUOTE_MAP)


def split_sentences(text: str, lang: str) -> list[str]:
    if lang == "zh":
        parts = [p.strip() for p in _split_after(text, _ZH_BOUNDARY_RE)]
        return [p for p in parts if p]
    # English: split after terminal punctuation, then re-join splits that
    # happened after a known abbreviation.
    raw = _split_after(text, _EN_BOUNDARY_RE)
    out: list[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if out:
            prev_word = out[-1].rstrip('"\')').rsplit(" ", 1)[-1].rstrip(".").lower()
            if prev_word in _EN_ABBREV:
                out[-1] = out[-1] + " " + piece
                continue
        out.append(piece)
    return out


def _split_after(text: str, boundary: re.Pattern) -> list[str]:
    """Split text after each boundary match (the match stays with the left piece)."""
    out, start = [], 0
    for m in boundary.finditer(text):
        out.append(text[start : m.end()])
        start = m.end()
    out.append(text[start:])
    return out


def chunk_text(text: str, lang: str, max_chars: int) -> list[str]:
    """Split text at sentence boundaries into pieces no longer than max_chars.

    A single sentence longer than max_chars is split at clause punctuation,
    and as a last resort at whitespace (en) or every max_chars chars (zh).
    """
    chunks: list[str] = []
    cur = ""
    joiner = "" if lang == "zh" else " "
    for sent in split_sentences(text, lang):
        for piece in _split_long(sent, lang, max_chars):
            if not cur:
                cur = piece
            elif len(cur) + len(joiner) + len(piece) <= max_chars:
                cur = cur + joiner + piece
            else:
                chunks.append(cur)
                cur = piece
    if cur:
        chunks.append(cur)
    return chunks


def _split_long(sent: str, lang: str, max_chars: int) -> list[str]:
    if len(sent) <= max_chars:
        return [sent]
    clause_re = re.compile(r"(?<=[，、；：])") if lang == "zh" else re.compile(r"(?<=[,;:])\s+")
    pieces = [p for p in clause_re.split(sent) if p.strip()]
    out: list[str] = []
    for p in pieces:
        if len(p) <= max_chars:
            out.append(p.strip())
            continue
        if lang == "zh":
            out.extend(p[i : i + max_chars] for i in range(0, len(p), max_chars))
        else:
            words = p.split()
            cur = ""
            for w in words:
                if cur and len(cur) + 1 + len(w) > max_chars:
                    out.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}" if cur else w
            if cur:
                out.append(cur)
    return out


# Rough speaking rates, used to budget batched TTS by expected audio length.
_CHARS_PER_S = {"zh": 4.0, "en": 15.0}
TOKENS_PER_S = 12  # Qwen3-TTS 12Hz codec
# Per-sequence fixed cost in a batch: the reference-clip prompt (a 12 s clip
# is ~150 codec tokens plus its text) and text prompt, attended by every step.
SEQUENCE_OVERHEAD_TOKENS = 300


def expected_tokens(text: str, lang: str) -> int:
    """Expected audio tokens for a text: seconds of speech x 12 tokens/s."""
    return int(len(text) / _CHARS_PER_S.get(lang, 8.0) * TOKENS_PER_S) + 1


def sequence_cost(text: str, lang: str) -> int:
    """What one line costs in a batch, for memory budgeting: prompt overhead
    plus expected generated tokens. 64 short lines spilled 16 GB even though
    their generated tokens summed to ~7k; the overhead is what they had in common."""
    return SEQUENCE_OVERHEAD_TOKENS + expected_tokens(text, lang)
