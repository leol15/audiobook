"""Rule-based dialogue extraction and speech-tag attribution. No model here.

Works on ingest output where all quote marks are canonical straight `"`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Span:
    kind: str            # "narration" | "dialogue"
    text: str
    speaker: str | None = None   # resolved by rules; None means "ask the LLM"
    confidence: float = 0.0
    quote_id: int | None = None  # dialogue only; stable within a chapter


@dataclass
class ParaSpans:
    para: int
    spans: list[Span] = field(default_factory=list)
    open_end: bool = False  # ends inside an open quote (speech continues next paragraph)


_QUOTE_RE = re.compile(r'"([^"]*)"')
# A quote that opens but never closes in this paragraph: dialogue continues
# into the next paragraph (standard typography for multi-paragraph speech).
_OPEN_TAIL_RE = re.compile(r'"([^"]*)$')

# Speech verbs, both languages.
_EN_VERBS = (
    r"(?:said|says|asked|asks|replied|replies|answered|cried|exclaimed|whispered|muttered|"
    r"shouted|called|continued|added|observed|remarked|returned|repeated|interrupted|"
    r"inquired|demanded|declared|began|went on|put in|murmured|laughed|sighed|snapped|"
    r"insisted|protested|suggested|agreed|admitted|announced|breathed|growled|hissed)"
)
# Up to four words of any case: "Elizabeth", "Mr. Bennet", "his lady". The
# cast lookup tries shorter prefixes, so trailing words are harmless.
_NAME = r"((?:[\w'\-]+\.?\s?){1,4})"
_PRONOUN = r"(?:he|she|they|I)"

# "..." said Elizabeth  /  "..." Elizabeth said  /  "..." she said (unresolvable -> None)
_EN_AFTER = [
    re.compile(rf"^\s*[,;:\-—]?\s*{_EN_VERBS}\s+{_NAME}"),
    re.compile(rf"^\s*[,;:\-—]?\s*{_NAME}\s*{_EN_VERBS}"),
]
# Elizabeth said, "..."  /  said Elizabeth: "..."
_EN_BEFORE = [
    re.compile(rf"{_NAME}\s*{_EN_VERBS}[^\"]{{0,40}}$"),
    re.compile(rf"{_EN_VERBS}\s+{_NAME}[^\"]{{0,40}}$"),
]
_EN_PRONOUN_TAG = re.compile(rf"^\s*[,;:\-—]?\s*(?:{_EN_VERBS}\s+{_PRONOUN}|{_PRONOUN}\s+{_EN_VERBS})\b", re.IGNORECASE)

_ZH_VERBS = r"(?:说|道|问|答|答道|说道|问道|叫道|喊道|笑道|叹道|回答|接着说|继续说|低声说|大声说|开口)"
_ZH_NAME = r"([一-鿿·]{1,6}?)"
_ZH_BEFORE = [re.compile(rf"{_ZH_NAME}{_ZH_VERBS}[：:，,]?\s*$")]
_ZH_AFTER = [re.compile(rf"^\s*[，,]?\s*{_ZH_NAME}{_ZH_VERBS}")]

_STOP = {"the", "a", "and", "but", "then", "there", "it", "that", "this"}


def extract(paragraphs: list[str], lang: str, resolve) -> list[ParaSpans]:
    """Split paragraphs into spans and attribute what the rules can.

    `resolve(name) -> canonical | None` maps a surface name to the cast.
    """
    out: list[ParaSpans] = []
    qid = 0
    carry_open = False  # previous paragraph ended inside an open quote
    for pi, para in enumerate(paragraphs):
        ps = ParaSpans(para=pi)
        text = para
        if carry_open:
            # Continued speech: the paragraph re-opens with `"` (standard
            # typography) and runs to the next `"`.
            text = text.removeprefix('"')
            m = re.match(r'^([^"]*)"?', text)
            lead = m.group(1).strip()
            if lead:
                qid += 1
                ps.spans.append(Span("dialogue", lead, quote_id=qid))
            text = text[m.end():]
            carry_open = False
        pos = 0
        for m in _QUOTE_RE.finditer(text):
            before = text[pos:m.start()].strip()
            if before:
                ps.spans.append(Span("narration", before, "narrator", 1.0))
            qid += 1
            inner = m.group(1).strip()
            if inner:
                ps.spans.append(Span("dialogue", inner, quote_id=qid))
            pos = m.end()
        tail = text[pos:]
        om = _OPEN_TAIL_RE.search(tail)
        if om:
            before = tail[:om.start()].strip()
            if before:
                ps.spans.append(Span("narration", before, "narrator", 1.0))
            qid += 1
            ps.spans.append(Span("dialogue", om.group(1).strip(), quote_id=qid))
            carry_open = ps.open_end = True
        elif tail.strip():
            ps.spans.append(Span("narration", tail.strip(), "narrator", 1.0))
        _attribute_rules(ps, lang, resolve)
        out.append(ps)
    _propagate_continuations(out)
    return out


def _attribute_rules(ps: ParaSpans, lang: str, resolve) -> None:
    spans = ps.spans
    after_pats, before_pats = (_ZH_AFTER, _ZH_BEFORE) if lang == "zh" else (_EN_AFTER, _EN_BEFORE)
    for i, sp in enumerate(spans):
        if sp.kind != "dialogue":
            continue
        name = None
        if i + 1 < len(spans) and spans[i + 1].kind == "narration":
            name = _match_name(after_pats, spans[i + 1].text, resolve)
        if name is None and i > 0 and spans[i - 1].kind == "narration":
            name = _match_name(before_pats, spans[i - 1].text, resolve)
        if name is not None:
            sp.speaker, sp.confidence = name, 0.95
    # Two quotes in one paragraph split by a short narration with a resolved tag:
    # the second quote is the same speaker ("...," said X, "...").
    for i in range(2, len(spans)):
        a, mid, b = spans[i - 2], spans[i - 1], spans[i]
        if a.kind == b.kind == "dialogue" and mid.kind == "narration" and len(mid.text) < 60:
            if a.speaker and not b.speaker:
                b.speaker, b.confidence = a.speaker, 0.9
            elif b.speaker and not a.speaker:
                a.speaker, a.confidence = b.speaker, 0.9


def _match_name(patterns, text: str, resolve) -> str | None:
    for pat in patterns:
        m = pat.search(text)
        if not m:
            continue
        for g in m.groups():
            if not g:
                continue
            cand = g.strip().rstrip(".,;:")
            if cand.lower() in _STOP:
                continue
            canon = resolve(cand)
            if canon:
                return canon
            # Try progressively shorter prefixes: "Elizabeth with a smile" -> "Elizabeth"
            words = cand.split()
            for n in range(len(words) - 1, 0, -1):
                canon = resolve(" ".join(words[:n]))
                if canon:
                    return canon
    return None


def _propagate_continuations(paras: list[ParaSpans]) -> None:
    """A paragraph that is a single unattributed quote following a paragraph
    that ended with an open quote (multi-paragraph speech) keeps the speaker."""
    prev_speaker = None
    prev_open = False
    for ps in paras:
        first = ps.spans[0] if ps.spans else None
        if prev_open and prev_speaker and first and first.kind == "dialogue" and not first.speaker:
            first.speaker, first.confidence = prev_speaker, 0.85
        last = ps.spans[-1] if ps.spans else None
        prev_open = ps.open_end
        prev_speaker = last.speaker if last and last.kind == "dialogue" else None
