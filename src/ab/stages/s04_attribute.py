"""Stage 4: chapters.norm.json + cast.yaml -> lines.jsonl

1. Rules split each paragraph into narration/dialogue spans and resolve speech
   tags ("said X") against the cast.
2. Unresolved quotes go to the LLM in windows, which returns speaker names for
   quote ids only. Text never round-trips through the model.
3. Lines marked locked: true in an existing chapter file are preserved.

Chapter-granular: 04-lines/cNNN.jsonl is stamped with the chapter text, the
language, and the cast's names and aliases, so editing one chapter (or the
cast) re-runs only what depends on it.

Without a cast.yaml the stage runs in narrator-only mode (milestone 1 behaviour).
"""

from __future__ import annotations

from rich.progress import track

from ab import cache
from ab.config import BookConfig, BookPaths, Cast
from ab.lines import chapter_indexes, read_attribution, read_lines, write_attribution, write_views
from ab.log import note
from ab.models import Chapter, ChapterList, Line
from ab.quotes import ParaSpans, extract

_SCHEMA = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote_id": {"type": "integer"},
                    "speaker": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["quote_id", "speaker", "confidence"],
            },
        }
    },
    "required": ["speakers"],
}
_SYSTEM = {
    "en": "You attribute dialogue in fiction to speakers for audiobook narration. Answer only with JSON.",
    "zh": "你为有声书朗读判断小说中每句对话的说话人。只用JSON回答。",
}
_PROMPT = {
    "en": (
        "Cast (use these exact names; use \"unknown\" if none fits):\n{cast}\n\n"
        "Passage. Quotes are marked [q<id>]. Some already have a known speaker in "
        "parentheses; determine the speaker of every quote marked (?). Consider who is "
        "present, alternation between speakers in a conversation, and who is being addressed.\n\n"
        "{passage}\n\nReturn the speaker of each (?) quote with a confidence from 0 to 1."
    ),
    "zh": (
        "角色表（只能使用这些名字；都不合适时用 \"unknown\"）：\n{cast}\n\n"
        "段落。对话用 [q<id>] 标记。有些已在括号中给出说话人；请判断每个标记为 (?) 的对话的说话人。"
        "考虑在场的人物、对话中的轮流发言、以及被称呼的对象。\n\n"
        "{passage}\n\n返回每个 (?) 对话的说话人和 0 到 1 的置信度。"
    ),
}
import re

_HAS_WORD = re.compile(r"\w")
WINDOW = 12   # max paragraphs per LLM call (fewer if the context budget fills first)
CONTEXT = 4   # preceding paragraphs shown for context


def cast_hash(cast: Cast) -> str:
    """Names and aliases only: a description edit must not re-run attribution."""
    return cache.content_hash({n: c.aliases for n, c in cast.characters.items()})


def chapter_inputs(ch: Chapter, cfg: BookConfig, cast_key: str) -> dict:
    return {"chapter": cache.content_hash(ch.title, ch.paragraphs), "language": cfg.language,
            "cast": cast_key}


def chapter_states(paths: BookPaths, cfg: BookConfig) -> list[tuple[int, list[str]]]:
    """(chapter index, stale reasons) per chapter of the normalized book; [] = fresh."""
    if not paths.chapters_norm.exists():
        return []
    book = ChapterList.model_validate_json(paths.chapters_norm.read_text(encoding="utf-8"))
    key = cast_hash(paths.load_cast())
    return [(ch.index, cache.stale_reasons(paths.chapter_lines(ch.index), chapter_inputs(ch, cfg, key)))
            for ch in book.chapters]


def run(paths: BookPaths, cfg: BookConfig, force: bool = False, chapters: set[int] | None = None):
    cast = paths.load_cast()
    key = cast_hash(cast)
    book = ChapterList.model_validate_json(paths.chapters_norm.read_text(encoding="utf-8"))
    todo = [ch for ch in book.chapters
            if (chapters is None or ch.index in chapters)
            and (force or not cache.is_fresh(paths.chapter_lines(ch.index), chapter_inputs(ch, cfg, key)))]
    # A chapter file the source no longer has (book shrank) would otherwise linger.
    for i in chapter_indexes(paths):
        if i >= len(book.chapters):
            paths.chapter_lines(i).unlink()
            cache.stamp_path(paths.chapter_lines(i)).unlink(missing_ok=True)
    if not todo:
        return paths.lines_dir

    stats = {"narration": 0, "rules": 0, "llm": 0, "unknown": 0, "locked": 0}
    llm = None
    if cast.characters:
        from ab.llm import Ollama
        llm = Ollama.from_config(cfg, log=paths.llm_log)

    for ch in track(todo, description="attribute"):
        locked = {ln.id: ln for ln in read_attribution(paths, ch.index) if ln.locked}
        lines = attribute_chapter(ch, cast, cfg, llm, stats, locked)
        write_attribution(paths, ch.index, lines)
        cache.mark_fresh(paths.chapter_lines(ch.index), chapter_inputs(ch, cfg, key))
    all_lines = read_lines(paths)
    write_views(paths, all_lines)
    _write_review(paths, all_lines)
    if llm:
        llm.unload()
    dialogue = sum(1 for ln in all_lines if ln.kind == "dialogue")
    note(paths, f"attribute: {len(todo)}/{len(book.chapters)} chapters run; {stats['narration']} "
         f"narration, {dialogue} dialogue in book ({stats['rules']} by rules, {stats['llm']} by LLM, "
         f"{stats['unknown']} unknown, {stats['locked']} locked in re-run chapters)")
    return paths.lines_dir


def attribute_chapter(ch: Chapter, cast: Cast, cfg: BookConfig, llm, stats: dict,
                      locked: dict[str, Line]) -> list[Line]:
    if not cast.characters:
        paras = [ParaSpans(para=i, spans=[_narr(p)]) for i, p in enumerate(ch.paragraphs)]
    else:
        paras = extract(ch.paragraphs, cfg.language, cast.resolve)
        _llm_fill(paras, cast, cfg.language, llm, stats, names=cast.names_for_chapter(ch.index))
    lines: list[Line] = []
    for ps in paras:
        for si, sp in enumerate(ps.spans):
            if not _HAS_WORD.search(sp.text):
                continue  # punctuation-only span such as a quoted "……"
            lid = f"c{ch.index:03d}p{ps.para:04d}s{si:02d}"
            if lid in locked and locked[lid].text == sp.text:
                lines.append(locked[lid])
                stats["locked"] += 1
                continue
            speaker = sp.speaker or "unknown"
            if sp.kind == "narration":
                stats["narration"] += 1
            elif sp.confidence >= 0.85:
                stats["rules"] += 1
            lines.append(Line(id=lid, chapter=ch.index, para=ps.para, kind=sp.kind,
                              speaker=speaker, text=sp.text, lang=cfg.language,
                              confidence=sp.confidence))
    return lines


def _narr(text: str):
    from ab.quotes import Span
    return Span("narration", text, "narrator", 1.0)


def _llm_fill(paras: list[ParaSpans], cast: Cast, lang: str, llm, stats: dict,
              names: list[str] | None = None) -> None:
    """Ask the LLM for the speakers the rules left open. The prompt lists only
    `names` (main cast plus this chapter's characters; default all), but the
    answer is resolved against the whole cast."""
    names = list(cast.characters) if names is None else names
    cast_txt = "\n".join(f"- {n}" + (f" ({', '.join(cast.characters[n].aliases)})"
                                    if cast.characters[n].aliases else "") for n in names)
    for start, end in windows(paras, llm, _PROMPT[lang].format(cast=cast_txt, passage="")
                              + _SYSTEM[lang]):
        window = paras[start:end]
        pending = {sp.quote_id: sp for ps in window for sp in ps.spans
                   if sp.kind == "dialogue" and not sp.speaker}
        if not pending:
            continue
        ctx = paras[max(0, start - CONTEXT) : start]
        passage = "\n\n".join(_render_para(ps) for ps in ctx + window)
        res = llm.json(_PROMPT[lang].format(cast=cast_txt, passage=passage), _SCHEMA,
                       system=_SYSTEM[lang])
        for item in res.get("speakers", []):
            sp = pending.get(item.get("quote_id"))
            if sp is None:
                continue
            canon = cast.resolve(str(item.get("speaker", "")).strip())
            conf = float(item.get("confidence", 0.5))
            if canon:
                sp.speaker, sp.confidence = canon, min(conf, 0.8)
                stats["llm"] += 1
        for sp in pending.values():
            if not sp.speaker:
                sp.speaker, sp.confidence = "unknown", 0.0
                stats["unknown"] += 1


def windows(paras: list[ParaSpans], llm, overhead: str) -> list[tuple[int, int]]:
    """(start, end) paragraph ranges per LLM call: at most WINDOW paragraphs, and
    never more than the context budget allows once CONTEXT preceding paragraphs
    are included. A single paragraph over budget is still sent alone, so that
    `Ollama.json` fails loudly instead of Ollama truncating."""
    from ab.llm import approx_tokens

    budget = llm.window_budget(overhead)
    sizes = [approx_tokens(_render_para(ps)) + 2 for ps in paras]
    out: list[tuple[int, int]] = []
    start = 0
    while start < len(paras):
        used = sum(sizes[max(0, start - CONTEXT):start])
        end = start
        while end < len(paras) and end - start < WINDOW and (end == start or used + sizes[end] <= budget):
            used += sizes[end]
            end += 1
        out.append((start, end))
        start = end
    return out


def _render_para(ps: ParaSpans) -> str:
    parts = []
    for sp in ps.spans:
        if sp.kind == "dialogue":
            tag = f"({sp.speaker})" if sp.speaker else "(?)"
            parts.append(f'[q{sp.quote_id}]{tag} "{sp.text}"')
        else:
            parts.append(sp.text)
    return " ".join(parts)


def _write_review(paths: BookPaths, lines: list[Line]) -> None:
    low = [ln for ln in lines if ln.kind == "dialogue" and ln.confidence < 0.85]
    f = paths.attribute_review
    with f.open("w", encoding="utf-8") as fh:
        fh.write("# Dialogue lines not resolved by rules. Fix with: ab fix <book> <id> --speaker NAME\n")
        for ln in low:
            fh.write(f"{ln.id}\t{ln.speaker}\t{ln.confidence:.2f}\t{ln.text[:100]}\n")
