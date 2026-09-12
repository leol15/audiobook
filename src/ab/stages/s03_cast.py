"""Stage 3: chapters.norm.json -> cast.yaml via the local LLM.

Never overwrites an existing cast.yaml (the user edits it to assign voices);
use --force to regenerate. Three passes, sized for a whole novel:

1. Discover, per chapter: who speaks, aliases, a one-line description. Each
   chapter's result is cached in work/03-cast/ keyed by the chapter text, so
   a --force after editing one chapter only re-asks about that chapter.
2. Merge, incrementally: a few chapters' entries at a time are matched
   against the running cast (existing name or new person), so the cast never
   has to fit one prompt.
3. Rank, no model: rule-based speech-tag attribution over the whole book
   counts dialogue lines per character; the top `cast.main_cap` are the main
   cast (`main: true`), the rest are listed to the attribute LLM only in the
   chapters where they appear and are voiced by `_default`.

The LLM returns names only; no book text is rewritten.
"""

from __future__ import annotations

import json

import yaml
from rich.progress import track

from ab import cache
from ab.config import BookConfig, BookPaths, Cast, Character, _fold
from ab.llm import Ollama, approx_tokens, pack
from ab.log import note
from ab.models import Chapter, ChapterList
from ab.quotes import extract

_DISCOVER_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["name", "aliases", "description"],
            },
        }
    },
    "required": ["characters"],
}
_MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["canonical", "aliases", "description"],
            },
        }
    },
    "required": ["groups"],
}

_SYSTEM = {
    "en": "You analyse fiction for audiobook production. Answer only with JSON.",
    "zh": "你在为有声书制作分析小说。只用JSON回答。",
}
_DISCOVER = {
    "en": (
        "List every character who SPEAKS dialogue in this chapter. For each give the name "
        "as most commonly written, other names/titles used for the same person, and a short "
        "description (role, gender, age, manner of speaking). Do not include characters who "
        "are only mentioned.\n\n{text}"
    ),
    "zh": (
        "列出本章中所有说过话的角色。对每个角色给出最常用的名字、指同一人的其他称呼、"
        "以及简短描述（身份、性别、年龄、说话方式）。不要包含只被提及而没有说话的角色。\n\n{text}"
    ),
}
_MERGE = {
    "en": (
        "Known cast of a novel so far (canonical name, other names):\n{known}\n\n"
        "New character entries collected from the next chapters of the same novel:\n{entries}\n\n"
        "Decide for each new entry whether it is one of the known characters or a new person, "
        "and merge new entries that refer to the same person. Return one group per distinct "
        "person among the new entries. If the person is already known, use the known "
        "canonical name exactly; otherwise choose the most common short name. Keep distinct "
        "people separate even if they share a surname or title."
    ),
    "zh": (
        "这部小说目前已知的角色（规范名，其他称呼）：\n{known}\n\n"
        "从接下来几章中收集到的新角色条目：\n{entries}\n\n"
        "请判断每个新条目是已知角色还是新人物，并合并指同一人物的新条目。"
        "对新条目中的每个不同人物返回一组：如果是已知角色，规范名必须与已知规范名完全一致；"
        "否则选择最常用的简称。不同的人物即使同姓或同称谓也要分开。"
    ),
}
_NONE = {"en": "(none yet)", "zh": "（暂无）"}


def run(paths: BookPaths, cfg: BookConfig, force: bool = False):
    if paths.cast.exists() and not force:
        return paths.cast
    lang = cfg.language
    llm = Ollama.from_config(cfg, log=paths.llm_log)
    book = ChapterList.model_validate_json(paths.chapters_norm.read_text(encoding="utf-8"))

    per_chapter = [discover(paths, llm, ch, lang) for ch in
                   track(book.chapters, description="cast: discover")]
    cast = Cast()
    merges = 0
    for batch in _merge_batches(book.chapters, per_chapter, llm, lang, cfg.cast.merge_chapters):
        merge(cast, batch, llm, lang)
        merges += 1
    rank(cast, book.chapters, lang, cfg.cast.main_cap)
    write_cast(paths, cast)
    main = sum(1 for c in cast.characters.values() if c.main)
    note(paths, f"cast: {len(cast.characters)} characters ({main} main) from {len(book.chapters)} "
         f"chapters in {merges} merge calls -> {paths.cast}")
    llm.unload()
    return paths.cast


# ---------------------------------------------------------------- 1. discover

def discover(paths: BookPaths, llm: Ollama, ch: Chapter, lang: str) -> list[dict]:
    """Characters speaking in one chapter, from the cache if its text is unchanged."""
    key = cache.content_hash("discover", lang, ch.paragraphs)
    f = paths.cast_work / f"c{ch.index:03d}.json"
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("hash") == key:
            return data["characters"]
    found: list[dict] = []
    budget = llm.window_budget(_DISCOVER[lang].format(text="") + _SYSTEM[lang])
    # Windows of whole paragraphs sized to the context budget, not a fixed
    # character count: 12k Chinese characters is ~8k tokens, twice Ollama's
    # default window, and used to be truncated silently.
    for window in pack(ch.paragraphs, budget):
        res = llm.json(_DISCOVER[lang].format(text="\n\n".join(window)), _DISCOVER_SCHEMA,
                       system=_SYSTEM[lang])
        found.extend(res.get("characters", []))
    entries = _dedupe(found)
    paths.cast_work.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"hash": key, "chapter": ch.index, "characters": entries},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return entries


def _dedupe(found: list[dict]) -> list[dict]:
    """Exact-name dedupe within one chapter's windows (cheap, before the model merges)."""
    by_name: dict[str, dict] = {}
    for c in found:
        n = str(c.get("name", "")).strip()
        if not n:
            continue
        e = by_name.setdefault(n, {"name": n, "aliases": [], "description": str(c.get("description", "")).strip()})
        for a in c.get("aliases", []):
            a = str(a).strip()
            if a and a != n and a not in e["aliases"]:
                e["aliases"].append(a)
    return list(by_name.values())


# ---------------------------------------------------------------- 2. merge

def _merge_batches(chapters, per_chapter, llm: Ollama, lang: str, max_chapters: int):
    """Yield lists of (chapter index, entries) that fit one merge prompt."""
    overhead = _MERGE[lang].format(known="", entries="") + _SYSTEM[lang]
    budget = llm.window_budget(overhead) // 2  # the other half is for the running cast
    batch: list[tuple[int, list[dict]]] = []
    used = 0
    for ch, entries in zip(chapters, per_chapter):
        if not entries:
            continue
        t = sum(approx_tokens(_entry_line(e)) + 1 for e in entries)
        if batch and (used + t > budget or len(batch) >= max_chapters):
            yield batch
            batch, used = [], 0
        batch.append((ch.index, entries))
        used += t
    if batch:
        yield batch


def _entry_line(e: dict) -> str:
    aka = ", ".join(e.get("aliases", [])) or "-"
    return f"- {e['name']} (aka {aka}): {e.get('description', '')}"


def merge(cast: Cast, batch: list[tuple[int, list[dict]]], llm: Ollama, lang: str) -> None:
    """Fold one batch of chapters' entries into the running cast with one LLM call."""
    # Which chapters each surface name came from, so merged entries keep their chapters.
    where: dict[str, set[int]] = {}
    seen: dict[str, dict] = {}
    for ci, entries in batch:
        for e in entries:
            for n in [e["name"], *e.get("aliases", [])]:
                where.setdefault(_fold(n), set()).add(ci)
            merged = seen.setdefault(e["name"], {"name": e["name"], "aliases": [], "description": e["description"]})
            for a in e.get("aliases", []):
                if a not in merged["aliases"] and a != e["name"]:
                    merged["aliases"].append(a)

    # Entries that already resolve to a known character by name need no model.
    direct, ask = [], []
    for e in seen.values():
        known = cast.resolve(e["name"])
        (direct if known else ask).append((known, e))
    for known, e in direct:
        _fold_into(cast, known, [e["name"], *e["aliases"]], e["description"], where)

    if ask:
        known_txt = "\n".join(f"- {n}" + (f" ({', '.join(c.aliases)})" if c.aliases else "")
                              for n, c in cast.characters.items()) or _NONE[lang]
        entries_txt = "\n".join(_entry_line(e) for _, e in ask)
        res = llm.json(_MERGE[lang].format(known=known_txt, entries=entries_txt), _MERGE_SCHEMA,
                       system=_SYSTEM[lang])
        covered: set[str] = set()
        for g in res.get("groups", []):
            canon = str(g.get("canonical", "")).strip()
            if not canon:
                continue
            names = [canon, *(str(a).strip() for a in g.get("aliases", []) if str(a).strip())]
            target = next((t for t in (cast.resolve(n) for n in names) if t), None)
            if target is None:
                target = canon
                cast.characters[target] = Character(description=str(g.get("description", "")).strip())
            _fold_into(cast, target, names, str(g.get("description", "")).strip(), where)
            covered.update(_fold(n) for n in names)
        # Anything the model dropped is kept as its own character rather than lost.
        for _, e in ask:
            if _fold(e["name"]) not in covered and not cast.resolve(e["name"]):
                cast.characters[e["name"]] = Character(description=e["description"])
                _fold_into(cast, e["name"], [e["name"], *e["aliases"]], e["description"], where)


def _fold_into(cast: Cast, target: str, names: list[str], description: str, where: dict) -> None:
    c = cast.characters[target]
    for n in names:
        if n and n != target and _fold(n) != _fold(target) and n not in c.aliases:
            c.aliases.append(n)
        for ci in where.get(_fold(n), ()):
            if ci not in c.chapters:
                c.chapters.append(ci)
    if description and not c.description:
        c.description = description
    c.chapters.sort()


# ---------------------------------------------------------------- 3. rank

def rank(cast: Cast, chapters: list[Chapter], lang: str, main_cap: int) -> None:
    """Count rule-attributed dialogue lines and name mentions per character (no
    model), mark the top `main_cap` as main, and order cast.yaml by rank."""
    lines: dict[str, int] = {n: 0 for n in cast.characters}
    mentions: dict[str, int] = {n: 0 for n in cast.characters}
    for ch in chapters:
        text = "\n".join(ch.paragraphs)
        for n, c in cast.characters.items():
            mentions[n] += sum(text.count(a) for a in {n, *c.aliases} if a)
        for ps in extract(ch.paragraphs, lang, cast.resolve):
            for sp in ps.spans:
                if sp.kind == "dialogue" and sp.speaker in lines:
                    lines[sp.speaker] += 1
                    if ch.index not in cast.characters[sp.speaker].chapters:
                        cast.characters[sp.speaker].chapters.append(ch.index)
    order = sorted(cast.characters, key=lambda n: (-lines[n], -mentions[n], n))
    ranked = {}
    for i, n in enumerate(order):
        c = cast.characters[n]
        c.lines, c.main = lines[n], i < main_cap
        c.chapters.sort()
        ranked[n] = c
    cast.characters = ranked


def write_cast(paths: BookPaths, cast: Cast) -> None:
    data = {"characters": {n: c.model_dump() for n, c in cast.characters.items()}}
    paths.cast.write_text(
        "# Generated by `ab cast`. Edit freely; assign voices in book.yaml under `voices:`.\n"
        "# main: true = main cast (in every attribute prompt; give it a voice), others use _default.\n"
        "# lines = dialogue lines resolved by rules at cast time (a ranking hint, not a total).\n"
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
