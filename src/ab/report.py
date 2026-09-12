"""Human-facing views of the work directory: script, status, report, fix."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from ab import cache
from ab.config import BookConfig, BookPaths
from ab.lines import chapter_indexes, read_chapter, read_lines, verify_failures
from ab.models import ChapterList, Line

# ---------------------------------------------------------------- script

def write_script(paths: BookPaths, lines: list[Line]) -> Path:
    """04-script.md: the book as a screenplay, one line per utterance.

    Markers:  (?) model-attributed (confidence < 0.85)   (!) unknown speaker
              [x] verify failed (error above threshold)  [lock] hand-fixed
    """
    titles = {}
    if paths.chapters_norm.exists():
        book = ChapterList.model_validate_json(paths.chapters_norm.read_text(encoding="utf-8"))
        titles = {c.index: c.title for c in book.chapters}
    verify_bad = verify_failures(paths)
    out = ["# Script", "",
           "Markers: `(?)` model-attributed · `(!)` unknown speaker · `[x]` verify failed · `[lock]` hand-fixed",
           "", "Fix a line with `ab fix <book> <id> --speaker NAME`.", ""]
    chapter = None
    for ln in lines:
        if ln.chapter != chapter:
            chapter = ln.chapter
            out += ["", f"## {titles.get(chapter, f'Chapter {chapter + 1}')}", ""]
        marks = []
        if ln.speaker == "unknown":
            marks.append("(!)")
        elif ln.kind == "dialogue" and ln.confidence < 0.85:
            marks.append("(?)")
        if ln.id in verify_bad:
            marks.append("[x]")
        if ln.locked:
            marks.append("[lock]")
        tag = "narrator" if ln.kind == "narration" else ln.speaker
        mark = (" " + " ".join(marks)) if marks else ""
        out.append(f"`{ln.id}` **{tag}**{mark}: {ln.text}")
    paths.script.write_text("\n".join(out) + "\n", encoding="utf-8")
    return paths.script


# ---------------------------------------------------------------- status

def chapter_status(paths: BookPaths, cfg: BookConfig) -> list[dict]:
    """One row per chapter: title, line count, and fresh/stale/missing per chapter stage."""
    from ab.stages import s04_attribute as attribute
    from ab.stages import s05_render as render
    from ab.stages import s06_verify as verify
    from ab.stages import s07_build as build

    if not paths.chapters_norm.exists():
        return []
    book = ChapterList.model_validate_json(paths.chapters_norm.read_text(encoding="utf-8"))
    rows = {ch.index: {"chapter": ch.index, "title": ch.title, "lines": 0} for ch in book.chapters}

    def fill(stage, states):
        for i, reasons in states:
            if i not in rows:
                continue
            if reasons == ["missing"]:
                state, detail = "missing", ""
            elif reasons:
                state, detail = "stale", ", ".join(reasons)
            else:
                state, detail = "fresh", ""
            rows[i][stage] = state
            rows[i][stage + "_detail"] = detail

    fill("attribute", attribute.chapter_states(paths, cfg))
    fill("render", render.chapter_states(paths, cfg, _backend_name(cfg.tts.backend)))
    fill("verify", verify.chapter_states(paths, cfg))
    fill("build", build.chapter_states(paths, cfg))
    for i in chapter_indexes(paths):
        if i in rows:
            rows[i]["lines"] = len(read_chapter(paths, i))
    for r in rows.values():
        for stage in ("attribute", "render", "verify", "build"):
            r.setdefault(stage, "missing")
            r.setdefault(stage + "_detail", "")
    return list(rows.values())


def stage_status(paths: BookPaths, cfg: BookConfig) -> list[dict]:
    """One row per stage: name, state (fresh|stale|missing), detail, artifact."""
    from ab.stages import s01_ingest as ingest
    from ab.stages import s02_normalize as normalize

    rows = []

    def stamped(name, artifact, inputs_fn, detail_fn):
        if not artifact.exists():
            rows.append({"stage": name, "state": "missing", "detail": "", "artifact": artifact})
            return
        reasons = cache.stale_reasons(artifact, inputs_fn(paths, cfg))
        state = "fresh" if not reasons else "stale"
        detail = detail_fn() if state == "fresh" else "changed: " + ", ".join(reasons)
        rows.append({"stage": name, "state": state, "detail": detail, "artifact": artifact})

    def n_chapters(p):
        b = ChapterList.model_validate_json(p.read_text(encoding="utf-8"))
        return f"{len(b.chapters)} chapters, {sum(len(c.paragraphs) for c in b.chapters)} paragraphs"

    stamped("ingest", paths.chapters, ingest.inputs, lambda: n_chapters(paths.chapters))
    stamped("normalize", paths.chapters_norm, normalize.inputs, lambda: n_chapters(paths.chapters_norm))
    cast = paths.load_cast()
    n_main = sum(1 for c in cast.characters.values() if c.main)
    rows.append({"stage": "cast", "state": "fresh" if paths.cast.exists() else "missing",
                 "detail": f"{len(cast.characters)} characters, {n_main} main" if cast.characters
                 else "narrator only",
                 "artifact": paths.cast})

    chapters = chapter_status(paths, cfg)
    lines = read_lines(paths)

    def agg(stage, artifact, detail_fn):
        states = [r[stage] for r in chapters]
        n, fresh = len(states), states.count("fresh")
        if not chapters or all(s == "missing" for s in states):
            rows.append({"stage": stage, "state": "missing", "detail": "", "artifact": artifact})
            return
        state = "fresh" if fresh == n else ("stale" if fresh or "stale" in states else "missing")
        reasons = sorted({r[stage + "_detail"] for r in chapters if r[stage] == "stale" and r[stage + "_detail"]})
        detail = f"{fresh}/{n} chapters fresh"
        if reasons:
            detail += " (stale: " + "; ".join(reasons)[:60] + ")"
        extra = detail_fn()
        rows.append({"stage": stage, "state": state, "detail": f"{detail}; {extra}" if extra else detail,
                     "artifact": artifact})

    def attr_detail():
        d = [ln for ln in lines if ln.kind == "dialogue"]
        low = [ln for ln in d if ln.confidence < 0.85]
        unk = [ln for ln in d if ln.speaker == "unknown"]
        return f"{len(lines)} lines, {len(d)} dialogue, {len(low)} model-attributed, {len(unk)} unknown"

    def render_detail():
        have = [ln for ln in lines if ln.audio and (paths.work / ln.audio).exists()]
        names = sorted({ln.backend for ln in have if ln.backend})
        return f"{len(have)}/{len(lines)} lines rendered" + (f" by {', '.join(names)}" if names else "")

    def verify_detail():
        checked = [ln for ln in lines if ln.error_rate is not None]
        if not checked:
            return "not run (optional)"
        mean = sum(ln.error_rate for ln in checked) / len(checked)
        return f"{len(checked)}/{len(lines)} checked, mean error {mean:.3f}, {len(verify_failures(paths))} flagged"

    def build_detail():
        outs = sorted(paths.out.glob("*.m4b"), key=lambda p: p.stat().st_mtime) if paths.out.exists() else []
        return f"{outs[-1].name} ({_age(outs[-1])})" if outs else "no whole-book m4b yet"

    agg("attribute", paths.lines_dir, attr_detail)
    agg("render", paths.render_dir, render_detail)
    agg("verify", paths.verify_dir, verify_detail)
    agg("build", paths.out, build_detail)
    return rows


def _backend_name(registry_name: str) -> str | None:
    """Registry key -> backend.name without loading a model. Keep in sync with ab.tts.*"""
    return {"kokoro": "kokoro-82m-v1.0", "tone": "tone-test", "chatterbox": "chatterbox-mtl",
            "qwen3tts": "qwen3tts-1.7B"}.get(registry_name)


def _age(p: Path) -> str:
    secs = time.time() - p.stat().st_mtime
    for unit, div in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= div:
            return f"{secs / div:.0f}{unit} ago"
    return f"{secs:.0f}s ago"


# ---------------------------------------------------------------- report

def write_report(paths: BookPaths, cfg: BookConfig) -> Path:
    """REPORT.md: everything you want to know after an unattended run."""
    rows = stage_status(paths, cfg)
    out = [f"# {cfg.title}", "",
           (f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')} · language `{cfg.language}` · "
           f"backend `{cfg.tts.backend}`"), "",
           "## Stages", "", "| Stage | State | Detail |", "|---|---|---|"]
    out += [f"| {r['stage']} | {r['state']} | {r['detail']} |" for r in rows]

    chapters = chapter_status(paths, cfg)
    if chapters:
        out += ["", "## Chapters", "", "| # | Title | Lines | Attribute | Render | Verify | Build |",
                "|---|---|---|---|---|---|---|"]
        out += [f"| {r['chapter']} | {r['title'][:40]} | {r['lines']} | {r['attribute']} | {r['render']} | "
                f"{r['verify']} | {r['build']} |" for r in chapters]

    lines = read_lines(paths)
    if lines:
        write_script(paths, lines)
        by_speaker: dict[str, int] = {}
        for ln in lines:
            by_speaker[ln.speaker] = by_speaker.get(ln.speaker, 0) + 1
        out += ["", "## Speakers", "", "| Speaker | Lines |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in sorted(by_speaker.items(), key=lambda kv: -kv[1])]
        low = [ln for ln in lines if ln.kind == "dialogue" and (ln.confidence < 0.85 or ln.speaker == "unknown")]
        if low:
            out += ["", "## Attribution to review", "",
                    "Model-attributed or unknown. `ab fix <book> <id> --speaker NAME` to correct.", ""]
            out += [f"- `{ln.id}` **{ln.speaker}** ({ln.confidence:.2f}): {ln.text[:90]}" for ln in low]
        bad = verify_failures(paths)
        if bad:
            out += ["", "## Verify failures", "", "Listen at `work/05-audio/by-line/<id>.wav`.", ""]
            out += [f"- `{ln.id}` (error {bad[ln.id]:.2f}): {ln.text[:90]}" for ln in lines if ln.id in bad]

    if paths.run_log.exists():
        tail = paths.run_log.read_text(encoding="utf-8").splitlines()[-25:]
        out += ["", "## Recent log", "", "```"] + tail + ["```"]
    paths.work.mkdir(parents=True, exist_ok=True)
    paths.report.write_text("\n".join(out) + "\n", encoding="utf-8")
    return paths.report


# ---------------------------------------------------------------- fix

def fix_line(paths: BookPaths, line_id: str, speaker: str | None = None, text: str | None = None,
             unlock: bool = False) -> Line:
    """Edit one line by id, lock it, and drop its audio so render redoes it.

    Only that chapter's attribute file changes (its stamp stays: attribution
    need not re-run), which makes the chapter's render stamp stale, so the
    next render re-plans just that chapter and re-renders just this line.
    """
    from ab.lines import (
        read_attribution,
        read_render,
        read_verify,
        write_attribution,
        write_render,
        write_verify,
        write_views,
    )

    try:
        chapter = int(line_id[1:4])
    except ValueError:
        raise SystemExit(f"bad line id {line_id!r}; ids look like c000p0012s00 (see {paths.script.name})") from None
    lines = read_attribution(paths, chapter)
    target = next((ln for ln in lines if ln.id == line_id), None)
    if target is None:
        raise SystemExit(f"no line {line_id!r}; ids are in {paths.script.name}")
    if speaker is not None:
        cast = paths.load_cast()
        canon = cast.resolve(speaker) if speaker != "narrator" else "narrator"
        if canon is None:
            raise SystemExit(f"{speaker!r} is not in cast.yaml (or 'narrator')")
        target.speaker = canon
        target.kind = "narration" if canon == "narrator" else "dialogue"
    if text is not None:
        target.text = text
    target.confidence = 1.0
    target.locked = not unlock
    write_attribution(paths, chapter, lines)
    # Forget this line's audio and verify result; render stamp goes stale via the file hash.
    state = read_render(paths, chapter)
    if line_id in state:
        state.pop(line_id)
        for ln in lines:
            st = state.get(ln.id)
            if st:
                ln.audio, ln.backend, ln.attempts = st.audio, st.backend, st.attempts
        write_render(paths, chapter, [ln for ln in lines if ln.id in state])
    results = read_verify(paths, chapter)
    if line_id in results:
        results.pop(line_id)
        write_verify(paths, chapter, results)
    write_views(paths)
    return target
