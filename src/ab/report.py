"""Human-facing views of the work directory: script, status, report."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from ab import cache
from ab.config import BookConfig, BookPaths
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
    verify_bad = _verify_failures(paths, lines)
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


def _verify_failures(paths: BookPaths, lines: list[Line] | None = None) -> set[str]:
    """Ids that failed verify. If lines are given, only lines whose current
    audio has been checked count (a re-rendered line has error_rate None)."""
    if not paths.verify.exists():
        return set()
    bad = set()
    for raw in paths.verify.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            d = json.loads(raw)
            if not d.get("ok", True):
                bad.add(d["id"])
    if lines is not None:
        bad &= {ln.id for ln in lines if ln.error_rate is not None}
    return bad


# ---------------------------------------------------------------- status

def stage_status(paths: BookPaths, cfg: BookConfig) -> list[dict]:
    """One row per stage: name, state (fresh|stale|missing|n/a), detail, artifact."""
    from ab.stages import attribute, ingest, normalize

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
    rows.append({"stage": "cast", "state": "fresh" if paths.cast.exists() else "missing",
                     "detail": f"{len(cast.characters)} characters" if cast.characters else "narrator only",
                     "artifact": paths.cast})

    lines: list[Line] = []
    if paths.lines.exists():
        lines = attribute.read_lines(paths)

    def attr_detail():
        d = [ln for ln in lines if ln.kind == "dialogue"]
        low = [ln for ln in d if ln.confidence < 0.85]
        unk = [ln for ln in d if ln.speaker == "unknown"]
        return f"{len(lines)} lines, {len(d)} dialogue, {len(low)} model-attributed, {len(unk)} unknown"

    stamped("attribute", paths.lines, attribute.inputs, attr_detail)

    backend = cfg.tts.backend
    if lines:
        have = [ln for ln in lines if ln.audio and (paths.work / ln.audio).exists()]
        names = {ln.backend for ln in have if ln.backend}
        state = "fresh" if len(have) == len(lines) else ("stale" if have else "missing")
        detail = f"{len(have)}/{len(lines)} lines rendered" + (f" by {', '.join(sorted(names))}" if names else "")
        if state == "fresh" and names and names != {_backend_name(backend)} and _backend_name(backend):
            state, detail = "stale", detail + f" (config says {backend})"
        rows.append({"stage": "render", "state": state, "detail": detail, "artifact": paths.audio})
        checked = [ln for ln in lines if ln.error_rate is not None]
        bad = _verify_failures(paths, lines)
        if checked:
            mean = sum(ln.error_rate for ln in checked) / len(checked)
            v_state = "fresh" if len(checked) == len(lines) else "stale"
            v_detail = f"{len(checked)}/{len(lines)} checked, mean error {mean:.3f}, {len(bad)} flagged"
        else:
            v_state, v_detail = "missing", "not run (optional)"
        rows.append({"stage": "verify", "state": v_state, "detail": v_detail, "artifact": paths.verify})
    else:
        rows.append({"stage": "render", "state": "missing", "detail": "", "artifact": paths.audio})
        rows.append({"stage": "verify", "state": "missing", "detail": "", "artifact": paths.verify})

    outs = sorted(paths.out.glob("*.m4b"), key=lambda p: p.stat().st_mtime) if paths.out.exists() else []
    if outs:
        newest = outs[-1]
        stale = paths.lines.exists() and paths.lines.stat().st_mtime > newest.stat().st_mtime
        rows.append({"stage": "build", "state": "stale" if stale else "fresh",
                         "detail": f"{newest.name} ({_age(newest)})" + (" older than lines" if stale else ""),
                         "artifact": newest})
    else:
        rows.append({"stage": "build", "state": "missing", "detail": "", "artifact": paths.out})
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
    from ab.stages import attribute

    rows = stage_status(paths, cfg)
    out = [f"# {cfg.title}", "",
           (f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')} · language `{cfg.language}` · "
           f"backend `{cfg.tts.backend}`"), "",
           "## Stages", "", "| Stage | State | Detail |", "|---|---|---|"]
    out += [f"| {r['stage']} | {r['state']} | {r['detail']} |" for r in rows]

    if paths.lines.exists():
        lines = attribute.read_lines(paths)
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
        bad = _verify_failures(paths, lines)
        if bad:
            out += ["", "## Verify failures", "", "Listen at `work/05-audio/by-line/<id>.wav`.", ""]
            out += [f"- `{ln.id}` (error {ln.error_rate:.2f}): {ln.text[:90]}" for ln in lines if ln.id in bad]

    if paths.run_log.exists():
        tail = paths.run_log.read_text(encoding="utf-8").splitlines()[-25:]
        out += ["", "## Recent log", "", "```"] + tail + ["```"]
    paths.work.mkdir(parents=True, exist_ok=True)
    paths.report.write_text("\n".join(out) + "\n", encoding="utf-8")
    return paths.report


# ---------------------------------------------------------------- fix

def fix_line(paths: BookPaths, line_id: str, speaker: str | None = None, text: str | None = None,
             unlock: bool = False) -> Line:
    """Edit one line by id, lock it, and drop its audio so render redoes it."""
    from ab.stages.attribute import read_lines, write_lines

    lines = read_lines(paths)
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
    target.audio, target.backend, target.error_rate, target.attempts = None, None, None, 0
    write_lines(paths, lines)
    return target
