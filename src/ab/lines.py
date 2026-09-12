"""Chapter-granular line storage.

A book's script is split per chapter into three files that different stages
own, so a stage rewriting its part of one chapter never touches the others:

    04-lines/cNNN.jsonl    attribute (and `ab fix`): the ATTRIBUTION_FIELDS of Line
    05-render/cNNN.jsonl   render: RenderState (audio path, backend, attempts)
    06-verify/cNNN.jsonl   verify: VerifyResult (transcript, error rate, audio checked)

`read_chapter`/`read_lines` merge them back into Line objects for everything
that only reads (script, report, voices, play). A verify result only counts
while its `audio` still matches the render state, so a re-rendered line shows
`error_rate None` until it is checked again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ab.config import BookPaths
from ab.models import ATTRIBUTION_FIELDS, Line, RenderState, VerifyResult

_CHAPTER_FILE = re.compile(r"^c(\d{3})\.jsonl$")


def chapter_indexes(paths: BookPaths) -> list[int]:
    """Chapters that have an attribute file, in order."""
    if not paths.lines_dir.exists():
        return []
    return sorted(int(m.group(1)) for f in paths.lines_dir.iterdir()
                  if (m := _CHAPTER_FILE.match(f.name)))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)  # a checkpoint interrupted mid-write never leaves a torn file


def read_attribution(paths: BookPaths, chapter: int) -> list[Line]:
    return [Line.model_validate(d) for d in _read_jsonl(paths.chapter_lines(chapter))]


def read_render(paths: BookPaths, chapter: int) -> dict[str, RenderState]:
    return {d["id"]: RenderState.model_validate(d) for d in _read_jsonl(paths.chapter_render(chapter))}


def read_verify(paths: BookPaths, chapter: int) -> dict[str, VerifyResult]:
    return {d["id"]: VerifyResult.model_validate(d) for d in _read_jsonl(paths.chapter_verify(chapter))}


def read_chapter(paths: BookPaths, chapter: int) -> list[Line]:
    """Lines of one chapter with render and verify state merged in."""
    lines = read_attribution(paths, chapter)
    render = read_render(paths, chapter)
    verify = read_verify(paths, chapter)
    for ln in lines:
        st = render.get(ln.id)
        if st:
            ln.audio, ln.backend, ln.attempts = st.audio, st.backend, st.attempts
        vr = verify.get(ln.id)
        if vr and ln.audio and vr.audio == ln.audio:
            ln.error_rate = vr.error_rate
    return lines


def read_lines(paths: BookPaths) -> list[Line]:
    return [ln for i in chapter_indexes(paths) for ln in read_chapter(paths, i)]


def write_attribution(paths: BookPaths, chapter: int, lines: list[Line]) -> Path:
    f = paths.chapter_lines(chapter)
    _write_jsonl(f, [ln.model_dump(include=set(ATTRIBUTION_FIELDS)) for ln in lines])
    return f


def write_render(paths: BookPaths, chapter: int, lines: list[Line]) -> Path:
    """Store the render state carried on these Line objects."""
    f = paths.chapter_render(chapter)
    _write_jsonl(f, [RenderState(id=ln.id, audio=ln.audio, backend=ln.backend,
                                 attempts=ln.attempts).model_dump() for ln in lines])
    return f


def write_verify(paths: BookPaths, chapter: int, results: dict[str, VerifyResult]) -> Path:
    f = paths.chapter_verify(chapter)
    _write_jsonl(f, [r.model_dump() for r in results.values()])
    return f


def write_views(paths: BookPaths, lines: list[Line] | None = None) -> None:
    """Regenerate the whole-book human views (04-script.md) from the chapter files."""
    from ab.report import write_script

    paths.work.mkdir(parents=True, exist_ok=True)
    write_script(paths, read_lines(paths) if lines is None else lines)


def verify_failures(paths: BookPaths, chapters: list[int] | None = None) -> dict[str, float]:
    """id -> error rate for lines whose *current* audio failed verify."""
    bad: dict[str, float] = {}
    for i in chapters if chapters is not None else chapter_indexes(paths):
        render = read_render(paths, i)
        for vid, vr in read_verify(paths, i).items():
            st = render.get(vid)
            if not vr.ok and st and st.audio and st.audio == vr.audio:
                bad[vid] = vr.error_rate
    return bad
