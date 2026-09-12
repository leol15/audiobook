from __future__ import annotations

import time
from pathlib import Path

import typer
from rich import print

from ab.config import BookPaths
from ab.log import note
from ab.stages import s01_ingest as ingest
from ab.stages import s02_normalize as normalize
from ab.stages import s03_cast as cast
from ab.stages import s04_attribute as attribute
from ab.stages import s05_render as render
from ab.stages import s06_verify as verify
from ab.stages import s07_build as build

app = typer.Typer(no_args_is_help=True, help="Local audiobook narration pipeline.")

STAGES = [
    ("ingest", ingest.run),
    ("normalize", normalize.run),
    ("cast", cast.run),
    ("attribute", attribute.run),
    ("render", render.run),
    ("verify", verify.run),
    ("build", build.run),
]
_ORDER = [n for n, _ in STAGES]


def _book(path: Path) -> BookPaths:
    p = BookPaths(path)
    if not p.config.exists():
        raise SystemExit(f"{p.config} not found")
    return p


_CHAPTER_STAGES = {"attribute", "render", "verify", "build"}


def _chapters(spec: str | None) -> set[int] | None:
    """'3', '0,4', '2-5' -> chapter indexes; None = all."""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def _make_cmd(name, fn):
    def cmd(
        book: Path = typer.Argument(..., help="Book directory"),
        force: bool = False,
        tts: str | None = typer.Option(None, help="Override tts.backend from book.yaml"),
        chapters: str | None = typer.Option(None, help="Only these chapter indexes, e.g. 0,3 or 2-5"),
    ):
        paths = _book(book)
        cfg = paths.load_config()
        if tts:
            cfg.tts.backend = tts
        if name in _CHAPTER_STAGES:
            out = fn(paths, cfg, force=force, chapters=_chapters(chapters))
        elif chapters:
            raise SystemExit(f"--chapters applies to {', '.join(sorted(_CHAPTER_STAGES))}")
        else:
            out = fn(paths, cfg, force=force)
        print(f"[green]{name}[/] -> {out}")

    cmd.__name__ = name
    return cmd


for _name, _fn in STAGES:
    app.command(_name)(_make_cmd(_name, _fn))


@app.command()
def run(
    book: Path = typer.Argument(..., help="Book directory"),
    to: str = typer.Option("build", help="Last stage to run"),
    force: str | None = typer.Option(None, help="Re-run from this stage onward"),
    skip: str = typer.Option("cast,verify", help="Comma-separated stages to skip"),
    tts: str | None = typer.Option(None, help="Override tts.backend from book.yaml"),
):
    """Run stages in order up to --to, skipping fresh ones."""
    paths = _book(book)
    cfg = paths.load_config()
    if tts:
        cfg.tts.backend = tts
        cfg.tts.params = {}
    skips = {s.strip() for s in skip.split(",") if s.strip()}
    forcing = False
    note(paths, f"run: to={to} skip={sorted(skips)} force={force or '-'} tts={cfg.tts.backend}")
    for name, fn in STAGES:
        if force and name == force:
            forcing = True
        if name in skips:
            print(f"[dim]{name}: skipped[/]")
        else:
            t0 = time.time()
            out = fn(paths, cfg, force=forcing)
            print(f"[green]{name}[/] -> {out}")
            note(paths, f"{name}: done in {time.time() - t0:.1f}s", console=False)
        if name == to:
            break
    from ab.report import write_report

    print(f"[dim]report -> {write_report(paths, cfg)}[/]")


@app.command()
def voices(backend: str = "kokoro"):
    """List voice ids for a backend."""
    from ab.tts import load_backend

    for v in load_backend(backend).voices():
        print(v)


@app.command()
def llm_check():
    """Check the Ollama connection and list models."""
    from ab.llm import Ollama

    o = Ollama()
    print(f"{o.url}: {o.ping()}")
    print(f"[dim]default model {o.model}, num_ctx {o.num_ctx} "
          f"(prompt budget {o.prompt_budget} tokens; set llm.num_ctx in book.yaml)[/]")


@app.command("voices-design")
def voices_design(
    book: Path = typer.Argument(..., help="Book directory"),
    backend: str = typer.Option("qwen3tts", help="Backend with a voice-design model"),
    force: bool = typer.Option(False, help="Regenerate clips that already exist"),
):
    """Create one reference clip per role from cast descriptions (Qwen3-TTS VoiceDesign)."""
    from ab import voices

    paths = _book(book)
    out = voices.run(paths, paths.load_config(), force=force, backend_name=backend)
    print(f"[green]voices-design[/] -> {out}")


@app.command()
def status(
    book: Path = typer.Argument(..., help="Book directory"),
    chapters: bool = typer.Option(False, "--chapters", help="Per-chapter grid instead of the stage table"),
):
    """Show each stage: fresh, stale (and why), or missing."""
    from rich.table import Table

    from ab.report import chapter_status, stage_status

    paths = _book(book)
    cfg = paths.load_config()
    color = {"fresh": "green", "stale": "yellow", "missing": "red"}
    if chapters:
        table = Table(title=f"{cfg.title} · chapters")
        for col in ("#", "title", "lines", "attribute", "render", "verify", "build"):
            table.add_column(col)
        for r in chapter_status(paths, cfg):
            cells = []
            for st in ("attribute", "render", "verify", "build"):
                c = color.get(r[st], "white")
                d = f" ({r[st + '_detail']})" if r[st + "_detail"] else ""
                cells.append(f"[{c}]{r[st]}[/]{d}")
            table.add_row(str(r["chapter"]), r["title"][:40], str(r["lines"]), *cells)
        print(table)
        return
    table = Table(title=f"{cfg.title} · {cfg.language} · tts={cfg.tts.backend}")
    table.add_column("stage")
    table.add_column("state")
    table.add_column("detail")
    table.add_column("artifact", style="dim")
    for r in stage_status(paths, cfg):
        c = color.get(r["state"], "white")
        table.add_row(r["stage"], f"[{c}]{r['state']}[/]", r["detail"],
                      str(Path(r["artifact"]).relative_to(paths.root)))
    print(table)


@app.command()
def report(book: Path = typer.Argument(..., help="Book directory")):
    """Write work/REPORT.md (stage table, speakers, lines to review) and print its path."""
    from ab.report import write_report

    paths = _book(book)
    print(f"[green]report[/] -> {write_report(paths, paths.load_config())}")


@app.command()
def fix(
    book: Path = typer.Argument(..., help="Book directory"),
    line_id: str = typer.Argument(..., help="Line id, e.g. c000p0012s00 (see 04-script.md)"),
    speaker: str | None = typer.Option(None, help="Cast name, alias, or 'narrator'"),
    text: str | None = typer.Option(None, help="Replace the line's text"),
    unlock: bool = typer.Option(False, help="Let the attribute stage overwrite this line again"),
):
    """Correct one line's speaker or text, lock it, and queue it for re-render."""
    from ab.report import fix_line

    paths = _book(book)
    ln = fix_line(paths, line_id, speaker=speaker, text=text, unlock=unlock)
    note(paths, f"fix: {ln.id} -> {ln.speaker} {'(unlocked)' if unlock else '(locked)'}: {ln.text[:60]}")
    print("[dim]run `ab render` (or `ab run`) to re-render it[/]")


@app.command()
def play(book: Path = typer.Argument(...), line_id: str = typer.Argument(...)):
    """Print the audio path for a line id (pipe to a player)."""
    paths = _book(book)
    p = paths.audio_by_line / f"{line_id}.wav"
    if not p.exists():
        raise SystemExit(f"no audio for {line_id}; rendered lines are listed in {paths.audio_by_line}")
    print(str(p.resolve()))


@app.command()
def new(
    book: Path = typer.Argument(..., help="Directory to create, e.g. books/my-novel"),
    source: Path = typer.Option(..., "--source", "-s", help="Text or Markdown file to copy in"),
    language: str = typer.Option("en", "--language", "-l", help="en | zh"),
    title: str | None = typer.Option(None, help="Defaults to the directory name"),
    author: str = typer.Option("", help="Author for the M4B tags"),
    chapter_regex: str | None = typer.Option(None, help="Regex for chapter heading lines"),
    backend: str = typer.Option("kokoro", help="Default tts.backend"),
):
    """Scaffold a book: copy the source, write a commented book.yaml, run ingest."""
    from ab.newbook import create, ingest_preview

    if language not in ("en", "zh"):
        raise SystemExit("language must be en or zh")
    paths = create(book, source=source, language=language, title=title, author=author,
                   chapter_regex=chapter_regex, backend=backend)
    ingest_preview(paths)
