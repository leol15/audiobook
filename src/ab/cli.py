from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from ab.config import BookPaths
from ab.stages import attribute, build, cast, ingest, normalize, render, verify

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


def _make_cmd(name, fn):
    def cmd(
        book: Path = typer.Argument(..., help="Book directory"),
        force: bool = False,
        tts: str | None = typer.Option(None, help="Override tts.backend from book.yaml"),
    ):
        paths = _book(book)
        cfg = paths.load_config()
        if tts:
            cfg.tts.backend = tts
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
    for name, fn in STAGES:
        if force and name == force:
            forcing = True
        if name in skips:
            print(f"[dim]{name}: skipped[/]")
        else:
            out = fn(paths, cfg, force=forcing)
            print(f"[green]{name}[/] -> {out}")
        if name == to:
            break


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


@app.command("voices-design")
def voices_design(
    book: Path = typer.Argument(..., help="Book directory"),
    backend: str = typer.Option("qwen3tts", help="Backend with a voice-design model"),
    force: bool = typer.Option(False, help="Regenerate clips that already exist"),
):
    """Create one reference clip per role from cast descriptions (Qwen3-TTS VoiceDesign)."""
    from ab.stages import voices

    paths = _book(book)
    out = voices.run(paths, paths.load_config(), force=force, backend_name=backend)
    print(f"[green]voices-design[/] -> {out}")
