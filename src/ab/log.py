"""Console + persistent run log (work/run.log)."""

from __future__ import annotations

from datetime import datetime

from rich import print as rprint

from ab.config import BookPaths


def note(paths: BookPaths, msg: str, *, console: bool = True) -> None:
    """Print a message and append it, timestamped, to work/run.log."""
    if console:
        rprint(msg)
    paths.work.mkdir(parents=True, exist_ok=True)
    plain = _strip_markup(msg)
    with paths.run_log.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {plain}\n")


def _strip_markup(s: str) -> str:
    import re

    return re.sub(r"\[/?[a-z ]+\]", "", s)
