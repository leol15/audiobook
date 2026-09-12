# ab — local audiobook narration

See `DESIGN.md` for the full design. Quick start (milestone 1, single narrator):

```bash
uv sync --extra kokoro --extra dev      # first time; pulls CUDA 12.8 torch
uv run pytest
uv run ab run books/sample              # ingest -> normalize -> attribute -> render -> build
ls books/sample/out/
```

Each stage is also a command (`uv run ab ingest books/sample`), and
`ab run --force render` re-runs from that stage onward. `ab voices` lists
voice ids; `ab llm-check` confirms the Windows Ollama is reachable.

To narrate your own book: copy `books/sample`, replace `source.txt`
(or `source.md`), and edit `book.yaml`.
