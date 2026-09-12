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

## Second backend and verification (milestone 3)

Chatterbox lives in its own venv because it pins an older torch:

```bash
cd backends/chatterbox && uv sync && cd ../..
uv sync --extra kokoro --extra verify --extra dev   # whisper for verification
```

A/B a book between backends. Outputs are named by backend so both coexist:

```bash
uv run ab run books/sample                       # kokoro (book.yaml default) -> out/<title>.kokoro-82m-v1.0.m4b
uv run ab render books/sample --tts chatterbox   # renders into the same cache, different keys
uv run ab verify books/sample                    # whisper check + auto re-render of drifted lines
uv run ab build  books/sample                    # -> out/<title>.chatterbox-mtl.m4b
```

Chatterbox voices are `default` or a path to a 5-15 s reference wav. Put them
in `book.yaml` under `voices:` exactly like Kokoro voice ids. Params under
`tts.params`: `exaggeration` (0-1), `cfg_weight`, `temperature`.

`work/verify.review.txt` lists lines still above the error threshold after
re-rendering; `work/llm.jsonl` logs every LLM exchange for debugging.
