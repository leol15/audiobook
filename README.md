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

## Qwen3-TTS backend

Apache 2.0, strong Mandarin, preset speakers or reference-clip cloning.
Same subprocess pattern as Chatterbox:

```bash
cd backends/qwen3tts && uv sync && cd ../..
uv run ab render books/small-chinese --tts qwen3tts   # first run downloads ~4 GB of weights
uv run ab verify books/small-chinese
uv run ab build  books/small-chinese                  # -> out/<title>.qwen3tts-1.7B.m4b
```

Voice ids are preset names (`Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`
for Mandarin; `Ryan`, `Aiden` for English) or a path to a reference wav.
Expect roughly 0.4x realtime; Chinese verification compares pinyin, so
whisper's script and homophone choices do not count as errors.
A `.txt` next to the wav with its transcript improves cloning. Params under
`tts.params`: `size` (`1.7B` default, `0.6B` for speed), `instruct` (style
text for presets, e.g. `"calm, low voice"`), `seed`.

## Voice design (Qwen3-TTS part two)

Turn each cast description into a reference clip, so a Chinese book is not
limited to the five Mandarin presets:

```bash
uv run ab voices-design books/small-chinese   # one clip per role -> books/small-chinese/voices/
uv run ab run books/small-chinese --tts qwen3tts --skip cast
```

The command reads `cast.yaml` descriptions (and `narrator_description` from
`book.yaml`, with a sensible default), asks the VoiceDesign model to speak a
short passage of that character's own lines, and saves `voices/<role>.wav`
plus a `.txt` transcript. It then writes `voices.qwen3tts` in `book.yaml` to
point at those clips; rendering clones them with the Base model. Existing
clips are kept unless `--force`. Edit a description in `cast.yaml` and rerun
with `--force` to redesign one voice; delete a clip you dislike and rerun to
regenerate just that one.

## Inspecting a book (debugging workflow)

```bash
uv run ab status books/x            # each stage: fresh / stale (and which input changed) / missing
uv run ab report books/x            # writes work/REPORT.md: stage table, speakers, lines to review, log tail
uv run ab fix books/x c000p0012s00 --speaker "Mrs. Bennet"   # correct + lock a line; render redoes it
uv run ab play books/x c000p0012s00 # path of that line's audio
```

`work/` is numbered by stage so it reads in pipeline order:

| File | What it is |
|---|---|
| `01-chapters.json` | ingest output |
| `02-chapters.norm.json` | normalize output |
| `03-llm.jsonl` | every prompt and raw response from cast and attribute |
| `04-lines.jsonl` | the machine-readable script; `04-script.md` is the same as a screenplay with `(?)` `(!)` `[x]` `[lock]` markers; `04-attribute.review.txt` lists lines the rules could not resolve |
| `05-audio/<hash>.wav` | one file per rendered line; `05-audio/by-line/<id>.wav` symlinks make a line easy to find |
| `06-verify.jsonl` | transcripts and error rates; `06-verify.review.txt` lists lines still failing |
| `07-chapters/`, `07-ffmetadata.txt` | build intermediates |
| `run.log` | timestamped stage summaries and durations across runs |
| `REPORT.md` | written at the end of every `ab run` |

Each `*.inputs` stamp stores the named input hashes a stage was built from,
which is how `ab status` can say "stale: cast changed".
