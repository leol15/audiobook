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

To narrate your own book:

```bash
uv run ab new books/my-novel --source ~/my-novel.txt --language en --author "Jane Austen"
#   -> copies the source, writes a fully commented book.yaml, runs ingest and
#      prints the chapters it found (set --chapter-regex if that looks wrong)
uv run ab cast books/my-novel            # discover characters (needs Ollama)
uv run ab voices-assign books/my-novel   # pick a Kokoro/preset voice per character from the descriptions
uv run ab check books/my-novel           # voices <-> cast consistency (also runs before every `ab run`)
uv run ab run books/my-novel
```

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

The LLM (cast and attribute stages) is configured per book:

```yaml
llm:
  model: qwen3:14b     # default: OLLAMA_MODEL env, then qwen3:14b
  num_ctx: 16384       # context window requested from Ollama (its default 4096 truncates silently)
```

Prompts are sized to `num_ctx` and a prompt that would not fit is an error
(`PromptTooLong`) telling you to raise `num_ctx`, never a truncated answer.
`ab llm-check` prints the model, window, and prompt budget in use.

## Cast for a whole novel

`ab cast` discovers speakers chapter by chapter (cached in `work/03-cast/`),
merges aliases a few chapters at a time against the running cast, then ranks
everyone by dialogue lines the rules could attribute. The top `cast.main_cap`
(default 20) are marked `main: true` in `cast.yaml`:

```yaml
cast:
  main_cap: 20        # characters that get a voice and appear in every attribute prompt
  merge_chapters: 5   # chapters merged per LLM call
```

Give main characters voices in `book.yaml`; everyone else uses `_default`
and is only offered to the attribute model in chapters where they speak.
Edit `cast.yaml` freely (set `main: true` to promote someone, move a name
into `aliases` to fix a merge); it is never overwritten without `--force`,
and `--force` re-asks the model only about chapters whose text changed.

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
The worker renders lines in batches (`tts.batch` in `book.yaml`, default 32
for this backend): one model call per batch takes about as long as a single
line, so a batch of 32 runs several times faster than realtime where one line
at a time ran at 0.4x. Lower it if you hit CUDA out-of-memory, raise it if
VRAM allows (each line in a batch costs well under 100 MB). Chinese
verification compares pinyin, so whisper's script and homophone choices do
not count as errors. Verification is batched too (`verify.batch`, default
16 lines per whisper call; set 1 for one file at a time).
A `.txt` next to the wav with its transcript improves cloning. Params under
`tts.params`: `size` (`1.7B` default, `0.6B` for speed), `instruct` (style
text for presets, e.g. `"calm, low voice"`), `seed`.

To measure throughput on your own machine (x realtime, GPU utilization,
peak VRAM) and score the result with the verify metric:

```bash
cd backends/qwen3tts && .venv/bin/python bench.py --lines ../../books/small-chinese/work/04-lines.jsonl \
    --ref ../../books/small-chinese/voices/narrator.wav --batch 1 --out /tmp/bench && cd ../..
uv run python backends/qwen3tts/bench_verify.py /tmp/bench
```

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
| `03-cast/cNNN.json` | per-chapter character discovery (cache for `ab cast --force`) |
| `04-lines/cNNN.jsonl` | the machine-readable script, one file per chapter; `04-script.md` is the whole book as a screenplay with `(?)` `(!)` `[x]` `[lock]` markers; `04-attribute.review.txt` lists lines the rules could not resolve |
| `05-audio/<hash>.wav` | one file per rendered line, shared by all chapters and backends; `05-audio/by-line/<id>.wav` symlinks make a line easy to find |
| `05-render/cNNN.jsonl` | per chapter: which audio file each line has, from which backend, and how many seeds were tried |
| `06-verify/cNNN.jsonl` | per chapter: transcripts and error rates; `06-verify.review.txt` lists lines still failing |
| `07-chapters/cNNN.wav`, `07-ffmetadata.txt` | build intermediates |
| `run.log` | timestamped stage summaries and durations across runs |
| `REPORT.md` | written at the end of every `ab run` |

Each `*.inputs` stamp stores the named input hashes a stage was built from,
which is how `ab status` can say "stale: cast changed". Stages 4-7 are
stamped per chapter, so editing one chapter's text or fixing one line re-runs
only that chapter:

```bash
uv run ab status --chapters books/x           # attribute / render / verify / build per chapter
uv run ab render books/x --chapters 3,7-9     # restrict any of attribute/render/verify/build
uv run ab build books/x                       # per-chapter mp3s for every rendered chapter;
                                              # the whole-book m4b once all chapters are ready
```

Outputs land in `out/<title>.<backend>.m4b` and, per chapter,
`out/<title>.<backend>/cNNN <title>.mp3` (`build.chapter_format: mp3|m4b|none`
in `book.yaml`). While a long render runs detached, `ab build` in another
shell emits the chapters finished so far.
