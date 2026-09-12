# ab — local audiobook narration pipeline (agent handoff)

Read this first, then `DESIGN.md` (why things are the way they are) and
`README.md` (user-facing commands). Everything below is operational context
that is not derivable from the code.

## What it is

Text/Markdown book → chaptered M4B with a narrator voice and a distinct voice
per character. All models local and free. English and Mandarin, set per book.
Seven resumable stages: ingest → normalize → cast → attribute → render →
verify → build, each a module in `src/ab/stages/s0N_*.py` with
`run(paths, cfg, force)` and `inputs(paths, cfg)` (stages 1-2) or
`run(paths, cfg, force, chapters)` and `chapter_states(paths, cfg)` (stages
4-7, which are chapter-granular: one file + stamp per chapter).

Core design rule: **the LLM labels spans by id and never emits text.** Do not
add a stage that round-trips book text through a model.

## Environment (this machine)

- WSL2 on Windows, RTX 5070 Ti 16 GB (Blackwell → needs `cu128` torch wheels),
  11 GB RAM, 4 cores. Python 3.13 system; project pinned to 3.12 via uv.
- `uv` at `~/.local/bin/uv` (not on PATH in non-login shells: `export PATH=$HOME/.local/bin:$PATH`).
- Three venvs, never merge them (torch pins conflict):
  - main: `uv sync --extra kokoro --extra verify --extra dev`
  - `backends/chatterbox/.venv` (py3.11, chatterbox-tts pins torch 2.6 → override to 2.7+ in its pyproject)
  - `backends/qwen3tts/.venv` (py3.12, torch and torchaudio must both come from the cu128 index)
- Ollama runs on **Windows**, bound to a LAN adapter IP, reachable from WSL via
  the `OLLAMA_URL` in `.env` (gitignored; `.env.example` shows the shape).
  `uv run ab llm-check` lists models. Default model `qwen3:14b`.
- Hugging Face weights cached under `~/.cache/huggingface/hub` (Kokoro,
  Chatterbox, Qwen3-TTS Base/CustomVoice/VoiceDesign, ~4 GB each). A
  `[qwen3tts] loading ...` line is a load from cache, not a download.
- ffmpeg is in WSL. `SoX could not be found` warnings from the Qwen venv are harmless.
- flash-attn 2.8.3 (community cu128/torch2.11 wheel) is installed in the Qwen venv but
  measured no faster than `sdpa`, which stays the default (`tts.params.attn`).

## Commands

```bash
export PATH=$HOME/.local/bin:$PATH
uv run pytest -q && uv run ruff check src tests      # must both pass before a commit
uv run ab run books/<slug> [--tts kokoro|qwen3tts|chatterbox|tone] [--skip cast,verify] [--force <stage>]
uv run ab status books/<slug>       # fresh/stale (and which input changed)/missing per stage
uv run ab report books/<slug>       # work/REPORT.md
uv run ab fix books/<slug> <line-id> --speaker NAME   # correct + lock + queue re-render
uv run ab voices-design books/<slug>                  # Qwen3-TTS VoiceDesign clips from cast descriptions
uv run ab new books/<slug> --source f.txt -l zh       # scaffold: commented book.yaml + ingest preview
uv run ab voices-assign books/<slug> [--tts kokoro]   # built-in voice per main character from descriptions
uv run ab check books/<slug>                          # voices/cast/backend validation (auto-runs before render)
```

Long renders (Qwen3-TTS is a few x realtime with batching, 0.4x without) must be launched detached, or they
die with the Claude session:
`setsid nohup bash -c "cd ... && uv run ab run books/x > log 2>&1" < /dev/null &`
Render checkpoints `04-lines.jsonl` every 10 lines, so an interrupted run
resumes from the cache with little loss.

## Code map

```
src/ab/
  cli.py          typer commands (one per stage + run/status/report/fix/play/voices-design/llm-check)
  config.py       book.yaml + cast.yaml models; BookPaths = the numbered work/ layout
  models.py       Chapter, Line (the unit of rendering), VerifyResult
  cache.py        content hashes; .inputs stamps store named components for `ab status`
  lines.py        per-chapter line files: 04-lines (attribution), 05-render (audio state),
                  06-verify (results); read_chapter/read_lines merge them into Line
  text.py         quote canonicalization, per-language sentence splitting, chunking
  quotes.py       rule-based dialogue extraction + speech-tag attribution (no model)
  llm.py          Ollama client, JSON-schema output, logs every exchange to work/03-llm.jsonl
  audio.py        crossfade concat, silence
  report.py       04-script.md writer, status table, REPORT.md, fix_line
  voices.py       `ab voices-design`; assign.py + voicepool.py `ab voices-assign`; check.py; newbook.py
  stages/         s01_ingest ... s07_build (imported as `from ab.stages import s04_attribute as attribute`)
  tts/            base.py protocol; kokoro.py (in-process); subprocess.py (JSON-lines worker);
                  chatterbox.py, qwen3tts.py (thin subclasses); tone.py (test backend, no model)
backends/<name>/  worker.py + its own pyproject/venv for out-of-process backends
books/<slug>/     source.txt|md, book.yaml, cast.yaml, overrides.yaml, voices/, work/, out/
tests/            no models, no Ollama: use the tone backend and ffmpeg
```

Conventions: stages write their own per-chapter file (`write_attribution`,
`write_render`, `write_verify`) and call `write_views()` at the end to
regenerate `04-script.md`; render plans whole-book but stamps per chapter;
verify results are valid only while their `audio` matches the render entry;
`ab fix` drops the line's 05/06 entries so the chapter's render stamp goes
stale by hash; render groups
pending lines by voice and sends `tts.batch` at a time to backends that have
`synthesize_batch` (qwen3tts worker protocol `kind: batch`); render sets
`line.backend` and clears `error_rate`; verify flags only apply to lines whose
current audio was checked; voice maps in `book.yaml` are per backend
(`voices: {kokoro: {...}, qwen3tts: {...}}`) and a value that is a file path
relative to the book dir is a reference clip; cache key =
(backend.name, voice, params incl. seed, text), so backends/voices never clobber
each other and build names the M4B after the backend.

## State of the books

- `books/sample` — Pride and Prejudice excerpt, English, 5 Kokoro voices, cast
  discovered by LLM. One line hand-fixed and locked (c001p0004s00 → Mrs. Bennet).
  Both books' `work/` were wiped and regenerated in the per-chapter layout on
  2026-09-12 (no migration code exists for the old single-file layout).
- `books/small-chinese` — a web-novel chapter ("058 ..."), Mandarin. `source.txt`
  is **untracked on purpose** (copyrighted; never `git add` it). Backend
  `qwen3tts` with six designed voices in `voices/` (gitignored, regenerable
  with `ab voices-design`). 188 lines; 4 lines are `unknown` (unnamed team
  members addressing 刀老大); 5 hoofbeat sound-effect quotes were fixed to
  narrator. Latest full regeneration (2026-09-12, per-chapter layout, batch
  32): render 215 s for 188 lines, verify mean phonetic error 0.052, nothing
  flagged; whole run (attribute → build) about 6 minutes.

Measured quality (see DESIGN.md table): Kokoro is best on English, Qwen3-TTS
is best on Mandarin. Chatterbox hallucinates on short lines. Chinese verify
compares toneless pinyin (whisper emits traditional script and homophones).

## Known issues / traps

1. Ollama `num_ctx` defaults to 4096 and truncates silently. Fixed: every
   request sends `llm.num_ctx` (book.yaml, default 16384), windows are sized
   from the budget, and an oversized prompt raises `PromptTooLong`. Token
   estimate is 1/CJK char, 1/3 other chars (measured ~1.5 and ~3.7 real).
2. Cast merge is incremental (batches of `cast.merge_chapters` against the
   running cast), ranked by rule-attributed lines, top `cast.main_cap` are
   `main`. Attribute prompts list main + chapter characters. Existing
   cast.yaml files without `main`/`chapters` fields are treated as all-main.
   Ranking is only as good as the speech-tag rules: on Chinese few lines
   resolve, so mentions decide most of the order.
3. `ab run` background jobs launched via the harness die with the session.
   Render checkpoints per-chapter state every 60 s, so little is lost.
4. `uv sync` in the main venv once removed `en-core-web-sm`; Kokoro English
   still worked. If English G2P breaks, that is the first suspect.
5. Preset and designed Qwen3-TTS renders share an output filename
   (`<title>.qwen3tts-1.7B.m4b` and the `<title>.qwen3tts-1.7B/` chapter
   folder); the newer overwrites.
6. `report._backend_name()` hardcodes registry→backend.name; keep in sync.

## Next tracks (agreed order)

Scale (whole-book) track: (a) `num_ctx` fix (done), (b) incremental cast
merge + main-cast cap + per-chapter cast in attribute prompts (done), (c)
chapter-granular artifacts for attribute/render/verify/build (done; render
checkpoints are time-based, every 60 s), (d) batched whisper in verify
(done, `verify.batch`, ~2x on the Chinese chapter). The track's remaining
unknown is a real 30-chapter run: nothing above has been exercised beyond
the two small books.

Speed track for Qwen3-TTS: batching is done (see DESIGN "Qwen3-TTS
throughput"; `tts.batch`); the decode loop is launch-bound, so the remaining
levers are CUDA graphs on the talker step or vLLM serving, not model size.

Backlog from DESIGN.md: emotion/style tags, EPUB ingest, mixed-language
books, review UI.

## Working agreements

- Keep `DESIGN.md` and `README.md` updated in the same commit as the change.
- Commit with `-c user.name=oreo -c user.email=leo211liao@gmail.com`; no global git identity is set.
- Never commit `books/*/source.*` for copyrighted texts, `.env`, venvs, `work/`, `out/`.
