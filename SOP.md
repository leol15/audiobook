# SOP: text file → multi-voice audiobook

Standard operating procedure for producing an M4B from a plain-text or
Markdown book with this pipeline. Written from the `books/chapter50s` run
(five web-novel chapters, Mandarin, Qwen3-TTS with designed voices):
70 minutes of audio, about 45 minutes of wall time end to end after setup.

Every command is run from the project root with uv on PATH:

```bash
cd ~/projects/audiobook && export PATH=$HOME/.local/bin:$PATH
```

## 0. One-time setup (already done on this machine)

| Need | Check | If missing |
|---|---|---|
| Main venv | `uv run ab --help` | `uv sync --extra kokoro --extra verify --extra dev` |
| Qwen3-TTS venv | `ls backends/qwen3tts/.venv/bin/python` | `cd backends/qwen3tts && uv sync` |
| Ollama on Windows | `uv run ab llm-check` lists `qwen3:14b` | start Ollama; `OLLAMA_URL` in `.env` must be its bound IP |
| ffmpeg | `ffmpeg -version` | `sudo apt install ffmpeg` |
| Model weights | first run downloads ~4 GB per Qwen3-TTS variant | wait; a `loading ...` line later is a cache load, not a download |

Keep Ollama running and the PC awake for the whole procedure. The GPU is
shared: never run cast/attribute (Ollama, ~9 GB) and a Qwen3-TTS render
(~8-11 GB) at the same time.

## 1. Scaffold the book

```bash
uv run ab new books/<slug> --source path/to/book.txt --language zh --author "Name" [--title "..."]
```

Creates `books/<slug>/` with the source copied in and a fully commented
`book.yaml`, then runs ingest and prints the chapters found.

**Check the chapter count.** If it says one chapter for a long text, the
headings are not in the default form. Look at how the file marks chapters
and set the regex, then re-ingest:

```yaml
# book.yaml — web-novel style "059 title" headings:
chapter_regex: '^\d{3}\s+\S.*$'
```
```bash
uv run ab ingest books/<slug> --force
```

**Choose the backend** in `book.yaml`. `kokoro` renders in seconds with
built-in voices (best for English); `qwen3tts` is best for Mandarin and
supports designed voices, at roughly 3-5x realtime with batching:

```yaml
tts:
  backend: qwen3tts
```

## 2. Discover the cast

```bash
uv run ab run books/<slug> --to cast --skip verify     # normalize + cast (Ollama, ~1-3 min per 5 chapters)
```

Open `books/<slug>/cast.yaml` and spend two minutes on it. This is the one
file whose quality decides everything downstream:

- Delete or demote (`main: false`) entries that are not individual speakers:
  groups ("三个御鬼者", "双胞胎少女"), objects, places.
- Fold obvious typos or alternate names into `aliases:` of the real entry.
- Descriptions drive voice choice (gender, age, manner). Sharpen any that are
  vague; they are also the voice-design prompts.

The cast stage never overwrites this file unless you pass `--force`.

## 3. Attribute lines

```bash
uv run ab run books/<slug> --to attribute --skip cast,verify   # ~5 min per 5 chapters
```

Rules resolve tagged speech; the LLM assigns the rest by quote id. Do this
before voices so voice design can sample each character's own lines.

Optional review now or after listening: `work/04-script.md` is the screenplay
view with `(?)` on model-attributed lines and `(!)` on unknown speakers.

## 4. Voices

`book.yaml` holds one voice map per backend; the one matching `tts.backend`
is used. Roles: `narrator`, `_default`, and cast names as in `cast.yaml`.

```yaml
voices:
  kokoro:   { narrator: zm_yunyang, 林风: zm_yunxi, _default: zf_xiaoxiao }
  qwen3tts: { narrator: voices/narrator.wav, 林风: voices/林风.wav, _default: voices/narrator.wav }
```

**Qwen3-TTS** ids are preset names (`Uncle_Fu`, `Vivian`, `Serena`, `Dylan`,
`Eric`; English `Ryan`, `Aiden`) or a clip path. Normal route:

```bash
uv run ab voices-design books/<slug>     # one 5-15 s clip per main role from its cast description; writes the map
```

Redo one voice: fix its description, delete its `.wav`/`.txt` in `voices/`,
rerun. Redo all: `--force`. Own recording: 5-15 s wav plus `.txt` transcript,
point the entry at it. Presets and clips can be mixed.

**Kokoro** ids are built-in (`af_`/`am_` American, `bf_`/`bm_` British,
`zf_`/`zm_` Mandarin; must match the book language):

```bash
uv run ab voices-assign books/<slug>     # by gender/age from descriptions; writes the map. --tts qwen3tts for presets
```

**Validate** (also runs before every render):

```bash
uv run ab check books/<slug>
```

Errors block: unknown cast name, unknown or wrong-language id, missing clip,
no narrator. Clips over ~15 s or cut mid-sentence make the clone loop: delete
and regenerate.

## 5. Render, verify, build

Long renders must be launched detached, or they die with the terminal:

```bash
L=/tmp/<slug>.log
setsid nohup bash -c "cd ~/projects/audiobook && export PATH=\$HOME/.local/bin:\$PATH && \
  uv run ab run books/<slug> --skip cast > $L 2>&1; echo exit=\$? >> $L" > /dev/null 2>&1 < /dev/null &
```

Monitor:

```bash
uv run ab status books/<slug> --chapters     # attribute / render / verify / build per chapter
ls books/<slug>/out/<Title>.<backend>/       # per-chapter MP3s appear as chapters finish
nvidia-smi                                   # ~8-11 GB, utilization 20-60 % is normal for Qwen3-TTS
tail -f $L
```

Expected timing for 70 minutes of audio: render 12-15 min, verify 5 min,
build 3 min. Verify transcribes every line with whisper, re-renders lines
whose phonetic error exceeds the threshold with a new seed (up to 3 tries),
and lists survivors in `work/06-verify.review.txt`.

**Stuck?** GPU at 100 %, VRAM at the ceiling, and no new files under
`work/05-audio` for several minutes means a runaway batch. The worker caps
generation length by text now, so this should not recur; if it does, kill
the run (`pkill -f "ab run books/<slug>"` from a *different* pattern than the
shell you type it in), lower `tts.batch_tokens` in `book.yaml` (default 8000;
it bounds GPU memory, `tts.batch` is only a line-count ceiling), and relaunch.
Nothing is lost: finished audio is cached.

## 6. Review and fix

```bash
cat books/<slug>/work/REPORT.md              # stage table, speakers, lines to review, verify failures
uv run ab play books/<slug> c003p0030s00     # path of one line's audio, to listen
```

Verify failures with error above ~1.0 or duration far longer than the text
are real (looping, hallucination). Errors around 0.2-0.3 on short lines are
usually whisper's own mishearing (traditional script, homophones) and can be
ignored after a listen.

Fix a wrong speaker or text, which locks the line and queues a re-render:

```bash
uv run ab fix books/<slug> c000p0088s00 --speaker narrator
uv run ab fix books/<slug> c001p0010s02 --speaker 林风 --text "corrected text"
uv run ab run books/<slug> --skip cast       # re-renders only the fixed lines, re-verifies, rebuilds
```

Locked lines survive re-running attribute. Sound effects in quotes
("哒哒~哒哒~") belong to the narrator.

## 7. Deliver

- Whole book: `books/<slug>/out/<Title>.<backend>.m4b` with chapter markers,
  cover and tags from `book.yaml`.
- Per chapter: `books/<slug>/out/<Title>.<backend>/cNNN.mp3`.

Change voices or pauses later and rerun step 5; only affected lines
re-render. Different backends write different output names, so an A/B
never overwrites.

## Reference: what each file is for

| File | Edit? | Purpose |
|---|---|---|
| `book.yaml` | yes | everything about this book: language, regex, backend, voices, pauses, thresholds |
| `cast.yaml` | yes | who speaks; descriptions are voice prompts; `main: false` = default voice |
| `voices/*.wav + .txt` | delete to regenerate | designed reference clips (qwen3tts) |
| `work/04-script.md` | read | the book as a screenplay with review markers |
| `work/REPORT.md` | read | post-run summary |
| `work/run.log` | read | timings and stage summaries across runs |
| `work/03-llm.jsonl` | read | every LLM prompt and raw answer, for attribution debugging |
