# Audiobook Narration Pipeline — Design

Hobby project. All models run locally and are free to use. English and
Mandarin Chinese, set per book. Input is plain text or Markdown. Output is an M4B audiobook with chapter
markers, a narrator voice, and a distinct built-in voice per character.

## Hardware and runtime

| Item | Value | Note |
|---|---|---|
| GPU | RTX 5070 Ti, 16 GB VRAM | Blackwell: needs PyTorch 2.7+ built for CUDA 12.8 (`cu128` wheels). |
| RAM | 11 GB (WSL2 default) | The LLM runs outside WSL, so this only has to hold TTS plus whisper. Raise to 16 GB in `%UserProfile%\.wslconfig` if you see OOM kills. |
| CPU | 4 cores | Fine; every heavy stage is GPU-bound. |
| Python | 3.13 | Some TTS packages lag on 3.13. Pin the project to 3.12 via uv. |
| Tools | ffmpeg in WSL; Ollama installed on Windows | WSL talks to Ollama over HTTP; see "Ollama from WSL" below. |

VRAM budget: Ollama runs on Windows but uses the same GPU, so its models
compete with WSL-side TTS and whisper for the 16 GB. Qwen3-14B at Q4 (~9 GB)
plus Kokoro (<1 GB) coexist. Larger TTS models (Chatterbox ~5 GB) should not
run alongside the 14B LLM. Stages are sequential anyway, so this is only a
concern if you parallelize. Note Ollama keeps a model loaded for 5 minutes
after the last request; the render stage should either wait or call
`/api/generate` with `keep_alive: 0` at the end of the attribute stage.

### Ollama from WSL

Ollama is installed on Windows, not inside WSL. WSL reaches it over HTTP:

- **Mirrored networking (simplest).** Add `networkingMode=mirrored` under
  `[wsl2]` in `.wslconfig`, then `http://localhost:11434` works from WSL with
  no other changes.
- **Default NAT networking.** Set the Windows environment variable
  `OLLAMA_HOST=0.0.0.0` (restart Ollama), allow port 11434 through Windows
  Firewall, and use the host IP from WSL:
  `http://$(ip route show default | awk '{print $3}'):11434`.

The LLM client reads `OLLAMA_URL` (default `http://localhost:11434`), so
either setup is a one-line config. Pull models from the Windows side:
`ollama pull qwen3:14b`.

### Context window

Ollama's default `num_ctx` is 4096 tokens regardless of what the model
supports, and a longer prompt is truncated silently: the model answers from
the tail of the passage and nothing downstream can tell. A 12k-character
Chinese cast window is ~8k tokens, so the cast stage was losing the first
half of every chapter before this was found. Rules now:

- Every request sends `options.num_ctx` (`llm.num_ctx` in `book.yaml`,
  default 16384; `llm.model` picks the model). `OUTPUT_RESERVE` (2048
  tokens) is kept free for the JSON answer, so the prompt budget is
  `num_ctx - 2048`.
- Token counts are estimated conservatively without a tokenizer: one token
  per CJK character, one per three other characters. Measured against
  Qwen3's own counts (Ollama reports `prompt_eval_count`), Chinese runs at
  ~1.5 characters per token and English at ~3.7, so the estimate overshoots
  by a third; that is the safety margin.
- Stages size their windows from the budget, not from a fixed character
  count: cast discovery packs whole paragraphs of a chapter until the budget
  is full (`llm.pack`), and attribute windows are at most 12 paragraphs but
  fewer if the cast list plus 4 context paragraphs plus the window would
  overflow (`attribute.windows`).
- `Ollama.json` raises `PromptTooLong` before sending anything over budget,
  and again after the reply if Ollama's `prompt_eval_count` exceeds the
  budget (it reports fewer tokens when a prefix was cached, never more). A
  too-long prompt is an error that names the fix, never a quietly wrong
  answer. `work/03-llm.jsonl` records `num_ctx` and the real token count of
  every exchange.

## Design principle: the LLM labels spans, it never emits text

Every character in `lines.jsonl` is read aloud, and the verify stage only
checks audio against that text. An LLM that re-outputs book text while
splitting or cleaning it will occasionally drop a clause, "fix" a typo, or
smooth a sentence, and nothing downstream can detect that. So:

- Splitting (chapters, paragraphs, quotes, sentences) is rule-based and
  deterministic. Deterministic splitting also keeps the content-hash cache
  stable across runs.
- The LLM receives spans with ids and returns ids plus labels: which line
  numbers start chapters, which quote id belongs to which speaker, which
  names are aliases. Text never round-trips through the model.
- If an LLM normalization pass is ever added, it returns
  `{span_id, replacement}` edits that are diffed and can be rejected, not a
  rewritten document.

## Pipeline

Every stage reads the previous stage's artifact from disk and writes its own.
Every stage is idempotent and resumable: outputs are keyed by a content hash of
their inputs, so re-running skips finished work and an edit re-renders only
what changed. A full book render takes minutes with Kokoro and hours with
larger models, so this property is what makes experimentation bearable.

```
source.md
  │  1. ingest      01-chapters.json         (chapter list, cleaned paragraphs)
  │  2. normalize   02-chapters.norm.json    (numbers, abbreviations expanded)
  │  3. cast        cast.yaml                (characters found, aliases merged; user maps voices)
  │  4. attribute   04-lines/cNNN.jsonl      (per chapter: each utterance's speaker, text, lock)
  │  5. render      05-audio/<hash>.wav      (one file per utterance, content-addressed)
  │                 05-render/cNNN.jsonl     (per chapter: audio path, backend, attempts per line)
  │  6. verify      06-verify/cNNN.jsonl     (per chapter: transcript, error rate; bad lines re-rendered)
  │  7. build       07-chapters/cNNN.wav     (per chapter, with pauses)
  │                 out/<book>.<backend>/cNNN.mp3   (each chapter as soon as it is rendered)
  │                 out/<book>.<backend>.m4b        (whole book once every chapter is ready)
```

### Chapter granularity

Stages 4-7 work per chapter so that a 30-chapter novel re-runs only what
changed, and chapters can be listened to while the rest renders. Each stage
writes one file per chapter with its own `.inputs` stamp, and the stamp is
the hash of the *previous stage's file for that chapter* plus the config the
stage depends on:

| Stage | File | Stamp inputs |
|---|---|---|
| attribute | `04-lines/cNNN.jsonl` | chapter title+paragraphs, language, cast names and aliases |
| render | `05-render/cNNN.jsonl` | hash of the 04 file, backend name, voice map, params |
| verify | `06-verify/cNNN.jsonl` | hash of the 05 file, whisper model, threshold |
| build | `07-chapters/cNNN.wav` (+ `cNNN.mp3`) | hash of the 05 file, pauses |

Consequences:

- Editing chapter 5's text changes only its 04 stamp (ingest and normalize
  re-run whole-file but are cheap); attribute re-runs chapter 5, which
  changes its 04 hash, so render re-plans chapter 5, and unchanged lines are
  cache hits. Chapter identity is positional: inserting a chapter shifts
  everything after it, and those chapters re-run.
- `ab fix` edits one 04 record and drops that line's 05 and 06 entries
  without touching the 04 stamp (attribution need not re-run). The chapter's
  render stamp is stale by hash, and render re-renders just that line.
- A voice or param change in `book.yaml` is now detected (it is in the
  render stamp); every unchanged line is a free cache hit.
- Verify results carry the audio path they checked, so after a re-render a
  line shows `error_rate None` until it is checked again, and only lines
  without a current result are transcribed. When verify drops a line's audio
  for a new seed it removes the chapter's render stamp, and the re-render
  loop runs render only on the failed chapters.
- Render still plans across the whole book so batches fill by voice across
  chapters, and checkpoints write only the chapter files it touched, at
  least every 60 s (`CHECKPOINT_SECONDS`), atomically (write to `.tmp`,
  rename).
- The cast stamp is the whole cast's names and aliases, not the chapter's
  subset: adding an alias can change rule resolution anywhere. Description
  edits re-run nothing; promoting a character to main re-attributes every
  chapter because every prompt changes.
- `Line` remains the in-memory model; `ab.lines.read_chapter` merges the
  three files back into it for everything that only reads (script, report,
  voices, play). Line ids (`cNNNpNNNNsNN`) and audio cache keys are the same
  as before the split.

### 1. ingest

- Markdown: chapters split on `#` / `##` headings. Front matter (title,
  author, cover path) read from a YAML block if present.
- Plain text: chapters split by a regex chosen by `language`. English
  default `^(chapter|part|book)\s+([0-9]+|[ivxlc]+|\w+)\b` case-insensitive,
  or a bare numeral line. Chinese default `^第[0-9一二三四五六七八九十百千零〇]+[章回节卷部]`.
  Override with `--chapter-regex`. If the regex finds zero or implausibly
  few chapters for the file length, an LLM fallback receives only the
  candidate heading lines (short lines, all-caps lines, lines followed by a
  blank) with line numbers and returns the line numbers that start chapters.
  Final fallback: whole file is one chapter.
- Cleanup: unwrap hard line breaks inside paragraphs, normalize quotes to
  straight `"` (attribution depends on this) including `“ ”`, `「 」`, and
  `『 』`, map full-width punctuation to a canonical set, collapse whitespace,
  drop Gutenberg-style headers and footers if detected.
- Language is declared in `book.yaml` (`language: en | zh`). Mixed-language
  books are out of scope for v1; the line schema carries a `lang` field so a
  per-line override can be added later without changing artifacts.
- Output keeps paragraph boundaries. Paragraphs are the unit for pauses and
  for attribution context.

### 2. normalize

Rule-based, no model, with one rule set per language. English: ordinals,
years, money, units, common abbreviations (Mr., Dr., St., etc.), roman
numerals in headings, em-dashes to a comma-pause. Chinese: Arabic numerals to
spoken form (`2024年` → `二零二四年`, `3个` → `三个`), units and percentages,
Latin abbreviations left as-is since the G2P spells them. Keep a per-book
`overrides.yaml` for name pronunciations (`Hermione: her-MY-oh-nee`; for
Chinese, polyphone fixes like `重庆: chong2 qing4`), applied as text
substitution since Kokoro accepts phoneme hints via its `misaki` G2P.

### 3. cast

Three passes with the local LLM (Ollama, Qwen3-14B Q4 default; Qwen3-8B if
you want speed), sized so a 30-chapter novel works:

1. **Discover, per chapter.** List named speaking characters with aliases and
   a one-line description. Each chapter's answer is cached in
   `work/03-cast/cNNN.json` keyed by the chapter's text, so `ab cast --force`
   after editing one chapter only re-asks about that chapter.
2. **Merge, incrementally.** The old design sent every chapter's entries in
   one prompt, which cannot fit a novel. Now a few chapters at a time
   (`cast.merge_chapters`, default 5, also bounded by the context budget)
   are matched against the *running* cast: the prompt lists known canonical
   names and the new entries, and the model returns groups whose canonical
   is either a known name or a new person. An entry whose name already
   resolves against the running cast is folded in without a model call. A
   group's aliases carry the chapter indexes of the entries they came from,
   and anything the model silently dropped is kept as its own character.
3. **Rank, no model.** Rule-based speech-tag attribution (the same
   `quotes.extract` the attribute stage uses) runs over the whole book and
   counts dialogue lines per character; name mentions break ties. The top
   `cast.main_cap` (default 20) are the main cast.

Writes `cast.yaml`, ordered by rank:

```yaml
characters:
  Elizabeth: { aliases: [Lizzy, Miss Bennet], description: "...", main: true, lines: 143, chapters: [0, 1, 4] }
  Darcy:     { aliases: [Mr. Darcy, Fitzwilliam], description: "...", main: true, lines: 97, chapters: [2, 4] }
  the butler: { aliases: [], description: "...", main: false, lines: 2, chapters: [7] }
```

`main: true` characters are listed in every attribute prompt and should get a
voice in `book.yaml`; `ab voices-design` designs clips only for them. Minor
characters are listed to the attribute LLM only in the chapters where they
were found speaking (`chapters`), so the prompt stays short and the model is
not offered a hundred names to confuse, and they fall through to the
`_default` voice in render. `lines` is the rule-resolved count at cast time,
a ranking hint rather than a total (Chinese speech tags resolve fewer lines
than English ones, so mentions matter more there). A `cast.yaml` without
these fields (written before they existed) treats every character as main.

You edit the file to fix merges or promote a character; the stage never
runs again unless you `--force` it or delete the file. Human review of this
one small file fixes most attribution errors upstream.

### 4. attribute

The stage most likely to produce wrong output, so it is split so the LLM does
as little as possible:

1. **Quote extraction (rules).** Regex over normalized quotes splits each
   paragraph into narration spans and dialogue spans. Handles nested single
   quotes and multi-paragraph quotes (open quote without close continues).
   Works identically for both languages because ingest canonicalized the
   quote marks.
2. **Speaker assignment (rules first).** Attribution tags adjacent to a quote
   resolve directly against the cast list: English `said X` / `X replied`,
   Chinese `X说` / `X道` / `X问` / `X答道` and the `X说：“…”` colon form.
   This resolves most lines with zero model calls.
3. **Speaker assignment (LLM).** Remaining quotes go to the LLM in windows of
   up to 12 paragraphs (fewer if the context budget fills) with the cast list
   for that chapter (main cast plus characters discovered in it), the
   preceding paragraphs as context, and a strict JSON schema:
   `{quote_id, speaker, confidence}`. Answers are resolved against the whole
   cast, so a name the model knows from context still counts. Alternating-speaker
   heuristics in two-person scenes are given as context. Qwen3 is strong in
   both languages, so the prompt is the same with the instructions written in
   the book's language.
4. **Output** `04-lines/cNNN.jsonl`, one record per utterance:
   `{id, chapter, para, kind: narration|dialogue, speaker, text, confidence, locked}`.
   Low-confidence lines are listed in `04-attribute.review.txt` for manual
   fixes; a record marked `locked: true` whose text is unchanged is kept
   verbatim when its chapter re-runs.

Emotion tags are out of scope for v1. The record schema has a free `style`
field reserved so a later backend can consume them.

### 5. render

The swappable core. One protocol, several backends, selected in `book.yaml`:

```python
class TTSBackend(Protocol):
    name: str            # used in cache keys, e.g. "kokoro-0.9"
    max_chars: int       # segmenter splits utterances above this
    sample_rate: int
    def voices(self) -> list[str]: ...
    def synthesize(self, text: str, voice: str, **params) -> np.ndarray: ...
```

| Backend | Size | License | Why it is here |
|---|---|---|---|
| **Kokoro-82M** (default) | 82M | Apache 2.0 | ~100x realtime on this GPU, 50+ English voices plus 8 Mandarin voices (`zf_*`, `zm_*`), very stable. English cast coverage is ample; the Chinese cast is limited to 8 distinct voices. Needs `misaki[zh]` for the Chinese G2P. |
| Chatterbox Multilingual | ~500M | MIT | Reference-audio cloning (unlimited distinct voices), an `exaggeration` knob for emotion, and Mandarin support. Roughly realtime. The "large model" to experiment against. |
| Qwen3-TTS | 0.6–1.7B | Apache 2.0 | Strongest Chinese quality of the free models. Three variants: CustomVoice (9 preset speakers, 5 Mandarin, plus a style instruction), Base (clone from a reference clip), VoiceDesign (voice from a text description). Runs in `backends/qwen3tts` through the subprocess backend; the worker loads variants lazily since each is ~4 GB VRAM. Autoregressive, so verify stays on. |
| Orpheus 3B / F5-TTS | 1–3B | varies | English-only or weaker Chinese; add only for comparison. F5's weights are non-commercial, fine for a hobby. |
| tone (test) | none | n/a | Emits a sine tone sized to the text. Exercises render, cache, pauses, and ffmpeg assembly in tests without any model or GPU. |

Each backend declares `languages: set[str]`; render refuses a voice map that
sends a `zh` book to an English-only backend. Kokoro needs `lang_code="z"`
and a Chinese-voice id together, and the backend enforces that pairing.

Rules:

- Each backend is an optional uv dependency group (`uv sync --extra kokoro`).
  These packages pin conflicting torch versions: confirmed on day one, since
  `chatterbox-tts` pins torch 2.6, which has no Blackwell kernels, while
  Kokoro needs 2.7+. So Chatterbox is not an extra at all. It gets its own
  venv and runs through `SubprocessBackend`, which talks JSON over
  stdin/stdout. The protocol is already serializable, so this costs nothing
  in the main code.
- Long utterances are split at sentence boundaries to `max_chars` and joined
  with a short crossfade. Splits never cross a speaker change.
- Cache key = `sha256(backend.name, voice, params, text)`. Switching backends
  or voices never clobbers earlier output, so you can A/B two renders of the
  same book side by side. Each line records which backend produced its audio,
  and build names the M4B after it, so `ab render --tts chatterbox` followed
  by `ab build` yields a second file next to the Kokoro one.
- Deterministic seeds where the backend supports them; seed is part of
  `params`, so "regenerate with a different seed" is just a new cache key.
- Voice names in `cast.yaml` are per-backend: a `voices:` map in `book.yaml`
  translates cast roles to backend voice ids so swapping backends means
  editing one map, not the cast.

### Qwen3-TTS throughput

Measured with `backends/qwen3tts/bench.py` (run in that venv): the first 20
narrator lines of `books/small-chinese` (297 characters, ~70 s of audio),
cloned from the designed narrator clip, timed after a warm-up call, with
`nvidia-smi` utilization sampled while generating. `bench_verify.py` (main
venv) scores the same wavs with the verify stage's pinyin metric so a speedup
cannot hide a quality regression.

Baseline, 2026-09-12, RTX 5070 Ti, torch 2.11 cu128, `qwen-tts` 12Hz 1.7B
Base, `sdpa` attention, one line per call (what the worker did until now):

| Config | x realtime | GPU busy | peak VRAM |
|---|---|---|---|
| 1.7B, sdpa, batch 1 | 0.43 | 16 % | 5.0 GB |

GPU busy at 16 % says the decoder loop is launch-bound: each codec frame costs
a fixed amount of Python and kernel-launch time, and the card idles between
launches. That points at batching (amortize the per-step cost over many
lines) rather than at a smaller model or faster attention kernels.

**Batching.** `generate_voice_clone` and `generate_custom_voice` accept lists
of texts and run them as one left-padded batch through the talker. The worker
protocol gained `{"kind": "batch", "items": [{text, out}, ...], voice, lang,
params}` (one model call, one response with `paths`), and the worker
announces `"batch": true` in its hello so the parent knows it may send them.
`SubprocessBackend.synthesize_batch` uses it and falls back to one request per
text if the worker is old or the batch fails. The render stage plans all
pending (line, chunk) items first, groups them by voice and params (a batch
must share one reference clip), sorts each group by text length so a batch
finishes together (batched generation runs until its longest member stops),
and sends `tts.batch` items per call (default from the backend, 1 for the
others). Multi-chunk lines are reassembled when all their chunks are back.
Batching is not part of the cache key; a re-render of one line in a batch of
different composition gives different sampled audio, the same way a new seed
would.

Two things were needed to make large batches fit in 16 GB:

- The codec decoder (vocoder) was called on the whole batch; its activations
  cost ~0.7 GB per sequence (batch 20 peaked at 19 GB and spilled). The
  worker now vocodes one sequence at a time (`decode_chunk`, default 1):
  identical audio, the talker's ~5 GB is the peak, and the decode step is
  under a second for 20 lines either way.
- The reference clip was re-encoded on every request. The worker caches the
  clone prompt per file, which also trims a little per-line overhead.

Same 20 lines, all `sdpa`, sorted batches, vocoder chunked unless noted:

| Config | calls | x realtime | GPU busy | peak VRAM |
|---|---|---|---|---|
| 1.7B, batch 1 | 20 | 0.43 | 16 % | 5.0 GB |
| 1.7B, batch 4 (unsorted, whole-batch vocode) | 5 | 0.92 | 18 % | 7.2 GB |
| 1.7B, batch 8 (unsorted, whole-batch vocode) | 3 | 1.28 | 20 % | 10.2 GB |
| 1.7B, batch 20 (unsorted, whole-batch vocode) | 1 | 1.20 | 59 % | 19.1 GB (spilled) |
| 1.7B, batch 8 | 3 | 1.58 | 24 % | 9.9 GB |
| 1.7B, batch 12 | 2 | 1.70 | 34 % | 12.4 GB |
| 1.7B, batch 16 | 2 | 1.87 | 19 % | 5.5 GB |
| 1.7B, batch 20 | 1 | 2.71 | 21 % | 6.0 GB |
| 0.6B, batch 1 | 20 | 0.40 | 15 % | |
| 0.6B, batch 16 | 2 | 1.83 | 18 % | |

Wall time per call is nearly independent of the batch size (about 25 s for
these ~10 s lines whether the call carries 1 line or 20), so throughput is
roughly linear in the batch until the GPU saturates, which it has not at 20.
The 0.6B model is no faster than 1.7B at any batch size, which is the same
finding from the other side: compute is not the bottleneck, so there is no
reason to give up the larger model's quality.

**Other levers, each measured separately** (same lines, `bench.py`):

- *flash-attn.* No wheel on PyPI for torch 2.11/cu128/py3.12, but a
  community build exists (`mjun0812/flash-attention-prebuild-wheels`,
  `flash_attn-2.8.3+cu128torch2.11`) and it loads on sm_120. It makes no
  difference: 0.35x at batch 1 and 2.65x at batch 20 against 0.43x / 2.71x
  for `sdpa`. Attention over a few hundred tokens is a small fraction of a
  launch-bound step. `sdpa` stays the default; `tts.params.attn:
  flash_attention_2` selects it for anyone who wants to re-check on other
  hardware.
- *0.6B model.* 0.40x at batch 1, 1.83x at batch 16, i.e. the same as 1.7B.
  Not used.
- *torch.compile / CUDA graphs.* Wrapping the talker module is a no-op
  because HF `generate()` calls the module's own `forward`; compiling the
  bound `forward` (`bench.py --compile`) failed in inductor's C helper build
  on this WSL install, and it could not have delivered the thing that would
  matter (CUDA-graphed decode steps) anyway: the model declares
  `_supports_static_cache = False`, so every step has a new KV length and the
  graph would be re-captured. A real fix is a hand-written decode loop with a
  static cache, or serving through vLLM, both out of scope for now.
- *Batch size.* On 40 lines: batch 16 → 3.05x, 32 → 3.48x, 40 → 5.57x at
  7.8 GB peak. It keeps scaling with the number of lines that share a call,
  so the default is 32 (about 7 GB with the 1.7B model; lower `tts.batch` on
  OOM, raise it on a bigger card). Whole-book batches fill better than the
  benchmark's, because all of a speaker's lines form one group.

**Quality check.** Mean pinyin error on the same 20 lines, whisper `small`:

| Config | x realtime | mean error | flagged |
|---|---|---|---|
| 1.7B sdpa batch 1 (old default) | 0.43 | 0.075 | 0 |
| 1.7B sdpa batch 8 | 1.58 | 0.068 | 0 |
| 1.7B sdpa batch 20 | 2.71 | 0.083 | 1 |
| 1.7B flash-attn batch 1 | 0.35 | 0.058 | 0 |
| 1.7B flash-attn batch 20 | 2.65 | 0.107 | 1 |
| 0.6B sdpa batch 1 | 0.40 | 0.085 | 0 |
| 0.6B sdpa batch 16 | 1.83 | 0.075 | 0 |
| 1.7B sdpa batch 40 (40 lines) | 5.57 | 0.082 | 1 |

The spread is sampling noise on a few four-character lines (林风一摸 heard
as 凌峰隐摸 on one draw, clean on the next), not a batching effect: the
whole book confirms it. Re-rendering all 188 lines of `books/small-chinese`
with the new default (batch 32, 11 model calls) took 3.5 minutes for 13.7
minutes of audiobook where it used to take about 35, and `ab verify`
reports mean error 0.048 with nothing flagged after its usual re-render
pass, against 0.060 for the one-line-at-a-time render it replaced.

### 6. verify

Autoregressive TTS skips sentences, repeats phrases, and invents words, and
the audio sounds fluent while doing it. Catch it mechanically. The stage is
off by default and enabled per book; it is not needed with Kokoro.

- Transcribe every rendered utterance with faster-whisper (`small.en` for
  English, multilingual `small` with `language="zh"` for Chinese; GPU,
  batched; adds a few minutes per book).
- Compute word error rate against the normalized source text for English.
  For Chinese, compare toneless pinyin syllables rather than characters:
  whisper freely emits traditional script and homophones (林峰 for 林风,
  蝴蝶节 for 蝴蝶结), none of which are TTS errors. Measured on the Chinese
  book, the character metric reported 0.109 mean error and flagged 9 lines
  for Qwen3-TTS; the phonetic metric on the same audio reported 0.023 and
  flagged none. The remaining weakness is whisper `small` itself on Mandarin;
  `verify.whisper_model: large-v3-turbo` is the upgrade if false positives
  persist.
- Utterances above a threshold (start at 0.15) are re-rendered with a new
  seed, up to 3 attempts. Still-failing lines go to `verify.review.txt`.
  Short lines tolerate one edit regardless of rate: a single misheard word on
  a three-word line is whisper's error more often than the TTS model's.
  Measured on the sample book:

  | Book | Backend | Mean error | Lines re-rendered | Still bad after 3 tries |
  |---|---|---|---|---|
  | English sample | Kokoro-82M | 0.037 (WER) | 0 | 0 |
  | English sample | Chatterbox Multilingual (default voice) | 0.200 (WER) | 6 | 3 |
  | Chinese sample | Kokoro-82M | 0.053 (pinyin) | 1 | 2 |
  | Chinese sample | Qwen3-TTS 1.7B CustomVoice (presets) | 0.023 (pinyin) | 0 | 0 |
  | Chinese sample | Qwen3-TTS 1.7B Base, cloning designed clips | 0.012 (pinyin) | 1 | 1 |

  The designed-and-cloned voices scored best of all, and the one remaining
  flag is whisper hearing 唯有 as 我有. Whisper's transcripts also write
  numbers as digits, so the Chinese comparison spells digits out before
  stripping punctuation.

  On Mandarin the ranking flips: Qwen3-TTS is both more accurate and, to the
  ear, far more natural than Kokoro's Mandarin voices. It costs speed: about
  0.4x realtime on this GPU without flash-attention (Kokoro is ~100x), so a
  six-minute chapter takes about fifteen minutes to render.

  Chatterbox's failures were hallucinations after short lines ("This was
  invitation enough." became a sentence and a half of invented speech). That
  is the autoregressive failure mode this stage exists for; Kokoro never
  produced it. Likely mitigation for later: pad very short utterances with
  surrounding narration or a minimum-length rule before sending to an
  autoregressive backend.
- Also flags audio-level problems: utterance duration far off the expected
  chars-per-second, or long internal silence.

### 7. build

- Pause insertion: 0.35 s between sentences of one speaker, 0.6 s between
  paragraphs, 1.2 s before dialogue by a new speaker, 2.5 s at chapter start.
  All configurable in `book.yaml`.
- Per-chapter WAV concatenation (`07-chapters/cNNN.wav`), rebuilt only when
  the chapter's render file or the pauses changed.
- Each chapter whose render is complete is also encoded on its own into
  `out/<title>.<backend>/cNNN <title>.mp3` (`build.chapter_format`: `mp3`,
  `m4b`, or `none`), with `loudnorm` and track tags, so `ab build` can run
  beside a detached render and you can listen to finished chapters.
- Once every chapter is ready: AAC 64 kbps mono 44.1 kHz into M4B with
  ffmpeg, `loudnorm` to -18 LUFS, chapter markers from the chapter list via
  an `ffmetadata` file, cover art and title/author tags from front matter.
  Until then build reports how many chapters are ready and skips the M4B.

## Debuggability

A long unattended render must be inspectable afterwards without re-running
anything, so the work directory is designed to be read by a person:

- Artifacts are numbered by stage (`01-chapters.json` ... `07-chapters/`) so
  a directory listing reads in pipeline order; stages 4-7 hold one file per
  chapter (`cNNN.jsonl`).
- Every stage's `.inputs` stamp stores its named input hashes as JSON, and
  `ab status` diffs them to say which input changed ("stale: cast changed")
  rather than only that something did. `ab status --chapters` shows the same
  per chapter, and `REPORT.md` includes the chapter grid.
- `04-script.md` is the book as a screenplay with markers for model-attributed,
  unknown, verify-failed, and hand-locked lines. It is regenerated at the end
  of every stage that writes lines and by `ab fix`, so it is never out of date.
- `05-audio/by-line/<id>.wav` symlinks map line ids to content-hashed files.
- `run.log` accumulates timestamped stage summaries and durations across
  runs; `REPORT.md` is written at the end of every `ab run` with the stage
  table, speaker counts, lines to review, and the log tail.
- `ab fix <book> <id> --speaker X` edits and locks a line and clears its
  audio, so the manual correction loop is one command plus a render.

## Project layout

```
audiobook/
  pyproject.toml            uv project, python 3.12, extras per TTS backend
  src/ab/
    cli.py                  typer: ab ingest|normalize|cast|attribute|render|verify|build|run
    stages/                 s01_ingest.py ... s07_build.py, numbered in execution order;
                            each: run(paths, cfg, force[, chapters]) -> artifact, and
                            inputs(paths, cfg) or chapter_states(paths, cfg) for `ab status`
    lines.py                per-chapter line files (04/05/06) and the merged Line view
    voices.py               `ab voices-design` (a tool, not a pipeline stage)
    report.py               script.md, status table, REPORT.md, fix-line
    log.py                  console + work/run.log
    tts/                    base.py (protocol), kokoro.py, chatterbox.py, subprocess.py
    llm.py                  thin Ollama client with JSON-schema enforcement and retries
    cache.py                hash-keyed artifact store
    audio.py                concat, crossfade, silence, loudness helpers (numpy + ffmpeg)
  books/<slug>/
    source.md
    book.yaml               language, backend, voice map, pauses, thresholds
    cast.yaml
    overrides.yaml
    work/                   numbered stage artifacts, run.log, REPORT.md (gitignored)
    out/                    final m4b
  tests/                    fixtures: 3-chapter public-domain excerpts, one English, one Chinese
```

`ab run --to render` executes stages in order, skipping ones whose inputs are
unchanged. `ab run --force attribute` invalidates from that stage onward.

## Build order

1. **End-to-end single narrator.** ingest, normalize, render (Kokoro), build.
   No LLM yet. Goal: a listenable M4B from a text file in one command. This
   proves the cache, the segmenter, and the ffmpeg assembly. Do English first,
   then run the Chinese fixture through the same path to shake out the
   punctuation, chapter regex, and G2P differences early.
2. **cast + attribute.** Point the LLM client at the Windows Ollama, pull
   Qwen3-14B, add rules-first attribution, then the LLM fallback. Measure how
   many lines the rules resolve before tuning prompts.
3. **Second backend + verify.** Add Chatterbox in its own venv behind the
   subprocess backend, then add faster-whisper verification and automatic
   re-render. These go together: Kokoro is non-autoregressive and almost
   never skips or repeats text, so verification only earns its cost once an
   autoregressive model (Chatterbox, Qwen3-TTS, Orpheus) is in the mix. Use
   it to A/B a chapter between backends with a measured error rate.
4. **Qwen3-TTS, part one.** Clone/preset backend behind the subprocess
   protocol, A/B against Kokoro on the Chinese book with verify on.
5. **Qwen3-TTS, part two: voice design.** `ab voices-design` turns each cast
   description into a reference clip with the VoiceDesign model, saved under
   the book's `voices/` directory with a transcript. Rendering then clones
   that clip with the Base model so a character sounds the same across the
   whole book. This removes the voice-count cap on Mandarin without
   recording anything. Design choices: the sample text is the character's
   own lines from `lines.jsonl` (so the clip is heard saying in-character
   words), the map is written into `book.yaml` so it is inspectable and
   editable, and existing clips are never overwritten without `--force`, so
   a voice you like survives re-runs. The cast description is the design
   prompt, which makes `cast.yaml` the single place to steer a voice.
6. Later, if wanted: emotion/style tags, EPUB ingest, per-line language
   switching for mixed books, a small review web UI for fixing attributions
   while listening.

## Known risks

- **Attribution quality** is the ceiling on the whole multi-voice idea. The
  rules-first design and the editable `lines.jsonl` are the mitigation;
  expect to hand-fix a few percent of lines on a first book.
- **Dependency conflicts** between TTS packages. Mitigated by extras plus the
  subprocess backend.
- **Python 3.13** support in TTS packages. Pin to 3.12.
- **Chinese voice variety.** Kokoro ships only 8 Mandarin voices, so a
  Chinese book with a large cast will reuse voices. Qwen3-TTS or Chatterbox
  cloning removes the cap; the voice map makes that a config change.
- **Chinese sentence splitting.** No spaces and different terminators
  (`。！？；`), so the segmenter needs a per-language splitter rather than
  a shared regex.
- **WSL to Windows networking.** Ollama on Windows is invisible from WSL
  until mirrored networking or `OLLAMA_HOST=0.0.0.0` is set. Check with
  `curl $OLLAMA_URL/api/tags` before building stage 3.
