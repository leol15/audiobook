"""Throughput benchmark for the Qwen3-TTS Base (voice clone) model.

Run inside backends/qwen3tts/.venv:

    .venv/bin/python bench.py --lines ../../books/small-chinese/work/04-lines.jsonl \
        --ref ../../books/small-chinese/voices/narrator.wav --batch 8 --size 1.7B \
        --attn sdpa --out /tmp/bench-b8

Synthesizes the first N narrator lines with the given reference clip and
prints seconds of audio produced per wall second (x realtime) and mean GPU
utilization while generating. --out writes <line id>.wav plus a
bench-lines.jsonl manifest so bench_verify.py (main venv) can score them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import soundfile as sf
import torch

_LANG = {"en": "English", "zh": "Chinese"}


class GpuMonitor:
    """Samples nvidia-smi utilization on a thread; mean() over the sampled window."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.samples.append(float(out.splitlines()[0]))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()

    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else float("nan")


def load_lines(path: Path, n: int, speaker: str) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            ln = json.loads(raw)
            if ln["speaker"] == speaker:
                out.append(ln)
            if len(out) == n:
                break
    return out


def chunked_decode(model, n: int) -> None:
    """Vocode n sequences per call instead of the whole batch: same audio, flat VRAM."""
    tok = model.model.speech_tokenizer
    orig = tok.decode

    def decode(items, *a, **kw):
        wavs, fs = [], None
        for i in range(0, len(items), n):
            w, fs = orig(items[i:i + n], *a, **kw)
            wavs.extend(w)
        return wavs, fs

    tok.decode = decode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path, help="reference wav; .txt next to it is ref_text")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--speaker", default="narrator")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--size", default="1.7B", choices=["0.6B", "1.7B"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--compile", action="store_true", help="torch.compile the talker decoder")
    ap.add_argument("--sort", action="store_true", help="order lines by length before batching")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode-chunk", type=int, default=0,
                    help="vocode this many sequences at a time (0 = whole batch; peak VRAM grows ~0.7 GB/seq)")
    ap.add_argument("--out", type=Path, help="write <id>.wav and bench-lines.jsonl here")
    ap.add_argument("--json", type=Path, help="append the result as one JSON line here")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel

    lines = load_lines(args.lines, args.n, args.speaker)
    if len(lines) < args.n:
        print(f"only {len(lines)} lines for speaker {args.speaker!r}", file=sys.stderr)
    lang = _LANG.get(lines[0].get("lang", "zh"), "Auto")

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        f"Qwen/Qwen3-TTS-12Hz-{args.size}-Base", device_map="cuda:0",
        dtype=torch.bfloat16, attn_implementation=args.attn)
    load_s = time.perf_counter() - t0
    if args.compile:
        model.model.talker = torch.compile(model.model.talker, mode="reduce-overhead", dynamic=True)

    if args.decode_chunk:
        chunked_decode(model, args.decode_chunk)

    txt = args.ref.with_suffix(".txt")
    ref_text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(args.ref), ref_text=ref_text or None, x_vector_only_mode=not ref_text)

    def gen(texts: list[str]):
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(
                text=texts, language=lang, voice_clone_prompt=prompt, max_new_tokens=2048)
        return [w.reshape(-1) for w in wavs], sr

    # Warm-up: kernels, allocator, (compile). Not timed.
    gen([lines[0]["text"]] * min(args.batch, 2))
    torch.cuda.synchronize()

    order = sorted(lines, key=lambda l: len(l["text"])) if args.sort else list(lines)
    batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
    audio: dict[str, tuple] = {}
    peak_before = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with GpuMonitor() as mon:
        t0 = time.perf_counter()
        for b in batches:
            wavs, sr = gen([l["text"] for l in b])
            for l, w in zip(b, wavs):
                audio[l["id"]] = (w, sr)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    secs = sum(len(w) / sr for w, sr in audio.values())
    chars = sum(len(l["text"]) for l in lines)
    res = {
        "size": args.size, "attn": args.attn, "batch": args.batch, "compile": args.compile,
        "sort": args.sort, "decode_chunk": args.decode_chunk, "n": len(lines), "chars": chars, "audio_s": round(secs, 1),
        "wall_s": round(wall, 1), "x_realtime": round(secs / wall, 2),
        "gpu_util": round(mon.mean(), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "load_s": round(load_s, 1),
    }
    print(json.dumps(res, ensure_ascii=False))
    if args.json:
        with args.json.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "bench-lines.jsonl").open("w", encoding="utf-8") as f:
            for l in lines:
                w, sr = audio[l["id"]]
                sf.write(args.out / f"{l['id']}.wav", w, sr)
                f.write(json.dumps({"id": l["id"], "text": l["text"], "lang": l.get("lang", "zh"),
                                    "audio": f"{l['id']}.wav"}, ensure_ascii=False) + "\n")
    del peak_before


if __name__ == "__main__":
    main()
