"""Score a bench.py output directory with the pipeline's verify metric.

Run in the MAIN venv (needs faster-whisper and ab):

    uv run python backends/qwen3tts/bench_verify.py /tmp/bench-b8 [--model small]

Prints the mean error (toneless pinyin for zh, WER for en) and lines above
the 0.15 threshold, the same numbers `ab verify` reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    from ab.stages.s06_verify import error_rate

    lines = [json.loads(l) for l in (args.dir / "bench-lines.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lang = lines[0]["lang"]
    model = WhisperModel(args.model or {"en": "small.en", "zh": "small"}[lang], device="cuda", compute_type="float16")
    rates, bad = [], []
    for ln in lines:
        segments, _ = model.transcribe(str(args.dir / ln["audio"]), language=lang, beam_size=1,
                                       condition_on_previous_text=False)
        transcript = ("" if lang == "zh" else " ").join(s.text.strip() for s in segments)
        rate, edits = error_rate(ln["text"], transcript, lang, with_edits=True)
        rates.append(rate)
        if rate > args.threshold and edits > 1:
            bad.append((ln["id"], round(rate, 2), ln["text"][:40], transcript[:40]))
    mean = sum(rates) / len(rates)
    print(json.dumps({"dir": str(args.dir), "n": len(rates), "mean_error": round(mean, 3),
                      "flagged": len(bad)}, ensure_ascii=False))
    for b in bad:
        print("  ", *b)


if __name__ == "__main__":
    main()
