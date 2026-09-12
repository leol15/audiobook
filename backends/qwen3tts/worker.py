"""Qwen3-TTS worker. Runs in its own venv; speaks JSON lines on stdin/stdout.

Protocol is a superset of backends/chatterbox/worker.py:
  Request:  {"text", "voice", "lang": "en"|"zh", "out": wav path, "params": {...}}
            or {"kind": "batch", "items": [{"text", "out"}, ...], "voice", "lang", "params"}
               -> one model call for all items (same voice); the response carries "paths"
            or {"kind": "design", "text", "lang", "instruct", "out"} -> VoiceDesign model
  Response: {"ok": true, "path", "sr"} | {"ok": true, "paths": [...], "sr"} | {"ok": false, "error"}
  Hello:    {"ready": true, "sr", "device", "batch": true} after the first model loads.

voice:
  - a preset speaker name (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, ...)
    -> CustomVoice model
  - a path to a reference wav -> Base model, voice clone. If a .txt with the
    same stem exists next to it, its content is used as ref_text (better
    quality); otherwise x-vector-only cloning is used. The encoded clone
    prompt is cached per file, so a voice is encoded once per run.
params:
  size: "1.7B" (default) | "0.6B"
  attn: "sdpa" (default) | "flash_attention_2" (needs flash-attn in this venv)
  instruct: style/emotion instruction for CustomVoice (e.g. "calm, low voice")
  seed: int
  max_new_tokens: int (default 2048)
  decode_chunk: sequences vocoded per codec call (default 1). The talker
    generates a whole batch in ~5 GB; the codec decoder needs ~0.7 GB per
    sequence, so vocoding one at a time keeps a batch of 20 under 6 GB.

Speed comes from batching: generation is launch-bound (GPU ~20% busy at any
batch size), so the wall time of one call barely depends on how many lines
are in it. See DESIGN.md "Qwen3-TTS throughput".
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import soundfile as sf
import torch

_LANG = {"en": "English", "zh": "Chinese"}
_PRESETS = {"Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"}


class Models:
    """Lazy loader: only the variants actually used get loaded (each ~4 GB VRAM)."""

    def __init__(self, size: str, device: str, attn: str = "sdpa", decode_chunk: int = 1):
        self.size, self.device, self.attn, self.decode_chunk = size, device, attn, decode_chunk
        self._m: dict[str, object] = {}
        self._prompts: dict[tuple[str, float], list] = {}

    def get(self, variant: str):
        key = f"{self.size}-{variant}"
        if key not in self._m:
            from qwen_tts import Qwen3TTSModel

            model_id = f"Qwen/Qwen3-TTS-12Hz-{key}"
            print(f"[qwen3tts] loading {model_id} ({self.attn})", file=sys.stderr, flush=True)
            m = Qwen3TTSModel.from_pretrained(
                model_id, device_map=self.device, dtype=torch.bfloat16,
                attn_implementation=self.attn)
            _chunk_vocoder(m, self.decode_chunk)
            self._m[key] = m
        return self._m[key]

    def clone_prompt(self, ref: Path) -> list:
        """Encoded reference clip for the Base model, cached by path and mtime."""
        key = (str(ref), ref.stat().st_mtime)
        if key not in self._prompts:
            txt = ref.with_suffix(".txt")
            ref_text = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            self._prompts[key] = self.get("Base").create_voice_clone_prompt(
                ref_audio=str(ref), ref_text=ref_text or None, x_vector_only_mode=not ref_text)
        return self._prompts[key]


def _chunk_vocoder(model, n: int) -> None:
    """Vocode n sequences per codec call instead of the whole batch (same audio, flat VRAM)."""
    if n <= 0:
        return
    tok = model.model.speech_tokenizer
    orig = tok.decode

    def decode(items, *a, **kw):
        wavs, fs = [], None
        for i in range(0, len(items), n):
            w, fs = orig(items[i:i + n], *a, **kw)
            wavs.extend(w)
        return wavs, fs

    tok.decode = decode


def synthesize(models: Models, texts: list[str], voice: str, lang: str, p: dict) -> tuple[list, int]:
    gen = {"max_new_tokens": int(p.get("max_new_tokens", 2048))}
    if voice in _PRESETS:
        return models.get("CustomVoice").generate_custom_voice(
            text=texts, language=lang, speaker=voice, instruct=str(p.get("instruct", "")), **gen)
    ref = Path(voice)
    if not ref.exists():
        raise FileNotFoundError(f"voice is neither a preset nor a wav path: {voice}")
    return models.get("Base").generate_voice_clone(
        text=texts, language=lang, voice_clone_prompt=models.clone_prompt(ref), **gen)


def write(wav, out: str, sr: int) -> None:
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    sf.write(out, wav.reshape(-1).astype("float32"), sr)


def main() -> None:
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr  # keep the protocol channel clean

    def send(obj) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = Models(os.environ.get("QWEN3TTS_SIZE", "1.7B"), device)
    send({"ready": True, "sr": 24000, "device": device, "batch": True})  # 12Hz codec models output 24 kHz

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            p = req.get("params", {})
            if "size" in p:
                models.size = str(p["size"])
            if "attn" in p:
                models.attn = str(p["attn"])
            if "decode_chunk" in p:
                models.decode_chunk = int(p["decode_chunk"])
            if "seed" in p:
                torch.manual_seed(int(p["seed"]))
            lang = _LANG.get(req.get("lang", "en"), "Auto")
            voice = req.get("voice", "Vivian")
            with torch.inference_mode():
                if req.get("kind") == "design":
                    # VoiceDesign exists only at 1.7B.
                    saved, models.size = models.size, "1.7B"
                    try:
                        wavs, sr = models.get("VoiceDesign").generate_voice_design(
                            text=req["text"], language=lang, instruct=req["instruct"],
                            max_new_tokens=int(p.get("max_new_tokens", 2048)))
                    finally:
                        models.size = saved
                    write(wavs[0], req["out"], sr)
                    send({"ok": True, "path": req["out"], "sr": sr})
                elif req.get("kind") == "batch":
                    items = req["items"]
                    wavs, sr = synthesize(models, [it["text"] for it in items], voice, lang, p)
                    for it, wav in zip(items, wavs):
                        write(wav, it["out"], sr)
                    send({"ok": True, "paths": [it["out"] for it in items], "sr": sr})
                else:
                    wavs, sr = synthesize(models, [req["text"]], voice, lang, p)
                    write(wavs[0], req["out"], sr)
                    send({"ok": True, "path": req["out"], "sr": sr})
        except Exception as e:  # noqa: BLE001 - report everything to the parent
            send({"ok": False, "error": f"{e}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    main()
