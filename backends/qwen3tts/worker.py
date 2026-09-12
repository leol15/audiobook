"""Qwen3-TTS worker. Runs in its own venv; speaks JSON lines on stdin/stdout.

Protocol is the same as backends/chatterbox/worker.py:
  Request:  {"text", "voice", "lang": "en"|"zh", "out": wav path, "params": {...}}
  Response: {"ok": true, "path", "sr"} | {"ok": false, "error"}
  Hello:    {"ready": true, "sr", "device"} after the first model loads.

voice:
  - a preset speaker name (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, ...)
    -> CustomVoice model
  - a path to a reference wav -> Base model, voice clone. If a .txt with the
    same stem exists next to it, its content is used as ref_text (better
    quality); otherwise x-vector-only cloning is used.
params:
  size: "1.7B" (default) | "0.6B"
  instruct: style/emotion instruction for CustomVoice (e.g. "calm, low voice")
  seed: int
  max_new_tokens: int (default 2048)
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

    def __init__(self, size: str, device: str):
        self.size, self.device = size, device
        self._m: dict[str, object] = {}

    def get(self, variant: str):
        if variant not in self._m:
            from qwen_tts import Qwen3TTSModel

            model_id = f"Qwen/Qwen3-TTS-12Hz-{self.size}-{variant}"
            print(f"[qwen3tts] loading {model_id}", file=sys.stderr, flush=True)
            self._m[variant] = Qwen3TTSModel.from_pretrained(
                model_id, device_map=self.device, dtype=torch.bfloat16,
                attn_implementation="sdpa",  # flash-attn not installed; sdpa is fine
            )
        return self._m[variant]


def main() -> None:
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr  # keep the protocol channel clean

    def send(obj) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = Models(os.environ.get("QWEN3TTS_SIZE", "1.7B"), device)
    send({"ready": True, "sr": 24000, "device": device})  # 12Hz codec models output 24 kHz

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            p = req.get("params", {})
            if "size" in p:
                models.size = str(p["size"])
            if "seed" in p:
                torch.manual_seed(int(p["seed"]))
            lang = _LANG.get(req.get("lang", "en"), "Auto")
            voice = req.get("voice", "Vivian")
            gen = {"max_new_tokens": int(p.get("max_new_tokens", 2048))}
            with torch.inference_mode():
                if voice in _PRESETS:
                    wavs, sr = models.get("CustomVoice").generate_custom_voice(
                        text=req["text"], language=lang, speaker=voice,
                        instruct=str(p.get("instruct", "")), **gen)
                else:
                    ref = Path(voice)
                    if not ref.exists():
                        raise FileNotFoundError(f"voice is neither a preset nor a wav path: {voice}")
                    txt = ref.with_suffix(".txt")
                    kw = {"ref_audio": str(ref)}
                    if txt.exists():
                        kw["ref_text"] = txt.read_text(encoding="utf-8").strip()
                    else:
                        kw["ref_text"] = ""
                        kw["x_vector_only_mode"] = True
                    wavs, sr = models.get("Base").generate_voice_clone(
                        text=req["text"], language=lang, **kw, **gen)
            audio = wavs[0]
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            audio = audio.reshape(-1).astype("float32")
            sf.write(req["out"], audio, sr)
            send({"ok": True, "path": req["out"], "sr": sr})
        except Exception as e:  # noqa: BLE001 - report everything to the parent
            send({"ok": False, "error": f"{e}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    main()
