"""Chatterbox TTS worker. Runs in its own venv; speaks JSON lines on stdin/stdout.

Request:  {"text": str, "voice": str, "lang": "en"|"zh", "out": wav path, "params": {...}}
Response: {"ok": true, "path": wav path, "sr": int} | {"ok": false, "error": str}
Hello:    on startup the worker prints {"ready": true, "sr": int} once the model is loaded.

voice: "default" for the built-in voice, or a path to a reference wav (5-15 s).
params: exaggeration (0-1, default 0.5), cfg_weight (default 0.5),
        temperature (default 0.8), seed (int).
"""

from __future__ import annotations

import json
import sys
import traceback

import soundfile as sf
import torch

_LANG = {"en": "en", "zh": "zh"}


def main() -> None:
    # Libraries print progress bars and warnings to stdout. Keep the protocol
    # channel private: send on a dup of the original stdout, point sys.stdout
    # at stderr for everything else.
    import os

    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    def send(obj) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    sr = model.sr
    send({"ready": True, "sr": sr, "device": device})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            p = req.get("params", {})
            if "seed" in p:
                torch.manual_seed(int(p["seed"]))
            voice = req.get("voice", "default")
            kwargs = dict(
                language_id=_LANG.get(req.get("lang", "en"), "en"),
                exaggeration=float(p.get("exaggeration", 0.5)),
                cfg_weight=float(p.get("cfg_weight", 0.5)),
                temperature=float(p.get("temperature", 0.8)),
            )
            if voice and voice != "default":
                kwargs["audio_prompt_path"] = voice
            with torch.inference_mode():
                wav = model.generate(req["text"], **kwargs)
            audio = wav.squeeze(0).cpu().numpy().astype("float32")
            sf.write(req["out"], audio, sr)
            send({"ok": True, "path": req["out"], "sr": sr})
        except Exception as e:  # noqa: BLE001 - report everything to the parent
            send({"ok": False, "error": f"{e}\n{traceback.format_exc()}"})


if __name__ == "__main__":
    main()
