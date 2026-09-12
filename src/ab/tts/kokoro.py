"""Kokoro-82M backend. English (a/b voices) and Mandarin (z voices)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

_LANG_CODE = {"en": "a", "zh": "z"}  # kokoro: a=American, b=British, z=Mandarin


class KokoroBackend:
    name = "kokoro-82m-v1.0"
    max_chars = 400
    sample_rate = 24000
    languages: ClassVar[set[str]] = {"en", "zh"}

    def __init__(self, speed: float = 1.0, device: str | None = None, **_):
        from kokoro import KPipeline  # heavy import; deferred

        self.speed = speed
        self._pipelines: dict[str, KPipeline] = {}
        self._device = device
        self._KPipeline = KPipeline

    def _pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = self._KPipeline(lang_code=lang_code, device=self._device)
        return self._pipelines[lang_code]

    def voices(self) -> list[str]:
        return [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", "am_echo", "am_eric",
            "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
        ]

    def synthesize(self, text: str, voice: str, *, lang: str, speed: float | None = None, **_):
        if lang == "zh" and not voice.startswith("z"):
            raise ValueError(f"voice {voice!r} is not a Mandarin voice (z*)")
        if lang == "en" and voice.startswith("z"):
            raise ValueError(f"voice {voice!r} is a Mandarin voice; book language is en")
        code = "b" if (lang == "en" and voice.startswith("b")) else _LANG_CODE[lang]
        pipe = self._pipeline(code)
        chunks = [np.asarray(audio, dtype=np.float32)
                  for _, _, audio in pipe(text, voice=voice, speed=speed or self.speed)]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
