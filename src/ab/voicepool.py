"""Built-in voice catalogs with the metadata needed to pick and validate voices.

Kokoro ships no metadata, so gender/age/character are hand-annotated from
listening. Qwen3-TTS presets come from the model card. `chatterbox` has only
a default voice; anything else must be a reference clip.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    id: str
    lang: str          # "en" | "zh"
    gender: str        # "f" | "m"
    age: str           # "young" | "adult" | "old"
    narrator: bool = False   # good default narrator
    note: str = ""


KOKORO = [
    Voice("af_heart", "en", "f", "young", note="warm, expressive"),
    Voice("af_bella", "en", "f", "young", note="bright"),
    Voice("af_nicole", "en", "f", "young", note="soft, breathy; avoid for narration"),
    Voice("af_sarah", "en", "f", "adult", note="even, clear"),
    Voice("af_sky", "en", "f", "young", note="light"),
    Voice("am_adam", "en", "m", "adult", note="flat, plain"),
    Voice("am_michael", "en", "m", "adult", True, note="steady; good narrator"),
    Voice("am_echo", "en", "m", "young", note="youthful"),
    Voice("am_eric", "en", "m", "adult", note="firm"),
    Voice("bf_emma", "en", "f", "adult", True, note="British; good narrator"),
    Voice("bf_isabella", "en", "f", "adult", note="British, gentle"),
    Voice("bm_george", "en", "m", "old", True, note="British; good narrator"),
    Voice("bm_lewis", "en", "m", "adult", note="British, dry"),
    Voice("zf_xiaobei", "zh", "f", "young", note="bright"),
    Voice("zf_xiaoni", "zh", "f", "young", note="gentle"),
    Voice("zf_xiaoxiao", "zh", "f", "adult", True, note="even; usable narrator"),
    Voice("zf_xiaoyi", "zh", "f", "adult", note="warm"),
    Voice("zm_yunjian", "zh", "m", "adult", note="deep"),
    Voice("zm_yunxi", "zh", "m", "young", note="clear"),
    Voice("zm_yunxia", "zh", "m", "young", note="boyish"),
    Voice("zm_yunyang", "zh", "m", "adult", True, note="steady; good narrator"),
]

QWEN3TTS = [
    Voice("Vivian", "zh", "f", "young", note="bright, slightly edgy"),
    Voice("Serena", "zh", "f", "young", note="warm, gentle"),
    Voice("Uncle_Fu", "zh", "m", "old", True, note="low, mellow; good narrator"),
    Voice("Dylan", "zh", "m", "young", note="Beijing accent, clear"),
    Voice("Eric", "zh", "m", "adult", note="Sichuan accent, lively"),
    Voice("Ryan", "en", "m", "adult", True, note="dynamic; good narrator"),
    Voice("Aiden", "en", "m", "young", note="sunny American"),
    Voice("Ono_Anna", "ja", "f", "young"),
    Voice("Sohee", "ko", "f", "adult"),
]

POOLS: dict[str, list[Voice]] = {"kokoro": KOKORO, "qwen3tts": QWEN3TTS}
# Backends whose voice ids may be reference-clip paths (validated as files).
CLIP_BACKENDS = {"qwen3tts", "chatterbox"}
# Literal ids that are always valid for a backend.
SPECIAL = {"chatterbox": {"default"}, "tone": {"a220", "a330", "a440"}}


def pool(backend: str, lang: str) -> list[Voice]:
    return [v for v in POOLS.get(backend, []) if v.lang == lang]


def known_ids(backend: str) -> set[str]:
    return {v.id for v in POOLS.get(backend, [])} | SPECIAL.get(backend, set())
