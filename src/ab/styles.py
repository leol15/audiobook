"""Per-line delivery styles: a small fixed label set shared by attribute
(the LLM picks one per quote), `ab voices-design` (one clip per character
per style) and render (voice key `<role>@<style>`).

Labels are English words in every language; the phrases here are what the
VoiceDesign model is told, appended to the character's description.
"""

from __future__ import annotations

NEUTRAL = "neutral"

# label -> VoiceDesign phrase per language
PHRASES: dict[str, dict[str, str]] = {
    "angry": {"en": "angry, raised voice, sharp and forceful",
              "zh": "愤怒，提高音量，语气尖锐有力"},
    "sad": {"en": "sad, low, slow and subdued",
            "zh": "悲伤，低沉，缓慢而压抑"},
    "soft": {"en": "soft, gentle and quiet, almost a whisper",
             "zh": "轻柔，温和，小声，近乎耳语"},
    "urgent": {"en": "urgent, fast and tense, out of breath",
               "zh": "急促，紧张，语速很快，带着喘息"},
}
DEFAULT_STYLES = list(PHRASES)

# Shown to the attribute LLM next to each label.
GLOSS: dict[str, dict[str, str]] = {
    "en": {"angry": "angry or shouting", "sad": "sad or grieving", "soft": "gentle, quiet, tender",
           "urgent": "hurried, alarmed, tense"},
    "zh": {"angry": "愤怒、吼叫", "sad": "悲伤、哀痛", "soft": "温柔、轻声、体贴",
           "urgent": "急促、惊慌、紧张"},
}


def phrase(style: str, lang: str) -> str:
    p = PHRASES.get(style)
    return p.get(lang, p["en"]) if p else style


def voice_key(role: str, style: str | None) -> str:
    """book.yaml voice-map key for a role in a style: `林风@angry`; neutral is the bare role."""
    return f"{role}@{style}" if style and style != NEUTRAL else role


def split_key(key: str) -> tuple[str, str | None]:
    role, sep, style = key.rpartition("@")
    return (role, style) if sep else (key, None)
