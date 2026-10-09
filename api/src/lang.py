"""Language handling shared by STT and TTS.

Requests carry an ISO 639-1 code (``"fi"``, ``"en"``) or nothing / ``"auto"``,
which means "let the model decide" for STT and "use the configured voice" for TTS.
"""

from __future__ import annotations

import re

AUTO_VALUES = {"", "auto", "none", "detect"}

_LANG_RE = re.compile(r"^[a-z]{2,3}([-_][a-z0-9]{2,8})?$", re.IGNORECASE)

# Default Piper voice per language, used when a request asks for a language the
# configured Piper voice does not speak. Finnish voice per the FI benchmark
# (fi_FI-harri-medium, CER 3.5 %).
PIPER_VOICES: dict[str, str] = {
    "fi": "fi_FI-harri-medium",
    "en": "en_US-lessac-medium",
    "sv": "sv_SE-nst-medium",
    "de": "de_DE-thorsten-medium",
}

# Kokoro (KPipeline) lang codes — see kokoro.pipeline.LANG_CODES. No Finnish.
KOKORO_LANG_CODES: dict[str, str] = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "ja": "j",
    "zh": "z",
}


class UnsupportedLanguageError(ValueError):
    """The selected engine cannot synthesize/recognize the requested language."""


def normalize_language(value: str | None) -> str | None:
    """Return a lower-case language code, or None for auto-detect.

    Raises ValueError for values that do not look like a language code.
    """
    if value is None:
        return None
    v = value.strip().lower().replace("_", "-")
    if v in AUTO_VALUES:
        return None
    if not _LANG_RE.match(v):
        raise ValueError(f"Invalid language code: {value!r}")
    return v


def base_language(code: str | None) -> str | None:
    """``"en-gb"`` → ``"en"``."""
    if code is None:
        return None
    return code.split("-", 1)[0]


def piper_voice_language(model: str) -> str:
    """``"fi_FI-harri-medium"`` → ``"fi"``."""
    return model.split("_", 1)[0].split("-", 1)[0].lower()


def piper_voice_for(language: str) -> str:
    voice = PIPER_VOICES.get(base_language(language) or "")
    if voice is None:
        raise UnsupportedLanguageError(
            f"No Piper voice configured for language '{language}'. "
            f"Supported: {', '.join(sorted(PIPER_VOICES))}."
        )
    return voice


def kokoro_lang_code(language: str | None) -> str:
    if language is None:
        return "a"
    code = KOKORO_LANG_CODES.get(language) or KOKORO_LANG_CODES.get(base_language(language) or "")
    if code is None:
        raise UnsupportedLanguageError(
            f"Kokoro does not support language '{language}'. "
            f"Supported: {', '.join(sorted(k for k in KOKORO_LANG_CODES if '-' not in k))}. "
            "For Finnish use Piper (fi_FI-harri-medium)."
        )
    return code
