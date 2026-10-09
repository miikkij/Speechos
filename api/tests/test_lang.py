from __future__ import annotations

import pytest

from src.lang import (
    UnsupportedLanguageError,
    kokoro_lang_code,
    normalize_language,
    piper_voice_for,
    piper_voice_language,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("auto", None), ("AUTO", None), ("fi", "fi"), (" EN ", "en"), ("en_GB", "en-gb")],
)
def test_normalize(raw, expected):
    assert normalize_language(raw) == expected


@pytest.mark.parametrize("raw", ["finnish!", "f", "../etc", "fi en"])
def test_normalize_rejects(raw):
    with pytest.raises(ValueError):
        normalize_language(raw)


def test_piper_voices():
    assert piper_voice_for("fi") == "fi_FI-harri-medium"
    assert piper_voice_for("en-gb") == "en_US-lessac-medium"
    assert piper_voice_language("fi_FI-harri-medium") == "fi"
    with pytest.raises(UnsupportedLanguageError):
        piper_voice_for("xh")


def test_kokoro_codes():
    assert kokoro_lang_code(None) == "a"
    assert kokoro_lang_code("en-gb") == "b"
    assert kokoro_lang_code("ja") == "j"
    with pytest.raises(UnsupportedLanguageError, match="Finnish use Piper"):
        kokoro_lang_code("fi")
