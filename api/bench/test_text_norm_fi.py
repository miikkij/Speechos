"""Tests for src/engines_fi/text_norm_fi.py (run: uv run --extra tts-fi pytest bench/test_text_norm_fi.py)."""

import pytest

pytest.importorskip("num2words")

from src.engines_fi.text_norm_fi import normalize_fi  # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("Maksu oli yhteensä 49,90 euroa.", "Maksu oli yhteensä neljäkymmentäyhdeksän euroa yhdeksänkymmentä senttiä."),
    ("Tapaaminen on sovittu tiistaina 24. maaliskuuta.",
     "Tapaaminen on sovittu tiistaina kahdentenakymmenentenäneljäntenä maaliskuuta."),
    ("Se oli 6. joulukuuta.", "Se oli kuudes joulukuuta."),
    ("Osoite on Mannerheimintie 12 A, Helsinki.", "Osoite on Mannerheimintie kaksitoista aa, Helsinki."),
    ("Esim. tämä lyhenne saattaa olla TTS-mallille hankala.",
     "Esimerkiksi tämä lyhenne saattaa olla tee tee äs mallille hankala."),
    ("Sähköposti lähetettiin osoitteeseen asiakaspalvelu@example.fi.",
     "Sähköposti lähetettiin osoitteeseen asiakaspalvelu ät example piste fi."),
    ("Kokous on klo 14.30 huoneessa 5.", "Kokous on kello neljätoista kolmekymmentä huoneessa viisi."),
    ("Hinta nousi 12 % ja maksaa 20 €.", "Hinta nousi kaksitoista prosenttia ja maksaa kaksikymmentä euroa."),
    ("Asun 3 kilometrin päässä.", "Asun kolmen kilometrin päässä."),
    ("Omenoita, päärynöitä jne.", "Omenoita, päärynöitä ja niin edelleen."),
    ("Luku on 3,05.", "Luku on kolme pilkku nolla viisi."),
    ("Asuin 1990-luvulla 2 vuotta Turussa.", "Asuin tuhat yhdeksänsataayhdeksänkymmentä-luvulla kaksi vuotta Turussa."),
])
def test_normalize_fi(raw: str, expected: str) -> None:
    assert normalize_fi(raw) == expected


@pytest.mark.parametrize("text", ["Joo, ymmärrän.", "Hei, miten voit tänään?", "Kerro lisää."])
def test_plain_text_untouched(text: str) -> None:
    assert normalize_fi(text) == text
