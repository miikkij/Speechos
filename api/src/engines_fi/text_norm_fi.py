"""Finnish text normalization for TTS input: numbers, dates, money, units, phone numbers, abbreviations,
acronyms, e-mail.

Local Finnish TTS engines (Piper, Chatterbox-Finnish, MMS) mostly fail on digits and abbreviations
("12 A", "24.", "49,90", "TTS-mallille"), see docs/plans/2026-10-public/report-b.md. This module turns
them into words before synthesis. It is deliberately rule-based and conservative: anything it does not
recognise is left as is.

Case handling (Finnish numerals inflect): dates after a weekday/time word in the essive ("tiistaina
24. maaliskuuta") get the essive ordinal; a number directly before a noun ending in -ssa/-ssä (inessive)
or a longer word in -lla/-llä (adessive) or vowel+n (genitive) copies that case; everything else is
nominative ("49 euroa", "12 A"). Wrong-case output is still far more intelligible than digits.

    >>> normalize_fi("Maksu oli yhteensä 49,90 euroa.")
    'Maksu oli yhteensä neljäkymmentäyhdeksän euroa yhdeksänkymmentä senttiä.'

Requires `num2words` (extra `tts-fi`).
"""

from __future__ import annotations

import re

from num2words import num2words

__all__ = ["normalize_fi"]

ABBREVIATIONS = {
    "esim.": "esimerkiksi",
    "Esim.": "Esimerkiksi",
    "mm.": "muun muassa",
    "jne.": "ja niin edelleen",
    "yms.": "ynnä muuta sellaista",
    "ym.": "ynnä muuta",
    "ns.": "niin sanottu",
    "ks.": "katso",
    "vrt.": "vertaa",
    "n.": "noin",
    "klo": "kello",
    "tri": "tohtori",
    "prof.": "professori",
    "nro": "numero",
    "puh.": "puhelin",
    "t.": "terveisin",
}

LETTER_NAMES = {
    "A": "aa", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "äf", "G": "gee", "H": "hoo",
    "I": "ii", "J": "jii", "K": "koo", "L": "äl", "M": "äm", "N": "än", "O": "oo", "P": "pee",
    "Q": "kuu", "R": "är", "S": "äs", "T": "tee", "U": "uu", "V": "vee", "W": "kaksoisvee", "X": "äks",
    "Y": "yy", "Z": "tset", "Å": "ruotsalainen oo", "Ä": "ää", "Ö": "öö",
}
# Acronyms that are read as words, not letter by letter.
ACRONYM_WORDS = {"NATO", "YLE", "KELA", "SAK"}
SPELLED_OVERRIDE = {"EU": "ee uu"}

MONTHS_PARTITIVE = ("tammikuuta|helmikuuta|maaliskuuta|huhtikuuta|toukokuuta|kesäkuuta|heinäkuuta|"
                    "elokuuta|syyskuuta|lokakuuta|marraskuuta|joulukuuta")

_PHONE = re.compile(r"(?<![\w+])(?:\+358 ?|0)\d{1,3}(?:[ -]?\d{2,4}){1,3}\b")
DIGITS = ["nolla", "yksi", "kaksi", "kolme", "neljä", "viisi", "kuusi", "seitsemän", "kahdeksan", "yhdeksän"]
# Words ending in vowel+n that are not genitive nouns ("ennen kello 16", "5 vuotta sitten").
_NOT_GENITIVE = {"ennen", "jälkeen", "sitten", "kuin", "aikaan", "myöhemmin", "aiemmin", "aikaisemmin", "pian"}
_EMAIL = re.compile(r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b")
_TIME = re.compile(r"\b(kello)\s+(\d{1,2})[.:](\d{2})\b")
_DATE = re.compile(rf"(\S+\s+)?\b(\d{{1,2}})\.\s+({MONTHS_PARTITIVE})\b")
_MONEY = re.compile(r"\b(\d+),(\d{2})\s*(?:€|euroa)")
_EURO_SIGN = re.compile(r"\b(\d+)\s*€")
_PERCENT = re.compile(r"\b(\d+(?:,\d+)?)\s*%")
_DECIMAL = re.compile(r"\b(\d+),(\d+)\b")
_NUM_LETTER = re.compile(r"\b(\d+)\s+([A-ZÅÄÖ])\b(?!-)")
_INT_BEFORE_WORD = re.compile(r"\b(\d+)(\s+)([a-zåäö]+)")
_INT = re.compile(r"\b\d+\b")
_ACRONYM = re.compile(r"\b([A-ZÅÄÖ]{2,5})\b")


def _card(n: int, case: str = "nominative") -> str:
    return num2words(n, lang="fi", case=case)


def _ord(n: int, case: str = "nominative") -> str:
    return num2words(n, lang="fi", to="ordinal", case=case)


def _noun_case(word: str) -> str:
    """Very small, high-precision case guess from a following noun."""
    if re.search(r"(ssa|ssä)$", word):
        return "inessive"
    if re.search(r"(lla|llä)$", word) and len(word) > 5:
        return "adessive"
    if re.search(r"[aeiouyäö]n$", word) and len(word) > 4 and word not in _NOT_GENITIVE:  # "3 kilometrin"
        return "genitive"
    return "nominative"


def _phone(m: re.Match) -> str:
    """Phone numbers digit by digit, groups separated by commas: "040 123" → "nolla neljä nolla, yksi kaksi kolme"."""
    raw, parts = m.group(0), []
    if raw.startswith("+358"):
        parts.append("plus kolme viisi kahdeksan")
        raw = raw[4:]
    parts += [" ".join(DIGITS[int(c)] for c in g) for g in re.split(r"[ -]+", raw.strip()) if g]
    return ", ".join(parts)


def _email(m: re.Match) -> str:
    local = " piste ".join(m.group(1).split("."))
    domain = " piste ".join(m.group(2).split("."))
    return f"{local} ät {domain}"


def _date(m: re.Match) -> str:
    prev, day, month = m.group(1) or "", int(m.group(2)), m.group(3)
    case = "essive" if re.search(r"(na|nä)\s+$", prev) else "nominative"
    return f"{prev}{_ord(day, case)} {month}"


def _money(m: re.Match) -> str:
    euros, cents = int(m.group(1)), int(m.group(2))
    text = f"{_card(euros)} euroa"
    if cents:
        text += f" {_card(cents)} senttiä"
    return text


def _decimal(m: re.Match) -> str:
    frac = m.group(2)
    frac_words = _card(int(frac)) if not frac.startswith("0") else " ".join(_card(int(d)) for d in frac)
    return f"{_card(int(m.group(1)))} pilkku {frac_words}"


def _acronym(m: re.Match) -> str:
    word = m.group(1)
    if word in SPELLED_OVERRIDE:
        return SPELLED_OVERRIDE[word]
    if word in ACRONYM_WORDS:
        return word
    return " ".join(LETTER_NAMES.get(ch, ch) for ch in word)


def normalize_fi(text: str) -> str:
    """Return `text` with numbers, money, dates, abbreviations, acronyms and e-mails spelled out."""
    text = _EMAIL.sub(_email, text)
    text = _PHONE.sub(_phone, text)
    for abbr, full in ABBREVIATIONS.items():
        # a sentence-final abbreviation keeps the full stop it was carrying ("… jne." → "… edelleen.")
        text = re.sub(rf"(?<![\w.]){re.escape(abbr)}(?=\s|$)",
                      lambda m, full=full, abbr=abbr: full + ("." if abbr.endswith(".") and m.end() == len(m.string)
                                                              else ""), text)
    text = _TIME.sub(lambda m: f"{m.group(1)} {_card(int(m.group(2)))} {_card(int(m.group(3)))}"
                     if int(m.group(3)) else f"{m.group(1)} {_card(int(m.group(2)))}", text)
    text = _DATE.sub(_date, text)
    text = _MONEY.sub(_money, text)
    text = _EURO_SIGN.sub(lambda m: f"{_card(int(m.group(1)))} euroa", text)
    text = _PERCENT.sub(lambda m: f"{m.group(1)} prosenttia", text)
    text = _DECIMAL.sub(_decimal, text)
    text = _NUM_LETTER.sub(lambda m: f"{_card(int(m.group(1)))} {LETTER_NAMES.get(m.group(2), m.group(2))}", text)
    text = _INT_BEFORE_WORD.sub(lambda m: f"{_card(int(m.group(1)), _noun_case(m.group(3)))}{m.group(2)}{m.group(3)}",
                                text)
    text = _INT.sub(lambda m: _card(int(m.group(0))), text)
    # Acronyms glued to a word with a hyphen ("TTS-mallille") are spelled and the hyphen kept as a space.
    text = re.sub(r"\b([A-ZÅÄÖ]{2,5})-(?=\w)", lambda m: _acronym(m) + " ", text)
    text = _ACRONYM.sub(_acronym, text)
    return re.sub(r"\s{2,}", " ", text).strip()
