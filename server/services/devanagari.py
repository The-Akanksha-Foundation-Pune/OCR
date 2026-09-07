"""
Fold a Student ID handwritten in Devanagari into its Latin form.

On Hindi and Marathi consent forms a parent may write the ID in either
script. The two halves behave very differently:

* the six digits are a date of birth and map one-to-one (० -> 0), so they
  survive transliteration exactly;
* the six letters are a transliteration of the student's own name
  (MOHSHA = MOHammad SHAikh), and Devanagari-to-Latin is many-to-many
  there, so they only ever yield a rough guess.

So we trust the digits to shortlist and use the letters only to rank.
"""

from __future__ import annotations

import re
import unicodedata

from server.config import DEVANAGARI_DIGITS

_DIGIT_MAP = {ch: str(i) for i, ch in enumerate(DEVANAGARI_DIGITS)}

# Rough consonant/vowel readings, longest-first so conjuncts win over the
# single letters they contain.
_LETTER_MAP: list[tuple[str, str]] = [
    ("क्ष", "KSH"), ("ज्ञ", "GY"), ("श्र", "SHR"), ("त्र", "TR"),
    ("ख", "KH"), ("घ", "GH"), ("छ", "CH"), ("झ", "JH"), ("ठ", "TH"),
    ("ढ", "DH"), ("थ", "TH"), ("ध", "DH"), ("फ", "PH"), ("भ", "BH"),
    ("श", "SH"), ("ष", "SH"), ("च", "CH"), ("ऋ", "RI"),
    ("क", "K"), ("ग", "G"), ("ङ", "N"), ("ज", "J"), ("ञ", "N"),
    ("ट", "T"), ("ड", "D"), ("ण", "N"), ("त", "T"), ("द", "D"),
    ("न", "N"), ("प", "P"), ("ब", "B"), ("म", "M"), ("य", "Y"),
    ("र", "R"), ("ल", "L"), ("व", "V"), ("स", "S"), ("ह", "H"),
    ("ळ", "L"),
    ("आ", "A"), ("अ", "A"), ("इ", "I"), ("ई", "I"), ("उ", "U"),
    ("ऊ", "U"), ("ए", "E"), ("ऐ", "AI"), ("ओ", "O"), ("औ", "AU"),
    ("ा", "A"), ("ि", "I"), ("ी", "I"), ("ु", "U"), ("ू", "U"),
    ("े", "E"), ("ै", "AI"), ("ो", "O"), ("ौ", "AU"),
    ("ं", "N"), ("ः", ""), ("ँ", "N"), ("्", ""), ("़", ""),
]


def has_devanagari(text: str) -> bool:
    """True when the text carries any Devanagari character."""
    return any("ऀ" <= ch <= "ॿ" for ch in text or "")


def fold_devanagari_digits(text: str) -> str:
    """Replace Devanagari numerals with ASCII ones, leaving the rest alone."""
    if not text:
        return ""
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def transliterate(text: str) -> str:
    """
    Best-effort Devanagari -> Latin for the letter half of an ID.

    Deliberately lossy: it exists to rank a handful of database candidates
    that already share a digit suffix, never to produce an ID on its own.
    """
    if not text:
        return ""

    out = fold_devanagari_digits(unicodedata.normalize("NFC", text))
    for source, target in _LETTER_MAP:
        out = out.replace(source, target)
    # Anything Devanagari the table missed
    out = "".join(ch for ch in out if not ("ऀ" <= ch <= "ॿ"))
    return re.sub(r"[^A-Za-z0-9]", "", out).upper()


def split_id(text: str) -> tuple[str, str]:
    """
    Split a folded ID into (letters, digits); either half may be empty.

    Devanagari OCR strews stray characters through the letter half
    (ल0नडन०११०२१ folds to L0NDN011021), so a clean letters-then-digits
    split often fails. Since a real ID ends in exactly six digits, anchor
    on that trailing run and treat whatever precedes it as the letters —
    that recovers the reliable half from an otherwise messy reading.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()

    match = re.fullmatch(r"([A-Z]*)(\d*)", cleaned)
    if match:
        return match.group(1), match.group(2)

    tail = re.search(r"(\d{6})$", cleaned)
    if tail:
        head = re.sub(r"[^A-Z]", "", cleaned[: tail.start()])
        return head, tail.group(1)

    letters = "".join(ch for ch in cleaned if ch.isalpha())
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return letters, digits


def prefix_similarity(guess: str, actual: str) -> float:
    """
    Score a transliterated letter prefix against a real one.

    Uses rapidfuzz where the shapes differ in length, since a transliteration
    routinely runs long or short (KH for ख, one letter or two).
    """
    if not guess or not actual:
        return 0.0
    from rapidfuzz import fuzz

    return max(
        fuzz.ratio(guess, actual),
        fuzz.partial_ratio(guess, actual),
    ) / 100.0
