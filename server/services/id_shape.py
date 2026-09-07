"""
Repair an OCR reading using the fixed shape of an Akanksha Student ID.

Every ID in active_student_data is six letters followed by six digits
(MOHSHA011021) - 16,340 of 16,352 of them, the rest a near-miss on length.
That constraint is worth far more than it looks.

OCR confuses O/0, I/1, S/5, L/1, B/8 constantly, and the pipeline used to
apply one blanket confusion map before fuzzy-matching. But the direction of
the mistake is not ambiguous once you know where the character sits: in the
first six positions a digit is certainly a misread letter, and in the last
six a letter is certainly a misread digit. Applying that turned 18 character
errors into 4 across the readings measured, and made two of them exact.

What remains after this are genuine letter-shape confusions in the name half
(X for Y, Q for R). Those the database lookup can resolve, because by then
the six digits - a date of birth, and the discriminating half - are right.
"""

from __future__ import annotations

import re

ID_LETTERS = 6
ID_DIGITS = 6
ID_LENGTH = ID_LETTERS + ID_DIGITS

# Digit shapes that are really letters, for the name half of the ID.
_AS_LETTER = str.maketrans(
    {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "4": "A"}
)

# Letter shapes that are really digits, for the date half.
_AS_DIGIT = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
        "G": "6",
        "T": "7",
        "A": "4",
    }
)


def _clean(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", token or "").upper()


def apply_shape(token: str, split: int = ID_LETTERS) -> str:
    """Force `split` letters followed by digits, fixing shape confusions."""
    cleaned = _clean(token)
    head, tail = cleaned[:split], cleaned[split:]
    return head.translate(_AS_LETTER) + tail.translate(_AS_DIGIT)


def shape_variants(token: str) -> list[str]:
    """
    Plausible corrections of an OCR reading, best first.

    A full-length reading has one obvious split. A short or long one has
    lost or gained a character and we do not know which half, so both
    neighbouring splits are offered and the database decides between them.
    """
    cleaned = _clean(token)
    if len(cleaned) < 8:
        return []

    if len(cleaned) == ID_LENGTH:
        splits = [ID_LETTERS]
    else:
        # Off-length: the drop could be in either half.
        splits = [ID_LETTERS, ID_LETTERS - 1, ID_LETTERS + 1]

    seen: list[str] = []
    for split in splits:
        if split < 1 or split >= len(cleaned):
            continue
        candidate = apply_shape(cleaned, split)
        if candidate and candidate not in seen and candidate != cleaned:
            seen.append(candidate)
    return seen


def is_canonical(token: str) -> bool:
    """True when the token already has the exact 6-letter, 6-digit shape."""
    return bool(re.fullmatch(r"[A-Z]{6}\d{6}", _clean(token)))
