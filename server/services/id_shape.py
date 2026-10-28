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


# Every shape a real Student ID takes, measured across all 16,749 rows of
# active_student_data: 16,576 are six letters and six digits, and the rest
# are five or six letters and six digits with a separator or a short
# trailing suffix (a letter, or "_" + letter + digit). One character either
# way on each half leaves room for a reading that dropped or doubled one.
#
# This is what keeps a parent's name, email or mobile number - all of which
# sit a few lines below the Student ID on the form - from ever being taken
# for one: a run of ten phone digits, an "@" fragment or a bare name cannot
# fit it.
_STUDENT_ID_SHAPE = re.compile(r"[A-Z]{4,7}\d{5,7}[A-Z]?\d?")
_ID_SEPARATORS = re.compile(r"[. _-]")

# Six digits at the end with no seventh in front: the date of birth intact.
_FULL_DATE_TAIL = re.compile(r"(?<!\d)\d{6}$")


def looks_like_student_id(text: str) -> bool:
    """Whether a reading has the shape of an Akanksha Student ID.

    Separators (".", space, "_", "-") are ignored and case does not matter,
    so ``"aadmis.310714"`` and ``"AADMIS310714_A1"`` both pass while a name,
    an email address or a phone number does not.

    Args:
        text: The reading to test, as OCR or a person produced it.

    Returns:
        True when the reading is four to seven letters followed by five to
        seven digits, with at most one trailing letter and one trailing digit.
    """
    compact = _ID_SEPARATORS.sub("", (text or "").upper())
    return bool(_STUDENT_ID_SHAPE.fullmatch(compact))


def as_student_id(text: str) -> str:
    """The reading as a Student ID, or "" when it cannot be one.

    A reading that already has the shape is returned as it is. One that
    only fails because of a shape confusion (``M0HSHA011021``, a zero for an
    O in the name half) is returned corrected, since the lookup would make
    the same correction anyway - turning it away would throw out exactly the
    readings this module exists to rescue.

    A correction is only accepted when it leaves six digits at the end.
    The date of birth is the half the database lookup anchors on, and an
    off-length reading offers more than one split: L0NDN011021 becomes
    LONDN011021, not LONDNO11021, and a truncated MOHSHA1021 is not dressed
    up as MOHSH41021.

    Args:
        text: A reading from OCR or typed by a person.

    Returns:
        The reading, upper-cased, in a shape that passes
        :func:`looks_like_student_id`, or ``""`` when neither the reading
        nor any of its shape corrections fits.
    """
    reading = (text or "").strip().upper()
    if looks_like_student_id(reading):
        return reading

    for variant in shape_variants(reading):
        if looks_like_student_id(variant) and _FULL_DATE_TAIL.search(variant):
            return variant
    return ""


def loggable_reading(text: str) -> str:
    """How a reading may appear in logs/ocr_debug.log.

    A crop that strays off the Student ID field reads back the parent's
    name, email address or mobile number, and the debug log is kept on disk
    indefinitely. A reading that is (or shape-corrects to) a Student ID is
    worth seeing as read, misreads and all; anything else is described by
    its length only.

    Args:
        text: A reading from any recogniser, or a printed line's OCR text.

    Returns:
        ``repr(text)`` when :func:`as_student_id` accepts it, otherwise a
        placeholder that gives the length and none of the characters.
    """
    if as_student_id(text):
        return repr(text)
    return f"<not an ID: {len(text or '')} chars>"
