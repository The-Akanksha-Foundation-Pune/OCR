"""Read the handwritten batch code in the corner of a scanned form page.

Staff write a code such as ``B1-12`` in the top-right corner of every page of
a form, where the printed forms already carry a circled serial: ``B`` and the
batch number, a dash, and the form's number within that batch. Both pages of
a two-part form (the Student ID page and the signature page) carry the same
code, which is how the app knows they belong together.

Handwriting and OCR mangle the code in a handful of predictable ways - a
``B`` read as ``8`` or ``3``, a ``1`` read as ``I``, ``l`` or ``|``, an ``O``
for a zero, any of several dashes or no dash at all - so the parser forgives
exactly those and refuses everything else. A code it is unsure of is not
guessed at: the page goes to a person with its image instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from server.config import BATCH_CODE_FALLBACK_REGION, BATCH_CODE_REGION
from server.services.debug_log import debug_log

# Upper bound for the batch and for the number within it. Three digits is
# far more than a day's scanning, and a longer run of digits is far more
# likely to be a misread phone number or date than a code.
MAX_CODE_PART = 999

# What a handwritten "B" comes back as. Only ever at the very start, where a
# digit cannot be part of the code itself.
_B_LOOKALIKES = frozenset("B83")

# Characters read for a digit: "1" as I, l (upper-cased to L) or |, and "0"
# as O. Nothing else is turned into a digit - "S" for 5 or "Z" for 2 would
# make almost any word parse.
_AS_DIGIT = str.maketrans({"I": "1", "L": "1", "|": "1", "O": "0"})

# What a handwritten dash is read as. A space is handled separately, since
# spaces are also ignored inside a number.
_DASHES = "-‐‑‒–—―−~_."

# Unicode names of the enclosed digits a recogniser may return for the
# printed serial: (3), a circled 3, and the dingbat and negative forms.
_ENCLOSED_PREFIXES = ("CIRCLED", "NEGATIVE CIRCLED", "DOUBLE CIRCLED", "DINGBAT", "PARENTHESIZED")

# Below this share of dark pixels a corner crop holds no writing at all, so
# it is not worth sending to a recogniser (or to Google).
_MIN_CORNER_INK = 0.002


@dataclass(frozen=True)
class BatchCode:
    """A parsed batch code: ``B<batch>-<number>``.

    Attributes:
        batch: The batch number, the digits after the B.
        number: The form's number within the batch, after the dash.
    """

    batch: int
    number: int

    @property
    def text(self) -> str:
        """The code as staff write it, e.g. ``B1-12``."""
        return f"B{self.batch}-{self.number}"

    @property
    def folder(self) -> str:
        """The name of this code's batch folder, e.g. ``B1``."""
        return f"B{self.batch}"

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class CodeReading:
    """What was read from a page's corner.

    Attributes:
        code: The parsed code, or None when nothing readable was there.
        confidence: The recogniser's confidence in the text that parsed.
        raw: The text that parsed, or the first text read when none did.
        region: Which corner it came from ("top-right" or "top-left"), or
            None when no code was found.
    """

    code: BatchCode | None
    confidence: float = 0.0
    raw: str = ""
    region: str | None = None


def parse_batch_code(text: str | None) -> BatchCode | None:
    """Parse a batch code as written or as OCR read it.

    Accepts ``B1-1``, ``b1-12``, ``81-1`` (B read as 8), ``31 - 5`` (B read
    as 3), ``BI-I2`` and ``Bl|-1`` (1 read as I, l or |), ``B1O-1`` (0 read
    as O), and a dash written or read as any of ``-``, en or em dash, ``~``,
    ``_``, ``.`` or a plain space. Spaces anywhere else are ignored.

    Args:
        text: The text to parse.

    Returns:
        The code, or None when the text is anything other than one code
        with a batch and a number each from 1 to 999.
    """
    if not text:
        return None
    # NFKC folds full-width and other compatibility digits to ASCII.
    folded = unicodedata.normalize("NFKC", text).upper().strip()
    if not folded or folded[0] not in _B_LOOKALIKES:
        return None
    body = folded[1:]
    for dash in _DASHES:
        body = body.replace(dash, "-")

    if "-" in body:
        # A dash doubled by a shaky hand ("B1--2") is still one dash; two
        # separate ones ("B1-2-3") are not a code.
        parts = re.split(r"\s*-+\s*", body.strip())
    else:
        # No dash: the gap between the two numbers is the separator, so
        # there must be exactly one.
        parts = body.split()
    if len(parts) != 2:
        return None

    batch = _as_number(parts[0])
    number = _as_number(parts[1])
    if batch is None or number is None:
        return None
    return BatchCode(batch=batch, number=number)


def _as_number(chunk: str) -> int | None:
    """One half of a code as a number from 1 to MAX_CODE_PART, or None."""
    compact = re.sub(r"\s+", "", chunk).translate(_AS_DIGIT)
    if not compact or not compact.isascii() or not compact.isdigit():
        return None
    if len(compact) > len(str(MAX_CODE_PART)):
        return None
    value = int(compact)
    if not 1 <= value <= MAX_CODE_PART:
        return None
    return value


def crop_region(image: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    """Crop a page to a region given as fractions (x0, y0, x1, y1) of its size.

    Args:
        image: The page, greyscale or BGR.
        region: Left, top, right and bottom as fractions of width and height.

    Returns:
        The cropped part of the page (a view, not a copy).
    """
    height, width = image.shape[:2]
    x0, y0, x1, y1 = region
    left, right = round(x0 * width), round(x1 * width)
    top, bottom = round(y0 * height), round(y1 * height)
    return image[max(0, top) : max(top + 1, bottom), max(0, left) : max(left + 1, right)]


def read_batch_code(page: np.ndarray, *, allow_vision: bool = True) -> CodeReading:
    """Read the batch code off an upright page.

    The top-right corner is read first, where the forms carry their circled
    serial; the top-left is tried after it, since some prints put the serial
    there. The first text that parses wins.

    Args:
        page: The upright page, greyscale or BGR.
        allow_vision: False when `page` may not really be upright - a page
            turned round on the chance it was scanned upside down. Its
            "top" corners may then be the bottom of the form, which holds
            signatures and names, so they are read on this machine only.

    Returns:
        What was read. ``code`` is None when neither corner held a code.
    """
    first_raw = ""
    for name, region in (("top-right", BATCH_CODE_REGION), ("top-left", BATCH_CODE_FALLBACK_REGION)):
        crop = crop_region(page, region)
        if _ink_share(crop) < _MIN_CORNER_INK:
            continue
        readings = recognise_corner_text(crop, allow_vision=allow_vision)
        for text, confidence in candidate_texts(readings):
            code = parse_batch_code(text)
            if code is not None:
                debug_log(f"[CODE] {name} corner read {code.text} (conf={confidence:.2f})")
                return CodeReading(code=code, confidence=confidence, raw=text, region=name)
        if readings and not first_raw:
            first_raw = readings[0][0]
    debug_log(f"[CODE] no batch code found ({len(first_raw)} chars read)")
    return CodeReading(code=None, raw=first_raw[:40])


def candidate_texts(readings: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """The texts to try parsing, in the order to try them.

    The corner also holds the form's printed serial (a circled 3, say),
    which a recogniser returns beside the code - and "B1-12 3" would parse
    as B1-123, since spaces are allowed inside a number. So circled digits
    are dropped first, then each word is tried on its own, then each line
    or box whole (for a code read with gaps, "B 1 - 12"), and last runs of
    words across boxes, shortest first ("B1" and "-12" read as two boxes).

    Args:
        readings: (text, confidence) per line or box, in reading order.

    Returns:
        (text, confidence) pairs, each text once.
    """
    lines = [(_drop_serials(text), conf) for text, conf in readings]
    lines = [(text, conf) for text, conf in lines if text]
    words = [(word, conf) for text, conf in lines for word in text.split()]
    out: list[tuple[str, float]] = []
    seen: set[str] = set()

    def offer(text: str, conf: float) -> None:
        if text not in seen:
            seen.add(text)
            out.append((text, conf))

    for word, conf in words:
        offer(word, conf)
    for text, conf in lines:
        offer(text, conf)
    for size in range(2, min(len(words), 6) + 1):
        for start in range(len(words) - size + 1):
            window = words[start : start + size]
            offer(" ".join(w for w, _ in window), sum(c for _, c in window) / size)
    return out


def _drop_serials(text: str) -> str:
    """Text with circled and bracketed digits (the printed serial) made spaces."""
    kept = "".join(" " if _is_enclosed_digit(ch) else ch for ch in text or "")
    return re.sub(r"\s+", " ", kept).strip()


def _is_enclosed_digit(char: str) -> bool:
    """Whether a character is a digit in a circle or brackets (one character)."""
    if char.isascii() or unicodedata.category(char) != "No":
        return False
    return unicodedata.name(char, "").startswith(_ENCLOSED_PREFIXES)


def recognise_corner_text(crop: np.ndarray, *, allow_vision: bool = True) -> list[tuple[str, float]]:
    """Every text a recogniser reads in a corner crop, in reading order.

    Google Vision reads it when configured and allowed, since it reads
    handwriting far better; the local RapidOCR model reads it otherwise, or
    when Vision returns nothing. The top corner of an upright form holds the
    code and the printed serial, never any of the form's personal details,
    so that crop may be sent to Vision - and only that crop.

    Args:
        crop: The corner crop, greyscale or BGR.
        allow_vision: False keeps the crop on this machine: it is not known
            to be the top of the form (see read_batch_code).

    Returns:
        (text, confidence) per line (Vision) or box (RapidOCR).
    """
    bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if crop.ndim == 2 else crop
    # Handwriting at 150 dpi is small for either recogniser; doubling it
    # costs little on a crop this size and reads noticeably better.
    if max(bgr.shape[:2]) < 600:
        bgr = cv2.resize(bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    from server.services import vision_ocr

    if allow_vision and vision_ocr.is_configured():
        readings = vision_ocr.recognize_text(bgr)
        if readings:
            return readings
    return _rapid_texts(bgr)


def _rapid_texts(crop_bgr: np.ndarray) -> list[tuple[str, float]]:
    """RapidOCR's boxes in a crop, top to bottom and left to right."""
    try:
        from server.services.field_detector import get_rapid_ocr

        result, _ = get_rapid_ocr()(crop_bgr)
    except Exception as exc:  # noqa: BLE001 - a recogniser failure is "nothing read"
        debug_log(f"[CODE] RapidOCR failed: {type(exc).__name__}: {exc}")
        return []
    def reading_order(item: Any) -> tuple[int, float]:
        xs = [point[0] for point in item[0]]
        ys = [point[1] for point in item[0]]
        # Boxes whose middles are within ~20 px are one line.
        return (int((min(ys) + max(ys)) / 2 // 40), min(xs))

    boxes = sorted(result or [], key=reading_order)
    return [(str(item[1]), float(item[2])) for item in boxes]


def _ink_share(crop: np.ndarray) -> float:
    """The share of dark pixels in a crop, 0.0 for an empty one."""
    if crop.size == 0:
        return 0.0
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(np.mean(grey < 128))
