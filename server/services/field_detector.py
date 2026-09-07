"""Detect printed Student ID labels and extract ID text from RapidOCR."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from server.config import (
    ROI_HEIGHT_MULTIPLIER,
    ROI_LEFT_PADDING_RATIO,
    ROI_RIGHT_MULTIPLIER,
    TEMPLATE_ROI_PAD_RATIO,
    TEMPLATE_VALUE_ROIS,
)
from server.services.debug_log import debug_log

_rapid_ocr = None

_ID_AFTER_LABEL = re.compile(
    r"student\s*id\s*:?\s*([A-Za-z0-9][A-Za-z0-9\-_/\s]{3,20})",
    re.IGNORECASE,
)


@dataclass
class LabelHit:
    text: str
    language: str
    box: list[list[float]]
    confidence: float
    source: str = "ocr"
    parsed_student_id: str = ""


def get_rapid_ocr():
    """Create or reuse RapidOCR for printed label detection."""
    global _rapid_ocr
    if _rapid_ocr is not None:
        return _rapid_ocr

    from rapidocr_onnxruntime import RapidOCR

    _rapid_ocr = RapidOCR()
    return _rapid_ocr


def run_page_ocr(image_bgr: np.ndarray) -> list:
    """Run RapidOCR on a page and return raw line list."""
    ocr = get_rapid_ocr()
    result, _ = ocr(image_bgr)
    return result or []


def find_student_id_label(image_bgr: np.ndarray) -> LabelHit | None:
    """Find the Student ID printed label on EN/HI/MR consent forms."""
    result = run_page_ocr(image_bgr)

    if result:
        best: LabelHit | None = None
        best_score = -1.0

        for item in result:
            if not item or len(item) < 3:
                continue
            box, text, confidence = item[0], item[1], item[2]
            language, score = _score_label(str(text))
            parsed = parse_student_id_from_text(str(text))
            if language is None and not parsed:
                continue
            if language is None:
                language = "english"
                score = 1.5
            ranked = score + float(confidence) + (1.5 if parsed else 0.0)
            if ranked > best_score:
                best_score = ranked
                best = LabelHit(
                    text=str(text),
                    language=language,
                    box=box,
                    confidence=float(confidence),
                    source="ocr",
                    parsed_student_id=parsed,
                )

        if best is not None:
            if not best.parsed_student_id:
                best.parsed_student_id = _id_from_nearby_boxes(result, best)
            debug_log(
                f"[DETECT] best_label={best.text!r} parsed={best.parsed_student_id!r}"
            )
            return best

        fallback = _fallback_after_grade(result)
        if fallback is not None:
            fallback.parsed_student_id = parse_student_id_from_text(fallback.text)
            if not fallback.parsed_student_id:
                fallback.parsed_student_id = _id_from_nearby_boxes(result, fallback)
            return fallback

    return _template_label_hit(image_bgr, preferred="english")


def parse_student_id_from_text(text: str) -> str:
    """Pull Student ID characters from OCR line text like 'Student ID:AADMES3Ozy'."""
    if not text:
        return ""

    match = _ID_AFTER_LABEL.search(text)
    if match:
        return _normalize_id_token(match.group(1))

    # Label (any language) and value glued in one OCR line, e.g.
    # "faaef3Us:TAROUR29081S" (Marathi "विद्यार्थी आयडी:" garbled + value).
    # Only the text AFTER the last colon is the handwritten value — never
    # glue the garbled label prefix onto it.
    if ":" in text:
        value_part = text.rpartition(":")[2]
        compact = re.sub(r"[^A-Za-z0-9]", "", value_part).upper()
        if 6 <= len(compact) <= 20 and any(c.isalpha() for c in compact) and any(
            c.isdigit() for c in compact
        ):
            return compact
        return ""

    # Standalone token that already looks ID-like
    compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if 8 <= len(compact) <= 20 and any(c.isalpha() for c in compact) and any(
        c.isdigit() for c in compact
    ):
        # Avoid returning pure label words
        if "STUDENT" in compact or "INFORMATION" in compact:
            return ""
        return compact
    return ""


def crop_student_id_value(image_bgr: np.ndarray, label: LabelHit) -> np.ndarray:
    """Crop the handwritten value region to the right of the label (or template ROI)."""
    if label.source == "template":
        return crop_template_value(image_bgr, label.language)

    xs = [point[0] for point in label.box]
    ys = [point[1] for point in label.box]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    label_width = max(x_max - x_min, 1.0)
    label_height = max(y_max - y_min, 1.0)

    height, width = image_bgr.shape[:2]
    # Glued "Student ID:AADMES3Ozy" boxes: take the right side where the value is.
    # Use the colon's character position within the OCR'd text as a proxy for
    # where the value starts across the box width, instead of a fixed ratio —
    # label lengths vary a lot across English/Hindi/Marathi.
    if label.parsed_student_id or ":" in label.text.lower():
        text = label.text
        if ":" in text:
            split_ratio = (text.rindex(":") + 1) / max(len(text), 1)
            split_ratio = min(max(split_ratio, 0.25), 0.75)
        else:
            split_ratio = 0.42
        left = int(x_min + label_width * split_ratio)
        right = int(min(width, x_max + label_width * 0.15))
    else:
        left = int(max(0, x_max + label_width * ROI_LEFT_PADDING_RATIO))
        right = int(min(width, x_max + label_width * ROI_RIGHT_MULTIPLIER))

    top = int(max(0, y_min - label_height * 0.8))
    bottom = int(min(height, y_max + label_height * 1.4))

    if right <= left or bottom <= top:
        raise ValueError("Invalid Student ID crop region")

    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Empty Student ID crop region")
    return crop


def crop_template_value(image_bgr: np.ndarray, language: str) -> np.ndarray:
    """Crop Student ID value using relative coordinates from blank form templates."""
    if language not in TEMPLATE_VALUE_ROIS:
        raise ValueError(f"No template ROI for language: {language}")
    return crop_value_roi(image_bgr, TEMPLATE_VALUE_ROIS[language])


def crop_value_roi(
    image_bgr: np.ndarray, roi: tuple[float, float, float, float]
) -> np.ndarray:
    """Crop an explicit (x0, y0, x1, y1) ROI given as page fractions."""
    x0, y0, x1, y1 = roi
    height, width = image_bgr.shape[:2]
    left = int(width * x0)
    right = int(width * x1)
    top = int(height * y0)
    bottom = int(height * y1)

    pad = max(4, int((bottom - top) * TEMPLATE_ROI_PAD_RATIO))
    top = max(0, top - pad)
    bottom = min(height, bottom + pad)

    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Empty template Student ID crop")
    return crop


def iter_template_languages() -> list[str]:
    return list(TEMPLATE_VALUE_ROIS.keys())


def _normalize_id_token(token: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", token).upper()
    return text


def _id_from_nearby_boxes(ocr_lines: list, label: LabelHit) -> str:
    """Find an alphanumeric token to the right of / on the Student ID label line."""
    xs = [point[0] for point in label.box]
    ys = [point[1] for point in label.box]
    label_x_max = max(xs)
    label_y = sum(ys) / 4.0
    label_h = max(ys) - min(ys)

    best = ""
    for item in ocr_lines:
        box, text, _ = item[0], str(item[1]), item[2]
        y_center = sum(point[1] for point in box) / 4.0
        x_min = min(point[0] for point in box)
        if abs(y_center - label_y) > max(28.0, label_h * 1.8):
            continue
        if x_min < label_x_max - 10:
            # Same box / overlapping — try parse from full text
            parsed = parse_student_id_from_text(text)
            if parsed and len(parsed) > len(best):
                best = parsed
            continue
        compact = _normalize_id_token(text)
        if 6 <= len(compact) <= 20 and any(c.isalpha() for c in compact) and any(
            c.isdigit() for c in compact
        ):
            if len(compact) > len(best):
                best = compact
    return best


def _template_label_hit(image_bgr: np.ndarray, preferred: str = "english") -> LabelHit:
    """Synthetic label hit so callers can crop via template ROI."""
    language = preferred if preferred in TEMPLATE_VALUE_ROIS else "english"
    height, width = image_bgr.shape[:2]
    x0, y0, x1, y1 = TEMPLATE_VALUE_ROIS[language]
    label_right = width * x0
    label_left = max(0.0, label_right - width * 0.18)
    top = height * y0
    bottom = height * y1
    box = [
        [label_left, top],
        [label_right, top],
        [label_right, bottom],
        [label_left, bottom],
    ]
    return LabelHit(
        text=f"template:{language}",
        language=language,
        box=box,
        confidence=0.4,
        source="template",
    )


def _score_label(text: str) -> tuple[str | None, float]:
    """Return (language, score) if text looks like a Student ID label."""
    normalized = text.strip().lower()
    compact = re.sub(r"\s+", "", normalized)

    if "information" in normalized or "जानका" in text or "माहित" in text or "माचित" in text:
        return None, 0.0

    if "studentid" in compact or re.search(r"student\s*id\s*:?", normalized):
        return "english", 3.0
    if re.search(r"\bstudent\b", normalized) and re.search(r"\bid\b", normalized):
        return "english", 2.5

    if "आयडी" in text or "आईडी" in text or "आईडि" in text or "आईिी" in text:
        language = "marathi" if "आयडी" in text or "आयिी" in text else "hindi"
        score = 3.0 if "विद्यार्थ" in text or "भवद्यार्थ" in text or "चवद्यार्थ" in text else 2.0
        return language, score

    if "vidyarthi" in compact and re.search(r"\bid\b", normalized):
        return "hindi", 2.0

    return None, 0.0


def _is_grade_label(text: str) -> bool:
    normalized = text.strip().lower()
    if "grade" in normalized or "class" in normalized:
        return True
    if "कक्षा" in text or "इयत्ता" in text:
        return True
    return False


def _fallback_after_grade(ocr_lines: list) -> LabelHit | None:
    """If Student ID label text is missed, use the line under Grade."""
    grade_hits: list[tuple[float, list, str, float]] = []
    for item in ocr_lines:
        box, text, confidence = item[0], str(item[1]), float(item[2])
        if not _is_grade_label(text):
            continue
        y_center = sum(point[1] for point in box) / 4.0
        grade_hits.append((y_center, box, text, confidence))

    if not grade_hits:
        return None

    grade_hits.sort(key=lambda row: row[0])
    grade_y, _, _, _ = grade_hits[0]

    candidates: list[tuple[float, list, str, float]] = []
    for item in ocr_lines:
        box, text, confidence = item[0], str(item[1]), float(item[2])
        y_center = sum(point[1] for point in box) / 4.0
        if y_center <= grade_y + 5:
            continue
        if y_center - grade_y > 140:
            continue
        candidates.append((y_center, box, text, confidence))

    if not candidates:
        return None

    candidates.sort(key=lambda row: row[0])
    _, box, text, confidence = candidates[0]
    language = "hindi" if any(ch in text for ch in "आईआयडीकक्षा") else "english"
    if "आयडी" in text or "इयत्ता" in text or "आयिी" in text:
        language = "marathi"
    return LabelHit(
        text=text,
        language=language,
        box=box,
        confidence=confidence,
        source="ocr",
    )
