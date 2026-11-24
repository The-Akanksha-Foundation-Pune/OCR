"""
Identify the consent-form language and locate the Student ID field by the
page's ruled fill-in lines, not by reading its text.

RapidOCR here runs the English model, so the Hindi/Marathi labels are
unreadable on a scan — there is no text signal to key off. What the three
templates *do* share is a distinctive ladder of underscore rules, and in
every language the Student ID is the 4th rule from the top. Matching that
ladder identifies the form and pins the value box in one step, and because
the match solves for scale and offset it tolerates the framing drift of a
real ADF scan far better than fixed page fractions do.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from server.config import FORM_RULE_FINGERPRINTS, TEMPLATE_VALUE_ROIS
from server.services.debug_log import debug_log

# A rule must span at least this fraction of the page to count as a
# fill-in line rather than underlined body text or a table edge.
MIN_RULE_WIDTH_FRAC = 0.10
# Rules closer together than this (as a fraction of page height) are the
# same line broken by scan noise.
RULE_MERGE_TOL = 0.008
# Beyond this mean row error the match is not trustworthy.
MAX_MATCH_ERROR = 0.020


@dataclass
class FormMatch:
    language: str
    confidence: float
    student_id_roi: tuple[float, float, float, float]
    rule_count: int
    error: float
    source: str  # "rules" | "fallback"
    # The language that came second. When confidence is low this is the
    # other plausible reading of the page, and the caller may want to keep
    # its recognisers (Devanagari) switched on too.
    runner_up_language: str | None = None


def find_rule_rows(image_bgr: np.ndarray) -> list[tuple[float, float, float]]:
    """Return the page's horizontal fill-in rules as (y, x0, x1) fractions."""
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )

    # Keep only long horizontal ink: an underscore run survives, glyphs do not.
    kernel_width = max(12, int(width * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(rules, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[tuple[float, float, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * MIN_RULE_WIDTH_FRAC:
            continue
        if h > max(6, int(height * 0.012)):  # a box edge or shaded band, not a rule
            continue
        found.append(((y + h / 2.0) / height, x / width, (x + w) / width))

    found.sort()
    return _merge_close_rows(found)


def _merge_close_rows(
    rows: list[tuple[float, float, float]]
) -> list[tuple[float, float, float]]:
    """Fuse rules split into fragments by a speck or a fold."""
    merged: list[tuple[float, float, float]] = []
    for y, x0, x1 in rows:
        if merged and abs(y - merged[-1][0]) <= RULE_MERGE_TOL:
            prev_y, prev_x0, prev_x1 = merged[-1]
            merged[-1] = ((prev_y + y) / 2.0, min(prev_x0, x0), max(prev_x1, x1))
        else:
            merged.append((y, x0, x1))
    return merged


# Same grid the loop used to walk. Built once: identify_form runs on every
# page, and the nested Python loops were a third of a second each.
_SCALES = np.arange(0.85, 1.151, 0.01)
_OFFSETS = np.arange(-0.12, 0.1201, 0.005)


def _match_language(
    observed: list[float], expected: list[float]
) -> tuple[float, float, float]:
    """
    Best (error, scale, offset) fitting expected rows onto the observed ones.

    A scan shifts and stretches the page slightly, so compare shapes rather
    than absolute positions: search a small scale/offset grid and score each
    expected row by its nearest observed row.
    """
    if not observed or not expected:
        return float("inf"), 1.0, 0.0

    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)

    # Real scans drift further than a tight grid used to allow: a Fujitsu ADF
    # crops differently from the template PDFs, and runs were settling on
    # scale=1.08 / offset=+0.06 - both pinned at the old limits, which means
    # the true fit was outside the range and the match landed on the wrong
    # rules. Widen the search rather than mis-fit.
    # projected shape: (scale, offset, expected row)
    projected = exp[None, None, :] * _SCALES[:, None, None] + _OFFSETS[None, :, None]
    errors = np.abs(projected[..., None] - obs).min(axis=-1).mean(axis=-1)
    scale_i, offset_i = np.unravel_index(int(errors.argmin()), errors.shape)
    return float(errors[scale_i, offset_i]), float(_SCALES[scale_i]), float(_OFFSETS[offset_i])


def identify_form(image_bgr: np.ndarray) -> FormMatch | None:
    """
    Work out which language template this page is and where its Student ID
    value box sits. Returns None when the rules are too unclear to trust,
    so the caller can fall back to trying every template.
    """
    rows = find_rule_rows(image_bgr)
    if len(rows) < 4:
        debug_log(f"[LAYOUT] only {len(rows)} rules found — no layout match")
        return None

    observed = [row[0] for row in rows]
    results: list[tuple[float, str, float, float]] = []
    for language, spec in FORM_RULE_FINGERPRINTS.items():
        error, scale, offset = _match_language(observed, spec["rows"])
        results.append((error, language, scale, offset))

    results.sort()
    error, language, scale, offset = results[0]
    runner_up = results[1][0] if len(results) > 1 else float("inf")
    runner_up_language = results[1][1] if len(results) > 1 else None

    if error > MAX_MATCH_ERROR:
        debug_log(
            f"[LAYOUT] best={language} error={error:.4f} exceeds "
            f"{MAX_MATCH_ERROR} — rejecting layout match"
        )
        return None

    spec = FORM_RULE_FINGERPRINTS[language]
    pinned = _roi_from_matched_rule(rows, spec, scale, offset)
    # Only a box on a rule the scan shows is followed past its end: a fixed
    # template box may sit on a line of printed text.
    roi = _follow_ink_right(image_bgr, pinned) if pinned else TEMPLATE_VALUE_ROIS[language]

    # Clear separation from the runner-up means a confident language call.
    margin = (runner_up - error) / max(runner_up, 1e-6)
    confidence = float(min(0.99, max(0.3, margin)))

    debug_log(
        f"[LAYOUT] language={language} error={error:.4f} runner_up={runner_up:.4f} "
        f"scale={scale:.3f} offset={offset:+.3f} rules={len(rows)} roi={roi}"
    )
    return FormMatch(
        language=language,
        confidence=confidence,
        student_id_roi=roi,
        rule_count=len(rows),
        error=error,
        source="rules",
        runner_up_language=runner_up_language,
    )


# Handwriting that runs on past the printed rule is followed until a gap at
# least this wide (a fraction of the page width) - wider than the space
# between two written characters - or the page's right margin.
INK_FOLLOW_GAP = 0.03
INK_FOLLOW_LIMIT = 0.97
# A column counts as ink when this share of the band's height is dark.
INK_FOLLOW_COLUMN = 0.06


def _follow_ink_right(
    image_bgr: np.ndarray, roi: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Widen the ROI to take in a Student ID written on past the rule's end.

    Parents often write the ID starting near the end of the printed rule, so
    its last characters fall outside a box that stops there: a Marathi MEMS
    form (Nov 2026) had the last digit of its ID cut in half, and Vision
    misread it. From the box's right edge the band is followed rightwards
    while ink continues with no gap wider than INK_FOLLOW_GAP. The box is
    never narrowed, and a line with nothing after the rule is left as it is.
    """
    height, width = image_bgr.shape[:2]
    left, top, right, bottom = roi
    y0, y1 = int(top * height), int(bottom * height)
    x_start, x_limit = int(right * width), int(INK_FOLLOW_LIMIT * width)
    if y1 - y0 < 4 or x_start >= x_limit:
        return roi
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    band = gray[y0:y1, x_start:x_limit] < 160
    inked = band.sum(axis=0) >= max(2, int((y1 - y0) * INK_FOLLOW_COLUMN))
    gap_limit = max(4, int(width * INK_FOLLOW_GAP))
    last, gap = -1, 0
    for index, has_ink in enumerate(inked):
        if has_ink:
            last, gap = index, 0
        else:
            gap += 1
            if gap > gap_limit:
                break
    if last < 0:
        return roi
    new_right = min(INK_FOLLOW_LIMIT, (x_start + last) / width + 0.010)
    if new_right <= right:
        return roi
    debug_log(f"[LAYOUT] Student ID runs on past the rule - box widened {right:.3f} -> {new_right:.3f}")
    return (left, top, new_right, bottom)


def _roi_from_matched_rule(
    rows: list[tuple[float, float, float]],
    spec: dict,
    scale: float,
    offset: float,
) -> tuple[float, float, float, float] | None:
    """
    Pin the ROI to the rule the scan actually shows rather than to the
    template's nominal position, so a shifted or stretched page still
    crops the right box.
    """
    expected_y = spec["rows"][spec["student_id_index"]] * scale + offset
    best_row = min(rows, key=lambda row: abs(row[0] - expected_y))
    if abs(best_row[0] - expected_y) > RULE_MERGE_TOL * 3:
        return None

    y, x0, x1 = best_row
    nominal = spec["value_height"] * scale
    # The value sits on the rule, so reach up from it rather than centring.
    top = max(0.0, y - nominal * 1.10)
    bottom = min(1.0, y + nominal * 0.35)
    left = max(0.0, x0 - 0.010)
    right = min(1.0, x1 + spec.get("value_right_reach", 0.010))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)
