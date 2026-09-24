"""Straighten a scanned page by its horizontal ink.

A page fed through the ADF often lands a degree or two off square, even when
the paper looks straight to the eye. The rule-ladder detector only counts
lines that are actually horizontal - a 1.5-degree tilt hides every one of
them and the reader falls through to a template guess that costs 20+ seconds.

This step measures the angle of the dominant long horizontal ink runs (the
form's printed rules, and the ~60 characters of body text underlines that go
with them) and rotates the page back. About 200 ms per page. Called only
when the first layout attempt failed, so straight pages pay nothing.
"""

from __future__ import annotations

import cv2
import numpy as np

# Angles below this are indistinguishable from measurement noise; ignore them.
_NEGLIGIBLE_DEG = 0.15

# Anything past this and the page probably was not just tilted - refuse to
# spin a face-down or 90-degree-turned sheet as if it were a small skew.
_MAX_CORRECTION_DEG = 8.0


def estimate_skew_deg(bgr: np.ndarray) -> float:
    """Median tilt angle of long horizontal ink runs, in degrees. 0 = straight."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    # Keep only ink that continues horizontally for a few dozen pixels: the
    # form's ruled lines and long body underlines survive, letter stems do not.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(horizontal, 40, 120)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=200,
        minLineLength=int(0.15 * bgr.shape[1]), maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))
        # Only near-horizontal runs count; anything past a modest tilt is a
        # diagonal stroke we don't want to rotate the whole page for.
        if abs(angle) < _MAX_CORRECTION_DEG:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def deskew(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (straightened_image, angle_applied_deg). No-op for tiny angles."""
    angle = estimate_skew_deg(bgr)
    if abs(angle) < _NEGLIGIBLE_DEG:
        return bgr, 0.0

    height, width = bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    straightened = cv2.warpAffine(
        bgr, matrix, (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,   # keep the page's white margin
    )
    return straightened, angle
