"""
Tell a form's later pages from its first page by the shape of the whole page.

Only page 1 carries a Student ID. The pages behind it - the consent text, the
"सहमति की पुष्टि" / "घोषणा" confirmation - have no fill-in ladder, so the
layout match gives up on them and they used to fall through to full-page OCR
and every template crop: about 11 seconds a page, ending either in an
"unreadable" error or, worse, a made-up ID read off the body text
(IFUNDRAISING3Q3 off the Hindi page 2) waiting to be confirmed.

Each printed page has its own pattern of text lines down the page. The
row-by-row ink profile of a scan is compared with that of every page of the
blank templates in data/forms, solving for the scale and offset a scan adds
(a Letter template printed on A4 measures about 0.87). A page is called a
continuation only when it matches a later template page closely and no
first page comes near, so a first page the layout missed still gets read.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from server.config import FORM_TEMPLATE_DIR
from server.services.debug_log import debug_log

# Rows the page is resampled to. Fine enough to separate one printed line
# from the next, coarse enough that the comparison takes ~40 ms.
PROFILE_BINS = 300
# Width the page is shrunk to before measuring ink. Matches nothing exact;
# it only keeps the row sums cheap on a 600 DPI scan.
PROFILE_WORK_WIDTH = 400
INK_THRESHOLD = 160
# Below this much overlap between the shifted template and the page, a fit
# is ignored rather than scored on a sliver.
MIN_OVERLAP = 0.70

_SCALES = np.arange(0.80, 1.151, 0.01)
_OFFSETS = np.arange(-0.12, 0.1201, 0.005)
_POSITIONS = (np.arange(PROFILE_BINS) + 0.5) / PROFILE_BINS


@dataclass(frozen=True)
class PageMatch:
    # "hindi:3" style key of the closest template page.
    best_page: str
    best_score: float
    # Closest first page of any language, and closest later page.
    first_page_score: float
    continuation_score: float

    @property
    def best_is_continuation(self) -> bool:
        return not self.best_page.endswith(":1")


_lock = threading.Lock()
_templates: dict[str, np.ndarray] | None = None


def row_profile(image_bgr: np.ndarray) -> np.ndarray:
    """Fraction of dark pixels in each row, resampled to PROFILE_BINS."""
    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    )
    height, width = gray.shape[:2]
    if width > PROFILE_WORK_WIDTH:
        new_height = max(1, int(height * PROFILE_WORK_WIDTH / width))
        gray = cv2.resize(gray, (PROFILE_WORK_WIDTH, new_height), interpolation=cv2.INTER_AREA)
    ink = (gray < INK_THRESHOLD).mean(axis=1)
    source = np.linspace(0.0, 1.0, len(ink))
    return np.interp(_POSITIONS, source, ink)


def _language_of(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("eng"):
        return "english"
    if stem.startswith("hin"):
        return "hindi"
    if stem.startswith(("mar", "mra")):
        return "marathi"
    return stem.split()[0]


def _load_templates() -> dict[str, np.ndarray]:
    """Profiles of every page of every blank template, built once."""
    global _templates
    if _templates is not None:
        return _templates
    with _lock:
        if _templates is not None:
            return _templates
        from server.services.preprocess import load_pages_from_bytes, preprocess_for_ocr

        built: dict[str, np.ndarray] = {}
        for path in sorted(Path(FORM_TEMPLATE_DIR).glob("*.pdf")):
            try:
                pages = load_pages_from_bytes(path.read_bytes(), path.name)
            except Exception as exc:  # noqa: BLE001 - one bad template must not stop the rest
                debug_log(f"[PROFILE] could not read template {path.name}: {exc}")
                continue
            language = _language_of(path)
            for number, page in enumerate(pages, start=1):
                built[f"{language}:{number}"] = row_profile(preprocess_for_ocr(page))
        debug_log(f"[PROFILE] {len(built)} template pages loaded from {FORM_TEMPLATE_DIR}")
        _templates = built
        return _templates


def _best_fit(observed: np.ndarray, template: np.ndarray) -> float:
    """
    Highest correlation between the page and the template under any scale
    and offset in the grid. The template value at page position y is
    template((y - offset) / scale).
    """
    where = (_POSITIONS[None, None, :] - _OFFSETS[None, :, None]) / _SCALES[:, None, None]
    inside = ((where >= 0.0) & (where <= 1.0)).astype(float)
    projected = np.interp(where, _POSITIONS, template)
    count = inside.sum(axis=-1)
    safe = np.maximum(count, 1.0)

    mean_t = (projected * inside).sum(axis=-1) / safe
    mean_o = (observed[None, None, :] * inside).sum(axis=-1) / safe
    dev_t = (projected - mean_t[..., None]) * inside
    dev_o = (observed[None, None, :] - mean_o[..., None]) * inside
    corr = (dev_t * dev_o).sum(axis=-1) / np.sqrt(
        (dev_t**2).sum(axis=-1) * (dev_o**2).sum(axis=-1) + 1e-12
    )
    corr = np.where(count >= PROFILE_BINS * MIN_OVERLAP, corr, -1.0)
    return float(corr.max())


def match_page(image_bgr: np.ndarray) -> PageMatch | None:
    """Closest template page for this scan, or None without templates."""
    templates = _load_templates()
    if not templates:
        return None
    observed = row_profile(image_bgr)
    scores = {key: _best_fit(observed, profile) for key, profile in templates.items()}
    best_page = max(scores, key=scores.get)
    first = [score for key, score in scores.items() if key.endswith(":1")]
    later = [score for key, score in scores.items() if not key.endswith(":1")]
    return PageMatch(
        best_page=best_page,
        best_score=scores[best_page],
        first_page_score=max(first) if first else -1.0,
        continuation_score=max(later) if later else -1.0,
    )


def warm() -> None:
    """Build the template profiles ahead of the first page."""
    _load_templates()
