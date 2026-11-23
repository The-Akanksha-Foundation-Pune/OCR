"""The Hindi form's back page ("सहमति की पुष्टि"), as BOPEMS prints it.

It is a Declaration page and stays with its form, even when it carries so
little ink - a signature and a date - that it would otherwise count as blank.
"""

from __future__ import annotations

import cv2
import numpy as np

from server.config import DECLARATION_EXTRA_HEADINGS
from server.services import extract_student_id as extract

# Where the heading sits on a real BOPEMS scan 1662 x 2350 pixels.
_PAGE = (2350, 1662)
_HEADING_AT = (224, 328)


def _back_page(extra_ink: float = 0.0) -> np.ndarray:
    """A white page with the printed heading at the top left, and optional scribble below."""
    page = np.full((*_PAGE, 3), 255, np.uint8)
    path, _, _ = DECLARATION_EXTRA_HEADINGS[0]
    heading = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert heading is not None
    x, y = _HEADING_AT
    page[y : y + heading.shape[0], x : x + heading.shape[1]] = cv2.cvtColor(heading, cv2.COLOR_GRAY2BGR)
    if extra_ink:
        rows = int(_PAGE[0] * 0.5 * extra_ink)
        page[_PAGE[0] // 2 : _PAGE[0] // 2 + rows, 100:-100] = 0
    return page


def test_a_sparse_hindi_back_page_is_a_declaration_not_blank() -> None:
    result = extract._extract(_back_page(), "back.jpg")

    assert result.skipped and result.skip_reason == "declaration"


def test_a_well_inked_hindi_back_page_is_a_declaration() -> None:
    result = extract._extract(_back_page(extra_ink=0.08), "back.jpg")

    assert result.skipped and result.skip_reason == "declaration"


def test_a_truly_blank_page_is_still_blank() -> None:
    page = np.full((*_PAGE, 3), 255, np.uint8)

    result = extract._extract(page, "blank.jpg")

    assert result.skipped and result.skip_reason == "blank"


def test_the_heading_is_looked_for_at_the_top_left_only() -> None:
    page = np.full((*_PAGE, 3), 255, np.uint8)
    path, _, _ = DECLARATION_EXTRA_HEADINGS[0]
    heading = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    # The same words half way down the page are body text, not the heading.
    y = _PAGE[0] // 2
    page[y : y + heading.shape[0], 224 : 224 + heading.shape[1]] = heading

    assert extract.looks_like_declaration(page) is False
