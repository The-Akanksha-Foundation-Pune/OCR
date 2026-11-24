"""The two-page Hindi form MEMS (Matoshri English Medium School) prints.

Its front has its own rule ladder, with the Student ID ("विद्यार्थी आईडी") on
the fourth rule and the handwriting running on past the rule's end, and the
printed stamp "School Index No. 11.15.263" beside the school name. Its back
opens with body text and has its "घोषणा" heading part way down, in a face
of its own.
"""

from __future__ import annotations

import cv2
import numpy as np

from server.config import DECLARATION_EXTRA_HEADINGS, FORM_RULE_FINGERPRINTS
from server.services import extract_student_id as extract
from server.services.form_layout import identify_form
from server.services.paper_forms import ONE_SHEET, TWO_SHEET

# A real MEMS scan at 200 dpi.
_PAGE = (2335, 1664)
# Each rule's left and right ends on that scan, as fractions of the width.
_RULE_X = [
    (0.243, 0.522), (0.239, 0.516), (0.162, 0.442), (0.227, 0.504),
    (0.248, 0.537), (0.198, 0.477), (0.198, 0.476), (0.129, 0.358),
]


def _front() -> np.ndarray:
    """A white page with the MEMS front's fill-in rules drawn where the scan has them."""
    page = np.full((*_PAGE, 3), 255, np.uint8)
    height, width = _PAGE
    for y, (x0, x1) in zip(FORM_RULE_FINGERPRINTS["hindi_mems"]["rows"], _RULE_X):
        row = int(y * height)
        page[row : row + 3, int(x0 * width) : int(x1 * width)] = 0
    return page


def _mems_heading() -> np.ndarray:
    path, _, _ = DECLARATION_EXTRA_HEADINGS[1]
    assert path.name == "ghoshana_mems.png"
    heading = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert heading is not None
    return cv2.cvtColor(heading, cv2.COLOR_GRAY2BGR)


def _back(heading_top: float = 0.367) -> np.ndarray:
    """A white page with the MEMS "घोषणा" heading at the left margin, part way down."""
    page = np.full((2339, 1656, 3), 255, np.uint8)
    heading = _mems_heading()
    y = int(page.shape[0] * heading_top)
    page[y : y + heading.shape[0], 164 : 164 + heading.shape[1]] = heading
    # Body text above it, so the page is not blank.
    page[200:1200:40, 180:1500] = 0
    return page


def test_the_mems_front_is_recognised_by_its_own_ladder() -> None:
    layout = identify_form(_front())

    assert layout is not None
    assert layout.language == "hindi_mems"
    assert layout.error <= 0.0025


def test_the_mems_student_id_crop_reaches_past_the_rule() -> None:
    layout = identify_form(_front())

    assert layout is not None
    left, top, right, bottom = layout.student_id_roi
    # On the Student ID rule (0.368), not the class rule above it (0.335).
    assert 0.34 < top < 0.368 < bottom < 0.38
    # The rule ends at 0.504; IDs are written on to 0.62 and beyond.
    assert left < 0.227 and right >= 0.66


def test_the_other_forms_keep_their_short_reach() -> None:
    for language, spec in FORM_RULE_FINGERPRINTS.items():
        if language != "hindi_mems":
            assert "value_right_reach" not in spec


def test_the_mems_back_is_a_declaration_page_sent_as_two_sheets() -> None:
    result = extract._extract(_back(), "back.jpg", TWO_SHEET)

    assert result.skipped and result.skip_reason == "declaration"
    assert result.declaration_side is False


def test_the_mems_back_is_a_declaration_side_sent_as_one_sheet() -> None:
    result = extract._extract(_back(), "back.jpg", ONE_SHEET)

    assert result.skipped and result.skip_reason == "declaration"
    assert result.declaration_side is True


def test_the_mems_heading_is_not_looked_for_below_half_way() -> None:
    page = _back(heading_top=0.70)

    assert extract.looks_like_declaration(page) is False


def test_the_printed_school_index_is_never_a_student_id() -> None:
    assert extract._is_printed_number("School Index No. 11.15.263")
    assert extract._is_printed_number("ex No. 11.15.263")
    assert extract._is_printed_number("11.15.263")
    assert not extract._is_printed_number("ABCDEF010101")
    assert not extract._is_printed_number("aadmis.310714")
    assert not extract._is_printed_number("SIMNO.310714")


def test_the_worker_never_turns_round_a_page_that_shows_its_heading_upright(monkeypatch) -> None:
    from server.services import ocr_worker
    from server.services.form_layout import FormMatch

    back = cv2.cvtColor(_back(), cv2.COLOR_BGR2GRAY)
    plain = np.full_like(back, 255)
    plain[100:200, 100:300] = 0  # some ink, not the heading, not symmetric
    loose = FormMatch("english", 0.5, (0.5, 0.2, 0.75, 0.23), 4, 0.0155, "rules")
    as_came: list[np.ndarray] = []

    def identify(image: np.ndarray) -> FormMatch | None:
        # Some ladder fits the page turned round, none as it came - as the
        # real MEMS back page does.
        turned = not np.array_equal(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), as_came[-1])
        return loose if turned else None

    monkeypatch.setattr(ocr_worker, "identify", identify)

    as_came.append(back)
    assert ocr_worker._layout_orientation(back, TWO_SHEET) is None
    # A page without the heading still turns round as before.
    as_came.append(plain)
    assert ocr_worker._layout_orientation(plain, TWO_SHEET) == 180


# ------------------------------------------------- the MEMS Marathi form
def _marathi_back(label_top: float = 0.21, print_run: int = 2) -> np.ndarray:
    """A white page with the "पालकांची स्वाक्षरी:" label at the left margin and almost no other ink.

    print_run 3 is the label from a second MEMS print run, printed larger.
    """
    path, _, _ = DECLARATION_EXTRA_HEADINGS[print_run]
    assert path.name.startswith("palakanchi_swakshari")
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert label is not None
    page = np.full((2339, 1666, 3), 255, np.uint8)
    y = int(page.shape[0] * label_top)
    page[y : y + label.shape[0], 186 : 186 + label.shape[1]] = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    return page


def test_the_marathi_back_page_is_a_declaration_not_blank() -> None:
    result = extract._extract(_marathi_back(), "back.jpg", TWO_SHEET)

    assert result.skipped and result.skip_reason == "declaration"


def test_the_marathi_signature_label_far_down_a_page_is_not_a_back() -> None:
    assert extract.looks_like_declaration(_marathi_back(label_top=0.60)) is False


def test_an_id_written_past_the_rule_is_followed_to_its_end() -> None:
    from server.services.form_layout import _follow_ink_right

    page = np.full((2339, 1656, 3), 255, np.uint8)
    # Handwriting from 0.50 to 0.70 of the width, on a box that stops at 0.60.
    page[1050:1090, 828:1160:20] = 0
    page[1050:1090, 829:1160:20] = 0

    left, top, right, bottom = _follow_ink_right(page, (0.24, 0.445, 0.60, 0.472))

    assert (left, top, bottom) == (0.24, 0.445, 0.472)
    assert 0.70 <= right <= 0.72


def test_a_box_with_nothing_after_the_rule_is_left_as_it_is() -> None:
    from server.services.form_layout import _follow_ink_right

    page = np.full((2339, 1656, 3), 255, np.uint8)
    page[1050:1090, 500:900] = 0  # inside the box only

    assert _follow_ink_right(page, (0.24, 0.445, 0.60, 0.472)) == (0.24, 0.445, 0.60, 0.472)


def test_ink_after_a_wide_gap_is_not_taken_in() -> None:
    from server.services.form_layout import _follow_ink_right

    page = np.full((2339, 1656, 3), 255, np.uint8)
    # Something well clear of the end of the box, past a gap of 0.06.
    page[1050:1090, 1094:1200] = 0

    assert _follow_ink_right(page, (0.24, 0.445, 0.60, 0.472))[2] == 0.60


def test_the_marathi_back_page_from_the_other_print_run_is_a_declaration() -> None:
    result = extract._extract(_marathi_back(label_top=0.19, print_run=3), "back.jpg", TWO_SHEET)

    assert result.skipped and result.skip_reason == "declaration"
