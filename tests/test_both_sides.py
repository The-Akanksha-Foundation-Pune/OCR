"""The form printed on both sides of one sheet: its back known by the heading part way down, each sheet one form.

That form is one sheet: its Student ID side, and on the back the rest of the
consent text with the "घोषणा" heading part way down - its Declaration side.
Scanned both sides, every sheet gives two images. With batch codes off, as
the app ships:

  - The reader knows a Declaration side by that heading anywhere down the
    left margin, as the page came or turned round - the worker then stores
    the page turned upright - and leaves it there, reading nothing on it,
    however the scanner made the page black and white. The heading at the
    top left of its own sheet (the two-sheet form) is a Declaration page as
    before, that check made first. A page whose own ruled layout pins its
    Student ID box, with handwriting in it, is never taken for a
    Declaration side, whatever heading the search down the margin finds.
  - A file of several pages scanned both sides pairs sheet by sheet, by
    place - pages 1 and 2, 3 and 4 - the Student ID side first in the
    document and the PDF, whichever came first; its sheets wait until every
    page of the file is read, and one whose Declaration side came first
    needs a person's look. A file that does not pair into sheets pairs by
    scan order like any other, and so do files of one image each that are
    not the numbered images of one scanner session (those pair sheet by
    sheet too: tests/test_scan_sessions.py) and the two-sheet form in one
    file.
  - Automatic upload: a stack of such files in plain order is clean; one
    with a sheet scanned Declaration side first, a file that does not pair
    into sheets, or such a file in among other scans waits for a person, as
    everything does with automatic upload off.
  - The page says to scan the one-sheet form both sides with skip blank
    pages off and the two-sheet form one side, and calls a sheet's two parts
    its Student ID side and its Declaration side.

Every page here is drawn: lines of printed words, ruled lines and the
heading picture in data/labels. The database is the FakeDB, Drive the
FakeDrive; nothing leaves the test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from flask.testing import FlaskClient
from PIL import Image

from server import config
from server.config import (
    DECLARATION_HEADING_PATH,
    DECLARATION_SIDE_MATCH_MIN,
    DECLARATION_TEMPLATE_PAGE_WIDTH,
    FORM_RULE_FINGERPRINTS,
    LAYOUT_TRUST_ERROR,
)
from server.services import extract_student_id as ex
from server.services import ocr_worker, scan_queue
from server.services.form_layout import FormMatch, identify_form
from tests import test_auto_upload_stack as stack_tests
from tests import test_batch_queue as queue_tests
from tests.test_app import _run_page, needs_node
from tests.test_auto_upload import _key, _queue, _refused_with_nothing_saved
from tests.test_auto_upload_stack import (
    KIDS,
    SETTLE,
    _auto,
    _filed,
    _image,
    _listed,
    _waits_for_a_person,
)
from tests.test_batch_codes_off import (
    _OFF_RESULTS_JS,
    ONE,
    _later,
    _outside_scripts,
    _skipped,
)
from tests.test_batch_queue import (
    FakeDrive,
    Students,
    _batches,
    _confirm,
    _form_page,
    _read,
    _send,
    _tiff,
    _writes,
)

# The queue tests' fixtures, here by name: the queue in the test's own folder
# (autouse), the fake database's students - every child of KIDS in it - the
# fake Drive and a Google sign-in.
queue_dir = queue_tests.queue_dir
students = queue_tests.students
kids = stack_tests.kids
drive = queue_tests.drive
google = queue_tests.google

# A drawn page: A4 at 150 dpi, as the scanner saves it in grey.
WIDTH, HEIGHT = 1240, 1754
# Printed words, each too short to be taken for a ruled line.
_WORDS = ("parents", "agree", "that", "photographs", "reports", "and", "stories", "of", "their", "child", "may")


@pytest.fixture(autouse=True)
def batch_codes(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Batch codes off, as the app ships (overrides tests/conftest.py's fixture)."""
    monkeypatch.setattr(config, "BATCH_CODES", False)
    return False


@pytest.fixture(autouse=True)
def _no_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pages are read by hand here, or by process_one - never by a reader thread of the app's."""
    monkeypatch.setattr(ocr_worker, "should_run_in_process", lambda _debug: False)


# ------------------------------------------------------------------ drawn pages
def _words(page: np.ndarray, top: float, bottom: float) -> None:
    """Lines of printed words from `top` to `bottom` down the page (fractions of its height) - no ruled line."""
    y, n = int(HEIGHT * top), 0
    while y < HEIGHT * bottom:
        line = " ".join(_WORDS[(n + i) % len(_WORDS)] for i in range(9))
        cv2.putText(page, line, (int(WIDTH * 0.06), y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
        y, n = y + 42, n + 3


def _heading(page: np.ndarray, at: float, left: float = 0.06, size: float = 1.0) -> None:
    """The printed "घोषणा" heading, its top `at` down the page and its left edge `left` across it.

    `size` prints it that much larger or smaller than the heading picture's own page.
    """
    picture = cv2.imread(str(DECLARATION_HEADING_PATH), cv2.IMREAD_GRAYSCALE)
    scale = WIDTH / DECLARATION_TEMPLATE_PAGE_WIDTH * size
    picture = cv2.resize(
        picture, (int(picture.shape[1] * scale), int(picture.shape[0] * scale)), interpolation=cv2.INTER_AREA
    )
    y, x = int(HEIGHT * at), int(WIDTH * left)
    page[y : y + picture.shape[0], x : x + picture.shape[1]] = picture


def _back(at: float = 0.52, left: float = 0.06, size: float = 1.0) -> np.ndarray:
    """The back of the one-sheet form: printed words, the heading `at` down the page, more words.

    At 0.04 it is instead the Declaration page of the two-sheet form, its
    heading at the top of its own sheet.
    """
    page = np.full((HEIGHT, WIDTH), 255, np.uint8)
    _words(page, 0.07, at - 0.02)
    _heading(page, at, left, size)
    _words(page, at + 0.05, 0.92)
    return page


def _black_and_white(page: np.ndarray, scan: str) -> np.ndarray:
    """The page as a 100 dpi Black & White scan saves it, then stored as intake stores it (page.jpg).

    `scan` is how the scanner makes it black and white: a threshold - a
    low one thins the strokes, a high one thickens them - or dithering.
    """
    small = cv2.resize(page, (round(WIDTH * 100 / 150), round(HEIGHT * 100 / 150)), interpolation=cv2.INTER_AREA)
    if scan == "dithered":
        made = np.array(Image.fromarray(small).convert("1"), dtype=np.uint8) * 255
    else:
        made = np.where(small < int(scan.split()[-1]), 0, 255).astype(np.uint8)
    return cv2.imdecode(np.frombuffer(scan_queue.encode_page(made), np.uint8), cv2.IMREAD_GRAYSCALE)


def _front(heading_at: float | None = None) -> np.ndarray:
    """A Student ID page: the Hindi form's ruled lines, a Student ID written on the fourth - and the heading too, if asked."""
    page = np.full((HEIGHT, WIDTH), 255, np.uint8)
    _words(page, 0.06, 0.22)
    rows = FORM_RULE_FINGERPRINTS["hindi"]["rows"]
    for n, row in enumerate(rows):
        y = int(HEIGHT * row)
        cv2.putText(page, f"field {n + 1}", (int(WIDTH * 0.06), y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        cv2.line(page, (int(WIDTH * 0.25), y), (int(WIDTH * 0.62), y), 0, 2)
    y = int(HEIGHT * rows[FORM_RULE_FINGERPRINTS["hindi"]["student_id_index"]])
    cv2.putText(page, "AATKHA200618", (int(WIDTH * 0.27), y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 3)
    _words(page, 0.80, 0.92)
    if heading_at is not None:
        _heading(page, heading_at)
    return page


def _bgr(page: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def nothing_read(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every way the reader reads a page for its Student ID, each listed here if it is used at all."""
    used: list[str] = []

    def stand_in(name: str) -> Any:
        def refuse(*_args: Any, **_kwargs: Any) -> Any:
            used.append(name)
            raise AssertionError(f"{name} was used")

        return refuse

    for name in (
        "find_student_id_label",
        "_read_id_from_crop",
        "crop_template_value",
        "_strip_says_declaration",
        "get_rapid_ocr",
        "recognize_with_vision",
    ):
        monkeypatch.setattr(ex, name, stand_in(name))
    return used


# ================================================================== the reader
@pytest.mark.parametrize("at", [0.30, 0.52, 0.56, 0.85])
def test_the_heading_part_way_down_the_left_margin_makes_a_declaration_side(
    nothing_read: list[str], at: float
) -> None:
    result = ex._extract(_bgr(_back(at)), source="duplex.tif (page 2/2)")

    assert (result.skipped, result.skip_reason, result.declaration_side, result.orientation) == (
        True,
        "declaration",
        True,
        0,
    )
    assert result.student_id == "" and result.message == "skipped:declaration"
    # Nothing on it is read: no Student ID looked for, no text read off it.
    assert nothing_read == []


def test_a_back_fed_upside_down_is_found_turned_round(nothing_read: list[str]) -> None:
    result = ex._extract(_bgr(cv2.rotate(_back(), cv2.ROTATE_180)), source="duplex.tif (page 1/2)")

    assert (result.skip_reason, result.declaration_side, result.orientation) == ("declaration", True, 180)
    assert nothing_read == []


@pytest.mark.parametrize("scan", ["threshold 96", "threshold 128", "threshold 160", "dithered"])
@pytest.mark.parametrize("size", [0.9, 1.1])
def test_a_back_scanned_black_and_white_with_its_heading_printed_smaller_or_larger_is_a_declaration_side(
    nothing_read: list[str], scan: str, size: float
) -> None:
    """The scanner setup's 100 dpi Black & White, however the scanner makes it - and a heading not at the picture's size.

    Matched at the picture's own size and sharp, these scored well under
    the threshold; blurred and at a range of sizes they match as a grey
    scan does.
    """
    result = ex._extract(_bgr(_black_and_white(_back(size=size), scan)), source="duplex.tif (page 2/2)")

    assert (result.skip_reason, result.declaration_side, result.orientation) == ("declaration", True, 0)
    assert nothing_read == []


@pytest.mark.parametrize("scan", ["grey", "threshold 96", "threshold 160", "dithered"])
def test_printed_words_down_the_left_margin_are_no_heading_however_scanned(scan: str) -> None:
    page = np.full((HEIGHT, WIDTH), 255, np.uint8)
    _words(page, 0.05, 0.95)
    if scan != "grey":
        page = _black_and_white(page, scan)
    gray = cv2.cvtColor(ex.preprocess_for_ocr(_bgr(page)), cv2.COLOR_BGR2GRAY)

    for turned in (gray, cv2.rotate(gray, cv2.ROTATE_180)):
        score, _top = ex._heading_match(turned)
        assert score < DECLARATION_SIDE_MATCH_MIN - 0.1, score


def test_the_heading_at_the_top_left_of_its_own_sheet_is_a_declaration_page_as_before(
    nothing_read: list[str],
) -> None:
    """The two-sheet form's Declaration page: the top-left check finds it, and it is no sheet's back."""
    page = _bgr(_back(0.04))
    assert ex._looks_like_declaration(ex.preprocess_for_ocr(page))

    upright = ex._extract(page, source="Scan 2.tiff")
    turned = ex._extract(_bgr(cv2.rotate(_back(0.04), cv2.ROTATE_180)), source="Scan 4.tiff")

    assert (upright.skip_reason, upright.declaration_side, upright.orientation) == ("declaration", False, 0)
    # Fed upside down it is found turned round now, and stays a page of its own sheet.
    assert (turned.skip_reason, turned.declaration_side, turned.orientation) == ("declaration", False, 180)
    assert nothing_read == []


def test_a_student_id_page_with_handwriting_in_its_box_is_never_taken_for_a_declaration_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever the search down the margin finds; the two-sheet form's top-left check is made before it, as before."""
    page = _bgr(_front(heading_at=0.735))
    prepared = ex.preprocess_for_ocr(page)
    # The heading alone would claim it ...
    score, _top = ex._heading_match(cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY))
    assert score >= DECLARATION_SIDE_MATCH_MIN
    # ... but its ruled layout pins the Student ID box, with writing in it.
    layout = identify_form(prepared)
    assert layout is not None and layout.language == "hindi" and layout.error <= LAYOUT_TRUST_ERROR
    read: list[str] = []

    def read_crop(crop: np.ndarray, language: str, tag: str, **_options: Any) -> list[tuple[str, float, str, str]]:
        read.append(tag)
        return [("AATKHA200618", 0.9, language, tag)]

    monkeypatch.setattr(ex, "FAST_MODE", True)
    monkeypatch.setattr(ex, "_save_debug_crop", lambda crop, tag: None)
    monkeypatch.setattr(ex, "vision_configured", lambda: False)
    monkeypatch.setattr(ex, "_read_id_from_crop", read_crop)

    result = ex._extract(page, source="duplex.tif (page 1/2)")

    assert not result.skipped and result.skip_reason == ""
    assert result.student_id == "AATKHA200618" and result.form_language == "hindi"
    assert read == ["layout:hindi"]


@pytest.mark.parametrize(
    ("page", "looked_at"),
    [
        # No ruled layout: the top strip, then the label line across the page.
        pytest.param(lambda: _back(left=0.55), ["top strip", "label"], id="the heading away from the left margin"),
        # Its layout locates the Student ID box, which is read.
        pytest.param(lambda: _front(), ["layout:hindi"], id="a student id page with no heading"),
    ],
)
def test_a_page_without_the_heading_at_its_left_margin_is_read_for_its_student_id(
    monkeypatch: pytest.MonkeyPatch, page: Any, looked_at: list[str]
) -> None:
    looked: list[str] = []

    def strip(image: np.ndarray) -> bool:
        looked.append("top strip")
        return False

    def label(image: np.ndarray) -> None:
        looked.append("label")

    def read_crop(crop: np.ndarray, language: str, tag: str, **_options: Any) -> list[tuple[str, float, str, str]]:
        looked.append(tag)
        return []

    def template(image: np.ndarray, language: str) -> np.ndarray:
        raise ValueError("no box")

    logged: list[str] = []
    monkeypatch.setattr(ex, "FAST_MODE", True)
    monkeypatch.setattr(ex, "_save_debug_crop", lambda crop, tag: None)
    monkeypatch.setattr(ex, "vision_configured", lambda: False)
    monkeypatch.setattr(ex, "_strip_says_declaration", strip)
    monkeypatch.setattr(ex, "find_student_id_label", label)
    monkeypatch.setattr(ex, "_read_id_from_crop", read_crop)
    monkeypatch.setattr(ex, "crop_template_value", template)
    monkeypatch.setattr(ex, "debug_log", logged.append)

    result = ex.extract_from_image(_bgr(page()), source="Scan 1.tiff")

    assert not result.skipped and result.student_id == ""
    assert result.error and result.error.startswith("No Student ID found on this page")
    assert looked == looked_at
    # The debug log says [SKIP] only of a page that is skipped: the heading
    # search and the ID box's ink are logged under their own names.
    assert not [line for line in logged if line.startswith("[SKIP]")]
    searched = [line for line in logged if line.startswith("[HEADING]")]
    assert len(searched) == (2 if looked_at[0] == "top strip" else 0)


@pytest.mark.parametrize(
    ("error", "skipped"),
    [(0.0010, True), (0.0062, False)],
    ids=["a box its ladder pins", "a box a loose fit only guesses at"],
)
def test_an_empty_student_id_box_skips_the_page_only_where_its_ladder_pins_the_box(
    monkeypatch: pytest.MonkeyPatch, error: float, skipped: bool
) -> None:
    """A loose fit's box lands rows from the Student ID, so its ink says nothing: that page is read like any other.

    Upright fronts that fit only loosely - kept upright now rather than
    turned at a loose fit's word - were otherwise skipped unread.
    """
    # The box this layout locates is a strip of empty paper below the page's words.
    layout = FormMatch(
        language="hindi", confidence=0.3, student_id_roi=(0.62, 0.93, 0.95, 0.97), rule_count=6, error=error,
        source="rules",
    )
    looked: list[str] = []

    def read_crop(crop: np.ndarray, language: str, tag: str, **_options: Any) -> list[tuple[str, float, str, str]]:
        looked.append(tag)
        return []

    def template(image: np.ndarray, language: str) -> np.ndarray:
        raise ValueError("no box")

    monkeypatch.setattr(ex, "identify_form", lambda image: layout)
    monkeypatch.setattr(ex, "FAST_MODE", True)
    monkeypatch.setattr(ex, "_save_debug_crop", lambda crop, tag: None)
    monkeypatch.setattr(ex, "vision_configured", lambda: False)
    monkeypatch.setattr(ex, "find_student_id_label", lambda image: looked.append("label"))
    monkeypatch.setattr(ex, "_read_id_from_crop", read_crop)
    monkeypatch.setattr(ex, "crop_template_value", template)

    result = ex.extract_from_image(_bgr(_front()), source="Scan 1.tiff")

    if skipped:
        assert (result.skipped, result.skip_reason) == (True, "empty_id") and looked == []
    else:
        assert not result.skipped and result.error.startswith("No Student ID found on this page")
        assert looked == ["label", "layout:hindi"]


def test_the_worker_keeps_a_back_as_it_came_and_turns_one_fed_upside_down(
    client: FlaskClient, fake_db: Any, queue_dir: Path
) -> None:
    """Neither is read for a Student ID; the one found only turned round is stored turned, so its thumbnail and PDF are upright."""
    upright = _back()
    response = _send(
        client, "duplex.tif", _tiff([Image.fromarray(upright), Image.fromarray(cv2.rotate(upright, cv2.ROTATE_180))])
    )
    assert response.status_code == 202, response.get_json()
    ids = [page["page_id"] for page in response.get_json()["pages"]]

    assert [ocr_worker.process_one() for _ in ids] == [ocr_worker.READ, ocr_worker.READ]

    for page_id, turned in zip(ids, (0, 180), strict=True):
        result = scan_queue.load_result(page_id)
        assert result is not None
        assert (result.skip_reason, result.declaration_side, result.orientation) == ("declaration", True, turned)
        assert result.student_id == "" and result.message == "Declaration page, no Student ID."
        stored = cv2.imread(str(scan_queue.page_dir(page_id) / scan_queue.PAGE_IMAGE), cv2.IMREAD_GRAYSCALE)
        score, top = ex._heading_match(stored)
        assert score >= DECLARATION_SIDE_MATCH_MIN and abs(top - 0.52) < 0.01
    # Nobody to look up on a Declaration page.
    assert fake_db.statements == []


# ================================================================== pairing
def _file(client: FlaskClient, pages: int, name: str = "Scan 1.tif", **form: str) -> list[str]:
    """One file of `pages` images, as the scanner saves a stack scanned both sides - sent with the switch on from scan page ONE."""
    images = [_form_page((620 + 10 * n, 874)) for n in range(pages)]
    response = _send(client, name, _tiff(images), auto_upload="1", sender=ONE, **form)
    assert response.status_code == 202, response.get_json()
    return [page["page_id"] for page in response.get_json()["pages"]]


def _id_side(page_id: str, n: int) -> None:
    """KIDS[n]'s Student ID side, read and matched exactly."""
    _read(page_id, None, KIDS[n]["student_id"], dict(KIDS[n], matched_via="exact"))


def _declaration_side(page_id: str) -> None:
    """A Declaration side, as the reader finds one: its heading part way down."""
    _skipped(page_id, declaration_side=True)


def _ids(doc: dict[str, Any]) -> list[str]:
    return [page["page_id"] for page in doc["pages"]]


def _sides(doc: dict[str, Any]) -> list[bool]:
    return [page["both_sides"] for page in doc["pages"]]


def test_a_file_scanned_both_sides_is_one_form_a_sheet(google: FlaskClient, fake_db: Any, kids: Students) -> None:
    pages = _file(google, 4)
    _id_side(pages[0], 0)
    _declaration_side(pages[1])
    _id_side(pages[2], 1)
    _declaration_side(pages[3])

    docs = _listed(google)

    assert set(docs) == {_key(pages[0]), _key(pages[2])}
    for at in (0, 2):
        doc = docs[_key(pages[at])]
        assert _ids(doc) == [pages[at], pages[at + 1]] and _sides(doc) == [True, True]
        assert doc["status"] == "ready" and doc["attention"] == []
    assert _batches(google)["unsorted"] == []


def test_a_sheet_fed_the_other_way_up_is_one_form_its_student_id_side_first_in_the_pdf(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    pages = _file(google, 4)
    _declaration_side(pages[0])
    _id_side(pages[1], 0)
    _id_side(pages[2], 1)
    _declaration_side(pages[3])

    docs = _listed(google)

    assert set(docs) == {_key(pages[1]), _key(pages[2])}
    assert _ids(docs[_key(pages[1])]) == [pages[1], pages[0]] and _sides(docs[_key(pages[1])]) == [True, True]
    assert _ids(docs[_key(pages[2])]) == [pages[2], pages[3]]
    # A person checks it on the paper first; the sheet after it is Ready.
    assert docs[_key(pages[1])]["status"] == "needs_attention"
    assert docs[_key(pages[1])]["attention"] == ["declaration_side_first"]
    assert docs[_key(pages[2])]["status"] == "ready" and docs[_key(pages[2])]["attention"] == []
    assert _batches(google)["unsorted"] == []

    response = _confirm(google, _key(pages[1]), KIDS[0]["student_id"])

    assert response.status_code == 200, response.get_json()
    assert _filed(drive, 0) == [_image(pages[1]), _image(pages[0])]


@pytest.mark.parametrize("automatic", [False, True], ids=["automatic upload off", "automatic upload on"])
def test_a_file_a_side_out_of_step_has_every_sheet_checked_by_a_person(
    google: FlaskClient, fake_db: Any, kids: Students, monkeypatch: pytest.MonkeyPatch, automatic: bool
) -> None:
    """Its first image a back whose front never came: each sheet is paired with the sheet before's Declaration side.

    By place every sheet then looks fed the other way up, and only the paper
    shows which it is - so each needs a person's look, whatever automatic
    upload is set to.
    """
    monkeypatch.setattr(config, "AUTO_UPLOAD", automatic)
    pages = _file(google, 4)
    _declaration_side(pages[0])
    _id_side(pages[1], 1)
    _declaration_side(pages[2])
    _id_side(pages[3], 2)
    _later(SETTLE + 1)

    docs = _listed(google)

    assert set(docs) == {_key(pages[1]), _key(pages[3])}
    for at in (1, 3):
        doc = docs[_key(pages[at])]
        assert _ids(doc) == [pages[at], pages[at - 1]]
        assert doc["status"] == "needs_attention" and doc["attention"] == ["declaration_side_first"]
        assert doc["message"] == (
            "Its Declaration side came before its Student ID side in the file - check on the paper that the two are"
            " one sheet."
        )
        assert doc["can_confirm"] is True and doc["auto_upload"] is False
        assert doc["auto_upload_refusal"] == (
            "its sheet was scanned Declaration side first" if automatic else "automatic upload is switched off on this server"
        )


def test_sheets_read_before_the_rest_of_their_file_cannot_be_confirmed_yet(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """The rest of the file may show it does not pair into sheets - here a one-sided page first and last.

    Pages 3 to 8 read before pages 1 and 2 pair by place as sheets fed the
    other way up, each with the sheet before's Declaration side: they wait,
    with no Confirm. Once the whole file is read it pairs by scan order.
    """
    cover, i0, d0, i1, d1, i2, d2, note = _file(google, 8)
    _id_side(i1, 1)
    _id_side(i2, 2)
    for back in (d0, d1, d2):
        _declaration_side(back)
    _read(note, None)

    so_far = _listed(google)

    for first, back in ((i1, d0), (i2, d1)):
        doc = so_far[_key(first)]
        assert _ids(doc) == [first, back]
        assert doc["status"] == "waiting" and doc["can_confirm"] is False
        assert doc["message"] == (
            "Waiting for the rest of its file to be read - a file scanned both sides is paired sheet by sheet only"
            " once every page of it has been."
        )
    response = _confirm(google, _key(i1), KIDS[1]["student_id"], [i1, d0])
    assert response.status_code == 409, response.get_json()
    assert response.get_json()["error"] == so_far[_key(i1)]["message"] + " Try again in a moment."
    assert _writes(fake_db) == [] and drive.calls == []

    _read(cover, None)
    _id_side(i0, 0)
    docs = _listed(google)

    for first, back in ((i0, d0), (i1, d1), (i2, d2)):
        doc = docs[_key(first)]
        assert _ids(doc) == [first, back] and _sides(doc) == [False, False]
        assert "gap_in_stack" in doc["attention"]


def test_a_file_scanned_both_sides_is_ready_once_its_last_page_is_read(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    pages = _file(google, 4)
    _id_side(pages[0], 0)
    _declaration_side(pages[1])
    _id_side(pages[2], 1)

    assert _listed(google)[_key(pages[0])]["status"] == "waiting"

    _declaration_side(pages[3])
    docs = _listed(google)

    assert [docs[_key(pages[at])]["status"] for at in (0, 2)] == ["ready", "ready"]


def test_a_side_read_before_its_other_side_is_taken_as_still_being_read(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """Not shown until its other side is read; nothing is held across it, and a form before it waits for it."""
    front = _queue(google, "Scan 1.tiff")[0]
    pages = _file(google, 4, "Scan 2.tif")
    after = _file(google, 2, "Scan 3.tif")
    _id_side(front, 0)
    # Both sheets fed the other way up; the first one's back is read before
    # its front, the second one's front before its back.
    _declaration_side(pages[0])
    _id_side(pages[3], 2)
    _id_side(after[0], 3)
    _declaration_side(after[1])

    so_far = _listed(google)

    assert set(so_far) == {_key(front), _key(after[0])}
    assert _batches(google)["unsorted"] == []
    assert so_far[_key(front)]["status"] == "waiting"
    assert so_far[_key(front)]["message"] == "Waiting for its Declaration page - the page after it is still being read."
    assert so_far[_key(after[0])]["attention"] == []

    _id_side(pages[1], 1)
    _declaration_side(pages[2])
    docs = _listed(google)

    assert _ids(docs[_key(pages[1])]) == [pages[1], pages[0]] and _ids(docs[_key(pages[3])]) == [pages[3], pages[2]]
    assert _ids(docs[_key(front)]) == [front] and "no_declaration" in docs[_key(front)]["attention"]


def test_a_file_that_does_not_pair_into_sheets_is_paired_by_scan_order(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """A back the reader missed, read as a page with no Student ID: the file pairs as any other, back fed first and all."""
    pages = _file(google, 4)
    _declaration_side(pages[0])
    _id_side(pages[1], 0)
    _id_side(pages[2], 1)
    _read(pages[3], None)

    docs = _listed(google)
    unsorted = {page["page_id"]: page for page in _batches(google)["unsorted"]}

    # By scan order the first back has no Student ID page before it, and the
    # one missed is offered to the form straight before it - nothing by place.
    assert set(docs) == {_key(pages[1]), _key(pages[2])}
    assert _ids(docs[_key(pages[1])]) == [pages[1]] and _ids(docs[_key(pages[2])]) == [pages[2]]
    assert set(unsorted) == {pages[0], pages[3]}
    assert unsorted[pages[0]]["message"] == "A Declaration page with no Student ID page right before it."
    assert unsorted[pages[3]]["attach_to"]["page_id"] == pages[2]
    assert not any(page["both_sides"] for doc in docs.values() for page in doc["pages"])
    assert not any(page["both_sides"] for page in unsorted.values())
    assert all("orphan_in_stack" in doc["attention"] for doc in docs.values())


def test_a_file_of_an_odd_number_of_pages_is_paired_by_scan_order(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    pages = _file(google, 3)
    _id_side(pages[0], 0)
    _declaration_side(pages[1])
    _id_side(pages[2], 1)

    docs = _listed(google)

    assert _ids(docs[_key(pages[0])]) == [pages[0], pages[1]] and _sides(docs[_key(pages[0])]) == [False, False]
    assert _ids(docs[_key(pages[2])]) == [pages[2]]


def test_files_of_one_image_each_with_no_running_number_pair_by_scan_order_as_before(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """A sheet sent as two files: its sides pair as the Student ID page and the Declaration page after it - or, back first, not at all.

    Their names end in no number, so they are no scanner session's images.
    """
    front, back = _queue(google, "front.tiff")[0], _queue(google, "back.tiff")[0]
    later_back, later_front = _queue(google, "back again.tiff")[0], _queue(google, "front again.tiff")[0]
    _id_side(front, 0)
    _declaration_side(back)
    _declaration_side(later_back)
    _id_side(later_front, 1)

    docs = _listed(google)
    [orphan] = _batches(google)["unsorted"]

    assert _ids(docs[_key(front)]) == [front, back] and _sides(docs[_key(front)]) == [False, False]
    assert _ids(docs[_key(later_front)]) == [later_front]
    assert orphan["page_id"] == later_back and orphan["message"].startswith(
        "A Declaration page right after another Declaration page"
    )
    assert "orphan_in_stack" in docs[_key(front)]["attention"]


def test_the_two_sheet_form_in_one_file_pairs_by_scan_order_as_before(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """Its Declaration pages are headed at the top of their own sheets: a stack fed Declaration page first is caught, not paired by place."""
    pages = _file(google, 4)
    _skipped(pages[0])
    _id_side(pages[1], 0)
    _skipped(pages[2])
    _id_side(pages[3], 1)

    docs = _listed(google)
    [orphan] = _batches(google)["unsorted"]

    assert orphan["page_id"] == pages[0]
    assert _ids(docs[_key(pages[1])]) == [pages[1], pages[2]] and _sides(docs[_key(pages[1])]) == [False, False]
    assert _ids(docs[_key(pages[3])]) == [pages[3]]
    assert all("orphan_in_stack" in doc["attention"] for doc in docs.values())


# ================================================================== automatic upload
def test_a_stack_of_files_scanned_both_sides_in_plain_order_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    first = _file(google, 4, "Scan 1.tif")
    second = _file(google, 2, "Scan 2.tif")
    sheets = [(first[0], first[1]), (first[2], first[3]), (second[0], second[1])]
    for n, (id_side, back) in enumerate(sheets):
        _id_side(id_side, n)
        _declaration_side(back)
    _later(SETTLE + 1)

    docs = _listed(google)

    for id_side, _back_side in sheets:
        assert docs[_key(id_side)]["auto_upload"] is True, docs[_key(id_side)]["auto_upload_refusal"]
    response = _auto(google, 2, second[0])
    assert response.status_code == 200, response.get_json()
    assert _filed(drive, 2) == [_image(second[0]), _image(second[1])]


def test_a_sheet_fed_the_other_way_up_keeps_every_form_of_its_stack_waiting_for_a_person(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    pages = _file(google, 4)
    _id_side(pages[0], 0)
    _declaration_side(pages[1])
    _declaration_side(pages[2])
    _id_side(pages[3], 1)
    _later(SETTLE + 1)

    _waits_for_a_person(google, pages[0], "its stack has a sheet scanned Declaration side first")
    _waits_for_a_person(google, pages[3], "its sheet was scanned Declaration side first")
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)
    _refused_with_nothing_saved(_auto(google, 1, pages[3]), fake_db, drive)


@pytest.mark.parametrize("fault", ["a back the reader missed", "a blank side", "an odd page"])
def test_a_file_that_does_not_pair_into_sheets_keeps_its_stack_waiting_for_a_person(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, fault: str
) -> None:
    pages = _file(google, 3 if fault == "an odd page" else 4)
    _id_side(pages[0], 0)
    _declaration_side(pages[1])
    _id_side(pages[2], 1)
    if fault == "a back the reader missed":
        _read(pages[3], None)
    elif fault == "a blank side":
        _skipped(pages[3], "blank")
    _later(SETTLE + 1)

    _waits_for_a_person(
        google,
        pages[0],
        "its stack has a file scanned both sides whose sheets are not each a Student ID side and a Declaration side",
    )
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)


def test_a_file_scanned_both_sides_in_among_other_scans_keeps_its_stack_waiting_for_a_person(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    sheet = _file(google, 2, "Scan 1.tif")
    _id_side(sheet[0], 0)
    _declaration_side(sheet[1])
    # One image each, named with no running number: no scanner session.
    front, back = _queue(google, "front.tiff")[0], _queue(google, "back.tiff")[0]
    _id_side(front, 1)
    _declaration_side(back)
    _later(SETTLE + 1)

    for page_id in (sheet[0], front):
        _waits_for_a_person(google, page_id, "its stack has a file scanned both sides in among other scans")
    _refused_with_nothing_saved(_auto(google, 0, sheet[0]), fake_db, drive)


def test_with_automatic_upload_off_a_clean_stack_of_sheets_still_waits_for_a_person(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "AUTO_UPLOAD", False)
    pages = _file(google, 2)
    _id_side(pages[0], 0)
    _declaration_side(pages[1])
    _later(SETTLE + 1)

    _waits_for_a_person(google, pages[0], "automatic upload is switched off on this server")
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)


# ================================================================== the page
def test_page_says_to_scan_the_one_sheet_form_both_sides_and_the_two_sheet_form_one_side(client: FlaskClient) -> None:
    said = " ".join(_outside_scripts(client.get("/").get_data(as_text=True)).split())

    for told in (
        (
            "For the form printed on both sides of one sheet, scan both sides into one file for each stack, with"
            " skip blank pages off and double-feed detection on - numbered images saved one a side are paired"
            f" too, once no more have come for {config.STACK_SETTLE_MINUTES} minutes; for the two-sheet form,"
            " one side."
        ),
        (
            "A file of several pages scanned both sides is paired sheet by sheet - pages 1 and 2, 3 and 4, and so"
            " on - once all of it has been read, and so are the numbered images a scanner saves one a side, by"
            f" their numbers, once no more have come for {config.STACK_SETTLE_MINUTES} minutes: each sheet's"
            " Student ID side and Declaration side are one document, the Student ID side first, whichever side"
            " came first; one whose Declaration side came first waits for you to check it."
        ),
        (
            "So does every form of a stack with a sheet scanned Declaration side first, a file scanned both sides"
            " that does not pair into sheets, or such a file in among other scans."
        ),
    ):
        assert told in said, told


@needs_node
@pytest.mark.parametrize(
    "navigator",
    [
        {"userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "platform": "Win32", "maxTouchPoints": 0},
        {"userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "platform": "MacIntel", "maxTouchPoints": 0},
        {"userAgent": "Mozilla/5.0 (X11; Linux x86_64)", "platform": "Linux x86_64", "maxTouchPoints": 0},
        {"userAgent": "Mozilla/5.0 (X11; CrOS x86_64 14541.0.0)", "platform": "Linux x86_64", "maxTouchPoints": 0},
    ],
)
def test_page_scanner_setup_says_which_sides_to_scan(
    client: FlaskClient, tmp_path: Path, navigator: dict[str, Any]
) -> None:
    out = _run_page(client, tmp_path, rf"""
      const page = loadPage({{
        navigator: {json.dumps(navigator)},
        showDirectoryPicker: async () => makeFolder(),
        indexedDB: makeIndexedDB(),
      }});
      return page.byId["scan-folder-steps"].innerHTML;
    """)

    sides = (
        "For the form printed on <strong>both sides of one sheet</strong>, scan <strong>both sides</strong> (duplex)"
        " into one file for each stack, with <strong>skip blank pages</strong> off, so every sheet gives two images:"
        " its Student ID side and its Declaration side. Numbered images saved one a side are paired sheet by sheet"
        f" too, once no more have come for {config.STACK_SETTLE_MINUTES} minutes. Keep <strong>double-feed"
        " detection</strong> on: two sheets fed as one come in as one child's Student ID side and the next child's"
        " Declaration side, and nothing else shows it. For the <strong>two-sheet form</strong>, scan one side"
        " (<strong>simplex</strong>)."
    )
    assert sides in out
    assert "Load the forms with page 1 of each - its Student ID page or side - face down" in out
    # One file for each scan is still what the steps ask for: numbered
    # images one a side are paired too, but only once the scan has finished.
    assert "JPEG" not in out and "JPG" not in out
    if navigator["platform"] == "Win32":
        # The profile leaves the sides to the step after the button.
        assert (
            "file type <strong>PDF</strong> (or TIFF) with all of a scan's pages in one file" in out
            and "<strong>multifeed detection</strong> on, and the sides your form needs (below)." in out
        )
        assert out.index("Assign to scanner button") < out.index(sides)
    elif navigator["platform"] == "MacIntel":
        assert "<strong>Format</strong> PDF with <strong>Combine into single document</strong> ticked" in out


@needs_node
def test_page_calls_a_sheets_two_parts_its_student_id_side_and_its_declaration_side(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Only a form the server paired by place; one paired by scan order is a Student ID page and a Declaration page."""
    out = _run_page(client, tmp_path, _OFF_RESULTS_JS + r"""
      const page = await openPage();
      const inFile = (id, at, extra) => pg(id, Object.assign({ source_name: "duplex.tif", source_page: at, source_pages: 6, both_sides: true }, extra));
      const back = inFile("p1", 1, { skip_reason: "declaration", message: "Declaration page, no Student ID." });
      const front = inFile("p2", 2, { student_id: ASHA.student_id, matched_student: ASHA, skip_reason: "" });
      const misread = inFile("p3", 3, { student_id: "AATKHA200619", skip_reason: "" });
      const itsBack = inFile("p4", 4, { skip_reason: "declaration" });
      const loose = inFile("p5", 5, { skip_reason: "", message: "No Student ID found on this page.",
        declaration_pages: [inFile("p6", 6, { skip_reason: "declaration" })] });
      const idPage = pg("p7", { source_name: "Scan 7.tiff", student_id: ASHA.student_id, matched_student: ASHA, skip_reason: "", both_sides: false });
      const declaration = pg("p8", { source_name: "Scan 8.tiff", skip_reason: "declaration", both_sides: false });
      await answerBatches(page, offDocs([
        pdoc(front, { parts: 2, pages: [front, back], student_id: ASHA.student_id, matched_student: ASHA }),
        pdoc(misread, { parts: 2, pages: [misread, itsBack], status: "needs_attention", attention: ["no_match"],
          student_id: "AATKHA200619", message: "No student with ID AATKHA200619 at School A. Check the ID against the paper." }),
        pdoc(idPage, { parts: 2, pages: [idPage, declaration], student_id: ASHA.student_id, matched_student: ASHA }),
      ], [loose]));
      const sheet = card(page, "duplex.tif - page 2");
      const check = card(page, "duplex.tif - page 3");
      const pair = card(page, "Scan 7.tiff");
      const [waiting] = unsortedCards(page);
      const parts = (c) => c._all().find((e) => e.classList.contains("ocr-doc-parts")).textContent;
      const labels = (c, name) => c._all().filter((e) => e.classList.contains(name)).map((e) => e.textContent);
      const shown = {
        sheetParts: parts(sheet), sheetThumbs: labels(sheet, "ocr-thumb-label"),
        checkParts: parts(check), checkScans: labels(check, "ocr-scan-caption"),
        pairParts: parts(pair), pairThumbs: labels(pair, "ocr-thumb-label"),
        waitingParts: parts(waiting), waitingThumbs: labels(waiting, "ocr-thumb-label"), waitingText: textIn(waiting),
      };
      fieldIn(waiting, "id").value = "mohsha011021";
      buttonIn(waiting, "Check ID").fire("click");
      await answer(page.calls, "/student-id", { page: {}, document: {} });
      await tick(5);
      const said = textIn(waiting);
      buttonIn(sheet, "Confirm and upload").fire("click");
      const confirm = await answer(page.calls, "/confirm", { document: { status: "uploaded", key: "Pp2~ab12cd34" } });
      return { shown, said, confirmBody: JSON.parse(confirm.opts.body) };
    """)

    shown = out["shown"]
    assert shown["sheetParts"] == "Student ID side + Declaration side"
    assert shown["sheetThumbs"] == ["Student ID side", "Declaration side"]
    assert shown["checkParts"] == "Student ID side + Declaration side"
    assert shown["checkScans"] == ["duplex.tif - page 3 - Student ID side", "duplex.tif - page 4 - Declaration side"]
    assert shown["pairParts"] == "Student ID page + Declaration page"
    assert shown["pairThumbs"] == ["Student ID page", "Declaration page"]
    assert shown["waitingParts"] == "duplex.tif - page 5 + Declaration side"
    assert shown["waitingThumbs"] == ["Student ID side", "Declaration side"]
    assert "Its Declaration side is the other side of this sheet - once the Student ID is typed" in shown["waitingText"]
    assert "the page and its Declaration side become one document" in out["said"]
    # The Student ID side first, as the server keeps it - though its back came first.
    assert out["confirmBody"] == {"student_id": "MOHSHA011021", "page_ids": ["p2", "p1"]}
