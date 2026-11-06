"""The form choice: which paper form a stack is, picked on the scan page, and each page read and paired by its own form.

There are two consent forms: the two-sheet form - a Student ID page, then a
Declaration page of its own, headed at the top left - and the one-sheet
form, printed on both sides, its Declaration side on the back with the
heading part way down. With batch codes off, as the app ships:

  - The scan page asks which form is being scanned before anything can be
    sent: until one is picked the scan folder, the drop zone and Choose files
    are locked, with one line saying so. It is remembered on that computer
    (and the page still works when the browser keeps nothing), shown at the
    top and changeable: a change applies to the scans that come in after it.
    The setup steps and the help say to scan both sides only for the
    one-sheet form, and one side for the two-sheet form.
  - Every scan carries the choice ("form": two_sheet or one_sheet); a value
    the server does not know is refused with nothing stored. A page with no
    form - sent by an older page, queued before the choice existed, read by
    the Drive inbox - is the two-sheet form.
  - The two-sheet form is read and paired as before the one-sheet form was
    known: its Declaration page by the heading at the top left only - never
    a Declaration side, however low in the frame it was scanned - an
    upside-down Student ID page turned upright at any better fit, and its
    pages paired by scan order, never by place or as a scanner session. The
    one-sheet form's rules are tests/test_both_sides.py and
    tests/test_scan_sessions.py.
  - A run that mixes the two forms reads and pairs each page by its own
    form, and with a one-sheet file or numbered scan paired by place in it,
    none of its forms goes up by itself.
  - With batch codes on, pages still group by their codes, and each is
    still read by its own form.

Every page here is drawn; the database is the FakeDB and Drive the
FakeDrive. Nothing leaves the test.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from flask.testing import FlaskClient
from PIL import Image

from server import config, inbox_watcher
from server.blueprints import batch_routes
from server.services import (
    batch_store,
    drive_inbox,
    ocr_worker,
    paper_forms,
    scan_queue,
)
from server.services import extract_student_id as ex
from tests import test_auto_upload_stack as stack_tests
from tests import test_batch_queue as queue_tests
from tests.test_app import _run_page, needs_node
from tests.test_auto_upload import _key, _queue, _refused_with_nothing_saved
from tests.test_auto_upload_stack import (
    KIDS,
    SETTLE,
    _auto,
    _listed,
    _waits_for_a_person,
)
from tests.test_batch_codes_off import ONE, _later, _skipped
from tests.test_batch_queue import (
    OWNER,
    Readers,
    _batches,
    _form_page,
    _jpeg,
    _read,
    _send,
    _tiff,
)
from tests.test_both_sides import (
    _back,
    _bgr,
    _declaration_side,
    _file,
    _id_side,
    _ids,
    _sides,
)

# The queue tests' fixtures, here by name: the queue in the test's own folder
# (autouse), the fake readers, the fake database's students - every child of
# KIDS in it - the fake Drive and a Google sign-in.
queue_dir = queue_tests.queue_dir
readers = queue_tests.readers
students = queue_tests.students
kids = stack_tests.kids
drive = queue_tests.drive
google = queue_tests.google

TWO_SHEET, ONE_SHEET = paper_forms.TWO_SHEET, paper_forms.ONE_SHEET
NODE = shutil.which("node")
MIXED = "its stack has a file scanned both sides in among other scans"
# What a one-sheet sheet says while its file, or its session, may still grow.
REST = ("Waiting for the rest of its file", "Waiting for the rest of its scan")


@pytest.fixture(autouse=True)
def batch_codes(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Batch codes off, as the app ships (overrides tests/conftest.py's fixture)."""
    monkeypatch.setattr(config, "BATCH_CODES", False)
    return False


@pytest.fixture(autouse=True)
def _no_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pages are read by hand here, or by process_one - never by a reader thread of the app's."""
    monkeypatch.setattr(ocr_worker, "should_run_in_process", lambda _debug: False)


def _sent_as(client: FlaskClient, form: str | None, pages: int = 1, name: str = "Scan 1.tiff", **fields: str) -> list[str]:
    """A file of `pages` drawn pages sent to POST /api/scans with `form` - with no form field at all for None."""
    if form is not None:
        fields["form"] = form
    images = [_form_page((620 + 10 * n, 874)) for n in range(pages)]
    response = _send(client, name, _tiff(images), **fields)
    assert response.status_code == 202, response.get_json()
    return [page["page_id"] for page in response.get_json()["pages"]]


def _numbered(client: FlaskClient, form: str, numbers: range) -> list[str]:
    """Images saved one a side as a scanner numbers them - Scan_0001.jpg, Scan_0002.jpg ... - each its own file, as `form`."""
    pages = []
    for number in numbers:
        data = _jpeg(_form_page((600 + 4 * number, 874)))
        response = _send(client, f"Scan_{number:04d}.jpg", data, auto_upload="1", sender=ONE, form=form)
        assert response.status_code == 202, response.get_json()
        pages.append(response.get_json()["pages"][0]["page_id"])
    return pages


def _unsorted(client: FlaskClient) -> dict[str, dict[str, Any]]:
    return {page["page_id"]: page for page in _batches(client)["unsorted"]}


@pytest.fixture
def read_as_production(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The reader's slow recognisers stood in for, each noted as it is used; and the search down the margin, which must not run.

    A page read for its Student ID goes through the top strip and the label
    line (no layout on a drawn page with no rules); none reads anything here.
    """
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

    def search(*_args: Any) -> Any:
        looked.append("searched down the margin")
        raise AssertionError("the two-sheet form is never searched down its margin")

    monkeypatch.setattr(ex, "FAST_MODE", True)
    monkeypatch.setattr(ex, "_save_debug_crop", lambda crop, tag: None)
    monkeypatch.setattr(ex, "vision_configured", lambda: False)
    monkeypatch.setattr(ex, "_strip_says_declaration", strip)
    monkeypatch.setattr(ex, "find_student_id_label", label)
    monkeypatch.setattr(ex, "_read_id_from_crop", read_crop)
    monkeypatch.setattr(ex, "crop_template_value", template)
    monkeypatch.setattr(ex, "_find_heading", search)
    monkeypatch.setattr(ex, "_heading_match", search)
    return looked


# ================================================================== the form each page is sent as
@pytest.mark.parametrize(
    ("sent", "kept"),
    [
        (ONE_SHEET, ONE_SHEET),
        (TWO_SHEET, TWO_SHEET),
        (" One_Sheet ", ONE_SHEET),
        # An older page, or any other client, sends none - or an empty one.
        (None, TWO_SHEET),
        ("", TWO_SHEET),
    ],
)
def test_every_page_of_a_scan_keeps_the_form_it_was_sent_as(
    client: FlaskClient, queue_dir: Path, sent: str | None, kept: str
) -> None:
    pages = _sent_as(client, sent, pages=3, name="stack.tif")

    for page_id in pages:
        assert json.loads((queue_dir / page_id / "meta.json").read_text())["form"] == kept
        assert scan_queue.load_meta(page_id).form == kept


@pytest.mark.parametrize("sent", ["duplex", "both_sides", "two-sheet", "1", "one_sheet,two_sheet"])
def test_a_scan_with_a_form_this_server_does_not_know_is_refused_with_nothing_stored(
    client: FlaskClient, queue_dir: Path, sent: str
) -> None:
    """Read as either form it would be a guess - a one-sheet stack read as two sheets has every back read as a page with no Student ID."""
    response = _send(client, "Scan 1.tiff", _tiff([_form_page()]), form=sent)

    assert response.status_code == 400
    assert response.get_json()["error"] == batch_routes.UNKNOWN_FORM
    assert scan_queue.open_page_ids(OWNER) == []
    assert not [path for path in queue_dir.rglob("meta.json")]


def test_a_page_queued_before_the_choice_existed_is_read_and_paired_as_two_sheets(
    client: FlaskClient, readers: Readers, queue_dir: Path
) -> None:
    [page_id] = _sent_as(client, ONE_SHEET)
    # As an older version wrote it: no form at all.
    path = queue_dir / page_id / "meta.json"
    written = json.loads(path.read_text())
    del written["form"]
    path.write_text(json.dumps(written))

    assert scan_queue.load_meta(page_id).form == TWO_SHEET
    assert ocr_worker.process_one() == ocr_worker.READ
    assert readers.forms == [TWO_SHEET]


def test_the_worker_tells_the_reader_the_form_each_page_was_sent_as(client: FlaskClient, readers: Readers) -> None:
    _sent_as(client, ONE_SHEET, pages=2, name="duplex.tif")
    _sent_as(client, TWO_SHEET, name="Scan 2.tiff")
    _sent_as(client, None, name="Scan 3.tiff")

    assert [ocr_worker.process_one() for _ in range(4)] == [ocr_worker.READ] * 4
    assert readers.forms == [ONE_SHEET, ONE_SHEET, TWO_SHEET, TWO_SHEET]


def test_the_drive_inbox_reads_every_page_as_the_two_sheet_form(
    monkeypatch: pytest.MonkeyPatch, read_as_production: list[str]
) -> None:
    """A one-sheet back in the inbox is read as a page with no Student ID, as before; a two-sheet Declaration page is known by its heading at the top left."""
    data = _tiff([Image.fromarray(_back()), Image.fromarray(_back(0.04))])
    monkeypatch.setattr(drive_inbox, "download", lambda _file_id: data)
    monkeypatch.setattr(drive_inbox, "file_link", lambda _file_id: "")
    monkeypatch.setattr(batch_store, "file_page_logged", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(batch_store, "file_pages_finished", lambda *_args, **_kwargs: None)
    read: list[ex.ExtractResult] = []

    def record(result: ex.ExtractResult, batch: dict[str, Any], _link: str, _tail: Any) -> tuple[str, int]:
        read.append(result)
        return "[unread]", int(batch["id"])

    monkeypatch.setattr(inbox_watcher, "_read_and_record", record)

    inbox_watcher._read_pages({"id": "file-1", "name": "Scan 9.tif"}, {"id": 1, "school": "", "location": ""}, 0)

    back, declaration = read
    assert not back.skipped and back.error.startswith("No Student ID found on this page")
    assert (declaration.skipped, declaration.skip_reason) == (True, "declaration")
    assert not back.declaration_side and not declaration.declaration_side
    assert "searched down the margin" not in read_as_production


# ================================================================== the two-sheet form, read as before
@pytest.mark.parametrize("at", [0.18, 0.30, 0.52])
@pytest.mark.parametrize("form", [TWO_SHEET, None], ids=["two_sheet", "no form"])
def test_a_declaration_page_low_in_the_frame_is_never_a_declaration_side_on_the_two_sheet_form(
    read_as_production: list[str], at: float, form: str | None
) -> None:
    """Photographed or scanned with its heading below the top strip: read for a Student ID, as before - never taken for a one-sheet back."""
    page = _bgr(_back(at))

    result = ex.extract_from_image(page, "Scan 2.jpg") if form is None else ex.extract_from_image(page, "Scan 2.jpg", form)

    assert not result.skipped and not result.declaration_side and result.orientation == 0
    assert result.error.startswith("No Student ID found on this page")
    assert read_as_production == ["top strip", "label"]


@pytest.mark.parametrize("at", [0.18, 0.52])
def test_the_same_page_on_the_one_sheet_form_is_a_declaration_side(at: float) -> None:
    """What the two-sheet form is kept from: the search down the margin claims it."""
    result = ex._extract(_bgr(_back(at)), source="Scan 2.jpg", form=ONE_SHEET)

    assert (result.skip_reason, result.declaration_side) == ("declaration", True)


def test_the_two_sheet_forms_declaration_page_is_known_by_its_heading_at_the_top_left_alone(
    read_as_production: list[str],
) -> None:
    upright = ex.extract_from_image(_bgr(_back(0.04)), "Scan 2.tiff", TWO_SHEET)
    turned = ex.extract_from_image(_bgr(cv2.rotate(_back(0.04), cv2.ROTATE_180)), "Scan 4.tiff", TWO_SHEET)

    assert (upright.skip_reason, upright.declaration_side, upright.orientation) == ("declaration", False, 0)
    # Fed upside down its heading is at the bottom right: not looked for turned round, it is read as before.
    assert not turned.skipped and (turned.declaration_side, turned.orientation) == (False, 0)
    assert read_as_production == ["top strip", "label"]


def _loose_fit_turned_round(monkeypatch: pytest.MonkeyPatch, upright_fit: float | None) -> None:
    """The layout fits loosely turned round (0.0062, wider than LAYOUT_TRUST_ERROR) and, as the page came, `upright_fit` or not at all."""

    def identify(image: np.ndarray) -> Any:
        mark = float(np.mean(image[: image.shape[0] // 10, image.shape[1] * 3 // 4 :] < 128)) > 0.05
        error = 0.0062 if mark else upright_fit
        return None if error is None else SimpleNamespace(error=error)

    assert 0.0062 > config.LAYOUT_TRUST_ERROR
    monkeypatch.setattr(ocr_worker, "identify", identify)


@pytest.mark.parametrize("upright_fit", [None, 0.0100], ids=["no fit as it came", "a looser fit as it came"])
@pytest.mark.parametrize("form", [TWO_SHEET, None], ids=["two_sheet", "no form"])
def test_an_upside_down_student_id_page_of_the_two_sheet_form_is_turned_upright_at_a_loose_fit(
    client: FlaskClient, readers: Readers, monkeypatch: pytest.MonkeyPatch, queue_dir: Path,
    upright_fit: float | None, form: str | None,
) -> None:
    """As before the one-sheet form was known: on real scans some of its upside-down Student ID pages fit only loosely turned round."""
    # Scanned upside down: its top-right mark is at the bottom left.
    response = _send(
        client, "Scan 1.tiff", _tiff([_form_page(mark="bottom-left")]), **({} if form is None else {"form": form})
    )
    [page_id] = [page["page_id"] for page in response.get_json()["pages"]]
    _loose_fit_turned_round(monkeypatch, upright_fit)

    ocr_worker.process_one()

    assert json.loads((queue_dir / page_id / "result.json").read_text())["orientation"] == 180
    stored = cv2.imread(str(queue_dir / page_id / "page.jpg"), cv2.IMREAD_GRAYSCALE)
    assert float(np.mean(stored[: stored.shape[0] // 10, stored.shape[1] * 3 // 4 :] < 128)) > 0.05
    assert readers.forms == [TWO_SHEET]


def test_the_one_sheet_form_still_keeps_such_a_page_as_it_came(
    client: FlaskClient, readers: Readers, monkeypatch: pytest.MonkeyPatch, queue_dir: Path
) -> None:
    response = _send(client, "Scan 1.tiff", _tiff([_form_page(mark="bottom-left")]), form=ONE_SHEET)
    [page_id] = [page["page_id"] for page in response.get_json()["pages"]]
    before = (queue_dir / page_id / "page.jpg").read_bytes()
    _loose_fit_turned_round(monkeypatch, None)

    ocr_worker.process_one()

    assert json.loads((queue_dir / page_id / "result.json").read_text())["orientation"] == 0
    assert (queue_dir / page_id / "page.jpg").read_bytes() == before


# ================================================================== the two-sheet form, paired as before
def test_a_two_sheet_file_pairs_by_scan_order_whatever_its_pages_were_read_as(
    google: FlaskClient, fake_db: Any, kids: Any
) -> None:
    """Read as Declaration sides - a page read before the choice, say - its Declaration pages still pair by scan order, never by place."""
    pages = _file(google, 4, form=TWO_SHEET)
    _declaration_side(pages[0])
    _id_side(pages[1], 0)
    _declaration_side(pages[2])
    _id_side(pages[3], 1)

    docs = _listed(google)
    unsorted = _unsorted(google)

    # By place, the one-sheet form, these would be two sheets fed Declaration side first.
    assert set(docs) == {_key(pages[1]), _key(pages[3])}
    assert _ids(docs[_key(pages[1])]) == [pages[1], pages[2]] and _sides(docs[_key(pages[1])]) == [False, False]
    assert _ids(docs[_key(pages[3])]) == [pages[3]]
    assert set(unsorted) == {pages[0]}
    assert unsorted[pages[0]]["message"] == "A Declaration page with no Student ID page right before it."
    assert not unsorted[pages[0]]["both_sides"]
    assert not any("declaration_side_first" in doc["attention"] for doc in docs.values())


def test_a_two_sheet_file_never_waits_for_the_rest_of_it(google: FlaskClient, fake_db: Any, kids: Any) -> None:
    """Its first form is Ready as soon as its two sheets are read - a one-sheet file's sheets wait for the whole file."""
    pages = _file(google, 4, form=TWO_SHEET)
    _id_side(pages[0], 0)
    _declaration_side(pages[1])

    doc = _listed(google)[_key(pages[0])]

    assert _ids(doc) == [pages[0], pages[1]] and _sides(doc) == [False, False]
    assert doc["status"] == "ready" and not doc["message"].startswith(REST)


def test_the_two_sheet_forms_numbered_images_are_no_scanner_session(
    google: FlaskClient, fake_db: Any, kids: Any
) -> None:
    """Named as a scanner numbers them, and read as a face-down session's would be: paired by scan order, at once."""
    pages = _numbered(google, TWO_SHEET, range(1, 5))
    _declaration_side(pages[0])
    _id_side(pages[1], 0)
    _declaration_side(pages[2])
    _id_side(pages[3], 1)

    docs = _listed(google)
    unsorted = _unsorted(google)

    assert set(docs) == {_key(pages[1]), _key(pages[3])}
    assert _ids(docs[_key(pages[1])]) == [pages[1], pages[2]] and _sides(docs[_key(pages[1])]) == [False, False]
    assert set(unsorted) == {pages[0]} and not unsorted[pages[0]]["both_sides"]
    assert not any(doc["message"].startswith(REST) for doc in docs.values())


# ================================================================== a run of both forms
def test_a_run_that_mixes_the_two_forms_pairs_each_page_by_its_own_form_and_waits_for_a_person(
    google: FlaskClient, fake_db: Any, kids: Any, drive: Any
) -> None:
    """A one-sheet stack, then a two-sheet one from the same scan page straight after: one run, its forms all checked by a person."""
    sheet = _file(google, 2, "Scan 1.tif", form=ONE_SHEET)
    _id_side(sheet[0], 0)
    _declaration_side(sheet[1])
    id_page = _queue(google, "Scan 2.tiff", form=TWO_SHEET)[0]
    declaration = _queue(google, "Scan 3.tiff", form=TWO_SHEET)[0]
    _id_side(id_page, 1)
    _skipped(declaration)
    _later(SETTLE + 1)

    docs = _listed(google)

    assert _ids(docs[_key(sheet[0])]) == [sheet[0], sheet[1]] and _sides(docs[_key(sheet[0])]) == [True, True]
    assert _ids(docs[_key(id_page)]) == [id_page, declaration] and _sides(docs[_key(id_page)]) == [False, False]
    assert all(doc["status"] == "ready" for doc in docs.values())
    for page_id in (sheet[0], id_page):
        _waits_for_a_person(google, page_id, MIXED)
    _refused_with_nothing_saved(_auto(google, 0, sheet[0]), fake_db, drive)
    _refused_with_nothing_saved(_auto(google, 1, id_page), fake_db, drive)


def test_two_sheet_images_numbered_on_from_a_one_sheet_session_are_not_in_it(
    google: FlaskClient, fake_db: Any, kids: Any, drive: Any
) -> None:
    """The scanner numbers on, the form changed: the session is its one-sheet images; the rest pair by scan order."""
    session = _numbered(google, ONE_SHEET, range(1, 3))
    _id_side(session[0], 0)
    _declaration_side(session[1])
    rest = _numbered(google, TWO_SHEET, range(3, 5))
    _id_side(rest[0], 1)
    # A Declaration page low in the frame, read as a side.
    _declaration_side(rest[1])
    _later(SETTLE + 1)

    docs = _listed(google)

    assert _ids(docs[_key(session[0])]) == [session[0], session[1]] and _sides(docs[_key(session[0])]) == [True, True]
    assert _ids(docs[_key(rest[0])]) == [rest[0], rest[1]] and _sides(docs[_key(rest[0])]) == [False, False]
    for page_id in (session[0], rest[0]):
        _waits_for_a_person(google, page_id, MIXED)
    _refused_with_nothing_saved(_auto(google, 1, rest[0]), fake_db, drive)


def test_a_stack_of_the_two_sheet_form_alone_still_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, kids: Any, drive: Any
) -> None:
    """Nothing mixed in: Student ID page, Declaration page, by turns, as before."""
    first, first_declaration = (_queue(google, name, form=TWO_SHEET)[0] for name in ("Scan 1.tiff", "Scan 2.tiff"))
    second, second_declaration = (_queue(google, name, form=TWO_SHEET)[0] for name in ("Scan 3.tiff", "Scan 4.tiff"))
    _id_side(first, 0)
    _skipped(first_declaration)
    _id_side(second, 1)
    _skipped(second_declaration)
    _later(SETTLE + 1)

    docs = _listed(google)

    assert docs[_key(first)]["auto_upload"] is True, docs[_key(first)]["auto_upload_refusal"]
    assert docs[_key(second)]["auto_upload"] is True, docs[_key(second)]["auto_upload_refusal"]


# ================================================================== batch codes on
def test_with_batch_codes_on_each_page_is_still_read_by_its_own_form(
    client: FlaskClient, readers: Readers, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    readers.codes = ["B1-1", "B1-1", "B1-2"]
    _sent_as(client, ONE_SHEET, pages=2, name="duplex.tif")
    _sent_as(client, TWO_SHEET, name="Scan 2.tiff")

    assert [ocr_worker.process_one() for _ in range(3)] == [ocr_worker.READ] * 3
    assert readers.forms == [ONE_SHEET, ONE_SHEET, TWO_SHEET]


def test_with_batch_codes_on_pages_of_both_forms_group_by_their_codes(
    google: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-sheet sheet fed the other way up is its parts in scan order, its back first - nothing pairs by place, whatever its form."""
    monkeypatch.setattr(config, "BATCH_CODES", True)
    sheet = _sent_as(google, ONE_SHEET, pages=2, name="duplex.tif")
    pair = _sent_as(google, TWO_SHEET, name="Scan 2.tiff") + _sent_as(google, TWO_SHEET, name="Scan 3.tiff")
    _skipped(sheet[0], code="B1-1", declaration_side=True)
    _read(sheet[1], "B1-1", KIDS[0]["student_id"], dict(KIDS[0], matched_via="exact"))
    _read(pair[0], "B1-2", KIDS[1]["student_id"], dict(KIDS[1], matched_via="exact"))
    _skipped(pair[1], code="B1-2")

    [batch] = _batches(google)["batches"]
    docs = {doc["key"]: doc for doc in batch["documents"]}

    assert set(docs) == {"B1-1", "B1-2"}
    assert _ids(docs["B1-1"]) == [sheet[0], sheet[1]] and _ids(docs["B1-2"]) == [pair[0], pair[1]]
    assert not any("both_sides" in page for doc in docs.values() for page in doc["pages"])


# ================================================================== the page
def test_the_page_script_parses(client: FlaskClient, tmp_path: Path) -> None:
    """node --check of the scan page's own script, as rendered and with its Jinja stripped from the template."""
    if NODE is None:
        pytest.skip("Node is needed to check the page script")
    template = (Path(config.BASE_DIR) / "client" / "templates" / "index.html").read_text(encoding="utf-8")
    raw = next(s for s in re.findall(r"<script>(.*?)</script>", template, re.DOTALL) if "const API_BASE" in s)
    stripped = re.sub(r"\{%.*?%\}", "", re.sub(r"\{\{.*?\}\}", "0", raw))
    rendered = next(
        s for s in re.findall(r"<script>(.*?)</script>", client.get("/").get_data(as_text=True), re.DOTALL)
        if "const API_BASE" in s
    )
    for name, script in (("stripped.js", stripped), ("rendered.js", rendered)):
        (tmp_path / name).write_text(script, encoding="utf-8")
        checked = subprocess.run([NODE, "--check", str(tmp_path / name)], capture_output=True, text=True, check=False)
        assert checked.returncode == 0, checked.stderr


def test_the_page_offers_both_forms_in_its_own_words(client: FlaskClient) -> None:
    html = " ".join(client.get("/").get_data(as_text=True).split())

    assert re.search(
        r'<input type="radio" name="paper-form" id="form-two-sheet" value="two_sheet" /> '
        r"<span>Two sheets: a Student ID page and a separate Declaration page</span>",
        html,
    )
    assert re.search(
        r'<input type="radio" name="paper-form" id="form-one-sheet" value="one_sheet" /> '
        r"<span>One sheet printed on both sides: Student ID on the front, Declaration on the back</span>",
        html,
    )
    # Neither is picked by the page itself.
    assert 'value="two_sheet" checked' not in html and 'value="one_sheet" checked' not in html
    assert "Pick the form being scanned first - nothing can be scanned or added until you do." in html
    # Beside the school picker, before the scanner.
    assert html.index('id="school-select"') < html.index('id="form-choice"') < html.index('id="scan-folder-card"')


@needs_node
def test_page_locks_scanning_and_adding_files_until_the_form_is_picked(client: FlaskClient, tmp_path: Path) -> None:
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      let pickerOpened = 0;
      const page = loadPage({
        clock, form: null, indexedDB: makeIndexedDB(),
        local: { "ocr.school": "School A", "ocr.location": "Pune" },
        showDirectoryPicker: async () => { pickerOpened += 1; return folder; },
      });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      await settle();
      const b = page.byId;
      let chooserOpened = 0;
      b["file-input"].addEventListener("click", () => { chooserOpened += 1; });
      const state = () => ({
        needed: !b["form-choice-needed"].hidden, hint: !b["form-choice-hint"].hidden,
        checked: [b["form-two-sheet"].checked, b["form-one-sheet"].checked],
        fileInput: b["file-input"].disabled, tiles: b["upload-tiles"].classList.contains("is-locked"),
        zone: b["drop-zone"].getAttribute("aria-disabled"),
        connect: b["scan-folder-connect"].disabled, steps: b["scan-folder-steps"].innerHTML,
      });
      const locked = state();
      b["drop-zone"].fire("click");
      b["drop-zone"].fire("keydown", { key: "Enter" });
      b["drop-zone"].fire("drop", { dataTransfer: { types: ["Files"], files: [{ name: "dropped.jpg" }] } });
      const dropped = b["drop-message"].textContent;
      await uploadFile(page, "chosen.jpg");
      b["scan-folder-connect"].fire("click");
      await clock.advance(6000);
      const whileLocked = { chooserOpened, pickerOpened, sends: everyScan(page).length, dropped, looks: folder.listings };

      pickForm(page, "one_sheet");
      const unlocked = state();
      await uploadFile(page, "chosen.jpg");
      const sent = sentForm(await answerScan(page, ["p1"]));
      b["drop-zone"].fire("click");
      return { locked, whileLocked, unlocked, sent, chooserOpened, kept: page.storage.getItem("ocr.form") };
    """)

    locked, unlocked = out["locked"], out["unlocked"]
    assert locked["needed"] is True and locked["hint"] is False and locked["checked"] == [False, False]
    assert locked["fileInput"] is True and locked["tiles"] is True and locked["zone"] == "true"
    assert locked["connect"] is True
    assert "Pick the form being scanned" in locked["steps"] and "<ol" not in locked["steps"]
    # Nothing opened, nothing read from a folder, nothing sent.
    assert out["whileLocked"] == {
        "chooserOpened": 0, "pickerOpened": 0, "sends": 0,
        "dropped": "Not added - pick the form being scanned first.", "looks": 0,
    }
    assert unlocked["needed"] is False and unlocked["hint"] is True and unlocked["checked"] == [False, True]
    assert unlocked["fileInput"] is False and unlocked["tiles"] is False and unlocked["zone"] == "false"
    assert unlocked["connect"] is False and "<ol" in unlocked["steps"]
    assert out["sent"]["form"] == "one_sheet" and out["sent"]["file"] == "chosen.jpg"
    assert out["chooserOpened"] == 1
    assert out["kept"] == "one_sheet"


@needs_node
def test_page_remembers_the_form_on_this_computer_and_works_when_the_browser_keeps_nothing(
    client: FlaskClient, tmp_path: Path
) -> None:
    out = _run_page(client, tmp_path, r"""
      const kept = new Map([["ocr.school", "School A"], ["ocr.location", "Pune"]]);
      const KEPT = {};
      const BROKEN = {};
      const realStorage = makeStorage;
      makeStorage = (seed) => {
        if (seed === KEPT) return { getItem: (k) => (kept.has(k) ? kept.get(k) : null), setItem: (k, v) => kept.set(k, String(v)), removeItem: (k) => kept.delete(k) };
        if (seed === BROKEN) {
          const refuse = () => { throw new Error("SecurityError: storage is disabled"); };
          return { getItem: refuse, setItem: refuse, removeItem: refuse };
        }
        return realStorage(seed);
      };
      const visit = async (local) => {
        const page = loadPage({ local, form: null });
        await answer(page.calls, "/api/schools", { schools: [{ school: "School A", location: "Pune" }] });
        await answer(page.calls, "/api/inbox/status", { configured: false });
        return page;
      };
      const send = async (page, name, id) => {
        await uploadFile(page, name);
        const call = await answerScan(page, [id]);
        return sentForm(call).form;
      };

      const first = await visit(KEPT);
      const firstLocked = first.byId["file-input"].disabled;
      pickForm(first, "one_sheet");
      const firstSent = await send(first, "a.jpg", "p1");

      const second = await visit(KEPT);
      const secondChecked = [second.byId["form-two-sheet"].checked, second.byId["form-one-sheet"].checked];
      const secondLocked = second.byId["file-input"].disabled;
      const secondSent = await send(second, "b.jpg", "p2");
      pickForm(second, "two_sheet");
      const changedSent = await send(second, "c.jpg", "p3");

      const third = await visit(BROKEN);
      const brokenLocked = third.byId["file-input"].disabled;
      pickForm(third, "one_sheet");
      const brokenSent = await send(third, "d.jpg", "p4");
      makeStorage = realStorage;
      return { firstLocked, firstSent, secondChecked, secondLocked, secondSent, changedSent,
               remembered: kept.get("ocr.form"), brokenLocked, brokenSent };
    """)

    assert out["firstLocked"] is True and out["firstSent"] == "one_sheet"
    assert out["secondChecked"] == [False, True] and out["secondLocked"] is False and out["secondSent"] == "one_sheet"
    assert out["changedSent"] == "two_sheet" and out["remembered"] == "two_sheet"
    # Nothing kept: asked again, and the page works on this visit.
    assert out["brokenLocked"] is True and out["brokenSent"] == "one_sheet"


@needs_node
def test_page_a_new_form_applies_to_the_scans_that_come_in_after_it(client: FlaskClient, tmp_path: Path) -> None:
    """As for a new school (#17): a scan the folder showed before the change goes as the form it was seen under."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      await clock.advance(2000);
      pickForm(page, "one_sheet");
      folder.add("Scan 002.jpg");
      await clock.advance(2000);
      await clock.advance(2000);
      const sends = [];
      sends.push(sentForm(await answerScan(page, ["p1"])));
      sends.push(sentForm(await answerScan(page, ["p2"])));
      await uploadFile(page, "added.jpg");
      sends.push(sentForm(await answerScan(page, ["p3"])));
      return { sends: sends.map((s) => [s.file, s.form]), state: page.byId["scan-folder-state"].textContent };
    """)

    assert out["sends"] == [["Scan 001.jpg", "two_sheet"], ["Scan 002.jpg", "one_sheet"], ["added.jpg", "one_sheet"]]
    assert "Watching" in out["state"]


@needs_node
def test_page_a_scan_folder_kept_from_before_waits_for_the_form(client: FlaskClient, tmp_path: Path) -> None:
    """Connected on a visit before the choice existed: not read at all until the form is picked, then read and sent with it."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      const idb = makeIndexedDB();
      idb.dbs.set("ocr-scan-folder", new Map([["handles", new Map([["watched", folder]])], ["sent", new Map()]]));
      folder.add("Scan 001.jpg");
      const page = loadPage({
        clock, form: null, indexedDB: idb,
        local: { "ocr.school": "School A", "ocr.location": "Pune" },
        showDirectoryPicker: async () => folder,
      });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      await clock.advance(10000);
      const waiting = { looks: folder.listings, sends: everyScan(page).length, state: page.byId["scan-folder-state"].textContent,
                        connect: page.byId["scan-folder-connect"].disabled, toggle: page.byId["scan-folder-toggle"].disabled };
      pickForm(page, "two_sheet");
      await clock.advance(6000);
      const sent = sentForm(await answerScan(page, ["p1"]));
      return { waiting, looked: folder.listings > 0, sent: [sent.file, sent.form, sent.school],
               state: page.byId["scan-folder-state"].textContent };
    """)

    assert out["waiting"]["looks"] == 0 and out["waiting"]["sends"] == 0
    assert out["waiting"]["state"] == '"OCR Scans" · Not watching yet - pick the form being scanned above'
    assert out["waiting"]["connect"] is True and out["waiting"]["toggle"] is True
    assert out["looked"] is True and out["sent"] == ["Scan 001.jpg", "two_sheet", "School A"]
    assert "Watching" in out["state"]


@needs_node
def test_page_setup_steps_and_help_follow_the_form(client: FlaskClient, tmp_path: Path) -> None:
    """Both sides (duplex), skip blank pages off, for the one-sheet form only; one side (simplex) for the two-sheet form."""
    out = _run_page(client, tmp_path, r"""
      const WINDOWS = { userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", platform: "Win32", maxTouchPoints: 0 };
      const HELP = ["how-sides-two", "how-sides-one", "ocr-note-sides", "auto-upload-sides", "inbox-form-note"];
      const look = (page) => ({
        shown: HELP.filter((id) => !page.byId[id].hidden),
        steps: page.byId["scan-folder-steps"].innerHTML,
      });
      const open = (form) => loadPage({ form, navigator: WINDOWS, showDirectoryPicker: async () => makeFolder(), indexedDB: makeIndexedDB() });
      const changing = open("two_sheet");
      const before = look(changing);
      pickForm(changing, "one_sheet");
      const after = look(changing);
      return { none: look(open(null)), two: look(open("two_sheet")), one: look(open("one_sheet")), before, after };
    """)

    assert out["none"]["shown"] == [] and "Pick the form being scanned" in out["none"]["steps"]
    assert out["two"]["shown"] == ["how-sides-two"]
    assert out["one"]["shown"] == ["how-sides-one", "ocr-note-sides", "auto-upload-sides", "inbox-form-note"]
    two, one = out["two"]["steps"], out["one"]["steps"]
    assert "one side (<strong>simplex</strong>)" in two and "file type <strong>JPEG</strong> (or PDF)" in two
    assert "duplex" not in two and "skip blank pages" not in two and "both sides" not in two
    assert "Load the forms with page 1 of each face down" in two
    assert "(duplex)" in one and "<strong>skip blank pages</strong> off" in one and "simplex" not in one
    assert "Load the forms with the Student ID side of each face down" in one
    assert out["before"] == out["two"] and out["after"] == out["one"]


@needs_node
def test_page_with_batch_codes_on_still_asks_for_the_form_and_sends_it(
    client: FlaskClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    out = _run_page(client, tmp_path, r"""
      const page = loadPage({ form: null, local: { "ocr.school": "School A", "ocr.location": "Pune" } });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      await uploadFile(page, "early.jpg");
      const early = everyScan(page).length;
      pickForm(page, "one_sheet");
      await uploadFile(page, "stack.pdf");
      const sent = sentForm(await answerScan(page, ["p1"]));
      return { early, sent, steps: page.byId["scan-folder-steps"].innerHTML };
    """)

    assert out["early"] == 0
    assert out["sent"]["form"] == "one_sheet" and "sender" not in out["sent"]
    assert "batch code (B1-1, B1-2 ...)" in out["steps"] and "(duplex)" in out["steps"]
