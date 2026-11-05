"""A scanner session's numbered images, one a side: taken as one file and paired sheet by sheet by their numbers.

A scanner that saves a scan one image a side numbers the images of one scan
in turn - 20260101120000_0001.jpg, 20260101120000_0002.jpg ... or Scan 1.jpeg,
Scan 2.jpeg ... With batch codes off, as the app ships:

  - documents.scan_session names a file's session - its name without the
    number, compared without case - and its number there; a name ending in
    no number is in no session.
  - The images of one session that one scan page sent in one run - from its
    scan folder, or apart from those added by hand there - are one file:
    from the lowest number, the first two are one sheet, the next two the
    next - in the order of their numbers, not of their coming - each sheet
    its Student ID side and its Declaration side in either order, the
    Student ID side first in the document and the PDF. A sheet whose
    Declaration side came first needs a person's look, and keeps its stack
    from going up by itself; so do images that came in out of the order of
    their numbers.
  - More images may join a session until its run has finished, so its
    sheets wait until then - nothing more from that scan page for
    STACK_SETTLE_MINUTES, nothing of it still being stored, every page read
    - with no Confirm, and a confirm is refused with nothing saved. A scan
    another scan page is storing for the same school never keeps it waiting.
  - A session that breaks the rule - a gap or a repeat in its numbers, an
    odd number of images - is paired by scan order as before, and its stack
    waits for a person; a session of one image, or with no Declaration side
    (the two-sheet form saved as numbered images), and images whose names
    end in no number, pair by scan order as before.
  - Automatic upload: a stack of sessions and files scanned both sides, every
    sheet Student ID side first, goes up by itself; one in among other scans
    waits for a person. With batch codes on nothing changes.

Every page here is drawn and read by hand, as the worker would read it. The
database is the FakeDB, Drive the FakeDrive; nothing leaves the test.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient

from server import config
from server.services import documents, ocr_worker, scan_queue
from tests import test_auto_upload_stack as stack_tests
from tests import test_batch_queue as queue_tests
from tests.test_app import _run_page, needs_node
from tests.test_auto_upload import _key, _refused_with_nothing_saved
from tests.test_auto_upload_stack import (
    BY_HAND,
    KIDS,
    SETTLE,
    _auto,
    _filed,
    _image,
    _listed,
    _waits_for_a_person,
)
from tests.test_batch_codes_off import _OFF_RESULTS_JS, ONE, TWO, _later, _skipped
from tests.test_batch_queue import (
    OWNER,
    SCHOOL,
    FakeDrive,
    Students,
    _batches,
    _confirm,
    _form_page,
    _jpeg,
    _read,
    _send,
    _tiff,
    _writes,
)
from tests.test_both_sides import _declaration_side, _id_side, _ids, _sides

# The queue tests' fixtures, here by name: the queue in the test's own folder
# (autouse), the fake database's students - every child of KIDS in it - the
# fake Drive and a Google sign-in.
queue_dir = queue_tests.queue_dir
students = queue_tests.students
kids = stack_tests.kids
drive = queue_tests.drive
google = queue_tests.google

# What a sheet of a session says while the session may still grow.
REST_OF_SCAN = (
    "Waiting for the rest of its scan - images saved one a side are paired sheet by sheet once no more have"
    f" come for {config.STACK_SETTLE_MINUTES} minutes."
)
# Why no form of a stack goes up by itself, for a session as for a file.
SWAPPED = "its stack has a sheet scanned Declaration side first"
UNPAIRED = (
    "its stack has a file scanned both sides whose sheets are not each a Student ID side and a Declaration side"
)
MIXED = "its stack has a file scanned both sides in among other scans"

# Every page drawn a little wider than the one before, so each page.jpg - and
# so each page of a PDF - is told apart.
_WIDTHS = itertools.count(600, 4)


@pytest.fixture(autouse=True)
def batch_codes(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Batch codes off, as the app ships (overrides tests/conftest.py's fixture)."""
    monkeypatch.setattr(config, "BATCH_CODES", False)
    return False


@pytest.fixture(autouse=True)
def _no_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pages are read by hand here - never by a reader thread of the app's."""
    monkeypatch.setattr(ocr_worker, "should_run_in_process", lambda _debug: False)


# ------------------------------------------------------------------ helpers
def _sent(client: FlaskClient, name: str, sender: str = ONE, **form: str) -> str:
    """One file of one image, as the scan page sends it with the switch on - a JPEG, or for .tif a TIFF."""
    page = _form_page((next(_WIDTHS), 874))
    data = _tiff([page]) if name.lower().endswith((".tif", ".tiff")) else _jpeg(page)
    response = _send(client, name, data, auto_upload="1", sender=sender, **form)
    assert response.status_code == 202, response.get_json()
    [sent] = response.get_json()["pages"]
    return str(sent["page_id"])


def _scan(client: FlaskClient, names: list[str], sides: str, sender: str = ONE, first_kid: int = 0) -> list[str]:
    """Send each image in turn, then read it as `sides` says - "f" a Student ID side, KIDS in turn, "b" a Declaration side."""
    pages = [_sent(client, name, sender) for name in names]
    kid = itertools.count(first_kid)
    for page_id, side in zip(pages, sides, strict=True):
        if side == "f":
            _id_side(page_id, next(kid))
        else:
            _declaration_side(page_id)
    return pages


def _numbered(count: int, stem: str = "20260101120000_{:04d}.jpg", first: int = 1) -> list[str]:
    """A session's names, as a scanner numbers them (a timestamp, then a running number) unless `stem` says otherwise."""
    return [stem.format(n) for n in range(first, first + count)]


def _unsorted(client: FlaskClient) -> list[str]:
    return [page["page_id"] for page in _batches(client)["unsorted"]]


def _held(doc: dict[str, Any]) -> list[str]:
    return [code for code in doc["attention"] if code in ("orphan_in_stack", "gap_in_stack", "after_held_stack")]


# ================================================================== the name
@pytest.mark.parametrize(
    ("name", "session"),
    [
        ("20260101120000_0001.jpg", ("20260101120000.jpg", 1)),
        ("20260101120000_0022.jpg", ("20260101120000.jpg", 22)),
        ("Scan 2.jpeg", ("scan.jpeg", 2)),
        ("SCAN 12.JPEG", ("scan.jpeg", 12)),
        ("IMG-0042.JPG", ("img.jpg", 42)),
        ("page (3).png", ("page.png", 3)),
        ("page(3).png", ("page.png", 3)),
        ("Scan_999999.tif", ("scan.tif", 999999)),
        ("Consent forms 7A 3.jpg", ("consent forms 7a.jpg", 3)),
        ("Scan 4", ("scan", 4)),
    ],
)
def test_a_name_ending_in_a_running_number_names_its_session_and_its_number(
    name: str, session: tuple[str, int]
) -> None:
    assert documents.scan_session(name) == session


@pytest.mark.parametrize(
    "name",
    [
        "20260101115900.jpg",
        "scan.jpg",
        "Scan.jpeg",
        "_0001.jpg",
        "0001.jpg",
        "Scan 1234567.jpg",
        "Scan 2a.jpeg",
        "Scan 2 copy.jpeg",
        "",
    ],
)
def test_a_name_ending_in_no_running_number_is_in_no_session(name: str) -> None:
    assert documents.scan_session(name) is None


def test_the_images_of_one_session_share_their_key_and_another_extension_is_another_session() -> None:
    one, two = documents.scan_session("Scan 1.JPEG"), documents.scan_session("scan 2.jpeg")
    assert one is not None and two is not None and one[0] == two[0]
    assert documents.scan_session("Scan 3.jpg") != ("scan.jpeg", 3)


# ================================================================== pairing
@pytest.mark.parametrize(
    "stem",
    ["20260101120000_{:04d}.jpg", "Scan {}.jpeg"],
    ids=["numbered after a timestamp", "numbered as Image Capture does"],
)
def test_a_session_pairs_sheet_by_sheet_once_its_scan_has_finished(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, stem: str
) -> None:
    pages = _scan(google, _numbered(6, stem), "fbfbfb")
    sheets = [(pages[0], pages[1]), (pages[2], pages[3]), (pages[4], pages[5])]

    waiting = _listed(google)
    refused = _confirm(google, _key(pages[0]), KIDS[0]["student_id"], [pages[0], pages[1]])

    assert set(waiting) == {_key(first) for first, _back in sheets}
    for first, back in sheets:
        doc = waiting[_key(first)]
        assert _ids(doc) == [first, back] and _sides(doc) == [True, True]
        assert doc["status"] == "waiting" and doc["can_confirm"] is False and doc["message"] == REST_OF_SCAN
    assert refused.status_code == 409, refused.get_json()
    assert refused.get_json()["error"] == REST_OF_SCAN + " Try again in a moment."
    assert _writes(fake_db) == [] and drive.calls == []

    _later(SETTLE + 1)
    docs = _listed(google)

    for first, back in sheets:
        doc = docs[_key(first)]
        assert _ids(doc) == [first, back] and _sides(doc) == [True, True]
        assert doc["status"] == "ready" and doc["attention"] == [] and doc["can_confirm"] is True
    assert _unsorted(google) == []
    response = _confirm(google, _key(pages[2]), KIDS[1]["student_id"])
    assert response.status_code == 200, response.get_json()
    assert _filed(drive, 1) == [_image(pages[2]), _image(pages[3])]


def test_a_sheet_of_a_session_fed_face_down_pairs_with_its_own_other_side(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """By scan order its back would join nothing and hold the stack, its front pair with nothing; by number it is one form."""
    pages = _scan(google, _numbered(6), "fbbffb")
    _later(SETTLE + 1)

    docs = _listed(google)

    assert set(docs) == {_key(pages[0]), _key(pages[3]), _key(pages[4])}
    assert _ids(docs[_key(pages[0])]) == [pages[0], pages[1]]
    assert _ids(docs[_key(pages[3])]) == [pages[3], pages[2]] and _sides(docs[_key(pages[3])]) == [True, True]
    assert _ids(docs[_key(pages[4])]) == [pages[4], pages[5]]
    # The sheet fed the other way up needs a person's look; nothing is held.
    assert docs[_key(pages[3])]["status"] == "needs_attention"
    assert docs[_key(pages[3])]["attention"] == ["declaration_side_first"]
    for at in (0, 4):
        assert docs[_key(pages[at])]["status"] == "ready" and docs[_key(pages[at])]["attention"] == []
    assert _unsorted(google) == []
    # No form of its stack goes up by itself.
    _waits_for_a_person(google, pages[0], SWAPPED)
    _waits_for_a_person(google, pages[3], "its sheet was scanned Declaration side first")
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)

    response = _confirm(google, _key(pages[3]), KIDS[1]["student_id"])
    assert response.status_code == 200, response.get_json()
    assert _filed(drive, 1) == [_image(pages[3]), _image(pages[2])]


def test_a_session_fed_face_down_has_every_sheet_checked_by_a_person(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """The scanner numbers each back before its front: every sheet still pairs with its own other side."""
    pages = _scan(google, _numbered(6, "Scan {}.jpeg"), "bfbfbf")
    _later(SETTLE + 1)

    docs = _listed(google)

    assert set(docs) == {_key(pages[1]), _key(pages[3]), _key(pages[5])}
    for at in (1, 3, 5):
        doc = docs[_key(pages[at])]
        assert _ids(doc) == [pages[at], pages[at - 1]] and _sides(doc) == [True, True]
        assert doc["status"] == "needs_attention" and doc["attention"] == ["declaration_side_first"]
        assert doc["can_confirm"] is True and doc["auto_upload"] is False
    assert _unsorted(google) == []


def test_a_session_a_side_out_of_step_has_every_sheet_checked_by_a_person(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """Its first image a back whose front never came: by number every sheet looks fed the other way up.

    Each is then paired with the sheet before's back, and only the paper
    shows which - so each needs a person's look, as in a file a side out of
    step.
    """
    pages = _scan(google, _numbered(4), "bfbf")
    _later(SETTLE + 1)

    docs = _listed(google)

    assert set(docs) == {_key(pages[1]), _key(pages[3])}
    for at in (1, 3):
        doc = docs[_key(pages[at])]
        assert _ids(doc) == [pages[at], pages[at - 1]]
        assert doc["status"] == "needs_attention" and doc["attention"] == ["declaration_side_first"]
        assert doc["auto_upload"] is False and doc["auto_upload_refusal"] == "its sheet was scanned Declaration side first"


def test_a_session_with_a_gap_in_its_numbers_pairs_by_scan_order_once_its_scan_has_finished(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """Images 3 and 4 never came: until the scan has finished they might; then it is paired by scan order."""
    pages = _scan(google, _numbered(2) + _numbered(2, first=5), "fbfb")

    so_far = _listed(google)

    for at in (0, 2):
        assert so_far[_key(pages[at])]["status"] == "waiting" and so_far[_key(pages[at])]["message"] == REST_OF_SCAN

    _later(SETTLE + 1)
    docs = _listed(google)

    assert set(docs) == {_key(pages[0]), _key(pages[2])}
    for at in (0, 2):
        assert _ids(docs[_key(pages[at])]) == [pages[at], pages[at + 1]] and _sides(docs[_key(pages[at])]) == [
            False,
            False,
        ]
    _waits_for_a_person(google, pages[0], UNPAIRED)
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)


def test_a_session_with_a_number_twice_pairs_by_scan_order_at_once(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """Image 2 sent again: a number that comes twice never pairs into sheets, so the session is paired by scan order straight away."""
    names = _numbered(2) + _numbered(1, first=2) + _numbered(2, first=3)
    pages = _scan(google, names, "fbbfb")

    docs = _listed(google)
    unsorted = _unsorted(google)

    assert set(docs) == {_key(pages[0]), _key(pages[3])}
    assert _ids(docs[_key(pages[0])]) == [pages[0], pages[1]] and _sides(docs[_key(pages[0])]) == [False, False]
    assert _ids(docs[_key(pages[3])]) == [pages[3], pages[4]]
    # The second copy joins nothing and holds the stack, as by scan order it always did.
    assert unsorted == [pages[2]]
    assert all(_held(doc) == ["orphan_in_stack"] for doc in docs.values())
    assert all(doc["status"] != "waiting" for doc in docs.values())


def test_a_session_of_an_odd_number_of_images_pairs_by_scan_order_once_its_scan_has_finished(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """While the scan may go on, its last image waits for its other side, listed nowhere yet."""
    pages = _scan(google, _numbered(5), "fbfbf")

    so_far = _listed(google)

    assert set(so_far) == {_key(pages[0]), _key(pages[2])}
    assert all(doc["status"] == "waiting" and doc["message"] == REST_OF_SCAN for doc in so_far.values())
    assert _unsorted(google) == []

    _later(SETTLE + 1)
    docs = _listed(google)

    assert set(docs) == {_key(pages[0]), _key(pages[2]), _key(pages[4])}
    assert _ids(docs[_key(pages[0])]) == [pages[0], pages[1]] and _sides(docs[_key(pages[0])]) == [False, False]
    assert _ids(docs[_key(pages[4])]) == [pages[4]] and "no_declaration" in docs[_key(pages[4])]["attention"]
    _waits_for_a_person(google, pages[0], UNPAIRED)
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)


def test_two_sessions_in_one_run_each_pair_sheet_by_sheet(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """The second scan's first sheet fed the other way up: by scan order its back would follow the first scan's last back and hold the stack."""
    first = _scan(google, _numbered(4), "fbfb")
    second = _scan(google, _numbered(2, "20260101121000_{:04d}.jpg"), "bf", first_kid=2)
    _later(SETTLE + 1)

    docs = _listed(google)

    assert set(docs) == {_key(first[0]), _key(first[2]), _key(second[1])}
    assert _ids(docs[_key(first[0])]) == [first[0], first[1]] and _ids(docs[_key(first[2])]) == [first[2], first[3]]
    assert _ids(docs[_key(second[1])]) == [second[1], second[0]]
    assert docs[_key(second[1])]["attention"] == ["declaration_side_first"]
    assert not any(_held(doc) for doc in docs.values()) and _unsorted(google) == []
    _waits_for_a_person(google, first[0], SWAPPED)


def test_sessions_from_two_scan_pages_and_files_added_by_hand_never_mix(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """The same names from two computers, sent turn about, and again added by hand in the scan page's own tab.

    Taken together their numbers would come twice; each scan page's - and
    the files added by hand - are a session of their own.
    """
    here = [_sent(google, "Scan 1.jpeg", ONE)]
    there = [_sent(google, "Scan 1.jpeg", TWO)]
    here.append(_sent(google, "Scan 2.jpeg", ONE))
    there.append(_sent(google, "Scan 2.jpeg", TWO))
    by_hand = [_sent(google, "Scan 1.jpeg", BY_HAND), _sent(google, "Scan 2.jpeg", BY_HAND)]
    for n, (front, back) in enumerate((here, there, by_hand)):
        _id_side(front, n)
        _declaration_side(back)
    _later(SETTLE + 1)

    docs = _listed(google)

    assert set(docs) == {_key(here[0]), _key(there[0]), _key(by_hand[0])}
    for front, back in (here, there, by_hand):
        assert _ids(docs[_key(front)]) == [front, back] and _sides(docs[_key(front)]) == [True, True]
        assert docs[_key(front)]["status"] == "ready"
    _waits_for_a_person(
        google, here[0], "its stack has files added by hand as well as scans from the scan folder"
    )
    _waits_for_a_person(
        google, there[0], "another computer or tab sent pages for the same school around the same time"
    )


@pytest.mark.parametrize(
    ("came", "why"),
    [
        ([1, 0, 3, 2], "its stack does not begin with a Student ID page"),
        ([0, 2, 1, 3], "a page came between its Student ID page and its Declaration page"),
    ],
    ids=["each back before its front", "the second sheet's front before the first sheet's back"],
)
def test_images_that_came_out_of_the_order_of_their_numbers_still_pair_by_their_numbers(
    google: FlaskClient, fake_db: Any, kids: Students, came: list[int], why: str
) -> None:
    """Each sheet Student ID side first by its numbers, whatever order they came in - but a person confirms them."""
    names = _numbered(4, "Scan {}.jpeg")
    pages: dict[int, str] = {at: _sent(google, names[at]) for at in came}
    for at in came:
        if at % 2:
            _declaration_side(pages[at])
        else:
            _id_side(pages[at], at // 2)
    _later(SETTLE + 1)

    docs = _listed(google)

    assert set(docs) == {_key(pages[0]), _key(pages[2])}
    for at in (0, 2):
        doc = docs[_key(pages[at])]
        assert _ids(doc) == [pages[at], pages[at + 1]] and _sides(doc) == [True, True]
        assert doc["status"] == "ready" and doc["attention"] == []
    assert _unsorted(google) == []
    # They came in out of turn, so the stack is not clean as it came.
    assert docs[_key(pages[0])]["auto_upload"] is False
    assert docs[_key(pages[0])]["auto_upload_refusal"] == why


def test_a_session_that_may_still_grow_waits_until_its_scan_has_finished(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """Its run finishes once nothing has come from its scan page for the settle time, nothing is being stored, every page read."""
    pages = _scan(google, _numbered(4), "fbfb")
    _later(SETTLE - 5)

    nearly = _listed(google)

    assert all(doc["status"] == "waiting" and doc["message"] == REST_OF_SCAN for doc in nearly.values())

    # More of the same scan: the session grows, and waits again from its new last image.
    more = [_sent(google, name) for name in _numbered(2, first=5)]
    _id_side(more[0], 2)
    _later(SETTLE + 1)

    reading = _listed(google)

    # Its last image still to be read: still waiting, its other side not shown yet.
    assert set(reading) == {_key(pages[0]), _key(pages[2])}
    assert all(doc["status"] == "waiting" for doc in reading.values())

    _declaration_side(more[1])
    with scan_queue._being_stored(OWNER, scan_queue.Storing(ONE, SCHOOL, "Mumbai")):
        storing = _listed(google)
        refused = _confirm(google, _key(pages[0]), KIDS[0]["student_id"], [pages[0], pages[1]])
    done = _listed(google)

    assert set(storing) == {_key(pages[0]), _key(pages[2]), _key(more[0])}
    assert all(doc["status"] == "waiting" and doc["message"] == REST_OF_SCAN for doc in storing.values())
    assert refused.status_code == 409 and refused.get_json()["error"] == REST_OF_SCAN + " Try again in a moment."
    assert _writes(fake_db) == [] and drive.calls == []
    for first, back in ((pages[0], pages[1]), (pages[2], pages[3]), (more[0], more[1])):
        assert _ids(done[_key(first)]) == [first, back] and done[_key(first)]["status"] == "ready"
        assert done[_key(first)]["auto_upload"] is True, done[_key(first)]["auto_upload_refusal"]


def test_a_scan_another_scan_page_is_storing_for_its_school_never_keeps_a_session_waiting(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """It may be beside the stack, so the stack does not go up by itself yet - but it is never in the session."""
    pages = _scan(google, _numbered(2), "fb")
    _later(SETTLE + 1)

    with scan_queue._being_stored(OWNER, scan_queue.Storing(TWO, SCHOOL, "Mumbai")):
        doc = _listed(google)[_key(pages[0])]
        response = _confirm(google, _key(pages[0]), KIDS[0]["student_id"])

    assert doc["status"] == "ready" and doc["can_confirm"] is True
    assert _ids(doc) == [pages[0], pages[1]] and _sides(doc) == [True, True]
    assert doc["auto_upload_pending"] is True and doc["auto_upload_refusal"] == "a scan of yours is still being stored"
    assert response.status_code == 200, response.get_json()
    assert _filed(drive, 0) == [_image(pages[0]), _image(pages[1])]


def test_a_two_sheet_form_saved_as_numbered_images_pairs_by_scan_order_as_before(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """Its Declaration pages headed at the top of their own sheets: no Declaration side, so nothing waits for the scan to finish."""
    pages = [_sent(google, name) for name in _numbered(4, "Scan {}.jpeg")]
    for n, at in enumerate((0, 2)):
        _id_side(pages[at], n)
        _skipped(pages[at + 1])

    docs = _listed(google)

    assert set(docs) == {_key(pages[0]), _key(pages[2])}
    for at in (0, 2):
        doc = docs[_key(pages[at])]
        assert _ids(doc) == [pages[at], pages[at + 1]] and _sides(doc) == [False, False]
        assert doc["status"] == "ready" and doc["attention"] == []
        assert doc["auto_upload_refusal"] == "its stack may still go on"
    _later(SETTLE + 1)
    assert all(doc["auto_upload"] is True for doc in _listed(google).values())


def test_a_single_numbered_image_pairs_by_scan_order_as_before(
    google: FlaskClient, fake_db: Any, kids: Students
) -> None:
    """A session of one image is no file of several: here the Declaration side straight after its front, in a file of its own."""
    front = _sent(google, "front.jpg")
    back = _sent(google, "Scan 7.jpeg")
    _id_side(front, 0)
    _declaration_side(back)

    docs = _listed(google)

    assert set(docs) == {_key(front)}
    assert _ids(docs[_key(front)]) == [front, back] and _sides(docs[_key(front)]) == [False, False]
    assert docs[_key(front)]["status"] == "ready" and docs[_key(front)]["attention"] == []


# ================================================================== automatic upload
def test_a_stack_of_sessions_and_files_scanned_both_sides_in_plain_order_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    images = [_sent(google, name) for name in _numbered(2, "Scan {}.jpeg")]
    response = _send(
        google, "duplex.tif", _tiff([_form_page((next(_WIDTHS), 874)) for _ in range(4)]), auto_upload="1", sender=ONE
    )
    assert response.status_code == 202, response.get_json()
    pages = [page["page_id"] for page in response.get_json()["pages"]]
    more = [_sent(google, name) for name in _numbered(2, "20260101122000_{:04d}.jpg")]
    sheets = [(images[0], images[1]), (pages[0], pages[1]), (pages[2], pages[3]), (more[0], more[1])]
    for n, (front, back) in enumerate(sheets):
        _id_side(front, n)
        _declaration_side(back)
    _later(SETTLE + 1)

    docs = _listed(google)

    for front, back in sheets:
        assert _ids(docs[_key(front)]) == [front, back]
        assert docs[_key(front)]["auto_upload"] is True, docs[_key(front)]["auto_upload_refusal"]
    response = _auto(google, 3, more[0])
    assert response.status_code == 200, response.get_json()
    assert _filed(drive, 3) == [_image(more[0]), _image(more[1])]


@pytest.mark.parametrize(
    "others",
    ["images named with no number", "the two-sheet form"],
)
def test_a_session_in_among_other_scans_keeps_its_stack_waiting_for_a_person(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, others: str
) -> None:
    session = _scan(google, _numbered(2), "fb")
    if others == "the two-sheet form":
        front, back = _sent(google, "Scan 1.tiff"), _sent(google, "Scan 2.tiff")
        _skipped(back)
    else:
        front, back = _sent(google, "front.jpg"), _sent(google, "back.jpg")
        _declaration_side(back)
    _id_side(front, 1)
    _later(SETTLE + 1)

    docs = _listed(google)

    assert _ids(docs[_key(session[0])]) == [session[0], session[1]] and _sides(docs[_key(session[0])]) == [True, True]
    assert _ids(docs[_key(front)]) == [front, back] and _sides(docs[_key(front)]) == [False, False]
    for page_id in (session[0], front):
        _waits_for_a_person(google, page_id, MIXED)
    _refused_with_nothing_saved(_auto(google, 0, session[0]), fake_db, drive)


def test_with_automatic_upload_off_a_clean_session_still_waits_for_a_person(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "AUTO_UPLOAD", False)
    pages = _scan(google, _numbered(2), "fb")
    _later(SETTLE + 1)

    doc = _listed(google)[_key(pages[0])]
    assert doc["status"] == "ready" and _sides(doc) == [True, True]
    _waits_for_a_person(google, pages[0], "automatic upload is switched off on this server")
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)


# ================================================================== batch codes on
def test_with_batch_codes_on_numbered_images_are_gathered_by_their_codes_as_before(
    google: FlaskClient, fake_db: Any, kids: Students, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sheet fed the other way up is its parts in scan order, its back first - nothing pairs by number."""
    monkeypatch.setattr(config, "BATCH_CODES", True)
    pages = [_sent(google, name) for name in _numbered(4)]
    _skipped(pages[0], code="B1-1", declaration_side=True)
    _read(pages[1], "B1-1", KIDS[0]["student_id"], dict(KIDS[0], matched_via="exact"))
    _read(pages[2], "B1-2", KIDS[1]["student_id"], dict(KIDS[1], matched_via="exact"))
    _skipped(pages[3], code="B1-2", declaration_side=True)

    [batch] = _batches(google)["batches"]
    docs = {doc["key"]: doc for doc in batch["documents"]}

    assert batch["label"] == "B1" and set(docs) == {"B1-1", "B1-2"}
    assert _ids(docs["B1-1"]) == [pages[0], pages[1]] and _ids(docs["B1-2"]) == [pages[2], pages[3]]
    assert all(doc["status"] == "ready" for doc in docs.values())
    assert not any("both_sides" in page for doc in docs.values() for page in doc["pages"])


# ================================================================== the page
@needs_node
def test_page_says_a_sheet_waits_for_the_rest_of_its_scan(client: FlaskClient, tmp_path: Path) -> None:
    """Not for its Declaration page, which it has: its card's pill and the status line say so, and there is no Confirm."""
    out = _run_page(client, tmp_path, _OFF_RESULTS_JS + r"""
      const page = await openPage();
      const RAVI = { student_id: "RAVKUM020315", student_name: "Ravi Kumar", school_name: "School A", grade_name: "5", division_name: "B" };
      const side = (id, extra) => pg(id, Object.assign({ both_sides: true, skip_reason: "" }, extra));
      const front = side("p1", { source_name: "20260101120000_0001.jpg", student_id: ASHA.student_id, matched_student: ASHA });
      const back = side("p2", { source_name: "20260101120000_0002.jpg", skip_reason: "declaration" });
      const alone = pg("p3", { source_name: "Scan 9.tiff", student_id: RAVI.student_id, matched_student: RAVI, both_sides: false });
      await answerBatches(page, offDocs([
        pdoc(front, { parts: 2, pages: [front, back], status: "waiting", student_id: ASHA.student_id,
          matched_student: ASHA, message: WAITING }),
        pdoc(alone, { status: "waiting", student_id: RAVI.student_id, matched_student: RAVI,
          message: "Waiting for its Declaration page - nothing has come after it yet." }),
      ]));
      const pill = (c) => c._all().find((e) => e.classList.contains("ocr-pill")).textContent;
      const sheet = card(page, "20260101120000_0001.jpg");
      const form = card(page, "Scan 9.tiff");
      return {
        sheetPill: pill(sheet), sheetText: textIn(sheet), sheetConfirm: Boolean(buttonIn(sheet, "Confirm and upload")),
        formPill: pill(form), status: page.byId["ocr-status"].textContent,
      };
    """.replace("WAITING", repr(REST_OF_SCAN)))

    assert out["sheetPill"] == "Waiting for the rest of its scan"
    assert REST_OF_SCAN in out["sheetText"] and out["sheetConfirm"] is False
    assert out["formPill"] == "Waiting for its Declaration page"
    assert "1 document waiting for a Declaration page" in out["status"]
    assert "1 document waiting for the rest of its scan" in out["status"]
