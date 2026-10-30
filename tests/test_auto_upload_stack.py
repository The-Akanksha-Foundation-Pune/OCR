"""Upload matched pages automatically, batch codes off: a form goes up by itself only with its whole stack.

A stack is a run (documents._runs): the pages one scan page sent, each within
RUN_GAP_MINUTES of the one before. None of its forms goes up by itself until
the whole stack qualifies (documents._judge_stack) - then each that passes
its own checks may, each with a confirm of its own:

  - finished: its last page reached the server STACK_SETTLE_MINUTES ago,
    every page of it has been read, none of the person's scans that may
    touch it is still being stored, and no stack sent before it that would
    hold it if it were held is still being read;
  - clean: Student ID page (its ID read by OCR), Declaration page (one the
    reader recognised), by turns from first to last, all sent the same way;
    no blank page, no page that could not be read, nothing held, nothing
    corrected by hand - nor a correction that cannot be read;
  - alone: no page for its school from any other scan page from
    RUN_GAP_MINUTES before it to STACK_SETTLE_MINUTES after it, no held
    stack before it from its own scan page, at any time, and none for its
    school from any scan page in the HELD_STACK_HOURS before it.

Covered here: each rule both ways, a form that fails only its own checks
waiting without stopping the rest, the ten stacks the red-team check found
going up wrongly and the ones its review found after them, a scan still
being stored when the clock passes the settle time, a person's correction
while a form of its stack is being confirmed by itself, batch codes on
unchanged - and the page (Node), which says when each Ready form goes up or
why it waits, keeps the card as it is while it does, asks again after a
refusal that was only for now, and confirms only what the server lists as
going up now.

What scan order cannot show is not tested as safe, because it is not: two
Declaration pages swapped, and two faults that cancel out into a stack that
takes turns as it should - a form missing its Declaration page with a lone
Declaration page straight after it, a few seconds apart, say. Nor is what a
pause longer than STACK_SETTLE_MINUTES cuts off: the forms before it are
judged as they were when it settled.

The database is the FakeDB, Drive the FakeDrive, OCR the stand-in readers
of tests/test_batch_queue.py; nothing leaves the test.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
import pytest
from flask import Flask
from flask.testing import FlaskClient

from server import config
from server.config import (
    DECLARATION_WAIT_SECONDS,
    HELD_STACK_HOURS,
    QUEUE_CLAIM_STALE_SECONDS,
    RUN_GAP_MINUTES,
    STACK_SETTLE_MINUTES,
)
from server.services import documents, ocr_worker, scan_queue
from server.services.clock import IST
from tests import test_batch_queue as queue_tests
from tests.test_app import _run_page, needs_node
from tests.test_auto_upload import (
    _AUTO_JS,
    _auto_confirm,
    _document,
    _key,
    _parts,
    _queue,
    _refused_with_nothing_saved,
    _settle,
    _tab,
)
from tests.test_batch_codes_off import ONE, THREE, TWO, _failed, _later, _skipped
from tests.test_batch_queue import (
    AATIF,
    OWNER,
    FakeDrive,
    Students,
    _batches,
    _form_page,
    _read,
    _send,
    _tiff,
    _writes,
)

# The queue tests' fixtures, here by name: the queue in the test's own folder
# (autouse), the stand-in readers, the fake database's students, the fake
# Drive and a Google sign-in.
queue_dir = queue_tests.queue_dir
readers = queue_tests.readers
students = queue_tests.students
drive = queue_tests.drive
google = queue_tests.google

# Children whose Student ID pages are read as exact matches, in stack order.
KIDS = [
    dict(AATIF, student_id=student_id, student_name=name)
    for student_id, name in (
        ("AATKHA200618", "Aatif Khan"),
        ("MOHSHA011021", "Mohammad Shaikh"),
        ("RIYPAT150719", "Riya Patil"),
        ("SNEJAD220820", "Sneha Jadhav"),
        ("KARMEH090517", "Karan Mehta"),
        ("PRIDES110618", "Priya Desai"),
        ("ROHGUP030716", "Rohan Gupta"),
        ("ANJRAO250918", "Anjali Rao"),
    )
]
SETTLE = STACK_SETTLE_MINUTES * 60
# A file added by hand in scan page ONE's own tab.
BY_HAND = f"{ONE}-1"


@pytest.fixture(autouse=True)
def batch_codes(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Batch codes off, as the app ships; the tests that need them on say so."""
    monkeypatch.setattr(config, "BATCH_CODES", False)
    return False


@pytest.fixture(autouse=True)
def _no_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pages are read by hand here, never by a reader thread of the app's."""
    monkeypatch.setattr(ocr_worker, "should_run_in_process", lambda _debug: False)


@pytest.fixture
def kids(students: Students) -> Students:
    """Every child of KIDS in the fake database, none marked scanned."""
    for kid in KIDS:
        students.rows[kid["student_id"]] = dict(kid)
    return students


def _id_page(client: FlaskClient, n: int, name: str, **form: str) -> str:
    """KIDS[n]'s Student ID page, sent with the switch on (scan page ONE unless `form` says) and read exactly."""
    [page_id] = _queue(client, name, **form)
    _read(page_id, None, KIDS[n]["student_id"], dict(KIDS[n], matched_via="exact"))
    return page_id


def _declaration(client: FlaskClient, name: str, **form: str) -> str:
    """A Declaration page, sent with the switch on and recognised by the reader."""
    [page_id] = _queue(client, name, **form)
    _skipped(page_id)
    return page_id


def _cover(client: FlaskClient, name: str, **form: str) -> str:
    """A sheet with no Student ID and no Declaration heading read: a cover sheet, or a Declaration page missed."""
    [page_id] = _queue(client, name, **form)
    _read(page_id, None)
    return page_id


def _forms(client: FlaskClient, kids: range, prefix: str = "Scan", **form: str) -> list[str]:
    """Forms fed as they should be - each Student ID page, then its Declaration page; every page id."""
    pages: list[str] = []
    for n in kids:
        pages.append(_id_page(client, n, f"{prefix} {len(pages) + 1}.tiff", **form))
        pages.append(_declaration(client, f"{prefix} {len(pages) + 1}.tiff", **form))
    return pages


def _listed(client: FlaskClient) -> dict[str, dict[str, Any]]:
    """Every document GET /api/batches lists, by key."""
    return {doc["key"]: doc for group in _batches(client)["batches"] for doc in group["documents"]}


def _auto(client: FlaskClient, n: int, page_id: str) -> Any:
    """What the scan page sends to confirm by itself KIDS[n]'s form, which `page_id` begins."""
    return _auto_confirm(client, _key(page_id), student_id=KIDS[n]["student_id"])


def _image(page_id: str) -> bytes:
    return (scan_queue.page_dir(page_id) / scan_queue.PAGE_IMAGE).read_bytes()


def _filed(drive: FakeDrive, n: int) -> list[bytes]:
    """The page images of the one PDF filed for KIDS[n], in order."""
    [pdf_file] = [f for f in drive.uploaded() if KIDS[n]["student_id"] in f["name"]]
    with fitz.open(stream=pdf_file["data"], filetype="pdf") as pdf:
        return [pdf.extract_image(page.get_images()[0][0])["image"] for page in pdf]


def _waits_for_a_person(client: FlaskClient, page_id: str, why: str) -> None:
    """The form `page_id` begins is listed as not going up by itself, for `why`, which no wait clears."""
    doc = _listed(client)[_key(page_id)]
    assert doc["auto_upload"] is False and doc["auto_upload_pending"] is False, doc
    assert doc["auto_upload_refusal"] == why and doc["auto_upload_at"] is None


def _stamp(received_ns: int) -> str:
    """An epoch-ns time as the server writes its India-time stamps."""
    return datetime.fromtimestamp(received_ns / 1_000_000_000, IST).replace(tzinfo=None).isoformat(timespec="seconds")


def _received_ns(page_id: str) -> int:
    meta = scan_queue.load_meta(page_id)
    assert meta is not None
    return int(meta.received_ns)


# ================================================================== finished
def test_a_clean_stack_goes_up_whole_once_nothing_more_has_come_for_the_settle_time(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    pages = _forms(google, range(3))

    settling = _listed(google)
    _later(SETTLE - 5)
    nearly = _listed(google)
    refused = _auto(google, 0, pages[0])

    for doc in (settling[_key(pages[at])] for at in (0, 2, 4)):
        assert doc["status"] == "ready" and doc["auto_upload"] is False and doc["auto_upload_pending"] is True
        assert doc["auto_upload_refusal"] == "its stack may still go on"
    # Every form goes up at the same time: the stack's last page and the settle time.
    assert {nearly[_key(pages[at])]["auto_upload_at"] for at in (0, 2, 4)} == {
        _stamp(_received_ns(pages[-1]) + SETTLE * 1_000_000_000)
    }
    error = _refused_with_nothing_saved(refused, fake_db, drive)
    assert error.startswith("Its stack may still go on, so it was not uploaded automatically")

    _later(10)

    done = _listed(google)
    assert [done[_key(pages[at])]["auto_upload"] for at in (0, 2, 4)] == [True, True, True]
    assert done[_key(pages[0])]["auto_upload_pending"] is False and done[_key(pages[0])]["auto_upload_refusal"] == ""
    for n, at in ((0, 0), (1, 2), (2, 4)):
        response = _auto(google, n, pages[at])
        assert response.status_code == 200, response.get_json()
        assert _filed(drive, n) == [_image(pages[at]), _image(pages[at + 1])]


def test_a_page_of_the_stack_still_being_read_keeps_every_form_of_it_waiting(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    pages = _forms(google, range(2))
    last = _queue(google, "Scan 5.tiff")[0]
    _later(SETTLE + 1)

    doc = _listed(google)[_key(pages[0])]

    # It may turn out to be a page out of place: until it is read, nothing
    # goes - and no time is given, which that page may yet change.
    assert doc["auto_upload"] is False and doc["auto_upload_pending"] is True
    assert doc["auto_upload_refusal"] == "a page of its stack is still being read"
    assert doc["auto_upload_at"] is None
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)

    # The next form's Student ID page: now the stack ends with one, and waits
    # for its Declaration page - which never comes.
    _read(last, None, KIDS[2]["student_id"], dict(KIDS[2], matched_via="exact"))
    _waits_for_a_person(google, pages[0], "its stack ends with a Student ID page")


def test_the_time_a_stack_goes_up_is_given_only_while_time_is_all_it_waits_for(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """A reader behind, or paused for a database outage: never a time that has passed, nor one a page may yet undo."""
    pages = _forms(google, range(1))
    _id_page(google, 1, "Scan 3.tiff")
    last = _queue(google, "Scan 4.tiff")[0]  # still waiting for the reader

    before = _listed(google)[_key(pages[0])]
    _later(SETTLE + 15 * 60)
    after = _listed(google)[_key(pages[0])]

    for doc in (before, after):
        assert doc["auto_upload_pending"] is True and doc["auto_upload_at"] is None
        assert doc["auto_upload_refusal"] == "a page of its stack is still being read"
    # Read at last, as it should be: nothing else to wait for.
    _skipped(last)
    assert _listed(google)[_key(pages[0])]["auto_upload"] is True


def test_a_stack_whose_last_form_has_no_declaration_page_yet_says_so_and_then_waits_for_a_person(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    pages = _forms(google, range(2))
    last = _id_page(google, 2, "Scan 5.tiff")

    so_far = _listed(google)[_key(pages[0])]

    # Not yet, and not at any time either: if nothing else comes, it waits.
    assert so_far["auto_upload"] is False and so_far["auto_upload_pending"] is True
    assert so_far["auto_upload_refusal"] == "its stack ends with a Student ID page so far"
    assert so_far["auto_upload_at"] is None
    assert _listed(google)[_key(last)]["status"] == "waiting"

    _later(SETTLE + 1)

    _waits_for_a_person(google, pages[0], "its stack ends with a Student ID page")
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)


@pytest.mark.parametrize(
    "scan",
    [
        # From its own scan page, for whichever school: it may be more of it.
        scan_queue.Storing(ONE, "Other School", "Pune"),
        scan_queue.Storing(BY_HAND, "Other School", "Pune"),
        # For its school and city, from any other: it may be beside it.
        scan_queue.Storing(TWO, "ABMPS", "Mumbai"),
        scan_queue.Storing("", "ABMPS", "Mumbai"),
    ],
)
def test_no_stack_is_finished_while_a_scan_that_may_touch_it_is_still_being_stored(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, scan: scan_queue.Storing
) -> None:
    pages = _forms(google, range(1))
    _later(SETTLE + 1)

    with scan_queue._being_stored(OWNER, scan):
        storing = _listed(google)[_key(pages[0])]
        refused = _auto(google, 0, pages[0])

    assert storing["auto_upload"] is False and storing["auto_upload_pending"] is True
    assert storing["auto_upload_refusal"] == "a scan of yours is still being stored"
    assert storing["auto_upload_at"] is None
    error = _refused_with_nothing_saved(refused, fake_db, drive)
    # Only for now: the page is told so, and may ask again.
    assert refused.get_json()["pending"] is True
    assert error.startswith("A scan of yours is still being stored, so it was not uploaded automatically yet.")
    assert _listed(google)[_key(pages[0])]["auto_upload"] is True


def test_a_scan_from_another_scan_page_for_another_school_being_stored_does_not_hold_a_finished_stack(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """Another computer scanning another school all the while: nothing it sends can be part of this stack."""
    pages = _forms(google, range(1))
    _settle()

    with scan_queue._being_stored(OWNER, scan_queue.Storing(TWO, "Other School", "Mumbai")):
        listed = _listed(google)[_key(pages[0])]
        response = _auto(google, 0, pages[0])

    assert listed["auto_upload"] is True, listed["auto_upload_refusal"]
    assert response.status_code == 200, response.get_json()


def test_a_storing_marker_that_cannot_be_read_holds_every_stack(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    pages = _forms(google, range(1))
    _settle()

    with scan_queue._being_stored(OWNER, scan_queue.Storing(TWO, "Other School", "Mumbai")) as marker:
        # A disk error, say: what it was marked for is not known.
        marker.write_text("{", encoding="utf-8")
        listed = _listed(google)[_key(pages[0])]

    assert listed["auto_upload"] is False and listed["auto_upload_refusal"] == "a scan of yours is still being stored"


def test_no_stack_is_finished_while_a_page_of_the_person_s_cannot_be_read_from_the_queue(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listed page whose meta.json cannot be read just now - a file being replaced - is not simply left out."""
    # Seen, the first sheet - a Declaration page with nothing before it -
    # holds the stack; left out, the rest would take turns as it should.
    lead = _declaration(google, "Scan 1.tiff")
    pages = _forms(google, range(2), prefix="Rest")
    _settle()
    real_load = scan_queue.load_meta
    monkeypatch.setattr(scan_queue, "load_meta", lambda page_id: None if page_id == lead else real_load(page_id))

    unseen = _listed(google)[_key(pages[0])]
    refused = _auto(google, 0, pages[0])

    assert unseen["auto_upload"] is False and unseen["auto_upload_pending"] is True
    assert unseen["auto_upload_refusal"] == "a page of yours could not be read from the queue just now"
    _refused_with_nothing_saved(refused, fake_db, drive)
    monkeypatch.setattr(scan_queue, "load_meta", real_load)
    assert _listed(google)[_key(pages[0])]["attention"] == ["orphan_in_stack"]


def test_a_storing_marker_left_by_a_process_that_died_is_cleared(queue_dir: Path) -> None:
    scan = scan_queue.Storing(ONE, "ABMPS", "Mumbai")
    with scan_queue._being_stored(OWNER, scan) as marker:
        assert scan_queue.storing(OWNER) == [scan]
        left = marker.with_name("0" * 16)
        left.write_bytes(marker.read_bytes())
    stale = time.time() - QUEUE_CLAIM_STALE_SECONDS - 1
    os.utime(left, (stale, stale))

    assert scan_queue.storing(OWNER) == []
    assert not left.exists()


def test_a_stack_after_one_from_its_scan_page_still_being_read_waits_for_it(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    # The first sheet of an earlier stack is still with a reader; a pause
    # longer than the run gap, and the same scan page sends the next.
    first = _queue(google, "Scan 1.tiff")[0]
    _later(RUN_GAP_MINUTES * 60 + 60)
    pages = _forms(google, range(1, 2), prefix="Later")
    _later(SETTLE + 1)

    reading = _listed(google)[_key(pages[0])]

    assert reading["auto_upload"] is False and reading["auto_upload_pending"] is True
    assert reading["auto_upload_refusal"] == "a stack sent before it is still being read"
    # It turns out to be a Declaration page with nothing before it: that
    # stack is held, and so is this one, which may be the rest of it.
    _skipped(first)
    _waits_for_a_person(
        google, pages[0], "its stack may go on from one scanned before it that has a page out of place"
    )


def test_a_stack_after_one_for_its_school_from_another_scan_page_still_being_read_waits_for_it(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """The reader is behind on another computer's stack: that one may yet turn out held, and this one its rest."""
    first = _queue(google, "Scan 1.tiff", sender=TWO)[0]
    _id_page(google, 0, "Scan 2.tiff", sender=TWO)
    _declaration(google, "Scan 3.tiff", sender=TWO)
    _later(RUN_GAP_MINUTES * 60 + 60)
    pages = _forms(google, range(1, 2), prefix="Later", sender=THREE)
    _later(SETTLE + 1)

    reading = _listed(google)[_key(pages[0])]

    assert reading["auto_upload"] is False and reading["auto_upload_pending"] is True
    assert reading["auto_upload_refusal"] == "a stack sent before it is still being read"
    # It is a Declaration page with nothing before it: held, and this one
    # waits for a person.
    _skipped(first)
    _waits_for_a_person(
        google, pages[0], "its stack may go on from one scanned before it that has a page out of place"
    )


# ================================================================== clean
@pytest.mark.parametrize(
    ("lead", "why"),
    [
        # The first form's Student ID page with nothing read on it, a blank
        # Student ID line, or a page that could not be read: its Declaration
        # page goes with it, but the stack's other forms wait too.
        ("cover", "its stack does not begin with a Student ID page"),
        ("empty_id", "its stack does not begin with a Student ID page"),
        ("failed", "its stack has a page that could not be read"),
        ("declaration", "its stack has a Declaration page without its Student ID page"),
    ],
)
def test_a_stack_that_does_not_begin_with_a_student_id_page_read_waits(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, lead: str, why: str
) -> None:
    [first] = _queue(google, "Scan 1.tiff")
    if lead == "cover":
        _read(first, None)
    elif lead == "failed":
        _failed(first)
    else:
        _skipped(first, lead)
    _declaration(google, "Scan 2.tiff")
    # The rest fed as it should be, each form straight after the other.
    pages = _forms(google, range(1, 3), prefix="Rest")
    _settle()

    _waits_for_a_person(google, pages[0], why)
    _waits_for_a_person(google, pages[2], why)
    _refused_with_nothing_saved(_auto(google, 1, pages[0]), fake_db, drive)


@pytest.mark.parametrize(
    ("where", "why"),
    [
        # Where a form should begin, with one straight after it: it may be
        # that form's Student ID page fed face down.
        ("between_forms", "its stack has a page with no Declaration page between it and the next form"),
        ("inside_a_form", "its stack has a blank page"),
        ("at_the_end", "its stack has a blank page"),
    ],
)
def test_a_stack_with_a_blank_page_anywhere_waits(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, where: str, why: str
) -> None:
    first = _id_page(google, 0, "Scan 1.tiff")
    if where == "inside_a_form":
        # Passed over: the Declaration page after it is the form's own.
        _skipped(_queue(google, "Blank.tiff")[0], "blank")
    _declaration(google, "Scan 2.tiff")
    if where == "between_forms":
        _skipped(_queue(google, "Blank.tiff")[0], "blank")
    rest = _forms(google, range(1, 2), prefix="Rest")
    if where == "at_the_end":
        _skipped(_queue(google, "Blank.tiff")[0], "blank")
    _settle()

    # The form with nothing of its own to say says why its stack waits.
    _waits_for_a_person(google, rest[0], why)
    _refused_with_nothing_saved(_auto(google, 1, rest[0]), fake_db, drive)
    if where == "inside_a_form":
        assert _listed(google)[_key(first)]["auto_upload_refusal"] == (
            "a page came between its Student ID page and its Declaration page"
        )


def test_a_declaration_page_the_reader_missed_keeps_its_stack_waiting(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    id_a = _id_page(google, 0, "Scan 1.tiff")
    _cover(google, "Scan 2.tiff")
    later = _forms(google, range(1, 2), prefix="Rest")
    _settle()

    _waits_for_a_person(google, later[0], "its stack does not go Student ID page, Declaration page, and so on")
    assert _listed(google)[_key(id_a)]["attention"] == ["no_declaration"]


@pytest.mark.parametrize("correction", ["typed", "typed_and_taken_back", "attached", "attached_and_taken_off"])
def test_a_stack_a_person_has_corrected_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, correction: str
) -> None:
    pages = _forms(google, range(2))
    if correction.startswith("typed"):
        # The person typed the same ID the reader read - their hand all the same.
        google.post(f"/api/pages/{pages[2]}/student-id", json={"student_id": KIDS[1]["student_id"]})
        if correction == "typed_and_taken_back":
            google.post(f"/api/pages/{pages[2]}/student-id", json={"student_id": ""})
    else:
        last = _id_page(google, 2, "Scan 5.tiff")
        missed = _cover(google, "Scan 6.tiff")
        assert google.post(f"/api/pages/{missed}/declaration", json={"of": last}).status_code == 200
        if correction == "attached_and_taken_off":
            assert google.post(f"/api/pages/{missed}/declaration", json={"of": ""}).status_code == 200
    _settle()

    _waits_for_a_person(google, pages[0], "a page of its stack was corrected by hand")
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)


@pytest.mark.parametrize("correction", ["student_id", "declaration"])
def test_a_correction_whose_file_cannot_be_read_still_keeps_its_stack_waiting(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, correction: str
) -> None:
    """A power cut left the file a person's correction is kept in empty: what they said is lost, not that they said it."""
    pages = _forms(google, range(2))
    if correction == "student_id":
        # The last sheet, read as a Declaration page, is another child's
        # Student ID page, the person says.
        typed = google.post(f"/api/pages/{pages[3]}/student-id", json={"student_id": KIDS[2]["student_id"]})
        assert typed.status_code == 200
        kept = scan_queue.page_dir(pages[3]) / "student_id.json"
    else:
        last = _id_page(google, 2, "Scan 5.tiff")
        missed = _cover(google, "Scan 6.tiff")
        assert google.post(f"/api/pages/{missed}/declaration", json={"of": last}).status_code == 200
        kept = scan_queue.page_dir(missed) / "declaration.json"
    _settle()
    kept.write_text("", encoding="utf-8")

    docs = _listed(google)

    assert [doc["auto_upload"] for doc in docs.values()] == [False] * len(docs)
    _waits_for_a_person(google, pages[0], "a page of its stack was corrected by hand")
    if correction == "student_id":
        # Shown as the reader saw it - a form again - but never going up as one.
        assert docs[_key(pages[2])]["auto_upload_refusal"] == "a correction made on one of its pages could not be read"
    for n, at in ((0, 0), (1, 2)):
        _refused_with_nothing_saved(_auto(google, n, pages[at]), fake_db, drive)


def test_a_correction_is_on_the_disk_before_it_counts(
    google: FlaskClient, kids: Students, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written through, so a power cut leaves the correction or none - not an empty file."""
    pages = _forms(google, range(1))
    flushed: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(scan_queue.os, "fsync", lambda fd: flushed.append(fd) or real_fsync(fd))

    typed = google.post(f"/api/pages/{pages[0]}/student-id", json={"student_id": KIDS[0]["student_id"]})

    assert typed.status_code == 200 and flushed


# ================================================================== every page sent the same way
@pytest.mark.parametrize("where", ["before", "inside", "after"])
def test_a_form_added_by_hand_in_the_scan_page_s_own_tab_keeps_its_stack_waiting(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, where: str
) -> None:
    """Added by hand in tab ONE, a file is part of the stack its scan folder sends - and may stand in for a sheet out of place."""
    added: list[str] = []
    if where == "before":
        added = _forms(google, range(5, 6), prefix="Added", sender=BY_HAND)
    pages = _forms(google, range(1))
    if where == "inside":
        added = _forms(google, range(5, 6), prefix="Added", sender=BY_HAND)
    pages += _forms(google, range(1, 2), prefix="Rest")
    if where == "after":
        added = _forms(google, range(5, 6), prefix="Added", sender=BY_HAND)
    _settle()

    # A form of its own, each of them - but not a stack sent one way.
    for page_id in (pages[0], pages[2], added[0]):
        _waits_for_a_person(google, page_id, "its stack has files added by hand as well as scans from the scan folder")
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)
    _refused_with_nothing_saved(_auto(google, 5, added[0]), fake_db, drive)


def test_a_stack_added_wholly_by_hand_in_one_tab_may_go_up(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    pages = _forms(google, range(2), prefix="Added", sender=BY_HAND)
    _settle()

    for n, at in ((0, 0), (1, 2)):
        assert _listed(google)[_key(pages[at])]["auto_upload"] is True
        assert _auto(google, n, pages[at]).status_code == 200
    assert _filed(drive, 1) == [_image(pages[2]), _image(pages[3])]


# ================================================================== alone
@pytest.mark.parametrize(
    "form",
    [
        {"sender": TWO},  # another computer, or a tab loaded again
        {"sender": f"{TWO}-1"},  # a file added by hand in another tab
        {"sender": ""},  # an older page, which sends no id
    ],
)
@pytest.mark.parametrize("when", ["before", "during", "after"])
def test_a_page_for_the_same_school_from_another_scan_page_around_the_stack_keeps_it_waiting(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, form: dict[str, str], when: str
) -> None:
    if when == "before":
        _id_page(google, 5, "Other 1.tiff", **form)
        _later(RUN_GAP_MINUTES * 60 - 10)
    pages = _forms(google, range(1))
    if when == "during":
        _id_page(google, 5, "Other 1.tiff", **form)
    pages += _forms(google, range(1, 2), prefix="Rest")
    if when == "after":
        _later(SETTLE - 10)
        _id_page(google, 5, "Other 1.tiff", **form)
    _settle()

    _waits_for_a_person(
        google, pages[0], "another computer or tab sent pages for the same school around the same time"
    )
    _refused_with_nothing_saved(_auto(google, 0, pages[0]), fake_db, drive)


@pytest.mark.parametrize("other", ["another_school", "before_the_window", "after_the_window"])
def test_a_page_from_another_scan_page_for_another_school_or_further_off_does_not_stop_the_stack(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, other: str
) -> None:
    if other == "before_the_window":
        _id_page(google, 5, "Other 1.tiff", sender=TWO)
        _later(RUN_GAP_MINUTES * 60 + 10)
    pages = _forms(google, range(2))
    if other == "another_school":
        _id_page(google, 5, "Other 1.tiff", sender=TWO, school="Other School")
    _settle()
    if other == "after_the_window":
        # Once the stack has settled: it went up as it was by then.
        _id_page(google, 5, "Other 1.tiff", sender=TWO)

    doc = _listed(google)[_key(pages[0])]

    assert doc["auto_upload"] is True, doc["auto_upload_refusal"]
    assert _auto(google, 0, pages[0]).status_code == 200
    assert _filed(drive, 0) == [_image(pages[0]), _image(pages[1])]


def test_a_stack_soon_after_another_scan_page_s_stack_that_ended_with_a_student_id_page_waits(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """Within RUN_GAP_MINUTES of a stack that is not clean, whoever sent it, nothing goes up by itself."""
    _forms(google, range(1), sender=TWO)
    _id_page(google, 1, "Other 3.tiff", sender=TWO)
    _later(5 * 60)
    pages = _forms(google, range(2, 4))
    _settle()

    _waits_for_a_person(
        google, pages[0], "another computer or tab sent pages for the same school around the same time"
    )


# ================================================================== after a held stack
@pytest.mark.parametrize("school", ["ABMPS", "Other School"])
def test_a_clean_stack_from_the_scan_page_of_a_held_one_waits_however_long_after(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, school: str
) -> None:
    # A stack fed Declaration page first: held.
    _declaration(google, "Scan 1.tiff")
    _id_page(google, 0, "Scan 2.tiff")
    _declaration(google, "Scan 3.tiff")
    _later(60 * 60)
    # An hour later, from the same scan page - for another school, perhaps.
    pages = _forms(google, range(1, 3), prefix="Later", school=school)
    _settle()

    _waits_for_a_person(
        google, pages[0], "its stack may go on from one scanned before it that has a page out of place"
    )


@pytest.mark.parametrize("sender", [THREE, TWO])  # the page loaded again, or the other computer
@pytest.mark.parametrize("minutes", [RUN_GAP_MINUTES + 1, 45, HELD_STACK_HOURS * 60 - 1])
def test_a_clean_stack_from_another_scan_page_after_a_held_one_for_its_school_waits(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, sender: str, minutes: int
) -> None:
    """It may be the rest of the held one, sent after a long break: from the page loaded again, or the other computer."""
    _declaration(google, "Scan 1.tiff")
    _id_page(google, 0, "Scan 2.tiff")
    _declaration(google, "Scan 3.tiff")
    _later(minutes * 60)
    pages = _forms(google, range(1, 3), prefix="Later", sender=sender)
    _settle()

    _waits_for_a_person(
        google, pages[0], "its stack may go on from one scanned before it that has a page out of place"
    )
    # Its forms are Ready: a person confirms them as they are.
    assert _listed(google)[_key(pages[0])]["status"] == "ready"
    _refused_with_nothing_saved(_auto(google, 1, pages[0]), fake_db, drive)


@pytest.mark.parametrize("other", ["long_after", "another_school"])
def test_a_clean_stack_from_another_scan_page_long_after_a_held_one_or_for_another_school_may_go_up(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, other: str
) -> None:
    # The held stack: for this school more than HELD_STACK_HOURS before, or
    # for another school a few minutes before.
    form = {"school": "Other School"} if other == "another_school" else {}
    _declaration(google, "Scan 1.tiff", **form)
    _id_page(google, 0, "Scan 2.tiff", **form)
    _declaration(google, "Scan 3.tiff", **form)
    _later(HELD_STACK_HOURS * 60 * 60 + 60 if other == "long_after" else RUN_GAP_MINUTES * 60 + 60)
    pages = _forms(google, range(1, 3), prefix="Later", sender=THREE)
    _settle()

    assert _listed(google)[_key(pages[0])]["auto_upload"] is True
    assert _auto(google, 1, pages[0]).status_code == 200


def test_a_clean_stack_from_the_scan_page_of_one_that_was_only_unfinished_may_go_up(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    # The last child of the class before returned only the Student ID page.
    _forms(google, range(1))
    _id_page(google, 1, "Scan 3.tiff")
    _later(RUN_GAP_MINUTES * 60 + 60)
    pages = _forms(google, range(2, 4), prefix="Later")
    _settle()

    assert _listed(google)[_key(pages[0])]["auto_upload"] is True


def test_the_middle_of_a_stack_cut_twice_from_its_scan_page_waits_with_the_held_part_before_it(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """Fed Declaration page first and cut twice: the middle part takes turns as it should, the cut is the tell."""
    _declaration(google, "Scan 1.tiff")
    _id_page(google, 0, "Scan 2.tiff")
    _declaration(google, "Scan 3.tiff")
    _later(RUN_GAP_MINUTES * 60 + 60)
    middle = [
        _id_page(google, 1, "Scan 4.tiff"),
        _declaration(google, "Scan 5.tiff"),
        _id_page(google, 2, "Scan 6.tiff"),
        _declaration(google, "Scan 7.tiff"),
    ]
    _settle()

    _waits_for_a_person(
        google, middle[0], "its stack may go on from one scanned before it that has a page out of place"
    )
    assert _writes(fake_db) == [] and drive.calls == []


# ================================================================== its own checks
def test_a_form_that_fails_only_its_own_checks_waits_without_stopping_the_rest_of_its_stack(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    near = _queue(google, "Scan 1.tiff")[0]
    # A reading the lookup reached only by a repair: Ready, but not exact.
    _read(near, None, "AATKHX200618", dict(KIDS[0], matched_via="constrained_repair"))
    _declaration(google, "Scan 2.tiff")
    marked = _queue(google, "Scan 3.tiff")[0]
    _read(marked, None, KIDS[1]["student_id"], dict(KIDS[1], matched_via="exact", scanned_at="2026-10-01 09:00:00"))
    _declaration(google, "Scan 4.tiff")
    rest = _forms(google, range(2, 4), prefix="Rest")
    _settle()

    docs = _listed(google)

    assert docs[_key(near)]["auto_upload_refusal"] == "its Student ID is not an exact match"
    assert docs[_key(marked)]["auto_upload_refusal"] == "the student is already marked scanned"
    assert docs[_key(near)]["auto_upload_pending"] is False and docs[_key(marked)]["auto_upload_pending"] is False
    assert [docs[_key(rest[at])]["auto_upload"] for at in (0, 2)] == [True, True]
    assert _auto(google, 2, rest[0]).status_code == 200
    assert _auto(google, 3, rest[2]).status_code == 200
    assert len(drive.uploaded()) == 2


def test_a_form_whose_declaration_page_came_after_it_was_shown_without_one_waits_alone(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """A jam between its two sheets longer than DECLARATION_WAIT_SECONDS: that form waits, the rest may go."""
    first = _id_page(google, 0, "Scan 1.tiff")
    _later(DECLARATION_WAIT_SECONDS + 30)
    assert _listed(google)[_key(first)]["attention"] == ["no_declaration"]
    _declaration(google, "Scan 2.tiff")
    rest = _forms(google, range(1, 2), prefix="Rest")
    _settle()

    docs = _listed(google)

    assert docs[_key(first)]["status"] == "ready" and docs[_key(first)]["auto_upload"] is False
    assert docs[_key(first)]["auto_upload_refusal"] == (
        f"its Declaration page came more than {DECLARATION_WAIT_SECONDS} seconds after its Student ID page"
    )
    assert docs[_key(rest[0])]["auto_upload"] is True


def test_the_server_checks_the_stack_again_inside_the_lock(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page from another computer stored while the confirm waits for its lock: the second look sees it."""
    pages = _forms(google, range(1))
    _settle()
    shown = _parts(google, _key(pages[0]))
    real_lock = scan_queue.confirm_lock

    def lock_once_another_page_is_in(owner: str, name: str) -> Any:
        # Received inside the stack's window - a send that took its time.
        [late] = scan_queue.intake(
            _tiff([_form_page()]), "Other.tiff", source_name="Other.tiff", owner_email=OWNER,
            school="ABMPS", location="Mumbai", auto_upload=True, sender=TWO,
        )
        meta = scan_queue.load_meta(late.page_id)
        assert meta is not None
        meta.received_ns = _received_ns(pages[-1]) + 60 * 1_000_000_000
        scan_queue.save_meta(meta)
        return real_lock(owner, name)

    monkeypatch.setattr(scan_queue, "confirm_lock", lock_once_another_page_is_in)

    response = _auto_confirm(google, _key(pages[0]), page_ids=shown)

    error = _refused_with_nothing_saved(response, fake_db, drive)
    assert error.startswith("Another computer or tab sent pages for the same school around the same time")


# ================================================================== the red-team stacks
def test_red_team_1a_a_stack_fed_declaration_first_whose_first_child_returned_only_the_student_id_page(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """ID0, D1, ID1, D2, ID2: every form would take the next child's Declaration page."""
    ids = [_id_page(google, 0, "Scan 1.tiff")]
    for n in (1, 2):
        _declaration(google, f"Scan {2 * n}.tiff")
        ids.append(_id_page(google, n, f"Scan {2 * n + 1}.tiff"))
    _settle()

    for page_id in ids[:2]:
        _waits_for_a_person(google, page_id, "its stack ends with a Student ID page")
    for n in (0, 1):
        _refused_with_nothing_saved(_auto(google, n, ids[n]), fake_db, drive)


def test_red_team_1b_a_cover_sheet_on_top_of_a_stack_fed_declaration_first(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """C, D0, ID0, D1, ID1, D2, ID2: the cover sheet takes the first Declaration page."""
    _cover(google, "Scan 1.tiff")
    ids = []
    for n in range(3):
        _declaration(google, f"Scan {2 * n + 2}.tiff")
        ids.append(_id_page(google, n, f"Scan {2 * n + 3}.tiff"))
    _settle()

    for page_id in ids[:2]:
        _waits_for_a_person(google, page_id, "its stack does not begin with a Student ID page")
    assert _writes(fake_db) == [] and drive.calls == []


@pytest.mark.parametrize("pause", [5, DECLARATION_WAIT_SECONDS + 60, 9 * 60])
@pytest.mark.parametrize("class_b", ["fed_declaration_first", "first_declaration_missed", "only_two_sheets"])
def test_red_team_2_the_next_class_s_first_declaration_page_after_a_form_without_one(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, pause: int, class_b: str
) -> None:
    """Class A ends with a child who returned no Declaration page; class B, same school and scan page, follows."""
    class_a = [_id_page(google, 0, "A 1.tiff"), _declaration(google, "A 2.tiff"), _id_page(google, 1, "A 3.tiff")]
    _later(pause)
    if class_b == "first_declaration_missed":
        _cover(google, "B 1.tiff")
    else:
        _declaration(google, "B 1.tiff")
    _id_page(google, 2, "B 2.tiff")
    if class_b != "only_two_sheets":
        for n in (3, 4):
            _declaration(google, f"B {2 * n - 3}.tiff")
            _id_page(google, n, f"B {2 * n - 2}.tiff")
    _settle()

    docs = _listed(google)

    assert [doc["auto_upload"] for doc in docs.values()] == [False] * len(docs)
    for n, page_id in ((0, class_a[0]), (1, class_a[2])):
        response = _auto(google, n, page_id)
        assert response.status_code == 409, response.get_json()
    assert _writes(fake_db) == [] and drive.calls == []


@pytest.mark.parametrize("pause", [DECLARATION_WAIT_SECONDS + 60, 9 * 60])
def test_red_team_2_a_lone_declaration_page_on_top_of_the_next_class_after_a_break(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, pause: int
) -> None:
    """IDa0 Da0 IDa1 | break | D7 IDb2 Db2 IDb3 Db3: the two faults cancel into a stack that takes turns.

    Scan order cannot see it - but the form that takes the lone page was shown
    without a Declaration page for minutes before it came, so that one form
    waits; the others are paired as on paper and go up.
    """
    a0, _, a1 = _id_page(google, 0, "A 1.tiff"), _declaration(google, "A 2.tiff"), _id_page(google, 1, "A 3.tiff")
    _later(pause)
    _declaration(google, "B 1.tiff")
    class_b = _forms(google, range(2, 4), prefix="B form")
    _settle()

    response = _auto(google, 1, a1)

    error = _refused_with_nothing_saved(response, fake_db, drive)
    assert error.startswith(f"Its Declaration page came more than {DECLARATION_WAIT_SECONDS} seconds after")
    for n, page_id in ((0, a0), (2, class_b[0]), (3, class_b[2])):
        assert _auto(google, n, page_id).status_code == 200
    assert _filed(drive, 2) == [_image(class_b[0]), _image(class_b[1])]
    assert len(drive.uploaded()) == 3


@pytest.mark.parametrize("cut", ["reload", "school_switched"])
@pytest.mark.parametrize("minutes", [RUN_GAP_MINUTES + 1, 45])
def test_red_team_3_the_rest_of_a_held_stack_fed_declaration_first_after_a_long_break(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, minutes: int, cut: str
) -> None:
    """D0 ID0 D1 ID1 D2 | a long break, then the page loaded again or the school switched | ID2 D3 ID3 D4 ID4."""
    _declaration(google, "Scan 1.tiff")
    for n in (0, 1):
        _id_page(google, n, f"Scan {2 * n + 2}.tiff")
        _declaration(google, f"Scan {2 * n + 3}.tiff")
    _later(minutes * 60)
    form = {"sender": THREE} if cut == "reload" else {"school": "Other School"}
    rest = []
    for n in (2, 3, 4):
        rest.append(_id_page(google, n, f"Scan {2 * n + 2}.tiff", **form))
        if n < 4:
            _declaration(google, f"Scan {2 * n + 3}.tiff", **form)
    _settle()

    for n, page_id in ((2, rest[0]), (3, rest[1])):
        doc = _listed(google)[_key(page_id)]
        assert doc["auto_upload"] is False and doc["auto_upload_pending"] is False
        _refused_with_nothing_saved(_auto(google, n, page_id), fake_db, drive)


@pytest.mark.parametrize("form", [{"sender": TWO}, {"sender": f"{TWO}-1"}, {"sender": ""}])
def test_red_team_4_and_6_another_scan_page_s_sheet_never_stands_in_for_this_stack_s_next_one(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, form: dict[str, str]
) -> None:
    """ID0 then a stray Declaration page (ONE) - and another computer's first sheet, before ONE's own next one."""
    id0 = _id_page(google, 0, "Scan 1.tiff")
    _declaration(google, "Scan 2.tiff")
    _later(10)
    _id_page(google, 5, "Other 1.tiff", **form)

    # For a person now, however long it then waits: nothing it waits for.
    _waits_for_a_person(google, id0, "another computer or tab sent pages for the same school around the same time")
    _refused_with_nothing_saved(_auto(google, 0, id0), fake_db, drive)
    # ONE's own next sheet, 30 s on: the form's real Declaration page, out
    # of place now.
    _later(30)
    _declaration(google, "Scan 3.tiff")
    _id_page(google, 1, "Scan 4.tiff")
    _settle()
    assert _listed(google)[_key(id0)]["attention"] == ["orphan_in_stack"]
    _refused_with_nothing_saved(_auto(google, 0, id0), fake_db, drive)


@pytest.mark.parametrize("jam", [DECLARATION_WAIT_SECONDS + 5, 5 * 60, RUN_GAP_MINUTES * 60 - 30])
def test_red_team_5_a_jam_right_after_a_declaration_page_out_of_place(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, jam: int
) -> None:
    """ID0, D1 (child 1's, slipped ahead) - the feeder jams on D0 - then D0, ID1: the jam is waited out."""
    id0 = _id_page(google, 0, "Scan 1.tiff")
    _declaration(google, "Scan 2.tiff")
    _later(jam)

    during = _listed(google)[_key(id0)]
    refused = _auto(google, 0, id0)

    assert during["auto_upload"] is False and during["auto_upload_pending"] is True
    _refused_with_nothing_saved(refused, fake_db, drive)
    _declaration(google, "Scan 3.tiff")
    _id_page(google, 1, "Scan 4.tiff")
    _settle()
    assert _listed(google)[_key(id0)]["attention"] == ["orphan_in_stack"]
    _refused_with_nothing_saved(_auto(google, 0, id0), fake_db, drive)


@pytest.mark.parametrize("cut", ["reload", "other_computer"])
def test_red_team_7_the_rest_of_a_reversed_stack_cut_mid_form_and_sent_from_another_scan_page(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, cut: str
) -> None:
    """D4 ID4 D3 | eleven minutes, the page loaded again or the other computer | ID3 D2 ID2 D1 ID1."""
    _declaration(google, "Scan 1.tiff")
    _id_page(google, 4, "Scan 2.tiff")
    _declaration(google, "Scan 3.tiff")
    _later(RUN_GAP_MINUTES * 60 + 60)
    sender = TWO if cut == "reload" else THREE
    rest = []
    for n in (3, 2, 1):
        rest.append(_id_page(google, n, f"Rest {2 * (3 - n) + 1}.tiff", sender=sender))
        _declaration(google, f"Rest {2 * (3 - n) + 2}.tiff", sender=sender)
    rest.append(_id_page(google, 0, "Rest 7.tiff", sender=sender))
    _settle()

    # Held as what may go on from the held part - and it ends with a Student
    # ID page as well.
    for n, page_id in ((3, rest[0]), (2, rest[1])):
        _waits_for_a_person(
            google, page_id, "its stack may go on from one scanned before it that has a page out of place"
        )
        _refused_with_nothing_saved(_auto(google, n, page_id), fake_db, drive)


def test_red_team_8_a_late_page_from_a_held_stack_s_old_tab(
    app: Flask, google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    """D3 ID3 D2 (tab ONE) | ID2 D1 ID1 D0 ID0 from a new tab TWO | then ONE's retry of ID2 gets through."""
    _declaration(google, "Scan 1.tiff")
    _id_page(google, 3, "Scan 2.tiff")
    _declaration(google, "Scan 3.tiff")
    _later(60)
    new_tab = _tab(app)
    rest = []
    for n in (2, 1, 0):
        rest.append(_id_page(new_tab, n, f"Scan {8 - 2 * n}.tiff", sender=TWO))
        if n:
            _declaration(new_tab, f"Scan {9 - 2 * n}.tiff", sender=TWO)
    _id_page(google, 2, "Scan 4.tiff", sender=ONE)
    _settle()

    # ID1 with form 0's Declaration page: nothing of its own to say. (Its
    # stack has another scan page's pages around it, too.)
    _waits_for_a_person(
        google, rest[1], "its stack may go on from one scanned before it that has a page out of place"
    )
    _refused_with_nothing_saved(_auto(google, 1, rest[1]), fake_db, drive)


def test_red_team_9_a_sheet_received_inside_the_settle_time_but_listed_after_it_holds_the_stack(
    app: Flask, google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The next sheet is received (stamped) just before the stack settles and listed just after: it is not missed."""
    pages = _forms(google, range(1))
    _later(SETTLE - 0.4)
    inside, go_on = threading.Event(), threading.Event()
    real_encode = scan_queue.encode_page

    def slow_encode(grey: Any) -> bytes:
        # A big scan's pages take a while to store, after it was received.
        if not inside.is_set():
            inside.set()
            assert go_on.wait(timeout=30)
        return real_encode(grey)

    monkeypatch.setattr(scan_queue, "encode_page", slow_encode)
    answer: dict[str, Any] = {}
    scanner = _tab(app)
    thread = threading.Thread(
        target=lambda: answer.setdefault(
            "response", _send(scanner, "Scan 3.tiff", _tiff([_form_page()]), auto_upload="1", sender=ONE)
        )
    )
    thread.start()
    assert inside.wait(timeout=30)
    while time.time_ns() - _received_ns(pages[-1]) < (SETTLE + 0.1) * 1_000_000_000:
        time.sleep(0.01)

    listed = _listed(google)[_key(pages[0])]
    refused = _auto(google, 0, pages[0])
    go_on.set()
    thread.join(timeout=30)

    assert listed["auto_upload"] is False and listed["auto_upload_pending"] is True
    assert listed["auto_upload_refusal"] == "a scan of yours is still being stored"
    _refused_with_nothing_saved(refused, fake_db, drive)
    # Received inside the settle time, it is part of the stack: a second
    # Declaration page in a row, out of place.
    [sheet] = [page["page_id"] for page in answer["response"].get_json()["pages"]]
    assert (_received_ns(sheet) - _received_ns(pages[-1])) < SETTLE * 1_000_000_000
    _skipped(sheet)
    _settle()
    assert _listed(google)[_key(pages[0])]["attention"] == ["orphan_in_stack"]


@pytest.mark.parametrize("on", ["declaration_page", "student_id_page"])
def test_red_team_10_a_correction_while_its_form_is_going_up_by_itself_is_refused_not_overridden(
    app: Flask, google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch, on: str,
) -> None:
    id_a, declaration = _forms(google, range(1))
    _settle()
    shown = _parts(google, _key(id_a))
    inside, go_on = threading.Event(), threading.Event()
    real_lookup = documents.confirm_student_id

    def round_trip(*args: Any, **kwargs: Any) -> Any:
        # The automatic confirm's database round trip, held while the person types.
        inside.set()
        assert go_on.wait(timeout=30)
        return real_lookup(*args, **kwargs)

    monkeypatch.setattr(documents, "confirm_student_id", round_trip)
    answer: dict[str, Any] = {}
    auto_tab = _tab(app)
    thread = threading.Thread(
        target=lambda: answer.setdefault("response", _auto_confirm(auto_tab, _key(id_a), page_ids=shown))
    )
    thread.start()
    assert inside.wait(timeout=30)

    page = declaration if on == "declaration_page" else id_a
    correction = google.post(f"/api/pages/{page}/student-id", json={"student_id": KIDS[1]["student_id"]})
    go_on.set()
    thread.join(timeout=30)

    assert correction.status_code == 409
    assert "being confirmed right now" in correction.get_json()["error"]
    assert scan_queue.load_overrides(page).student_id is None
    # The form went up as it was checked, and nothing the person said is lost:
    # it was refused, and is refused now that the page is closed.
    assert answer["response"].status_code == 200, answer["response"].get_json()
    late = google.post(f"/api/pages/{page}/student-id", json={"student_id": KIDS[1]["student_id"]})
    assert late.status_code == 409


def test_red_team_10_a_correction_made_first_stops_the_automatic_confirm(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    id_a, declaration = _forms(google, range(1))
    _settle()
    shown = _parts(google, _key(id_a))

    assert google.post(f"/api/pages/{declaration}/student-id", json={"student_id": KIDS[1]["student_id"]}).status_code == 200

    _refused_with_nothing_saved(_auto_confirm(google, _key(id_a), page_ids=shown), fake_db, drive)


def test_a_correction_is_refused_while_any_form_of_its_stack_is_being_confirmed(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive
) -> None:
    pages = _forms(google, range(2))
    missed = _cover(google, "Scan 5.tiff")

    with scan_queue.confirm_lock(OWNER, _key(pages[0])):
        typed = google.post(f"/api/pages/{pages[3]}/student-id", json={"student_id": KIDS[4]["student_id"]})
        attached = google.post(f"/api/pages/{missed}/declaration", json={"of": pages[2]})

    for refused in (typed, attached):
        assert refused.status_code == 409 and "being confirmed right now" in refused.get_json()["error"]
    assert scan_queue.load_overrides(pages[3]).student_id is None
    assert scan_queue.load_overrides(missed).declaration_of is None
    # Once nothing is being confirmed, it is kept.
    assert google.post(f"/api/pages/{pages[3]}/student-id", json={"student_id": KIDS[4]["student_id"]}).status_code == 200


# ================================================================== the review's stacks
@pytest.mark.parametrize("jam", [0, 30])
def test_review_a_student_id_page_added_by_hand_after_a_stray_declaration_page_keeps_the_stack_waiting(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, jam: int
) -> None:
    """ID0 D0 ID1, a stray Declaration page - and, with the feeder stopped, a Student ID page added by hand in the same tab; then D1 ID2 D2.

    As read the stack takes turns: the page added by hand makes the stray
    one look like the Declaration page of the form before it.
    """
    pages = _forms(google, range(1))
    id1 = _id_page(google, 1, "Scan 3.tiff")
    _declaration(google, "Scan 4.tiff")  # another child's, between form 1's two sheets
    _later(jam)
    added = _id_page(google, 5, "Added.tiff", sender=BY_HAND)
    _later(jam)
    _declaration(google, "Scan 5.tiff")  # form 1's own
    _forms(google, range(2, 3), prefix="Then")
    _settle()

    docs = _listed(google)

    assert [doc["auto_upload"] for doc in docs.values()] == [False] * len(docs)
    _waits_for_a_person(google, id1, "its stack has files added by hand as well as scans from the scan folder")
    for n, page_id in ((0, pages[0]), (1, id1), (5, added)):
        assert _auto(google, n, page_id).status_code == 409
    assert _writes(fake_db) == [] and drive.calls == []


@pytest.mark.parametrize("added", ["declaration_after", "both_ends"])
def test_review_a_stack_fed_declaration_first_and_made_to_take_turns_by_pages_added_by_hand_waits(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, added: str
) -> None:
    """ID0 D1 ID1 D2 ID2 (finding 1) with a Declaration page added by hand after it - or D0 ID0 D1 ID1 D2 ID2 with a Student ID page before it as well."""
    ids: list[str] = []
    if added == "both_ends":
        _id_page(google, 5, "Added 1.tiff", sender=BY_HAND)
        _declaration(google, "Scan 0.tiff")
    ids.append(_id_page(google, 0, "Scan 1.tiff"))
    for n in (1, 2):
        _declaration(google, f"Scan {2 * n}.tiff")
        ids.append(_id_page(google, n, f"Scan {2 * n + 1}.tiff"))
    _later(DECLARATION_WAIT_SECONDS + 10)
    _declaration(google, "Added 2.tiff", sender=BY_HAND)
    _settle()

    for n, page_id in enumerate(ids):
        assert _listed(google)[_key(page_id)]["auto_upload"] is False
        assert _auto(google, n, page_id).status_code == 409
    assert _writes(fake_db) == [] and drive.calls == []


@pytest.mark.parametrize("sender", [THREE, TWO])  # the page loaded again, or the other computer
@pytest.mark.parametrize("pause", [RUN_GAP_MINUTES * 60 + 1, RUN_GAP_MINUTES * 60 + 60])
def test_review_the_rest_of_a_held_stack_that_ends_with_a_declaration_page_waits(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, sender: str, pause: int
) -> None:
    """D0 ID0 D1 - fed Declaration page first, held - then, after a break, from another scan page: ID1 D2 ID2 D3.

    The last child returned only its Declaration page, so the rest takes
    turns as it should and ends with one.
    """
    _declaration(google, "Scan 1.tiff")
    _id_page(google, 0, "Scan 2.tiff")
    _declaration(google, "Scan 3.tiff")
    _later(pause)
    rest = []
    for n in (1, 2):
        rest.append(_id_page(google, n, f"Rest {2 * n - 1}.tiff", sender=sender))
        _declaration(google, f"Rest {2 * n}.tiff", sender=sender)
    _settle()

    for n, page_id in ((1, rest[0]), (2, rest[1])):
        _waits_for_a_person(
            google, page_id, "its stack may go on from one scanned before it that has a page out of place"
        )
        _refused_with_nothing_saved(_auto(google, n, page_id), fake_db, drive)


@pytest.mark.parametrize("sender", [THREE, TWO])  # the page loaded again, or the other computer
def test_review_the_middle_of_a_stack_cut_twice_sent_from_another_scan_page_waits(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, sender: str
) -> None:
    """D0 ID0 D1 | eleven minutes, another scan page | ID1 D2 | eleven minutes | ID2 D3 ID3."""
    _declaration(google, "Scan 1.tiff")
    _id_page(google, 0, "Scan 2.tiff")
    _declaration(google, "Scan 3.tiff")
    _later(RUN_GAP_MINUTES * 60 + 60)
    middle = [_id_page(google, 1, "Middle 1.tiff", sender=sender), _declaration(google, "Middle 2.tiff", sender=sender)]
    _later(RUN_GAP_MINUTES * 60 + 60)

    # Settled during the second break, and taking turns as it should.
    _waits_for_a_person(
        google, middle[0], "its stack may go on from one scanned before it that has a page out of place"
    )
    _refused_with_nothing_saved(_auto(google, 1, middle[0]), fake_db, drive)
    rest = [_id_page(google, 2, "Rest 1.tiff", sender=sender)]
    _declaration(google, "Rest 2.tiff", sender=sender)
    rest.append(_id_page(google, 3, "Rest 3.tiff", sender=sender))
    _settle()
    for n, page_id in ((1, middle[0]), (2, rest[0])):
        _refused_with_nothing_saved(_auto(google, n, page_id), fake_db, drive)


# ================================================================== batch codes on
def test_with_batch_codes_on_a_ready_document_goes_up_by_itself_with_no_stack_to_wait_for(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    first, second = _queue(google, "Scan 1.tiff")[0], _queue(google, "Scan 2.tiff")[0]
    # Another computer sends for the same school at the same moment.
    other = _queue(google, "Other 1.tiff", sender=TWO)[0]
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))
    _read(second, "B1-1")
    _read(other, "B1-2")

    doc = _document(google, "B1-1")

    assert doc["status"] == "ready" and doc["auto_upload"] is True
    assert "auto_upload_pending" not in doc and "auto_upload_refusal" not in doc and "auto_upload_at" not in doc
    assert _auto_confirm(google, "B1-1").status_code == 200


def test_with_batch_codes_a_document_whose_typed_code_cannot_be_read_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The code a person typed over a misread one is lost to a power cut: the misread code is not taken as theirs."""
    monkeypatch.setattr(config, "BATCH_CODES", True)
    [page] = _queue(google, "Scan 1.tiff")
    _read(page, "B1-7", "AATKHA200618", dict(AATIF, matched_via="exact"))
    assert google.post(f"/api/pages/{page}/code", json={"code": "B1-1"}).status_code == 200
    (scan_queue.page_dir(page) / "code.json").write_text("", encoding="utf-8")

    doc = _document(google, "B1-7")

    assert doc["status"] == "ready" and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, "B1-7"), fake_db, drive)
    assert error.startswith("A correction made on one of its pages could not be read")


# ================================================================== the page
_STACK_JS = _AUTO_JS + r"""
  // As the server lists a Ready form not going up by itself yet, or not at all.
  function settling(id, student, at) {
    return one(id, student, { auto_upload: false, auto_upload_pending: true, auto_upload_at: at,
      auto_upload_refusal: "its stack may still go on" });
  }
  function waiting(id, student, why) {
    return one(id, student, { auto_upload: false, auto_upload_pending: false, auto_upload_at: null,
      auto_upload_refusal: why });
  }
  function autoLine(page, name) {
    const found = card(page, name)._all().find((e) => e.classList.contains("ocr-doc-auto"));
    return found ? found.textContent : "";
  }
  // A time today by the page's own clock, as the server stamps it.
  function todayAt(clock, time) {
    const now = new clock.Date();
    const day = [now.getFullYear(), String(now.getMonth() + 1).padStart(2, "0"), String(now.getDate()).padStart(2, "0")];
    return day.join("-") + "T" + time;
  }
"""


@needs_node
def test_page_says_when_each_ready_form_goes_up_by_itself_or_why_it_waits(client: FlaskClient, tmp_path: Path) -> None:
    """The server's word on each Ready card: a time, a reason for now, or the one reason a person is needed."""
    out = _run_page(client, tmp_path, _STACK_JS + r"""
      const clock = makeClock();
      const page = loadPage({ clock, local: { "ocr.school": "School A", "ocr.location": "Pune" } });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      const docs = listing([
        settling("p1", ASHA, todayAt(clock, "10:42:05")),
        one("p2", RAVI, { auto_upload: false, auto_upload_pending: true, auto_upload_at: null,
          auto_upload_refusal: "its stack ends with a Student ID page so far" }),
        waiting("p3", ASHA, "another computer or tab sent pages for the same school around the same time"),
        waiting("p4", RAVI, "its Student ID is not an exact match"),
      ]);
      await answerBatches(page, docs);
      await tick(20);
      const lines = ["p1", "p2", "p3", "p4"].map((id) => autoLine(page, "Scan " + id + ".jpg"));
      // The switch off on this browser: none of it is true here.
      setSwitch(page, false);
      await tick(20);
      const off = ["p1", "p3"].map((id) => autoLine(page, "Scan " + id + ".jpg"));
      return { lines, off, confirms: confirms(page).length };
    """)

    assert out["lines"] == [
        "Goes up by itself at 10:42 if nothing else comes and this page is still open on screen.",
        "Not yet: its stack ends with a Student ID page so far.",
        "Waits for you: another computer or tab sent pages for the same school around the same time.",
        "Waits for you: its Student ID is not an exact match.",
    ]
    assert out["off"] == ["", ""]
    assert out["confirms"] == 0


@needs_node
def test_page_confirms_a_settling_stack_only_once_the_server_lists_it_as_going_up_now(
    client: FlaskClient, tmp_path: Path
) -> None:
    """While the stack settles it keeps asking and confirms nothing; listed as going up, each form goes, once."""
    out = _run_page(client, tmp_path, _STACK_JS + r"""
      const clock = makeClock();
      const page = loadPage({ clock, local: { "ocr.school": "School A", "ocr.location": "Pune" } });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      const looks = () => page.log.filter((c) => c.url.includes("/api/batches")).length;
      const at = todayAt(clock, "10:10:00");
      const stack = listing([settling("p1", ASHA, at), settling("p2", RAVI, at)]);
      await answerBatches(page, stack);
      for (let i = 0; i < 3; i += 1) {
        await clock.advance(3000);
        await answerBatches(page, stack);
      }
      const whileSettling = { looks: looks(), confirms: confirms(page).length, line: autoLine(page, "Scan p1.jpg") };
      // An answer with one gone up by itself and one still not: only the first is sent.
      await clock.advance(3000);
      await answerBatches(page, listing([one("p1", ASHA), waiting("p2", RAVI, "its stack has a blank page")]));
      const sent = await waitingFor(page, "/api/documents/Pp1/confirm");
      await answer(page.calls, "/api/documents/Pp1/confirm", {
        document: { status: "uploaded", key: "Pp1~ab12cd34", confirmed_automatically: true }, uploaded: true, finished: true,
      });
      await answerBatches(page, listing([waiting("p2", RAVI, "its stack has a blank page")]));
      await tick(20);
      return { whileSettling, body: JSON.parse(sent.opts.body), all: confirms(page).map(([url]) => url),
               line: autoLine(page, "Scan p2.jpg") };
    """)

    assert out["whileSettling"]["confirms"] == 0 and out["whileSettling"]["looks"] == 4
    assert out["whileSettling"]["line"] == (
        "Goes up by itself at 10:10 if nothing else comes and this page is still open on screen."
    )
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p1d"], "automatic": True}
    assert out["all"] == ["/api/documents/Pp1/confirm"]
    assert out["line"] == "Waits for you: its stack has a blank page."


@needs_node
def test_page_asks_again_after_a_refusal_only_for_now_and_never_after_one_for_good(
    client: FlaskClient, tmp_path: Path
) -> None:
    """A scan being stored meanwhile: asked for again on the next look, not straight away. Refused for good: never."""
    out = _run_page(client, tmp_path, _STACK_JS + r"""
      const clock = makeClock();
      const page = loadPage({ clock, local: { "ocr.school": "School A", "ocr.location": "Pune" } });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      const both = listing([one("p1", ASHA), one("p2", RAVI)]);
      await answerBatches(page, both);
      await answer(page.calls, "/api/documents/Pp1/confirm", {
        error: "A scan of yours is still being stored, so it was not uploaded automatically yet. Nothing was saved - it is tried again by itself.",
        pending: true,
      }, 409);
      await answer(page.calls, "/api/documents/Pp2/confirm", {
        error: "Its Student ID is not an exact match, so it was not uploaded automatically. Nothing was saved - check it and press Confirm and upload.",
      }, 409);
      await tick(20);
      const firstLook = confirms(page).map(([url]) => url);
      const said = textIn(card(page, "Scan p1.jpg"));
      // The next look lists both as going up again: only the first is asked for.
      await answerBatches(page, both);
      await waitingFor(page, "/api/documents/Pp1/confirm");
      await answer(page.calls, "/api/documents/Pp1/confirm", {
        document: { status: "uploaded", key: "Pp1~ab12cd34", confirmed_automatically: true }, uploaded: true, finished: true,
      });
      // The one refused for good, listed as settling once more: no promise on its card.
      await answerBatches(page, listing([settling("p2", RAVI, todayAt(clock, "10:42:05"))]));
      await tick(20);
      await clock.advance(30000);
      return { firstLook, said, all: confirms(page).map(([url]) => url), line: autoLine(page, "Scan p2.jpg") };
    """)

    assert out["firstLook"] == ["/api/documents/Pp1/confirm", "/api/documents/Pp2/confirm"]
    assert "A scan of yours is still being stored, so it was not uploaded automatically yet." in out["said"]
    assert "Not uploaded automatically:" not in out["said"]
    assert out["all"] == ["/api/documents/Pp1/confirm", "/api/documents/Pp2/confirm", "/api/documents/Pp1/confirm"]
    assert out["line"] == ""


@needs_node
def test_page_makes_no_promise_for_a_card_whose_box_holds_another_id(client: FlaskClient, tmp_path: Path) -> None:
    """Someone typed another ID in the box, not looked up yet: the card is theirs, and says so."""
    out = _run_page(client, tmp_path, _STACK_JS + r"""
      const clock = makeClock();
      const page = loadPage({ clock, local: { "ocr.school": "School A", "ocr.location": "Pune" } });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      await answerBatches(page, listing([settling("p1", ASHA, todayAt(clock, "10:42:05"))]));
      await tick(20);
      const box = fieldIn(card(page, "Scan p1.jpg"), "id");
      box.value = "RAVKUM020315";
      box.fire("input");
      const typedLine = autoLine(page, "Scan p1.jpg");
      await clock.advance(3000);
      await answerBatches(page, listing([settling("p1", ASHA, todayAt(clock, "10:42:05"))]));
      await tick(20);
      const lookedAgain = autoLine(page, "Scan p1.jpg");
      // Listed as going up now: the page leaves it.
      await clock.advance(3000);
      await answerBatches(page, listing([one("p1", ASHA)]));
      await clock.advance(10000);
      // The ID matched put back: the promise is back too.
      box.value = "MOHSHA011021";
      box.fire("input");
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, listing([settling("p1", ASHA, todayAt(clock, "10:42:05"))]));
      await tick(20);
      return { typedLine, lookedAgain, confirms: confirms(page).length, back: autoLine(page, "Scan p1.jpg") };
    """)

    said = "Waits for you: its Student ID box no longer holds the ID matched."
    assert out["typedLine"] == said and out["lookedAgain"] == said
    assert out["confirms"] == 0
    assert out["back"] == "Goes up by itself at 10:42 if nothing else comes and this page is still open on screen."


@needs_node
def test_page_keeps_a_ready_card_as_it_is_while_its_line_changes(client: FlaskClient, tmp_path: Path) -> None:
    """Each sheet of its stack changes what the card says about going up - in place, not by drawing it again."""
    out = _run_page(client, tmp_path, _STACK_JS + r"""
      const clock = makeClock();
      const page = loadPage({ clock, local: { "ocr.school": "School A", "ocr.location": "Pune" } });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      const answers = [
        { auto_upload_pending: true, auto_upload_at: null, auto_upload_refusal: "a page of its stack is still being read" },
        { auto_upload_pending: true, auto_upload_at: null, auto_upload_refusal: "its stack ends with a Student ID page so far" },
        { auto_upload_pending: true, auto_upload_at: todayAt(clock, "10:10:00"), auto_upload_refusal: "its stack may still go on" },
      ];
      const lines = [];
      let first = null;
      for (const said of answers) {
        if (first) await clock.advance(3000);
        await answerBatches(page, listing([one("p1", ASHA, Object.assign({ auto_upload: false }, said))]));
        await tick(20);
        if (!first) {
          first = card(page, "Scan p1.jpg");
          // Someone is typing in its Student ID box meanwhile.
          page.document.activeElement = fieldIn(first, "id");
        }
        lines.push(autoLine(page, "Scan p1.jpg"));
      }
      const last = card(page, "Scan p1.jpg");
      return { lines, same: first === last, focus: last.contains(page.document.activeElement) };
    """)

    assert out["lines"] == [
        "Not yet: a page of its stack is still being read.",
        "Not yet: its stack ends with a Student ID page so far.",
        "Goes up by itself at 10:10 if nothing else comes and this page is still open on screen.",
    ]
    assert out["same"] is True and out["focus"] is True


@needs_node
def test_page_on_a_hidden_tab_neither_looks_nor_confirms_until_it_is_shown_again(
    client: FlaskClient, tmp_path: Path
) -> None:
    """The time on the card holds only while the page is open on screen: hidden, nothing goes up - and on its return it does."""
    out = _run_page(client, tmp_path, _STACK_JS + r"""
      const clock = makeClock();
      const page = loadPage({ clock, local: { "ocr.school": "School A", "ocr.location": "Pune" } });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      const looks = () => page.log.filter((c) => c.url.includes("/api/batches")).length;
      await answerBatches(page, listing([settling("p1", ASHA, todayAt(clock, "10:10:00"))]));
      await tick(20);
      const line = autoLine(page, "Scan p1.jpg");
      page.document.visibilityState = "hidden";
      page.document.fire("visibilitychange");
      const before = looks();
      await clock.advance(15 * 60 * 1000);
      const hidden = { looks: looks() - before, confirms: confirms(page).length };
      page.document.visibilityState = "visible";
      page.document.fire("visibilitychange");
      await answerBatches(page, listing([one("p1", ASHA)]));
      await waitingFor(page, "/api/documents/Pp1/confirm");
      return { line, hidden, confirms: confirms(page).length };
    """)

    assert out["line"] == "Goes up by itself at 10:10 if nothing else comes and this page is still open on screen."
    assert out["hidden"] == {"looks": 0, "confirms": 0}
    assert out["confirms"] == 1


@needs_node
def test_page_says_not_yet_for_a_stack_whose_page_is_still_being_read_long_after_it_settled(
    google: FlaskClient, fake_db: Any, kids: Students, drive: FakeDrive, tmp_path: Path
) -> None:
    """The reader is behind - paused for a database outage, say: the server's own answer, on the page, promises no time."""
    pages = _forms(google, range(1))
    _id_page(google, 1, "Scan 3.tiff")
    _queue(google, "Scan 4.tiff")  # still waiting for the reader
    _later(SETTLE * 2)
    data = _batches(google)

    out = _run_page(google, tmp_path, _STACK_JS + r"""
      const page = await openWith();
      await answerBatches(page, DATA);
      await tick(20);
      return { line: autoLine(page, "Scan 1.tiff"), confirms: confirms(page).length };
    """.replace("DATA", json.dumps(data)))

    assert _listed(google)[_key(pages[0])]["auto_upload_at"] is None
    assert out == {"line": "Not yet: a page of its stack is still being read.", "confirms": 0}
