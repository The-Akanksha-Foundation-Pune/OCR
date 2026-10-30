"""Batch codes switched off: each form is its Student ID page and the Declaration page straight after it.

The app ships with BATCH_CODES off (server/config.py). Every other test runs
with codes on (tests/conftest.py); this file overrides that fixture and
checks what changes without them:

  - The worker never reads the code corner, here or with Google; the layout
    still turns a page the right way up, and a page with no layout to match
    is kept as it was scanned.
  - Pages are paired only inside a run: what one scan page sent (its
    sender id), each within RUN_GAP_MINUTES of the one before. Two computers
    on one sign-in each pair their own pages; files added by hand pair with
    each other and with what the scan folder sent; a long pause starts a new
    run; a sender id of any other shape is kept as none.
  - Each form is one document: its Student ID page - read, matched or typed -
    and the one Declaration page scanned straight after it, which has no
    Student ID. A blank page between them is passed over. Any other
    Declaration page - with nothing before it, after another, after a
    confirmed form (however long ago), or sent for another school - joins
    nothing, waits under Unsorted saying only what is true of it, and holds
    every form of its run for a person. So does a form's first page with no
    Declaration page between it and the next form's - two Student ID pages
    in a row, or a stack fed Declaration page first whose first sheet was
    not read as one - and a held run holds the run that may be the rest of
    its stack, after a pause or a reload, but no other.
  - A form is Ready when it matched and has its Declaration page; Waiting,
    with no Confirm, while the page after its Student ID page may still be
    that page or nothing has come yet for DECLARATION_WAIT_SECONDS; and
    Needs attention after that, when it matched nobody, or when its run is
    held. A form with no Student ID waits under Unsorted, its Declaration
    page with it, until one is typed. A Declaration page the reader missed
    can be attached by hand to the form straight before it - never a page
    straight after one offered itself - and taken off, on the form's card or
    under Unsorted; it is a page of its own again once the page before it no
    longer begins a form, or a Student ID is typed on it.
  - Confirm and Retry upload work as before; the PDF in Drive is
    <Name>_<ID>.pdf, the Student ID page first, and a confirm shown only the
    Student ID page is refused once its Declaration page has joined. A
    student confirmed lately is flagged when scanned again.
  - Pages read with a code before the switch are sorted like any other, and
    the documents confirmed with a code are still listed; with batch codes
    on again, nothing of the pairing is used.
  - The page asks for no codes, shows none, shows each form's two pages on
    one card, says what waits, what is held and what is out of place, offers
    to attach a missed Declaration page - saying what that comes to - takes
    one off under Unsorted too, and leaves the scan folder as it is.

The database is the FakeDB, Drive the FakeDrive, OCR the stand-in readers
of tests/test_batch_queue.py; nothing leaves the test.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import fitz
import numpy as np
import pytest
from flask.testing import FlaskClient

from server import config
from server.config import DECLARATION_WAIT_SECONDS, RUN_GAP_MINUTES
from server.services import documents, ocr_worker, scan_queue
from tests import test_batch_queue as queue_tests
from tests.test_app import _run_page, needs_node
from tests.test_batch_queue import (
    AATIF,
    MOHSHA,
    OWNER,
    FakeDrive,
    Readers,
    Students,
    _batches,
    _confirm,
    _form_page,
    _queue,
    _read,
    _ready_document,
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


@pytest.fixture(autouse=True)
def batch_codes(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Batch codes off, as the app ships (overrides tests/conftest.py's fixture)."""
    monkeypatch.setattr(config, "BATCH_CODES", False)
    return False


def _documents(client: FlaskClient) -> list[dict[str, Any]]:
    """The one list of documents GET /api/batches gives without batch codes."""
    groups = _batches(client)["batches"]
    if not groups:
        return []
    [group] = groups
    assert group["batch"] is None and group["label"] == ""
    return group["documents"]


def _key(page_id: str) -> str:
    return f"P{page_id}"


# Scan pages' ids, as each makes one for itself when it is loaded.
ONE = "0123456789abcdef" * 2
TWO = "fedcba9876543210" * 2
THREE = "00112233445566778899aabbccddeeff"


def _scan(client: FlaskClient, name: str = "Scan 1.tiff", sender: str = "") -> str:
    """One single-page scan, sent from the scan page whose id is `sender` (none, as an older page); its page id."""
    response = _send(client, name, _tiff([_form_page()]), **({"sender": sender} if sender else {}))
    assert response.status_code == 202, response.get_json()
    [page] = response.get_json()["pages"]
    return str(page["page_id"])


def _later(seconds: float) -> None:
    """As if `seconds` had gone by: every page queued so far reached the server that much earlier."""
    for folder in scan_queue.queue_root().iterdir():
        meta = scan_queue.load_meta(folder.name)
        if meta is not None:
            meta.received_ns -= int(seconds * 1_000_000_000)
            scan_queue.save_meta(meta)


def _skipped(page_id: str, reason: str = "declaration", **result: Any) -> None:
    """Mark a queued page read as the worker reads a skipped page: no Student ID read off it.

    A Declaration page - a form's second sheet - unless `reason` says otherwise.
    """
    (scan_queue.queue_root() / "_pending" / page_id).unlink()
    scan_queue.save_result(
        page_id, scan_queue.PageResult(skip_reason=reason, blank=reason == "blank", **result)
    )
    meta = scan_queue.load_meta(page_id)
    assert meta is not None
    meta.status = scan_queue.STATUS_READ
    scan_queue.save_meta(meta)


def _failed(page_id: str) -> None:
    """Mark a queued page as the worker leaves one it could not read: failed, with nothing read off it."""
    (scan_queue.queue_root() / "_pending" / page_id).unlink()
    meta = scan_queue.load_meta(page_id)
    assert meta is not None
    meta.status, meta.attempts, meta.error = scan_queue.STATUS_FAILED, 3, "MemoryError: out of memory"
    scan_queue.save_meta(meta)


def _form(
    client: FlaskClient,
    student: dict[str, Any],
    name: str = "Scan.tiff",
    declaration_name: str = "Scan 1.tiff",
) -> list[str]:
    """One form sent as the scanner sends it - its Student ID page, then its Declaration page - and read."""
    id_page = _queue(client, name)[0]
    declaration = _queue(client, declaration_name)[0]
    _read(id_page, None, student["student_id"], dict(student))
    _skipped(declaration)
    return [id_page, declaration]


def _ids(pages: list[dict[str, Any]]) -> list[str]:
    return [page["page_id"] for page in pages]


def _image_of(page_id: str) -> bytes:
    """A queued page's page.jpg, which goes into the PDF as it is."""
    return (scan_queue.page_dir(page_id) / scan_queue.PAGE_IMAGE).read_bytes()


# ================================================================== the worker
def test_the_corner_is_not_read_and_the_page_still_is(
    client: FlaskClient, fake_db: Any, students: Students, readers: Readers, queue_dir: Path
) -> None:
    [page_id] = _queue(client)
    readers.codes = ["B1-1"]
    readers.ids = ["AATKHA200618"]

    assert ocr_worker.process_one() == ocr_worker.READ

    result = json.loads((queue_dir / page_id / "result.json").read_text())
    assert result["code"] is None and result["code_raw"] == "" and result["code_region"] is None
    assert result["student_id"] == "AATKHA200618"
    assert result["matched_student"]["student_name"] == "Aatif Khan"
    # Neither corner was read - not by RapidOCR, not by Google.
    assert readers.code_calls == 0 and readers.vision_allowed == []
    assert _writes(fake_db) == []


def test_a_page_with_no_layout_is_kept_as_it_was_scanned(
    client: FlaskClient, students: Students, readers: Readers, queue_dir: Path
) -> None:
    # Upside down, and no form layout to tell: with codes on, the code read
    # in the turned page's corner would have turned it round.
    [page_id] = _queue(client, data=_tiff([_form_page(mark="bottom-left")]))
    readers.code_rule = lambda _page: "B2-3"
    before = (queue_dir / page_id / "page.jpg").read_bytes()

    ocr_worker.process_one()

    result = json.loads((queue_dir / page_id / "result.json").read_text())
    assert result["orientation"] == 0 and result["code"] is None
    assert readers.code_calls == 0
    assert (queue_dir / page_id / "page.jpg").read_bytes() == before


def test_the_layout_still_turns_an_upside_down_page(
    client: FlaskClient, students: Students, readers: Readers, monkeypatch: pytest.MonkeyPatch,
    queue_dir: Path,
) -> None:
    [page_id] = _queue(client, data=_tiff([_form_page(mark="bottom-left")]))

    def identify(image: np.ndarray) -> Any:
        upright = float(np.mean(image[: image.shape[0] // 10, image.shape[1] * 3 // 4 :] < 128)) > 0.05
        return SimpleNamespace(error=0.001 if upright else 0.015)

    monkeypatch.setattr(ocr_worker, "identify", identify)

    ocr_worker.process_one()

    result = json.loads((queue_dir / page_id / "result.json").read_text())
    assert result["orientation"] == 180 and result["code"] is None
    assert readers.code_calls == 0
    stored = cv2.imread(str(queue_dir / page_id / "page.jpg"), cv2.IMREAD_GRAYSCALE)
    assert float(np.mean(stored[: stored.shape[0] // 10, stored.shape[1] * 3 // 4 :] < 128)) > 0.05


# ================================================================== documents
def test_each_form_is_its_student_id_page_and_the_declaration_page_after_it(client: FlaskClient) -> None:
    # As the scanner sends them: one sheet a scan, each form's two one after the other.
    first = _form(client, AATIF, "Scan.tiff", "Scan 1.tiff")
    second = _form(client, MOHSHA, "Scan 2.tiff", "Scan 3.tiff")
    loose = _queue(client, "Scan 4.tiff")[0]
    loose_declaration = _queue(client, "Scan 5.tiff")[0]
    _read(loose, None)
    _skipped(loose_declaration)

    body = _batches(client)
    docs = _documents(client)

    # Newest scan first, both sheets in each, keyed by the Student ID page.
    assert [d["key"] for d in docs] == [_key(second[0]), _key(first[0])]
    assert all(scan_queue.is_document_key(d["key"]) for d in docs)
    for doc, parts, student in ((docs[0], second, MOHSHA), (docs[1], first, AATIF)):
        assert doc["parts"] == 2 and _ids(doc["pages"]) == parts
        assert doc["code"] == "" and doc["status"] == "ready" and doc["can_confirm"] is True
        assert doc["student_id"] == student["student_id"] and doc["attention"] == []
        assert doc["message"] == f"Matched {student['student_name']}."
        assert [p["skip_reason"] for p in doc["pages"]] == ["", "declaration"]
        assert [p["document_key"] for p in doc["pages"]] == [doc["key"], doc["key"]]
        assert [p["needs_person"] for p in doc["pages"]] == [False, False]
    # The form with no Student ID waits under Unsorted as one card, its
    # Declaration page with it rather than on a card of its own.
    [waiting] = body["unsorted"]
    assert waiting["page_id"] == loose and waiting["needs_person"] is True
    assert _ids(waiting["declaration_pages"]) == [loose_declaration]
    assert waiting["declaration_pages"][0]["needs_person"] is False
    assert waiting["attach_to"] is None
    # The same keys on every look, and each can be fetched by it - never by
    # its Declaration page.
    assert [d["key"] for d in _documents(client)] == [d["key"] for d in docs]
    fetched = client.get(f"/api/documents/{_key(first[0])}")
    assert fetched.status_code == 200 and fetched.get_json()["document"]["student_id"] == "AATKHA200618"
    assert _ids(fetched.get_json()["document"]["pages"]) == first
    assert client.get(f"/api/documents/{_key(first[1])}").status_code == 404
    assert client.get(f"/api/documents/{_key(loose)}").status_code == 404


def test_an_id_that_matched_nobody_needs_attention(client: FlaskClient) -> None:
    [page_id] = _queue(client)
    _skipped(_queue(client, "Scan 2.tiff")[0])
    _read(page_id, None, "AATKHA200619", None)

    [doc] = _documents(client)

    assert doc["status"] == "needs_attention" and doc["attention"] == ["no_match"] and doc["parts"] == 2
    assert doc["student_id"] == "AATKHA200619" and doc["can_confirm"] is True
    assert doc["message"] == "No student with ID AATKHA200619 at ABMPS. Check the ID against the paper."


def test_a_student_id_page_waits_while_the_page_after_it_may_be_its_declaration_page(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    first = _queue(google, "Scan.tiff")[0]
    after = _queue(google, "Scan 1.tiff")[0]  # not read yet
    _read(first, None, "AATKHA200618", dict(AATIF))

    [doc] = _documents(google)

    assert doc["status"] == "waiting" and doc["can_confirm"] is False and doc["auto_upload"] is False
    assert doc["message"] == "Waiting for its Declaration page - the page after it is still being read."
    assert _ids(doc["pages"]) == [first] and doc["attention"] == []
    refused = _confirm(google, key=_key(first))
    assert refused.status_code == 409 and "still being read" in refused.get_json()["error"]
    assert _writes(fake_db) == [] and drive.calls == []
    # However long the reading takes.
    _later(DECLARATION_WAIT_SECONDS + 60)
    assert _documents(google)[0]["status"] == "waiting"

    _skipped(after)

    [doc] = _documents(google)
    assert doc["status"] == "ready" and _ids(doc["pages"]) == [first, after]


def test_a_student_id_page_with_nothing_after_it_waits_a_little_then_may_be_confirmed_alone(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google, "Scan.tiff")
    _read(page_id, None, "AATKHA200618", dict(AATIF))

    # The scanner may still be saving the form's second sheet: no Confirm yet.
    [doc] = _documents(google)
    assert doc["status"] == "waiting" and doc["can_confirm"] is False and doc["auto_upload"] is False
    assert doc["message"] == "Waiting for its Declaration page - nothing has come after it yet."
    refused = _confirm(google, key=_key(page_id))
    assert refused.status_code == 409
    assert refused.get_json()["error"] == (
        "Waiting for its Declaration page - nothing has come after it yet. Try again in a moment."
    )
    assert _writes(fake_db) == [] and drive.calls == []
    _later(DECLARATION_WAIT_SECONDS - 5)
    assert _documents(google)[0]["status"] == "waiting"

    _later(10)

    [doc] = _documents(google)
    assert doc["status"] == "needs_attention" and doc["attention"] == ["no_declaration"]
    assert doc["can_confirm"] is True and doc["auto_upload"] is False
    assert doc["message"] == (
        "No Declaration page came after this Student ID page - check the form."
        " It can still be confirmed without one."
    )
    # A person may confirm it as it is: its PDF is the Student ID page alone.
    response = _confirm(google, key=_key(page_id))
    assert response.status_code == 200, response.get_json()
    [pdf_file] = drive.uploaded()
    with fitz.open(stream=pdf_file["data"], filetype="pdf") as pdf:
        assert pdf.page_count == 1
        images = [pdf.extract_image(page.get_images()[0][0])["image"] for page in pdf]
    assert images == [_image_of(page_id)]


def test_a_student_id_page_with_the_next_form_after_it_has_no_declaration_page(client: FlaskClient) -> None:
    first = _queue(client, "Scan.tiff")[0]
    second = _queue(client, "Scan 1.tiff")[0]
    _read(first, None, "AATKHA200618", dict(AATIF))
    _read(second, None, "MOHSHA011021", dict(MOHSHA))

    docs = {d["key"]: d for d in _documents(client)}

    # The page after the first is the next form's Student ID page, so there
    # is nothing to wait for - and the stack is out of step there: every
    # form after it may have the next one's Declaration page.
    alone = docs[_key(first)]
    assert alone["status"] == "needs_attention" and alone["attention"] == ["no_declaration", "gap_in_stack"]
    assert _ids(alone["pages"]) == [first] and alone["can_confirm"] is True
    assert alone["message"].endswith(
        "This stack has a page with no Declaration page between it and the next form, so its forms"
        " may be paired wrongly - check them."
    )
    # Nothing has come after the second yet.
    assert docs[_key(second)]["status"] == "waiting" and docs[_key(second)]["can_confirm"] is False


# ------------------------------------------------------------------ runs
def test_a_pause_longer_than_the_run_gap_starts_a_new_run(client: FlaskClient) -> None:
    id_page = _scan(client, "Scan.tiff", ONE)
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _later(RUN_GAP_MINUTES * 60 + 1)
    declaration = _scan(client, "Scan 1.tiff", ONE)
    _skipped(declaration)

    body = _batches(client)
    [doc] = _documents(client)

    # Not paired: the scan page paused too long between them.
    assert _ids(doc["pages"]) == [id_page] and doc["attention"] == ["no_declaration"]
    [page] = body["unsorted"]
    assert page["page_id"] == declaration and page["skip_reason"] == "declaration"
    assert page["message"] == "A Declaration page with no Student ID page right before it."
    assert page["document_key"] is None


def test_a_pause_within_the_run_gap_keeps_the_run(client: FlaskClient) -> None:
    id_page = _scan(client, "Scan.tiff", ONE)
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _later(RUN_GAP_MINUTES * 60 - 30)
    declaration = _scan(client, "Scan 1.tiff", ONE)
    _skipped(declaration)

    [doc] = _documents(client)

    assert _ids(doc["pages"]) == [id_page, declaration] and doc["status"] == "ready"


def test_a_page_from_another_scan_page_is_never_paired(client: FlaskClient) -> None:
    id_page = _scan(client, "Scan.tiff", ONE)
    declaration = _scan(client, "Scan 1.tiff", TWO)
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _skipped(declaration)

    body = _batches(client)
    [doc] = _documents(client)

    # Its own stack has nothing after it yet.
    assert _ids(doc["pages"]) == [id_page] and doc["status"] == "waiting"
    [page] = body["unsorted"]
    assert page["page_id"] == declaration
    assert page["message"] == "A Declaration page with no Student ID page right before it."


def test_two_computers_on_one_sign_in_each_pair_their_own_pages(client: FlaskClient) -> None:
    # Two desks scanning one school at once, signed in as one person: the
    # scans reach the server interleaved.
    first = _scan(client, "Scan.tiff", ONE)
    other = _scan(client, "Scan.tiff", TWO)
    first_declaration = _scan(client, "Scan 1.tiff", ONE)
    other_declaration = _scan(client, "Scan 1.tiff", TWO)
    _read(first, None, "AATKHA200618", dict(AATIF))
    _read(other, None, "MOHSHA011021", dict(MOHSHA))
    _skipped(first_declaration)
    _skipped(other_declaration)

    docs = {d["key"]: d for d in _documents(client)}

    assert _ids(docs[_key(first)]["pages"]) == [first, first_declaration]
    assert _ids(docs[_key(other)]["pages"]) == [other, other_declaration]
    assert [d["status"] for d in docs.values()] == ["ready", "ready"]
    assert _batches(client)["unsorted"] == []


def test_files_added_by_hand_pair_with_each_other_and_with_what_the_scan_folder_sent(client: FlaskClient) -> None:
    # With no scan folder each sheet is added as it is scanned, one pick at a
    # time; and a sheet the scan folder could not send may be added by hand.
    picked = _scan(client, "Scan.tiff", f"{ONE}-1")
    picked_declaration = _scan(client, "Scan 1.tiff", f"{ONE}-1")
    from_folder = _scan(client, "Scan 2.tiff", ONE)
    added = _scan(client, "Scan 3.tiff", f"{ONE}-1")
    _read(picked, None, "AATKHA200618", dict(AATIF))
    _skipped(picked_declaration)
    _read(from_folder, None, "MOHSHA011021", dict(MOHSHA))
    _skipped(added)

    docs = {d["key"]: d for d in _documents(client)}

    # One scan page, so one stack.
    assert _ids(docs[_key(picked)]["pages"]) == [picked, picked_declaration]
    assert _ids(docs[_key(from_folder)]["pages"]) == [from_folder, added]
    assert [d["status"] for d in docs.values()] == ["ready", "ready"]
    assert _batches(client)["unsorted"] == []


def test_a_scan_keeps_its_sender_only_in_the_shape_the_page_makes_it(client: FlaskClient, queue_dir: Path) -> None:
    kept = {}
    for sent in (ONE, f"{ONE}-1", f"{ONE}-2", ONE.upper(), "a1b2", f"{ONE}-0", "../../etc", ""):
        response = _send(client, "Scan.tiff", _tiff([_form_page()]), sender=sent)
        assert response.status_code == 202, response.get_json()
        [page] = response.get_json()["pages"]
        kept[sent] = json.loads((queue_dir / page["page_id"] / "meta.json").read_text())["sender"]

    # Its id, or with "-1" for a file added by hand. Anything else is kept
    # as none: the scan itself is never refused for it.
    assert kept == {ONE: ONE, f"{ONE}-1": f"{ONE}-1", f"{ONE}-2": "", ONE.upper(): "", "a1b2": "",
                    f"{ONE}-0": "", "../../etc": "", "": ""}


# ------------------------------------------------------------------ out of place
def test_a_stack_fed_declaration_first_holds_every_form_in_it(
    client: FlaskClient, fake_db: Any, students: Students
) -> None:
    # Each form fed Declaration page first: every Student ID page takes the
    # next form's Declaration page, and only the first page shows it.
    stray, first, wrong, second = (_scan(client, f"Scan {n}.tiff", ONE) for n in range(4))
    _skipped(stray)
    _read(first, None, "AATKHA200618", dict(AATIF))
    _skipped(wrong)
    _read(second, None, "MOHSHA011021", dict(MOHSHA))
    _later(DECLARATION_WAIT_SECONDS + 1)

    body = _batches(client)
    docs = {d["key"]: d for d in _documents(client)}

    [page] = body["unsorted"]
    assert page["page_id"] == stray and page["message"] == "A Declaration page with no Student ID page right before it."
    held = docs[_key(first)]
    assert _ids(held["pages"]) == [first, wrong]
    assert held["status"] == "needs_attention" and held["attention"] == ["orphan_in_stack"]
    assert held["message"] == (
        "This stack has a Declaration page without its Student ID page, so its forms"
        " may be paired wrongly - check them."
    )
    assert held["can_confirm"] is True and held["auto_upload"] is False
    assert docs[_key(second)]["attention"] == ["no_declaration", "orphan_in_stack"]

    # A Student ID typed on the stray page does not put the stack right.
    client.post(f"/api/pages/{stray}/student-id", json={"student_id": "AATKHA200619"})

    docs = {d["key"]: d for d in _documents(client)}
    assert docs[_key(first)]["attention"] == ["orphan_in_stack"]
    assert docs[_key(stray)]["attention"] == ["no_match", "no_declaration", "orphan_in_stack"]


def test_a_missing_student_id_page_leaves_an_orphan_that_holds_its_stack(client: FlaskClient) -> None:
    # The second form's Student ID page never reached the server.
    first, first_declaration, orphan, third, third_declaration = (
        _scan(client, f"Scan {n}.tiff", ONE) for n in (0, 1, 3, 4, 5)
    )
    _read(first, None, "AATKHA200618", dict(AATIF))
    _read(third, None, "MOHSHA011021", dict(MOHSHA))
    for page_id in (first_declaration, orphan, third_declaration):
        _skipped(page_id)

    body = _batches(client)
    docs = {d["key"]: d for d in _documents(client)}

    assert _ids(docs[_key(first)]["pages"]) == [first, first_declaration]
    assert _ids(docs[_key(third)]["pages"]) == [third, third_declaration]
    for doc in docs.values():
        assert doc["status"] == "needs_attention" and doc["attention"] == ["orphan_in_stack"]
    [page] = body["unsorted"]
    assert page["page_id"] == orphan and page["message"] == (
        "A Declaration page right after another Declaration page - a form has one,"
        " so the Student ID page before this one may be missing."
    )


def test_two_sheets_swapped_across_two_forms_hold_the_stack(client: FlaskClient) -> None:
    # Form A's Declaration page fed after form B's Student ID page.
    id_a, id_b, d_a, d_b = (_scan(client, f"Scan {n}.tiff", ONE) for n in range(4))
    _read(id_a, None, "AATKHA200618", dict(AATIF))
    _read(id_b, None, "MOHSHA011021", dict(MOHSHA))
    _skipped(d_a)

    docs = {d["key"]: d for d in _documents(client)}

    # B takes A's Declaration page, exactly one, straight after it: only the
    # two Student ID pages in a row show that the stack is out of step.
    assert _ids(docs[_key(id_a)]["pages"]) == [id_a]
    assert docs[_key(id_a)]["attention"] == ["no_declaration", "gap_in_stack"]
    assert _ids(docs[_key(id_b)]["pages"]) == [id_b, d_a]
    assert docs[_key(id_b)]["status"] == "needs_attention" and docs[_key(id_b)]["attention"] == ["gap_in_stack"]
    assert docs[_key(id_b)]["auto_upload"] is False

    _skipped(d_b)

    # B's own Declaration page, after it, is the orphan.
    docs = {d["key"]: d for d in _documents(client)}
    assert docs[_key(id_b)]["attention"] == ["orphan_in_stack"]
    [page] = _batches(client)["unsorted"]
    assert page["page_id"] == d_b and page["message"].startswith("A Declaration page right after another")


@pytest.mark.parametrize("first", ["missed", "blank", "failed"])
def test_a_stack_fed_declaration_first_is_held_when_its_first_sheet_is_not_read_as_one(
    client: FlaskClient, first: str
) -> None:
    # Form A's Declaration page first, its heading missed, fed face down or
    # not read at all: nothing says it is a Declaration page.
    lead, id_a, wrong, id_b = (_scan(client, f"Scan {n}.tiff", ONE) for n in range(4))
    if first == "missed":
        _read(lead, None)
    elif first == "blank":
        _skipped(lead, "blank")
    else:
        _failed(lead)
    _read(id_a, None, "AATKHA200618", dict(AATIF))
    _skipped(wrong)
    _read(id_b, None, "MOHSHA011021", dict(MOHSHA))

    body = _batches(client)
    docs = {d["key"]: d for d in _documents(client)}

    # A takes B's Declaration page - but it begins straight after a page
    # with no Declaration page after it, so its whole stack is held.
    held = docs[_key(id_a)]
    assert _ids(held["pages"]) == [id_a, wrong]
    assert held["status"] == "needs_attention" and held["attention"] == ["gap_in_stack"]
    assert held["message"] == (
        "This stack has a page with no Declaration page between it and the next form, so its forms"
        " may be paired wrongly - check them."
    )
    assert held["can_confirm"] is True and held["auto_upload"] is False
    assert "gap_in_stack" in docs[_key(id_b)]["attention"]
    [page] = body["unsorted"]
    assert page["page_id"] == lead and page["attach_to"] is None


@pytest.mark.parametrize("cut", ["pause", "reload"])
def test_a_stack_fed_declaration_first_and_cut_in_two_holds_the_rest_of_it_too(client: FlaskClient, cut: str) -> None:
    stray, id_a, wrong = (_scan(client, f"Scan {n}.tiff", ONE) for n in range(3))
    _skipped(stray)
    _read(id_a, None, "AATKHA200618", dict(AATIF))
    _skipped(wrong)
    # A jam longer than RUN_GAP_MINUTES, or the scan page loaded again (a new
    # id): the rest is a run of its own, beginning with a Student ID page.
    if cut == "pause":
        _later(RUN_GAP_MINUTES * 60 + 1)
    sender = ONE if cut == "pause" else TWO
    id_b, wrong_too = _scan(client, "Scan 3.tiff", sender), _scan(client, "Scan 4.tiff", sender)
    _read(id_b, None, "MOHSHA011021", dict(MOHSHA))
    _skipped(wrong_too)

    docs = {d["key"]: d for d in _documents(client)}

    assert docs[_key(id_a)]["attention"] == ["orphan_in_stack"]
    rest = docs[_key(id_b)]
    assert _ids(rest["pages"]) == [id_b, wrong_too]
    assert rest["status"] == "needs_attention" and rest["attention"] == ["after_held_stack"]
    assert rest["message"] == (
        "This stack may go on from one scanned before it that has a page out of place, so its forms"
        " may be paired wrongly - check them."
    )
    assert rest["can_confirm"] is True and rest["auto_upload"] is False


def test_a_run_that_cannot_be_the_rest_of_a_held_one_is_not_held(client: FlaskClient) -> None:
    _skipped(_scan(client, "Scan.tiff", ONE))
    # Straight after it, another scan page sending for another school; long
    # after it, another one for the same school.
    other_school = [
        _send(client, f"Scan {n}.tiff", _tiff([_form_page()]), sender=TWO, school="Other School")
        .get_json()["pages"][0]["page_id"]
        for n in (1, 2)
    ]
    _read(other_school[0], None, "AATKHA200618", dict(AATIF))
    _skipped(other_school[1])
    _later(RUN_GAP_MINUTES * 60 + 1)
    much_later = _scan(client, "Scan 3.tiff", THREE), _scan(client, "Scan 4.tiff", THREE)
    _read(much_later[0], None, "MOHSHA011021", dict(MOHSHA))
    _skipped(much_later[1])

    docs = {d["key"]: d for d in _documents(client)}

    assert _ids(docs[_key(other_school[0])]["pages"]) == other_school
    assert _ids(docs[_key(much_later[0])]["pages"]) == list(much_later)
    assert [(d["status"], d["attention"]) for d in docs.values()] == [("ready", []), ("ready", [])]


def test_a_second_declaration_page_in_a_row_joins_nothing_and_holds_its_stack(client: FlaskClient) -> None:
    id_page = _queue(client, "Scan.tiff")[0]
    declaration = _queue(client, "Scan 1.tiff")[0]
    again = _queue(client, "Scan 2.tiff")[0]
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _skipped(declaration)
    _skipped(again)

    body = _batches(client)
    [doc] = _documents(client)

    # A form has one Declaration page.
    assert _ids(doc["pages"]) == [id_page, declaration] and doc["parts"] == 2
    assert doc["status"] == "needs_attention" and doc["attention"] == ["orphan_in_stack"]
    [page] = body["unsorted"]
    assert page["page_id"] == again and page["skip_reason"] == "declaration"
    assert page["message"].startswith("A Declaration page right after another Declaration page")
    assert page["document_key"] is None and page["declaration_pages"] == [] and page["attach_to"] is None


def test_a_declaration_page_with_no_student_id_page_before_it_waits_under_unsorted_alone(
    client: FlaskClient,
) -> None:
    stray = _queue(client, "Scan.tiff")[0]
    _skipped(stray)

    [page] = _batches(client)["unsorted"]
    assert page["page_id"] == stray and page["message"] == "A Declaration page with no Student ID page right before it."
    assert page["needs_person"] is True and _documents(client) == []

    # A form scanned after it does not take it - and waits for a person.
    form = _form(client, AATIF, "Scan 1.tiff", "Scan 2.tiff")

    [doc] = _documents(client)
    assert _ids(doc["pages"]) == form and doc["attention"] == ["orphan_in_stack"]
    assert _ids(_batches(client)["unsorted"]) == [stray]


def test_a_blank_page_between_a_student_id_page_and_its_declaration_page_is_passed_over(
    client: FlaskClient,
) -> None:
    id_page = _queue(client, "Scan.tiff")[0]
    blank = _queue(client, "Scan 1.tiff")[0]
    declaration = _queue(client, "Scan 2.tiff")[0]
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _skipped(blank, "blank")

    # A blank page is never a Declaration page: the one after it may be.
    [doc] = _documents(client)
    assert doc["status"] == "waiting" and _ids(doc["pages"]) == [id_page]

    _skipped(declaration)

    [doc] = _documents(client)
    assert doc["status"] == "ready" and _ids(doc["pages"]) == [id_page, declaration]
    # The blank page stays as it is, under Unsorted.
    [page] = _batches(client)["unsorted"]
    assert page["page_id"] == blank and page["skip_reason"] == "blank" and page["blank"] is True
    assert page["attach_to"] is None


@pytest.mark.parametrize("sent", [{"school": "Other School"}, {"location": "Pune"}])
def test_a_declaration_page_sent_for_another_school_or_city_is_not_joined(
    client: FlaskClient, sent: dict[str, str]
) -> None:
    id_page = _queue(client, "Scan.tiff")[0]
    response = _send(client, "Scan 1.tiff", _tiff([_form_page()]), **sent)
    other = response.get_json()["pages"][0]["page_id"]
    _read(id_page, None, "AATKHA200618", dict(AATIF))

    # Not even waiting for it: a page sent for another school is never
    # this form's Declaration page.
    [doc] = _documents(client)
    assert doc["status"] == "needs_attention" and doc["attention"] == ["no_declaration"]

    _skipped(other)

    [doc] = _documents(client)
    assert _ids(doc["pages"]) == [id_page] and doc["attention"] == ["no_declaration", "orphan_in_stack"]
    [page] = _batches(client)["unsorted"]
    assert page["page_id"] == other
    assert page["message"] == "A Declaration page sent for another school or city than the page before it."


def test_a_declaration_page_read_before_its_student_id_page_waits_for_it(client: FlaskClient) -> None:
    id_page = _queue(client, "Scan.tiff")[0]
    declaration = _queue(client, "Scan 1.tiff")[0]
    # A second reader finished the Declaration page first.
    _skipped(declaration)

    body = _batches(client)
    assert body["unsorted"] == [] and _documents(client) == []
    assert body["queue"]["queued"] == 1

    _read(id_page, None, "AATKHA200618", dict(AATIF))

    [doc] = _documents(client)
    assert _ids(doc["pages"]) == [id_page, declaration] and doc["status"] == "ready"


def test_a_declaration_page_read_after_its_form_was_confirmed_joins_nothing(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    earlier = _queue(google, "Scan.tiff")[0]
    id_page = _queue(google, "Scan 1.tiff")[0]
    _read(earlier, None, "MOHSHA011021", dict(MOHSHA))
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    # Nothing came after it in time, so a person confirmed it on its own.
    _later(DECLARATION_WAIT_SECONDS + 1)
    assert _confirm(google, key=_key(id_page)).status_code == 200
    declaration = _queue(google, "Scan 2.tiff")[0]

    _skipped(declaration)

    docs = {d["key"]: d for d in _documents(google) if d["status"] != "uploaded"}
    # Not handed to the form before, and never said to be the confirmed one's.
    assert _ids(docs[_key(earlier)]["pages"]) == [earlier]
    assert docs[_key(earlier)]["attention"] == ["no_declaration", "orphan_in_stack"]
    [page] = _batches(google)["unsorted"]
    assert page["page_id"] == declaration
    assert page["message"] == "A Declaration page that came after a form that was already confirmed."


def test_a_declaration_page_after_a_form_confirmed_with_its_own_says_only_that(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    form = _form(google, AATIF, "Scan.tiff", "Scan 1.tiff")
    assert _confirm(google, key=_key(form[0])).status_code == 200
    late = _queue(google, "Scan 2.tiff")[0]

    _skipped(late)

    # Never that it is the confirmed form's: that form had its own.
    [page] = _batches(google)["unsorted"]
    assert page["page_id"] == late and page["message"] == (
        "A Declaration page right after another Declaration page - a form has one,"
        " so the Student ID page before this one may be missing."
    )


def test_a_declaration_page_after_a_form_confirmed_long_ago_still_joins_nothing(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    earlier = _queue(google, "Scan.tiff")[0]
    id_page = _queue(google, "Scan 1.tiff")[0]
    _read(earlier, None, "MOHSHA011021", dict(MOHSHA))
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _later(DECLARATION_WAIT_SECONDS + 1)
    assert _confirm(google, key=_key(id_page)).status_code == 200
    declaration = _queue(google, "Scan 2.tiff")[0]
    _skipped(declaration)
    # The confirmed form drops off the list after DONE_DOCUMENTS_HOURS.
    old = time.time() - (config.DONE_DOCUMENTS_HOURS + 1) * 3600
    for record in (scan_queue.owner_dir(OWNER) / "docs").glob("*.json"):
        os.utime(record, (old, old))

    docs = _documents(google)

    # Its Student ID page still stands between them: nothing pairs differently.
    assert [d["key"] for d in docs] == [_key(earlier)]
    assert _ids(docs[0]["pages"]) == [earlier] and docs[0]["attention"] == ["no_declaration", "orphan_in_stack"]
    [page] = _batches(google)["unsorted"]
    assert page["page_id"] == declaration
    assert page["message"] == "A Declaration page that came after a form that was already confirmed."


# ------------------------------------------------------------------ attached by hand
def test_a_declaration_page_the_reader_missed_can_be_attached_to_the_form_before_it(
    client: FlaskClient, fake_db: Any, students: Students
) -> None:
    id_page = _queue(client, "Scan.tiff")[0]
    missed = _queue(client, "Scan 1.tiff")[0]
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    # Its heading was not found: read as a page with no Student ID.
    _read(missed, None)

    [doc] = _documents(client)
    assert _ids(doc["pages"]) == [id_page] and doc["attention"] == ["no_declaration"]
    [page] = _batches(client)["unsorted"]
    assert page["page_id"] == missed and page["skip_reason"] == ""
    assert page["attach_to"] == {"page_id": id_page, "student_name": "Aatif Khan"}

    response = client.post(f"/api/pages/{missed}/declaration", json={"of": id_page})

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["page"]["declaration_by"] == OWNER
    attached = body["document"]
    assert attached["key"] == _key(id_page) and _ids(attached["pages"]) == [id_page, missed]
    assert attached["status"] == "ready" and attached["attention"] == [] and attached["auto_upload"] is False
    # Kept like a typed Student ID: who, and when.
    kept = json.loads((scan_queue.page_dir(missed) / "declaration.json").read_text())
    assert kept["of"] == id_page and kept["by"] == OWNER and kept["at"]
    assert _batches(client)["unsorted"] == []
    assert _writes(fake_db) == []

    # Taken off again, it is back under Unsorted, offered as before.
    assert client.post(f"/api/pages/{missed}/declaration", json={"of": ""}).status_code == 200

    [doc] = _documents(client)
    assert _ids(doc["pages"]) == [id_page]
    [page] = _batches(client)["unsorted"]
    assert page["page_id"] == missed and page["declaration_by"] == ""
    assert page["attach_to"] == {"page_id": id_page, "student_name": "Aatif Khan"}


def test_a_page_is_offered_for_the_form_before_it_only_straight_after_one_without_a_declaration_page(
    client: FlaskClient,
) -> None:
    # Straight after a form with a Declaration page, after a blank page, and
    # after the next form's page with no Student ID read: never.
    after_declaration = _form(client, AATIF, "Scan.tiff", "Scan 1.tiff") + [_scan(client, "Scan 2.tiff")]
    _read(after_declaration[2], None)
    _later(RUN_GAP_MINUTES * 60 + 1)
    id_page, blank, after_blank = (_scan(client, f"Scan {n}.tiff") for n in (3, 4, 5))
    _read(id_page, None, "MOHSHA011021", dict(MOHSHA))
    _skipped(blank, "blank")
    _read(after_blank, None)
    _later(RUN_GAP_MINUTES * 60 + 1)
    # No student matched: offered for "the form before it".
    unmatched, unnamed = _scan(client, "Scan 6.tiff"), _scan(client, "Scan 7.tiff")
    _read(unmatched, None, "AATKHA200619", None)
    _read(unnamed, None)

    offers = {page["page_id"]: page["attach_to"] for page in _batches(client)["unsorted"]}

    assert offers[after_declaration[2]] is None
    assert offers[blank] is None and offers[after_blank] is None
    assert offers[unnamed] == {"page_id": unmatched, "student_name": ""}


def test_a_page_straight_after_one_offered_as_a_declaration_page_is_not_offered_too(client: FlaskClient) -> None:
    # Form A's Declaration page with its heading missed, then form B's
    # Student ID page with its ID not read: both are pages with no Student ID.
    id_a, missed, unread_id = (_queue(client, f"Scan {n}.tiff")[0] for n in range(3))
    _read(id_a, None, "AATKHA200618", dict(AATIF))
    _read(missed, None)
    _read(unread_id, None)

    offers = {page["page_id"]: page["attach_to"] for page in _batches(client)["unsorted"]}

    # The first is the likelier Declaration page: only it is offered.
    assert offers == {missed: {"page_id": id_a, "student_name": "Aatif Khan"}, unread_id: None}
    refused = client.post(f"/api/pages/{unread_id}/declaration", json={"of": missed})
    assert refused.status_code == 409
    assert not (scan_queue.page_dir(unread_id) / "declaration.json").exists()
    # Nor is the stack held for them: the missed page stands where a
    # Declaration page would.
    [doc] = _documents(client)
    assert doc["attention"] == ["no_declaration"]


def test_a_page_attached_to_a_form_whose_student_id_was_not_read_waits_with_it_and_can_be_taken_off(
    client: FlaskClient, students: Students
) -> None:
    # A form whose Student ID was not read, and its Declaration page with its
    # heading missed: two pages with no Student ID, the first of its stack.
    unread_id, missed = _queue(client, "Scan.tiff")[0], _queue(client, "Scan 1.tiff")[0]
    _read(unread_id, None)
    _read(missed, None)
    offers = {page["page_id"]: page["attach_to"] for page in _batches(client)["unsorted"]}
    assert offers == {unread_id: None, missed: {"page_id": unread_id, "student_name": ""}}

    assert client.post(f"/api/pages/{missed}/declaration", json={"of": unread_id}).status_code == 200

    # Under Unsorted with it, to be taken off there - or one document with
    # it once its Student ID is typed.
    [waiting] = _batches(client)["unsorted"]
    assert waiting["page_id"] == unread_id and _ids(waiting["declaration_pages"]) == [missed]
    assert waiting["declaration_pages"][0]["declaration_by"] == OWNER
    assert client.post(f"/api/pages/{missed}/declaration", json={"of": ""}).status_code == 200
    offers = {page["page_id"]: page["attach_to"] for page in _batches(client)["unsorted"]}
    assert offers == {unread_id: None, missed: {"page_id": unread_id, "student_name": ""}}
    client.post(f"/api/pages/{missed}/declaration", json={"of": unread_id})

    client.post(f"/api/pages/{unread_id}/student-id", json={"student_id": "AATKHA200618"})

    [doc] = _documents(client)
    assert _ids(doc["pages"]) == [unread_id, missed] and doc["status"] == "ready"
    assert doc["auto_upload"] is False


def test_a_page_attached_by_hand_and_then_given_a_student_id_of_its_own_is_a_form_of_its_own(
    client: FlaskClient, students: Students
) -> None:
    id_page, attached = _queue(client, "Scan.tiff")[0], _queue(client, "Scan 1.tiff")[0]
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _read(attached, None)
    client.post(f"/api/pages/{attached}/declaration", json={"of": id_page})
    students.rows["MOHSHA011021"] = dict(MOHSHA)

    # Another tab still showed it under Unsorted: a Student ID is typed on it.
    client.post(f"/api/pages/{attached}/student-id", json={"student_id": "MOHSHA011021"})

    docs = {d["key"]: d for d in _documents(client)}

    # Two Student ID pages in a row, as the person now says - not a
    # Declaration page without its Student ID page.
    assert _ids(docs[_key(id_page)]["pages"]) == [id_page]
    assert docs[_key(id_page)]["attention"] == ["no_declaration", "gap_in_stack"]
    assert _ids(docs[_key(attached)]["pages"]) == [attached]
    assert "orphan_in_stack" not in docs[_key(attached)]["attention"]
    assert docs[_key(attached)]["pages"][0]["declaration_by"] == ""

    # Taken off again, it is the form's Declaration page once more, as it was attached.
    client.post(f"/api/pages/{attached}/student-id", json={"student_id": ""})

    [doc] = _documents(client)
    assert _ids(doc["pages"]) == [id_page, attached] and doc["attention"] == []


def test_a_page_attached_to_one_that_no_longer_begins_a_form_is_a_page_of_its_own_again(
    client: FlaskClient, students: Students
) -> None:
    id_page, blank, missed = (_queue(client, f"Scan {n}.tiff")[0] for n in range(3))
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _skipped(blank, "blank")
    _read(missed, None)
    # The blank page was the next form's Student ID page after all, and the
    # missed page its Declaration page.
    students.rows["MOHSHA011021"] = dict(MOHSHA)
    client.post(f"/api/pages/{blank}/student-id", json={"student_id": "MOHSHA011021"})
    assert client.post(f"/api/pages/{missed}/declaration", json={"of": blank}).status_code == 200

    # The typed ID is taken off: the page is blank again.
    client.post(f"/api/pages/{blank}/student-id", json={"student_id": ""})

    # The attached page does not join the form before the blank page.
    [doc] = _documents(client)
    assert _ids(doc["pages"]) == [id_page]
    pages = {page["page_id"]: page for page in _batches(client)["unsorted"]}
    assert set(pages) == {blank, missed}
    assert pages[missed]["declaration_by"] == "" and pages[missed]["attach_to"] is None


def test_attaching_is_refused_once_the_page_is_no_longer_straight_after_a_form_without_one(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    id_page = _queue(google, "Scan.tiff")[0]
    missed = _queue(google, "Scan 1.tiff")[0]
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _read(missed, None)
    assert _batches(google)["unsorted"][0]["attach_to"]["page_id"] == id_page
    # In another tab the form was confirmed without it.
    assert _confirm(google, key=_key(id_page)).status_code == 200

    stale = google.post(f"/api/pages/{missed}/declaration", json={"of": id_page})
    wrong = google.post(f"/api/pages/{missed}/declaration", json={"of": missed})
    unnamed = google.post(f"/api/pages/{missed}/declaration", json={"of": "not-a-page"})

    assert stale.status_code == 409 and stale.get_json()["error"] == (
        "This page is no longer straight after a form waiting for its Declaration page, so it was "
        "not attached. Nothing was saved - reload the page."
    )
    assert wrong.status_code == 409
    assert unnamed.status_code == 400
    assert not (scan_queue.page_dir(missed) / "declaration.json").exists()
    [page] = _batches(google)["unsorted"]
    assert page["page_id"] == missed and page["attach_to"] is None
    assert len(drive.uploaded()) == 1


def test_attaching_is_refused_while_its_form_is_being_confirmed(client: FlaskClient) -> None:
    id_page = _queue(client, "Scan.tiff")[0]
    missed = _queue(client, "Scan 1.tiff")[0]
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _read(missed, None)

    with scan_queue.confirm_lock(OWNER, _key(id_page)):
        busy = client.post(f"/api/pages/{missed}/declaration", json={"of": id_page})

    assert busy.status_code == 409 and "being confirmed right now" in busy.get_json()["error"]
    assert not (scan_queue.page_dir(missed) / "declaration.json").exists()


def test_the_pdf_of_a_form_with_an_attached_declaration_page_holds_both_pages_in_order(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    id_page = _queue(google, "Scan.tiff")[0]
    missed = _queue(google, "Scan 1.tiff")[0]
    _read(id_page, None, "AATKHA200618", dict(AATIF))
    _read(missed, None)
    google.post(f"/api/pages/{missed}/declaration", json={"of": id_page})

    response = _confirm(google, key=_key(id_page))

    assert response.status_code == 200, response.get_json()
    [pdf_file] = drive.uploaded()
    with fitz.open(stream=pdf_file["data"], filetype="pdf") as pdf:
        images = [pdf.extract_image(page.get_images()[0][0])["image"] for page in pdf]
    assert images == [_image_of(id_page), _image_of(missed)]


# ------------------------------------------------------------------ typing an ID
def test_a_form_whose_student_id_was_not_read_becomes_a_document_once_its_id_is_typed(
    client: FlaskClient, fake_db: Any, students: Students
) -> None:
    [page_id] = _queue(client)
    declaration = _queue(client, "Scan 2.tiff")[0]
    _read(page_id, None, "", None)
    _skipped(declaration)
    [waiting] = _batches(client)["unsorted"]
    assert waiting["page_id"] == page_id and _ids(waiting["declaration_pages"]) == [declaration]

    response = client.post(f"/api/pages/{page_id}/student-id", json={"student_id": "aatkha200618"})

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["page"]["student_id"] == "AATKHA200618" and body["page"]["student_id_source"] == "typed"
    assert body["document"]["key"] == _key(page_id)
    assert _ids(body["document"]["pages"]) == [page_id, declaration]
    assert body["document"]["status"] == "ready" and body["document"]["student_id_source"] == "typed"
    assert _batches(client)["unsorted"] == []
    assert _writes(fake_db) == []

    # Cleared, it goes back to waiting for one, its Declaration page still with it.
    client.post(f"/api/pages/{page_id}/student-id", json={"student_id": ""})
    assert _documents(client) == []
    [waiting] = _batches(client)["unsorted"]
    assert waiting["page_id"] == page_id and _ids(waiting["declaration_pages"]) == [declaration]


def test_a_failed_page_waits_under_unsorted_until_its_id_is_typed(
    client: FlaskClient, students: Students
) -> None:
    [page_id] = _queue(client)
    (scan_queue.queue_root() / "_pending" / page_id).unlink()
    meta = scan_queue.load_meta(page_id)
    assert meta is not None
    meta.status, meta.attempts, meta.error = scan_queue.STATUS_FAILED, 3, "MemoryError: out of memory"
    scan_queue.save_meta(meta)
    _later(DECLARATION_WAIT_SECONDS + 1)

    [waiting] = _batches(client)["unsorted"]
    assert waiting["status"] == "failed"
    assert waiting["message"] == "Could not read this page after 3 tries. Type its Student ID from the image."
    assert _documents(client) == []

    client.post(f"/api/pages/{page_id}/student-id", json={"student_id": "AATKHA200618"})

    [doc] = _documents(client)
    assert doc["key"] == _key(page_id) and doc["status"] == "failed" and doc["can_confirm"] is True
    assert doc["pages"][0]["needs_person"] is True


@pytest.mark.parametrize(
    ("reason", "attention"),
    [
        ("blank", ["no_declaration"]),
        # A page the reader took for a Declaration page that is a form of its
        # own after all: its stack is held, in case it was one.
        ("declaration", ["no_declaration", "orphan_in_stack"]),
        ("empty_id", ["no_declaration"]),
    ],
)
def test_a_skipped_page_waits_under_unsorted_until_its_id_is_typed(
    client: FlaskClient, students: Students, reason: str, attention: list[str]
) -> None:
    [page_id] = _queue(client)
    _skipped(page_id, reason, code="B1-1")
    _later(DECLARATION_WAIT_SECONDS + 1)

    [page] = _batches(client)["unsorted"]
    assert page["page_id"] == page_id and page["skip_reason"] == reason
    assert _documents(client) == []

    # The skip may have been wrong: an ID typed from the scan is not ignored.
    response = client.post(f"/api/pages/{page_id}/student-id", json={"student_id": "AATKHA200618"})

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["document"]["key"] == _key(page_id)
    [doc] = _documents(client)
    assert doc["key"] == _key(page_id) and doc["student_id_source"] == "typed"
    # Nothing came after it, so no Declaration page either.
    assert doc["status"] == "needs_attention" and doc["attention"] == attention
    assert _batches(client)["unsorted"] == []


def test_a_typed_code_is_refused(client: FlaskClient) -> None:
    [page_id] = _queue(client)
    _read(page_id, None)

    response = client.post(f"/api/pages/{page_id}/code", json={"code": "B1-1"})

    assert response.status_code == 409
    assert "Batch codes are switched off" in response.get_json()["error"]
    assert scan_queue.load_overrides(page_id).code is None


# ================================================================== confirm
def test_confirm_marks_the_student_and_uploads_one_pdf_of_both_pages_named_without_a_code(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id, declaration] = _form(google, AATIF, "Scan 7.tiff", "Scan 8.tiff")

    response = _confirm(google, key=_key(page_id))

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["uploaded"] is True and body["finished"] is True
    doc = body["document"]
    assert re.fullmatch(rf"P{page_id}~[0-9a-f]{{8}}", doc["key"])
    assert doc["status"] == "uploaded" and doc["code"] == "" and doc["parts"] == 2
    assert doc["drive_folder"] == "OCR/ABMPS/Grade 5/A/Aatif Khan_AATKHA200618"
    assert doc["drive_name"] == "Aatif Khan_AATKHA200618.pdf"

    # One PDF: the Student ID page first, its Declaration page after it,
    # each page.jpg as it is.
    [pdf_file] = drive.uploaded()
    assert pdf_file["name"] == "Aatif Khan_AATKHA200618.pdf"
    assert drive.path_of(pdf_file["id"]) == ["OCR", "ABMPS", "Grade 5", "A", "Aatif Khan_AATKHA200618"]
    with fitz.open(stream=pdf_file["data"], filetype="pdf") as pdf:
        assert pdf.page_count == 2
        images = [pdf.extract_image(page.get_images()[0][0])["image"] for page in pdf]
    assert images == [_image_of(page_id), _image_of(declaration)]
    record = scan_queue.load_document_record(OWNER, doc["key"])
    assert record is not None and record["page_ids"] == [page_id, declaration]

    [update] = fake_db.executed(r"^UPDATE active_student_data SET scanned_at")
    assert update[1][1] == "AATKHA200618"
    [insert] = fake_db.executed(r"^INSERT INTO ocr_scan_log")
    assert insert[1][1] == "AATKHA200618" and insert[1][2] == 1
    assert insert[1][6] == "scan:Scan 7.tiff"
    assert insert[1][8] == body["drive_link"]

    # Listed as uploaded under its confirmed key; nothing left to confirm.
    assert scan_queue.open_page_ids(OWNER) == []
    [listed] = _documents(google)
    assert listed["key"] == doc["key"] and listed["status"] == "uploaded"
    assert _ids(listed["pages"]) == [page_id, declaration]
    assert _batches(google)["unsorted"] == []
    assert google.get(f"/api/documents/{doc['key']}").get_json()["document"]["status"] == "uploaded"
    assert _confirm(google, key=_key(page_id), page_ids=[page_id, declaration]).status_code == 404
    assert len(fake_db.executed(r"^UPDATE active_student_data")) == 1


def test_a_confirm_shown_only_the_student_id_page_is_refused_once_its_declaration_page_joins(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google, "Scan.tiff")
    _read(page_id, None, "AATKHA200618", dict(AATIF))
    # Nothing came after it in time: the page shows it on its own, to be
    # checked, and Confirm may be pressed.
    _later(DECLARATION_WAIT_SECONDS + 1)
    looked = google.get(f"/api/documents/{_key(page_id)}").get_json()["document"]
    shown = _ids(looked["pages"])
    assert shown == [page_id] and looked["can_confirm"] is True

    # The scanner sends the form's Declaration page before Confirm is pressed.
    declaration = _queue(google, "Scan 1.tiff")[0]
    while_read = _confirm(google, key=_key(page_id), page_ids=shown)
    _skipped(declaration)
    once_read = _confirm(google, key=_key(page_id), page_ids=shown)

    assert while_read.status_code == 409 and "still being read" in while_read.get_json()["error"]
    assert once_read.status_code == 409 and once_read.get_json()["error"] == documents.CHANGED_MESSAGE
    assert _writes(fake_db) == [] and drive.calls == []
    assert scan_queue.open_page_ids(OWNER) == sorted([page_id, declaration])
    # Shown both, it is confirmed as one.
    again = _confirm(google, key=_key(page_id))
    assert again.status_code == 200, again.get_json()
    assert again.get_json()["document"]["parts"] == 2


def test_confirm_saves_only_the_page_it_was_shown(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google)
    other = _queue(google, "Scan 2.tiff")[0]
    _read(page_id, None, "AATKHA200618", dict(AATIF))
    _read(other, None, "AATKHA200618", dict(AATIF))

    response = _confirm(google, key=_key(page_id), page_ids=[page_id, other])

    assert response.status_code == 409
    assert _writes(fake_db) == [] and drive.calls == []


def test_a_drive_failure_leaves_it_confirmed_and_the_retry_uploads(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id, _declaration] = _form(google, AATIF)
    drive.fail_uploads = True

    body = _confirm(google, key=_key(page_id)).get_json()

    doc = body["document"]
    assert body["uploaded"] is False and doc["status"] == "confirmed" and doc["can_retry_upload"] is True
    [listed] = _documents(google)
    assert listed["status"] == "confirmed"

    drive.fail_uploads = False
    retried = google.post(f"/api/documents/{doc['key']}/upload")

    assert retried.status_code == 200, retried.get_json()
    assert retried.get_json()["document"]["status"] == "uploaded"
    assert [f["name"] for f in drive.uploaded()] == ["Aatif Khan_AATKHA200618.pdf"]
    assert len(fake_db.executed(r"^UPDATE active_student_data")) == 1
    assert len(fake_db.executed(r"^INSERT INTO ocr_scan_log")) == 1


def test_a_student_confirmed_lately_is_flagged_when_scanned_again(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    [first, _declaration] = _form(google, AATIF, "Scan 1.tiff", "Scan 2.tiff")
    _confirm(google, key=_key(first))
    [again, _again_declaration] = _form(google, AATIF, "Scan 3.tiff", "Scan 4.tiff")
    [someone_else, _other_declaration] = _form(google, MOHSHA, "Scan 5.tiff", "Scan 6.tiff")

    docs = {d["key"]: d for d in _documents(google)}

    flagged = docs[_key(again)]
    assert flagged["status"] == "needs_attention" and flagged["attention"] == ["same_student"]
    assert flagged["can_confirm"] is True
    assert flagged["message"] == (
        "A form for Aatif Khan (AATKHA200618) was already confirmed. "
        "Check the paper - this may be the same form scanned twice."
    )
    assert docs[_key(someone_else)]["status"] == "ready"


def test_a_failed_page_still_says_the_student_was_confirmed_lately(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    [first, _declaration] = _form(google, AATIF, "Scan 1.tiff", "Scan 2.tiff")
    _confirm(google, key=_key(first))
    [again] = _queue(google, "Scan 3.tiff")
    _skipped(_queue(google, "Scan 4.tiff")[0])
    (scan_queue.queue_root() / "_pending" / again).unlink()
    meta = scan_queue.load_meta(again)
    assert meta is not None
    meta.status, meta.attempts, meta.error = scan_queue.STATUS_FAILED, 3, "MemoryError: out of memory"
    scan_queue.save_meta(meta)
    google.post(f"/api/pages/{again}/student-id", json={"student_id": "AATKHA200618"})

    flagged = next(d for d in _documents(google) if d["key"] == _key(again))

    assert flagged["status"] == "failed" and flagged["attention"] == ["same_student"]
    assert flagged["parts"] == 2
    assert flagged["message"] == (
        "This page could not be read. Check the Student ID against its image. "
        "A form for Aatif Khan (AATKHA200618) was already confirmed. "
        "Check the paper - this may be the same form scanned twice."
    )


def test_pages_read_with_codes_are_one_page_documents_and_coded_records_still_show(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Read and confirmed while batch codes were on.
    monkeypatch.setattr(config, "BATCH_CODES", True)
    _ready_document(google)
    confirmed = _confirm(google, key="B1-1").get_json()["document"]
    id_page = _queue(google, "Scan 3.tiff")[0]
    signature_page = _queue(google, "Scan 4.tiff")[0]
    _read(id_page, "B1-2", "AATKHA200618", dict(AATIF))
    _read(signature_page, "B1-2")
    monkeypatch.setattr(config, "BATCH_CODES", False)

    body = _batches(google)
    docs = {d["key"]: d for d in _documents(google)}

    # The coded document confirmed before still shows, both its parts.
    old = docs[confirmed["key"]]
    assert old["code"] == "B1-1" and old["parts"] == 2 and old["status"] == "uploaded"
    # The ID page is a document of its own; the signature page, not read as
    # a Declaration page, is not attached. Its student is the one confirmed
    # with a code, so it is flagged.
    new = docs[_key(id_page)]
    assert new["parts"] == 1 and new["code"] == ""
    assert new["attention"] == ["same_student", "no_declaration"]
    assert [p["page_id"] for p in body["unsorted"]] == [signature_page]
    assert set(docs) == {confirmed["key"], _key(id_page)}

    response = _confirm(google, key=_key(id_page))

    assert response.status_code == 200, response.get_json()
    assert sorted(f["name"] for f in drive.uploaded()) == [
        "Aatif Khan_AATKHA200618.pdf", "Aatif Khan_AATKHA200618_B1-1.pdf",
    ]


def test_a_one_page_record_is_finished_after_codes_come_back(
    google: FlaskClient, students: Students, drive: FakeDrive, monkeypatch: pytest.MonkeyPatch
) -> None:
    [page_id] = _queue(google)
    _read(page_id, None, "AATKHA200618", dict(AATIF))
    _later(DECLARATION_WAIT_SECONDS + 1)
    drive.fail_uploads = True
    key = _confirm(google, key=_key(page_id)).get_json()["document"]["key"]
    monkeypatch.setattr(config, "BATCH_CODES", True)
    drive.fail_uploads = False

    _ready_document(google)

    # Listed apart from the batches, with no batch or label - never "B0".
    groups = _batches(google)["batches"]
    assert [(g["batch"], g["label"], [d["key"] for d in g["documents"]]) for g in groups] == [
        (1, "B1", ["B1-1"]), (None, "", [key]),
    ]

    retried = google.post(f"/api/documents/{key}/upload")

    assert retried.status_code == 200, retried.get_json()
    assert retried.get_json()["document"]["drive_name"] == "Aatif Khan_AATKHA200618.pdf"


def test_with_batch_codes_on_pages_are_sorted_by_their_code_as_before(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    # A Declaration page first, and two scan pages: none of it matters with codes.
    stray = _scan(client, "Scan.tiff", ONE)
    id_page = _scan(client, "Scan 1.tiff", TWO)
    signature = _scan(client, "Scan 2.tiff", ONE)
    _skipped(stray, code="B1-1")
    _read(id_page, "B1-1", "AATKHA200618", dict(AATIF))
    _read(signature, "B1-1")
    _read(_scan(client, "Scan 3.tiff", ONE), None)

    body = _batches(client)
    [group] = body["batches"]
    [doc] = group["documents"]

    assert doc["key"] == "B1-1" and _ids(doc["pages"]) == [stray, id_page, signature]
    assert doc["status"] == "ready" and doc["attention"] == []
    # The page is told nothing of Declaration pages, senders or attaching.
    for page in doc["pages"] + body["unsorted"]:
        assert not {"skip_reason", "declaration_by", "declaration_pages", "attach_to", "sender"} & set(page)
    refused = client.post(f"/api/pages/{signature}/declaration", json={"of": id_page})
    assert refused.status_code == 409 and "Batch codes are switched on" in refused.get_json()["error"]


# ================================================================== the page
def _outside_scripts(html: str) -> str:
    """The page as a person reads it: no script, no HTML comments."""
    return re.sub(r"<script>.*?</script>|<!--.*?-->", "", html, flags=re.DOTALL)


def test_page_asks_for_no_batch_codes(client: FlaskClient) -> None:
    html = client.get("/").get_data(as_text=True)
    text = _outside_scripts(html)

    for gone in ("B1-1", "batch code", "Batch code", "top-right corner", "Code each form", "or code can"):
        assert gone not in text, gone
    said = " ".join(text.split())
    for told in (
        # The top of the page, the switch, the note over the results and How it works.
        (
            "each form - its Student ID page and the Declaration page scanned straight after it - is one"
            " document, one PDF"
        ),
        (
            "is confirmed and uploaded by itself once its whole stack has come in clean - each form's Student ID"
            " page, then its Declaration page, nothing out of place or corrected by hand - and nothing more has"
            " come for 10 minutes."
        ),
        (
            "a document whose Student ID - as read - is exactly one of that school's students is confirmed and"
            " uploaded to your Drive without pressing Confirm - but only once its whole stack, as this computer"
            " sent it, is in: Student ID page, Declaration page, by turns from first to last, every page read,"
            " and nothing more from this computer or any other for that school for 10 minutes - and only while"
            " this page is open on screen, as it is this page that does it. Each Ready card says when it goes"
            " up, or why it waits."
        ),
        (
            "a Student ID page with no Declaration page straight after it, a stack with a page out of place,"
            " blank or unreadable, a Student ID you typed or a Declaration page you attached or took off, files"
            " added by hand in among what the scan folder sent, another computer or tab scanning for the same"
            " school at the same time, a stack for a school in the 12 hours after one of its stacks had a page"
            " out of place, or anything that needs checking still waits for you."
        ),
        "go to Drive as one PDF, the Student ID page first",
        (
            "A Declaration page with no Student ID page right before it waits there on its own, and the forms"
            " scanned with it wait for you to check them - as they do after a Student ID page with no"
            " Declaration page after it;"
        ),
        (
            "in a stack that came in whole and clean, Student ID page and Declaration page by turns with nothing"
            " out of place, nothing corrected by hand and no other computer scanning for the school, and has had"
            " nothing more for 10 minutes - is confirmed and uploaded by itself while this page is open on screen,"
            " and says so"
        ),
        "a Declaration page the reader missed can be attached there to the form straight before it.",
        "Scan each form's Student ID page and, straight after it, its Declaration page: the two are one document.",
        "its Student ID page first and its Declaration page after it",
    ):
        assert told in said, told
    assert "No Student ID was read - type it from the scan" in text
    assert "const BATCH_CODES = false;" in html


def test_page_asks_for_batch_codes_when_they_are_on(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)

    html = client.get("/").get_data(as_text=True)
    text = _outside_scripts(html)

    assert "Code each form, then scan" in text
    assert "Declaration page" not in text
    assert "const BATCH_CODES = true;" in html


# What /api/batches answers without batch codes: one group, no batch.
_OFF_RESULTS_JS = r"""
  function offDocs(documents, unsorted = [], queue = { queued: 0, reading: 0, failed: 0 }) {
    const pages = [...unsorted];
    documents.forEach((d) => pages.push(...d.pages));
    const groups = documents.length ? [{ batch: null, label: "", documents, counts: {} }] : [];
    return { batches: groups, unsorted, pages, queue };
  }
  function pdoc(page, extra) {
    return Object.assign({ key: "P" + page.page_id, code: "", parts: 1, pages: [page], student_id: "", matched_student: null, status: "ready" }, extra || {});
  }
"""


@needs_node
def test_page_without_codes_lists_one_page_documents_and_types_ids_under_unsorted(
    client: FlaskClient, tmp_path: Path
) -> None:
    """One list with no batch headings; no code box anywhere; Unsorted takes the Student ID."""
    out = _run_page(client, tmp_path, _OFF_RESULTS_JS + r"""
      const page = await openPage();
      const asha = pg("p1", { source_name: "Scan 001.jpg", student_id: "MOHSHA011021", matched_student: ASHA });
      const misread = pg("p2", { source_name: "stack.pdf", source_page: 2, source_pages: 3, student_id: "AATKHA200619" });
      const loose = pg("p3", { source_name: "Scan 003.jpg", message: "No Student ID found on this page." });
      const declaration = pg("p4", { source_name: "Scan 004.jpg", skip_reason: "declaration", message: "Declaration page, no Student ID." });
      const blank = pg("p5", { source_name: "Scan 005.jpg", skip_reason: "blank", blank: true, message: "This page looks blank." });
      await answerBatches(page, offDocs([
        pdoc(asha, { student_id: "MOHSHA011021", matched_student: ASHA }),
        pdoc(misread, { status: "needs_attention", attention: ["no_match"], student_id: "AATKHA200619",
          message: "No student with ID AATKHA200619 at School A. Check the ID against the paper." }),
      ], [loose, declaration, blank]));
      const all = cards(page);
      const headings = page.byId["ocr-batches"]._all().filter((e) => e.classList.contains("ocr-batch-title"));
      const ready = card(page, "Scan 001.jpg");
      const check = card(page, "stack.pdf - page 2");
      const [looseCard, declarationCard, blankCard] = unsortedCards(page);
      const shown = {
        cards: all.length, headings: headings.map((e) => e.textContent),
        codeFields: all.filter((c) => fieldIn(c, "code") || c._all().some((e) => e.dataset && String(e.dataset.field || "").startsWith("code-"))).length,
        setCode: all.filter((c) => buttonIn(c, "Set code")).length,
        looseText: textIn(looseCard), looseId: Boolean(fieldIn(looseCard, "id")),
        declarationId: Boolean(fieldIn(declarationCard, "id")), declarationText: textIn(declarationCard),
        blankId: Boolean(fieldIn(blankCard, "id")), blankText: textIn(blankCard),
        checkScans: bigScans(check).length, status: page.byId["ocr-status"].textContent,
        everything: all.map(textIn).join(" | "),
      };
      fieldIn(looseCard, "id").value = "mohsha011021";
      buttonIn(looseCard, "Check ID").fire("click");
      const typed = await answer(page.calls, "/api/pages/p3/student-id", { page: {}, document: {} });
      await tick(5);
      const said = textIn(looseCard);
      buttonIn(ready, "Confirm and upload").fire("click");
      const confirm = await answer(page.calls, "/confirm", { document: { status: "uploaded", key: "Pp1~ab12cd34" } });
      return { shown, typed: JSON.parse(typed.opts.body), said, confirmUrl: confirm.url, confirmBody: JSON.parse(confirm.opts.body) };
    """)

    shown = out["shown"]
    assert shown["cards"] == 5
    assert shown["headings"] == []
    assert shown["codeFields"] == 0 and shown["setCode"] == 0
    assert "B1" not in shown["everything"] and "batch code" not in shown["everything"].lower()
    assert "No Student ID" in shown["looseText"] and "Needs a Student ID" in shown["looseText"]
    assert shown["looseId"] is True
    # A declaration page says it was skipped, but takes an ID in case the
    # skip was wrong; a blank page only shows its scan.
    assert shown["declarationId"] is True and "Declaration page" in shown["declarationText"]
    assert shown["blankId"] is False and "Looks blank" in shown["blankText"]
    assert shown["checkScans"] == 1
    assert "3 pages without a Student ID" in shown["status"]
    assert out["typed"] == {"student_id": "MOHSHA011021"}
    assert "becomes a document of its own" in out["said"]
    assert out["confirmUrl"] == "/api/documents/Pp1/confirm"
    assert out["confirmBody"] == {"student_id": "MOHSHA011021", "page_ids": ["p1"]}


@needs_node
def test_page_shows_each_form_on_one_card_with_its_declaration_page(client: FlaskClient, tmp_path: Path) -> None:
    """Both sheets on the document's card, and on the Unsorted card of a form waiting for its Student ID."""
    out = _run_page(client, tmp_path, _OFF_RESULTS_JS + r"""
      const page = await openPage();
      const idPage = pg("p1", { source_name: "Scan.tiff", student_id: "MOHSHA011021", matched_student: ASHA, skip_reason: "" });
      const declaration = pg("p2", { source_name: "Scan 1.tiff", skip_reason: "declaration", message: "Declaration page, no Student ID." });
      const misread = pg("p3", { source_name: "Scan 2.tiff", student_id: "AATKHA200619", skip_reason: "" });
      const itsDeclaration = pg("p4", { source_name: "Scan 3.tiff", skip_reason: "declaration" });
      const loose = pg("p5", { source_name: "Scan 4.tiff", skip_reason: "", message: "No Student ID found on this page.",
        declaration_pages: [pg("p6", { source_name: "Scan 5.tiff", skip_reason: "declaration" })] });
      await answerBatches(page, offDocs([
        pdoc(idPage, { parts: 2, pages: [idPage, declaration], student_id: "MOHSHA011021", matched_student: ASHA }),
        pdoc(misread, { parts: 2, pages: [misread, itsDeclaration], status: "needs_attention", attention: ["no_match"],
          student_id: "AATKHA200619", message: "No student with ID AATKHA200619 at School A. Check the ID against the paper." }),
      ], [loose]));
      const ready = card(page, "Scan.tiff");
      const check = card(page, "Scan 2.tiff");
      const [waiting] = unsortedCards(page);
      const parts = (c) => c._all().find((e) => e.classList.contains("ocr-doc-parts")).textContent;
      const labels = (c, name) => c._all().filter((e) => e.classList.contains(name)).map((e) => e.textContent);
      const shown = {
        cards: cards(page).length, unsorted: unsortedCards(page).length,
        readyParts: parts(ready), readyThumbs: labels(ready, "ocr-thumb-label"), readyImages: imagesIn(ready),
        checkParts: parts(check), checkScans: labels(check, "ocr-scan-caption"),
        waitingParts: parts(waiting), waitingThumbs: labels(waiting, "ocr-thumb-label"),
        waitingImages: imagesIn(waiting), waitingText: textIn(waiting),
      };
      // An ID typed on a form's card goes on its Student ID page only.
      fieldIn(check, "id").value = "MOHSHA011021";
      buttonIn(check, "Check ID").fire("click");
      await answer(page.calls, "/student-id", { page: {}, document: {} });
      await tick(20);
      const lookups = page.log.filter((c) => c.url.includes("/student-id")).map((c) => c.url);
      // Under Unsorted, it goes on the form's first page.
      fieldIn(waiting, "id").value = "mohsha011021";
      buttonIn(waiting, "Check ID").fire("click");
      const typed = await answer(page.calls, "/student-id", { page: {}, document: {} });
      await tick(5);
      const said = textIn(waiting);
      buttonIn(ready, "Confirm and upload").fire("click");
      const confirm = await answer(page.calls, "/confirm", { document: { status: "uploaded", key: "Pp1~ab12cd34" } });
      return { shown, lookups, typedUrl: typed.url, said, confirmUrl: confirm.url, confirmBody: JSON.parse(confirm.opts.body) };
    """)

    shown = out["shown"]
    # Two forms and one waiting for its ID; no Declaration page has a card of its own.
    assert shown["cards"] == 3 and shown["unsorted"] == 1
    assert shown["readyParts"] == "Student ID page + Declaration page"
    assert shown["readyThumbs"] == ["Student ID page", "Declaration page"]
    assert shown["readyImages"] == ["/api/pages/p1/image?size=thumb", "/api/pages/p2/image?size=thumb"]
    assert shown["checkParts"] == "Student ID page + Declaration page"
    assert shown["checkScans"] == ["Scan 2.tiff - Student ID page", "Scan 3.tiff - Declaration page"]
    assert shown["waitingParts"] == "Scan 4.tiff - page 1 + Declaration page"
    assert shown["waitingThumbs"] == ["Student ID page", "Declaration page"]
    assert "/api/pages/p6/image?size=thumb" in shown["waitingImages"]
    assert "once the Student ID is typed, both are one document" in shown["waitingText"]
    assert out["lookups"] == ["/api/pages/p3/student-id"]
    assert out["typedUrl"] == "/api/pages/p5/student-id"
    assert "the page and its Declaration page become one document" in out["said"]
    assert out["confirmUrl"] == "/api/documents/Pp1/confirm"
    assert out["confirmBody"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p2"]}


@needs_node
def test_page_shows_what_waits_what_is_held_and_what_is_out_of_place(client: FlaskClient, tmp_path: Path) -> None:
    """Waiting forms offer no Confirm; Unsorted says why a Declaration page is alone and offers to attach a missed one."""
    out = _run_page(client, tmp_path, _OFF_RESULTS_JS + r"""
      const RAVI = { student_id: "RAVKUM020315", student_name: "Ravi Kumar", school_name: "School A", grade_name: "5", division_name: "B" };
      const page = await openPage();
      const waiting = pdoc(pg("p1", { source_name: "Scan 1.tiff", student_id: ASHA.student_id, matched_student: ASHA, skip_reason: "" }),
        { status: "waiting", can_confirm: false, student_id: ASHA.student_id, matched_student: ASHA,
          message: "Waiting for its Declaration page - nothing has come after it yet." });
      const heldPages = [pg("p2", { source_name: "Scan 2.tiff", student_id: RAVI.student_id, matched_student: RAVI, skip_reason: "" }),
        pg("p3", { source_name: "Scan 3.tiff", skip_reason: "declaration" })];
      const held = pdoc(heldPages[0], { parts: 2, pages: heldPages, status: "needs_attention", attention: ["orphan_in_stack"],
        student_id: RAVI.student_id, matched_student: RAVI,
        message: "This stack has a Declaration page without its Student ID page, so its forms may be paired wrongly - check them." });
      const attachedPages = [pg("p4", { source_name: "Scan 4.tiff", student_id: ASHA.student_id, matched_student: ASHA, skip_reason: "" }),
        pg("p5", { source_name: "Scan 5.tiff", skip_reason: "", declaration_by: "staff@akanksha.org" })];
      const attached = pdoc(attachedPages[0], { parts: 2, pages: attachedPages, student_id: ASHA.student_id, matched_student: ASHA });
      // Confirmed with a code before the switch: its signature page was not read as a Declaration page.
      const codedPages = [pg("p6", { source_name: "Scan 6.tiff", code: "B1-1", student_id: RAVI.student_id, matched_student: RAVI }),
        pg("p7", { source_name: "Scan 7.tiff", code: "B1-1" })];
      const coded = { key: "B1-1~ab12cd34", code: "B1-1", parts: 2, pages: codedPages, student_id: RAVI.student_id,
        matched_student: RAVI, status: "uploaded" };
      const orphan = pg("p8", { source_name: "Scan 8.tiff", skip_reason: "declaration",
        message: "A Declaration page with no Student ID page right before it." });
      const missed = pg("p9", { source_name: "Scan 9.tiff", skip_reason: "", message: "No Student ID found on this page.",
        attach_to: { page_id: "p4", student_name: "Asha Patil" } });
      const unnamed = pg("p10", { source_name: "Scan 10.tiff", skip_reason: "", attach_to: { page_id: "p11", student_name: "" } });
      await answerBatches(page, offDocs([waiting, held, attached, coded], [orphan, missed, unnamed]));
      const [orphanCard, missedCard, unnamedCard] = unsortedCards(page);
      const thumbs = (c) => c._all().filter((e) => e.classList.contains("ocr-thumb")).map((e) => e.attrs["aria-label"]);
      const parts = (c) => c._all().find((e) => e.classList.contains("ocr-doc-parts")).textContent;
      const pill = (c) => c._all().find((e) => e.classList.contains("ocr-pill")).textContent;
      const shown = {
        status: page.byId["ocr-status"].textContent,
        waitingPill: pill(card(page, "Scan 1.tiff")), waiting: textIn(card(page, "Scan 1.tiff")),
        waitingConfirm: Boolean(buttonIn(card(page, "Scan 1.tiff"), "Confirm and upload")),
        held: textIn(card(page, "Scan 2.tiff")), heldConfirm: Boolean(buttonIn(card(page, "Scan 2.tiff"), "Confirm and upload")),
        attachedParts: parts(card(page, "Scan 4.tiff")), attachedThumbs: thumbs(card(page, "Scan 4.tiff")),
        codedParts: parts(card(page, "Scan 6.tiff")), codedThumbs: thumbs(card(page, "Scan 6.tiff")),
        codedLabels: card(page, "Scan 6.tiff")._all().filter((e) => e.classList.contains("ocr-thumb-label")).map((e) => e.textContent),
        orphan: textIn(orphanCard), orphanButtons: orphanCard._all().filter((e) => e.tagName === "BUTTON").map((e) => e.textContent),
        orphanId: Boolean(fieldIn(orphanCard, "id")),
        unnamed: Boolean(buttonIn(unnamedCard, "Attach as the Declaration page of the form before it")),
      };
      buttonIn(missedCard, "Attach as the Declaration page of Asha Patil").fire("click");
      const attach = await answer(page.calls, "/api/pages/p9/declaration", { page: {}, document: {} });
      await tick(5);
      const said = textIn(missedCard);
      buttonIn(card(page, "Scan 4.tiff"), "Take off the Declaration page").fire("click");
      const detach = await answer(page.calls, "/api/pages/p5/declaration", { page: {}, document: {} });
      return { shown, attach: JSON.parse(attach.opts.body), said, detach: JSON.parse(detach.opts.body) };
    """)

    shown = out["shown"]
    assert "1 document waiting for a Declaration page" in shown["status"]
    assert shown["waitingPill"] == "Waiting for its Declaration page" and shown["waitingConfirm"] is False
    assert "nothing has come after it yet" in shown["waiting"]
    assert "may be paired wrongly - check them" in shown["held"] and shown["heldConfirm"] is True
    assert shown["attachedParts"] == "Student ID page + Declaration page"
    assert shown["attachedThumbs"] == [
        "Open the Student ID page of Scan 4.tiff full size", "Open the Declaration page of Scan 4.tiff full size",
    ]
    # A page that is not a Declaration page is never called one.
    assert shown["codedParts"] == "2 parts" and shown["codedLabels"] == ["Part 1", "Part 2"]
    assert shown["codedThumbs"] == ["Open part 1 of Scan 6.tiff full size", "Open part 2 of Scan 6.tiff full size"]
    assert "A Declaration page with no Student ID page right before it." in shown["orphan"]
    assert "Check ID" in shown["orphanButtons"] and shown["orphanId"] is True
    assert not [text for text in shown["orphanButtons"] if text.startswith("Attach")]
    assert shown["unnamed"] is True
    assert out["attach"] == {"of": "p4"}
    assert "the two become one document in a moment, for you to confirm" in out["said"]
    assert out["detach"] == {"of": ""}


@needs_node
def test_page_shows_a_declaration_page_attached_under_unsorted_and_takes_it_off_there(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Attached to a page whose Student ID is still to be typed: said so, shown with it, and taken off from its card."""
    out = _run_page(client, tmp_path, _OFF_RESULTS_JS + r"""
      const page = await openPage();
      const unread = pg("p1", { source_name: "Scan 1.tiff", skip_reason: "", message: "No Student ID found on this page." });
      const missed = pg("p2", { source_name: "Scan 2.tiff", skip_reason: "", attach_to: { page_id: "p1", student_name: "" } });
      await answerBatches(page, offDocs([], [missed, unread]));
      const offered = unsortedCards(page).find((c) => textIn(c).includes("Scan 2.tiff"));
      buttonIn(offered, "Attach as the Declaration page of the form before it").fire("click");
      const attach = await answer(page.calls, "/api/pages/p2/declaration", { page: {}, document: null });
      await tick(5);
      const said = textIn(offered);
      const attached = pg("p2", { source_name: "Scan 2.tiff", skip_reason: "", declaration_by: "staff@akanksha.org" });
      await answerBatches(page, offDocs([], [Object.assign({}, unread, { declaration_pages: [attached] })]));
      const [waiting] = unsortedCards(page);
      const shown = {
        cards: unsortedCards(page).length, text: textIn(waiting),
        buttons: waiting._all().filter((e) => e.tagName === "BUTTON").map((e) => e.textContent),
      };
      buttonIn(waiting, "Take off the Declaration page").fire("click");
      const detach = await answer(page.calls, "/api/pages/p2/declaration", { page: {}, document: null });
      await tick(5);
      return { attach: JSON.parse(attach.opts.body), said, shown, detach: JSON.parse(detach.opts.body), after: textIn(waiting) };
    """)

    assert out["attach"] == {"of": "p1"}
    # Not one document yet: that takes the Student ID of the page before it.
    assert "type the Student ID of the page before it, and the two become one document" in out["said"]
    assert "for you to confirm" not in out["said"]
    shown = out["shown"]
    assert shown["cards"] == 1 and "once the Student ID is typed, both are one document" in shown["text"]
    assert "Check ID" in shown["buttons"] and "Take off the Declaration page" in shown["buttons"]
    assert out["detach"] == {"of": ""}
    assert "Taken off - the page goes back under Unsorted in a moment." in out["after"]


@needs_node
def test_page_with_batch_codes_on_names_the_parts_as_before(
    client: FlaskClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Part 1, Part 2 and "2 parts", and "Open part 1 of B1-1 full size" - never a Declaration page."""
    monkeypatch.setattr(config, "BATCH_CODES", True)
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const parts = [pg("p1", { code: "B1-1", student_id: ASHA.student_id, matched_student: ASHA }),
        pg("p2", { code: "B1-1", skip_reason: "declaration" })];
      await answerBatches(page, batches([[1, [doc("B1-1", parts, { student_id: ASHA.student_id, matched_student: ASHA })]]]));
      const shown = card(page, "B1-1");
      return {
        parts: shown._all().find((e) => e.classList.contains("ocr-doc-parts")).textContent,
        labels: shown._all().filter((e) => e.classList.contains("ocr-thumb-label")).map((e) => e.textContent),
        aria: shown._all().filter((e) => e.classList.contains("ocr-thumb")).map((e) => e.attrs["aria-label"]),
        text: textIn(shown),
      };
    """)

    assert out["parts"] == "2 parts" and out["labels"] == ["Part 1", "Part 2"]
    assert out["aria"] == ["Open part 1 of B1-1 full size", "Open part 2 of B1-1 full size"]
    assert "Declaration" not in out["text"]


@needs_node
def test_page_without_codes_leaves_the_scan_folder_as_it_is(client: FlaskClient, tmp_path: Path) -> None:
    """No B<n> folders, nothing moved, nothing sent twice - a reload included."""
    out = _run_page(client, tmp_path, _OFF_RESULTS_JS + r"""
      const idb = makeIndexedDB();
      const clock = makeClock();
      const folder = makeFolder("OCR Scans", { move: true });
      const page = await openScannerPage({ clock, idb, folder });
      const steps = page.byId["scan-folder-steps"].innerHTML;
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      folder.add("Scan 002.jpg");
      await clock.advance(4000);
      for (const id of ["p1", "p2"]) await answerScan(page, [id]);
      await clock.advance(3000);
      await answerBatches(page, offDocs([
        pdoc(pg("p2", { source_name: "Scan 002.jpg", student_id: "MOHSHA011021", matched_student: ASHA }), { student_id: "MOHSHA011021", matched_student: ASHA }),
        pdoc(pg("p1", { source_name: "Scan 001.jpg", student_id: "MOHSHA011021", matched_student: ASHA }), { status: "uploaded", key: "Pp1~ab12cd34" }),
      ]));
      await clock.advance(10000);
      const sort = page.byId["scan-folder-sort"];
      const first = {
        top: folder.names(), folders: folder.folders(), made: folder.made, moves: folder.moves.length,
        removed: folder.removed, sent: everyScan(page).map(sentName), picker: page.pickerOptions,
        sortHidden: sort.hidden, error: page.byId["scan-folder-error"].textContent,
      };
      // Another visit: the scans still in the folder are known, not sent again.
      const clock2 = makeClock();
      const second = await openScannerPage({ clock: clock2, idb, folder });
      await clock2.advance(6000);
      const resent = everyScan(second).map(sentName);
      folder.add("Scan 003.jpg");
      await clock2.advance(4000);
      return { steps, first, resent, after: everyScan(second).map(sentName) };
    """)

    assert "B1" not in out["steps"] and "batch" not in out["steps"].lower()
    assert "let the browser view its files" in out["steps"]
    first = out["first"]
    assert first["picker"] == {"id": "ocr-scan-folder", "mode": "read"}
    assert first["top"] == ["Scan 001.jpg", "Scan 002.jpg"]
    assert first["folders"] == [] and first["made"] == []
    assert first["moves"] == 0 and first["removed"] == []
    assert first["sent"] == ["Scan 001.jpg", "Scan 002.jpg"]
    assert first["sortHidden"] is True and first["error"] == ""
    assert out["resent"] == []
    assert out["after"] == ["Scan 003.jpg"]


@needs_node
def test_page_with_codes_back_on_leaves_scans_confirmed_without_a_code_where_they_are(
    client: FlaskClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codes on again: a scan confirmed while they were off goes into no "B0" folder, and is listed under no heading."""
    monkeypatch.setattr(config, "BATCH_CODES", True)
    out = _run_page(client, tmp_path, _OFF_RESULTS_JS + r"""
      const idb = makeIndexedDB();
      const clock = makeClock();
      const folder = makeFolder("OCR Scans", { move: true });
      const page = await openScannerPage({ clock, idb, folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      folder.add("Scan 002.jpg");
      await clock.advance(4000);
      for (const id of ["p1", "p2"]) await answerScan(page, [id]);
      await clock.advance(3000);
      // As the server lists them: the coded document under B1, the one
      // confirmed while codes were off in a group with no batch.
      const coded = doc("B1-1", [pg("p2", { code: "B1-1", source_name: "Scan 002.jpg", student_id: "MOHSHA011021", matched_student: ASHA })],
        { student_id: "MOHSHA011021", matched_student: ASHA });
      const uncoded = pdoc(pg("p1", { source_name: "Scan 001.jpg", student_id: "MOHSHA011021", matched_student: ASHA }),
        { status: "uploaded", key: "Pp1~ab12cd34", student_id: "MOHSHA011021", matched_student: ASHA });
      const data = batches([[1, [coded]]]);
      data.batches.push({ batch: null, label: "", documents: [uncoded], counts: {} });
      data.pages.push(...uncoded.pages);
      await answerBatches(page, data);
      await clock.advance(10000);
      const headings = page.byId["ocr-batches"]._all().filter((e) => e.classList.contains("ocr-batch-title"));
      return {
        top: folder.names(), folders: folder.folders(), b1: folder.folders().includes("B1") ? folder.sub("B1").names() : [],
        headings: headings.map((e) => e.textContent), cards: cards(page).length,
      };
    """)

    assert out["folders"] == ["B1"] and out["b1"] == ["Scan 002.jpg"]
    assert out["top"] == ["Scan 001.jpg"]
    assert out["headings"] == ["B1"] and out["cards"] == 2
