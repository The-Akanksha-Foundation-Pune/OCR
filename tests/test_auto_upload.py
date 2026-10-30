"""Upload matched pages automatically: an exact match at one school goes up by itself.

With the switch on, the scan page confirms a document without anyone pressing
Confirm - but only when all four hold (documents.auto_upload_refusal):

  1. it is Ready: matched, nothing to check, nothing failed or still read -
     with batch codes off, a whole form as the reader put it together: its
     one Declaration page, recognised by the reader rather than attached by
     hand, scanned straight after its Student ID page - nothing between,
     not even a blank page, and not after the form was shown without one -
     from the same scan page and the same way (the scan folder, or by
     hand); and only once its whole stack is finished, clean and alone
     (tests/test_auto_upload_stack.py has those rules): nothing new for
     STACK_SETTLE_MINUTES, which the page keeps looking through;
  2. the match is exact: the Student ID read by OCR is the matched
     student's own once trimmed and upper-cased, not a misread the lookup
     repaired towards a near ID (with batch codes off a typed one never
     goes up by itself);
  3. one school was picked for its pages, not only a city;
  4. every page was sent with the switch on - pages queued before the switch
     existed load with it off.

Covered here: each condition both ways, the flag on POST /api/scans, the
server checking it all again and refusing a stale or ineligible automatic
confirm with nothing saved, no document confirmed twice, failures left for a
person, the record and History saying it was automatic, batch codes on and
off - and the page itself (Node): it confirms an eligible document once,
sends the switch and its own id with every scan, waits while scans are still
coming, keeps looking while a form may still go up, and leaves everything
else alone.

The database is the FakeDB, Drive the FakeDrive, OCR the stand-in readers
of tests/test_batch_queue.py; nothing leaves the test.
"""

from __future__ import annotations

import dataclasses
import json
import re
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
    RUN_GAP_MINUTES,
    STACK_SETTLE_MINUTES,
)
from server.services import documents, drive_store, ocr_worker, scan_queue
from tests import test_batch_queue as queue_tests
from tests.test_app import _run_page, needs_node
from tests.test_batch_codes_off import ONE, TWO, _failed, _later, _skipped
from tests.test_batch_queue import (
    AATIF,
    MOHSHA,
    OWNER,
    SCHOOL,
    FakeDrive,
    FakeResponse,
    Readers,
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
# (autouse), the stand-in readers, the fake database's students, the fake
# Drive and a Google sign-in.
queue_dir = queue_tests.queue_dir
readers = queue_tests.readers
students = queue_tests.students
drive = queue_tests.drive
google = queue_tests.google


@pytest.fixture(autouse=True)
def batch_codes(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Batch codes off, as the app ships; the tests that need them on say so."""
    monkeypatch.setattr(config, "BATCH_CODES", False)
    return False


@pytest.fixture
def near_ids(fake_db: Any, students: Students) -> Students:
    """The fake database also lists a school's Student IDs, so a misread can be repaired."""
    answer = fake_db.handler

    def handler(sql: str, params: Any) -> tuple[list[dict[str, Any]] | None, int | None]:
        if sql.startswith("SELECT student_id FROM active_student_data WHERE UPPER(TRIM(school_name))"):
            return [{"student_id": student_id} for student_id in students.rows], None
        return answer(sql, params)

    fake_db.handler = handler
    return students


def _queue(client: FlaskClient, name: str = "Scan 1.tiff", **form: str) -> list[str]:
    """Send one scan as the page does with the switch on, from scan page ONE, unless `form` says otherwise."""
    fields = {"auto_upload": "1", "sender": ONE, **form}
    response = _send(client, name, _tiff([_form_page()]), **fields)
    assert response.status_code == 202, response.get_json()
    return [page["page_id"] for page in response.get_json()["pages"]]


def _key(page_id: str) -> str:
    return f"P{page_id}"


def _document(client: FlaskClient, key: str) -> dict[str, Any]:
    response = client.get(f"/api/documents/{key}")
    assert response.status_code == 200, response.get_json()
    return response.get_json()["document"]


def _parts(client: FlaskClient, key: str) -> list[str]:
    """A document's page ids as the page shows them: the Student ID page, then its Declaration page."""
    return [page["page_id"] for page in _document(client, key)["pages"]]


def _auto_confirm(
    client: FlaskClient, key: str, student_id: str = "AATKHA200618", page_ids: list[str] | None = None
) -> Any:
    """What the page sends when it confirms a document by itself."""
    if page_ids is None:
        page_ids = [page["page_id"] for page in _document(client, key)["pages"]]
    return client.post(
        f"/api/documents/{key}/confirm",
        json={"student_id": student_id, "page_ids": page_ids, "automatic": True},
    )


def _refused_with_nothing_saved(response: Any, fake_db: Any, drive: FakeDrive) -> str:
    assert response.status_code == 409, response.get_json()
    error = response.get_json()["error"]
    assert "Nothing was saved" in error
    assert _writes(fake_db) == [] and drive.calls == []
    return error


def _ready_exact(client: FlaskClient, name: str = "Scan 1.tiff", **form: str) -> str:
    """One form sent with the switch on and read; its document's key.

    Its Student ID page, matched exactly, then its Declaration page - a
    stack of its own: nothing came after it for STACK_SETTLE_MINUTES.
    """
    [page_id] = _queue(client, name, **form)
    declaration = _queue(client, f"{Path(name).stem} declaration.tiff", **form)[0]
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _skipped(declaration)
    _settle()
    return _key(page_id)


def _settle() -> None:
    """As if nothing more had come for STACK_SETTLE_MINUTES: every stack so far is finished."""
    _later(STACK_SETTLE_MINUTES * 60 + 1)


def _declaration_after(client: FlaskClient, name: str = "Scan 2.tiff", **form: str) -> str:
    """A Declaration page sent with the switch on after what is queued so far, and read."""
    [page_id] = _queue(client, name, **form)
    _skipped(page_id)
    return page_id


def _tab(app: Flask) -> FlaskClient:
    """Another tab - or computer - of the same person, signed in with Google."""
    tab = app.test_client()
    with tab.session_transaction() as session:
        session["user"] = {"email": OWNER, "name": "Staff", "picture": ""}
        session["google_token"] = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_at": time.time() + 3600,
        }
    return tab


def _at_once(*calls: Any) -> list[int]:
    """Run each call in a thread of its own, all starting together; their status codes."""
    start = threading.Barrier(len(calls))
    answers: list[int] = []

    def run(call: Any) -> None:
        start.wait()
        answers.append(call().status_code)

    threads = [threading.Thread(target=run, args=(call,)) for call in calls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return sorted(answers)


# ================================================================== the flag at send time
def test_each_page_of_a_scan_sent_with_the_switch_on_carries_the_flag(
    client: FlaskClient, queue_dir: Path
) -> None:
    response = _send(client, "stack.tiff", _tiff([_form_page(), _form_page()]), auto_upload="1")

    assert response.status_code == 202
    for page in response.get_json()["pages"]:
        meta = json.loads((queue_dir / page["page_id"] / "meta.json").read_text())
        assert meta["auto_upload"] is True and meta["school"] == SCHOOL


@pytest.mark.parametrize("sent", [None, "0", "", "off"])
def test_a_scan_sent_without_the_switch_is_not_flagged(
    client: FlaskClient, queue_dir: Path, sent: str | None
) -> None:
    form = {} if sent is None else {"auto_upload": sent}
    [page] = _send(client, "Scan 1.tiff", _tiff([_form_page()]), **form).get_json()["pages"]

    meta = json.loads((queue_dir / page["page_id"] / "meta.json").read_text())
    assert meta["auto_upload"] is False


def test_a_page_queued_before_the_switch_existed_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive, queue_dir: Path
) -> None:
    key = _ready_exact(google)
    # meta.json as this version's predecessor wrote it: no auto_upload at all.
    path = queue_dir / key[1:] / "meta.json"
    meta = json.loads(path.read_text())
    del meta["auto_upload"]
    path.write_text(json.dumps(meta))

    loaded = scan_queue.load_meta(key[1:])
    doc = _document(google, key)

    assert loaded is not None and loaded.auto_upload is False
    assert doc["status"] == "ready" and doc["can_confirm"] is True and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert "sent with automatic upload off" in error
    # A person can still confirm it.
    assert _confirm(google, key=key).status_code == 200


# ================================================================== 1. Ready
def test_a_ready_exact_match_at_one_school_sent_with_the_switch_on_may_go_up(client: FlaskClient) -> None:
    key = _ready_exact(client)

    [listed] = _batches(client)["batches"][0]["documents"]

    assert listed["key"] == key and listed["status"] == "ready" and listed["parts"] == 2
    assert listed["auto_upload"] is True and listed["confirmed_automatically"] is False
    assert _document(client, key)["auto_upload"] is True


def test_a_document_that_needs_checking_waits_for_a_person(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    no_match = _queue(google, "Scan 1.tiff")[0]
    _read(no_match, None, "AATKHA200619", None)
    _declaration_after(google, "Scan 2.tiff")
    failed = _queue(google, "Scan 3.tiff")[0]
    _declaration_after(google, "Scan 4.tiff")
    (scan_queue.queue_root() / "_pending" / failed).unlink()
    meta = scan_queue.load_meta(failed)
    assert meta is not None
    meta.status, meta.attempts, meta.error = scan_queue.STATUS_FAILED, 3, "MemoryError: out of memory"
    scan_queue.save_meta(meta)
    google.post(f"/api/pages/{failed}/student-id", json={"student_id": "AATKHA200618"})

    docs = {d["key"]: d for d in _batches(google)["batches"][0]["documents"]}

    assert docs[_key(no_match)]["status"] == "needs_attention" and docs[_key(no_match)]["auto_upload"] is False
    assert docs[_key(failed)]["status"] == "failed" and docs[_key(failed)]["auto_upload"] is False
    error = _refused_with_nothing_saved(
        _auto_confirm(google, _key(no_match), student_id="AATKHA200619"), fake_db, drive
    )
    assert "needs checking" in error
    _refused_with_nothing_saved(_auto_confirm(google, _key(failed)), fake_db, drive)


def test_a_student_id_page_without_its_declaration_page_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google, "Scan 1.tiff")
    after = _queue(google, "Scan 2.tiff")[0]
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    key = _key(page_id)

    # The page after it may be its Declaration page, still to be read.
    waiting = _document(google, key)
    early = _auto_confirm(google, key, page_ids=[page_id])

    assert waiting["status"] == "waiting" and waiting["auto_upload"] is False
    assert early.status_code == 409 and "still being read" in early.get_json()["error"]
    assert _writes(fake_db) == [] and drive.calls == []

    # It was the next form's Student ID page: this form has no Declaration page.
    _read(after, None, "MOHSHA011021", dict(MOHSHA, matched_via="exact"))
    alone = _document(google, key)

    assert alone["status"] == "needs_attention" and alone["attention"] == ["no_declaration", "gap_in_stack"]
    assert alone["can_confirm"] is True and alone["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("It has no Declaration page, so it was not uploaded automatically")
    assert scan_queue.open_page_ids(OWNER) == sorted([page_id, after])
    # A person may still confirm it.
    assert _confirm(google, key=key).status_code == 200


def test_a_declaration_page_sent_with_the_switch_off_holds_its_form_back(client: FlaskClient) -> None:
    [page_id] = _queue(client, "Scan 1.tiff")
    _declaration_after(client, "Scan 2.tiff", auto_upload="0")
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))

    doc = _document(client, _key(page_id))
    found = documents.snapshot(OWNER).find(_key(page_id))

    assert doc["status"] == "ready" and doc["parts"] == 2 and doc["auto_upload"] is False
    assert found is not None and documents.auto_upload_refusal(found) == "it was sent with automatic upload off"


def test_no_form_of_a_stack_fed_declaration_first_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    # The first sheet is a Declaration page: the Student ID page after it
    # takes the next form's Declaration page, exactly one, straight after it.
    _declaration_after(google, "Scan 1.tiff")
    [page_id] = _queue(google, "Scan 2.tiff")
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _declaration_after(google, "Scan 3.tiff")
    key = _key(page_id)

    doc = _document(google, key)

    assert doc["parts"] == 2 and doc["attention"] == ["orphan_in_stack"]
    assert doc["can_confirm"] is True and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith(
        "Its stack has a Declaration page without its Student ID page, so it was not uploaded automatically"
    )
    # A person may still confirm it.
    assert _confirm(google, key=key).status_code == 200


def test_a_form_followed_by_a_second_declaration_page_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google, "Scan 1.tiff")
    _declaration_after(google, "Scan 2.tiff")
    # The next form's Student ID page went missing, or a stray Declaration
    # page came between this form's two sheets: another Declaration page
    # follows this one's, on the server but not read yet.
    again = _queue(google, "Scan 3.tiff")[0]
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    key = _key(page_id)

    reading = _document(google, key)

    # Until that page is read the form waits for it - nothing saved.
    assert reading["status"] == "ready" and reading["auto_upload"] is False
    assert reading["auto_upload_pending"] is True
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("A page of its stack is still being read")

    _skipped(again)
    _settle()

    doc = _document(google, key)
    assert doc["parts"] == 2 and doc["auto_upload"] is False and doc["attention"] == ["orphan_in_stack"]
    _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)


def test_a_form_goes_up_by_itself_only_once_its_whole_stack_has_come_in(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    students.rows["MOHSHA011021"] = dict(MOHSHA)
    [page_id] = _queue(google, "Scan 1.tiff")
    _declaration_after(google, "Scan 2.tiff")
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    key = _key(page_id)

    # Nothing after its Declaration page yet: the scanner's next file may be
    # a page out of place, which would hold it. The page keeps looking.
    alone = _document(google, key)
    assert alone["status"] == "ready" and alone["can_confirm"] is True
    assert alone["auto_upload"] is False and alone["auto_upload_pending"] is True
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("Its stack may still go on, so it was not uploaded automatically")

    # The next sheet has come, and is still to be read.
    next_id = _queue(google, "Scan 3.tiff")[0]
    reading = _document(google, key)
    assert reading["auto_upload"] is False and reading["auto_upload_pending"] is True
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("A page of its stack is still being read")

    # It is the next form's Student ID page: the stack is not whole until
    # that form's Declaration page has come too.
    _read(next_id, None, "MOHSHA011021", dict(MOHSHA, matched_via="exact"))
    so_far = _document(google, key)
    assert so_far["auto_upload"] is False and so_far["auto_upload_pending"] is True
    assert so_far["auto_upload_at"] is None
    _declaration_after(google, "Scan 4.tiff")
    settling = _document(google, key)
    assert settling["auto_upload"] is False and settling["auto_upload_pending"] is True
    assert settling["auto_upload_at"]

    _settle()

    settled = _document(google, key)
    assert settled["auto_upload"] is True and settled["auto_upload_pending"] is False
    assert _auto_confirm(google, key).status_code == 200
    assert _auto_confirm(google, _key(next_id), student_id="MOHSHA011021").status_code == 200


def test_the_last_form_of_a_stack_goes_up_by_itself_once_nothing_has_come_after_it_for_a_while(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google, "Scan 1.tiff")
    _declaration_after(google, "Scan 2.tiff")
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    key = _key(page_id)
    _later(STACK_SETTLE_MINUTES * 60 - 5)
    assert _document(google, key)["auto_upload_pending"] is True

    _later(10)

    doc = _document(google, key)
    assert doc["auto_upload"] is True and doc["auto_upload_pending"] is False
    assert _auto_confirm(google, key).status_code == 200


@pytest.mark.parametrize(
    ("after", "why"),
    [
        ("blank", "Its stack has a blank page"),
        ("missed", "Its stack does not go Student ID page, Declaration page, and so on"),
        ("failed", "Its stack has a page that could not be read"),
    ],
)
def test_a_form_whose_declaration_page_has_no_student_id_page_after_it_waits_for_a_person(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive, after: str, why: str
) -> None:
    [page_id] = _queue(google, "Scan 1.tiff")
    _declaration_after(google, "Scan 2.tiff")
    next_page = _queue(google, "Scan 3.tiff")[0]
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    # A blank sheet, one with no Student ID - or Declaration heading - read,
    # or one not read at all: this form's own Declaration page may be that
    # one, and the one before it a stray.
    if after == "blank":
        _skipped(next_page, "blank")
    elif after == "missed":
        _read(next_page, None)
    else:
        _failed(next_page)
    _settle()
    key = _key(page_id)

    doc = _document(google, key)

    assert doc["status"] == "ready" and doc["auto_upload"] is False and doc["auto_upload_pending"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith(why)
    if after != "blank":
        # Its Student ID typed makes it the next form's Student ID page - by
        # a person's hand, so the stack still waits for one.
        google.post(f"/api/pages/{next_page}/student-id", json={"student_id": "MOHSHA011021"})
        doc = _document(google, key)
        assert doc["auto_upload"] is False and doc["auto_upload_refusal"] == "a page of its stack was corrected by hand"


def test_a_form_at_the_end_of_its_run_waits_for_a_person_when_another_scan_page_sends_after_it(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google, "Scan 1.tiff")
    _declaration_after(google, "Scan 2.tiff")
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    # The scan page is loaded again, and the next sheet comes from it: it
    # may be the rest of this stack, whatever it turns out to be.
    after = _queue(google, "Scan 3.tiff", sender=TWO)[0]
    key = _key(page_id)

    reading = _document(google, key)
    assert reading["auto_upload"] is False and reading["auto_upload_pending"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("Another computer or tab sent pages for the same school around the same time")

    _read(after, None, "MOHSHA011021", dict(MOHSHA, matched_via="exact"))
    _settle()

    doc = _document(google, key)
    assert doc["auto_upload"] is False and doc["auto_upload_pending"] is False
    _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)


def test_two_sheets_swapped_across_two_forms_never_go_up_by_themselves(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    # Form A's Declaration page fed after form B's Student ID page, and B's
    # own Declaration page not sent yet.
    id_a, id_b = _queue(google, "Scan 1.tiff")[0], _queue(google, "Scan 2.tiff")[0]
    _declaration_after(google, "Scan 3.tiff")
    _read(id_a, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _read(id_b, None, "MOHSHA011021", dict(MOHSHA, matched_via="exact"))
    students.rows["MOHSHA011021"] = dict(MOHSHA)
    _later(DECLARATION_WAIT_SECONDS + 1)

    docs = {d["key"]: d for d in _batches(google)["batches"][0]["documents"]}

    # B has exactly one Declaration page, straight after it - A's.
    assert _parts(google, _key(id_b))[0] == id_b and docs[_key(id_b)]["parts"] == 2
    assert [docs[_key(page_id)]["auto_upload"] for page_id in (id_a, id_b)] == [False, False]
    error = _refused_with_nothing_saved(
        _auto_confirm(google, _key(id_b), student_id="MOHSHA011021"), fake_db, drive
    )
    assert error.startswith("Its stack has a page with no Declaration page between it and the next form")
    _declaration_after(google, "Scan 4.tiff")
    assert _document(google, _key(id_b))["auto_upload"] is False


def test_no_form_goes_up_by_itself_while_a_page_scanned_before_it_is_still_being_read(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    # The stack's first sheet is still with a reader while the next ones are
    # read: it may be a Student ID page, or a Declaration page fed first.
    lead = _queue(google, "Scan 1.tiff")[0]
    _declaration_after(google, "Scan 2.tiff")
    [page_id] = _queue(google, "Scan 3.tiff")
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _declaration_after(google, "Scan 4.tiff")
    _settle()
    key = _key(page_id)

    reading = _document(google, key)

    assert reading["status"] == "ready" and reading["auto_upload"] is False
    assert reading["auto_upload_pending"] is True
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("A page of its stack is still being read")

    _skipped(lead)

    held = _document(google, key)
    assert held["attention"] == ["orphan_in_stack"] and held["auto_upload"] is False


def test_a_stack_out_of_turn_waits_for_a_person_whatever_its_page_still_being_read_turns_out_to_be(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    # A stack fed Declaration page first, its first sheet still with a
    # reader: the Student ID page in a Declaration page's place says enough.
    _queue(google, "Scan 1.tiff")
    [page_id] = _queue(google, "Scan 2.tiff")
    _declaration_after(google, "Scan 3.tiff")
    next_id = _queue(google, "Scan 4.tiff")[0]
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _read(next_id, None, "MOHSHA011021", dict(MOHSHA, matched_via="exact"))
    _settle()
    key = _key(page_id)

    doc = _document(google, key)

    assert doc["status"] == "ready" and doc["auto_upload"] is False and doc["auto_upload_pending"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("Its stack does not go Student ID page, Declaration page, and so on")


@pytest.mark.parametrize("cut", ["pause", "reload"])
def test_a_stack_fed_declaration_first_and_cut_in_two_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive, cut: str
) -> None:
    _declaration_after(google, "Scan 1.tiff")
    [id_a] = _queue(google, "Scan 2.tiff")
    _read(id_a, None, "MOHSHA011021", dict(MOHSHA, matched_via="exact"))
    _declaration_after(google, "Scan 3.tiff")
    # A jam longer than RUN_GAP_MINUTES, or the scan page loaded again.
    if cut == "pause":
        _later(RUN_GAP_MINUTES * 60 + 1)
    form = {"sender": TWO} if cut == "reload" else {}
    [id_b] = _queue(google, "Scan 4.tiff", **form)
    _read(id_b, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _declaration_after(google, "Scan 5.tiff", **form)
    _later(DECLARATION_WAIT_SECONDS + 1)
    key = _key(id_b)

    doc = _document(google, key)

    assert doc["attention"] == ["after_held_stack"] and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("Its stack may go on from one scanned before it that has a page out of place")


def test_a_stack_loaded_again_while_its_first_run_is_still_being_read_waits(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    # The first sheet - a Declaration page - is still with a reader when the
    # scan page is loaded again and the rest of the stack comes in.
    lead = _queue(google, "Scan 1.tiff")[0]
    [id_a] = _queue(google, "Scan 2.tiff")
    _read(id_a, None, "MOHSHA011021", dict(MOHSHA, matched_via="exact"))
    _declaration_after(google, "Scan 3.tiff")
    [id_b] = _queue(google, "Scan 4.tiff", sender=TWO)
    _read(id_b, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _declaration_after(google, "Scan 5.tiff", sender=TWO)
    _settle()
    key = _key(id_b)

    doc = _document(google, key)

    # The two scan pages' pages came within minutes of each other: a person
    # looks, whatever the first sheet turns out to be.
    assert doc["status"] == "ready" and doc["auto_upload"] is False and doc["auto_upload_pending"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("Another computer or tab sent pages for the same school around the same time")

    _skipped(lead)

    assert _document(google, key)["attention"] == ["after_held_stack"]


def test_a_form_added_by_hand_in_two_picks_may_go_up_by_itself(client: FlaskClient) -> None:
    # A browser with no scan folder: each sheet added as it is scanned.
    hand = {"sender": f"{ONE}-1"}
    [page_id] = _queue(client, "Scan 1.tiff", **hand)
    declaration = _declaration_after(client, "Scan 2.tiff", **hand)
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _settle()

    doc = _document(client, _key(page_id))

    assert _parts(client, _key(page_id)) == [page_id, declaration]
    assert doc["status"] == "ready" and doc["auto_upload"] is True


def test_a_form_with_one_sheet_from_the_scan_folder_and_one_added_by_hand_waits_for_a_person(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    # A file added by hand while the scanner was between a form's two
    # sheets - or a sheet the scan folder could not send, added by hand.
    [page_id] = _queue(google, "Scan 1.tiff")
    declaration = _declaration_after(google, "Scan 2.tiff", sender=f"{ONE}-1")
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _later(DECLARATION_WAIT_SECONDS + 1)
    key = _key(page_id)

    doc = _document(google, key)

    # Paired - one scan page, one stack - but not by itself.
    assert _parts(google, key) == [page_id, declaration] and doc["status"] == "ready"
    assert doc["auto_upload"] is False and doc["auto_upload_pending"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("One of its pages was added by hand and the other came from the scan folder")
    assert _confirm(google, key=key).status_code == 200


def test_a_form_with_a_blank_page_between_its_two_sheets_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google, "Scan 1.tiff")
    blank = _queue(google, "Scan 2.tiff")[0]
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _skipped(blank, "blank")
    _declaration_after(google, "Scan 3.tiff")
    key = _key(page_id)

    doc = _document(google, key)

    # Paired, for a person to confirm: the blank page may be the next form's
    # Student ID page, fed face down.
    assert doc["status"] == "ready" and doc["parts"] == 2 and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("A page came between its Student ID page and its Declaration page")
    assert _confirm(google, key=key).status_code == 200


def test_a_form_whose_declaration_page_was_attached_by_hand_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google, "Scan 1.tiff")
    missed = _queue(google, "Scan 2.tiff")[0]
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _read(missed, None)
    assert google.post(f"/api/pages/{missed}/declaration", json={"of": page_id}).status_code == 200
    key = _key(page_id)

    doc = _document(google, key)

    assert doc["status"] == "ready" and doc["parts"] == 2 and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("Its Declaration page was attached by hand, so it was not uploaded automatically")
    # A person presses Confirm.
    assert _confirm(google, key=key).status_code == 200
    with fitz.open(stream=drive.uploaded()[0]["data"], filetype="pdf") as pdf:
        assert pdf.page_count == 2


def test_a_form_sent_without_the_scan_page_s_id_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    # An older page still open in a tab sends the switch, but no id: two of
    # them on one sign-in would mix their stacks.
    key = _ready_exact(google, sender="")

    doc = _document(google, key)

    assert doc["status"] == "ready" and doc["parts"] == 2 and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("It is not known that one computer sent both its pages")


@pytest.mark.parametrize("school", ["ABMPS", "Other School"])
def test_forms_from_two_computers_on_one_sign_in_each_pair_whole_but_go_up_alone_only(
    client: FlaskClient, school: str
) -> None:
    first = _queue(client, "Scan 1.tiff")[0]
    other = _queue(client, "Scan 1.tiff", sender=TWO, school=school)[0]
    first_declaration = _declaration_after(client, "Scan 2.tiff")
    other_declaration = _declaration_after(client, "Scan 2.tiff", sender=TWO, school=school)
    _read(first, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    _read(other, None, "MOHSHA011021", dict(MOHSHA, matched_via="exact"))
    # The last forms of their stacks: nothing came after either since.
    _settle()

    docs = {d["key"]: d for d in _batches(client)["batches"][0]["documents"]}

    assert _parts(client, _key(first)) == [first, first_declaration]
    assert _parts(client, _key(other)) == [other, other_declaration]
    # For one school, either may be the rest of the other's stack: both wait.
    # For two, each stack is alone for its own school.
    alone = school != "ABMPS"
    assert [docs[_key(page_id)]["auto_upload"] for page_id in (first, other)] == [alone, alone]
    if not alone:
        assert docs[_key(first)]["auto_upload_refusal"] == (
            "another computer or tab sent pages for the same school around the same time"
        )


def test_a_form_waiting_for_its_declaration_page_never_goes_up_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google, "Scan 1.tiff")
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact"))
    key = _key(page_id)

    waiting = _document(google, key)
    refused = _auto_confirm(google, key)
    _later(DECLARATION_WAIT_SECONDS + 1)
    alone = _document(google, key)

    assert waiting["status"] == "waiting" and waiting["auto_upload"] is False
    assert refused.status_code == 409 and refused.get_json()["error"].startswith("Waiting for its Declaration page")
    assert alone["status"] == "needs_attention" and alone["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("It has no Declaration page")


def test_a_student_confirmed_lately_is_not_uploaded_again_by_itself(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    first = _ready_exact(google, "Scan 1.tiff")
    assert _auto_confirm(google, first).status_code == 200
    again = _ready_exact(google, "Scan 2.tiff")

    doc = _document(google, again)

    assert doc["status"] == "needs_attention" and doc["attention"] == ["same_student"]
    assert doc["auto_upload"] is False
    assert _auto_confirm(google, again).status_code == 409
    assert len(drive.uploaded()) == 1


def test_with_batch_codes_a_document_waiting_for_a_part_or_whose_parts_disagree_waits(
    google: FlaskClient, students: Students, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    # Waiting: the page after it is not read yet, and may be its next part.
    waiting = _queue(google, "Scan 1.tiff")[0]
    _queue(google, "Scan 2.tiff")
    _read(waiting, "B1-1", "AATKHA200618", dict(AATIF))

    [doc] = _batches(google)["batches"][0]["documents"]
    assert doc["status"] == "waiting" and doc["auto_upload"] is False

    # Parts that match different students.
    conflict = _queue(google, "Scan 3.tiff")[0]
    other = _queue(google, "Scan 4.tiff")[0]
    _read(conflict, "B2-1", "AATKHA200618", dict(AATIF))
    _read(other, "B2-1", "MOHSHA011021", dict(MOHSHA))

    found = _document(google, "B2-1")
    assert found["attention"] == ["conflict"] and found["auto_upload"] is False


def test_with_batch_codes_a_code_used_again_waits(
    google: FlaskClient, students: Students, drive: FakeDrive, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    first = _queue(google, "Scan 1.tiff")[0]
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))
    assert _auto_confirm(google, "B1-1").status_code == 200
    later = _queue(google, "Scan 2.tiff")[0]
    _read(later, "B1-1", "AATKHA200618", dict(AATIF))

    doc = _document(google, "B1-1")

    assert doc["attention"] == ["code_reused"] and doc["auto_upload"] is False
    assert _auto_confirm(google, "B1-1").status_code == 409
    assert len(drive.uploaded()) == 1


def test_two_open_documents_for_one_student_both_wait_for_a_person(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    # The same form scanned twice, or two pages of one form that both carry the ID.
    first = _ready_exact(google, "Scan 1.tiff")
    second = _ready_exact(google, "Scan 2.tiff")

    docs = {d["key"]: d for d in _batches(google)["batches"][0]["documents"]}

    for key in (first, second):
        assert docs[key]["status"] == "ready" and docs[key]["can_confirm"] is True
        assert docs[key]["auto_upload"] is False
        error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
        assert error.startswith("Another document - open, or confirmed lately - is for the same student")
    # A person may confirm either; the other then says so.
    assert _confirm(google, key=first).status_code == 200
    assert _document(google, second)["attention"] == ["same_student"]


def test_two_documents_for_one_student_confirmed_by_themselves_at_once_save_neither(
    app: Flask, google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _ready_exact(google, "Scan 1.tiff")
    second = _ready_exact(google, "Scan 2.tiff")
    real_lookup = documents.confirm_student_id

    def round_trip(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.3)
        return real_lookup(*args, **kwargs)

    monkeypatch.setattr(documents, "confirm_student_id", round_trip)
    one, two = _tab(app), _tab(app)

    first_parts, second_parts = _parts(google, first), _parts(google, second)

    answers = _at_once(
        lambda: _auto_confirm(one, first, page_ids=first_parts),
        lambda: _auto_confirm(two, second, page_ids=second_parts),
    )

    assert answers == [409, 409]
    assert _writes(fake_db) == [] and drive.uploaded() == []
    assert scan_queue.open_page_ids(OWNER) == sorted(first_parts + second_parts)


def test_a_document_read_while_another_for_its_student_is_being_confirmed_waits(
    app: Flask, google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _ready_exact(google, "Scan 1.tiff")
    first_parts = _parts(google, first)
    inside, go_on = threading.Event(), threading.Event()
    real_lookup = documents.confirm_student_id

    def held_lookup(*args: Any, **kwargs: Any) -> Any:
        # The first confirm's database round trip, held while the second
        # scan comes in.
        if not inside.is_set():
            inside.set()
            go_on.wait(timeout=30)
        return real_lookup(*args, **kwargs)

    monkeypatch.setattr(documents, "confirm_student_id", held_lookup)
    tab = _tab(app)
    answers: list[int] = []
    confirming = threading.Thread(
        target=lambda: answers.append(_auto_confirm(tab, first, page_ids=first_parts).status_code)
    )
    confirming.start()
    assert inside.wait(timeout=30)

    # Another tab reads the same form again while the first is in its lookup.
    second = _ready_exact(google, "Scan 2.tiff")
    listed = _document(google, second)
    refused = _auto_confirm(google, second, page_ids=_parts(google, second))
    go_on.set()
    confirming.join(timeout=60)

    assert listed["status"] == "ready" and listed["auto_upload"] is False
    assert refused.status_code == 409 and "for the same student" in refused.get_json()["error"]
    assert answers == [200]
    assert len(fake_db.executed(r"^UPDATE active_student_data")) == 1
    assert len(fake_db.executed(r"^INSERT INTO ocr_scan_log")) == 1
    assert len(drive.uploaded()) == 1
    assert _document(google, second)["attention"] == ["same_student"]


def test_a_confirm_saving_its_record_is_never_missed_by_another_s_look(
    google: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record is saved, then its pages leave the open list; a look in between sees one or the other."""
    first = _ready_exact(google, "Scan 1.tiff")
    second = _ready_exact(google, "Scan 2.tiff")
    first_parts = _parts(google, first)
    record = {
        "key": f"{first}~0a1b2c3d", "code": "", "owner_email": OWNER, "page_ids": first_parts,
        "status": "confirmed", "student_id": "AATKHA200618", "student": dict(AATIF),
        "confirmed_at": "2026-10-29T10:00:00+05:30", "automatic": True,
    }
    finished: list[str] = []

    def then_a_confirm_finishes(real: Any) -> Any:
        # Straight after the first of the two looks, whichever it is, the
        # other confirm saves its record and takes its page off the list.
        def look(*args: Any, **kwargs: Any) -> Any:
            found = real(*args, **kwargs)
            if not finished:
                finished.append(real.__name__)
                scan_queue.save_document_record(OWNER, record)
                scan_queue.close_pages(OWNER, first_parts)
            return found

        return look

    monkeypatch.setattr(scan_queue, "open_page_ids", then_a_confirm_finishes(scan_queue.open_page_ids))
    monkeypatch.setattr(
        scan_queue, "load_document_records", then_a_confirm_finishes(scan_queue.load_document_records)
    )
    doc = documents.snapshot(OWNER).find(second)

    assert doc is not None and doc.shares_student is True
    assert documents.auto_upload_refusal(doc) != ""


def test_a_student_already_marked_scanned_waits_for_a_person(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    [page_id] = _queue(google)
    _declaration_after(google)
    # Marked when the page was looked up: an earlier form, perhaps a colleague's.
    _read(page_id, None, "AATKHA200618", dict(AATIF, matched_via="exact", scanned_at="2026-10-01 11:20:00"))
    key = _key(page_id)

    doc = _document(google, key)

    assert doc["status"] == "ready" and doc["can_confirm"] is True and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("The student is already marked scanned")
    assert fake_db.statements == []
    # A person may still confirm it.
    assert _confirm(google, key=key).status_code == 200


def test_a_student_marked_scanned_since_the_page_was_read_is_not_uploaded_by_itself(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)
    assert _document(google, key)["auto_upload"] is True
    # Confirmed meanwhile by someone else, from another form for the child.
    students.rows["AATKHA200618"]["scanned_at"] = datetime(2026, 10, 29, 9, 15)

    response = _auto_confirm(google, key)

    error = _refused_with_nothing_saved(response, fake_db, drive)
    assert error.startswith("Aatif Khan (AATKHA200618) is already marked scanned")
    assert scan_queue.open_page_ids(OWNER) == sorted(_parts(google, key))
    assert scan_queue.load_document_records(OWNER) == []
    # A person's confirm marks the child again, as it always has.
    assert _confirm(google, key=key).status_code == 200
    assert len(fake_db.executed(r"^UPDATE active_student_data SET scanned_at")) == 1


def test_an_automatic_confirm_marks_the_student_only_if_still_unmarked(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)
    answer = fake_db.handler

    def marked_meanwhile(sql: str, params: Any) -> tuple[list[dict[str, Any]] | None, int | None]:
        # Another confirm marked the child between the lookup and the UPDATE.
        if sql.startswith("UPDATE active_student_data SET scanned_at") and "scanned_at IS NULL" in sql:
            return [], 0
        return answer(sql, params)

    fake_db.handler = marked_meanwhile

    response = _auto_confirm(google, key)

    assert response.status_code == 409
    assert "already marked scanned, so it was not uploaded automatically" in response.get_json()["error"]
    [update] = fake_db.executed(r"^UPDATE active_student_data SET scanned_at")
    assert update[0].endswith("AND scanned_at IS NULL")
    assert fake_db.executed(r"^INSERT") == [] and drive.calls == []
    assert scan_queue.load_document_records(OWNER) == []
    assert scan_queue.open_page_ids(OWNER) == sorted(_parts(google, key))


def test_a_person_s_confirm_marks_the_student_as_before(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)

    assert _confirm(google, key=key).status_code == 200

    [update] = fake_db.executed(r"^UPDATE active_student_data SET scanned_at")
    assert "scanned_at IS NULL" not in update[0]


def test_with_batch_codes_a_student_with_another_document_waits(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    first = _queue(google, "Scan 1.tiff")[0]
    second = _queue(google, "Scan 2.tiff")[0]
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))
    _read(second, "B1-2", "AATKHA200618", dict(AATIF))
    _read(_queue(google, "Scan 3.tiff")[0], None)

    docs = {d["key"]: d for d in _batches(google)["batches"][0]["documents"]}
    assert docs["B1-1"]["status"] == "ready" and docs["B1-1"]["auto_upload"] is False
    assert docs["B1-2"]["status"] == "ready" and docs["B1-2"]["auto_upload"] is False

    # Confirmed by a person, B1-1 still holds B1-2 back: with batch codes on
    # nothing else warns of a second form for the student.
    assert _confirm(google, key="B1-1").status_code == 200
    later = _document(google, "B1-2")
    assert later["status"] == "ready" and later["auto_upload"] is False
    refused = _auto_confirm(google, "B1-2")
    assert refused.status_code == 409 and "for the same student" in refused.get_json()["error"]
    assert len(fake_db.executed(r"^UPDATE active_student_data")) == 1
    assert len(drive.uploaded()) == 1


def test_with_batch_codes_nothing_goes_up_by_itself_while_pages_are_still_to_be_read(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    first = _queue(google, "Scan 1.tiff")[0]
    after = _queue(google, "Scan 2.tiff")[0]
    later = _queue(google, "Scan 3.tiff")[0]
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))
    # The next form has begun, so B1-1 is Ready - but a scan behind it is
    # still to be read, and the server cannot tell whose part it is.
    _read(after, "B1-2")

    doc = _document(google, "B1-1")

    assert doc["status"] == "ready" and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, "B1-1"), fake_db, drive)
    assert error.startswith("Some of your pages are still being read")
    _read(later, "B1-2")
    assert _document(google, "B1-1")["auto_upload"] is True


# ================================================================== 2. an exact match
@pytest.mark.parametrize("reading", ["AATKHA200618", "aatkha200618", "AATKHA-200618"])
def test_an_id_read_exactly_as_the_students_own_may_go_up(
    client: FlaskClient, fake_db: Any, students: Students, readers: Readers, reading: str
) -> None:
    [page_id] = _queue(client)
    declaration = _queue(client, "Scan 2.tiff")[0]
    readers.ids = [reading]

    assert ocr_worker.process_one() == ocr_worker.READ
    _skipped(declaration)
    _settle()

    doc = _document(client, _key(page_id))
    assert doc["matched_student"]["matched_via"] == "exact"
    assert doc["status"] == "ready" and doc["auto_upload"] is True


@pytest.mark.parametrize(
    ("reading", "via"),
    [
        # X for A in the name half: fuzzy-matched to the one close ID.
        ("AATKHX200618", "constrained_repair"),
        # Z for 2 in the date half: put right by the ID's shape.
        ("AATKHAZ00618", "shape_fix"),
    ],
)
def test_an_id_the_lookup_repaired_towards_a_near_one_waits(
    google: FlaskClient, fake_db: Any, near_ids: Students, readers: Readers, drive: FakeDrive,
    reading: str, via: str,
) -> None:
    [page_id] = _queue(google)
    declaration = _queue(google, "Scan 2.tiff")[0]
    readers.ids = [reading]
    ocr_worker.process_one()
    _skipped(declaration)

    doc = _document(google, _key(page_id))

    # Ready - a person may well confirm it - but not by itself.
    assert doc["matched_student"]["student_id"] == "AATKHA200618"
    assert doc["matched_student"]["matched_via"] == via
    assert doc["status"] == "ready" and doc["can_confirm"] is True and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, _key(page_id)), fake_db, drive)
    assert "not an exact match" in error
    assert _confirm(google, key=_key(page_id)).status_code == 200


def test_an_id_typed_exactly_is_ready_but_never_goes_up_by_itself(
    client: FlaskClient, fake_db: Any, students: Students
) -> None:
    [page_id] = _queue(client)
    declaration = _declaration_after(client)
    _read(page_id, None, "", None)
    _settle()

    response = client.post(f"/api/pages/{page_id}/student-id", json={"student_id": " aatkha200618 "})

    assert response.status_code == 200, response.get_json()
    doc = response.get_json()["document"]
    assert doc["student_id_source"] == "typed" and doc["status"] == "ready"
    assert [page["page_id"] for page in doc["pages"]] == [page_id, declaration]
    # A person's hand is on its stack: a person presses Confirm.
    assert doc["auto_upload"] is False and doc["auto_upload_pending"] is False
    assert doc["auto_upload_refusal"] == "a page of its stack was corrected by hand"
    assert _writes(fake_db) == []


def test_with_batch_codes_an_id_typed_exactly_may_go_up(
    client: FlaskClient, fake_db: Any, students: Students, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    [page_id] = _queue(client)
    _read(page_id, "B1-1", "", None)
    _read(_queue(client, "Scan 2.tiff")[0], None)

    response = client.post(f"/api/pages/{page_id}/student-id", json={"student_id": " aatkha200618 "})

    assert response.status_code == 200, response.get_json()
    doc = response.get_json()["document"]
    assert doc["student_id_source"] == "typed" and doc["status"] == "ready"
    assert doc["auto_upload"] is True and "auto_upload_refusal" not in doc
    assert _writes(fake_db) == []


def test_an_id_typed_that_matched_only_a_near_id_waits(client: FlaskClient) -> None:
    [page_id] = _queue(client)
    _declaration_after(client)
    _read(page_id, None, "", None)
    scan_queue.set_student_id_override(
        page_id, "AATKHA200613", dict(AATIF, matched_via="constrained_repair"), OWNER
    )

    doc = _document(client, _key(page_id))

    assert doc["status"] == "ready" and doc["student_id_source"] == "typed"
    assert doc["auto_upload"] is False


def test_with_batch_codes_every_part_with_an_id_must_be_exact(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    ids, signature = _queue(client, "Scan 1.tiff")[0], _queue(client, "Scan 2.tiff")[0]
    _read(ids, "B1-1", "AATKHA200618", dict(AATIF))
    _read(signature, "B1-1")
    misread, its_signature = _queue(client, "Scan 3.tiff")[0], _queue(client, "Scan 4.tiff")[0]
    _read(misread, "B1-2", "MOHSHA011021", dict(MOHSHA))
    # A part read as another ID, which matched nobody: Ready by the matched
    # part, but the two readings disagree.
    _read(its_signature, "B1-2", "MOHSHA011020")
    tail = _queue(client, "Scan 5.tiff")[0]
    _read(tail, None)

    docs = {d["key"]: d for d in _batches(client)["batches"][0]["documents"]}

    assert docs["B1-1"]["status"] == "ready" and docs["B1-1"]["parts"] == 2
    assert docs["B1-1"]["auto_upload"] is True
    assert docs["B1-2"]["status"] == "ready" and docs["B1-2"]["auto_upload"] is False


# ================================================================== 3. one school
def test_a_page_scanned_for_a_whole_city_waits(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google, school="", location="Pune")

    doc = _document(google, key)

    assert doc["school"] == "" and doc["location"] == "Pune"
    assert doc["status"] == "ready" and doc["auto_upload"] is False
    error = _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)
    assert error.startswith("No single school was picked for it")


def test_with_batch_codes_parts_sent_for_two_schools_wait(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    first = _queue(client, "Scan 1.tiff")[0]
    second = _queue(client, "Scan 2.tiff", school="Other School")[0]
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))
    _read(second, "B1-1")
    _read(_queue(client, "Scan 3.tiff")[0], None)

    doc = _document(client, "B1-1")

    assert doc["status"] == "ready" and doc["auto_upload"] is False


# ================================================================== 4. sent with the switch on
def test_a_page_sent_with_the_switch_off_waits(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google, auto_upload="0")

    doc = _document(google, key)

    assert doc["status"] == "ready" and doc["auto_upload"] is False
    _refused_with_nothing_saved(_auto_confirm(google, key), fake_db, drive)


def test_with_batch_codes_every_part_must_be_sent_with_the_switch_on(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    first = _queue(client, "Scan 1.tiff")[0]
    second = _queue(client, "Scan 2.tiff", auto_upload="0")[0]
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))
    _read(second, "B1-1")
    _read(_queue(client, "Scan 3.tiff")[0], None)

    assert _document(client, "B1-1")["auto_upload"] is False


# ================================================================== the automatic confirm
def test_an_automatic_confirm_marks_the_student_uploads_and_says_it_was_automatic(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google, "Scan 7.tiff")

    response = _auto_confirm(google, key)

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["uploaded"] is True and body["finished"] is True
    doc = body["document"]
    assert doc["status"] == "uploaded" and doc["confirmed_automatically"] is True
    assert doc["auto_upload"] is False
    assert doc["confirmed_by"] == OWNER
    assert doc["message"] == "Uploaded automatically to Drive for Aatif Khan."
    # The whole form: its Student ID page and its Declaration page.
    assert doc["parts"] == 2
    [pdf_file] = drive.uploaded()
    assert pdf_file["name"] == "Aatif Khan_AATKHA200618.pdf"
    with fitz.open(stream=pdf_file["data"], filetype="pdf") as pdf:
        assert pdf.page_count == 2

    # The student was found by the exact ID only: no list of near IDs asked for.
    [update] = fake_db.executed(r"^UPDATE active_student_data SET scanned_at")
    assert update[1][1] == "AATKHA200618"
    assert fake_db.executed(r"^SELECT student_id FROM active_student_data WHERE UPPER\(TRIM\(school_name\)\)") == []
    # The record and History both say who, and that it was automatic.
    record = scan_queue.load_document_record(OWNER, doc["key"])
    assert record is not None
    assert record["automatic"] is True and record["confirmed_by"] == OWNER
    assert record["source"] == "auto:scan:Scan 7.tiff"
    assert record["student"]["matched_via"] == "exact"
    [insert] = fake_db.executed(r"^INSERT INTO ocr_scan_log")
    assert insert[1][6] == "auto:scan:Scan 7.tiff"
    assert insert[1][1] == "AATKHA200618" and insert[1][9] == "exact"

    # Listed as uploaded automatically, with nothing left to confirm.
    [listed] = _batches(google)["batches"][0]["documents"]
    assert listed["key"] == doc["key"] and listed["confirmed_automatically"] is True
    assert scan_queue.open_page_ids(OWNER) == []


def test_a_person_s_confirm_is_not_marked_automatic(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)

    doc = _confirm(google, key=key).get_json()["document"]

    assert doc["confirmed_automatically"] is False and doc["message"] == "Uploaded to Drive for Aatif Khan."
    record = scan_queue.load_document_record(OWNER, doc["key"])
    assert record is not None and record["automatic"] is False
    assert record["source"] == "scan:Scan 1.tiff"
    [insert] = fake_db.executed(r"^INSERT INTO ocr_scan_log")
    assert insert[1][6] == "scan:Scan 1.tiff"


def test_with_batch_codes_a_ready_document_goes_up_by_itself_in_all_its_parts(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    first = _queue(google, "Scan 1.tiff")[0]
    second = _queue(google, "Scan 2.tiff")[0]
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))
    _read(second, "B1-1")
    _read(_queue(google, "Scan 3.tiff")[0], None)
    assert _document(google, "B1-1")["auto_upload"] is True

    response = _auto_confirm(google, "B1-1")

    assert response.status_code == 200, response.get_json()
    doc = response.get_json()["document"]
    assert doc["parts"] == 2 and doc["confirmed_automatically"] is True
    [pdf_file] = drive.uploaded()
    assert pdf_file["name"] == "Aatif Khan_AATKHA200618_B1-1.pdf"
    with fitz.open(stream=pdf_file["data"], filetype="pdf") as pdf:
        assert pdf.page_count == 2
    [insert] = fake_db.executed(r"^INSERT INTO ocr_scan_log")
    assert insert[1][6] == "auto:batch:B1-1"


# ================================================================== the server checks again
def test_an_automatic_confirm_of_a_document_that_changed_since_is_refused(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)
    shown = [page["page_id"] for page in _document(google, key)["pages"]]
    # In another tab, someone typed an ID over the match that matches nobody.
    google.post(f"/api/pages/{shown[0]}/student-id", json={"student_id": "AATKHA200619"})

    response = _auto_confirm(google, key, page_ids=shown)

    error = _refused_with_nothing_saved(response, fake_db, drive)
    assert "not uploaded automatically" in error
    assert scan_queue.open_page_ids(OWNER) == shown


def test_an_automatic_confirm_for_another_student_is_refused(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    students.rows["MOHSHA011021"] = dict(MOHSHA)
    key = _ready_exact(google)

    response = _auto_confirm(google, key, student_id="MOHSHA011021")

    error = _refused_with_nothing_saved(response, fake_db, drive)
    assert "not the one it matched" in error


def test_an_automatic_confirm_checks_again_once_it_holds_the_lock(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _ready_exact(google)
    page_id, shown = key[1:], _parts(google, key)
    real_lock = scan_queue.confirm_lock

    def lock_while_a_student_id_is_typed(owner: str, name: str) -> Any:
        # Someone types an ID over the match while the confirm waits.
        scan_queue.set_student_id_override(page_id, "AATKHA200619", None, OWNER)
        return real_lock(owner, name)

    monkeypatch.setattr(scan_queue, "confirm_lock", lock_while_a_student_id_is_typed)

    response = _auto_confirm(google, key, page_ids=shown)

    _refused_with_nothing_saved(response, fake_db, drive)


def test_an_automatic_confirm_finds_the_student_by_the_exact_id_only(
    google: FlaskClient, fake_db: Any, near_ids: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)
    # Since the page was read, the student's ID was corrected in the
    # database: only a near ID is there now, which a repair would reach.
    near_ids.rows = {"AATKHA200619": dict(AATIF, student_id="AATKHA200619")}

    response = _auto_confirm(google, key)

    error = _refused_with_nothing_saved(response, fake_db, drive)
    assert "no longer exactly a student at ABMPS" in error
    assert fake_db.executed(r"^SELECT student_id FROM active_student_data WHERE UPPER\(TRIM\(school_name\)\)") == []
    assert scan_queue.open_page_ids(OWNER) == sorted(_parts(google, key))


def test_an_automatic_confirm_still_needs_the_parts_shown(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "BATCH_CODES", True)
    first = _queue(google, "Scan 1.tiff")[0]
    second = _queue(google, "Scan 2.tiff")[0]
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))
    _read(second, "B1-1")
    _read(_queue(google, "Scan 3.tiff")[0], None)

    response = _auto_confirm(google, "B1-1", page_ids=[first])

    error = _refused_with_nothing_saved(response, fake_db, drive)
    assert error.startswith("This document changed")


# ================================================================== never twice
def test_an_automatic_confirm_sent_twice_saves_once(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)
    shown = [page["page_id"] for page in _document(google, key)["pages"]]

    first = _auto_confirm(google, key, page_ids=shown)
    again = _auto_confirm(google, key, page_ids=shown)
    by_hand = _confirm(google, key=key, page_ids=shown)

    assert first.status_code == 200
    assert again.status_code == 404 and by_hand.status_code == 404
    assert len(fake_db.executed(r"^UPDATE active_student_data")) == 1
    assert len(fake_db.executed(r"^INSERT INTO ocr_scan_log")) == 1
    assert len(drive.uploaded()) == 1


def test_an_automatic_confirm_while_another_is_running_saves_nothing(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)
    with scan_queue.confirm_lock(OWNER, key):
        busy = _auto_confirm(google, key)

    assert busy.status_code == 409
    assert busy.get_json()["error"] == "This document is already being confirmed."
    assert _writes(fake_db) == [] and drive.calls == []
    assert scan_queue.open_page_ids(OWNER) == sorted(_parts(google, key))


def test_two_tabs_confirming_one_document_by_themselves_at_once_save_it_once(
    app: Flask, google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)
    shown = [page["page_id"] for page in _document(google, key)["pages"]]
    one, two = _tab(app), _tab(app)
    drive.latency = 0.1

    answers = _at_once(
        lambda: _auto_confirm(one, key, page_ids=shown),
        lambda: _auto_confirm(two, key, page_ids=shown),
    )

    assert answers in ([200, 404], [200, 409])
    assert len(fake_db.executed(r"^UPDATE active_student_data")) == 1
    assert len(fake_db.executed(r"^INSERT INTO ocr_scan_log")) == 1
    assert len(drive.uploaded()) == 1


# ================================================================== left for a person
def test_a_drive_failure_leaves_it_confirmed_for_retry_upload_and_not_open_again(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)
    shown = _parts(google, key)
    drive.fail_uploads = True

    body = _auto_confirm(google, key).get_json()

    doc = body["document"]
    assert body["uploaded"] is False and doc["status"] == "confirmed"
    assert doc["can_retry_upload"] is True and doc["confirmed_automatically"] is True
    assert doc["message"].startswith("Saved automatically, but the upload to Drive failed")
    # Nothing is left that the page could confirm by itself again.
    [listed] = _batches(google)["batches"][0]["documents"]
    assert listed["key"] == doc["key"] and listed["auto_upload"] is False
    assert _auto_confirm(google, key, page_ids=shown).status_code == 404

    drive.fail_uploads = False
    retried = google.post(f"/api/documents/{doc['key']}/upload")

    assert retried.status_code == 200, retried.get_json()
    assert retried.get_json()["document"]["message"] == "Uploaded automatically to Drive for Aatif Khan."
    assert len(fake_db.executed(r"^UPDATE active_student_data")) == 1
    [insert] = fake_db.executed(r"^INSERT INTO ocr_scan_log")
    assert insert[1][6].startswith("auto:")


def test_an_automatic_confirm_during_an_outage_saves_nothing_and_stays_for_a_person(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)
    students.down = True

    response = _auto_confirm(google, key)

    assert response.status_code == 503
    assert _writes(fake_db) == [] and drive.calls == []
    assert scan_queue.open_page_ids(OWNER) == sorted(_parts(google, key))


def test_an_automatic_confirm_without_a_google_sign_in_saves_nothing(
    client: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    with client.session_transaction() as session:
        session["signed_in_via"] = "tether"
    key = _ready_exact(client)

    response = _auto_confirm(client, key)

    assert response.status_code == 409
    assert response.get_json()["error"].startswith("Sign in with Google to upload to Drive")
    assert fake_db.statements == [] and drive.calls == []


def test_an_automatic_confirm_with_a_google_sign_in_that_has_run_out_saves_nothing(
    client: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    # Present, so has_drive_credentials is satisfied - but out of date, with
    # nothing to renew it by.
    with client.session_transaction() as session:
        session["google_token"] = {"access_token": "old-access-token", "expires_at": time.time() - 30}
    key = _ready_exact(client)

    response = _auto_confirm(client, key)

    assert response.status_code == 409
    error = response.get_json()["error"]
    assert error.startswith("Your Google sign-in has run out, so it was not uploaded automatically")
    assert "sign in with Google again" in error and "Nothing was saved" in error
    assert fake_db.statements == [] and drive.calls == []
    assert scan_queue.open_page_ids(OWNER) == sorted(_parts(client, key))


def test_an_automatic_confirm_whose_sign_in_google_will_not_renew_saves_nothing(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with google.session_transaction() as session:
        session["google_token"] = dict(session["google_token"], expires_at=time.time() - 30)
    renewals: list[str] = []

    def post(url: str, **kwargs: Any) -> FakeResponse:
        if url == drive_store.TOKEN_URL:
            # The sign-in was revoked.
            renewals.append(url)
            return FakeResponse(400, {"error": "invalid_grant"})
        return drive.post(url, **kwargs)

    monkeypatch.setattr(drive_store.requests, "post", post)
    key = _ready_exact(google)

    response = _auto_confirm(google, key)

    assert renewals == [drive_store.TOKEN_URL]
    assert response.status_code == 409 and "sign-in has run out" in response.get_json()["error"]
    assert fake_db.statements == [] and drive.calls == []


def test_an_automatic_confirm_renews_a_sign_in_that_has_run_out_and_goes_up(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with google.session_transaction() as session:
        session["google_token"] = dict(session["google_token"], expires_at=time.time() - 30)

    def post(url: str, **kwargs: Any) -> FakeResponse:
        if url == drive_store.TOKEN_URL:
            return FakeResponse(200, {"access_token": "renewed-access-token", "expires_in": 3600})
        return drive.post(url, **kwargs)

    monkeypatch.setattr(drive_store.requests, "post", post)
    key = _ready_exact(google)

    response = _auto_confirm(google, key)

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["uploaded"] is True
    with google.session_transaction() as session:
        assert session["google_token"]["access_token"] == "renewed-access-token"


def test_the_rule_is_one_function_the_listing_and_the_confirm_share(client: FlaskClient) -> None:
    key = _ready_exact(client)
    snap = documents.snapshot(OWNER)
    doc = snap.find(key)

    assert doc is not None and documents.auto_upload_refusal(doc) == "" and doc.auto_upload is True
    doc.pages[0].auto_upload = False
    assert documents.auto_upload_refusal(doc) == "it was sent with automatic upload off"
    # Without its Declaration page it is refused whatever its status says.
    doc.pages[0].auto_upload = True
    first, declaration = whole = list(doc.pages)
    doc.pages = whole[:1]
    assert doc.status == "ready" and documents.auto_upload_refusal(doc) == "it has no Declaration page"
    # So it is with two, ...
    doc.pages = [*whole, dataclasses.replace(declaration, page_id="0" * 32)]
    assert documents.auto_upload_refusal(doc) == "it has more than one Declaration page"
    doc.pages = whole
    # ... one attached by hand, ...
    declaration.declaration_of = first.page_id
    assert documents.auto_upload_refusal(doc) == "its Declaration page was attached by hand"
    declaration.declaration_of = ""
    # ... a page between them, ...
    doc.between = 1
    assert documents.auto_upload_refusal(doc) == "a page came between its Student ID page and its Declaration page"
    doc.between = 0
    # ... its Declaration page after it had been shown without one, ...
    arrived = declaration.received_ns
    declaration.received_ns = first.received_ns + (DECLARATION_WAIT_SECONDS + 1) * 1_000_000_000
    assert documents.auto_upload_refusal(doc) == (
        f"its Declaration page came more than {DECLARATION_WAIT_SECONDS} seconds after its Student ID page"
    )
    declaration.received_ns = arrived
    # ... a stack with a page out of place, whichever it is, ...
    for held, why in (
        (documents.ORPHAN_IN_STACK, "its stack has a Declaration page without its Student ID page"),
        (documents.GAP_IN_STACK, "its stack has a page with no Declaration page between it and the next form"),
        (
            documents.AFTER_HELD_STACK,
            "its stack may go on from one scanned before it that has a page out of place",
        ),
    ):
        doc.held = held
        assert documents.auto_upload_refusal(doc) == why and doc.auto_upload_pending is False
    doc.held = ""
    # ... pages not known to come from one scan page ...
    declaration.sender = TWO
    assert documents.auto_upload_refusal(doc) == "it is not known that one computer sent both its pages"
    first.sender = declaration.sender = ""
    assert documents.auto_upload_refusal(doc) == "it is not known that one computer sent both its pages"
    # ... and one from the scan folder with one added by hand.
    first.sender, declaration.sender = ONE, f"{ONE}-1"
    assert documents.auto_upload_refusal(doc) == (
        "one of its pages was added by hand and the other came from the scan folder"
    )
    declaration.sender = ONE
    assert documents.auto_upload_refusal(doc) == "" and doc.auto_upload_pending is False

    # It also waits for its whole stack, as the look found it - for a
    # person, or while it may still come right by itself, when the page is
    # told to keep looking and when it may go up.
    stack = doc.stack
    doc.stack = documents.StackVerdict("its stack has a blank page")
    assert documents.auto_upload_refusal(doc) == "its stack has a blank page"
    assert doc.auto_upload_pending is False and doc.auto_upload_at is None
    settles = time.time_ns() + 60 * 1_000_000_000
    doc.stack = documents.StackVerdict("its stack may still go on", True, settles)
    assert documents.auto_upload_refusal(doc) == "its stack may still go on"
    listed = doc.to_dict()
    assert listed["auto_upload"] is False and listed["auto_upload_pending"] is True
    assert listed["auto_upload_refusal"] == "its stack may still go on"
    expected = datetime.fromtimestamp(settles / 1_000_000_000, documents.IST).replace(tzinfo=None)
    assert listed["auto_upload_at"] == expected.isoformat(timespec="seconds")
    doc.stack = documents.StackVerdict("its stack ends with a Student ID page so far", True)
    assert doc.auto_upload_pending is True and doc.auto_upload_at is None
    # Only while nothing else stands in its way.
    first.auto_upload = False
    assert documents.auto_upload_refusal(doc) == "it was sent with automatic upload off"
    assert doc.auto_upload_pending is False
    first.auto_upload = True
    doc.stack = stack
    assert documents.auto_upload_refusal(doc) == ""


# ================================================================== the page
# What /api/batches answers without batch codes, and the page's own pieces.
_AUTO_JS = r"""
  const RAVI = { student_id: "RAVKUM020315", student_name: "Ravi Kumar", school_name: "School A", grade_name: "5", division_name: "B" };
  function listing(documents) {
    const pages = [];
    documents.forEach((d) => pages.push(...d.pages));
    const groups = documents.length ? [{ batch: null, label: "", documents, counts: {} }] : [];
    return { batches: groups, unsorted: [], pages, queue: { queued: 0, reading: 0, failed: 0 } };
  }
  // A form - its Student ID page `id`, then its Declaration page `id`d -
  // Ready and, unless `extra` says otherwise, one the server lists as going
  // up by itself.
  function one(id, student, extra) {
    const page = pg(id, { source_name: "Scan " + id + ".jpg", student_id: student.student_id, matched_student: student, skip_reason: "" });
    const declaration = pg(id + "d", { source_name: "Scan " + id + "d.jpg", skip_reason: "declaration", message: "Declaration page, no Student ID." });
    return Object.assign({
      key: "P" + id, code: "", parts: 2, pages: [page, declaration], student_id: student.student_id, matched_student: student,
      status: "ready", can_confirm: true, auto_upload: true,
    }, extra || {});
  }
  // The page, with School A of Pune - and anything else - remembered.
  async function openWith(local) {
    const page = loadPage({ local: Object.assign({ "ocr.school": "School A", "ocr.location": "Pune" }, local || {}) });
    await answerSchools(page);
    await answer(page.calls, "/api/inbox/status", { configured: false });
    return page;
  }
  // A request the page has made and is still waiting on, left unanswered.
  async function waitingFor(page, match) {
    for (let i = 0; i < 600; i += 1) {
      const found = page.calls.find((c) => c.url.includes(match));
      if (found) return found;
      await tick(5);
    }
    throw new Error("no request for " + match);
  }
  function confirms(page) {
    return page.log.filter((c) => c.url.endsWith("/confirm")).map((c) => [c.url, JSON.parse(c.opts.body)]);
  }
  function setSwitch(page, on) {
    const toggle = page.byId["auto-upload"];
    toggle.checked = on;
    toggle.fire("change");
  }
"""


@needs_node
def test_page_confirms_a_document_the_server_lists_as_auto_upload_once(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Confirmed by itself as soon as it is listed; never again, however often the list is asked for."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const page = await openWith();
      const eligible = listing([one("p1", ASHA)]);
      await answerBatches(page, eligible);
      const sent = await waitingFor(page, "/api/documents/Pp1/confirm");
      const pending = textIn(card(page, "Scan p1.jpg"));
      // Another look while it is in the air, and one after it answered with
      // the server still listing it open: neither confirms it again.
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, eligible);
      await tick(20);
      const whileInAir = confirms(page).length;
      await answer(page.calls, "/api/documents/Pp1/confirm", {
        document: { status: "uploaded", key: "Pp1~ab12cd34", confirmed_automatically: true,
          drive_folder: "OCR/School A/5/B/Asha Patil_MOHSHA011021", drive_link: "https://drive.google.com/x" },
        uploaded: true, finished: true,
      });
      const said = page.byId["ocr-live"].textContent;
      await answerBatches(page, eligible);
      await tick(20);
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, listing([one("p1", ASHA, {
        key: "Pp1~ab12cd34", status: "uploaded", can_confirm: false, auto_upload: false, confirmed_automatically: true,
        drive_folder: "OCR/School A/5/B/Asha Patil_MOHSHA011021", drive_link: "https://drive.google.com/x",
      })]));
      await tick(20);
      return { url: sent.url, body: JSON.parse(sent.opts.body), pending, whileInAir, said,
               confirms: confirms(page).length, done: textIn(card(page, "Scan p1.jpg")) };
    """)

    assert out["url"] == "/api/documents/Pp1/confirm"
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p1d"], "automatic": True}
    assert "uploading it to your Drive automatically" in out["pending"]
    assert out["whileInAir"] == 1
    assert "uploaded to your Drive automatically" in out["said"]
    assert out["confirms"] == 1
    assert "Uploaded automatically" in out["done"]
    assert "Asha Patil_MOHSHA011021" in out["done"]


@needs_node
def test_page_never_uploads_a_student_id_page_by_itself_without_its_declaration_page(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Nothing goes up while the form's Declaration page may still be on its way; once it is with it, both go up."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const page = await openWith();
      // As the server lists a Student ID page with nothing read after it yet.
      const alone = one("p1", ASHA, { parts: 1, status: "needs_attention", attention: ["no_declaration"], auto_upload: false });
      alone.pages = alone.pages.slice(0, 1);
      // And one an answer lists as going up by itself - out of date, say -
      // all the same.
      const listedAlone = one("p3", RAVI, { parts: 1 });
      listedAlone.pages = listedAlone.pages.slice(0, 1);
      await answerBatches(page, listing([alone, listedAlone]));
      await tick(20);
      const before = confirms(page).length;
      // The form's Declaration page came and has been read.
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, listing([one("p1", ASHA), listedAlone]));
      const sent = await waitingFor(page, "/api/documents/Pp1/confirm");
      await answer(page.calls, "/api/documents/Pp1/confirm", {
        document: { status: "uploaded", key: "Pp1~ab12cd34", confirmed_automatically: true }, uploaded: true, finished: true,
      });
      await answerBatches(page, listing([listedAlone]));
      await tick(20);
      return { before, body: JSON.parse(sent.opts.body), all: confirms(page).map(([url]) => url) };
    """)

    assert out["before"] == 0
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p1d"], "automatic": True}
    assert out["all"] == ["/api/documents/Pp1/confirm"]


@needs_node
def test_page_sends_the_switch_with_every_scan_and_remembers_it(client: FlaskClient, tmp_path: Path) -> None:
    """On by default; sent as 1 or 0 with each upload; kept by the browser - and fine when it keeps nothing."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      // A browser that keeps what it is told across visits, and one whose
      // storage refuses everything (a private window, or blocked site data).
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
        const page = loadPage({ local });
        await answer(page.calls, "/api/schools", { schools: [{ school: "School A", location: "Pune" }] });
        await answer(page.calls, "/api/inbox/status", { configured: false });
        await answerBatches(page, EMPTY);
        return page;
      };
      const send = async (page, name, id) => {
        await uploadFile(page, name);
        return sentForm(await answerScan(page, [id]));
      };

      const first = await visit(KEPT);
      const startsOn = first.byId["auto-upload"].checked;
      const on = await send(first, "a.jpg", "p1");
      setSwitch(first, false);
      const off = await send(first, "b.jpg", "p2");
      const remembered = kept.get("ocr.autoUpload");

      const second = await visit(KEPT);
      const secondOn = second.byId["auto-upload"].checked;
      const stillOff = await send(second, "c.jpg", "p3");
      setSwitch(second, true);
      const backOn = await send(second, "d.jpg", "p4");

      const third = await visit(BROKEN);
      const brokenOn = third.byId["auto-upload"].checked;
      setSwitch(third, false);
      const brokenOff = await send(third, "e.jpg", "p5");
      makeStorage = realStorage;
      return { startsOn, on, off, remembered, secondOn, stillOff, backOn, rememberedOn: kept.get("ocr.autoUpload"),
               brokenOn, brokenOff };
    """)

    assert out["startsOn"] is True
    # With the page's id for files added by hand (test_page_sends_its_own_id_with_every_scan).
    assert re.fullmatch(r"[0-9a-f]{32}-1", out["on"].pop("sender"))
    assert out["on"] == {"file": "a.jpg", "school": "School A", "location": "Pune", "auto_upload": "1"}
    assert out["off"]["auto_upload"] == "0"
    assert out["remembered"] == "off"
    assert out["secondOn"] is False and out["stillOff"]["auto_upload"] == "0"
    assert out["backOn"]["auto_upload"] == "1" and out["rememberedOn"] == "on"
    assert out["brokenOn"] is True and out["brokenOff"]["auto_upload"] == "0"


@needs_node
def test_page_leaves_every_other_document_for_a_person(client: FlaskClient, tmp_path: Path) -> None:
    """Nothing the server does not list as auto_upload, nothing while the switch is off, no card being typed in."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const page = await openWith({ "ocr.autoUpload": "off" });
      const docs = listing([
        one("p1", ASHA, { auto_upload: undefined }),
        one("p2", ASHA, { auto_upload: false }),
        one("p3", ASHA, { status: "needs_attention", attention: ["same_student"] }),
        one("p4", RAVI),
        one("p5", ASHA),
      ]);
      await answerBatches(page, docs);
      await tick(20);
      const whileOff = confirms(page).length;
      // Someone types another ID into the fourth before switching it on.
      const typing = fieldIn(card(page, "Scan p4.jpg"), "id");
      typing.value = "MOHSHA011021";
      typing.fire("input");
      setSwitch(page, true);
      const sent = await waitingFor(page, "/confirm");
      await answer(page.calls, "/api/documents/Pp5/confirm", {
        document: { status: "uploaded", key: "Pp5~ab12cd34", confirmed_automatically: true }, uploaded: true, finished: true,
      });
      await answerBatches(page, docs);
      await tick(20);
      return { whileOff, url: sent.url, all: confirms(page), switchOn: page.byId["auto-upload"].checked };
    """)

    assert out["whileOff"] == 0
    assert out["switchOn"] is True
    assert out["url"] == "/api/documents/Pp5/confirm"
    assert out["all"] == [
        ["/api/documents/Pp5/confirm", {"student_id": "MOHSHA011021", "page_ids": ["p5", "p5d"], "automatic": True}],
    ]


@needs_node
def test_page_does_not_try_again_after_a_refusal_an_outage_or_a_drive_failure(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Each document is tried once a visit; what failed stays on its card for Confirm or Retry upload."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const page = await openWith();
      const docs = listing([one("p1", ASHA), one("p2", RAVI), one("p3", ASHA)]);
      await answerBatches(page, docs);
      await answer(page.calls, "/api/documents/Pp1/confirm",
        { error: "The student database could not be reached. Nothing was saved - try again." }, 503);
      await answer(page.calls, "/api/documents/Pp2/confirm",
        { error: "Its Student ID is not an exact match, so it was not uploaded automatically. Nothing was saved - check it and press Confirm and upload." }, 409);
      await answer(page.calls, "/api/documents/Pp3/confirm", {
        document: { status: "confirmed", key: "Pp3~ab12cd34", confirmed_automatically: true },
        uploaded: false, finished: false,
        message: "Saved automatically, but the upload to Drive failed. Use Retry upload.",
      });
      for (let i = 0; i < 3; i += 1) {
        await answerBatches(page, docs);
        await tick(20);
        page.byId["ocr-refresh"].fire("click");
      }
      await answerBatches(page, docs);
      await tick(20);
      const tried = confirms(page).map(([url]) => url);
      const outage = card(page, "Scan p1.jpg");
      const refused = card(page, "Scan p2.jpg");
      const drive = card(page, "Scan p3.jpg");
      const shown = {
        outage: textIn(outage), outageButton: Boolean(buttonIn(outage, "Confirm and upload")),
        drive: textIn(drive), retryButton: Boolean(buttonIn(drive, "Retry upload")),
        refused: textIn(refused),
        uploads: page.log.filter((c) => c.url.endsWith("/upload")).length,
      };
      // A person confirms the first one after all: an ordinary confirm.
      buttonIn(outage, "Confirm and upload").fire("click");
      const byHand = await answer(page.calls, "/api/documents/Pp1/confirm", { document: { status: "uploaded", key: "Pp1~cd34ab12" } });
      return { tried, shown, byHand: JSON.parse(byHand.opts.body) };
    """)

    assert out["tried"] == [
        "/api/documents/Pp1/confirm", "/api/documents/Pp2/confirm", "/api/documents/Pp3/confirm",
    ]
    shown = out["shown"]
    assert "Not uploaded automatically: The student database could not be reached." in shown["outage"]
    assert shown["outageButton"] is True
    assert "upload to Drive failed" in shown["drive"] and shown["retryButton"] is True
    assert shown["uploads"] == 0
    assert "Its Student ID is not an exact match, so it was not uploaded automatically." in shown["refused"]
    assert "Not uploaded automatically: Its" not in shown["refused"]
    assert out["byHand"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p1d"]}


@needs_node
def test_page_stops_uploading_by_itself_when_drive_needs_a_google_sign_in(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Signed in through Tether there is no Drive to upload to: the first refusal stops the rest."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const page = await openWith();
      const docs = listing([one("p1", ASHA), one("p2", RAVI)]);
      await answerBatches(page, docs);
      await answer(page.calls, "/api/documents/Pp1/confirm",
        { error: "Sign in with Google to upload to Drive. Nothing was saved." }, 409);
      await answerBatches(page, docs);
      await tick(20);
      const first = card(page, "Scan p1.jpg");
      const stopped = page.byId["auto-upload-stopped"];
      return { tried: confirms(page).map(([url]) => url), text: textIn(first), links: linksIn(first),
               stopped: stopped.textContent, stoppedShown: !stopped.hidden };
    """)

    assert out["tried"] == ["/api/documents/Pp1/confirm"]
    assert "Sign in with Google to upload to Drive" in out["text"]
    assert "/login/google" in out["links"]
    assert out["stoppedShown"] is True
    assert out["stopped"].startswith("Automatic upload stopped for this visit. Drive needs you to sign in with Google.")


@needs_node
def test_page_stops_uploading_by_itself_after_a_drive_failure(client: FlaskClient, tmp_path: Path) -> None:
    """Saved without its PDF - a sign-in run out, say: the rest would go the same way, so none is sent."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const page = await openWith();
      const docs = listing([one("p1", ASHA), one("p2", RAVI)]);
      await answerBatches(page, docs);
      const before = page.byId["auto-upload-stopped"].textContent;
      await answer(page.calls, "/api/documents/Pp1/confirm", {
        document: { status: "confirmed", key: "Pp1~ab12cd34", confirmed_automatically: true },
        uploaded: false, finished: false,
        message: "Saved automatically, but the upload to Drive failed: Google Drive refused to list your folders. Use Retry upload.",
      });
      for (let i = 0; i < 3; i += 1) {
        await answerBatches(page, docs);
        await tick(20);
        page.byId["ocr-refresh"].fire("click");
      }
      await answerBatches(page, docs);
      await tick(20);
      const waiting = card(page, "Scan p2.jpg");
      const stopped = page.byId["auto-upload-stopped"];
      return {
        before, tried: confirms(page).map(([url]) => url), stopped: stopped.textContent, stoppedShown: !stopped.hidden,
        retry: Boolean(buttonIn(card(page, "Scan p1.jpg"), "Retry upload")),
        waitingConfirm: Boolean(buttonIn(waiting, "Confirm and upload")), waiting: textIn(waiting),
      };
    """)

    assert out["before"] == ""
    assert out["tried"] == ["/api/documents/Pp1/confirm"]
    assert out["stoppedShown"] is True
    assert out["stopped"] == (
        "Automatic upload stopped for this visit. Scan p1.jpg was saved, but Drive did not take its PDF"
        " - press Retry upload on it. Reload the page to start it again."
    )
    assert out["retry"] is True
    assert out["waitingConfirm"] is True and "Ready to confirm" in out["waiting"]


@needs_node
def test_page_leaves_a_card_whose_id_box_was_emptied(client: FlaskClient, tmp_path: Path) -> None:
    """A person who clears the box to type the right ID is not overtaken by the match they erased."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const page = await openWith({ "ocr.autoUpload": "off" });
      const docs = listing([one("p1", ASHA), one("p2", RAVI)]);
      await answerBatches(page, docs);
      await tick(20);
      const box = fieldIn(card(page, "Scan p2.jpg"), "id");
      box.value = "";
      box.fire("input");
      setSwitch(page, true);
      await answer(page.calls, "/api/documents/Pp1/confirm", {
        document: { status: "uploaded", key: "Pp1~ab12cd34", confirmed_automatically: true }, uploaded: true, finished: true,
      });
      for (let i = 0; i < 2; i += 1) {
        await answerBatches(page, docs);
        await tick(20);
        page.byId["ocr-refresh"].fire("click");
      }
      await answerBatches(page, docs);
      await tick(20);
      return { all: confirms(page), box: fieldIn(card(page, "Scan p2.jpg"), "id").value };
    """)

    assert out["all"] == [
        ["/api/documents/Pp1/confirm", {"student_id": "MOHSHA011021", "page_ids": ["p1", "p1d"], "automatic": True}],
    ]
    assert out["box"] == ""


@needs_node
def test_page_never_uploads_by_itself_a_form_that_waits_is_held_was_attached_or_has_two_declaration_pages(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Even listed as auto_upload - an answer out of date, say - only a whole form as the reader put it together goes up."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const page = await openWith();
      const waiting = one("p1", ASHA, { status: "waiting", can_confirm: false, parts: 1,
        message: "Waiting for its Declaration page - nothing has come after it yet." });
      waiting.pages = waiting.pages.slice(0, 1);
      // Its stack has a Declaration page out of place: listed Ready all the same, and checked.
      const held = one("p2", RAVI, { attention: ["orphan_in_stack"] });
      const heldChecked = one("p3", RAVI, { status: "needs_attention", attention: ["orphan_in_stack"] });
      const attached = one("p4", RAVI);
      attached.pages[1] = pg("p4d", { source_name: "Scan p4d.jpg", skip_reason: "", declaration_by: "staff@akanksha.org" });
      const two = one("p5", RAVI, { parts: 3 });
      two.pages.push(pg("p5e", { source_name: "Scan p5e.jpg", skip_reason: "declaration" }));
      const whole = one("p6", ASHA);
      const docs = [waiting, held, heldChecked, attached, two];
      await answerBatches(page, listing([...docs, whole]));
      const sent = await waitingFor(page, "/api/documents/Pp6/confirm");
      await answer(page.calls, "/api/documents/Pp6/confirm", {
        document: { status: "uploaded", key: "Pp6~ab12cd34", confirmed_automatically: true }, uploaded: true, finished: true,
      });
      for (let i = 0; i < 2; i += 1) {
        await answerBatches(page, listing(docs));
        await tick(20);
        page.byId["ocr-refresh"].fire("click");
      }
      await answerBatches(page, listing(docs));
      await tick(20);
      const card1 = card(page, "Scan p1.jpg");
      return { all: confirms(page).map(([url]) => url), body: JSON.parse(sent.opts.body),
               waiting: textIn(card1), waitingConfirm: Boolean(buttonIn(card1, "Confirm and upload")) };
    """)

    assert out["all"] == ["/api/documents/Pp6/confirm"]
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p6", "p6d"], "automatic": True}
    assert "Waiting for its Declaration page" in out["waiting"] and out["waitingConfirm"] is False


@needs_node
def test_page_never_uploads_by_itself_while_scans_are_still_coming(client: FlaskClient, tmp_path: Path) -> None:
    """Without batch codes too: the sheet after a form's Declaration page may still be on its way, and out of place."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const page = await openWith();
      const reading = listing([one("p1", ASHA)]);
      reading.queue = { queued: 0, reading: 1, failed: 0 };
      await answerBatches(page, reading);
      await tick(20);
      const whileReading = confirms(page).length;
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, listing([one("p1", ASHA)]));
      const sent = await waitingFor(page, "/api/documents/Pp1/confirm");
      return { whileReading, body: JSON.parse(sent.opts.body) };
    """)

    assert out["whileReading"] == 0
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p1d"], "automatic": True}


@needs_node
def test_page_keeps_looking_while_a_form_may_still_go_up_by_itself(client: FlaskClient, tmp_path: Path) -> None:
    """Nothing after a form's Declaration page yet: the page asks again until it may go up - but not for one that waits for a person."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const open = async (clock, local) => {
        const page = loadPage({ clock, local: Object.assign({ "ocr.school": "School A", "ocr.location": "Pune" }, local || {}) });
        await answerSchools(page);
        await answer(page.calls, "/api/inbox/status", { configured: false });
        return page;
      };
      const looks = (page) => page.log.filter((c) => c.url.includes("/api/batches")).length;
      const settling = listing([one("p1", ASHA, { auto_upload: false, auto_upload_pending: true })]);

      const clock = makeClock();
      const page = await open(clock);
      await answerBatches(page, settling);
      await clock.advance(3000);
      await answerBatches(page, settling);
      await clock.advance(3000);
      const looked = looks(page);
      await answerBatches(page, listing([one("p1", ASHA)]));
      const sent = await waitingFor(page, "/api/documents/Pp1/confirm");

      // Ready, but for a person to confirm: no more looks by themselves.
      const idleClock = makeClock();
      const idle = await open(idleClock);
      await answerBatches(idle, listing([one("p1", ASHA, { auto_upload: false })]));
      await idleClock.advance(10000);
      // Settling, with the switch off: none either - until it is on again.
      const offClock = makeClock();
      const off = await open(offClock, { "ocr.autoUpload": "off" });
      await answerBatches(off, settling);
      await offClock.advance(10000);
      const whileOff = looks(off);
      setSwitch(off, true);
      await offClock.advance(3000);
      return { looked, body: JSON.parse(sent.opts.body), idle: looks(idle), off: whileOff, backOn: looks(off) };
    """)

    assert out["looked"] == 3
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p1d"], "automatic": True}
    assert out["idle"] == 1 and out["off"] == 1 and out["backOn"] == 2


@needs_node
def test_page_sends_its_own_id_with_every_scan(client: FlaskClient, tmp_path: Path) -> None:
    """One id a visit, sent with every scan - with "-1" on each file added by hand, however many picks; a reload makes a new id."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const idb = makeIndexedDB();
      const clock = makeClock();
      const folder = makeFolder("OCR Scans");
      const page = await openScannerPage({ clock, idb, folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      folder.add("Scan 002.jpg");
      await clock.advance(4000);
      for (const id of ["p1", "p2"]) await answerScan(page, [id]);
      await uploadFile(page, "a.jpg", "b.jpg");
      for (const id of ["p3", "p4"]) await answerScan(page, [id]);
      await uploadFile(page, "c.jpg");
      await answerScan(page, ["p5"]);
      folder.add("Scan 003.jpg");
      await clock.advance(4000);
      await answerScan(page, ["p6"]);
      const sent = everyScan(page).map((call) => [sentName(call), sentForm(call).sender]);
      const again = await openPage();
      await uploadFile(again, "d.jpg");
      return { sent, reloaded: sentForm(await answerScan(again, ["p7"])).sender };
    """)

    sent = dict(out["sent"])
    assert list(sent) == ["Scan 001.jpg", "Scan 002.jpg", "a.jpg", "b.jpg", "c.jpg", "Scan 003.jpg"]
    own = sent["Scan 001.jpg"]
    assert re.fullmatch(r"[0-9a-f]{32}", own) and sent["Scan 002.jpg"] == sent["Scan 003.jpg"] == own
    # One stack with the scan folder's, so two picks of one form pair - but
    # marked, so a form of one of each never goes up by itself.
    assert [sent[name] for name in ("a.jpg", "b.jpg", "c.jpg")] == [f"{own}-1", f"{own}-1", f"{own}-1"]
    assert all(scan_queue.is_sender(sender) for sender in sent.values())
    assert re.fullmatch(r"[0-9a-f]{32}-1", out["reloaded"]) and not out["reloaded"].startswith(own)


# With batch codes on: /api/batches as test_app.py's helpers build it.
_CODED_JS = r"""
  function coded(pages, extra) {
    return batches([[1, [doc("B1-1", pages.map((id) => pg(id, { code: "B1-1", student_id: ASHA.student_id, matched_student: ASHA })),
      Object.assign({ key: "B1-1", student_id: ASHA.student_id, matched_student: ASHA, status: "ready", can_confirm: true,
        auto_upload: true }, extra || {}))]]]);
  }
"""


@needs_node
def test_page_with_batch_codes_waits_for_a_scan_held_for_a_retry(
    client: FlaskClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The form's second page is held for a retry: the first is not filed alone, and both go up once it is in."""
    monkeypatch.setattr(config, "BATCH_CODES", True)
    out = _run_page(client, tmp_path, _AUTO_JS + _CODED_JS + r"""
      const clock = makeClock();
      const page = loadPage({ clock, local: { "ocr.school": "School A", "ocr.location": "Pune" } });
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", { configured: false });
      await answerBatches(page, EMPTY);
      await uploadFile(page, "a.jpg", "b.jpg");
      await answerScan(page, ["p1"]);
      await answer(page.calls, "/api/scans", { error: "The server is busy - try again." }, 503);
      // The server lists B1-1 Ready with its first part only - all it has.
      await clock.advance(3000);
      await answerBatches(page, coded(["p1"]));
      await tick(20);
      const whileHeld = confirms(page).length;
      // The retry gets the second part through, and it is read.
      await clock.advance(10000);
      await answerScan(page, ["p2"]);
      await clock.advance(3000);
      await answerBatches(page, coded(["p1", "p2"]));
      const sent = await waitingFor(page, "/api/documents/B1-1/confirm");
      return { whileHeld, body: JSON.parse(sent.opts.body) };
    """)

    assert out["whileHeld"] == 0
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p2"], "automatic": True}


@needs_node
def test_page_with_batch_codes_waits_for_a_scan_still_settling_in_the_folder(
    client: FlaskClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scan the scanner is still saving may be the form's last part: nothing goes up until it is in."""
    monkeypatch.setattr(config, "BATCH_CODES", True)
    out = _run_page(client, tmp_path, _AUTO_JS + _CODED_JS + r"""
      const clock = makeClock();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      await clock.advance(4000);
      await answerScan(page, ["p1"]);
      // The scanner starts on the form's second page before the results come.
      folder.add("Scan 002.jpg");
      await clock.advance(3000);
      await answerBatches(page, coded(["p1"]));
      await tick(20);
      const whileSaving = confirms(page).length;
      await clock.advance(4000);
      await answerScan(page, ["p2"]);
      await clock.advance(3000);
      await answerBatches(page, coded(["p1", "p2"]));
      const sent = await waitingFor(page, "/api/documents/B1-1/confirm");
      return { whileSaving, body: JSON.parse(sent.opts.body), sentNames: everyScan(page).map(sentName) };
    """)

    assert out["whileSaving"] == 0
    assert out["sentNames"] == ["Scan 001.jpg", "Scan 002.jpg"]
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p2"], "automatic": True}
