"""The Drive inbox watcher: every page logged once, under the batch it arrived in.

Drive, OCR and the student lookup are stood in for, and the three tables the
watcher writes (batches, inbox files, the scan log) live in `Scene` below, so
these tests run whole passes the way the service does - against nothing real.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pymysql  # type: ignore[import-untyped]  # no stubs installed
import pytest
import requests
from flask.testing import FlaskClient
from google.auth.exceptions import RefreshError

from server import inbox_watcher
from server.services import batch_store, drive_inbox
from server.services.clock import now_ist
from server.services.extract_student_id import ExtractResult, LookupUnavailable

# The real rename, kept before the scene stands in for Drive, for the tests
# that need what it does when the network fails underneath it.
REAL_MARK_DONE = drive_inbox.mark_done

FOLDER = "folder-1"
OWNER = "staff@akanksha.org"  # the signed-in user of the `client` fixture
IST_OFFSET = timedelta(hours=5, minutes=30)
FORBIDDEN = "Drive PATCH files/x -> 403: The user does not have sufficient permissions"

LOG_COLUMNS = (
    "created_at", "confirmed_student_id", "found_in_db", "matched_student_id",
    "student_name", "school_name", "source", "message", "image_link",
    "matched_via", "scan_school", "scan_location",
)

Rows = tuple[list[dict[str, Any]], int | None]


# --------------------------------------------------------------- the scene
@dataclass
class Scene:
    """One inbox folder, its batches, Drive, OCR and the student database."""

    batches: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)  # ocr_inbox_files
    log: list[dict[str, Any]] = field(default_factory=list)  # ocr_scan_log
    log_inserts: int = 0
    fail_log_inserts: set[int] = field(default_factory=set)  # 1-based attempt numbers
    lock_free: bool = True
    lock_mine: bool = True  # False once MySQL has dropped the lock's connection
    lock_calls: list[str] = field(default_factory=list)
    session_settings: list[tuple[Any, ...]] = field(default_factory=list)

    drive: dict[str, dict[str, Any]] = field(default_factory=dict)
    pages: dict[str, list[str]] = field(default_factory=dict)  # file id -> IDs read
    rename_failures: int = 0  # how many renames Drive refuses; -1 for all of them
    renames: list[tuple[str, str]] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)
    listings: int = 0

    students: dict[str, str] = field(default_factory=dict)  # Student ID -> name
    # Student ID -> the school that has them. A student not listed here is
    # found whichever school is asked.
    schools: dict[str, str] = field(default_factory=dict)
    lookup_down: bool = False
    lookups: list[tuple[str, str | None]] = field(default_factory=list)
    said: list[str] = field(default_factory=list)

    # -- set-up
    def add_batch(
        self, school: str, *, opened: datetime, closed: datetime | None = None
    ) -> int:
        batch_id = 7 + len(self.batches)
        self.batches.append({
            "id": batch_id, "folder_id": FOLDER, "owner_email": OWNER,
            "school": school, "location": "Mumbai", "opened_by": OWNER,
            "opened_at": opened, "closed_at": closed, "pages_done": 0,
        })
        return batch_id

    def add_file(
        self, file_id: str, name: str, ids: list[str], *, created: datetime | None = None
    ) -> None:
        entry: dict[str, Any] = {"id": file_id, "name": name, "size": "2048",
                                 "mimeType": "image/jpeg"}
        if created is not None:
            # Drive reports createdTime in UTC; the batches are kept in IST.
            entry["createdTime"] = (created - IST_OFFSET).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        self.drive[file_id] = entry
        self.pages[file_id] = ids

    def batch(self, batch_id: int) -> dict[str, Any]:
        return next(b for b in self.batches if b["id"] == batch_id)

    def name_of(self, file_id: str) -> str:
        return str(self.drive[file_id]["name"])

    # -- the database
    def handle(self, sql: str, params: Any) -> Rows:
        p = tuple(params or ())
        if sql.startswith("SELECT * FROM ocr_scan_batches"):
            return self._select_batches(sql, p), None
        if sql == "UPDATE ocr_scan_batches SET pages_done = pages_done + %s WHERE id = %s":
            self.batch(p[1])["pages_done"] += p[0]
            return [], 1
        if sql == "UPDATE ocr_scan_batches SET closed_at = %s WHERE id = %s AND closed_at IS NULL":
            row = self.batch(p[1])
            if row["closed_at"] is not None:
                return [], 0
            row["closed_at"] = p[0]
            return [], 1
        if sql.startswith("SELECT * FROM ocr_inbox_files WHERE file_id IN"):
            return [dict(self.files[f]) for f in p if f in self.files], None
        if sql.startswith("INSERT INTO ocr_inbox_files"):
            file_id, folder_id, batch_id, name, state, updated = p
            self.files[file_id] = {
                "file_id": file_id, "folder_id": folder_id, "batch_id": batch_id,
                "name": name, "state": state, "pages_logged": 0, "new_name": None,
                "attempts": 0, "last_error": None, "updated_at": updated,
            }
            return [], 1
        if sql.startswith("UPDATE ocr_inbox_files SET pages_logged = %s"):
            pages, updated, file_id = p
            self.files[file_id].update(pages_logged=pages, updated_at=updated)
            return [], 1
        if sql.startswith("UPDATE ocr_inbox_files SET state = %s, new_name = %s"):
            state, new_name, pages, updated, file_id, expected = p
            claimed = self.files.get(file_id)
            if claimed is None or claimed["state"] != expected:
                return [], 0
            claimed.update(state=state, new_name=new_name, pages_logged=pages, attempts=0,
                           last_error=None, updated_at=updated)
            return [], 1
        if sql.startswith("UPDATE ocr_inbox_files SET attempts = attempts + 1"):
            why, updated, file_id = p
            if file_id not in self.files:
                return [], 0
            row = self.files[file_id]
            row.update(attempts=row["attempts"] + 1, last_error=why, updated_at=updated)
            return [], 1
        if sql.startswith("SELECT attempts, state FROM ocr_inbox_files"):
            known = self.files.get(p[0])
            return ([{"attempts": known["attempts"], "state": known["state"]}] if known else []), None
        if sql.startswith("UPDATE ocr_inbox_files SET state = %s, updated_at = %s"):
            state, updated, file_id = p
            self.files[file_id].update(state=state, updated_at=updated)
            return [], 1
        if sql.startswith("INSERT INTO ocr_scan_log"):
            self.log_inserts += 1
            if self.log_inserts in self.fail_log_inserts:
                raise pymysql.err.OperationalError(2013, "Lost connection to MySQL server")
            self.log.append(dict(zip(LOG_COLUMNS, p)))
            return [], 1
        if sql.startswith("INSERT INTO ocr_inbox_watcher"):
            return [], 1
        if sql == "SET SESSION wait_timeout = %s":
            self.session_settings.append(p)
            return [], 0
        for call in ("GET_LOCK", "IS_USED_LOCK", "RELEASE_LOCK"):
            if call in sql:
                self.lock_calls.append(call)
                return [{"got": int(self.lock_free), "mine": int(self.lock_mine),
                         "released": 1}], None
        raise AssertionError(f"SQL the watcher tests do not know: {sql}")

    def _select_batches(self, sql: str, p: tuple[Any, ...]) -> list[dict[str, Any]]:
        where = re.sub(r"^SELECT \* FROM ocr_scan_batches WHERE ", "", sql)
        rows = self.batches
        if where == "closed_at IS NULL ORDER BY id":
            picked = [b for b in rows if b["closed_at"] is None]
        elif where == "closed_at IS NULL OR closed_at >= %s ORDER BY id":
            picked = [b for b in rows if b["closed_at"] is None or b["closed_at"] >= p[0]]
        elif where == "folder_id = %s ORDER BY id":
            picked = [b for b in rows if b["folder_id"] == p[0]]
        elif where == "folder_id = %s AND (closed_at IS NULL OR closed_at > %s) ORDER BY id":
            picked = [b for b in rows if b["folder_id"] == p[0]
                      and (b["closed_at"] is None or b["closed_at"] > p[1])]
        elif where == "folder_id = %s AND closed_at <= %s ORDER BY id DESC LIMIT 1":
            picked = [b for b in rows if b["folder_id"] == p[0]
                      and b["closed_at"] is not None and b["closed_at"] <= p[1]][-1:]
        elif where == "id = %s":
            picked = [b for b in rows if b["id"] == p[0]]
        elif where == "owner_email = %s AND closed_at IS NULL ORDER BY id DESC LIMIT 1":
            picked = [b for b in rows if b["owner_email"] == p[0] and b["closed_at"] is None][-1:]
        else:
            raise AssertionError(f"batch query the watcher tests do not know: {sql}")
        return [dict(b) for b in sorted(picked, key=lambda b: b["id"])]

    # -- Drive
    def list_new_pages(self, folder_id: str) -> list[dict[str, Any]]:
        self.listings += 1
        return [dict(e) for e in self.drive.values() if not e["name"].startswith("[")]

    def download(self, file_id: str) -> bytes:
        self.downloads.append(file_id)
        return file_id.encode()

    def mark_done(self, file_id: str, new_name: str) -> str:
        if self.rename_failures:
            if self.rename_failures > 0:
                self.rename_failures -= 1
            raise drive_inbox.DriveInboxError(FORBIDDEN)
        self.drive[file_id]["name"] = new_name
        self.renames.append((file_id, new_name))
        return f"https://drive.google.com/file/d/{file_id}/view"

    # -- OCR and the student lookup
    def read(self, data: bytes, filename: str) -> list[ExtractResult]:
        ids = self.pages[data.decode()]
        return [
            ExtractResult(
                student_id=sid, confidence=0.9 if sid else 0.0, form_language="en",
                source="upload", message="ok" if sid else "no Student ID found on this page",
                error=None if sid else "no Student ID found on this page",
                page_number=n, page_count=len(ids),
            )
            for n, sid in enumerate(ids, start=1)
        ]

    def confirm(
        self, student_id: str, source: str = "manual-confirm", *,
        image_link: str | None = None, defer_log: bool = False,
        school: str | None = None, location: str | None = None,
    ) -> ExtractResult:
        if self.lookup_down:
            raise LookupUnavailable()
        self.lookups.append((student_id, school))  # where scanned_at is written
        result = ExtractResult(
            student_id=student_id, confidence=1.0, form_language="manual", source=source,
            label_text="user-confirmed", message="ok", ocr_student_id=student_id,
        )
        name = self.students.get(student_id)
        if name and self.schools.get(student_id, school) != school:
            name = None  # a real student, but not at the school asked about
        if name:
            result.found_in_db = True
            result.student = {"student_id": student_id, "student_name": name,
                              "school_name": school, "matched_via": "exact"}
        else:
            result.message = "student_id_not_found_in_db"
        return result

    # -- what came of it
    def rows_for(self, student_id: str) -> list[dict[str, Any]]:
        return [r for r in self.log if r["confirmed_student_id"] == student_id]

    def failed_rows(self) -> list[dict[str, Any]]:
        return [r for r in self.log if str(r["message"]).startswith("Could not process")]


@pytest.fixture
def scene(fake_db: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[Scene]:
    world = Scene(students={"AATKHA200618": "Aatif Khan", "AARSAY150821": "Aarav Sayyed",
                            "AADMIS310714": "Aadya Mistry"})
    fake_db.handler = world.handle
    monkeypatch.setattr(drive_inbox, "list_new_pages", world.list_new_pages)
    monkeypatch.setattr(drive_inbox, "download", world.download)
    monkeypatch.setattr(drive_inbox, "mark_done", world.mark_done)
    monkeypatch.setattr(inbox_watcher, "extract_student_ids_from_upload_batch", world.read)
    monkeypatch.setattr(inbox_watcher, "confirm_student_id", world.confirm)
    monkeypatch.setattr(inbox_watcher, "say", world.said.append)
    monkeypatch.setattr(inbox_watcher, "warm_models_in_background", lambda: None)
    # Per-process memory of the watcher, fresh for every test.
    monkeypatch.setattr(inbox_watcher, "_drained", set(), raising=False)
    monkeypatch.setattr(inbox_watcher, "_attempts", Counter(), raising=False)
    yield world


def passes(count: int) -> None:
    for _ in range(count):
        inbox_watcher.run_once()


# ------------------------------------------------- #5 / #8: logged once only
def test_a_file_drive_will_not_rename_is_logged_once_then_left_alone(scene: Scene) -> None:
    """Shared as Viewer: every rename is refused. The pages used to be read,
    looked up and logged again on every pass, for as long as the batch was open."""
    now = now_ist()
    batch_id = scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "scan.pdf", ["AATKHA200618", "AARSAY150821"],
                   created=now - timedelta(minutes=5))
    scene.rename_failures = -1

    passes(8)

    assert len(scene.rows_for("AATKHA200618")) == 1
    assert len(scene.rows_for("AARSAY150821")) == 1
    assert len(scene.log) == 2  # and no "Could not process this file" rows
    assert scene.downloads == ["f1"]  # read (and sent to Vision) once
    assert scene.lookups == [("AATKHA200618", "ABMPS"), ("AARSAY150821", "ABMPS")]
    assert scene.batch(batch_id)["pages_done"] == 2
    assert scene.files["f1"]["state"] == "parked"
    assert scene.renames == []


def test_a_rename_that_fails_once_is_retried_without_logging_again(scene: Scene) -> None:
    now = now_ist()
    batch_id = scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "scan.pdf", ["AATKHA200618", "AARSAY150821"],
                   created=now - timedelta(minutes=5))
    scene.rename_failures = 1

    passes(3)

    assert scene.renames == [("f1", "[done 2 pages] scan.pdf")]
    assert len(scene.log) == 2
    assert scene.downloads == ["f1"]
    assert scene.batch(batch_id)["pages_done"] == 2
    assert scene.files["f1"]["state"] == "done"


def test_a_page_that_fails_part_way_through_a_pdf_does_not_relog_the_pages_before_it(
    scene: Scene,
) -> None:
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    # Page 2 has no Student ID, and the insert for it hits a dropped connection.
    scene.add_file("f1", "stack.pdf", ["AATKHA200618", "", "AADMIS310714"],
                   created=now - timedelta(minutes=5))
    scene.fail_log_inserts = {2}

    passes(3)

    assert len(scene.rows_for("AATKHA200618")) == 1
    assert len(scene.rows_for("")) == 1
    assert len(scene.rows_for("AADMIS310714")) == 1
    assert scene.lookups.count(("AATKHA200618", "ABMPS")) == 1
    assert scene.renames == [("f1", "[done 3 pages] stack.pdf")]


def test_a_file_that_keeps_failing_is_parked_once_not_over_and_over(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-memory attempt count was reset after each park, so a file that
    could not be renamed "[failed]" either was parked again every few passes."""
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "scan.jpg", ["AATKHA200618"], created=now - timedelta(minutes=5))
    scene.rename_failures = -1  # so even "[failed] scan.jpg" cannot be written

    def broken(file_id: str) -> bytes:
        scene.downloads.append(file_id)
        raise drive_inbox.DriveInboxError("Drive GET files/f1 -> 500: backend error")

    monkeypatch.setattr(drive_inbox, "download", broken)
    passes(9)

    assert len(scene.failed_rows()) == 1
    assert scene.downloads == ["f1"] * inbox_watcher.MAX_ATTEMPTS
    assert scene.files["f1"]["state"] == "parked"


def _broken_download(scene: Scene) -> Callable[[str], bytes]:
    def broken(file_id: str) -> bytes:
        scene.downloads.append(file_id)
        raise drive_inbox.DriveInboxError(f"Drive GET files/{file_id} -> 500: backend error")

    return broken


def test_a_failed_row_that_cannot_be_written_is_written_later_not_lost(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file was renamed "[failed]" and settled before its History row was
    written, so an insert that failed left a renamed file and no row."""
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "scan.jpg", ["AATKHA200618"], created=now - timedelta(minutes=5))
    monkeypatch.setattr(drive_inbox, "download", _broken_download(scene))
    scene.fail_log_inserts = {1}  # the "[failed]" row hits a dropped connection

    passes(inbox_watcher.MAX_ATTEMPTS)

    assert scene.log == [] and scene.renames == []  # nothing renamed without its row

    passes(4)

    assert len(scene.failed_rows()) == 1
    assert scene.renames == [("f1", "[failed] scan.jpg")]
    assert scene.files["f1"]["state"] == "done"


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("HTTPSConnectionPool(host='www.googleapis.com', port=443): "
                                 "Max retries exceeded with url: /drive/v3/files/f1"),
        requests.Timeout("HTTPSConnectionPool(host='www.googleapis.com'): Read timed out."),
    ],
)
def test_a_network_error_renaming_a_failed_file_still_parks_it_once(
    scene: Scene, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """A dropped connection got past `except DriveInboxError`: the file was
    never settled, and was downloaded and OCR'd again on every pass."""
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "scan.jpg", ["AATKHA200618"], created=now - timedelta(minutes=5))
    monkeypatch.setattr(drive_inbox, "download", _broken_download(scene))

    def unreachable(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(drive_inbox, "mark_done", REAL_MARK_DONE)
    monkeypatch.setattr(drive_inbox, "_token", lambda: "token")
    monkeypatch.setattr(drive_inbox.requests, "request", unreachable)

    passes(inbox_watcher.MAX_ATTEMPTS + 4)

    assert scene.downloads == ["f1"] * inbox_watcher.MAX_ATTEMPTS
    assert len(scene.failed_rows()) == 1
    assert scene.files["f1"]["state"] == "parked"


@pytest.mark.parametrize("where", ["request", "token"])
def test_drive_being_unreachable_is_a_drive_inbox_error_with_no_detail_in_it(
    monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    secret_url = "https://www.googleapis.com/drive/v3/files/f1?supportsAllDrives=true"

    def unreachable(*_args: Any, **_kwargs: Any) -> Any:
        raise requests.ConnectionError(f"Max retries exceeded with url: {secret_url}")

    def refused() -> str:
        raise RefreshError("invalid_grant: account not found", {"error": "invalid_grant"})

    monkeypatch.setattr(drive_inbox, "_token", refused if where == "token" else lambda: "token")
    monkeypatch.setattr(drive_inbox.requests, "request", unreachable)

    with pytest.raises(drive_inbox.DriveInboxError) as raised:
        drive_inbox.mark_done("f1", "[failed] scan.jpg")

    # Shown in History and the watcher's output; the detail stays in the log.
    assert "Could not reach Drive" in str(raised.value)
    assert "googleapis" not in str(raised.value) and "invalid_grant" not in str(raised.value)


# ------------------------------------------------ #9: a lookup outage waits
def test_a_lookup_outage_leaves_the_file_untouched_until_the_database_is_back(
    scene: Scene,
) -> None:
    now = now_ist()
    batch_id = scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=6))
    scene.add_file("f2", "page-002.jpg", ["AARSAY150821"], created=now - timedelta(minutes=5))
    scene.lookup_down = True

    passes(5)

    assert scene.renames == []  # never "[not found]", never "[failed]"
    assert scene.log == []
    assert scene.files["f1"]["attempts"] == 0
    assert scene.files["f1"]["state"] == "reading"
    # The pass stops at the outage instead of OCR'ing every other page to
    # fail the same way.
    assert "f2" not in scene.downloads

    scene.lookup_down = False
    passes(1)

    assert scene.name_of("f1") == "[saved] Aatif Khan_AATKHA200618.jpg"
    assert scene.name_of("f2") == "[saved] Aarav Sayyed_AARSAY150821.jpg"
    assert [r["found_in_db"] for r in scene.log] == [1, 1]
    assert scene.batch(batch_id)["pages_done"] == 2


def test_a_failed_history_insert_leaves_the_file_for_a_retry(scene: Scene) -> None:
    """log_confirm_result swallows a failed insert; the watcher then renamed
    the page "[saved] ..." with no row in History for it at all."""
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=5))
    scene.fail_log_inserts = {1}

    passes(1)

    assert scene.renames == []
    assert scene.log == []

    passes(1)

    assert len(scene.rows_for("AATKHA200618")) == 1
    assert scene.name_of("f1") == "[saved] Aatif Khan_AATKHA200618.jpg"


# ------------------------------------- #10: a page belongs to its own batch
def test_a_page_left_when_a_batch_closed_is_read_for_that_batch_not_the_next(
    scene: Scene,
) -> None:
    now = now_ist()
    first = scene.add_batch("ABMPS", opened=now - timedelta(hours=3),
                            closed=now - timedelta(hours=1))
    second = scene.add_batch("LAPMEMS", opened=now - timedelta(minutes=55))
    scene.add_file("left", "abmps-page-031.jpg", ["AATKHA200618"],
                   created=now - timedelta(hours=2))
    # Arrived during the second batch. Compared without converting Drive's
    # UTC to IST it would look five and a half hours older - inside the first.
    scene.add_file("new", "page-001.jpg", ["AARSAY150821"],
                   created=now - timedelta(minutes=30))

    passes(1)

    assert scene.lookups == [("AATKHA200618", "ABMPS"), ("AARSAY150821", "LAPMEMS")]
    left, new = scene.rows_for("AATKHA200618"), scene.rows_for("AARSAY150821")
    assert [(r["source"], r["scan_school"]) for r in left] == [(f"drive:{first}", "ABMPS")]
    assert [(r["source"], r["scan_school"]) for r in new] == [(f"drive:{second}", "LAPMEMS")]
    assert scene.batch(first)["pages_done"] == 1
    assert scene.batch(second)["pages_done"] == 1


def test_pages_waiting_at_stop_are_still_read_with_no_batch_open(scene: Scene) -> None:
    now = now_ist()
    batch_id = scene.add_batch("ABMPS", opened=now - timedelta(hours=1),
                               closed=now - timedelta(minutes=1))
    scene.add_file("f1", "page-031.jpg", ["AATKHA200618"], created=now - timedelta(minutes=10))

    passes(1)

    assert scene.lookups == [("AATKHA200618", "ABMPS")]
    assert scene.name_of("f1") == "[saved] Aatif Khan_AATKHA200618.jpg"
    assert scene.batch(batch_id)["pages_done"] == 1


def test_a_page_that_reaches_drive_after_stop_is_not_read_for_any_school(
    scene: Scene,
) -> None:
    """It may be the last school's tail still syncing or the next school's
    first page; read under whichever batch opened next, a school-A form was
    looked up among school B's children."""
    now = now_ist()
    first = scene.add_batch("ABMPS", opened=now - timedelta(hours=1),
                            closed=now - timedelta(minutes=10))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=5))
    second = scene.add_batch("LAPMEMS", opened=now - timedelta(minutes=2))

    passes(2)

    assert scene.lookups == [] and scene.downloads == []  # not read at all
    assert len(scene.log) == 1
    row = scene.log[0]
    assert row["confirmed_student_id"] == "" and not row["found_in_db"]
    assert row["scan_school"] is None and row["scan_location"] is None
    assert row["source"] == f"drive:{second}"  # listed in the panel that found it
    assert "not read for any school" in row["message"]
    assert scene.name_of("f1") == "[unread] page-001.jpg"
    assert scene.batch(first)["pages_done"] == 0
    assert scene.batch(second)["pages_done"] == 0


def test_a_page_that_reaches_drive_after_stop_is_set_aside_with_nothing_open(
    scene: Scene,
) -> None:
    now = now_ist()
    first = scene.add_batch("ABMPS", opened=now - timedelta(hours=1),
                            closed=now - timedelta(minutes=3))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=1))

    passes(1)

    assert scene.lookups == []
    assert [(r["source"], r["scan_school"]) for r in scene.log] == [(f"drive:{first}", None)]
    assert scene.name_of("f1") == "[unread] page-001.jpg"

    scene.add_batch("LAPMEMS", opened=now_ist())
    passes(1)

    assert len(scene.log) == 1 and scene.lookups == []  # not read again later


def test_the_last_sheet_still_uploading_when_the_next_school_starts_is_read_for_its_own_school(
    scene: Scene,
) -> None:
    """Saved at T, Stop at T+5s, the next school started at T+20s, the upload
    landed at T+30s. Drive dates the page by the upload, inside the second
    batch, so it was looked up among LAPMEMS children only, logged not found
    and renamed "[not found] ..." - never to be read again."""
    now = now_ist()
    first = scene.add_batch("ABMPS", opened=now - timedelta(hours=1),
                            closed=now - timedelta(seconds=55))
    second = scene.add_batch("LAPMEMS", opened=now - timedelta(seconds=40))
    scene.schools = {"AATKHA200618": "ABMPS", "AARSAY150821": "LAPMEMS"}
    scene.add_file("tail", "page-031.jpg", ["AATKHA200618"], created=now - timedelta(seconds=30))
    scene.add_file("next", "page-001.jpg", ["AARSAY150821"], created=now - timedelta(seconds=20))

    passes(2)

    assert scene.lookups == [
        ("AATKHA200618", "LAPMEMS"),  # where it would go by its upload time
        ("AATKHA200618", "ABMPS"),    # the school stopped just before it
        ("AARSAY150821", "LAPMEMS"),  # a page of the new school: one lookup
    ]
    tail, new = scene.rows_for("AATKHA200618"), scene.rows_for("AARSAY150821")
    assert [(r["source"], r["scan_school"], r["found_in_db"]) for r in tail] == [
        (f"drive:{first}", "ABMPS", True)
    ]
    assert [(r["source"], r["scan_school"], r["found_in_db"]) for r in new] == [
        (f"drive:{second}", "LAPMEMS", True)
    ]
    assert scene.name_of("tail") == "[saved] Aatif Khan_AATKHA200618.jpg"
    assert scene.name_of("next") == "[saved] Aarav Sayyed_AARSAY150821.jpg"
    assert scene.batch(first)["pages_done"] == 1
    assert scene.batch(second)["pages_done"] == 1


def test_a_pdf_landing_just_after_a_school_switch_counts_each_page_where_it_was_found(
    scene: Scene,
) -> None:
    now = now_ist()
    first = scene.add_batch("ABMPS", opened=now - timedelta(hours=1),
                            closed=now - timedelta(minutes=1))
    second = scene.add_batch("LAPMEMS", opened=now - timedelta(minutes=1))
    scene.schools = {"AATKHA200618": "ABMPS", "AARSAY150821": "LAPMEMS"}
    scene.add_file("f1", "stack.pdf", ["AATKHA200618", "AARSAY150821"],
                   created=now - timedelta(seconds=30))

    passes(1)

    assert [(r["confirmed_student_id"], r["source"]) for r in scene.log] == [
        ("AATKHA200618", f"drive:{first}"), ("AARSAY150821", f"drive:{second}"),
    ]
    assert scene.batch(first)["pages_done"] == 1
    assert scene.batch(second)["pages_done"] == 1
    assert scene.name_of("f1") == "[done 2 pages] stack.pdf"


def test_a_sheet_found_in_neither_school_is_not_found_for_the_open_batch(scene: Scene) -> None:
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(hours=1), closed=now - timedelta(minutes=1))
    second = scene.add_batch("LAPMEMS", opened=now - timedelta(minutes=1))
    scene.add_file("f1", "page-001.jpg", ["ZZZZZZ999999"], created=now - timedelta(seconds=30))

    passes(1)

    assert scene.lookups == [("ZZZZZZ999999", "LAPMEMS"), ("ZZZZZZ999999", "ABMPS")]
    assert [(r["source"], r["scan_school"], r["found_in_db"]) for r in scene.log] == [
        (f"drive:{second}", "LAPMEMS", False)
    ]
    assert scene.name_of("f1") == "[not found] ZZZZZZ999999.jpg"


def test_a_page_well_after_the_last_stop_is_only_looked_up_in_the_open_school(
    scene: Scene,
) -> None:
    """Only a sheet that could still have been uploading gets a second look:
    a page of the new school whose student is missing from the database must
    not be matched against the last school's children an hour later."""
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(hours=3), closed=now - timedelta(hours=1))
    second = scene.add_batch("LAPMEMS", opened=now - timedelta(hours=1))
    scene.schools = {"AATKHA200618": "ABMPS"}
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=5))

    passes(1)

    assert scene.lookups == [("AATKHA200618", "LAPMEMS")]
    assert [(r["source"], r["found_in_db"]) for r in scene.log] == [(f"drive:{second}", False)]


def test_a_restart_on_the_same_school_looks_a_sheet_up_once(scene: Scene) -> None:
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(hours=1), closed=now - timedelta(minutes=1))
    second = scene.add_batch("ABMPS", opened=now - timedelta(minutes=1))
    scene.add_file("f1", "page-001.jpg", ["ZZZZZZ999999"], created=now - timedelta(seconds=30))

    passes(1)

    assert scene.lookups == [("ZZZZZZ999999", "ABMPS")]
    assert [r["source"] for r in scene.log] == [f"drive:{second}"]


def test_pages_dropped_in_before_the_first_start_are_read_for_it(scene: Scene) -> None:
    now = now_ist()
    batch_id = scene.add_batch("ABMPS", opened=now - timedelta(minutes=1))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=20))

    passes(1)

    assert scene.lookups == [("AATKHA200618", "ABMPS")]
    assert scene.batch(batch_id)["pages_done"] == 1


def test_a_closed_batch_with_nothing_left_lets_its_folder_go(scene: Scene) -> None:
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(hours=1), closed=now - timedelta(minutes=10))

    passes(3)

    assert scene.listings == 1


@pytest.mark.parametrize(
    ("created", "expected"),
    [
        ("2026-10-27T05:12:33.123Z", datetime(2026, 10, 27, 10, 42, 33, 123000)),
        ("2026-10-27T20:00:00Z", datetime(2026, 10, 28, 1, 30)),
        ("2026-10-27T05:12:33.123+00:00", datetime(2026, 10, 27, 10, 42, 33, 123000)),
        ("", None),
        (None, None),
        ("yesterday", None),
    ],
)
def test_drive_created_time_is_read_as_utc_and_turned_into_ist(
    created: str | None, expected: datetime | None
) -> None:
    assert inbox_watcher._drive_time(created) == expected


def test_stop_with_pages_waiting_says_what_will_happen_to_them(
    scene: Scene, client: FlaskClient
) -> None:
    now = now_ist()
    batch_id = scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=1))

    response = client.post("/api/inbox/stop", json={})

    assert response.status_code == 409
    body = response.get_json()
    assert body["reason"] == "pending" and body["pending"] == 1
    # The confirmation promised the waiting pages would be left unread and
    # unattributed; they are read for this batch's school, and scanning them
    # again would only log them twice. What becomes of a sheet that lands
    # after Stop depends on whether another school has been started by then.
    message = body["message"]
    assert "still read for this school" in message
    assert "no need to scan them again" in message
    assert "after you stop" in message and "unread if no school has been started" in message
    assert "not attributed to a school" not in message
    assert scene.batch(batch_id)["closed_at"] is None


def test_stop_does_not_wait_on_pages_already_recorded(scene: Scene, client: FlaskClient) -> None:
    """A file whose pages are logged but which Drive will not let us rename
    keeps its own name. Counted as waiting, it held Stop at 409 for ever."""
    now = now_ist()
    batch_id = scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=5))
    scene.rename_failures = -1
    passes(inbox_watcher.MAX_ATTEMPTS)
    assert scene.name_of("f1") == "page-001.jpg"

    response = client.post("/api/inbox/stop", json={})

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["closed"] is True
    assert scene.batch(batch_id)["closed_at"] is not None


# --------------------------------------------- #19: a heartbeat of its own
class _StopWatching(Exception):
    """Ends watch()'s endless loop after its first pass."""


def test_the_heartbeat_keeps_beating_through_a_long_pass(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pass reading hard pages outlasted the page's 45 s limit, and staff
    were told the watcher was not running while it read their pages."""
    beats: list[float] = []
    during: list[int] = []

    def slow_pass(progress: Callable[[], None] | None = None) -> int:
        before = len(beats)
        deadline = time.monotonic() + 3
        while len(beats) < before + 3 and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        during.append(len(beats) - before)
        return 0

    def end_loop(_seconds: float) -> None:
        raise _StopWatching

    monkeypatch.setattr(drive_inbox, "configured", lambda: (True, ""))
    monkeypatch.setattr(batch_store, "heartbeat", lambda _host="": beats.append(time.monotonic()))
    monkeypatch.setattr(inbox_watcher, "run_once", slow_pass)
    monkeypatch.setattr(inbox_watcher, "HEARTBEAT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(
        inbox_watcher, "time", SimpleNamespace(sleep=end_loop, monotonic=time.monotonic)
    )

    with pytest.raises(_StopWatching):
        inbox_watcher.watch()

    assert during and during[0] >= 3


def test_the_heartbeat_does_not_vouch_for_a_stuck_or_locked_out_watcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beats: list[str] = []
    monkeypatch.setattr(batch_store, "heartbeat", lambda host="": beats.append(host))
    beat = inbox_watcher.Heartbeat("host-a", interval=1.0, stuck_after=0.05)

    assert beat.beat_once() is False  # another watcher holds the lock

    beat.active = True
    beat.progress()
    assert beat.beat_once() is True

    threading.Event().wait(0.1)  # the main loop has not moved since
    assert beat.beat_once() is False

    beat.progress()
    assert beat.beat_once() is True
    assert beats == ["host-a", "host-a"]


def _wait_for(condition: Callable[[], bool], seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    return condition()


def test_a_failed_heartbeat_does_not_end_the_heartbeat_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the thread itself: a database blip on one beat must not end it,
    or the page says the watcher is down for as long as it keeps running."""
    calls: list[str] = []

    def flaky(host: str = "") -> None:
        calls.append(host)
        if len(calls) <= 2:
            raise pymysql.err.OperationalError(2003, "Can't connect")

    monkeypatch.setattr(batch_store, "heartbeat", flaky)
    beat = inbox_watcher.Heartbeat("host-a", interval=0.01)
    beat.active = True
    beat.progress()
    beat.start()
    try:
        assert _wait_for(lambda: len(calls) >= 5)
        assert beat._thread is not None and beat._thread.is_alive()
    finally:
        beat.stop()
    assert beat._thread is not None and not beat._thread.is_alive()


# --------------------------------------------------- #20: one watcher only
def test_a_second_watcher_run_once_does_not_touch_the_inboxes(scene: Scene) -> None:
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=5))
    scene.lock_free = False  # the service is running and holds the lock

    with pytest.raises(SystemExit) as stopped:
        inbox_watcher.main(["--once"])

    assert "another watcher" in str(stopped.value.code)
    assert scene.listings == 0 and scene.downloads == [] and scene.log == []
    assert scene.renames == []


def test_run_once_takes_and_lets_go_of_the_lock(scene: Scene) -> None:
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=5))

    inbox_watcher.main(["--once"])

    assert scene.lock_calls == ["GET_LOCK", "RELEASE_LOCK"]
    assert scene.name_of("f1") == "[saved] Aatif Khan_AATKHA200618.jpg"


def test_the_watch_loop_waits_while_another_watcher_holds_the_lock(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    passes_run: list[int] = []
    beats: list[str] = []

    def one_pass(progress: Callable[[], None] | None = None) -> int:
        passes_run.append(1)
        return 0

    monkeypatch.setattr(inbox_watcher, "run_once", one_pass)
    monkeypatch.setattr(batch_store, "heartbeat", lambda host="": beats.append(host))
    lock = batch_store.WatcherLock()
    beat = inbox_watcher.Heartbeat("host-b", interval=1.0)

    scene.lock_free = False
    assert inbox_watcher.watch_pass(lock, beat) == "waiting"
    assert passes_run == [] and beats == []  # nor does it say it is listening

    scene.lock_free = True
    assert inbox_watcher.watch_pass(lock, beat) == "ran"
    assert inbox_watcher.watch_pass(lock, beat) == "ran"
    assert passes_run == [1, 1]
    # Taken once and then kept, on the connection that took it.
    assert scene.lock_calls == ["GET_LOCK", "GET_LOCK", "IS_USED_LOCK"]
    lock.release()
    assert scene.lock_calls[-1] == "RELEASE_LOCK"


def test_the_watch_loop_has_mysql_free_its_lock_soon_after_its_host_dies(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET_LOCK lives as long as its connection. A host that died without
    closing it (power cut, network gone) left MySQL holding the lock until
    wait_timeout - eight hours by default - while every replacement watcher
    only waited and no inbox was read."""
    monkeypatch.setattr(drive_inbox, "configured", lambda: (True, ""))
    monkeypatch.setattr(inbox_watcher, "run_once", lambda progress=None: 0)

    def end_loop(_seconds: float) -> None:
        raise _StopWatching

    monkeypatch.setattr(
        inbox_watcher, "time", SimpleNamespace(sleep=end_loop, monotonic=time.monotonic)
    )

    with pytest.raises(_StopWatching):
        inbox_watcher.watch()

    assert scene.session_settings == [(inbox_watcher.LOCK_IDLE_SECONDS,)]
    # Short, but far longer than the gap between two keep-alives.
    assert inbox_watcher.LOCK_IDLE_SECONDS >= 6 * inbox_watcher.HEARTBEAT_SECONDS
    assert scene.lock_calls[0] == "GET_LOCK" and scene.lock_calls[-1] == "RELEASE_LOCK"


def test_a_lock_taken_with_an_idle_limit_sets_it_on_its_own_connection(
    scene: Scene, fake_db: Any
) -> None:
    lock = batch_store.WatcherLock(idle_timeout=120)

    assert lock.acquire() and lock.held

    sql = [s for s, _ in fake_db.statements]
    assert sql.index("SET SESSION wait_timeout = %s") < sql.index("SELECT GET_LOCK(%s, 0) AS got")
    assert scene.session_settings == [(120,)]


def test_the_heartbeat_thread_keeps_the_lock_connection_alive(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With MySQL dropping the lock's connection after a short silence, a long
    pass would lose the lock unless something used the connection meanwhile."""
    monkeypatch.setattr(batch_store, "heartbeat", lambda host="": None)
    lock = batch_store.WatcherLock(idle_timeout=120)
    assert lock.acquire()
    beat = inbox_watcher.Heartbeat("host-a", interval=0.01, lock=lock)
    beat.active = True
    beat.progress()
    beat.start()
    try:
        assert _wait_for(lambda: scene.lock_calls.count("IS_USED_LOCK") >= 3)
    finally:
        beat.stop()
    assert lock.held


def test_a_long_file_keeps_the_lock_though_the_heartbeat_stops_vouching(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One file of a long PDF can outlast STUCK_AFTER_SECONDS. The page is
    then told nothing is moving, but the lock must not idle out under it."""
    beats: list[str] = []
    monkeypatch.setattr(batch_store, "heartbeat", lambda host="": beats.append(host))
    lock = batch_store.WatcherLock(idle_timeout=120)
    assert lock.acquire()
    beat = inbox_watcher.Heartbeat("host-a", interval=0.01, stuck_after=0.0, lock=lock)
    beat.active = True
    beat.start()
    try:
        assert _wait_for(lambda: scene.lock_calls.count("IS_USED_LOCK") >= 3)
    finally:
        beat.stop()
    assert beats == [] and lock.held


def test_the_heartbeat_does_not_vouch_for_a_watcher_that_lost_its_lock(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    beats: list[str] = []
    monkeypatch.setattr(batch_store, "heartbeat", lambda host="": beats.append(host))
    lock = batch_store.WatcherLock(idle_timeout=120)
    assert lock.acquire()
    beat = inbox_watcher.Heartbeat("host-a", interval=1.0, lock=lock)
    beat.active = True
    beat.progress()

    assert beat.beat_once() is True
    scene.lock_mine = False
    assert beat.beat_once() is False and not lock.held
    assert beats == ["host-a"]


def test_a_watcher_that_loses_its_lock_mid_pass_stops_before_the_next_file(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the lock is gone, a waiting watcher can take it and start on the
    same folder; carrying on would have two of them logging the same pages."""
    now = now_ist()
    scene.add_batch("ABMPS", opened=now - timedelta(minutes=30))
    scene.add_file("f1", "page-001.jpg", ["AATKHA200618"], created=now - timedelta(minutes=6))
    scene.add_file("f2", "page-002.jpg", ["AARSAY150821"], created=now - timedelta(minutes=5))
    monkeypatch.setattr(batch_store, "heartbeat", lambda host="": None)
    lock = batch_store.WatcherLock(idle_timeout=120)
    beat = inbox_watcher.Heartbeat("host-a", interval=1.0, lock=lock)

    def download_then_lose_the_lock(file_id: str) -> bytes:
        data = scene.download(file_id)
        scene.lock_mine = False  # MySQL dropped the lock's connection
        beat.beat_once()  # what the heartbeat thread does every few seconds
        return data

    monkeypatch.setattr(drive_inbox, "download", download_then_lose_the_lock)

    assert inbox_watcher.watch_pass(lock, beat) == "lost"

    assert scene.downloads == ["f1"]  # the file under way is finished...
    assert scene.name_of("f1") == "[saved] Aatif Khan_AATKHA200618.jpg"
    assert "f2" not in scene.files  # ...and the next is left for the lock's holder
    assert beat.active is False
