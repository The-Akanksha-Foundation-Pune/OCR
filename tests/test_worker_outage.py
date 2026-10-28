"""The OCR worker during a student database outage: it pauses, and says so.

Covered here:

  - The first outage on a page pauses the worker: the page goes back in the
    queue with its OCR reading (ocr.json) kept and its tries unchanged, and
    no further page is taken - not even to be read - while it is paused.
  - A database error straight from the lookup is an outage too, never a try.
  - While paused the database is probed after 30 s, 60, 120, 240 and then
    every 300 s; a probe that answers resumes the worker, and the waiting
    page then completes with only its lookup done again.
  - The probe itself: one SELECT 1 with a five-second connect timeout, while
    every other connection keeps its twenty seconds.
  - _worker_state.json: written whole, kept fresh through a long pause, read
    back by GET /api/batches; one left alone too long is "not_running".
  - The scan page's Scanned documents line, from GET /api/batches.

The database is tests/conftest.py's FakeDB, the recognisers and the probe
are stand-ins, and the clock is moved by hand. The queue lives in the test's
own folder - never the real data/queue.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Self

import cv2
import numpy as np
import pymysql
import pytest
from flask.testing import FlaskClient

from server.config import QUEUE_DIR as REAL_QUEUE_DIR
from server.services import ocr_worker, scan_queue, student_lookup
from server.services.batch_code import BatchCode, CodeReading
from server.services.extract_student_id import ExtractResult
from tests.test_app import _run_page, needs_node

OWNER = "staff@akanksha.org"  # who the `client` fixture is signed in as
SCHOOL = "ABMPS"
AATIF = {
    "student_id": "AATKHA200618",
    "student_name": "Aatif Khan",
    "school_name": "ABMPS",
    "grade_name": "Grade 5",
    "division_name": "A",
    "academic_year": "2026-2027",
    "gender": "M",
    "status": "Active - Enrolled",
    "scanned_at": None,
}
START = datetime(2026, 10, 28, 13, 59, 45)  # India time, as the worker's clock reads
PAUSED_LINE = (
    "OCR paused - the student database can't be reached. Your scans are kept and "
    "will be read automatically when it's back (next try 14:00)."
)
NOT_RUNNING_LINE = "OCR worker is not running - scans are kept and will be read when it starts."


def _down() -> pymysql.err.OperationalError:
    return pymysql.err.OperationalError(2003, "Can't connect to MySQL server on 'db' (timed out)")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def queue_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The queue, its state file and the log in the test's own folder."""
    import server.services.debug_log as debug_module

    folder = tmp_path / "queue"
    monkeypatch.setattr(scan_queue, "QUEUE_DIR", folder)
    monkeypatch.setattr(debug_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(debug_module, "LOG_FILE", tmp_path / "logs" / "ocr_debug.log")
    monkeypatch.setattr(student_lookup, "_id_cache", {})
    monkeypatch.setattr(student_lookup, "_id_cache_at", {})
    # The stale-claim sweep runs at most once a minute per process.
    monkeypatch.setattr(ocr_worker, "_last_reclaim", time.monotonic())
    assert scan_queue.queue_root() == folder != Path(REAL_QUEUE_DIR)
    return folder


@dataclass
class Clock:
    """The worker's clock, moved by hand."""

    now: datetime = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(ocr_worker, "clock", fake)
    return fake


@dataclass
class Database:
    """The student database, down until a test brings it back.

    Attributes:
        down: Every statement and every probe fails while this is set.
        lookups: Lookups the worker asked for, answered or not.
        probes: When the worker probed, by its own clock.
    """

    down: bool = True
    lookups: int = 0
    probes: list[datetime] = field(default_factory=list)


@pytest.fixture
def database(fake_db: Any, monkeypatch: pytest.MonkeyPatch, clock: Clock) -> Database:
    known = Database()

    def handler(sql: str, params: Any) -> tuple[list[dict[str, Any]] | None, int | None]:
        if known.down:
            raise _down()
        if "WHERE UPPER(TRIM(student_id)) = UPPER(%s)" in sql:
            row = AATIF if str(params[0]).upper() == AATIF["student_id"] else None
            return ([dict(row)] if row else []), None
        return [], 0

    fake_db.handler = handler
    real_lookup = ocr_worker.lookup_student

    def counted_lookup(student_id: str, **kwargs: Any) -> dict[str, Any] | None:
        known.lookups += 1
        return real_lookup(student_id, **kwargs)

    def probe() -> None:
        known.probes.append(clock.now)
        if known.down:
            raise _down()

    monkeypatch.setattr(ocr_worker, "lookup_student", counted_lookup)
    monkeypatch.setattr(ocr_worker, "probe", probe)
    return known


@dataclass
class Readers:
    """Stand-ins for the recognisers: every page reads B1-1 and Aatif's ID."""

    code_calls: int = 0
    id_calls: int = 0


@pytest.fixture
def readers(monkeypatch: pytest.MonkeyPatch) -> Readers:
    fake = Readers()

    def read_code(_page: np.ndarray, *, allow_vision: bool = True) -> CodeReading:
        fake.code_calls += 1
        return CodeReading(code=BatchCode(1, 1), confidence=0.9, raw="B1-1", region="top-right")

    def read_student_id(_image_bgr: np.ndarray, source: str) -> ExtractResult:
        fake.id_calls += 1
        return ExtractResult(
            student_id="AATKHA200618", confidence=0.9, form_language="english", source=source
        )

    monkeypatch.setattr(ocr_worker, "read_code", read_code)
    monkeypatch.setattr(ocr_worker, "read_student_id", read_student_id)
    monkeypatch.setattr(ocr_worker, "identify", lambda _image: None)
    return fake


# ------------------------------------------------------------------ helpers
def _page_jpeg() -> bytes:
    """A white page with enough ink not to count as blank."""
    page = np.full((874, 620), 255, dtype=np.uint8)
    cv2.rectangle(page, (50, 50), (570, 150), 0, thickness=-1)
    ok, encoded = cv2.imencode(".jpg", page)
    assert ok
    return encoded.tobytes()


def _queue_pages(count: int) -> list[str]:
    """Queue `count` one-page scans for OWNER, oldest first."""
    ids: list[str] = []
    for number in range(1, count + 1):
        [meta] = scan_queue.intake(
            _page_jpeg(), "scan.jpg", source_name=f"Scan {number}.jpg", owner_email=OWNER, school=SCHOOL
        )
        ids.append(meta.page_id)
        time.sleep(0.002)  # the next page's id sorts after this one's
    return ids


def _write_state(state: str, updated: datetime, **extra: Any) -> None:
    """A state file as some worker process left it."""
    record = ocr_worker.WorkerState(state=state, since=updated.isoformat(timespec="seconds"), **extra)
    record.updated = updated.isoformat(timespec="seconds")
    scan_queue.write_json_atomic(scan_queue.worker_state_path(), asdict(record))


def _stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


# ================================================================== the pause
def test_an_outage_on_the_first_page_pauses_the_worker_and_takes_no_more_pages(
    queue_dir: Path, clock: Clock, database: Database, readers: Readers
) -> None:
    first, second, third = _queue_pages(3)

    assert ocr_worker.run_pass(budget_seconds=60) == 0

    # The first page is back in the queue with its reading kept, and the
    # outage is not one of its tries.
    assert (queue_dir / "_pending" / first).exists()
    assert not (queue_dir / first / "result.json").exists()
    kept = json.loads((queue_dir / first / "ocr.json").read_text())
    assert kept["code"] == "B1-1" and kept["student_id"] == "AATKHA200618"
    meta = scan_queue.load_meta(first)
    assert meta is not None and meta.status == "queued" and meta.attempts == 0 and meta.error == ""
    # The worker says it is paused, why, and when it asks again.
    state = json.loads((queue_dir / "_worker_state.json").read_text())
    assert state["state"] == "paused"
    assert state["reason"] == ocr_worker.OUTAGE_REASON
    assert state["since"] == "2026-10-28T13:59:45"
    assert state["next_try"] == "2026-10-28T14:00:15"
    assert (state["host"], state["pid"]) == (socket.gethostname(), os.getpid())

    # Until the back-off is up, every look takes nothing at all.
    for _ in range(5):
        clock.advance(5)
        assert ocr_worker.run_pass(budget_seconds=60) == 0
        assert ocr_worker.process_next() == ocr_worker.PAUSED

    assert database.lookups == 1 and database.probes == []
    assert (readers.code_calls, readers.id_calls) == (1, 1)  # only the first page was read
    for page_id in (first, second, third):
        assert (queue_dir / "_pending" / page_id).exists()
        meta = scan_queue.load_meta(page_id)
        assert meta is not None and meta.status == "queued" and meta.attempts == 0
    for page_id in (second, third):
        assert not (queue_dir / page_id / "ocr.json").exists()
    assert list((queue_dir / "_reading").iterdir()) == []


def test_a_database_error_straight_from_the_lookup_is_an_outage_not_a_try(
    queue_dir: Path, clock: Clock, readers: Readers, monkeypatch: pytest.MonkeyPatch
) -> None:
    [page_id] = _queue_pages(1)

    def unwrapped(_student_id: str, **_kwargs: Any) -> dict[str, Any] | None:
        raise pymysql.err.OperationalError(2013, "Lost connection to MySQL server during query")

    monkeypatch.setattr(ocr_worker, "lookup_student", unwrapped)

    assert ocr_worker.process_next() == ocr_worker.UNAVAILABLE

    meta = scan_queue.load_meta(page_id)
    assert meta is not None and meta.status == "queued" and meta.attempts == 0 and meta.error == ""
    assert (queue_dir / page_id / "ocr.json").exists()
    state = ocr_worker.load_state()
    assert state is not None and state.state == ocr_worker.STATE_PAUSED
    assert ocr_worker.process_next() == ocr_worker.PAUSED


def test_while_paused_the_database_is_probed_after_30_60_120_240_then_every_300_seconds(
    queue_dir: Path, clock: Clock, database: Database, readers: Readers
) -> None:
    [page_id] = _queue_pages(1)
    assert ocr_worker.run_pass(budget_seconds=60) == 0
    paused_at = clock.now

    for _ in range(0, 1100, 5):
        clock.advance(5)
        assert ocr_worker.run_pass(budget_seconds=60) == 0

    offsets = [(moment - paused_at).total_seconds() for moment in database.probes]
    assert offsets == [30, 90, 210, 450, 750, 1050]
    waits = [later - earlier for earlier, later in zip([0.0, *offsets[:-1]], offsets, strict=True)]
    assert waits == [30, 60, 120, 240, 300, 300]
    state = ocr_worker.load_state()
    assert state is not None and state.state == ocr_worker.STATE_PAUSED
    assert state.backoff_seconds == 300
    assert state.next_try == _stamp(paused_at + timedelta(seconds=1350))
    assert state.since == _stamp(paused_at)  # one pause, however many probes
    # Not one page was taken in all that time, and the outage is still no try.
    assert database.lookups == 1 and readers.id_calls == 1
    meta = scan_queue.load_meta(page_id)
    assert meta is not None and meta.attempts == 0


def test_a_probe_that_answers_resumes_and_the_page_completes_with_only_its_lookup_redone(
    queue_dir: Path,
    clock: Clock,
    database: Database,
    readers: Readers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    [page_id] = _queue_pages(1)
    assert ocr_worker.run_pass(budget_seconds=60) == 0
    assert (readers.code_calls, readers.id_calls) == (1, 1)

    # The reading is in ocr.json now: OCR must not run on this page again.
    def no_more_ocr(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("OCR ran again")

    for name in ("read_code", "read_student_id", "identify"):
        monkeypatch.setattr(ocr_worker, name, no_more_ocr)

    database.down = False
    clock.advance(29)
    assert ocr_worker.run_pass(budget_seconds=60) == 0  # back, but not asked yet
    assert database.probes == []
    clock.advance(1)
    assert ocr_worker.run_pass(budget_seconds=60) == 1

    assert len(database.probes) == 1 and database.lookups == 2
    result = json.loads((queue_dir / page_id / "result.json").read_text())
    assert result["code"] == "B1-1" and result["student_id"] == "AATKHA200618"
    assert result["matched_student"]["student_name"] == "Aatif Khan"
    meta = scan_queue.load_meta(page_id)
    assert meta is not None and meta.status == "read" and meta.attempts == 1 and meta.error == ""
    state = ocr_worker.load_state()
    assert state is not None and state.state == ocr_worker.STATE_IDLE  # nothing left to read


def test_the_worker_thread_takes_no_page_while_paused(
    queue_dir: Path,
    clock: Clock,
    database: Database,
    readers: Readers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _queue_pages(2)
    outcomes: list[str] = []
    real_next = ocr_worker.process_next

    def counted_next() -> str:
        outcome = real_next()
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(ocr_worker, "process_next", counted_next)
    stop = threading.Event()
    thread = threading.Thread(
        target=ocr_worker.run_forever,
        args=(stop,),
        kwargs={"idle_seconds": 0.01, "outage_seconds": 0.01},
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while outcomes.count(ocr_worker.PAUSED) < 20 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcomes[0] == ocr_worker.UNAVAILABLE
    assert set(outcomes[1:]) == {ocr_worker.PAUSED} and len(outcomes) > 20
    assert database.lookups == 1 and readers.id_calls == 1


# ================================================================== the probe
@dataclass
class _ProbeConnection:
    statements: list[str] = field(default_factory=list)
    closed: bool = False

    def cursor(self) -> Self:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, sql: str) -> int:
        self.statements.append(sql)
        return 1

    def fetchone(self) -> tuple[int]:
        return (1,)

    def close(self) -> None:
        self.closed = True


def test_the_probe_is_one_select_1_with_a_five_second_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[dict[str, Any]] = []
    connection = _ProbeConnection()

    def fake_connect(**kwargs: Any) -> _ProbeConnection:
        opened.append(kwargs)
        return connection

    # Every setting a stand-in: nothing from .env may reach a failure message.
    settings = {"DB_HOST": "db.example.invalid", "DB_USER": "u", "DB_PASS": "p", "DB_NAME": "n", "DB_PORT": 3306}
    for name, value in settings.items():
        monkeypatch.setattr(student_lookup, name, value)
    monkeypatch.setattr(student_lookup.pymysql, "connect", fake_connect)

    student_lookup.probe_database()

    assert opened[0]["connect_timeout"] == 5
    # A server that accepts the connection but never answers must not hang
    # the paused worker: the probe also limits each reply.
    assert opened[0]["read_timeout"] == 5 and opened[0]["write_timeout"] == 5
    assert connection.statements == ["SELECT 1"] and connection.closed
    # Every other caller keeps the twenty seconds it always had, and no
    # reply limit (PyMySQL's default), so long lookups are not cut short.
    student_lookup.get_connection()
    assert opened[1]["connect_timeout"] == 20
    assert opened[1]["read_timeout"] is None and opened[1]["write_timeout"] is None

    def refused(**_kwargs: Any) -> Any:
        raise _down()

    monkeypatch.setattr(student_lookup.pymysql, "connect", refused)
    with pytest.raises(student_lookup.DB_ERRORS):
        student_lookup.probe_database()
    monkeypatch.setattr(student_lookup, "DB_HOST", "")
    with pytest.raises(student_lookup.DatabaseNotConfigured):
        student_lookup.probe_database()


# ================================================================== the state file
def test_the_state_is_written_whole_and_read_back_by_api_batches(
    queue_dir: Path,
    clock: Clock,
    database: Database,
    readers: Readers,
    client: FlaskClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    [page_id] = _queue_pages(1)
    state_file = queue_dir / "_worker_state.json"
    moves: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def watched_replace(source: Any, target: Any) -> None:
        moves.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(scan_queue.os, "replace", watched_replace)
    assert ocr_worker.run_pass(budget_seconds=60) == 0
    monkeypatch.setattr(scan_queue.os, "replace", real_replace)

    # Each write was a whole file, moved into place from beside it.
    sources = [source for source, target in moves if target == state_file]
    assert len(sources) == 2  # running, then paused
    assert all(
        source.parent == queue_dir and re.fullmatch(r"\._worker_state\.json\.[0-9a-f]+\.tmp", source.name)
        for source in sources
    )
    assert list(queue_dir.glob("._worker_state.json*")) == []  # no temporary left behind

    body = client.get("/api/batches").get_json()
    assert body["queue"]["queued"] == 1
    assert body["queue"]["worker"] == {
        "state": "paused",
        "reason": ocr_worker.OUTAGE_REASON,
        "since": "2026-10-28T13:59:45",
        "next_try": "2026-10-28T14:00:15",
        "updated": "2026-10-28T13:59:45",
    }

    # Once the person has nothing waiting, there is nothing to explain.
    database.down = False
    clock.advance(30)
    assert ocr_worker.run_pass(budget_seconds=60) == 1
    body = client.get("/api/batches").get_json()
    assert body["queue"] == {"queued": 0, "reading": 0, "read": 1, "failed": 0}
    assert scan_queue.load_result(page_id) is not None


def test_a_state_that_cannot_be_written_leaves_the_last_whole_one(
    queue_dir: Path,
    clock: Clock,
    database: Database,
    readers: Readers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _queue_pages(1)
    assert ocr_worker.run_pass(budget_seconds=60) == 0
    state_file = queue_dir / "_worker_state.json"
    before = state_file.read_bytes()
    real_replace = os.replace

    def disk_full(source: Any, target: Any) -> None:
        if Path(target) == state_file:
            raise OSError(28, "No space left on device")
        real_replace(source, target)

    monkeypatch.setattr(scan_queue.os, "replace", disk_full)
    clock.advance(30)  # the probe is due, and booking the next one fails

    assert ocr_worker.run_pass(budget_seconds=60) == 0  # the worker carries on
    assert len(database.probes) == 1
    assert state_file.read_bytes() == before
    assert json.loads(before)["state"] == "paused"


def test_a_paused_worker_keeps_its_state_fresh_without_writing_on_every_look(
    queue_dir: Path,
    clock: Clock,
    database: Database,
    readers: Readers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _queue_pages(1)
    assert ocr_worker.run_pass(budget_seconds=60) == 0
    state_file = queue_dir / "_worker_state.json"
    writes: list[datetime] = []
    real_replace = os.replace

    def counted_replace(source: Any, target: Any) -> None:
        if Path(target) == state_file:
            writes.append(clock.now)
        real_replace(source, target)

    monkeypatch.setattr(scan_queue.os, "replace", counted_replace)
    # Thirty minutes of the watcher looking every ten seconds, through the
    # five-minute waits between probes.
    for _ in range(180):
        clock.advance(10)
        assert ocr_worker.run_pass(budget_seconds=60) == 0
        assert ocr_worker.reported_state()["state"] == "paused"

    assert len(database.probes) == 8  # at 30, 90, 210, 450, 750, 1050, 1350, 1650 s
    gaps = [(later - earlier).total_seconds() for earlier, later in itertools.pairwise(writes)]
    assert max(gaps) <= ocr_worker.STATE_REFRESH_SECONDS
    assert len(writes) <= 1800 / ocr_worker.STATE_REFRESH_SECONDS + len(database.probes)


@pytest.mark.parametrize("state", ["running", "paused"])
def test_a_busy_worker_silent_for_two_minutes_is_not_running(
    queue_dir: Path, clock: Clock, client: FlaskClient, state: str
) -> None:
    _queue_pages(1)
    _write_state(state, clock.now, next_try=_stamp(clock.now + timedelta(seconds=300)))

    clock.advance(120)
    assert ocr_worker.reported_state()["state"] == state
    assert client.get("/api/batches").get_json()["queue"]["worker"]["state"] == state

    clock.advance(1)
    assert ocr_worker.reported_state() == {"state": "not_running"}
    assert client.get("/api/batches").get_json()["queue"]["worker"] == {"state": "not_running"}


def test_a_quiet_idle_worker_is_idle_until_pages_wait_for_it(queue_dir: Path, clock: Clock) -> None:
    _write_state("idle", clock.now)
    clock.advance(3600)
    assert ocr_worker.reported_state()["state"] == "idle"

    _queue_pages(1)  # a live worker would have taken it
    assert ocr_worker.reported_state() == {"state": "not_running"}


@pytest.mark.parametrize("contents", [None, "{half a file", '{"state": "dancing"}', '["paused"]'])
def test_no_state_or_one_that_makes_no_sense_is_not_running(
    queue_dir: Path, clock: Clock, contents: str | None
) -> None:
    if contents is not None:
        queue_dir.mkdir(parents=True, exist_ok=True)
        (queue_dir / "_worker_state.json").write_text(contents, encoding="utf-8")

    assert ocr_worker.reported_state() == {"state": "not_running"}
    assert ocr_worker.process_next() == ocr_worker.IDLE  # and it never stops a worker


# ================================================================== the scan page
_STATUS_SCENARIO = r"""
  const page = await openPage();
  await answerBatches(page, __API__);
  const line = page.byId["received-status"];
  return { text: line.textContent, className: line.className };
"""


@needs_node
@pytest.mark.parametrize(
    ("state", "line", "look"),
    [
        ("paused", f"1 waiting for OCR. {PAUSED_LINE}", "section-status is-warn"),
        ("not_running", f"1 waiting for OCR. {NOT_RUNNING_LINE}", "section-status is-warn"),
        ("running", "1 waiting for OCR.", "section-status"),
    ],
)
def test_the_scan_page_says_why_scans_wait_from_api_batches(
    queue_dir: Path,
    clock: Clock,
    database: Database,
    readers: Readers,
    client: FlaskClient,
    tmp_path: Path,
    state: str,
    line: str,
    look: str,
) -> None:
    html = client.get("/").get_data(as_text=True)
    assert re.search(r'<p class="section-status" id="received-status" aria-live="polite">', html)
    if state == "paused":
        _queue_pages(1)
        assert ocr_worker.run_pass(budget_seconds=60) == 0  # the outage pauses it
    elif state == "not_running":
        _queue_pages(1)
        _write_state("running", clock.now - timedelta(minutes=5))
    else:
        _write_state("running", clock.now)
        _queue_pages(1)
    api = client.get("/api/batches").get_json()
    assert api["queue"]["worker"]["state"] == state

    out = _run_page(client, tmp_path, _STATUS_SCENARIO.replace("__API__", json.dumps(api)))

    assert out == {"text": line, "className": look}
