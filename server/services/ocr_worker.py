"""Read the pages waiting in the scan queue, one at a time, in the background.

Scanning stores pages and returns at once (services/scan_queue.py); this is
where they are read. For each page it turns the page upright, reads the
handwritten batch code in its corner, reads the Student ID with the same
pipeline the scan page always used, and looks the student up - read-only,
within the page's school or city - so the person checking the document sees
who it matched. It never writes to the database and never uploads anything:
that happens only when a person presses Confirm (services/documents.py).

Where it runs: in development a daemon thread inside the web app
(start_in_process_worker, from create_app); in production the inbox
watcher calls run_pass once per loop, so the server needs no new service.
Several readers may run at once - each claim is a single atomic rename, so
no page is read twice.

When the student database cannot be reached, the worker pauses, as a
circuit breaker would. The page whose lookup failed goes back in the queue
with its OCR reading kept in ocr.json, so only the lookup is done again
later, and the outage is never counted as one of its tries. Until the
database answers again the worker takes no page at all: each page's lookup
would otherwise wait out the connect timeout in turn, one after another,
and hold the queue for as long as the outage lasts. The whole worker
pauses, OCR included. OCR needs no database and could go on, but nothing
it read could be confirmed until the database is back (a confirm looks the
student up), and pausing everything is the one behaviour with no half-read
page to keep track of. While paused it asks the database one cheap
question - SELECT 1, with a short connect timeout - after 30 s, then 60,
120, 240, and every 300 s after that; the first answer resumes it.

What the worker is doing - running, paused or idle - is written to
<QUEUE_DIR>/_worker_state.json, so the web app can tell people why their
scans are waiting even when the worker is another process. The pause lives
in that file too: every reader of the queue honours it, and a watcher that
is restarted carries on with the back-off rather than starting it again.
"""

from __future__ import annotations

import contextlib
import math
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from server.config import (
    BLANK_PAGE_INK_RATIO,
    OCR_MAX_LONG_SIDE,
    OCR_WORKER_IN_PROCESS,
    PAGE_MAX_DPI,
    PAGE_MAX_LONG_SIDE,
    QUEUE_MAX_ATTEMPTS,
)
from server.services import scan_queue
from server.services.batch_code import read_batch_code
from server.services.clock import now_ist
from server.services.debug_log import debug_log
from server.services.extract_student_id import (
    LookupUnavailable,
    extract_from_image,
    lookup_student_read_only,
)
from server.services.form_layout import identify_form
from server.services.id_shape import is_canonical, looks_like_student_id
from server.services.preprocess import load_page_grey
from server.services.student_lookup import DB_ERRORS, probe_database

# What process_one did.
IDLE = "idle"  # nothing was waiting
READ = "read"
FAILED = "failed"  # gave up on the page after QUEUE_MAX_ATTEMPTS
REQUEUED = "requeued"  # it failed this time and will be tried again
UNAVAILABLE = "unavailable"  # the student database could not be asked
LOST = "lost"  # another reader took the page over
# What process_next did instead: nothing, the worker is paused for an outage.
PAUSED = "paused"

# What the worker is doing, as _worker_state.json says it.
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_IDLE = "idle"
# Never written: what the web app reports when no worker has written lately.
STATE_NOT_RUNNING = "not_running"
_WRITTEN_STATES = frozenset({STATE_RUNNING, STATE_PAUSED, STATE_IDLE})

OUTAGE_REASON = "The student database could not be reached."
# The pause's back-off: the first probe this long after the outage, then
# twice as long after each probe that fails, up to the cap.
OUTAGE_FIRST_BACKOFF_SECONDS = 30.0
OUTAGE_MAX_BACKOFF_SECONDS = 300.0
# A worker rewrites its state at least this often, even with nothing new to
# say, so that a state left alone for STATE_STALE_SECONDS can only mean the
# worker has stopped. A pause's five-minute wait is no exception.
STATE_REFRESH_SECONDS = 30.0
STATE_STALE_SECONDS = 120.0

BLANK_MESSAGE = "This page looks blank."

# The stale-claim sweep lists _reading/; once a minute is plenty for a
# claim that is only stale after ten.
RECLAIM_EVERY_SECONDS = 60.0
_last_reclaim = 0.0

# The readers, as module attributes so the tests can stand in for them.
read_code = read_batch_code
read_student_id = extract_from_image
lookup_student = lookup_student_read_only
identify = identify_form
probe = probe_database
# What the pause and the state file are timed by: India time, like every
# other stamp this app writes. A test moves it by hand.
clock: Callable[[], datetime] = now_ist


# ------------------------------------------------------------- one page
def process_next() -> str:
    """Read the next waiting page, unless the worker is paused for an outage.

    What the loops call. While paused this takes no page; once the back-off
    is up it probes the database, and reads on if the database answers.

    Returns:
        PAUSED while the database is still down; otherwise what process_one
        returned.
    """
    if not _may_read():
        return PAUSED
    return process_one()


def process_one() -> str:
    """Read the oldest waiting page now, if there is one - paused or not.

    The loops call process_next, which honours a pause; this always tries.

    Returns:
        IDLE, READ, FAILED, REQUEUED, UNAVAILABLE or LOST.
    """
    _maybe_reclaim()
    claim = scan_queue.claim_next()
    if claim is None:
        _enter(STATE_IDLE)
        return IDLE
    _enter(STATE_RUNNING)
    with _kept_fresh():
        return _process_claim(claim)


def _process_claim(claim: scan_queue.Claim) -> str:
    meta = scan_queue.load_meta(claim.page_id)
    if meta is None:
        # Its folder has gone; there is nothing to read.
        scan_queue.finish_claim(claim)
        return LOST

    if meta.attempts >= QUEUE_MAX_ATTEMPTS:
        # Its last reader died holding it (the sweep handed it back), and
        # that was the last try. A page that kills its reader every time
        # must not take the queue down with it for ever.
        return _give_up(claim, meta, "its reader stopped part-way")

    meta.attempts += 1
    meta.status = scan_queue.STATUS_READING
    scan_queue.save_meta(meta)
    try:
        result = read_page(meta)
    except LookupUnavailable:
        # An outage is not the page's fault: it is not counted as a try,
        # and the reading is kept in ocr.json so the retry only looks it up.
        meta.attempts -= 1
        meta.status = scan_queue.STATUS_QUEUED
        if scan_queue.still_held(claim):
            scan_queue.save_meta(meta)
        scan_queue.release_claim(claim)
        debug_log(f"[WORKER] {claim.page_id}: student database unavailable; queued again")
        # Every other page would wait out the same connect timeout in turn.
        _pause(OUTAGE_REASON)
        return UNAVAILABLE
    except Exception as exc:  # noqa: BLE001 - one bad page must not stop the queue
        why = f"{type(exc).__name__}: {exc}"
        if meta.attempts >= QUEUE_MAX_ATTEMPTS:
            return _give_up(claim, meta, why)
        meta.status = scan_queue.STATUS_QUEUED
        meta.error = why[:250]
        if scan_queue.still_held(claim):
            scan_queue.save_meta(meta)
        scan_queue.release_claim(claim)
        debug_log(f"[WORKER] {claim.page_id}: try {meta.attempts} failed ({why}); queued again")
        return REQUEUED

    # The result goes in first and the claim comes off after: a crash in
    # between leaves a page that is read, never one that is nothing at all.
    scan_queue.save_result(claim.page_id, result)
    if not scan_queue.finish_claim(claim):
        return LOST
    meta.status = scan_queue.STATUS_READ
    meta.error = ""
    scan_queue.save_meta(meta)
    debug_log(
        f"[WORKER] {claim.page_id}: code={result.code!r} "
        f"id={'yes' if result.student_id else 'no'} "
        f"matched={'yes' if result.matched_student else 'no'} orientation={result.orientation}"
    )
    return READ


def _give_up(claim: scan_queue.Claim, meta: scan_queue.PageMeta, why: str) -> str:
    meta.status = scan_queue.STATUS_FAILED
    meta.error = why[:250]
    # Failed is written before the claim comes off, for the same reason the
    # result is: never a page that is neither queued, read nor failed.
    scan_queue.save_meta(meta)
    scan_queue.finish_claim(claim)
    debug_log(f"[WORKER] {claim.page_id}: gave up after {meta.attempts} tries ({why})")
    return FAILED


def read_page(meta: scan_queue.PageMeta) -> scan_queue.PageResult:
    """Read one queued page and look its student up, read-only.

    Args:
        meta: The page.

    Returns:
        What to store as its result.json.

    Raises:
        LookupUnavailable: The student database could not be asked, however
            the lookup said so.
    """
    reading = scan_queue.load_ocr(meta.page_id)
    if reading is None:
        reading = read_ocr(meta)
        scan_queue.save_ocr(meta.page_id, reading)

    matched: dict[str, Any] | None = None
    message = reading.message
    if reading.student_id:
        try:
            matched = lookup_student(
                reading.student_id, school=meta.school or None, location=meta.location or None
            )
        except DB_ERRORS as exc:
            # The lookup is the only part of a page that needs the database,
            # so a database error here is an outage even when it comes up
            # unwrapped - never a try held against the page.
            raise LookupUnavailable() from exc
        if matched:
            message = f"Matched {matched.get('student_name') or matched.get('student_id')}"
        else:
            where = meta.school or (f"{meta.location} schools" if meta.location else "the student list")
            message = f"No student with ID {reading.student_id} at {where}."
    return replace(reading, matched_student=matched, message=message, read_at=_stamp())


def read_ocr(meta: scan_queue.PageMeta) -> scan_queue.PageResult:
    """OCR a page: orientation, batch code and Student ID. No database.

    The page is tried both ways up and kept the way its form layout matches
    better. A page with no layout to match (the signature page) is kept
    whichever way up its batch code reads; the upside-down try is read by
    the local recogniser only. An upside-down page has its page.jpg turned
    round, so the thumbnails and the PDF are upright too.
    """
    grey = scan_queue.load_page_grey_image(meta.page_id)
    if grey is None:
        grey = _normalise_again(meta)
    work = _working_copy(grey)

    angle = _layout_orientation(work)
    if angle is None:
        code = read_code(grey)
        angle = 0
        if code.code is None:
            # Only a guess that the page is upside down: if it is not, the
            # "top" corners of the turned page are the bottom of the form -
            # signatures and names - so they are read here, never by Google.
            flipped = read_code(cv2.rotate(grey, cv2.ROTATE_180), allow_vision=False)
            if flipped.code is not None:
                angle, code = 180, flipped
    else:
        code = read_code(grey if angle == 0 else cv2.rotate(grey, cv2.ROTATE_180))

    if angle == 180:
        grey = cv2.rotate(grey, cv2.ROTATE_180)
        work = cv2.rotate(work, cv2.ROTATE_180)
        scan_queue.replace_page_image(meta.page_id, grey)

    reading = scan_queue.PageResult(
        code=code.code.text if code.code else None,
        code_confidence=round(float(code.confidence), 3),
        code_raw=code.raw,
        code_region=code.region,
        orientation=angle,
        read_at=_stamp(),
    )
    if float(np.mean(work < 160)) < BLANK_PAGE_INK_RATIO:
        # The reader would only conclude the same, after a costly first pass.
        return replace(reading, blank=True, skip_reason="blank", message=BLANK_MESSAGE)

    label = meta.source_name or "scan"
    if meta.source_pages > 1:
        label = f"{label} (page {meta.source_page}/{meta.source_pages})"
    extracted = read_student_id(cv2.cvtColor(work, cv2.COLOR_GRAY2BGR), label)
    if extracted.skipped:
        reason = extracted.skip_reason or "blank"
        skipped_message = {
            "blank": BLANK_MESSAGE,
            "declaration": "Declaration page, no Student ID.",
            "empty_id": "Student ID line is empty.",
        }.get(reason, "This page was skipped.")
        return replace(
            reading,
            blank=reason == "blank",
            skip_reason=reason,
            message=skipped_message,
            form_language=extracted.form_language,
        )
    student_id = extracted.student_id if looks_like_student_id(extracted.student_id) else ""
    return replace(
        reading,
        student_id=student_id,
        student_raw_shape_ok=bool(student_id) and is_canonical(student_id),
        candidates=[student_id] if student_id else [],
        message=extracted.error or ("" if student_id else extracted.message),
        form_language=extracted.form_language if student_id else "",
        confidence=round(float(extracted.confidence), 3) if student_id else 0.0,
    )


def _layout_orientation(work: np.ndarray) -> int | None:
    """0 or 180, whichever way up the form's ruled layout matches better.

    None when neither does (a page with no Student ID rules on it).
    """
    best: tuple[float, int] | None = None
    for angle in (0, 180):
        image = work if angle == 0 else cv2.rotate(work, cv2.ROTATE_180)
        match = identify(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        if match is not None and (best is None or match.error < best[0]):
            best = (match.error, angle)
    return None if best is None else best[1]


def _working_copy(grey: np.ndarray) -> np.ndarray:
    """The page at the reader's working size (see OCR_MAX_LONG_SIDE)."""
    height, width = grey.shape[:2]
    longest = max(height, width)
    if longest <= OCR_MAX_LONG_SIDE:
        return grey
    scale = OCR_MAX_LONG_SIDE / longest
    return cv2.resize(
        grey, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )


def _normalise_again(meta: scan_queue.PageMeta) -> np.ndarray:
    """Rebuild a missing page.jpg from the stored original."""
    folder = scan_queue.page_dir(meta.page_id)
    original = folder / (meta.original or "")
    if not meta.original or not original.is_file():
        raise ValueError("the page image and its original are both missing")
    page = load_page_grey(
        original.read_bytes(), original.name, 1, max_dpi=PAGE_MAX_DPI, max_long_side=PAGE_MAX_LONG_SIDE
    )
    scan_queue.replace_page_image(meta.page_id, page.image)
    return page.image


def _maybe_reclaim() -> None:
    global _last_reclaim
    now = time.monotonic()
    if now - _last_reclaim < RECLAIM_EVERY_SECONDS:
        return
    _last_reclaim = now
    try:
        scan_queue.reclaim_orphaned()
        scan_queue.reclaim_stale()
    except OSError as exc:
        debug_log(f"[WORKER] stale-claim sweep failed: {exc}")


def _stamp() -> str:
    return now_ist().isoformat(timespec="seconds")


# ------------------------------------------------------------- the worker's state
@dataclass
class WorkerState:
    """_worker_state.json: what the OCR worker is doing, for the web app to show.

    Attributes:
        state: running, paused or idle.
        reason: Why it is paused; "" otherwise.
        since: When it entered this state, India time.
        next_try: When a paused worker next asks the database, India time;
            None when it is not paused.
        host: The machine the worker runs on.
        pid: The worker's process.
        updated: When this was last written, India time.
        backoff_seconds: How long the pause waits before next_try; doubled
            by each probe that fails, up to OUTAGE_MAX_BACKOFF_SECONDS.
    """

    state: str
    reason: str = ""
    since: str = ""
    next_try: str | None = None
    host: str = ""
    pid: int = 0
    updated: str = ""
    backoff_seconds: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerState | None:
        """Build from the file, or None when it holds no state this version knows."""
        if data.get("state") not in _WRITTEN_STATES:
            return None
        try:
            return cls(
                state=str(data["state"]),
                reason=str(data.get("reason") or ""),
                since=str(data.get("since") or ""),
                next_try=str(data["next_try"]) if data.get("next_try") else None,
                host=str(data.get("host") or ""),
                pid=int(data.get("pid") or 0),
                updated=str(data.get("updated") or ""),
                backoff_seconds=float(data.get("backoff_seconds") or 0.0),
            )
        except (TypeError, ValueError):
            return None

    def for_page(self) -> dict[str, Any]:
        """What the scan page is told: everything but the worker's host and pid.

        Those stay on the server, like a database error's detail does.
        """
        return {
            "state": self.state,
            "reason": self.reason,
            "since": self.since,
            "next_try": self.next_try,
            "updated": self.updated,
        }


# The worker's own writes, one at a time: the thread that keeps a long page's
# state fresh must never put "running" back over a pause just written.
_state_lock = threading.Lock()


def load_state(path: Path | None = None) -> WorkerState | None:
    """The worker's state as last written, or None when there is none to read.

    Args:
        path: The state file; the queue's own when None.

    Returns:
        The state, or None for a missing, unreadable or unknown one.
    """
    data = scan_queue.read_json(path or scan_queue.worker_state_path())
    return WorkerState.from_dict(data) if data else None


def reported_state() -> dict[str, Any]:
    """What the OCR worker is doing, as GET /api/batches tells the scan page.

    The state file as the worker last wrote it, or {"state": "not_running"}
    when there is none, or when a worker that said it was busy has not
    written it for STATE_STALE_SECONDS - a live one rewrites it far more
    often. An idle worker may fall quiet, but not while pages are waiting:
    a live one would have taken them.

    Returns:
        WorkerState.for_page(), or {"state": "not_running"}.
    """
    current = load_state()
    if current is None:
        return {"state": STATE_NOT_RUNNING}
    if _age_seconds(current.updated, clock()) > STATE_STALE_SECONDS and (
        current.state != STATE_IDLE or scan_queue.queue_state().pending
    ):
        return {"state": STATE_NOT_RUNNING}
    return current.for_page()


def _enter(
    state: str,
    *,
    reason: str = "",
    next_try: datetime | None = None,
    backoff: float = 0.0,
    path: Path | None = None,
) -> WorkerState:
    """Record that the worker is in `state`, keeping `since` when it already was.

    Saying the same again only rewrites the file once STATE_REFRESH_SECONDS
    have passed: often enough to show the worker is alive, without a write
    on every two-second look at an empty queue.
    """
    target = path or scan_queue.worker_state_path()
    with _state_lock:
        now = clock()
        current = load_state(target)
        wanted = WorkerState(
            state=state,
            reason=reason,
            since=_iso(now),
            next_try=None if next_try is None else _iso(next_try),
            backoff_seconds=backoff,
        )
        if current is not None and current.state == state:
            wanted.since = current.since or wanted.since
            same = (current.reason, current.next_try, current.backoff_seconds) == (
                wanted.reason,
                wanted.next_try,
                wanted.backoff_seconds,
            )
            if same and 0 <= _age_seconds(current.updated, now) < STATE_REFRESH_SECONDS:
                return current
        _save_state(wanted, target, now)
        return wanted


def _save_state(state: WorkerState, path: Path, now: datetime) -> None:
    """Write the state whole under a temporary name and move it into place.

    A web app reading at the same moment sees the old state or the new one,
    never half of either. A failed write is logged and the worker carries
    on: reading pages matters more than reporting on them.
    """
    state.host = socket.gethostname()
    state.pid = os.getpid()
    state.updated = _iso(now)
    try:
        scan_queue.write_json_atomic(path, asdict(state))
    except OSError as exc:
        debug_log(f"[WORKER] could not write {path.name}: {exc}")


def _refresh_running(path: Path) -> None:
    """Rewrite a running worker's state as it stands, to show it is still alive."""
    with _state_lock:
        current = load_state(path)
        if current is not None and current.state == STATE_RUNNING:
            _save_state(current, path, clock())


@contextlib.contextmanager
def _kept_fresh() -> Iterator[None]:
    """Keep a running worker's state fresh while it reads one page.

    A page can take minutes (Vision, a lookup waiting for its connection),
    longer than the web app waits before calling a silent worker stopped.
    The file is fixed before the thread starts, so it writes only where
    this page's worker does, and the thread ends with the page.
    """
    path = scan_queue.worker_state_path()
    done = threading.Event()

    def keep() -> None:
        while not done.wait(STATE_REFRESH_SECONDS):
            _refresh_running(path)

    keeper = threading.Thread(target=keep, name="ocr-worker-state", daemon=True)
    keeper.start()
    try:
        yield
    finally:
        done.set()
        keeper.join(timeout=5.0)


def _pause(reason: str) -> None:
    """Take no more pages until the database answers again (see _may_read)."""
    first = OUTAGE_FIRST_BACKOFF_SECONDS
    _enter(STATE_PAUSED, reason=reason, next_try=clock() + timedelta(seconds=first), backoff=first)
    debug_log(f"[WORKER] paused: {reason} The database is asked again in {first:.0f}s.")


def _may_read() -> bool:
    """Whether the loops may take a page now.

    Yes, unless the worker is paused. Paused, it waits out the back-off and
    then probes the database once: an answer resumes the worker, no answer
    doubles the wait, up to OUTAGE_MAX_BACKOFF_SECONDS. The next wait is
    booked before the probe goes, so a probe that hangs or breaks in some
    unforeseen way still cannot have the loop probe again straight away.
    """
    current = load_state()
    if current is None or current.state != STATE_PAUSED:
        return True
    now = clock()
    backoff = min(
        max(current.backoff_seconds, OUTAGE_FIRST_BACKOFF_SECONDS), OUTAGE_MAX_BACKOFF_SECONDS
    )
    due = _parse_stamp(current.next_try)
    # A next try further off than the whole wait means the clock went back:
    # it is due now, not after however far the clock went.
    if due is not None and now < due <= now + timedelta(seconds=backoff):
        # Nothing new to say; this only shows the paused worker is alive.
        _enter(STATE_PAUSED, reason=current.reason, next_try=due, backoff=backoff)
        return False
    longer = min(backoff * 2, OUTAGE_MAX_BACKOFF_SECONDS)
    _enter(
        STATE_PAUSED,
        reason=current.reason or OUTAGE_REASON,
        next_try=now + timedelta(seconds=longer),
        backoff=longer,
    )
    try:
        probe()
    except DB_ERRORS as exc:
        debug_log(
            f"[WORKER] the student database still cannot be reached "
            f"({type(exc).__name__}: {exc}); asked again in {longer:.0f}s"
        )
        return False
    debug_log("[WORKER] the student database answers again; reading the queue")
    _enter(STATE_RUNNING)
    return True


def _seconds_to_next_try() -> float | None:
    """How long until a paused worker's next probe; None when it is not paused."""
    current = load_state()
    if current is None or current.state != STATE_PAUSED:
        return None
    due = _parse_stamp(current.next_try)
    return None if due is None else max(0.0, (due - clock()).total_seconds())


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _parse_stamp(text: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def _age_seconds(stamp: str, now: datetime) -> float:
    """Seconds since `stamp` (India time); infinite when it cannot be read."""
    moment = _parse_stamp(stamp)
    return math.inf if moment is None else (now - moment).total_seconds()


# ------------------------------------------------------------- loops
def run_pass(
    budget_seconds: float = 60.0,
    progress: Callable[[], None] | None = None,
    max_pages: int | None = None,
) -> int:
    """Read waiting pages until none is left or the time budget is spent.

    The inbox watcher calls this once per loop; the budget keeps a long
    queue from holding up the Drive inboxes. It stops at the first database
    outage, and takes no page while the worker is paused for one - only the
    probe, once the back-off is up.

    Args:
        budget_seconds: Start no new page after this long.
        progress: Called after each page, so the watcher's heartbeat can
            tell a long pass from a stuck one.
        max_pages: Stop after this many pages.

    Returns:
        How many pages were read, failed or re-queued.
    """
    deadline = time.monotonic() + budget_seconds
    handled = 0
    while time.monotonic() < deadline and (max_pages is None or handled < max_pages):
        outcome = process_next()
        if outcome in (IDLE, UNAVAILABLE, PAUSED):
            # Every other page would hit the same outage; wait for the next pass.
            break
        handled += 1
        if progress is not None:
            progress()
    return handled


def run_forever(
    stop: threading.Event, idle_seconds: float = 2.0, outage_seconds: float = STATE_REFRESH_SECONDS
) -> None:
    """Keep reading the queue until `stop` is set.

    Args:
        stop: Set it to end the loop.
        idle_seconds: How long to wait when nothing is queued.
        outage_seconds: The longest wait while the database is down: the
            loop wakes for the next probe, and at least this often to keep
            the paused state fresh for the web app.
    """
    debug_log("[WORKER] reading the scan queue")
    while not stop.is_set():
        try:
            outcome = process_next()
        except Exception as exc:  # noqa: BLE001 - the loop must outlive any one failure
            debug_log(f"[WORKER] pass failed: {type(exc).__name__}: {exc}")
            outcome = UNAVAILABLE
        if outcome == IDLE:
            stop.wait(idle_seconds)
        elif outcome in (UNAVAILABLE, PAUSED):
            until = _seconds_to_next_try()
            # At least a second (or outage_seconds, when that is shorter), so
            # a state that cannot be written - whose probe would always look
            # due - never makes a busy loop.
            stop.wait(outage_seconds if until is None else min(outage_seconds, max(until, 1.0)))


# ------------------------------------------------------------- in the web app
_thread: threading.Thread | None = None
_stop = threading.Event()
_thread_lock = threading.Lock()


def should_run_in_process(debug: bool) -> bool:
    """Whether this web app process should read the queue itself.

    OCR_WORKER_IN_PROCESS decides when it is set; otherwise only a debug
    (development) server does, since under gunicorn every worker would
    start a reader of its own.
    """
    return debug if OCR_WORKER_IN_PROCESS is None else OCR_WORKER_IN_PROCESS


def start_in_process_worker() -> bool:
    """Start this process's one reader thread, if it is not running already.

    Returns:
        True when a thread was started by this call.
    """
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return False
        _stop.clear()
        _thread = threading.Thread(
            target=run_forever, args=(_stop,), name="ocr-queue-worker", daemon=True
        )
        _thread.start()
        return True


def stop_in_process_worker(timeout: float = 5.0) -> None:
    """Ask the reader thread to stop after its current page, and wait a little."""
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=timeout)


def in_process_worker_running() -> bool:
    """Whether this process's reader thread is alive."""
    return _thread is not None and _thread.is_alive()
