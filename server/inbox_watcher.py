"""Turn pages landing in people's Drive inboxes into scan results.

Runs as its own process, not inside the web app: gunicorn starts several
workers, and a watcher inside each would read the same page several times.
Only one watcher processes at a time - each takes a MySQL lock first, and a
second one (a stray --once, an old process a restart left behind) waits.

    python -m server.inbox_watcher            keep watching (production)
    python -m server.inbox_watcher --once     one pass, then exit (testing)
    python -m server.inbox_watcher --folders  who has shared an inbox with us
    python -m server.inbox_watcher --status   show open batches
    python -m server.inbox_watcher --open jyoti1.yadav@akanksha.org --school ABMPS --location Mumbai
    python -m server.inbox_watcher --close 12
    python -m server.inbox_watcher --queue-once   read the scan queue once, then exit

It also reads the scan queue (services/ocr_worker.py): pages sent from the
scan page wait there for OCR, and in production this process is the one that
reads them - once per loop, for up to QUEUE_PASS_SECONDS - so the server needs
no service of its own for it. Reading the queue writes nothing to the
database; a person confirms each document on the scan page.

For every open batch it lists that person's inbox folder, and for each new
file: downloads it, reads every page, looks each Student ID up within the
batch's school, records the result exactly as the scan page would have, and
renames the file to say what became of it - "[saved] Name_ID.jpg",
"[not found] ID.jpg", "[unread] page.jpg". The History row links to that
file, in the person's own Drive. No second upload.

A page belongs to the batch that was open when it reached Drive, so pages
still waiting when someone presses Stop are finished under that batch's
school, not read as whichever school is scanned next. A page that reaches
Drive after Stop, before the next Start, is read for no school: it is
listed as unread and renamed "[unread] page.jpg". Drive dates a page by when
its upload landed, not when it was scanned, so a page that lands just after
the next school was started may be the last school's final sheet: if its
student is not found in the new school, the school before is asked too.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from server.config import INBOX_POLL_SECONDS
from server.services import batch_store, drive_inbox, ocr_worker
from server.services.clock import IST, now_ist
from server.services.debug_log import debug_log
from server.services.extract_student_id import (
    ExtractError,
    ExtractResult,
    LookupUnavailable,
    confirm_student_id,
    extract_student_ids_from_upload_batch,
)
from server.services.history_store import log_scan_attempt
from server.services.warmup import warm_models_in_background

# A file that keeps failing is parked rather than retried forever: after this
# many attempts it is renamed "[failed] <name>" so a person sees it, and left
# alone from then on. Counted in the database, so a restart does not start
# the count again.
MAX_ATTEMPTS = 3

# The scan page calls the watcher dead once its heartbeat is 45 s old
# (inbox_routes.WATCHER_STALE_AFTER), and one pass can take far longer than
# that - a hard page goes through every OCR fallback. So the heartbeat has a
# thread of its own and beats every few seconds whatever the pass is doing...
HEARTBEAT_SECONDS = 10.0
# ...but not for a loop that has stopped moving. Nothing should go this long
# without reaching the next file; if it does, the watcher is stuck, and the
# page should say it is not running.
STUCK_AFTER_SECONDS = 30 * 60.0
# MySQL drops the watcher lock's connection, and the lock with it, after this
# long without a word on it - so a watcher whose host died without closing
# the connection holds the lock for minutes, not wait_timeout's eight hours.
# The heartbeat thread uses the connection every HEARTBEAT_SECONDS.
LOCK_IDLE_SECONDS = 120

# Pages still waiting when a batch closes are finished under that batch, so
# its folder is still looked at for a while after Stop...
LEFTOVER_WINDOW = timedelta(hours=6)
# ...including pages the scanning PC was still uploading when Stop was
# pressed. After this, a folder with nothing left for its closed batch is let
# go until its next Start.
SYNC_GRACE = timedelta(minutes=2)
# Drive dates a page by when its upload landed. One landing this soon after a
# batch closed may be that school's last sheet, still uploading when the next
# school was started, so its student is looked for in both schools. Longer
# than SYNC_GRACE: a missed second look files a real student as "[not found]"
# for good, while a needless one only runs where the open school has no match.
LATE_TAIL = timedelta(minutes=10)
# A renamed file seen again without its [result] bracket. Sooner than this it
# is Drive's listing lagging behind the rename; later, a person has taken the
# bracket off by hand to have the page read again.
RELIST_AFTER = timedelta(minutes=10)

# How long each loop may spend reading the scan queue before it goes back to
# the Drive inboxes. One page takes seconds to tens of seconds, so a long
# queue is read across loops and the inboxes are still looked at every minute
# or so.
QUEUE_PASS_SECONDS = 60.0

# The History row for a page that reached Drive between batches.
SET_ASIDE_MESSAGE = (
    "Reached Drive after Stop scanning and before the next Start, so it was "
    "not read for any school. Scan it again while a batch is open."
)

# Closed batches whose folder had nothing left for them on a pass after the
# sync grace. Their folder is not listed again until another batch opens.
_drained: set[int] = set()


def say(message: str) -> None:
    print(message, flush=True)
    debug_log(f"[INBOX] {message}")


# ------------------------------------------------------------- times
def _drive_time(value: str | None) -> datetime | None:
    """Drive's createdTime ("2026-10-27T05:12:33.123Z", UTC) as naive IST -
    the clock batches are opened and closed by (see clock.now_ist)."""
    # strptime's %z takes the trailing "Z" on every Python 3 the server may
    # run; fromisoformat only learnt it in 3.11.
    for pattern in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            moment = datetime.strptime(value or "", pattern)
        except ValueError:
            continue
        return moment.astimezone(IST).replace(tzinfo=None)
    return None


def _stored_time(value: datetime | str | None) -> datetime | None:
    """A DATETIME column as read (naive IST), or as batch_store serialises it."""
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def batch_for_page(
    created: datetime | None, batches: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, bool]:
    """The batch a page belongs to: the one open when it reached Drive.

    A page still waiting when its batch closed is read for that batch. One
    that reached Drive after a batch closed and before the next was opened
    is read for no school at all: it may be the tail of the last school's
    stack still syncing, or the start of the next, and a guess either way
    looks the form up among the wrong school's children. One that reached
    Drive just after the next batch opened may still be the last school's
    sheet; `tail_batch_for` says when to look there too. A page older than
    every batch on the folder (dropped in before its first Start) goes to
    the first batch.

    Args:
        created: When the page reached Drive, naive IST. None if Drive did
            not say, in which case only the open batch can take it.
        batches: One folder's batches, oldest first, starting with the last
            one that had closed before the oldest page arrived. They never
            overlap, since opening a batch closes the one before.

    Returns:
        (batch, False) to read the page under that batch; (None, True) to
        set it aside as unread and not attributed to any school; or
        (None, False) to leave it where it is for now.
    """
    after_a_close = False
    for batch in batches:
        is_open = batch.get("closed_at") is None
        if created is None:
            if is_open:
                return batch, False
            continue
        opened = _stored_time(batch.get("opened_at"))
        if opened is not None and created < opened:
            # Reached Drive before this batch was started.
            return (None, True) if after_a_close else (batch, False)
        closed = _stored_time(batch.get("closed_at"))
        if is_open or (closed is not None and created < closed):
            return batch, False
        after_a_close = True
    return None, after_a_close


def tail_batch_for(
    created: datetime | None, batch: dict[str, Any], batches: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """The batch whose last sheet a page may be, though it landed in `batch`.

    The scanning PC saves a sheet and Drive for Desktop uploads it a little
    later. Press Stop (or switch school) and then Start the next school before
    that upload lands, and Drive dates the sheet inside the new batch: looked
    up among the new school's children only, it was logged not found and
    renamed "[not found] ..." for good. Such a page is looked up in the
    school before too when the new one has no match (_read_and_record).

    Args:
        created: When the page reached Drive, naive IST.
        batch: The batch `batch_for_page` placed it in.
        batches: The folder's batches, oldest first, as `batch_for_page` saw
            them.

    Returns:
        The batch closed just before `batch` opened, when the page reached
        Drive within LATE_TAIL of that close and it was for another school;
        otherwise None.
    """
    if created is None:
        return None
    before: dict[str, Any] | None = None
    for candidate in batches:
        if int(candidate["id"]) == int(batch["id"]):
            break
        before = candidate
    else:
        return None  # not one of the folder's batches as listed this pass
    if before is None:
        return None
    opened = _stored_time(batch.get("opened_at"))
    closed = _stored_time(before.get("closed_at"))
    if opened is None or closed is None or created < opened or created - closed >= LATE_TAIL:
        return None
    same_school = (
        (before.get("school") or None, before.get("location") or None)
        == (batch.get("school") or None, batch.get("location") or None)
    )
    return None if same_school else before


# ------------------------------------------------------------- one page
def _read_and_record(
    result: ExtractResult,
    batch: dict[str, Any],
    link: str,
    tail: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Look one page's Student ID up and log it.

    Raises LookupUnavailable when the student database cannot be asked: the
    page is then neither logged nor named, and the file is tried again on a
    later pass. An outage is not a "not found".

    Args:
        result: What OCR read off the page.
        batch: The batch the page was placed in.
        link: The file's Drive link, for the History row.
        tail: The batch closed just before, when the page may be its last
            sheet (`tail_batch_for`). A student found only there is logged
            under it; one found in neither school is not found for `batch`.

    Returns:
        The name to give the file, and the id of the batch the page was
        logged under.
    """
    source = batch_store.source_tag(batch["id"])
    school = batch.get("school") or None
    location = batch.get("location") or None
    logged_under = batch

    if result.skipped or result.skip_reason:
        # Blank, declaration, or an empty ID box. Counted on the file below,
        # not written to History as an unread form.
        return "[skipped]", int(batch["id"])

    unread_message = result.error or result.message or "Could not read a Student ID"
    confirmed: ExtractResult | None = None
    if result.student_id:
        try:
            # Same call the browser makes when "check each form automatically" is on.
            confirmed = confirm_student_id(
                result.student_id, source=source, defer_log=True,
                school=school, location=location,
            )
            if tail is not None and not confirmed.found_in_db:
                # A not-found writes nothing, so asking the school before
                # second costs nothing when the student is in neither.
                earlier = confirm_student_id(
                    result.student_id, source=batch_store.source_tag(tail["id"]),
                    defer_log=True,
                    school=tail.get("school") or None, location=tail.get("location") or None,
                )
                if earlier.found_in_db:
                    say(f"batch {batch['id']}: {result.student_id} is not at {school} but "
                        f"is at {tail.get('school')}; its sheet landed just after batch "
                        f"{tail['id']} closed, so it is recorded there")
                    confirmed, logged_under = earlier, tail
        except LookupUnavailable:
            raise
        except ExtractError as exc:
            # Refused outright (not shaped like a Student ID). Trying again
            # gives the same answer, so it is recorded as an unread page.
            unread_message = exc.message

    if confirmed is None:
        # Nothing readable on the page. Recorded so the person sees a row for
        # every sheet they fed, not a silent gap. Never with whatever OCR
        # read: on an unread page that is a name, an email or a phone number.
        log_scan_attempt(
            confirmed_student_id="",
            found_in_db=False,
            source=source,
            message=unread_message,
            image_link=link,
            scan_school=school,
            scan_location=location,
        )
        return "[unread]", int(batch["id"])

    if not confirmed.ocr_student_id:
        confirmed.ocr_student_id = result.student_id
    student = confirmed.student or {}
    # The same row log_confirm_result writes, but written here so that a
    # failed insert raises: log_confirm_result swallows it to keep the scan
    # page's confirm working, which for the watcher meant renaming a page
    # that left no row in History. Raised, the file stays unrenamed and the
    # page is logged on a later pass.
    log_scan_attempt(
        confirmed_student_id=confirmed.ocr_student_id or confirmed.student_id,
        found_in_db=confirmed.found_in_db,
        matched_student_id=student.get("student_id"),
        student_name=student.get("student_name"),
        school_name=student.get("school_name"),
        source=batch_store.source_tag(logged_under["id"]),
        message=confirmed.message,
        image_link=link,
        matched_via=student.get("matched_via"),
        scan_school=logged_under.get("school") or None,
        scan_location=logged_under.get("location") or None,
    )

    if confirmed.found_in_db:
        name = (student.get("student_name") or "").strip() or confirmed.student_id
        return f"[saved] {name}_{confirmed.student_id}", int(logged_under["id"])
    return f"[not found] {confirmed.student_id}", int(logged_under["id"])


# ------------------------------------------------------------- one file
def _result_name(name: str, labels: list[str | None]) -> str:
    """The name a read file is given. A label is None for a page logged on
    an earlier pass, whose result was not kept."""
    # One page -> the app's usual name. A multi-page PDF keeps its own name
    # with a marker, since it holds several students.
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ".jpg"
    kept = [label for label in labels if label != "[skipped]"]
    if not kept and any(label == "[skipped]" for label in labels):
        return f"[skipped] {name}"
    labels = kept
    if len(labels) != 1:
        return f"[done {len(labels)} pages] {name}"
    label = labels[0]
    if label is None:
        return f"[done] {name}"
    # An unread page keeps its original name so it can still be found.
    return f"[unread] {name}" if label == "[unread]" else f"{label}{ext}"


def _read_pages(
    entry: dict[str, Any],
    batch: dict[str, Any],
    already: int,
    tail: dict[str, Any] | None = None,
) -> str:
    """Read every page of a claimed file and log the ones not yet logged.
    Returns the name to give the file."""
    file_id, name = entry["id"], entry.get("name", "scan")
    link = drive_inbox.file_link(file_id)
    results = extract_student_ids_from_upload_batch(drive_inbox.download(file_id), name)
    labels: list[str | None] = []
    # Pages per batch: a sheet found only in the school before is counted
    # there, where it is logged.
    counts: Counter[int] = Counter()
    for page, result in enumerate(results):
        if page < already:
            labels.append(None)  # logged on an earlier pass; not logged again
            counts[int(batch["id"])] += 1
            continue
        label, logged_under = _read_and_record(result, batch, link, tail)
        labels.append(label)
        counts[logged_under] += 1
        # Remembered page by page: when a later page fails, the retry
        # carries on from the next one instead of logging this one again.
        batch_store.file_page_logged(file_id, page + 1)
    new_name = _result_name(name, labels)
    batch_store.file_pages_finished(
        file_id, batch_id=int(batch["id"]), new_name=new_name, pages=len(results),
        counts=counts,
    )
    return new_name


def _set_aside(entry: dict[str, Any], batch: dict[str, Any]) -> str:
    """Record a page that reached Drive between batches without reading it.

    It is listed as unread, with no school, under the batch whose pass
    found it - what the Stop confirmation tells the person will happen.
    Returns the name to give the file.
    """
    file_id, name = entry["id"], entry.get("name", "scan")
    log_scan_attempt(
        confirmed_student_id="",
        found_in_db=False,
        source=batch_store.source_tag(batch["id"]),
        message=SET_ASIDE_MESSAGE,
        image_link=drive_inbox.file_link(file_id),
        scan_school=None,
        scan_location=None,
    )
    new_name = f"[unread] {name}"
    # No pages counted: none of it was read for the batch.
    batch_store.file_pages_finished(file_id, batch_id=int(batch["id"]), new_name=new_name, pages=0)
    return new_name


def process_file(
    entry: dict[str, Any],
    batch: dict[str, Any],
    claim: dict[str, Any] | None = None,
    *,
    set_aside: bool = False,
    tail: dict[str, Any] | None = None,
) -> bool:
    """Read one inbox file under `batch`, log its pages and rename it.

    The file is claimed in the database before anything is read, and each
    step is recorded as it is done, so a failure part-way never repeats a
    finished step: a page already logged is not logged again, and a file
    whose pages are all logged is only renamed. The rename is the last,
    cosmetic step - a folder the server cannot rename in (shared as Viewer)
    no longer has its pages read and logged again on every pass.

    Args:
        entry: The Drive file, as drive_inbox.list_new_pages lists it.
        batch: The batch the file belongs to, or with `set_aside` the batch
            whose pass found it.
        claim: What the database already holds for this file, if anything.
        set_aside: The file reached Drive between batches: log it as unread
            with no school instead of reading it.
        tail: The batch closed just before `batch`, when the file may be its
            last sheet still uploading (`tail_batch_for`).

    Returns:
        False if the file is not ready yet (Drive is still writing it).
    """
    file_id, name = entry["id"], entry.get("name", "scan")
    if int(entry.get("size") or 0) == 0:
        return False  # still being written by Drive; it will be back next pass

    if claim is None:
        batch_store.claim_file(
            file_id, folder_id=batch["folder_id"], batch_id=int(batch["id"]), name=name
        )
        claim = {"state": batch_store.FILE_READING, "pages_logged": 0}

    new_name = str(claim.get("new_name") or "")
    if claim.get("state") == batch_store.FILE_READING or not new_name:
        if set_aside:
            new_name = _set_aside(entry, batch)
        else:
            new_name = _read_pages(
                entry, batch, already=int(claim.get("pages_logged") or 0), tail=tail
            )

    drive_inbox.mark_done(file_id, new_name)
    batch_store.settle_file(file_id, batch_store.FILE_DONE)
    say(f"batch {batch['id']} ({batch['owner_email']}): {name} -> {new_name}")
    return True


def _park_failed(entry: dict[str, Any], batch: dict[str, Any], why: str) -> None:
    """Give up on a file: put one row in History saying why, then rename it
    "[failed] ..." if Drive lets us.

    The row comes first. Renamed and settled before it, a failed insert left
    a "[failed]" file with no row, never to be looked at again; now the file
    stays unsettled and the row is written on a later pass. It is parked
    straight after the row, so a rename that fails is not retried, and the
    rename - only cosmetic - comes last.
    """
    name = entry.get("name", "scan")
    log_scan_attempt(
        confirmed_student_id="", found_in_db=False,
        source=batch_store.source_tag(batch["id"]),
        message=f"Could not process this file: {why}"[:250],
        image_link=drive_inbox.file_link(entry["id"]),
        scan_school=batch.get("school") or None,
        scan_location=batch.get("location") or None,
    )
    batch_store.settle_file(entry["id"], batch_store.FILE_PARKED)
    try:
        drive_inbox.mark_done(entry["id"], f"[failed] {name}")
    except drive_inbox.DriveInboxError as exc:
        say(f"could not rename {name!r}: {exc}")
        return
    # Renamed: DONE rather than PARKED, so that taking the bracket off by
    # hand has it read again (see _claim_action).
    batch_store.settle_file(entry["id"], batch_store.FILE_DONE)


def _record_failure(entry: dict[str, Any], batch: dict[str, Any], exc: Exception) -> None:
    """Count a failed attempt at a file, and park it once it has failed too often.

    Never called for LookupUnavailable: an outage is not the file's fault,
    and counting it would park every waiting page as "[failed]" after a
    database blip of a few passes.
    """
    name = entry.get("name")
    why = exc.message if isinstance(exc, ExtractError) else str(exc)
    try:
        row = batch_store.file_failed(entry["id"], why)
        attempts = int((row or {}).get("attempts") or 0)
        say(f"batch {batch['id']}: {name!r} failed ({attempts}/{MAX_ATTEMPTS}): {why}")
        if row is None or attempts < MAX_ATTEMPTS:
            return
        if row.get("state") == batch_store.FILE_LOGGED:
            # Every page is in History already; only the rename keeps
            # failing. Stop trying - nothing would be read or logged again
            # either way, and there is no need for a "failed" row.
            batch_store.settle_file(entry["id"], batch_store.FILE_PARKED)
            say(f"batch {batch['id']}: leaving {name!r} under its own name - "
                f"its pages are recorded, but it cannot be renamed")
            return
        _park_failed(entry, batch, why)
    except Exception as again:  # noqa: BLE001 - the database may be what failed
        say(f"batch {batch['id']}: {name!r} failed: {why} (could not record it: {again})")


# ------------------------------------------------------------- one pass
def _claim_action(claim: dict[str, Any] | None) -> str:
    """What to do with a listed file, given what the database remembers of
    it: "new" (claim and read it), "resume" (carry on from where it got to)
    or "skip"."""
    if not claim:
        return "new"
    state = claim.get("state")
    if state in (batch_store.FILE_READING, batch_store.FILE_LOGGED):
        return "resume"
    if state == batch_store.FILE_DONE:
        updated = _stored_time(claim.get("updated_at"))
        if updated is None or now_ist() - updated >= RELIST_AFTER:
            return "new"
    return "skip"


def _run_folder(
    folder_id: str, latest: dict[str, Any], progress: Callable[[], None]
) -> tuple[int, bool]:
    """Process what is waiting in one inbox folder.

    Returns:
        (files handled, whether the student database was unreachable). On an
        outage the folder's remaining files are left for a later pass.
    """
    try:
        waiting = drive_inbox.list_new_pages(folder_id)
    except drive_inbox.DriveInboxError as exc:
        say(f"batch {latest['id']}: cannot list inbox of {latest['owner_email']}: {exc}")
        return 0, False

    claims = batch_store.file_claims([entry["id"] for entry in waiting]) if waiting else {}
    created = {entry["id"]: _drive_time(entry.get("createdTime")) for entry in waiting}
    known = [moment for moment in created.values() if moment is not None]
    batches = (
        batch_store.folder_batches(folder_id, closed_after=min(known) if known else None)
        if waiting else []
    )
    by_id = {int(batch["id"]): batch for batch in batches}

    handled = acted = 0
    for entry in waiting:
        claim = claims.get(entry["id"])
        action = _claim_action(claim)
        if action == "skip":
            continue
        if action == "new":
            claim = None
        batch: dict[str, Any] | None
        set_aside = False
        if claim is not None and (
            claim.get("state") == batch_store.FILE_LOGGED
            or int(claim.get("pages_logged") or 0) > 0
        ):
            # Some of it is in History already, under its batch: the rest
            # goes with it.
            batch_id = int(claim["batch_id"])
            batch = by_id.get(batch_id) or batch_store.get_batch(batch_id)
        else:
            # Placed by when it reached Drive, every time, so that a page
            # set aside is set aside again rather than read under the batch
            # whose pass first found it.
            batch, set_aside = batch_for_page(created[entry["id"]], batches)
            if set_aside:
                batch = latest
        if batch is None:
            continue  # Drive gave no time and nothing is open: wait for a Start
        tail = None if set_aside else tail_batch_for(created[entry["id"]], batch, batches)
        acted += 1
        progress()
        try:
            if process_file(entry, batch, claim, set_aside=set_aside, tail=tail):
                handled += 1
        except LookupUnavailable:
            # Nothing was logged or renamed for the page that hit it, and
            # its file is not counted as failing: it is read again once the
            # database answers. Every later file would only be downloaded
            # and OCR'd to fail the same way, so they wait too.
            say(f"batch {batch['id']}: the student database could not be reached; "
                f"{entry.get('name')!r} and the rest wait for a later pass")
            return handled, True
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
            _record_failure(entry, batch, exc)

    if not latest.get("open") and acted == 0:
        closed = _stored_time(latest.get("closed_at"))
        if closed is None or now_ist() - closed >= SYNC_GRACE:
            _drained.add(int(latest["id"]))
    return handled, False


def run_once(progress: Callable[[], None] | None = None) -> int:
    """Process everything waiting for every batch being watched.

    Args:
        progress: Called as the pass moves from one folder or file to the
            next, so the heartbeat can tell a long pass from a stuck one.

    Returns:
        The number of files read and renamed.
    """
    touch = progress or (lambda: None)
    by_folder: dict[str, list[dict[str, Any]]] = {}
    for batch in batch_store.batches_to_watch(closed_since=now_ist() - LEFTOVER_WINDOW):
        by_folder.setdefault(batch["folder_id"], []).append(batch)

    handled = 0
    for folder_id, batches in by_folder.items():
        latest = batches[-1]
        if not latest.get("open") and int(latest["id"]) in _drained:
            continue
        touch()
        done, unreachable = _run_folder(folder_id, latest, touch)
        handled += done
        if unreachable:
            break  # the same database serves every folder
    return handled


# ------------------------------------------------------------- watching
class Heartbeat:
    """Stamps the watcher's heartbeat every few seconds, from its own thread.

    Beating once per pass let one slow pass - a hard page, a PDF, a stack of
    pages synced at once - outlast the scan page's 45 s limit, and staff were
    told the watcher was not running while it was busy reading their pages.
    It only beats while this process holds the watcher lock (`active`) and
    the main loop is still moving (see STUCK_AFTER_SECONDS). Given the lock,
    each beat also keeps the lock's connection from going idle
    (LOCK_IDLE_SECONDS) and notices when the lock has gone.
    """

    def __init__(
        self,
        host: str,
        interval: float | None = None,
        stuck_after: float | None = None,
        lock: batch_store.WatcherLock | None = None,
    ) -> None:
        self.host = host
        self.interval = HEARTBEAT_SECONDS if interval is None else interval
        self.stuck_after = STUCK_AFTER_SECONDS if stuck_after is None else stuck_after
        self.lock = lock
        self.active = False
        self._last_progress = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def progress(self) -> None:
        """The main loop has moved on (a new pass, folder or file)."""
        self._last_progress = time.monotonic()

    def beat_once(self) -> bool:
        """Keep the lock's connection in use, then write one heartbeat if
        there is a working watcher to vouch for.

        Returns:
            True if a heartbeat was written.
        """
        if self.lock is not None:
            # Whether or not the loop looks stuck: one file of a long PDF can
            # outlast STUCK_AFTER_SECONDS, and letting MySQL drop the lock
            # then would free it mid-file for a waiting watcher. Without a
            # connection to keep, this asks MySQL nothing.
            try:
                held = self.lock.keepalive()
            except Exception as exc:  # noqa: BLE001 - a missed check must not end the thread
                debug_log(f"[INBOX] could not check the watcher lock: {exc}")
                held = False
            if not held:
                # Another watcher may hold it now; this one is no longer the
                # one to vouch for. The main loop takes it again, or waits.
                return False
        if not self.active or time.monotonic() - self._last_progress > self.stuck_after:
            return False
        try:
            batch_store.heartbeat(self.host)
        except Exception as exc:  # noqa: BLE001 - a missed beat must not end the thread
            debug_log(f"[INBOX] heartbeat failed: {exc}")
            return False
        return True

    def start(self) -> None:
        """Start beating in a background thread."""
        self._thread = threading.Thread(target=self._run, name="inbox-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 5)

    def _run(self) -> None:
        while True:
            self.beat_once()
            if self._stop.wait(self.interval):
                return


class LockLost(RuntimeError):
    """The watcher lock went part-way through a pass."""


def watch_pass(lock: batch_store.WatcherLock, beat: Heartbeat) -> str:
    """One turn of the watch loop: make sure this process holds the watcher
    lock, then do a pass.

    Returns:
        "ran"; "waiting" when another watcher holds the lock; "error" when
        the lock could not be checked (the database is unreachable); or
        "lost" when the lock went part-way through and the pass was cut short.
    """
    try:
        held = lock.acquire()
    except Exception as exc:  # noqa: BLE001 - keep the watcher alive
        beat.active = False
        say(f"pass skipped: could not check the watcher lock: {exc}")
        return "error"
    if not held:
        beat.active = False
        return "waiting"
    beat.progress()
    if not beat.active:
        say("processing (this watcher holds the lock)")
        beat.active = True
        # Beat straight away rather than at the thread's next tick: the scan
        # page uses it to warn when nobody is listening.
        beat.beat_once()

    def moving() -> None:
        beat.progress()
        # Once the lock has gone (its connection dropped), a waiting watcher
        # can take it and start on the same folder. Stop before the next
        # file rather than have two of them logging the same pages.
        if not lock.held:
            raise LockLost

    try:
        run_once(moving)
    except LockLost:
        beat.active = False
        say("lost the watcher lock part-way through a pass; stopped before the next file")
        return "lost"
    except Exception as exc:  # noqa: BLE001 - keep the watcher alive
        say(f"pass failed: {exc}")
    return "ran"


def read_scan_queue(progress: Callable[[], None] | None = None) -> int:
    """One pass over the scan queue, never letting a failure end the watcher.

    Args:
        progress: Called after each page (the heartbeat's progress).

    Returns:
        How many pages were handled.
    """
    try:
        handled = ocr_worker.run_pass(budget_seconds=QUEUE_PASS_SECONDS, progress=progress)
    except Exception as exc:  # noqa: BLE001 - keep the watcher alive
        say(f"scan queue pass failed: {exc}")
        return 0
    if handled:
        say(f"scan queue: read {handled} page(s)")
    return handled


def watch_queue_only() -> None:
    """Keep reading the scan queue when there are no Drive inboxes to watch."""
    warm_models_in_background()
    while True:
        if read_scan_queue() == 0:
            time.sleep(INBOX_POLL_SECONDS)


def watch() -> None:
    ok, why = drive_inbox.configured()
    if not ok:
        # Pages sent from the scan page still need reading, so the queue is
        # read even when the Drive inbox is not set up.
        say(f"Drive inbox not configured ({why}); reading the scan queue only")
        watch_queue_only()
        return
    say(f"watching Drive inboxes every {INBOX_POLL_SECONDS}s (Ctrl+C to stop)")
    warm_models_in_background()
    lock = batch_store.WatcherLock(idle_timeout=LOCK_IDLE_SECONDS)
    beat = Heartbeat(socket.gethostname(), lock=lock)
    beat.start()
    waiting_said = False
    try:
        while True:
            if watch_pass(lock, beat) == "waiting":
                if not waiting_said:
                    say("another watcher is processing the inboxes; "
                        "this one waits until it stops")
                    waiting_said = True
            else:
                waiting_said = False
            time.sleep(INBOX_POLL_SECONDS)
            # After the wait, not before it: the queue's claims are atomic,
            # so this needs no watcher lock, and every page counts as
            # progress for the heartbeat.
            read_scan_queue(beat.progress)
    finally:
        beat.stop()
        lock.release()


def run_once_locked() -> int | None:
    """One pass, only if no other watcher is processing.

    Returns:
        Files handled, or None if another watcher holds the lock.
    """
    lock = batch_store.WatcherLock()
    if not lock.acquire():
        return None
    try:
        return run_once()
    finally:
        lock.release()


# ------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--folders", action="store_true", help="list inboxes shared with us")
    parser.add_argument("--status", action="store_true", help="show open batches")
    parser.add_argument("--open", metavar="OWNER_EMAIL", help="open a batch for this person's inbox")
    parser.add_argument("--school", default="", help="school acronym for --open")
    parser.add_argument("--location", default="", help="Pune / Mumbai / Nagpur for --open")
    parser.add_argument("--close", type=int, metavar="BATCH_ID", help="close a batch")
    parser.add_argument(
        "--queue-once", action="store_true", help="read the scan queue once, then exit"
    )
    args = parser.parse_args(argv)

    if args.queue_once:
        print(f"read {read_scan_queue()} page(s) from the scan queue")
        return

    if args.folders:
        print(f"share your '{drive_inbox.DRIVE_INBOX_FOLDER_NAME}' folder with: "
              f"{drive_inbox.service_account_email()}\n")
        inboxes = drive_inbox.list_inboxes()
        if not inboxes:
            print("no inbox folders have been shared with the service account yet")
        for f in inboxes:
            print(f"{f['owner_email']:<36} {f['owner_name']:<20} {f['id']}")
        return

    if args.status:
        rows = batch_store.list_open_batches()
        if not rows:
            print("no open batches")
        for b in rows:
            print(f"batch {b['id']:<4} {b['owner_email']:<36} "
                  f"{b.get('school') or '-':<10} {b.get('location') or '-':<8} "
                  f"pages {b['pages_done']}  since {b['opened_at']}")
        return

    if args.open:
        inbox = drive_inbox.find_inbox(args.open)
        if not inbox:
            sys.exit(f"{args.open} has not shared an '{drive_inbox.DRIVE_INBOX_FOLDER_NAME}' "
                     f"folder with {drive_inbox.service_account_email()}")
        batch_id = batch_store.open_batch(
            folder_id=inbox["id"], owner_email=inbox["owner_email"],
            school=args.school or None, location=args.location or None,
            opened_by="cli",
        )
        print(f"opened batch {batch_id} for {inbox['owner_email']} "
              f"(school={args.school or '-'}, location={args.location or '-'})")
        return

    if args.close:
        print("closed" if batch_store.close_batch(args.close) else "no such open batch")
        return

    if args.once:
        handled = run_once_locked()
        if handled is None:
            # The running service already reads these inboxes; a second
            # reader would log and count every page it also picked up.
            sys.exit("another watcher is processing the inboxes; not starting a second one")
        print(f"handled {handled} file(s)")
        return

    watch()


if __name__ == "__main__":
    main()
