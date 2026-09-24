"""Turn pages landing in people's Drive inboxes into scan results.

Runs as its own process, not inside the web app: gunicorn starts several
workers, and a watcher inside each would read the same page several times.

    python -m server.inbox_watcher            keep watching (production)
    python -m server.inbox_watcher --once     one pass, then exit (testing)
    python -m server.inbox_watcher --folders  who has shared an inbox with us
    python -m server.inbox_watcher --status   show open batches
    python -m server.inbox_watcher --open jyoti1.yadav@akanksha.org --school ABMPS --location Mumbai
    python -m server.inbox_watcher --close 12

For every open batch it lists that person's inbox folder, and for each new
file: downloads it, reads every page, looks each Student ID up within the
batch's school, records the result exactly as the scan page would have, and
renames the file to say what became of it - "[saved] Name_ID.jpg",
"[not found] ID.jpg", "[unread] page.jpg". The History row links to that
file, in the person's own Drive. No second upload.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from collections import Counter

from server.config import INBOX_POLL_SECONDS
from server.services import batch_store, drive_inbox
from server.services.debug_log import debug_log
from server.services.extract_student_id import (
    ExtractError,
    ExtractResult,
    confirm_student_id,
    extract_student_ids_from_upload_batch,
    log_confirm_result,
)
from server.services.history_store import log_scan_attempt
from server.services.warmup import warm_models_in_background

# A file that keeps failing is parked rather than retried forever: after this
# many attempts it is renamed "[failed] <name>" so a person sees it.
MAX_ATTEMPTS = 3
_attempts: Counter[str] = Counter()


def say(message: str) -> None:
    print(message, flush=True)
    debug_log(f"[INBOX] {message}")


# ------------------------------------------------------------- one page
def _read_and_record(result: ExtractResult, batch: dict, link: str) -> str:
    """Look one page's Student ID up and log it. Returns the name to give the file."""
    source = batch_store.source_tag(batch["id"])
    school = batch.get("school") or None
    location = batch.get("location") or None

    if not result.student_id:
        # Nothing readable on the page. Recorded so the person sees a row for
        # every sheet they fed, not a silent gap.
        log_scan_attempt(
            confirmed_student_id=result.ocr_student_id or "",
            found_in_db=False,
            source=source,
            message=result.error or result.message or "Could not read a Student ID",
            image_link=link,
            scan_school=school,
            scan_location=location,
        )
        return "[unread]"

    # Same call the browser makes when "check each form automatically" is on.
    confirmed = confirm_student_id(
        result.student_id, source=source, defer_log=True,
        school=school, location=location,
    )
    if not confirmed.ocr_student_id:
        confirmed.ocr_student_id = result.student_id
    log_confirm_result(confirmed, image_link=link, scan_school=school, scan_location=location)

    student = confirmed.student or {}
    if confirmed.found_in_db:
        name = (student.get("student_name") or "").strip() or confirmed.student_id
        return f"[saved] {name}_{confirmed.student_id}"
    return f"[not found] {confirmed.student_id}"


# ------------------------------------------------------------- one file
def process_file(entry: dict, batch: dict) -> None:
    file_id, name = entry["id"], entry.get("name", "scan")
    if int(entry.get("size") or 0) == 0:
        return  # still being written by Drive; it will be back next pass

    link = drive_inbox.file_link(file_id)
    results = extract_student_ids_from_upload_batch(drive_inbox.download(file_id), name)
    labels = [_read_and_record(result, batch, link) for result in results]

    # One page -> the app's usual name. A multi-page PDF keeps its own name
    # with a marker, since it holds several students.
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ".jpg"
    if len(labels) == 1:
        # An unread page keeps its original name so it can still be found.
        new_name = f"[unread] {name}" if labels[0] == "[unread]" else f"{labels[0]}{ext}"
    else:
        new_name = f"[done {len(labels)} pages] {name}"

    drive_inbox.mark_done(file_id, new_name)
    batch_store.add_pages_done(batch["id"], len(results))
    say(f"batch {batch['id']} ({batch['owner_email']}): {name} -> {new_name}")


def _park_failed(entry: dict, batch: dict, why: str) -> None:
    try:
        drive_inbox.mark_done(entry["id"], f"[failed] {entry.get('name', 'scan')}")
    except drive_inbox.DriveInboxError as exc:
        say(f"could not rename {entry.get('name')!r}: {exc}")
    log_scan_attempt(
        confirmed_student_id="", found_in_db=False,
        source=batch_store.source_tag(batch["id"]),
        message=f"Could not process this file: {why}"[:250],
        image_link=drive_inbox.file_link(entry["id"]),
        scan_school=batch.get("school") or None,
        scan_location=batch.get("location") or None,
    )


# ------------------------------------------------------------- one pass
def run_once() -> int:
    """Process everything waiting for every open batch. Returns files handled."""
    handled = 0
    for batch in batch_store.list_open_batches():
        try:
            waiting = drive_inbox.list_new_pages(batch["folder_id"])
        except drive_inbox.DriveInboxError as exc:
            say(f"batch {batch['id']}: cannot list inbox of {batch['owner_email']}: {exc}")
            continue

        for entry in waiting:
            key = entry["id"]
            try:
                process_file(entry, batch)
                handled += 1
                _attempts.pop(key, None)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
                _attempts[key] += 1
                say(f"batch {batch['id']}: {entry.get('name')!r} failed "
                    f"({_attempts[key]}/{MAX_ATTEMPTS}): {exc}")
                if _attempts[key] >= MAX_ATTEMPTS:
                    _park_failed(entry, batch, str(exc))
                    _attempts.pop(key, None)
    return handled


def watch() -> None:
    ok, why = drive_inbox.configured()
    if not ok:
        say(f"not configured: {why}")
        sys.exit(2)
    say(f"watching Drive inboxes every {INBOX_POLL_SECONDS}s (Ctrl+C to stop)")
    warm_models_in_background()
    host = socket.gethostname()
    while True:
        try:
            # Heartbeat first: the scan page uses it to warn when nobody is
            # listening, so it must beat even on a pass with nothing to do.
            batch_store.heartbeat(host)
            run_once()
        except Exception as exc:  # noqa: BLE001 - keep the watcher alive
            say(f"pass failed: {exc}")
        time.sleep(INBOX_POLL_SECONDS)


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
    args = parser.parse_args(argv)

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
        print(f"handled {run_once()} file(s)")
        return

    watch()


if __name__ == "__main__":
    main()
