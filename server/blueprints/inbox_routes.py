"""Start / stop scanning into the signed-in person's Drive inbox, and watch results.

The scan page calls these. Start opens a batch tying this person's "OCR Inbox"
folder to the school they picked; the watcher (a separate process) reads pages
that land there and logs results; the page polls for them; Stop closes the
batch. Nothing here touches the scanner - the person's scanner app saves
straight into the folder, which is what makes the same route work on Windows
and Mac.
"""

from __future__ import annotations

from datetime import timedelta

from flask import Blueprint, Response, jsonify, request

from server.services import batch_store, drive_inbox
from server.services.auth import current_user, require_login
from server.services.clock import now_ist
from server.services.debug_log import debug_log
from server.services.student_lookup import school_progress

inbox_bp = Blueprint("inbox", __name__)
inbox_bp.before_request(require_login)

# The watcher stamps a heartbeat every few seconds, from a thread of its own
# so that a long pass does not go quiet. Older than this and it is not
# running, and pages will sit unread - say so up front.
WATCHER_STALE_AFTER = timedelta(seconds=45)

DRIVE_UNREACHABLE = "Could not reach Google Drive just now. Try again in a minute."

# What a Stop with pages still waiting actually does, for the confirmation.
# The watcher matches each page to the batch that was open when it reached
# Drive (inbox_watcher.batch_for_page): the waiting pages are still read for
# this school, so scanning them again would only log them twice. A sheet
# that lands after Stop is set aside as unread if nothing has been started by
# then; if the next school has been, it is read for that school, or - when it
# lands within inbox_watcher.LATE_TAIL of this Stop - for this one if only
# this school has the student (inbox_watcher.tail_batch_for).
STOP_PENDING_MESSAGE = (
    "Stop now anyway? Pages already in your Drive folder are still read for "
    "this school, so there is no need to scan them again. A sheet that "
    "reaches Drive after you stop is listed as unread if no school has been "
    "started by then. If one has, it is read for that school, or for this "
    "one if its student is found here and not there."
)


def _me() -> str:
    return ((current_user() or {}).get("email") or "").strip().lower()


def _watcher_state() -> dict:
    seen = batch_store.watcher_last_seen()
    # The watcher stamps last_seen from the same IST clock, so this compares
    # like with like wherever either process happens to run.
    alive = bool(seen and now_ist() - seen < WATCHER_STALE_AFTER)
    return {"alive": alive, "last_seen": seen.isoformat(sep=" ", timespec="seconds") if seen else None}


def _pending_in_inbox(folder_id: str | None) -> int | None:
    """How many pages are sitting in this inbox unread. None if it cannot be
    determined right now (Drive hiccup, no folder yet) - the panel should
    just not show the number rather than lie."""
    if not folder_id:
        return None
    try:
        waiting = drive_inbox.list_new_pages(folder_id)
    except drive_inbox.DriveInboxError as exc:
        debug_log(f"[INBOX] pending count failed: {exc}")
        return None
    # A file whose pages are all recorded but which could not be renamed (a
    # folder shared as Viewer), or one the watcher has given up on, keeps
    # its own name but is not waiting for anything. Counted, it held Stop at
    # "pages still waiting" for ever.
    try:
        settled = batch_store.settled_file_ids([entry["id"] for entry in waiting])
    except Exception as exc:  # noqa: BLE001 - only a count; Drive's number will do
        debug_log(f"[INBOX] could not check which pages are settled: {exc}")
        settled = set()
    return sum(1 for entry in waiting if entry["id"] not in settled)


@inbox_bp.get("/api/inbox/status")
def inbox_status() -> tuple[Response, int]:
    """Everything the page needs to draw the panel in one call."""
    ok, why = drive_inbox.configured()
    if not ok:
        return jsonify({"configured": False, "reason": why}), 200

    email = _me()
    open_batch = batch_store.open_batch_for(email)
    # Progress numbers: prefer the open batch's own school (that is the one
    # being scanned right now); otherwise honour whatever the page passes in
    # its query so the counter updates the moment the school picker changes.
    if open_batch:
        progress = school_progress(open_batch.get("school"), open_batch.get("location"))
    else:
        progress = school_progress(
            request.args.get("school"),
            request.args.get("location"),
        )
    payload = {
        "configured": True,
        "service_account": drive_inbox.service_account_email(),
        "folder_name": drive_inbox.DRIVE_INBOX_FOLDER_NAME,
        "inbox": {"found": False},
        "batch": open_batch,
        "watcher": _watcher_state(),
        "progress": progress,
    }
    try:
        inbox = drive_inbox.find_inbox(email)
    except drive_inbox.DriveInboxError as exc:
        # Drive's own message stays in the server log; the page gets a
        # sentence a person can act on.
        debug_log(f"[INBOX] status: could not look up the inbox: {exc}")
        payload["inbox"] = {"found": False, "error": DRIVE_UNREACHABLE}
        return jsonify(payload), 200
    if inbox:
        payload["inbox"] = {"found": True, "folder_id": inbox["id"], "shared_at": inbox["shared_at"]}
        # So the idle panel can say "3 pages already waiting in Drive" before
        # anyone presses Start - otherwise you cannot tell if the scanner has
        # been feeding pages into the folder while nobody was watching.
        payload["pending"] = _pending_in_inbox(inbox["id"])
    return jsonify(payload), 200


@inbox_bp.post("/api/inbox/start")
def inbox_start() -> tuple[Response, int]:
    body = request.get_json(silent=True) or {}
    school = str(body.get("school") or "").strip()
    location = str(body.get("location") or "").strip()
    email = _me()

    try:
        inbox = drive_inbox.find_inbox(email)
    except drive_inbox.DriveInboxError as exc:
        debug_log(f"[INBOX] start: could not look up the inbox: {exc}")
        return jsonify({"error": DRIVE_UNREACHABLE}), 502
    if not inbox:
        return jsonify({
            "error": (
                f"No '{drive_inbox.DRIVE_INBOX_FOLDER_NAME}' folder has been shared from "
                f"{email}. Share it with {drive_inbox.service_account_email()} as Editor, "
                f"then try again."
            )
        }), 409

    batch_id = batch_store.open_batch(
        folder_id=inbox["id"], owner_email=email,
        school=school or None, location=location or None, opened_by=email,
    )
    debug_log(f"[INBOX] {email} started batch {batch_id} school={school!r} location={location!r}")
    return jsonify({
        "batch": batch_store.get_batch(batch_id),
        "watcher": _watcher_state(),
        "progress": school_progress(school or None, location or None),
    }), 200


@inbox_bp.post("/api/inbox/stop")
def inbox_stop() -> tuple[Response, int]:
    batch = batch_store.open_batch_for(_me())
    if not batch:
        return jsonify({"closed": False, "batch": None}), 200

    # The panel asks for a confirmation when there are files still in the
    # inbox. Pages already there when the batch closes are still read for
    # this batch's school - the watcher goes by when each page reached Drive
    # - but one that arrives after Stop may not be (STOP_PENDING_MESSAGE); the
    # person should know that before closing. Force=true is the confirmed-close path.
    body = request.get_json(silent=True) or {}
    if not body.get("force"):
        pending = _pending_in_inbox(batch.get("folder_id")) or 0
        if pending > 0:
            return jsonify({
                "closed": False,
                "batch": batch,
                "pending": pending,
                "reason": "pending",
                "message": STOP_PENDING_MESSAGE,
            }), 409

    batch_store.close_batch(int(batch["id"]))
    return jsonify({"closed": True, "batch": batch_store.get_batch(int(batch["id"]))}), 200


@inbox_bp.get("/api/inbox/results")
def inbox_results():
    """Rows a batch has produced since `after` (a log row id). The page polls
    this while scanning and appends whatever is new."""
    try:
        batch_id = int(request.args.get("batch") or 0)
        after = int(request.args.get("after") or 0)
    except ValueError:
        return jsonify({"error": "batch and after must be numbers."}), 400

    batch = batch_store.get_batch(batch_id)
    if not batch or (batch.get("owner_email") or "") != _me():
        # Not yours to look at, or gone. Same answer either way.
        return jsonify({"error": "That batch is not available."}), 404

    return jsonify({
        "batch": batch,
        "rows": batch_store.batch_results(batch_id, after_id=after),
        "watcher": _watcher_state(),
        "progress": school_progress(batch.get("school"), batch.get("location")),
        "pending": _pending_in_inbox(batch.get("folder_id")),
    }), 200
