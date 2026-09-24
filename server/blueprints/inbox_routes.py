"""Start / stop scanning into the signed-in person's Drive inbox, and watch results.

The scan page calls these. Start opens a batch tying this person's "OCR Inbox"
folder to the school they picked; the watcher (a separate process) reads pages
that land there and logs results; the page polls for them; Stop closes the
batch. Nothing here touches the scanner - the person's scanner app saves
straight into the folder, which is what makes the same route work on Windows
and Mac.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request

from server.services import batch_store, drive_inbox
from server.services.auth import current_user, require_login
from server.services.debug_log import debug_log

inbox_bp = Blueprint("inbox", __name__)
inbox_bp.before_request(require_login)

# The watcher stamps a heartbeat every pass (every few seconds). Older than
# this and it is not running, and pages will sit unread - say so up front.
WATCHER_STALE_AFTER = timedelta(seconds=45)


def _me() -> str:
    return ((current_user() or {}).get("email") or "").strip().lower()


def _watcher_state() -> dict:
    seen = batch_store.watcher_last_seen()
    alive = bool(seen and datetime.now() - seen < WATCHER_STALE_AFTER)
    return {"alive": alive, "last_seen": seen.isoformat(sep=" ", timespec="seconds") if seen else None}


@inbox_bp.get("/api/inbox/status")
def inbox_status():
    """Everything the page needs to draw the panel in one call."""
    ok, why = drive_inbox.configured()
    if not ok:
        return jsonify({"configured": False, "reason": why}), 200

    email = _me()
    payload = {
        "configured": True,
        "service_account": drive_inbox.service_account_email(),
        "folder_name": drive_inbox.DRIVE_INBOX_FOLDER_NAME,
        "inbox": {"found": False},
        "batch": batch_store.open_batch_for(email),
        "watcher": _watcher_state(),
    }
    try:
        inbox = drive_inbox.find_inbox(email)
    except drive_inbox.DriveInboxError as exc:
        payload["inbox"] = {"found": False, "error": str(exc)}
        return jsonify(payload), 200
    if inbox:
        payload["inbox"] = {"found": True, "folder_id": inbox["id"], "shared_at": inbox["shared_at"]}
    return jsonify(payload), 200


@inbox_bp.post("/api/inbox/start")
def inbox_start():
    body = request.get_json(silent=True) or {}
    school = str(body.get("school") or "").strip()
    location = str(body.get("location") or "").strip()
    email = _me()

    try:
        inbox = drive_inbox.find_inbox(email)
    except drive_inbox.DriveInboxError as exc:
        return jsonify({"error": f"Could not reach Google Drive: {exc}"}), 502
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
    return jsonify({"batch": batch_store.get_batch(batch_id), "watcher": _watcher_state()}), 200


@inbox_bp.post("/api/inbox/stop")
def inbox_stop():
    batch = batch_store.open_batch_for(_me())
    if not batch:
        return jsonify({"closed": False, "batch": None}), 200
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
    }), 200
