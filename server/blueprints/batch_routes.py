"""The scan queue's API: send scans in, watch them read, confirm documents.

Scanning and OCR are separate. POST /api/scans only stores the file's pages
in the queue and answers 202 straight away; the OCR worker reads them in
the background; the page polls GET /api/batches to see each page read and
sorted into documents by its handwritten batch code. Nothing is written to
the database or Drive until a person presses Confirm on a document - the
only write path is POST /api/documents/<key>/confirm.

Every route is for the signed-in person's own pages only: another person's
page is answered exactly as one that does not exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request, send_file, url_for
from flask.typing import ResponseReturnValue

from server.blueprints.ocr_routes import (
    UNSUPPORTED_TYPE_MESSAGE,
    _is_allowed,
    _upload_filename,
)
from server.services import documents, scan_queue
from server.services.auth import current_user, require_login
from server.services.debug_log import debug_log
from server.services.extract_student_id import LookupUnavailable

batch_bp = Blueprint("batch", __name__)
batch_bp.before_request(require_login)

NOT_FOUND = "No such page."
# The detail of an unexpected failure (a database error names the host)
# stays in the server log. Where the document got to shows on the next poll.
UNEXPECTED = "Something went wrong. Wait for the list to refresh to see whether it was saved."


def _me() -> str:
    return scan_queue.normalise_owner((current_user() or {}).get("email"))


def _image_url(page_id: str, size: str) -> str:
    return url_for("batch.page_image", page_id=page_id, size=size)


def _error(message: str, status: int) -> ResponseReturnValue:
    return jsonify({"error": message}), status


def _json_body() -> dict[str, Any]:
    """The request's JSON object; a non-JSON body is refused with 415 before this returns."""
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _page_and_document(owner: str, page_id: str) -> dict[str, Any]:
    """A page and the document it now belongs to, after a correction."""
    snap = documents.snapshot(owner)
    page = next((p for p in snap.pages if p.page_id == page_id), None)
    doc = next(
        (d for d in snap.documents if any(p.page_id == page_id for p in d.pages)), None
    )
    return {
        "page": page.to_dict(_image_url) if page else None,
        "document": doc.to_dict(_image_url) if doc else None,
    }


@batch_bp.post("/api/scans")
def scans_api() -> ResponseReturnValue:
    """Store a scan's pages in the queue for OCR. Reads nothing: answers 202 at once."""
    if "file" not in request.files:
        return _error("Missing file field. Upload as multipart with key 'file'.", 400)
    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return _error("No file selected.", 400)

    filename = _upload_filename(uploaded.filename)
    if not _is_allowed(filename):
        return _error(UNSUPPORTED_TYPE_MESSAGE, 400)

    # The name as the person's computer had it, for the list of scans - not
    # secure_filename's version, which drops every non-ASCII character.
    source_name = Path(uploaded.filename.replace("\\", "/")).name[:200] or filename
    school = (request.form.get("school") or "").strip()
    location = (request.form.get("location") or "").strip()
    try:
        metas = scan_queue.intake(
            uploaded.read(),
            filename,
            source_name=source_name,
            owner_email=_me(),
            school=school,
            location=location,
        )
    except ValueError as exc:
        return _error(str(exc), 400)
    except OSError as exc:
        debug_log(f"[API] SCANS could not store {source_name!r}: {exc!r}")
        return _error("Could not store this scan on the server. Try again.", 500)

    return (
        jsonify(
            {
                "source_name": source_name,
                "page_count": len(metas),
                "pages": [
                    {"page_id": meta.page_id, "source_page": meta.source_page} for meta in metas
                ],
            }
        ),
        202,
    )


@batch_bp.get("/api/batches")
def batches_api() -> ResponseReturnValue:
    """The signed-in person's documents by batch, unsorted pages and the queue."""
    snap = documents.snapshot(_me())
    return (
        jsonify(
            {
                "batches": snap.batches(_image_url),
                "unsorted": [page.to_dict(_image_url) for page in snap.unsorted],
                "pages": [page.to_dict(_image_url) for page in snap.pages],
                "queue": snap.queue_counts(),
            }
        ),
        200,
    )


@batch_bp.get("/api/documents/<key>")
def document_api(key: str) -> ResponseReturnValue:
    """One of the signed-in person's documents, with its parts and readings."""
    doc = documents.get_document(_me(), key)
    if doc is None:
        return _error("No such document.", 404)
    return jsonify({"document": doc.to_dict(_image_url)}), 200


@batch_bp.get("/api/pages/<page_id>/image")
def page_image(page_id: str) -> ResponseReturnValue:
    """A page's scanned image (or ?size=thumb, ~400 px wide), for its owner only."""
    meta = scan_queue.load_meta(page_id) if scan_queue.is_page_id(page_id) else None
    if meta is None or meta.owner_email != _me():
        return _error(NOT_FOUND, 404)
    size = "thumb" if request.args.get("size") == "thumb" else "full"
    path = scan_queue.page_image_path(page_id, size)
    if path is None:
        return _error(NOT_FOUND, 404)
    # Revalidated every time: page.jpg is replaced when the worker finds a
    # page was scanned upside down, and the ETag changes with it.
    response = send_file(path, mimetype="image/jpeg", conditional=True, etag=True, max_age=0)
    response.cache_control.private = True
    response.cache_control.no_cache = True
    return response


@batch_bp.post("/api/pages/<page_id>/code")
def page_code_api(page_id: str) -> ResponseReturnValue:
    """Set (or with "" clear) a page's batch code by hand; it regroups at once."""
    payload = _json_body()
    owner = _me()
    try:
        documents.set_page_code(owner, page_id, str(payload.get("code") or ""), owner)
    except documents.DocumentError as exc:
        return _error(exc.message, exc.status_code)
    return jsonify(_page_and_document(owner, page_id)), 200


@batch_bp.post("/api/pages/<page_id>/student-id")
def page_student_id_api(page_id: str) -> ResponseReturnValue:
    """Set (or with "" clear) a page's Student ID by hand, and look it up read-only."""
    payload = _json_body()
    owner = _me()
    try:
        documents.set_page_student_id(owner, page_id, str(payload.get("student_id") or ""), owner)
    except documents.DocumentError as exc:
        return _error(exc.message, exc.status_code)
    except LookupUnavailable as exc:
        return _error(
            "The student database could not be reached, so the ID was not kept. Try again.",
            exc.status_code,
        )
    return jsonify(_page_and_document(owner, page_id)), 200


@batch_bp.post("/api/documents/<key>/confirm")
def confirm_document_api(key: str) -> ResponseReturnValue:
    """Confirm a document: mark the student scanned, upload the PDF, log it.

    The body is {student_id, page_ids}: page_ids are the parts the page was
    showing, and only exactly those are confirmed. The only path from the
    scan queue to the database and Drive. 409 with nothing saved when the
    document has gained or lost a part since it was shown, when the ID is
    not a student at the school, when the person signed in through Tether
    (no Google token for Drive), or when the document is still being read;
    400 with nothing saved when page_ids is missing; 503 with nothing saved
    when the database cannot be reached. A Drive failure after the student
    is marked answers 200 with uploaded false and the document left
    "confirmed"; so does a History row that could not be written after the
    upload worked (uploaded true, finished false).
    """
    payload = _json_body()
    owner = _me()
    try:
        outcome = documents.confirm_document(
            owner, key, str(payload.get("student_id") or ""), owner, payload.get("page_ids")
        )
    except documents.DocumentError as exc:
        return _error(exc.message, exc.status_code)
    except LookupUnavailable as exc:
        return _error(exc.message, exc.status_code)
    except Exception as exc:  # noqa: BLE001 - answer JSON, keep the detail in the log
        debug_log(f"[API] DOCUMENT CONFIRM key={key!r} failed: {exc!r}")
        return _error(UNEXPECTED, 500)
    debug_log(
        f"[API] DOCUMENT CONFIRM key={key!r} as={outcome.document.key!r} "
        f"student_id={outcome.document.student_id!r} uploaded={outcome.uploaded}"
    )
    return jsonify(_outcome_json(outcome)), 200


@batch_bp.post("/api/documents/<key>/upload")
def retry_upload_api(key: str) -> ResponseReturnValue:
    """Finish a confirmed document: retry its Drive upload, or only its History row.

    The body is not read. Answers 502 when it is still not finished: unlike
    a confirm, a retry did nothing else, so a failure is its whole answer.
    The document stays confirmed and can be retried once more.
    """
    try:
        outcome = documents.retry_upload(_me(), key)
    except documents.DocumentError as exc:
        return _error(exc.message, exc.status_code)
    except Exception as exc:  # noqa: BLE001 - answer JSON, keep the detail in the log
        debug_log(f"[API] DOCUMENT UPLOAD key={key!r} failed: {exc!r}")
        return _error(UNEXPECTED, 500)
    body = _outcome_json(outcome)
    if not outcome.finished:
        record = outcome.document.record or {}
        if outcome.uploaded:
            failure = record.get("log_error") or "Could not write the History entry yet."
            body["error"] = f"{failure} The student is saved and the PDF is in Drive; try again."
        else:
            failure = record.get("upload_error") or "The upload failed."
            body["error"] = f"{failure} The student is saved; try the upload again."
        return jsonify(body), 502
    return jsonify(body), 200


def _outcome_json(outcome: documents.ConfirmOutcome) -> dict[str, Any]:
    doc = outcome.document.to_dict(_image_url)
    return {
        "document": doc,
        "uploaded": outcome.uploaded,
        "finished": outcome.finished,
        "drive_link": doc["drive_link"],
        "message": outcome.message,
    }
