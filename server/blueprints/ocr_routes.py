"""OCR upload routes and simple web UI.

The scan page no longer scans through this module: it sends each file to
POST /api/scans (blueprints/batch_routes.py), which only queues it, and each
document is confirmed there - by a person, or for an exact match by the page
itself - before anything is saved. What stays
here is the page itself, the school list, the one-ID /api/confirm the
page's "type a Student ID" box uses, and /api/extract and /api/extract-batch,
kept for scripts and older pages that still call them. None of these is an
automatic save: /api/confirm writes only the ID a person typed and sent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request
from flask.typing import ResponseReturnValue
from werkzeug.utils import secure_filename

from server.config import (
    ALLOWED_EXTENSIONS,
    HELD_STACK_HOURS,
    STACK_SETTLE_MINUTES,
    auto_upload_on,
    batch_codes_on,
)
from server.services.auth import require_login
from server.services.debug_log import debug_log
from server.services.drive_store import upload_scan_to_drive
from server.services.extract_student_id import (
    ExtractError,
    confirm_student_id,
    extract_student_id_from_upload,
    extract_student_ids_from_upload_batch,
    log_confirm_result,
)
from server.services.scan_store import page_image_jpeg, stash_upload
from server.services.student_lookup import (
    list_schools_with_location,
)

ocr_bp = Blueprint("ocr", __name__)
ocr_bp.before_request(require_login)

UNSUPPORTED_TYPE_MESSAGE = (
    "Unsupported file type. Use JPG, PNG, BMP, TIFF, WebP or PDF - the scanner's "
    "software can save any of these."
)


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _upload_filename(original: str) -> str:
    """A safe name to store an upload under, keeping its extension.

    secure_filename drops every non-ASCII character, so a scan saved as
    "प्रवेश_फॉर्म.pdf" comes out as plain "pdf" - no suffix, and a good file
    refused as an unsupported type. When that happens the original's own
    extension is kept, on a stand-in name.
    """
    filename = secure_filename(original)
    if not Path(filename).suffix:
        suffix = Path(original).suffix.lower()
        if suffix in ALLOWED_EXTENSIONS:
            return f"upload{suffix}"
    return filename


def _upload_error(message: str, status: int, scan_id: str | None) -> ResponseReturnValue:
    """An error answer for an upload, still carrying its scan_id once stored.

    The image is stashed before OCR runs, so a page OCR cannot read still
    has one. The page keeps the scan_id on that row so the ID someone types
    in from the paper is saved with this image; without it the student was
    marked scanned with no copy of the form anywhere.
    """
    body: dict[str, Any] = {"error": message}
    if scan_id:
        body["scan_id"] = scan_id
        body["page_number"] = 1
    return jsonify(body), status


@ocr_bp.get("/")
def index() -> str:
    # The scanner's own software does the scanning; pages reach the server
    # only as files the browser uploads (from the watched scan folder, or
    # added by hand) or through the Drive inbox. So nothing here depends on
    # the OS of the server or of the computer the scanner is plugged into.
    # batch_codes: whether the page asks for batch codes and sorts by them.
    # auto_upload: whether the page offers Upload matched pages automatically;
    # stack_settle_minutes, how long a stack waits before it goes up by itself,
    # and held_stack_hours, how long one with a page out of place keeps the
    # next ones for its school from going up by themselves.
    return render_template(
        "index.html",
        active_page="scanner",
        batch_codes=batch_codes_on(),
        auto_upload=auto_upload_on(),
        stack_settle_minutes=STACK_SETTLE_MINUTES,
        held_stack_hours=HELD_STACK_HOURS,
    )


@ocr_bp.get("/api/schools")
def schools_api() -> ResponseReturnValue:
    """Schools to choose from before scanning a stack, grouped by city."""
    try:
        return jsonify({"schools": list_schools_with_location()}), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        # The detail (a pymysql error names the database host) stays in the
        # server log; the browser only needs to know it did not work.
        debug_log(f"[API] SCHOOLS failed: {exc!r}")
        return jsonify({"error": "Could not load schools. Reload the page to try again."}), 500


@ocr_bp.post("/api/extract")
def extract_api() -> ResponseReturnValue:
    """Read one page at once and answer with the reading (legacy; see the module docstring)."""
    if "file" not in request.files:
        return jsonify({"error": "Missing file field. Upload as multipart with key 'file'."}), 400

    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = _upload_filename(uploaded.filename)
    if not _is_allowed(filename):
        return jsonify({"error": UNSUPPORTED_TYPE_MESSAGE}), 400

    scan_id: str | None = None
    try:
        data = uploaded.read()
        scan_id = stash_upload(data, filename)
        result = extract_student_id_from_upload(data, filename)
        payload = result.to_dict()
        payload["scan_id"] = scan_id
        payload["page_number"] = payload.get("page_number") or 1
        debug_log(
            f"[API] PREVIEW file={filename!r} student_id={payload.get('student_id')!r} "
            f"scan_id={scan_id!r} found_in_db={payload.get('found_in_db')} "
            f"message={payload.get('message')!r}"
        )
        return jsonify(payload), 200
    except ExtractError as exc:
        return _upload_error(exc.message, exc.status_code, scan_id)
    except ValueError as exc:
        return _upload_error(str(exc), 400, scan_id)
    except Exception as exc:  # noqa: BLE001 — return safe API error
        debug_log(f"[API] PREVIEW file={filename!r} scan_id={scan_id!r} failed: {exc!r}")
        return _upload_error("Could not read this page. Try again.", 500, scan_id)


@ocr_bp.post("/api/extract-batch")
def extract_batch_api() -> ResponseReturnValue:
    """
    Legacy: the scan page now sends files to /api/scans instead.

    Same OCR-only preview as /api/extract, but reads EVERY page — a
    printer/scanner machine often batch-scans a whole stack of forms into
    one multi-page PDF or TIFF. Returns one preview per page for the UI to
    walk through and confirm one by one. Single-page uploads (a JPG or PNG)
    just come back as a 1-page batch.
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing file field. Upload as multipart with key 'file'."}), 400

    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = _upload_filename(uploaded.filename)
    if not _is_allowed(filename):
        return jsonify({"error": UNSUPPORTED_TYPE_MESSAGE}), 400

    scan_id: str | None = None
    try:
        data = uploaded.read()
        scan_id = stash_upload(data, filename)
        results = extract_student_ids_from_upload_batch(data, filename)
        pages = []
        for result in results:
            page = result.to_dict()
            page["scan_id"] = scan_id
            pages.append(page)
        payload = {"filename": filename, "page_count": len(pages), "pages": pages, "scan_id": scan_id}
        debug_log(
            f"[API] BATCH PREVIEW file={filename!r} page_count={len(pages)} "
            f"scan_id={scan_id!r} student_ids={[p.get('student_id') for p in pages]!r}"
        )
        return jsonify(payload), 200
    except ExtractError as exc:
        return _upload_error(exc.message, exc.status_code, scan_id)
    except ValueError as exc:
        return _upload_error(str(exc), 400, scan_id)
    except Exception as exc:  # noqa: BLE001 — return safe API error
        debug_log(f"[API] BATCH PREVIEW file={filename!r} scan_id={scan_id!r} failed: {exc!r}")
        return _upload_error("Could not read this file. Try again.", 500, scan_id)


@ocr_bp.post("/api/confirm")
def confirm_api() -> ResponseReturnValue:
    """
    Final step: user confirms Student ID → DB lookup first → upload scan
    to Drive (found or not) → log DrivelinkImage.
    """
    payload = request.get_json(silent=True) or {}
    student_id = str(payload.get("student_id", "")).strip()
    if not student_id:
        return jsonify({"error": "Student ID cannot be empty."}), 400

    source = str(payload.get("source") or "manual-confirm").strip() or "manual-confirm"
    scan_id = str(payload.get("scan_id") or "").strip()
    page_number = int(payload.get("page_number") or 1)
    # Which school's forms are being scanned. Narrowing the candidate list
    # is what makes unattended saving safe enough to do: it removes every
    # cross-school near-twin, and it disambiguates the Student IDs that are
    # duplicated across two schools.
    school = str(payload.get("school") or "").strip()
    # Location is the fallback scope: with the school left on "All Pune
    # schools" it still keeps the search inside Pune.
    location = str(payload.get("location") or "").strip()

    try:
        # 1) Search DB (and mark scanned_at when found)
        result = confirm_student_id(
            student_id,
            source=source,
            defer_log=True,
            school=school or None,
            location=location or None,
        )
        data = result.to_dict()
        image_link: str | None = None

        # 2) Always save the scan image to Drive when we have the file
        #    (found → StudentName_ID; not found → NOTFOUND_ID)
        if scan_id:
            student = result.student or {}
            if result.found_in_db:
                student_name = (student.get("student_name") or "").strip() or result.student_id
                display_name = f"{student_name}_{result.student_id}"
            else:
                display_name = f"NOTFOUND_{result.student_id}"

            page_bytes = page_image_jpeg(scan_id, page_number)
            if page_bytes:
                jpeg_data, _ext = page_bytes
                drive_meta = upload_scan_to_drive(
                    jpeg_data,
                    f"{display_name}.jpg",
                    display_name=display_name,
                )
                if drive_meta:
                    image_link = drive_meta.get("webViewLink") or None
                    data["drive"] = {
                        "id": drive_meta.get("id"),
                        "name": drive_meta.get("name"),
                        "webViewLink": image_link,
                    }

        # 3) Log attempt with Drive link in DrivelinkImage
        log_confirm_result(
            result,
            image_link=image_link,
            scan_school=school or None,
            scan_location=location or None,
        )
        data["image_link"] = image_link
        data["DrivelinkImage"] = image_link

        debug_log(
            f"[API] CONFIRM student_id={data.get('student_id')!r} "
            f"found_in_db={data.get('found_in_db')} DrivelinkImage={image_link!r} "
            f"message={data.get('message')!r}"
        )
        return jsonify(data), 200
    except ExtractError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception as exc:  # noqa: BLE001 — return safe API error
        # Logged, not sent: database errors carry the host, user and address.
        debug_log(f"[API] CONFIRM student_id={student_id!r} failed: {exc!r}")
        return jsonify({"error": "Lookup failed. Try again."}), 500
