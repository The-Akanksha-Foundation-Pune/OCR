"""OCR upload routes and simple web UI."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, render_template, request
from werkzeug.utils import secure_filename

from server.config import ALLOWED_EXTENSIONS
from server.services.auth import require_login
from server.services.debug_log import debug_log
from server.services.drive_store import upload_scan_to_drive
from server.services.extract_student_id import (
    ExtractError,
    confirm_student_id,
    extract_one_image,
    extract_one_page_from_stash,
    extract_student_id_from_upload,
    extract_student_ids_from_upload_batch,
    log_confirm_result,
)
from server.services.preprocess import count_pages_from_bytes
from server.services.scan_store import page_image_jpeg, stash_upload
from server.services.scan_job import (
    discard_job,
    get_job,
    job_status,
    page_paths,
    start_scan,
)
from server.services.scanner import ScannerError, scan_to_pdf
from server.services.student_lookup import (
    find_student_by_id,
    list_schools_with_location,
)

ocr_bp = Blueprint("ocr", __name__)
ocr_bp.before_request(require_login)


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@ocr_bp.get("/")
def index():
    return render_template("index.html", active_page="scanner")


@ocr_bp.get("/api/schools")
def schools_api():
    """Schools to choose from before scanning a stack, grouped by city."""
    try:
        return jsonify({"schools": list_schools_with_location()}), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not load schools: {exc}"}), 500


@ocr_bp.post("/api/extract")
def extract_api():
    if "file" not in request.files:
        return jsonify({"error": "Missing file field. Upload as multipart with key 'file'."}), 400

    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(uploaded.filename)
    if not _is_allowed(filename):
        return jsonify(
            {
                "error": "Unsupported file type. Use JPG, PNG, or PDF from CamScanner or printer scanner."
            }
        ), 400

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
        return jsonify({"error": exc.message}), exc.status_code
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Extraction failed: {exc}"}), 500


@ocr_bp.post("/api/extract-batch")
def extract_batch_api():
    """
    Same OCR-only preview as /api/extract, but reads EVERY page — a
    printer/scanner machine often batch-scans a whole stack of forms into
    one multi-page PDF. Returns one preview per page for the UI to walk
    through and confirm one by one. Single-page uploads (incl. phone
    photos) just come back as a 1-page batch.
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing file field. Upload as multipart with key 'file'."}), 400

    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(uploaded.filename)
    if not _is_allowed(filename):
        return jsonify(
            {
                "error": "Unsupported file type. Use JPG, PNG, or PDF from CamScanner or printer scanner."
            }
        ), 400

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
        return jsonify({"error": exc.message}), exc.status_code
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Batch extraction failed: {exc}"}), 500


@ocr_bp.post("/api/scan/start")
def scan_start_api():
    """
    Begin a scan and return at once.

    Pages are read as they land rather than after the whole stack has been
    fed - feeding dominates the wall clock, so overlapping the two is where
    the time is, and results start appearing within seconds.
    """
    try:
        job = start_scan()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not start the scanner: {exc}"}), 502
    debug_log(f"[API] SCAN START job={job.job_id}")
    return jsonify({"job_id": job.job_id}), 200


@ocr_bp.get("/api/scan/status")
def scan_status_api():
    """How many pages are captured so far, and whether the scanner stopped."""
    job = get_job((request.args.get("job") or "").strip())
    if job is None:
        return jsonify({"error": "That scan is no longer available."}), 404
    return jsonify(job_status(job)), 200


@ocr_bp.post("/api/scan/read")
def scan_read_api():
    """OCR one captured page of a running scan."""
    payload = request.get_json(silent=True) or {}
    job = get_job(str(payload.get("job_id") or "").strip())
    if job is None:
        return jsonify({"error": "That scan is no longer available."}), 404

    try:
        page_number = int(payload.get("page_number") or 1)
    except (TypeError, ValueError):
        return jsonify({"error": "page_number must be a number."}), 400

    pages = page_paths(job)
    if page_number < 1 or page_number > len(pages):
        return jsonify({"error": f"Page {page_number} has not been scanned yet."}), 409

    path = pages[page_number - 1]
    try:
        # Stash the image so /api/confirm can still upload it to Drive.
        scan_id = stash_upload(path.read_bytes(), f"page-{page_number:03d}.jpg")
        result = extract_one_image(path, page_number, len(pages))
        page = result.to_dict()
        page["scan_id"] = scan_id
        debug_log(
            f"[API] SCAN READ job={job.job_id} page={page_number} "
            f"student_id={page.get('student_id')!r}"
        )
        return jsonify(page), 200
    except ExtractError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not read page {page_number}: {exc}"}), 500


@ocr_bp.post("/api/scan/finish")
def scan_finish_api():
    """Drop a finished scan's images."""
    payload = request.get_json(silent=True) or {}
    discard_job(str(payload.get("job_id") or "").strip())
    return jsonify({"ok": True}), 200


@ocr_bp.post("/api/scan")
def scan_api():
    """
    Scan straight from the attached scanner - no file to save and upload.

    Returns the same shape as /api/extract-batch so the page handles a
    scanned stack and an uploaded PDF through exactly the same path.
    """
    try:
        data, filename = scan_to_pdf()
    except ScannerError as exc:
        return jsonify({"error": exc.message}), exc.status_code

    try:
        scan_id = stash_upload(data, filename)
        page_count = count_pages_from_bytes(data, filename)
        debug_log(
            f"[API] SCAN captured scan_id={scan_id!r} page_count={page_count}"
        )
        # Deliberately no OCR here. Reading a stack takes minutes, and a
        # request that long gets dropped by the browser - the pages are
        # read one at a time through /api/read-page instead.
        return jsonify(
            {"filename": filename, "page_count": page_count, "scan_id": scan_id}
        ), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Scan processing failed: {exc}"}), 500


@ocr_bp.post("/api/read-page")
def read_page_api():
    """OCR one page of a stashed scan. One short request per page."""
    payload = request.get_json(silent=True) or {}
    scan_id = str(payload.get("scan_id") or "").strip()
    try:
        page_number = int(payload.get("page_number") or 1)
    except (TypeError, ValueError):
        return jsonify({"error": "page_number must be a number."}), 400

    if not scan_id:
        return jsonify({"error": "scan_id is required."}), 400

    try:
        result = extract_one_page_from_stash(scan_id, page_number)
        page = result.to_dict()
        page["scan_id"] = scan_id
        debug_log(
            f"[API] READ-PAGE scan_id={scan_id!r} page={page_number} "
            f"student_id={page.get('student_id')!r}"
        )
        return jsonify(page), 200
    except ExtractError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not read page {page_number}: {exc}"}), 500


@ocr_bp.post("/api/confirm")
def confirm_api():
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
        return jsonify({"error": f"Lookup failed: {exc}"}), 500
