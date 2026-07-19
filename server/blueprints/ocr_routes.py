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
    extract_student_id_from_upload,
    extract_student_ids_from_upload_batch,
    log_confirm_result,
)
from server.services.scan_store import page_image_jpeg, stash_upload

ocr_bp = Blueprint("ocr", __name__)
ocr_bp.before_request(require_login)


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@ocr_bp.get("/")
def index():
    return render_template("index.html", active_page="scanner")


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

    try:
        # 1) Search DB (and mark scanned_at when found)
        result = confirm_student_id(student_id, source=source, defer_log=True)
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
        log_confirm_result(result, image_link=image_link)
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
