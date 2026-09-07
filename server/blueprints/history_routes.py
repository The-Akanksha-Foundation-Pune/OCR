"""Read-only reporting routes: scanned students + not-found scan attempts."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from server.services.auth import current_user, require_login
from server.services.history_store import (
    delete_scan_attempt,
    get_attempt,
    get_not_found,
    mark_reviewed,
    resolve_attempt,
    unreviewed_student_ids,
)
from server.services.student_lookup import (
    find_student_by_id,
    get_scanned_students,
    list_schools_with_location,
)

history_bp = Blueprint("history", __name__)
history_bp.before_request(require_login)

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100


def _pagination_args() -> tuple[int, int, str, str, str]:
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    try:
        page_size = int(request.args.get("page_size", DEFAULT_PAGE_SIZE))
    except ValueError:
        page_size = DEFAULT_PAGE_SIZE
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    search = (request.args.get("search") or "").strip()
    school = (request.args.get("school") or "").strip()
    location = (request.args.get("location") or "").strip()
    return page, page_size, search, school, location


@history_bp.post("/api/scanned/review")
def review_scanned_api():
    """
    Sign off fuzzy-matched rows after checking them against the paper.

    Either a list of student_ids, or {"all": true} to cover everything
    currently unreviewed within the active filter - never the whole table.
    """
    payload = request.get_json(silent=True) or {}
    user = current_user() or {}
    who = (user.get("email") or "").strip()

    try:
        if payload.get("all"):
            school = (payload.get("school") or "").strip()
            location = (payload.get("location") or "").strip()
            ids = unreviewed_student_ids(school=school, location=location)
        else:
            ids = [str(x) for x in (payload.get("student_ids") or [])]

        if not ids:
            return jsonify({"reviewed": 0, "student_ids": []}), 200

        updated = mark_reviewed(ids, reviewed_by=who)
        return jsonify({"reviewed": updated, "student_ids": ids}), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not mark as checked: {exc}"}), 500


@history_bp.get("/api/history-schools")
def history_schools_api():
    """School list for the History page filters."""
    try:
        return jsonify({"schools": list_schools_with_location()}), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not load schools: {exc}"}), 500


@history_bp.get("/history")
def history_page():
    return render_template("history.html", active_page="history")


@history_bp.get("/api/scanned")
def scanned_api():
    page, page_size, search, school, location = _pagination_args()
    try:
        items, total = get_scanned_students(
            limit=page_size,
            offset=(page - 1) * page_size,
            search=search,
            school=school,
            location=location,
        )
        return jsonify(
            {"items": items, "total": total, "page": page, "page_size": page_size}
        ), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not load scanned students: {exc}"}), 500


@history_bp.get("/api/not-found")
def not_found_api():
    page, page_size, search, school, location = _pagination_args()
    try:
        items, total = get_not_found(
            limit=page_size,
            offset=(page - 1) * page_size,
            search=search,
            school=school,
            location=location,
        )
        return jsonify(
            {"items": items, "total": total, "page": page, "page_size": page_size}
        ), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not load not-found list: {exc}"}), 500


@history_bp.post("/api/not-found/<int:attempt_id>/recheck")
def recheck_not_found_api(attempt_id: int):
    """
    Re-run the lookup for a not-found entry with a corrected Student ID.

    The OCR reading is often close but not exact, so this is the path back
    for a form that failed: someone reads the ID off the paper, types it,
    and if it resolves the student is marked scanned and the entry moves
    out of the Not Found list.
    """
    payload = request.get_json(silent=True) or {}
    corrected = str(payload.get("student_id", "")).strip()
    if not corrected:
        return jsonify({"error": "Enter a Student ID to recheck."}), 400

    attempt = get_attempt(attempt_id)
    if attempt is None:
        return jsonify({"error": "Entry not found."}), 404
    if attempt.get("found_in_db"):
        return jsonify({"error": "That entry has already been resolved."}), 409

    try:
        student = find_student_by_id(corrected, mark_scanned=True)
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Lookup failed: {exc}"}), 500

    if student is None:
        return jsonify(
            {
                "found": False,
                "student_id": corrected,
                "message": "Still no match. Check the ID against the paper form.",
            }
        ), 200

    resolve_attempt(
        attempt_id,
        corrected_student_id=corrected,
        matched_student_id=student.get("student_id") or corrected,
        student_name=student.get("student_name"),
        school_name=student.get("school_name"),
        matched_via=student.get("matched_via"),
    )
    return jsonify({"found": True, "student": student}), 200


@history_bp.delete("/api/not-found/<int:attempt_id>")
def delete_not_found_api(attempt_id: int):
    """Remove one "Not found" log entry — this only touches ocr_scan_log,
    never active_student_data."""
    try:
        deleted = delete_scan_attempt(attempt_id)
        if not deleted:
            return jsonify({"error": "Entry not found."}), 404
        return jsonify({"ok": True}), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not delete entry: {exc}"}), 500
