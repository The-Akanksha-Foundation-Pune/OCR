"""Read-only reporting routes: scanned students + not-found scan attempts."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from server.services.auth import require_login
from server.services.history_store import delete_scan_attempt, get_not_found
from server.services.student_lookup import get_scanned_students

history_bp = Blueprint("history", __name__)
history_bp.before_request(require_login)

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100


def _pagination_args() -> tuple[int, int, str]:
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
    return page, page_size, search


@history_bp.get("/history")
def history_page():
    return render_template("history.html", active_page="history")


@history_bp.get("/api/scanned")
def scanned_api():
    page, page_size, search = _pagination_args()
    try:
        items, total = get_scanned_students(
            limit=page_size, offset=(page - 1) * page_size, search=search
        )
        return jsonify(
            {"items": items, "total": total, "page": page, "page_size": page_size}
        ), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not load scanned students: {exc}"}), 500


@history_bp.get("/api/not-found")
def not_found_api():
    page, page_size, search = _pagination_args()
    try:
        items, total = get_not_found(
            limit=page_size, offset=(page - 1) * page_size, search=search
        )
        return jsonify(
            {"items": items, "total": total, "page": page, "page_size": page_size}
        ), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not load not-found list: {exc}"}), 500


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
