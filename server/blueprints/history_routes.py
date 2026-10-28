"""Read-only reporting routes: scanned students + not-found scan attempts."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, jsonify, render_template, request

from server.services.auth import current_user, require_login
from server.services.clock import now_ist
from server.services.debug_log import debug_log
from server.services.history_store import (
    EXACT_MATCH_WAYS,
    delete_scan_attempt,
    get_attempt,
    get_not_found,
    mark_reviewed,
    pending_review,
    resolve_attempt,
    review_token,
    schools_holding_student_id,
)
from server.services.student_lookup import (
    find_student_by_id,
    get_scanned_students,
    list_schools_with_location,
    resolve_scope,
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


def _payload_text(payload: dict[str, Any], key: str) -> str:
    return str(payload.get(key) or "").strip()


@history_bp.post("/api/scanned/review")
def review_scanned_api() -> tuple[Response, int]:
    """
    Sign off fuzzy-matched rows after checking them against the paper.

    Either a list of student_ids, or {"all": true} with the Scanned list's
    own school, location and search, plus "expected" and "token": the count
    and the review_token that list was loaded with. "All" is every row
    needing a check in exactly that list, across its pages - never rows it
    does not show, and only for one school or one city.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = {}
    user = current_user() or {}
    who = (user.get("email") or "").strip()

    try:
        through_id: int | None = None
        if payload.get("all"):
            school = _payload_text(payload, "school")
            location = _payload_text(payload, "location")
            search = _payload_text(payload, "search")
            expected = payload.get("expected")
            token = payload.get("token")
            # No confirmed count and token means a copy of the page from
            # before these checks, still open in a tab: it sends no search
            # either, so its "all" would sign off rows the operator never saw.
            if (
                not isinstance(expected, int)
                or isinstance(expected, bool)
                or not isinstance(token, str)
                or not token
            ):
                return jsonify(
                    {"error": "Reload the page, then use Mark all checked again."}
                ), 400
            # Across every school the list runs to pages nobody opens, and a
            # sign-off says a person held each of those forms. One school or
            # one city keeps "all" to a list someone can actually go through.
            if not school and not location:
                return jsonify(
                    {
                        "error": (
                            "Choose a school or a city first. Mark all checked "
                            "covers one school's or one city's list, not every "
                            "school at once."
                        )
                    }
                ), 400
            pending = pending_review(search=search, school=school, location=location)
            # Forms keep arriving while the page is open, and others sign
            # rows off. If the rows have moved since the operator was shown
            # them - even to the same count - signing them off would cover
            # rows nobody looked at, so mark nothing and send the new count
            # back for them to look again.
            if len(pending) != expected or review_token(pending) != token:
                return jsonify(
                    {
                        "error": (
                            f"The list has changed: {len(pending)} row(s) now need "
                            "checking. Look again before marking them all."
                        ),
                        "needs_review": len(pending),
                    }
                ), 409
            ids = [student_id for student_id, _last_id in pending]
            through_id = max((last_id for _sid, last_id in pending), default=None)
        else:
            ids = [str(x) for x in (payload.get("student_ids") or [])]

        if not ids:
            return jsonify({"reviewed": 0, "student_ids": []}), 200

        updated = mark_reviewed(ids, reviewed_by=who, through_id=through_id)
        return jsonify({"reviewed": updated, "student_ids": ids}), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        debug_log(f"[HISTORY] mark checked failed: {exc!r}")
        return jsonify({"error": "Could not mark as checked. Try again."}), 500


@history_bp.get("/api/history-schools")
def history_schools_api() -> tuple[Response, int]:
    """School list for the History page filters."""
    try:
        return jsonify({"schools": list_schools_with_location()}), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        debug_log(f"[HISTORY] school list failed: {exc!r}")
        return jsonify({"error": "Could not load schools."}), 500


@history_bp.get("/history")
def history_page():
    return render_template("history.html", active_page="history")


@history_bp.get("/api/scanned")
def scanned_api() -> tuple[Response, int]:
    page, page_size, search, school, location = _pagination_args()
    try:
        items, total = get_scanned_students(
            limit=page_size,
            offset=(page - 1) * page_size,
            search=search,
            school=school,
            location=location,
        )
    except Exception as exc:  # noqa: BLE001 — return safe API error
        debug_log(f"[HISTORY] scanned list failed: {exc!r}")
        return jsonify({"error": "Could not load scanned students."}), 500

    # How many rows of this whole filtered list - every page, not just
    # this one - still need checking, and a token for exactly which: the
    # review bar shows the count and "Mark all checked" sends both back.
    # Counted apart from the list so a failure here hides the bar (null)
    # instead of the list someone is reading. review=0 is the page's tab
    # count preload, which wants only the total and not a scan of the log.
    needs_review: int | None = None
    token: str | None = None
    if request.args.get("review") != "0":
        try:
            pending = pending_review(search=search, school=school, location=location)
            needs_review = len(pending)
            token = review_token(pending)
        except Exception as exc:  # noqa: BLE001 — the list still stands without it
            debug_log(f"[HISTORY] review count failed: {exc!r}")

    return jsonify(
        {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "needs_review": needs_review,
            "review_token": token,
        }
    ), 200


@history_bp.get("/api/not-found")
def not_found_api() -> tuple[Response, int]:
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
        debug_log(f"[HISTORY] not-found list failed: {exc!r}")
        return jsonify({"error": "Could not load the not-found list."}), 500


@history_bp.post("/api/not-found/<int:attempt_id>/recheck")
def recheck_not_found_api(attempt_id: int) -> tuple[Response, int]:
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

    # Search where the form was scanned, as the scan itself did: the school,
    # or failing that its city. Unscoped, a typed ID - or a one-character
    # typo in it - could be fuzzy-matched to a near-twin in another school,
    # and that child would be marked as having handed in this form.
    school = str(attempt.get("scan_school") or "").strip()
    location = str(attempt.get("scan_location") or "").strip()
    try:
        # Empty when the row names no school or city (set-aside pages, rows
        # from before those columns, All/All scans) or a city with no
        # schools on file - and empty means the lookup searches everywhere.
        scope = resolve_scope(school or None, location or None)
        # Read-only: the child is marked scanned below, and only once this
        # entry has been claimed.
        student = find_student_by_id(
            corrected,
            mark_scanned=False,
            school=school or None,
            location=location or None,
        )
        # One school holds at most one child per ID; anything wider can
        # hold two, and the lookup would return whichever came first.
        holders = (
            schools_holding_student_id(str(student.get("student_id") or corrected))
            if student is not None and len(scope) != 1
            else []
        )
    except Exception as exc:  # noqa: BLE001 — return safe API error
        debug_log(f"[HISTORY] recheck lookup failed for {attempt_id}: {exc!r}")
        return jsonify(
            {"error": "Could not check the student database. Try again."}
        ), 500

    if student is None:
        where = school or (f"{location} schools" if location else "")
        return jsonify(
            {
                "found": False,
                "student_id": corrected,
                "message": (
                    f"Still no match in {where}. " if where else "Still no match. "
                )
                + "Check the ID against the paper form.",
            }
        ), 200

    matched_id = str(student.get("student_id") or corrected)
    via = str(student.get("matched_via") or "").lower()
    # With no school to search, a fuzzy repair can land on a near-twin
    # anywhere in the city or the country. Only an ID typed exactly as it
    # is on file is safe to take.
    if not scope and via not in EXACT_MATCH_WAYS:
        debug_log(
            f"[HISTORY] recheck {attempt_id}: unscoped {via} match to "
            f"{matched_id!r} not taken"
        )
        return jsonify(
            {
                "found": False,
                "student_id": corrected,
                "message": (
                    f"No exact match for {corrected}. This scan has no school on "
                    "record to search, so only an exact Student ID is accepted - "
                    "check it against the paper form."
                ),
            }
        ), 200

    in_scope = {name.upper() for name in scope}
    sharing = [name for name in holders if not in_scope or name.upper() in in_scope]
    if len(sharing) > 1:
        return jsonify(
            {
                "found": False,
                "student_id": corrected,
                "message": (
                    f"{matched_id} belongs to more than one child "
                    f"({', '.join(sharing)}). Nothing was saved - delete this "
                    "entry and scan the form again with its school chosen."
                ),
            }
        ), 200

    scanned_at = now_ist()
    try:
        claimed = resolve_attempt(
            attempt_id,
            corrected_student_id=corrected,
            matched_student_id=matched_id,
            student_name=student.get("student_name"),
            school_name=student.get("school_name"),
            matched_via=student.get("matched_via"),
            scanned_at=scanned_at,
        )
    except Exception as exc:  # noqa: BLE001 — return safe API error
        debug_log(f"[HISTORY] recheck save failed for {attempt_id}: {exc!r}")
        return jsonify({"error": "Could not save the match. Try again."}), 500

    if not claimed:
        # Deleted, or rechecked, in another tab while this one looked the
        # ID up. Nobody was marked scanned by this request.
        return jsonify(
            {
                "error": (
                    "That entry was rechecked or deleted somewhere else while "
                    "this one was checking. Nothing was saved - reload the list."
                )
            }
        ), 409

    student["scanned_at"] = scanned_at.isoformat(sep=" ", timespec="seconds")
    return jsonify({"found": True, "student": student}), 200


@history_bp.delete("/api/not-found/<int:attempt_id>")
def delete_not_found_api(attempt_id: int) -> tuple[Response, int]:
    """Remove one "Not found" log entry — this only touches ocr_scan_log,
    never active_student_data, and never a row that has been resolved."""
    try:
        deleted = delete_scan_attempt(attempt_id)
        if not deleted:
            return jsonify(
                {
                    "error": (
                        "That entry is no longer in the Not found list - it may "
                        "have been rechecked or deleted already."
                    )
                }
            ), 404
        return jsonify({"ok": True}), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        debug_log(f"[HISTORY] delete of attempt {attempt_id} failed: {exc!r}")
        return jsonify({"error": "Could not delete the entry. Try again."}), 500
