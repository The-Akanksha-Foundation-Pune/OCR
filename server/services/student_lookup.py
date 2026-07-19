"""Look up OCR Student ID in AFDW active_student_data."""

from __future__ import annotations

import re
import time
from typing import Any

import pymysql
from rapidfuzz import fuzz, process

from server.config import (
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
    MARK_SCANNED_ON_LOOKUP,
    STUDENT_TABLE,
)
from server.services.debug_log import debug_log

# Cache of every active student_id (small table — a few thousand rows) so
# fuzzy repair can score against the WHOLE list, not just rows that happen
# to share an (error-prone) OCR prefix. Refreshed every few minutes.
_ID_CACHE_TTL_SECONDS = 300
_id_cache: list[str] = []
_id_cache_at: float = 0.0

# Common handwriting / OCR confusions in digit sections
_OCR_VARIANTS = str.maketrans(
    {
        "O": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
        "G": "6",
    }
)


def get_connection():
    if not DB_HOST or not DB_USER or not DB_NAME:
        raise RuntimeError("Database is not configured. Check .env DB_* values.")

    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        port=DB_PORT,
        connect_timeout=20,
        cursorclass=pymysql.cursors.DictCursor,
    )


def find_student_by_id(
    student_id: str, mark_scanned: bool | None = None
) -> dict[str, Any] | None:
    """
    Find an active student by Student ID.

    1) Exact match
    2) Safe same-shape OCR fixes (O->0 in digit section)
    3) Constrained repair: noisy OCR (AADMES3OZY) may map to a DB ID
       only when one candidate clearly wins (e.g. AADMIS310714)

    mark_scanned overrides MARK_SCANNED_ON_LOOKUP for this call — pass
    False for a read-only preview (e.g. before the user confirms the OCR
    result), and True once the user has confirmed it's correct.
    """
    should_mark = MARK_SCANNED_ON_LOOKUP if mark_scanned is None else mark_scanned
    cleaned = re.sub(r"[^A-Za-z0-9]", "", (student_id or "")).upper().strip()
    if not cleaned:
        debug_log("[DB] empty student_id — skip lookup")
        return None

    debug_log("=" * 50)
    debug_log(f"[DB] VALUE SENT TO SQL = {cleaned!r}")
    debug_log(f"[DB] table={STUDENT_TABLE} column=student_id")
    debug_log("=" * 50)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            row = _exact_lookup(cur, cleaned)
            matched_via = "exact"

            if row is None:
                for variant in _safe_ocr_variants(cleaned):
                    debug_log(f"[DB] trying safe OCR variant {variant!r}")
                    row = _exact_lookup(cur, variant)
                    if row is not None:
                        matched_via = "safe_variant"
                        cleaned = variant
                        break

            if row is None:
                row = _constrained_repair_lookup(cur, cleaned)
                if row is not None:
                    matched_via = "constrained_repair"

            if row is None:
                debug_log(f"[DB] RESULT: NOT FOUND for {cleaned!r}")
                return None

            matched_id = row.get("student_id") or cleaned
            debug_log(
                f"[DB] RESULT: FOUND via={matched_via} student_id={matched_id!r} "
                f"name={row.get('student_name')!r} (ocr_input={student_id!r})"
            )

            if should_mark:
                cur.execute(
                    f"""
                    UPDATE {STUDENT_TABLE}
                    SET scanned_at = NOW()
                    WHERE UPPER(TRIM(student_id)) = UPPER(%s)
                    """,
                    (matched_id,),
                )
                conn.commit()
                debug_log(f"[DB] scanned_at updated for {matched_id!r}")
                cur.execute(
                    f"""
                    SELECT scanned_at
                    FROM {STUDENT_TABLE}
                    WHERE UPPER(TRIM(student_id)) = UPPER(%s)
                    LIMIT 1
                    """,
                    (matched_id,),
                )
                updated = cur.fetchone()
                if updated:
                    row["scanned_at"] = updated["scanned_at"]

            return _serialize_student(row)
    finally:
        conn.close()


def get_scanned_students(
    limit: int = 25, offset: int = 0, search: str = ""
) -> tuple[list[dict[str, Any]], int]:
    """Students already marked scanned_at, newest first (for the History
    page's Scanned tab). Reads straight from active_student_data — no
    schema changes, no separate log needed since scanned_at is already
    the source of truth."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            where = "WHERE scanned_at IS NOT NULL"
            params: list[Any] = []
            if search:
                where += " AND (UPPER(student_id) LIKE %s OR UPPER(student_name) LIKE %s)"
                like = f"%{search.upper()}%"
                params.extend([like, like])

            cur.execute(f"SELECT COUNT(*) AS total FROM {STUDENT_TABLE} {where}", params)
            total = (cur.fetchone() or {}).get("total", 0)

            cur.execute(
                f"""
                SELECT
                    student_id, student_name, school_name, grade_name,
                    division_name, academic_year, gender, status, scanned_at
                FROM {STUDENT_TABLE}
                {where}
                ORDER BY scanned_at DESC
                LIMIT %s OFFSET %s
                """,
                (*params, limit, offset),
            )
            rows = cur.fetchall() or []
            return [_serialize_student(row) for row in rows], total
    finally:
        conn.close()


def _exact_lookup(cur, student_id: str) -> dict[str, Any] | None:
    cur.execute(
        f"""
        SELECT
            student_id,
            student_name,
            school_name,
            grade_name,
            division_name,
            academic_year,
            gender,
            status,
            scanned_at
        FROM {STUDENT_TABLE}
        WHERE UPPER(TRIM(student_id)) = UPPER(%s)
        LIMIT 1
        """,
        (student_id,),
    )
    return cur.fetchone()


def _get_all_student_ids(cur) -> list[str]:
    """Cached list of every active Student ID (letters+digits shape only)."""
    global _id_cache, _id_cache_at
    now = time.monotonic()
    if _id_cache and (now - _id_cache_at) < _ID_CACHE_TTL_SECONDS:
        return _id_cache

    cur.execute(f"SELECT student_id FROM {STUDENT_TABLE}")
    rows = cur.fetchall() or []
    ids = []
    for row in rows:
        student_id = (row.get("student_id") or "").upper().strip()
        if re.fullmatch(r"[A-Z]{3,12}\d{4,12}", student_id):
            ids.append(student_id)

    _id_cache = ids
    _id_cache_at = now
    debug_log(f"[DB] refreshed student_id fuzzy-match cache: {len(ids)} ids")
    return ids


def _constrained_repair_lookup(cur, ocr_id: str) -> dict[str, Any] | None:
    """
    Recover the real Student ID from noisy handwriting OCR, e.g.
    TAROUR29081S -> TARGUR290815, by fuzzy-matching against EVERY active
    Student ID — the OCR mistake can land anywhere (including the first
    few letters), so we don't rely on an exact prefix filter.

    Safety rules (to avoid ever assigning the wrong student):
    - DB ID must be letters+digits format
    - Best similarity score >= 0.70 (rapidfuzz ratio, 0-1 scale)
    - Best score must beat the 2nd-best by >= 0.04 (reject ambiguous ties)
    """
    if len(ocr_id) < 6:
        return None

    candidates = _get_all_student_ids(cur)
    if not candidates:
        return None

    # Score both the raw OCR token AND a common-confusion-normalized version
    # (O<->0, I<->1, S<->5, L<->1, B<->8, Z<->2, G<->6). Handwriting mixes
    # these up anywhere in the ID (not only in a "digit section"), so
    # normalizing the whole string before scoring recovers matches like
    # TAROUR29O8IS -> TAR0UR290815 -> TARGUR290815 (single-char diff) even
    # though the raw string alone scores too low to trust.
    ocr_norm = ocr_id.translate(_OCR_VARIANTS)
    queries = {ocr_id, ocr_norm}

    best_by_id: dict[str, float] = {}
    for query in queries:
        for db_id, score, _ in process.extract(query, candidates, scorer=fuzz.ratio, limit=10):
            best_by_id[db_id] = max(best_by_id.get(db_id, 0.0), score / 100.0)

    if not best_by_id:
        return None

    ranked = sorted(best_by_id.items(), key=lambda item: item[1], reverse=True)
    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    debug_log(
        f"[DB] repair candidates top={best_id!r} score={best_score:.3f} "
        f"second={second_score:.3f} ocr={ocr_id!r} norm={ocr_norm!r}"
    )

    if best_score < 0.70:
        debug_log("[DB] repair rejected: score too low")
        return None
    if second_score and (best_score - second_score) < 0.04:
        debug_log("[DB] repair rejected: ambiguous between multiple students")
        return None

    return _exact_lookup(cur, best_id)


def _safe_ocr_variants(student_id: str) -> list[str]:
    """Same-length digit-section fixes only (e.g. O->0), keeping LETTERS+DIGITS shape."""
    match = re.fullmatch(r"([A-Z]+)([0-9OILSBZG]+)", student_id)
    if not match:
        return []
    prefix, rest = match.groups()
    digitish = rest.translate(_OCR_VARIANTS)
    if not digitish.isdigit():
        return []
    variant = prefix + digitish
    if variant == student_id or not re.fullmatch(r"[A-Z]{3,12}\d{4,12}", variant):
        return []
    return [variant]


def _serialize_student(row: dict[str, Any]) -> dict[str, Any]:
    scanned_at = row.get("scanned_at")
    return {
        "student_id": row.get("student_id") or "",
        "student_name": row.get("student_name") or "",
        "school_name": row.get("school_name") or "",
        "grade_name": row.get("grade_name") or "",
        "division_name": row.get("division_name") or "",
        "academic_year": row.get("academic_year") or "",
        "gender": row.get("gender") or "",
        "status": row.get("status") or "",
        "scanned_at": scanned_at.isoformat(sep=" ", timespec="seconds")
        if scanned_at is not None
        else None,
    }
