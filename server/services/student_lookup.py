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
    DEVANAGARI_SUFFIX_LOOKUP,
    EMPLOYEES_TABLE,
    MARK_SCANNED_ON_LOOKUP,
    MAX_REPAIR_LENGTH_DELTA,
    MIN_REPAIR_MARGIN,
    MIN_REPAIR_SCORE,
    STUDENT_TABLE,
)
from server.services.debug_log import debug_log
from server.services.devanagari import prefix_similarity, split_id
from server.services.id_shape import shape_variants

# Cache of every active student_id (small table — a few thousand rows) so
# fuzzy repair can score against the WHOLE list, not just rows that happen
# to share an (error-prone) OCR prefix. Refreshed every few minutes.
_ID_CACHE_TTL_SECONDS = 300
_id_cache: dict = {}
_id_cache_at: dict = {}

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


def resolve_scope(school: str | None, location: str | None) -> list[str]:
    """
    Which schools a lookup may match against.

    A school beats a location. A location on its own still has to narrow
    the search - leaving the school on "All Pune schools" and searching
    every school in the country is exactly the mistake that let a Pune
    form match a child in Nagpur.

    An empty list means no restriction.
    """
    named = (school or "").strip()
    if named:
        return [named]
    city = (location or "").strip()
    if city:
        return schools_in_location(city)
    return []


def find_student_by_id(
    student_id: str,
    mark_scanned: bool | None = None,
    school: str | None = None,
    location: str | None = None,
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

    scope = resolve_scope(school, location)
    if scope:
        debug_log(f"[DB] search limited to {len(scope)} school(s): {scope[:4]}...")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            row = _exact_lookup(cur, cleaned, scope=scope)
            matched_via = "exact"

            if row is None:
                # Fix the reading against the known 6-letter/6-digit shape
                # before anything fuzzy. This is the cheapest and safest
                # correction available: it uses the format rather than
                # guessing which of 16,000 students was meant, and it turns
                # a good share of readings into exact hits.
                for variant in shape_variants(cleaned):
                    debug_log(f"[DB] trying shape-corrected {variant!r}")
                    row = _exact_lookup(cur, variant, scope=scope)
                    if row is not None:
                        matched_via = "shape_fix"
                        cleaned = variant
                        break

            if row is None:
                for variant in _safe_ocr_variants(cleaned):
                    debug_log(f"[DB] trying safe OCR variant {variant!r}")
                    row = _exact_lookup(cur, variant, scope=scope)
                    if row is not None:
                        matched_via = "safe_variant"
                        cleaned = variant
                        break

            if row is None:
                row = _constrained_repair_lookup(cur, cleaned, scope=scope)
                if row is not None:
                    matched_via = "constrained_repair"

            if row is None and DEVANAGARI_SUFFIX_LOOKUP:
                row = _digit_suffix_lookup(cur, cleaned, scope=scope)
                if row is not None:
                    matched_via = "digit_suffix"

            if row is None:
                debug_log(f"[DB] RESULT: NOT FOUND for {cleaned!r}")
                return None

            matched_id = row.get("student_id") or cleaned
            # Carry how we got here up to the caller. An exact hit and a
            # fuzzy repair are both "found", but only one of them is
            # certain, and an auto-committing flow needs to tell them
            # apart afterwards.
            row["matched_via"] = matched_via
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


def schools_in_location(location: str) -> list[str]:
    """
    Every school in a city.

    Picking a location has to narrow the rows, not just the school
    dropdown - otherwise choosing "Mumbai" and leaving the school on
    "All Mumbai schools" sends no filter at all and quietly shows
    everything.
    """
    wanted = (location or "").strip()
    if not wanted:
        return []
    return [
        entry["school"]
        for entry in list_schools_with_location()
        if entry["location"].lower() == wanted.lower()
    ]


def get_scanned_students(
    limit: int = 25,
    offset: int = 0,
    search: str = "",
    school: str = "",
    location: str = "",
) -> tuple[list[dict[str, Any]], int]:
    """Students already marked scanned_at, newest first (for the History
    page's Scanned tab). Reads straight from active_student_data — no
    schema changes, no separate log needed since scanned_at is already
    the source of truth."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            where = "WHERE s.scanned_at IS NOT NULL"
            params: list[Any] = []
            if school:
                where += " AND UPPER(TRIM(s.school_name)) = UPPER(%s)"
                params.append(school)
            elif location:
                in_schools = schools_in_location(location)
                if in_schools:
                    slots = ", ".join(["%s"] * len(in_schools))
                    where += f" AND UPPER(TRIM(s.school_name)) IN ({slots})"
                    params.extend(name.upper() for name in in_schools)
                else:
                    where += " AND 1 = 0"
            if search:
                where += (
                    " AND (UPPER(s.student_id) LIKE %s"
                    " OR UPPER(s.student_name) LIKE %s)"
                )
                like = f"%{search.upper()}%"
                params.extend([like, like])

            cur.execute(
                f"SELECT COUNT(*) AS total FROM {STUDENT_TABLE} s {where}", params
            )
            total = (cur.fetchone() or {}).get("total", 0)

            # matched_via comes from the most recent log row for this
            # student, so the history can show which saves were exact reads
            # and which were recovered by fuzzy matching and deserve a look.
            cur.execute(
                f"""
                SELECT
                    s.student_id, s.student_name, s.school_name, s.grade_name,
                    s.division_name, s.academic_year, s.gender, s.status,
                    s.scanned_at,
                    (
                        SELECT l.matched_via
                        FROM ocr_scan_log l
                        WHERE l.matched_student_id = s.student_id
                          AND l.found_in_db = 1
                        ORDER BY l.id DESC
                        LIMIT 1
                    ) AS matched_via,
                    (
                        SELECT l.confirmed_student_id
                        FROM ocr_scan_log l
                        WHERE l.matched_student_id = s.student_id
                          AND l.found_in_db = 1
                        ORDER BY l.id DESC
                        LIMIT 1
                    ) AS ocr_read,
                    (
                        SELECT l.reviewed_at
                        FROM ocr_scan_log l
                        WHERE l.matched_student_id = s.student_id
                          AND l.found_in_db = 1
                        ORDER BY l.id DESC
                        LIMIT 1
                    ) AS reviewed_at,
                    (
                        SELECT l.reviewed_by
                        FROM ocr_scan_log l
                        WHERE l.matched_student_id = s.student_id
                          AND l.found_in_db = 1
                        ORDER BY l.id DESC
                        LIMIT 1
                    ) AS reviewed_by
                FROM {STUDENT_TABLE} s
                {where}
                ORDER BY s.scanned_at DESC
                LIMIT %s OFFSET %s
                """,
                (*params, limit, offset),
            )
            rows = cur.fetchall() or []
            return [_serialize_student(row) for row in rows], total
    finally:
        conn.close()


def _exact_lookup(
    cur, student_id: str, scope: list[str] | None = None
) -> dict[str, Any] | None:
    # With the scope known, a duplicated Student ID (10 of them exist in
    # active_student_data) resolves to the right child instead of whichever
    # row the database happens to return first.
    school_clause = ""
    params: list[Any] = [student_id]
    if scope:
        slots = ", ".join(["%s"] * len(scope))
        school_clause = f" AND UPPER(TRIM(school_name)) IN ({slots})"
        params.extend(name.upper() for name in scope)
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
        WHERE UPPER(TRIM(student_id)) = UPPER(%s){school_clause}
        LIMIT 1
        """,
        tuple(params),
    )
    return cur.fetchone()


def _get_all_student_ids(cur, scope: list[str] | None = None) -> list[str]:
    """
    Candidate Student IDs for fuzzy matching, optionally one school only.

    Narrowing to a school is the single most effective guard against
    matching the wrong child: it cuts ~16,500 candidates to a few hundred,
    and every cross-school near-twin stops being reachable at all.
    """
    global _id_cache, _id_cache_at
    key = "|".join(sorted(name.strip().upper() for name in (scope or [])))
    now = time.monotonic()

    cached = _id_cache.get(key)
    if cached and (now - _id_cache_at.get(key, 0.0)) < _ID_CACHE_TTL_SECONDS:
        return cached

    if scope:
        slots = ", ".join(["%s"] * len(scope))
        cur.execute(
            f"SELECT student_id FROM {STUDENT_TABLE} "
            f"WHERE UPPER(TRIM(school_name)) IN ({slots})",
            tuple(name.strip().upper() for name in scope),
        )
    else:
        cur.execute(f"SELECT student_id FROM {STUDENT_TABLE}")

    ids = []
    for row in cur.fetchall() or []:
        student_id = (row.get("student_id") or "").upper().strip()
        if re.fullmatch(r"[A-Z]{3,12}\d{4,12}", student_id):
            ids.append(student_id)

    _id_cache[key] = ids
    _id_cache_at[key] = now
    debug_log(
        f"[DB] fuzzy-match candidates for scope={key or 'ALL'}: {len(ids)} ids"
    )
    return ids


def list_schools() -> list[str]:
    """Distinct school names, for the picker on the scan page."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT school_name FROM {STUDENT_TABLE} "
                f"WHERE school_name IS NOT NULL AND TRIM(school_name) <> '' "
                f"ORDER BY school_name"
            )
            return [(r.get("school_name") or "").strip() for r in cur.fetchall() or []]
    finally:
        conn.close()


def list_schools_with_location() -> list[dict[str, str]]:
    """
    Schools paired with their city, for the location filter.

    active_student_data has no location column, so the city is derived from
    where the school's staff are based - employees.location against
    employees.school_acronym. Reading it live rather than hardcoding a list
    means the mapping follows the HR data instead of drifting from it.

    A school with no staff record (a couple exist) is grouped under "Other"
    rather than dropped - it still has students whose forms need scanning.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT school_name FROM {STUDENT_TABLE} "
                f"WHERE school_name IS NOT NULL AND TRIM(school_name) <> '' "
                f"ORDER BY school_name"
            )
            schools = [
                (r.get("school_name") or "").strip() for r in cur.fetchall() or []
            ]

            cur.execute(
                f"""
                SELECT UPPER(TRIM(school_acronym)) AS school,
                       TRIM(location) AS location,
                       COUNT(*) AS n
                FROM {EMPLOYEES_TABLE}
                WHERE school_acronym IS NOT NULL AND TRIM(school_acronym) <> ''
                  AND location IS NOT NULL AND TRIM(location) <> ''
                GROUP BY school, location
                """
            )
            # A school can show more than one city if a staff member moved;
            # the city most of its staff sit in is the right answer.
            best: dict[str, tuple[str, int]] = {}
            for row in cur.fetchall() or []:
                key = row["school"]
                count = int(row["n"])
                if key not in best or count > best[key][1]:
                    best[key] = (row["location"], count)
    finally:
        conn.close()

    return [
        {"school": name, "location": best.get(name.upper(), ("Other", 0))[0]}
        for name in schools
    ]


def _constrained_repair_lookup(
    cur, ocr_id: str, scope: list[str] | None = None
) -> dict[str, Any] | None:
    """
    Recover the real Student ID from noisy handwriting OCR, e.g.
    TAROUR29081S -> TARGUR290815, by fuzzy-matching against EVERY active
    Student ID — the OCR mistake can land anywhere (including the first
    few letters), so we don't rely on an exact prefix filter.

    Safety rules (to avoid ever assigning the wrong student):
    - DB ID must be letters+digits format
    - Reading must be within MAX_REPAIR_LENGTH_DELTA of the candidate's
      length, since IDs are a fixed 6 letters + 6 digits and a short read
      has dropped characters we would only be guessing at
    - Best similarity >= MIN_REPAIR_SCORE
    - Best must beat the runner-up by >= MIN_REPAIR_MARGIN

    The margin matters most. Every candidate here is a real child, so a
    near-tie is not "probably this one" - it is two students the reading
    cannot distinguish, and picking either silently marks the wrong one
    as scanned.
    """
    if len(ocr_id) < 6:
        return None

    candidates = _get_all_student_ids(cur, scope)
    if not candidates:
        return None

    # Score both the raw OCR token AND a common-confusion-normalized version
    # (O<->0, I<->1, S<->5, L<->1, B<->8, Z<->2, G<->6). Handwriting mixes
    # these up anywhere in the ID (not only in a "digit section"), so
    # normalizing the whole string before scoring recovers matches like
    # TAROUR29O8IS -> TAR0UR290815 -> TARGUR290815 (single-char diff) even
    # though the raw string alone scores too low to trust.
    ocr_norm = ocr_id.translate(_OCR_VARIANTS)
    # Shape-corrected forms score far better than the raw reading - the
    # digits are usually perfect once the format is applied - so give the
    # matcher those as well.
    queries = {ocr_id, ocr_norm, *shape_variants(ocr_id)}

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

    if best_score < MIN_REPAIR_SCORE:
        debug_log(
            f"[DB] repair rejected: score {best_score:.3f} below {MIN_REPAIR_SCORE}"
        )
        return None
    if second_score and (best_score - second_score) < MIN_REPAIR_MARGIN:
        debug_log(
            f"[DB] repair rejected: ambiguous - {best_score:.3f} vs {second_score:.3f} "
            f"is under the {MIN_REPAIR_MARGIN} margin"
        )
        return None
    if abs(len(best_id) - len(ocr_id)) > MAX_REPAIR_LENGTH_DELTA:
        debug_log(
            f"[DB] repair rejected: read {len(ocr_id)} chars against a "
            f"{len(best_id)}-char ID, too much is missing to guess from"
        )
        return None

    return _exact_lookup(cur, best_id, scope=scope)


def _digit_suffix_lookup(
    cur, ocr_id: str, scope: list[str] | None = None
) -> dict[str, Any] | None:
    """
    Anchor on the digits, rank on the letters.

    A Devanagari-written ID transliterates asymmetrically: the six digits
    are a date of birth and convert exactly, while the six letters are a
    name transliteration and only come back approximately. Whole-string
    fuzzy matching wastes that, so require the digits to match exactly and
    let the letters merely choose between the students who share them.

    The digits are a birth date, so up to 13 students share one — the
    margin check below is what keeps that from picking the wrong child.
    """
    letters, digits = split_id(ocr_id)
    if len(digits) < 4 or not letters:
        return None

    shortlist = [
        candidate
        for candidate in _get_all_student_ids(cur, scope)
        if candidate.endswith(digits)
    ]
    if not shortlist:
        debug_log(f"[DB] digit-suffix {digits!r}: no candidates")
        return None

    scored = sorted(
        (
            (prefix_similarity(letters, split_id(candidate)[0]), candidate)
            for candidate in shortlist
        ),
        reverse=True,
    )
    best_score, best_id = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    debug_log(
        f"[DB] digit-suffix {digits!r}: {len(shortlist)} candidates, "
        f"best={best_id!r} score={best_score:.3f} second={second_score:.3f}"
    )

    if best_score < 0.55:
        debug_log("[DB] digit-suffix rejected: letter prefix too dissimilar")
        return None
    if second_score and (best_score - second_score) < 0.08:
        debug_log("[DB] digit-suffix rejected: ambiguous between students")
        return None

    return _exact_lookup(cur, best_id, scope=scope)


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
        "matched_via": row.get("matched_via") or "",
        "ocr_read": row.get("ocr_read") or "",
        "reviewed_at": (
            row["reviewed_at"].isoformat(sep=" ", timespec="seconds")
            if row.get("reviewed_at")
            else ""
        ),
        "reviewed_by": row.get("reviewed_by") or "",
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
