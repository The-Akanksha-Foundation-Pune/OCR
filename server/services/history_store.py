"""Durable log of scan-confirmation attempts.

Stored as a table in the SAME MySQL database as active_student_data (not a
local SQLite file) — so it survives server rebuilds/moves and gets backed up
along with the rest of the database. It only powers the "Not Found" list in
the History UI and never feeds back into lookups.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from server.config import STUDENT_TABLE
from server.services.clock import now_ist
from server.services.debug_log import debug_log
from server.services.student_lookup import get_connection

LOG_TABLE = "ocr_scan_log"
DRIVE_LINK_COLUMN = "DrivelinkImage"
# How the student was found: "exact", "safe_variant", "constrained_repair",
# "digit_suffix". Only "exact" means the ID was read cleanly off the paper;
# the rest were recovered by fuzzy matching and are worth being able to
# audit later.
MATCHED_VIA_COLUMN = "matched_via"
# The school and city chosen on the scan page. school_name above is copied
# from the matched student, so it is NULL on every not-found row - which
# made those rows invisible to any school or location filter. These record
# what was being scanned, match or no match.
SCAN_SCHOOL_COLUMN = "scan_school"
SCAN_LOCATION_COLUMN = "scan_location"
# Set when a person has checked a fuzzy-matched row against the paper form.
# A repaired match is the software's best guess; this records that a human
# has since agreed with it, which is the only thing that turns it from
# "probably right" into "confirmed".
REVIEWED_AT_COLUMN = "reviewed_at"
REVIEWED_BY_COLUMN = "reviewed_by"
# Matches that were read rather than guessed, so never need a person's
# check: a clean read, or one fixed to the known ID shape. Must agree with
# EXACT_WAYS on the History page, which colours the rows from it.
EXACT_MATCH_WAYS = ("exact", "shape_fix")

_initialized = False


def _column_exists(cur, column_name: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) AS c
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (LOG_TABLE, column_name),
    )
    return (cur.fetchone() or {}).get("c", 0) > 0


def _ensure_schema() -> None:
    global _initialized
    if _initialized:
        return
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # The CURRENT_TIMESTAMP default is in the database's own zone
            # (UTC by default on RDS), so it is never relied on:
            # log_scan_attempt always passes created_at itself, from the IST
            # clock.
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {LOG_TABLE} (
                    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    confirmed_student_id VARCHAR(64) NOT NULL,
                    found_in_db TINYINT(1) NOT NULL,
                    matched_student_id VARCHAR(64) NULL,
                    student_name VARCHAR(255) NULL,
                    school_name VARCHAR(255) NULL,
                    source VARCHAR(255) NULL,
                    message VARCHAR(255) NULL,
                    {DRIVE_LINK_COLUMN} TEXT NULL,
                    {MATCHED_VIA_COLUMN} VARCHAR(32) NULL,
                    {SCAN_SCHOOL_COLUMN} VARCHAR(255) NULL,
                    {SCAN_LOCATION_COLUMN} VARCHAR(64) NULL,
                    {REVIEWED_AT_COLUMN} DATETIME NULL,
                    {REVIEWED_BY_COLUMN} VARCHAR(255) NULL,
                    INDEX idx_found_created (found_in_db, id DESC)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

            for column, ddl in (
                (SCAN_SCHOOL_COLUMN, "VARCHAR(255) NULL"),
                (SCAN_LOCATION_COLUMN, "VARCHAR(64) NULL"),
                (REVIEWED_AT_COLUMN, "DATETIME NULL"),
                (REVIEWED_BY_COLUMN, "VARCHAR(255) NULL"),
            ):
                if not _column_exists(cur, column):
                    cur.execute(
                        f"ALTER TABLE {LOG_TABLE} ADD COLUMN {column} {ddl}"
                    )
                    debug_log(f"[HISTORY] added {column} column to {LOG_TABLE!r}")

            if not _column_exists(cur, MATCHED_VIA_COLUMN):
                cur.execute(
                    f"ALTER TABLE {LOG_TABLE} "
                    f"ADD COLUMN {MATCHED_VIA_COLUMN} VARCHAR(32) NULL"
                )
                debug_log(
                    f"[HISTORY] added {MATCHED_VIA_COLUMN} column to {LOG_TABLE!r}"
                )

            has_drive_link = _column_exists(cur, DRIVE_LINK_COLUMN)
            has_image_link = _column_exists(cur, "ImageLink")

            if not has_drive_link and has_image_link:
                cur.execute(
                    f"ALTER TABLE {LOG_TABLE} "
                    f"CHANGE COLUMN ImageLink {DRIVE_LINK_COLUMN} TEXT NULL"
                )
                debug_log(
                    f"[HISTORY] renamed ImageLink to {DRIVE_LINK_COLUMN} on {LOG_TABLE!r}"
                )
            elif not has_drive_link:
                cur.execute(
                    f"ALTER TABLE {LOG_TABLE} "
                    f"ADD COLUMN {DRIVE_LINK_COLUMN} TEXT NULL"
                )
                debug_log(
                    f"[HISTORY] added {DRIVE_LINK_COLUMN} column to {LOG_TABLE!r}"
                )
            elif has_image_link:
                # Both exist (partial migration) — copy then drop old column
                cur.execute(
                    f"""
                    UPDATE {LOG_TABLE}
                    SET {DRIVE_LINK_COLUMN} = ImageLink
                    WHERE {DRIVE_LINK_COLUMN} IS NULL AND ImageLink IS NOT NULL
                    """
                )
                cur.execute(f"ALTER TABLE {LOG_TABLE} DROP COLUMN ImageLink")
                debug_log(f"[HISTORY] migrated ImageLink data into {DRIVE_LINK_COLUMN}")

        conn.commit()
        debug_log(f"[HISTORY] ensured MySQL table {LOG_TABLE!r} exists")
    finally:
        conn.close()
    _initialized = True


def log_scan_attempt(
    *,
    confirmed_student_id: str,
    found_in_db: bool,
    matched_student_id: str | None = None,
    student_name: str | None = None,
    school_name: str | None = None,
    source: str = "",
    message: str = "",
    image_link: str | None = None,
    matched_via: str | None = None,
    scan_school: str | None = None,
    scan_location: str | None = None,
) -> int | None:
    """Record one /api/confirm attempt (found or not) for the History page.

    Returns:
        The new row's id, so a link added later (a document whose Drive
        upload is retried) can be written onto the same row.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {LOG_TABLE} (
                    created_at, confirmed_student_id, found_in_db,
                    matched_student_id, student_name, school_name, source, message,
                    {DRIVE_LINK_COLUMN}, {MATCHED_VIA_COLUMN},
                    {SCAN_SCHOOL_COLUMN}, {SCAN_LOCATION_COLUMN}
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    now_ist(),
                    confirmed_student_id,
                    1 if found_in_db else 0,
                    matched_student_id,
                    student_name,
                    school_name,
                    source,
                    message,
                    image_link,
                    matched_via,
                    scan_school,
                    scan_location,
                ),
            )
            row_id = int(cur.lastrowid or 0) or None
        conn.commit()
        return row_id
    finally:
        conn.close()


def set_attempt_image_link(attempt_id: int, image_link: str) -> bool:
    """Put the Drive link on a logged attempt that was saved without one.

    A confirmed document is logged as soon as the student is marked
    scanned; when its upload to Drive failed, the retry adds the link here
    rather than logging the same confirm twice.

    Args:
        attempt_id: The row log_scan_attempt returned.
        image_link: The document's Drive link.

    Returns:
        True when a row was updated.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {LOG_TABLE} SET {DRIVE_LINK_COLUMN} = %s WHERE id = %s",
                (image_link, int(attempt_id)),
            )
            changed = cur.rowcount > 0
        conn.commit()
        return changed
    finally:
        conn.close()


def delete_scan_attempt(attempt_id: int) -> bool:
    """Delete one not-found attempt by id (used by the "Not found" tab's
    delete button). Returns True if a row was actually removed."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Only a not-found row. A Not found list left open in another
            # tab still offers Delete on an entry someone has since
            # rechecked, and a found row is the only record of that scan:
            # its Drive link, the reading, and the matched_via that flags a
            # guessed match for checking. The student would stay marked
            # scanned with no trace of how.
            cur.execute(
                f"DELETE FROM {LOG_TABLE} WHERE id = %s AND found_in_db = 0",
                (attempt_id,),
            )
            deleted = cur.rowcount > 0
        conn.commit()
        if deleted:
            debug_log(f"[HISTORY] deleted scan_attempt id={attempt_id}")
        return deleted
    finally:
        conn.close()


def get_attempt(attempt_id: int) -> dict[str, Any] | None:
    """Fetch one logged attempt, so a recheck can reuse its Drive link and
    search the same school (or city) the form was scanned for."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, created_at, confirmed_student_id, found_in_db,
                       source, message, {DRIVE_LINK_COLUMN}, {MATCHED_VIA_COLUMN},
                       {SCAN_SCHOOL_COLUMN}, {SCAN_LOCATION_COLUMN}
                FROM {LOG_TABLE}
                WHERE id = %s
                """,
                (attempt_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def resolve_attempt(
    attempt_id: int,
    *,
    corrected_student_id: str,
    matched_student_id: str,
    student_name: str | None,
    school_name: str | None,
    matched_via: str | None,
    scanned_at: datetime | None = None,
) -> bool:
    """Turn a not-found entry into a found one after someone corrected the ID.

    The row is updated in place rather than deleted and re-inserted, so the
    original reading stays visible in confirmed_student_id - it is the
    record of what the OCR actually saw, which is what makes the log worth
    keeping.

    With scanned_at, the matched child is marked scanned in the same
    transaction, and only once this entry has been claimed. Marking first
    left a child scanned with no log row behind it whenever the entry was
    deleted, or rechecked, in another tab while the lookup ran.

    Args:
        attempt_id: The not-found log row being resolved.
        corrected_student_id: The ID the person typed off the paper.
        matched_student_id: The ID of the child it resolved to.
        student_name: That child's name, copied onto the log row.
        school_name: That child's school. With scanned_at it also picks
            which child is marked: a Student ID is not unique on its own.
        matched_via: How the lookup found the child.
        scanned_at: When to record the child as scanned, or None to leave
            active_student_data alone.

    Returns:
        True if this call resolved the entry; False if it was no longer a
        not-found row, in which case nothing at all was changed.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {LOG_TABLE}
                SET found_in_db = 1,
                    matched_student_id = %s,
                    student_name = %s,
                    school_name = %s,
                    message = %s,
                    {MATCHED_VIA_COLUMN} = %s
                WHERE id = %s AND found_in_db = 0
                """,
                (
                    matched_student_id,
                    student_name,
                    school_name,
                    f"resolved by recheck as {corrected_student_id}",
                    matched_via,
                    attempt_id,
                ),
            )
            claimed = cur.rowcount > 0
            if not claimed:
                # Deleted or resolved elsewhere since it was read. The row
                # lock this UPDATE took is what makes the answer final.
                conn.rollback()
                return False
            if scanned_at is not None:
                school = (school_name or "").strip()
                school_clause = (
                    "UPPER(TRIM(school_name)) = UPPER(%s)"
                    if school
                    else "(school_name IS NULL OR TRIM(school_name) = '')"
                )
                cur.execute(
                    f"""
                    UPDATE {STUDENT_TABLE}
                    SET scanned_at = %s
                    WHERE UPPER(TRIM(student_id)) = UPPER(%s)
                      AND {school_clause}
                    """,
                    (scanned_at, matched_student_id, *([school] if school else [])),
                )
        conn.commit()
        debug_log(f"[HISTORY] attempt {attempt_id} resolved -> {matched_student_id!r}")
        return True
    finally:
        conn.close()


def schools_holding_student_id(student_id: str) -> list[str]:
    """Every school with a child under this exact Student ID.

    Student IDs are not unique: a handful are shared by two children in
    different schools. A recheck with no school to go on uses this to
    refuse rather than mark whichever of them the database returns first.

    Args:
        student_id: The ID to look for, compared trimmed and case-blind.

    Returns:
        The distinct school names, sorted; empty when nobody has the ID.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT school_name
                FROM {STUDENT_TABLE}
                WHERE UPPER(TRIM(student_id)) = UPPER(%s)
                """,
                (student_id.strip(),),
            )
            # Keyed case-blind, as every school comparison here is, so one
            # school spelt two ways is not taken for two children.
            by_key = {
                (row.get("school_name") or "").strip().upper(): (
                    row.get("school_name") or ""
                ).strip()
                for row in cur.fetchall() or []
            }
            return sorted(by_key.values())
    finally:
        conn.close()


def mark_reviewed(
    student_ids: list[str],
    *,
    reviewed_by: str = "",
    through_id: int | None = None,
) -> int:
    """
    Record that a person has checked these matches against the paper.

    Keyed on the matched student rather than the log row id: the Scanned tab
    is a list of students, and a student may have several log rows if their
    form was scanned more than once. Marking the student settles all of them.

    through_id, when given, leaves alone any log row saved after it: a
    guess that lands while "Mark all checked" runs was never on screen.
    """
    wanted = [sid.strip().upper() for sid in student_ids if sid and sid.strip()]
    if not wanted:
        return 0

    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            slots = ", ".join(["%s"] * len(wanted))
            bound = "" if through_id is None else " AND id <= %s"
            cur.execute(
                f"""
                UPDATE {LOG_TABLE}
                SET {REVIEWED_AT_COLUMN} = %s,
                    {REVIEWED_BY_COLUMN} = %s
                WHERE found_in_db = 1
                  AND {REVIEWED_AT_COLUMN} IS NULL
                  AND UPPER(TRIM(matched_student_id)) IN ({slots}){bound}
                """,
                (
                    now_ist(),
                    reviewed_by or None,
                    *wanted,
                    *([] if through_id is None else [through_id]),
                ),
            )
            updated = cur.rowcount
        conn.commit()
        debug_log(f"[HISTORY] marked {updated} row(s) reviewed by {reviewed_by!r}")
        return updated
    finally:
        conn.close()


def _scanned_list_where(
    search: str, school: str, location: str
) -> tuple[str, list[Any]]:
    """
    The WHERE clause of the History > Scanned list, over `s`.

    A copy of the filter in student_lookup.get_scanned_students, and it has
    to stay one: "Mark all checked" signs off exactly the rows that list
    shows for the same school, location and search. The log's own
    scan_school is no stand-in - the list files a row under the matched
    child's school, so a cross-school match would be counted under a school
    whose list never shows it. tests/test_history.py pages through the real
    list under every kind of filter and fails if the two disagree.
    """
    where = "WHERE s.scanned_at IS NOT NULL"
    params: list[Any] = []
    if school:
        where += " AND UPPER(TRIM(s.school_name)) = UPPER(%s)"
        params.append(school)
    elif location:
        from server.services.student_lookup import schools_in_location

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
    return where, params


def pending_review(
    search: str = "", school: str = "", location: str = ""
) -> list[tuple[str, int]]:
    """The Scanned list's rows that show "Recheck needed", on every page.

    The list takes a student's matched_via and reviewed_at from their
    NEWEST found log row, so this does too: a student whose latest save was
    exact is not pending however many old guesses they have, and one signed
    off before but guessed again since is.

    Args:
        search: The list's search box (Student ID or name, any part).
        school: The school filter; beats location when both are set.
        location: The city filter, used when no school is chosen.

    Returns:
        (student_id, id of that student's newest found log row) per
        pending student, sorted. Its length is the number the review bar
        shows; review_token() of it is what "Mark all checked" confirms.
    """
    _ensure_schema()
    where, params = _scanned_list_where(search, school, location)
    exact_ways = ", ".join(f"'{way}'" for way in EXACT_MATCH_WAYS)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT s.student_id AS student_id, l.id AS last_id
                FROM {STUDENT_TABLE} s
                JOIN (
                    SELECT matched_student_id, MAX(id) AS last_id
                    FROM {LOG_TABLE}
                    WHERE found_in_db = 1
                    GROUP BY matched_student_id
                ) latest ON latest.matched_student_id = s.student_id
                JOIN {LOG_TABLE} l ON l.id = latest.last_id
                {where}
                  AND l.{REVIEWED_AT_COLUMN} IS NULL
                  AND l.{MATCHED_VIA_COLUMN} IS NOT NULL
                  AND l.{MATCHED_VIA_COLUMN} <> ''
                  AND LOWER(l.{MATCHED_VIA_COLUMN}) NOT IN ({exact_ways})
                """,
                params,
            )
            pending = {
                ((row.get("student_id") or "").strip(), int(row.get("last_id") or 0))
                for row in cur.fetchall() or []
                if (row.get("student_id") or "").strip()
            }
            return sorted(pending)
    finally:
        conn.close()


def review_token(pending: Iterable[tuple[str, int]]) -> str:
    """A short fingerprint of exactly which rows were counted as pending.

    The count alone cannot tell "one signed off in another tab, one new
    guess saved" from no change at all, and then the new guess would be
    signed off unseen. The newest log id is in it too, so a child guessed
    again since - a new reading behind the same orange row - changes it.

    Args:
        pending: (student_id, newest found log id) pairs, as from
            pending_review().

    Returns:
        A hex string that differs whenever the set differs.
    """
    text = "\n".join(
        f"{student_id.upper()}:{last_id}" for student_id, last_id in sorted(pending)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def get_not_found(
    limit: int = 25,
    offset: int = 0,
    search: str = "",
    school: str = "",
    location: str = "",
) -> tuple[list[dict[str, Any]], int]:
    """Most recent confirm attempts that did NOT resolve to a student."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            where = "WHERE found_in_db = 0"
            params: list[Any] = []
            if school:
                # COALESCE so a found row still matches on the student's
                # school even if it predates these columns.
                where += (
                    f" AND UPPER(TRIM(COALESCE({SCAN_SCHOOL_COLUMN}, school_name)))"
                    " = UPPER(%s)"
                )
                params.append(school)
            elif location:
                from server.services.student_lookup import schools_in_location

                in_schools = schools_in_location(location)
                slots = ", ".join(["%s"] * len(in_schools)) if in_schools else ""
                clause = f" AND (UPPER(TRIM({SCAN_LOCATION_COLUMN})) = UPPER(%s)"
                values: list[Any] = [location]
                if slots:
                    clause += (
                        f" OR UPPER(TRIM(COALESCE({SCAN_SCHOOL_COLUMN}, school_name)))"
                        f" IN ({slots})"
                    )
                    values.extend(name.upper() for name in in_schools)
                clause += ")"
                where += clause
                params.extend(values)
            if search:
                where += " AND UPPER(confirmed_student_id) LIKE %s"
                params.append(f"%{search.upper()}%")

            cur.execute(f"SELECT COUNT(*) AS c FROM {LOG_TABLE} {where}", params)
            total = (cur.fetchone() or {}).get("c", 0)

            cur.execute(
                f"""
                SELECT id, created_at, confirmed_student_id, source, message,
                       {DRIVE_LINK_COLUMN}, {MATCHED_VIA_COLUMN},
                       {SCAN_SCHOOL_COLUMN}, {SCAN_LOCATION_COLUMN}
                FROM {LOG_TABLE}
                {where}
                ORDER BY id DESC
                LIMIT %s OFFSET %s
                """,
                (*params, limit, offset),
            )
            rows = cur.fetchall() or []
            for row in rows:
                created = row.get("created_at")
                if created is not None:
                    row["created_at"] = created.isoformat(sep=" ", timespec="seconds")

            return [dict(row) for row in rows], total
    finally:
        conn.close()
