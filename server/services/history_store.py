"""Durable log of scan-confirmation attempts.

Stored as a table in the SAME MySQL database as active_student_data (not a
local SQLite file) — so it survives server rebuilds/moves and gets backed up
along with the rest of the database. It only powers the "Not Found" list in
the History UI and never feeds back into lookups.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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
) -> None:
    """Record one /api/confirm attempt (found or not) for the History page."""
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
                    datetime.now(),
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
        conn.commit()
    finally:
        conn.close()


def delete_scan_attempt(attempt_id: int) -> bool:
    """Delete one logged attempt by id (used by the "Not found" tab's
    delete button). Returns True if a row was actually removed."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {LOG_TABLE} WHERE id = %s", (attempt_id,))
            deleted = cur.rowcount > 0
        conn.commit()
        if deleted:
            debug_log(f"[HISTORY] deleted scan_attempt id={attempt_id}")
        return deleted
    finally:
        conn.close()


def get_attempt(attempt_id: int) -> dict[str, Any] | None:
    """Fetch one logged attempt, so a recheck can reuse its Drive link."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, created_at, confirmed_student_id, found_in_db,
                       source, message, {DRIVE_LINK_COLUMN}, {MATCHED_VIA_COLUMN}
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
) -> bool:
    """
    Turn a not-found entry into a found one after someone corrected the ID.

    The row is updated in place rather than deleted and re-inserted, so the
    original reading stays visible in confirmed_student_id - it is the
    record of what the OCR actually saw, which is what makes the log worth
    keeping.
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
            updated = cur.rowcount > 0
        conn.commit()
        if updated:
            debug_log(
                f"[HISTORY] attempt {attempt_id} resolved -> {matched_student_id!r}"
            )
        return updated
    finally:
        conn.close()


def mark_reviewed(
    student_ids: list[str],
    *,
    reviewed_by: str = "",
) -> int:
    """
    Record that a person has checked these matches against the paper.

    Keyed on the matched student rather than the log row id: the Scanned tab
    is a list of students, and a student may have several log rows if their
    form was scanned more than once. Marking the student settles all of them.
    """
    wanted = [sid.strip().upper() for sid in student_ids if sid and sid.strip()]
    if not wanted:
        return 0

    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            slots = ", ".join(["%s"] * len(wanted))
            cur.execute(
                f"""
                UPDATE {LOG_TABLE}
                SET {REVIEWED_AT_COLUMN} = %s,
                    {REVIEWED_BY_COLUMN} = %s
                WHERE found_in_db = 1
                  AND {REVIEWED_AT_COLUMN} IS NULL
                  AND UPPER(TRIM(matched_student_id)) IN ({slots})
                """,
                (datetime.now(), reviewed_by or None, *wanted),
            )
            updated = cur.rowcount
        conn.commit()
        debug_log(f"[HISTORY] marked {updated} row(s) reviewed by {reviewed_by!r}")
        return updated
    finally:
        conn.close()


def unreviewed_student_ids(school: str = "", location: str = "") -> list[str]:
    """
    Students whose match still needs a human eye, within the current filter.

    Used by "mark all": the button must only cover what the operator can
    actually see, never the whole database.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            where = (
                f"WHERE found_in_db = 1 AND {REVIEWED_AT_COLUMN} IS NULL "
                f"AND {MATCHED_VIA_COLUMN} IS NOT NULL "
                f"AND {MATCHED_VIA_COLUMN} NOT IN ('exact', 'shape_fix')"
            )
            params: list[Any] = []
            if school:
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
                params.append(location)
                if slots:
                    clause += (
                        f" OR UPPER(TRIM(COALESCE({SCAN_SCHOOL_COLUMN}, school_name)))"
                        f" IN ({slots})"
                    )
                    params.extend(name.upper() for name in in_schools)
                clause += ")"
                where += clause

            cur.execute(
                f"SELECT DISTINCT matched_student_id FROM {LOG_TABLE} {where}",
                params,
            )
            return [
                (r.get("matched_student_id") or "").strip()
                for r in cur.fetchall() or []
                if (r.get("matched_student_id") or "").strip()
            ]
    finally:
        conn.close()


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
