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
                    INDEX idx_found_created (found_in_db, id DESC)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
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
                    {DRIVE_LINK_COLUMN}
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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


def get_not_found(
    limit: int = 25, offset: int = 0, search: str = ""
) -> tuple[list[dict[str, Any]], int]:
    """Most recent confirm attempts that did NOT resolve to a student."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            where = "WHERE found_in_db = 0"
            params: list[Any] = []
            if search:
                where += " AND UPPER(confirmed_student_id) LIKE %s"
                params.append(f"%{search.upper()}%")

            cur.execute(f"SELECT COUNT(*) AS c FROM {LOG_TABLE} {where}", params)
            total = (cur.fetchone() or {}).get("c", 0)

            cur.execute(
                f"""
                SELECT id, created_at, confirmed_student_id, source, message,
                       {DRIVE_LINK_COLUMN}
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
