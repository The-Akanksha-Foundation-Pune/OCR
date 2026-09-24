"""Who is scanning which school right now.

A page landing in someone's Drive inbox does not say which school it came
from. The person picks the school in the browser and presses Start; that opens
a batch tied to their inbox folder, and every page landing there is read
against that school until they press Stop. One open batch per folder, so a
page can never be filed under two schools.

Kept in the same MySQL database as everything else: the watcher is a separate
process from the web app, and this table is how the two agree.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from server.services.debug_log import debug_log
from server.services.student_lookup import get_connection

BATCH_TABLE = "ocr_scan_batches"
LOG_TABLE = "ocr_scan_log"
# One row: when the watcher last did a pass, and from which machine. The scan
# page reads it to warn when nothing is listening for pages.
HEARTBEAT_TABLE = "ocr_inbox_watcher"

_initialized = False


def _ensure_schema() -> None:
    global _initialized
    if _initialized:
        return
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {BATCH_TABLE} (
                    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    folder_id VARCHAR(128) NOT NULL,
                    owner_email VARCHAR(255) NOT NULL,
                    school VARCHAR(255) NULL,
                    location VARCHAR(64) NULL,
                    opened_by VARCHAR(255) NULL,
                    opened_at DATETIME NOT NULL,
                    closed_at DATETIME NULL,
                    pages_done INT NOT NULL DEFAULT 0,
                    INDEX idx_folder_open (folder_id, closed_at),
                    INDEX idx_owner_open (owner_email, closed_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {HEARTBEAT_TABLE} (
                    id TINYINT UNSIGNED PRIMARY KEY,
                    last_seen DATETIME NOT NULL,
                    host VARCHAR(255) NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        conn.commit()
        debug_log(f"[BATCH] ensured MySQL tables {BATCH_TABLE!r}, {HEARTBEAT_TABLE!r} exist")
    finally:
        conn.close()
    _initialized = True


def source_tag(batch_id: int) -> str:
    """The `source` written on every log row this batch produces. It is how
    results are found again without adding a column to the log table."""
    return f"drive:{batch_id}"


def open_batch(
    *,
    folder_id: str,
    owner_email: str,
    school: str | None,
    location: str | None,
    opened_by: str | None,
) -> int:
    """Start a batch for an inbox folder, closing any batch still open on it."""
    _ensure_schema()
    owner_email = (owner_email or "").strip().lower()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {BATCH_TABLE} SET closed_at = %s "
                f"WHERE folder_id = %s AND closed_at IS NULL",
                (datetime.now(), folder_id),
            )
            cur.execute(
                f"""
                INSERT INTO {BATCH_TABLE}
                    (folder_id, owner_email, school, location, opened_by, opened_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (folder_id, owner_email, school or None, location or None,
                 opened_by or None, datetime.now()),
            )
            batch_id = int(cur.lastrowid)
        conn.commit()
        debug_log(
            f"[BATCH] opened {batch_id} owner={owner_email!r} "
            f"school={school!r} location={location!r} by={opened_by!r}"
        )
        return batch_id
    finally:
        conn.close()


def close_batch(batch_id: int) -> bool:
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {BATCH_TABLE} SET closed_at = %s WHERE id = %s AND closed_at IS NULL",
                (datetime.now(), batch_id),
            )
            closed = cur.rowcount > 0
        conn.commit()
        if closed:
            debug_log(f"[BATCH] closed {batch_id}")
        return closed
    finally:
        conn.close()


def get_batch(batch_id: int) -> dict[str, Any] | None:
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {BATCH_TABLE} WHERE id = %s", (batch_id,))
            row = cur.fetchone()
            return _serialize(row) if row else None
    finally:
        conn.close()


def open_batch_for(owner_email: str) -> dict[str, Any] | None:
    """The batch this person currently has open, if any."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {BATCH_TABLE} WHERE owner_email = %s AND closed_at IS NULL "
                f"ORDER BY id DESC LIMIT 1",
                ((owner_email or "").strip().lower(),),
            )
            row = cur.fetchone()
            return _serialize(row) if row else None
    finally:
        conn.close()


def list_open_batches() -> list[dict[str, Any]]:
    """What the watcher polls: only folders somebody is actively scanning
    into. Pages arriving in any other folder wait until a batch is opened."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {BATCH_TABLE} WHERE closed_at IS NULL ORDER BY id")
            return [_serialize(r) for r in cur.fetchall() or []]
    finally:
        conn.close()


def add_pages_done(batch_id: int, count: int = 1) -> None:
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {BATCH_TABLE} SET pages_done = pages_done + %s WHERE id = %s",
                (count, batch_id),
            )
        conn.commit()
    finally:
        conn.close()


def batch_results(batch_id: int, after_id: int = 0) -> list[dict[str, Any]]:
    """Log rows this batch produced, oldest first. `after_id` lets the page
    ask only for what it has not shown yet."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, created_at, confirmed_student_id, found_in_db,
                       matched_student_id, student_name, school_name, message,
                       matched_via, DrivelinkImage
                FROM {LOG_TABLE}
                WHERE source = %s AND id > %s
                ORDER BY id
                """,
                (source_tag(batch_id), after_id),
            )
            rows = cur.fetchall() or []
        out = []
        for row in rows:
            item = dict(row)
            created = item.get("created_at")
            if created is not None:
                item["created_at"] = created.isoformat(sep=" ", timespec="seconds")
            item["found_in_db"] = bool(item.get("found_in_db"))
            out.append(item)
        return out
    finally:
        conn.close()


def heartbeat(host: str = "") -> None:
    """The watcher calls this every pass."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {HEARTBEAT_TABLE} (id, last_seen, host) VALUES (1, %s, %s)
                ON DUPLICATE KEY UPDATE last_seen = VALUES(last_seen), host = VALUES(host)
                """,
                (datetime.now(), host or None),
            )
        conn.commit()
    finally:
        conn.close()


def watcher_last_seen() -> datetime | None:
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT last_seen FROM {HEARTBEAT_TABLE} WHERE id = 1")
            row = cur.fetchone()
            return row.get("last_seen") if row else None
    finally:
        conn.close()


def _serialize(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    for key in ("opened_at", "closed_at"):
        if item.get(key) is not None:
            item[key] = item[key].isoformat(sep=" ", timespec="seconds")
    item["open"] = row.get("closed_at") is None
    return item
