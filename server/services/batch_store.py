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

import threading
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import pymysql  # type: ignore[import-untyped]  # no stubs installed

from server.services.clock import now_ist
from server.services.debug_log import debug_log
from server.services.student_lookup import get_connection

BATCH_TABLE = "ocr_scan_batches"
LOG_TABLE = "ocr_scan_log"
# One row: when the watcher last did a pass, and from which machine. The scan
# page reads it to warn when nothing is listening for pages.
HEARTBEAT_TABLE = "ocr_inbox_watcher"
# One row per inbox file the watcher has started on: which batch it belongs
# to and how far it got. The rename in Drive used to be the only record that
# a file had been read, so a rename that failed (a folder shared as Viewer, a
# Drive hiccup) had every page read and logged again on the next pass. With
# this, a page is logged once whatever happens afterwards.
FILE_TABLE = "ocr_inbox_files"

# Where a file has got to. READING: claimed, `pages_logged` of its pages are
# in History. LOGGED: every page is in History; only the rename is left.
# DONE: renamed. PARKED: given up on, and left alone from then on.
FILE_READING = "reading"
FILE_LOGGED = "logged"
FILE_DONE = "done"
FILE_PARKED = "parked"
# Nothing more will happen to these, so they are not "waiting" in the folder.
SETTLED_STATES = (FILE_LOGGED, FILE_DONE, FILE_PARKED)

# MySQL named lock held by the one watcher allowed to process inboxes.
WATCHER_LOCK_NAME = "ocr_inbox_watcher"

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
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {FILE_TABLE} (
                    file_id VARCHAR(128) NOT NULL PRIMARY KEY,
                    folder_id VARCHAR(128) NOT NULL,
                    batch_id BIGINT UNSIGNED NOT NULL,
                    name VARCHAR(512) NULL,
                    state VARCHAR(16) NOT NULL,
                    pages_logged INT NOT NULL DEFAULT 0,
                    new_name VARCHAR(512) NULL,
                    attempts INT NOT NULL DEFAULT 0,
                    last_error VARCHAR(255) NULL,
                    updated_at DATETIME NOT NULL,
                    INDEX idx_batch (batch_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        conn.commit()
        debug_log(
            f"[BATCH] ensured MySQL tables {BATCH_TABLE!r}, {HEARTBEAT_TABLE!r}, "
            f"{FILE_TABLE!r} exist"
        )
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
                (now_ist(), folder_id),
            )
            cur.execute(
                f"""
                INSERT INTO {BATCH_TABLE}
                    (folder_id, owner_email, school, location, opened_by, opened_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (folder_id, owner_email, school or None, location or None,
                 opened_by or None, now_ist()),
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
                (now_ist(), batch_id),
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


def batches_to_watch(closed_since: datetime) -> list[dict[str, Any]]:
    """Open batches, and batches closed since `closed_since`, oldest first.

    A batch closed while pages were still waiting in its folder still owns
    those pages, so the watcher keeps looking at its folder for a while after
    Stop rather than leaving them for whichever school is scanned next.

    Args:
        closed_since: Naive IST. Batches closed before this are left out.

    Returns:
        The batches, serialised as `get_batch` returns them.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {BATCH_TABLE} "
                f"WHERE closed_at IS NULL OR closed_at >= %s ORDER BY id",
                (closed_since,),
            )
            return [_serialize(r) for r in cur.fetchall() or []]
    finally:
        conn.close()


def folder_batches(folder_id: str, closed_after: datetime | None = None) -> list[dict[str, Any]]:
    """One folder's batches, oldest first.

    Batches on a folder never overlap - opening one closes the last - so
    these are the windows a page's arrival time is matched against; the one
    closed by `closed_after` is how a page that arrived after it, between
    batches, is told apart from one dropped in before any Start.

    Args:
        folder_id: The Drive inbox folder.
        closed_after: Naive IST, usually when the oldest waiting page reached
            Drive. None for every batch the folder has had.

    Returns:
        The open batch, those closed after `closed_after`, and the last one
        that had closed by then.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if closed_after is None:
                cur.execute(
                    f"SELECT * FROM {BATCH_TABLE} WHERE folder_id = %s ORDER BY id",
                    (folder_id,),
                )
                return [_serialize(r) for r in cur.fetchall() or []]
            cur.execute(
                f"SELECT * FROM {BATCH_TABLE} WHERE folder_id = %s "
                f"AND closed_at <= %s ORDER BY id DESC LIMIT 1",
                (folder_id, closed_after),
            )
            before = list(cur.fetchall() or [])
            cur.execute(
                f"SELECT * FROM {BATCH_TABLE} WHERE folder_id = %s "
                f"AND (closed_at IS NULL OR closed_at > %s) ORDER BY id",
                (folder_id, closed_after),
            )
            return [_serialize(r) for r in before + list(cur.fetchall() or [])]
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
    """The watcher calls this every few seconds, from a thread of its own."""
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {HEARTBEAT_TABLE} (id, last_seen, host) VALUES (1, %s, %s)
                ON DUPLICATE KEY UPDATE last_seen = VALUES(last_seen), host = VALUES(host)
                """,
                (now_ist(), host or None),
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


# ------------------------------------------------------------ inbox files
def file_claims(file_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """What the watcher remembers about each of these Drive files.

    Args:
        file_ids: Drive file ids.

    Returns:
        Each file's row, by file id. Files it has never started on are
        simply absent.
    """
    if not file_ids:
        return {}
    _ensure_schema()
    placeholders = ", ".join(["%s"] * len(file_ids))
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {FILE_TABLE} WHERE file_id IN ({placeholders})",
                tuple(file_ids),
            )
            return {str(row["file_id"]): dict(row) for row in cur.fetchall() or []}
    finally:
        conn.close()


def settled_file_ids(file_ids: Sequence[str]) -> set[str]:
    """Those of these files the watcher has finished with or given up on.

    Args:
        file_ids: Drive file ids, as listed in an inbox.

    Returns:
        The ids still in the folder under their own name, but not waiting
        for anything.
    """
    return {
        file_id
        for file_id, row in file_claims(file_ids).items()
        if row.get("state") in SETTLED_STATES
    }


def claim_file(file_id: str, *, folder_id: str, batch_id: int, name: str) -> None:
    """Record that the watcher is starting on a file, before any of it is
    read or logged. Starting again on a file seen before (its [result]
    bracket taken off by hand) begins its record afresh.

    Args:
        file_id: The Drive file.
        folder_id: The inbox folder it is in.
        batch_id: The batch it is read for.
        name: Its name in Drive now.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {FILE_TABLE}
                    (file_id, folder_id, batch_id, name, state, pages_logged,
                     new_name, attempts, last_error, updated_at)
                VALUES (%s, %s, %s, %s, %s, 0, NULL, 0, NULL, %s)
                ON DUPLICATE KEY UPDATE
                    folder_id = VALUES(folder_id), batch_id = VALUES(batch_id),
                    name = VALUES(name), state = VALUES(state), pages_logged = 0,
                    new_name = NULL, attempts = 0, last_error = NULL,
                    updated_at = VALUES(updated_at)
                """,
                (file_id, folder_id, batch_id, (name or "")[:512], FILE_READING, now_ist()),
            )
        conn.commit()
    finally:
        conn.close()


def file_page_logged(file_id: str, pages_logged: int) -> None:
    """Remember how far into a file the watcher has got.

    Args:
        file_id: The Drive file.
        pages_logged: How many of its pages, from the first, are in History.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {FILE_TABLE} SET pages_logged = %s, updated_at = %s WHERE file_id = %s",
                (pages_logged, now_ist(), file_id),
            )
        conn.commit()
    finally:
        conn.close()


def file_pages_finished(
    file_id: str,
    *,
    batch_id: int,
    new_name: str,
    pages: int,
    counts: Mapping[int, int] | None = None,
) -> None:
    """Every page of the file is in History: keep the name it is to be given
    and count its pages on their batches, in one transaction, so that neither
    happens twice however often the rename has to be retried.

    Args:
        file_id: The Drive file.
        batch_id: The batch it was read for.
        new_name: The name to give it in Drive.
        pages: How many pages it has.
        counts: Pages to add to each batch's count, by batch id, when they
            did not all go to `batch_id` (a school's last sheet still
            uploading after the next school was started). None counts all
            `pages` on `batch_id`.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {FILE_TABLE}
                SET state = %s, new_name = %s, pages_logged = %s, attempts = 0,
                    last_error = NULL, updated_at = %s
                WHERE file_id = %s AND state = %s
                """,
                (FILE_LOGGED, (new_name or "")[:512], pages, now_ist(), file_id, FILE_READING),
            )
            if cur.rowcount > 0:
                for counted_on, count in sorted((counts or {batch_id: pages}).items()):
                    cur.execute(
                        f"UPDATE {BATCH_TABLE} SET pages_done = pages_done + %s WHERE id = %s",
                        (count, counted_on),
                    )
        conn.commit()
    finally:
        conn.close()


def file_failed(file_id: str, why: str) -> dict[str, Any] | None:
    """Count a failed attempt at a file.

    Args:
        file_id: The Drive file.
        why: What went wrong, kept (cut to 255 characters) for whoever looks.

    Returns:
        Its `attempts` and `state` afterwards, or None if the watcher never
        got to claim it.
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {FILE_TABLE} SET attempts = attempts + 1, last_error = %s, "
                f"updated_at = %s WHERE file_id = %s",
                ((why or "")[:255], now_ist(), file_id),
            )
            cur.execute(
                f"SELECT attempts, state FROM {FILE_TABLE} WHERE file_id = %s",
                (file_id,),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        conn.close()


def settle_file(file_id: str, state: str) -> None:
    """Record that the watcher has finished with a file.

    Args:
        file_id: The Drive file.
        state: FILE_DONE (renamed) or FILE_PARKED (left alone from now on).
    """
    _ensure_schema()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {FILE_TABLE} SET state = %s, updated_at = %s WHERE file_id = %s",
                (state, now_ist(), file_id),
            )
        conn.commit()
    finally:
        conn.close()


class WatcherLock:
    """The MySQL named lock that lets only one watcher process inboxes.

    Two watchers listing the same folder both read, log and count every page
    they find before either renames it. GET_LOCK belongs to the connection
    that took it, so this keeps that connection open for as long as the lock
    is wanted. If the process dies, or the connection drops, the lock goes
    with it.

    A host that dies without closing the connection (a power cut, the network
    gone) leaves MySQL holding it, and the lock, until the connection has been
    silent for wait_timeout - eight hours by default, with every replacement
    watcher only waiting meanwhile. `idle_timeout` shortens that for this one
    connection; whoever holds the lock must then use it more often than that
    (`keepalive`, from the watcher's heartbeat thread).
    """

    def __init__(self, name: str = WATCHER_LOCK_NAME, idle_timeout: int | None = None) -> None:
        self.name = name
        self.idle_timeout = idle_timeout
        self._conn: Any = None
        # The heartbeat thread uses the connection too, and a pymysql
        # connection must not be used from two threads at once.
        self._mutex = threading.Lock()

    @property
    def held(self) -> bool:
        """Whether this process held the lock when it was last checked."""
        return self._conn is not None

    def acquire(self) -> bool:
        """Take the lock without waiting, or confirm it is still ours.

        Returns:
            True if this process holds the lock, False if another does.
        """
        with self._mutex:
            if self._still_held():
                return True
            conn = get_connection()
            try:
                with conn.cursor() as cur:
                    if self.idle_timeout:
                        cur.execute("SET SESSION wait_timeout = %s", (int(self.idle_timeout),))
                    cur.execute("SELECT GET_LOCK(%s, 0) AS got", (self.name,))
                    row = cur.fetchone() or {}
            except Exception:
                conn.close()
                raise
            if row.get("got") == 1:
                self._conn = conn
                debug_log(f"[BATCH] took the watcher lock {self.name!r}")
                return True
            conn.close()
            return False

    def keepalive(self) -> bool:
        """Use the lock's connection, so MySQL does not drop it as idle, and
        check the lock is still ours.

        Returns:
            True if this process still holds the lock.
        """
        with self._mutex:
            return self._still_held()

    def release(self) -> None:
        """Let the lock go, if held."""
        with self._mutex:
            if self._conn is None:
                return
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT RELEASE_LOCK(%s) AS released", (self.name,))
            except pymysql.MySQLError as exc:
                debug_log(f"[BATCH] could not release the watcher lock: {exc}")
            finally:
                self._drop()

    def _still_held(self) -> bool:
        if self._conn is None:
            return False
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT IS_USED_LOCK(%s) = CONNECTION_ID() AS mine", (self.name,)
                )
                row = cur.fetchone() or {}
            if row.get("mine") == 1:
                return True
            debug_log(f"[BATCH] the watcher lock {self.name!r} is no longer ours")
        except pymysql.MySQLError as exc:
            # The connection dropped, and the lock with it.
            debug_log(f"[BATCH] lost the watcher lock connection: {exc}")
        self._drop()
        return False

    def _drop(self) -> None:
        conn, self._conn = self._conn, None
        try:
            conn.close()
        except pymysql.MySQLError:
            pass  # already closed by the server


def _serialize(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    for key in ("opened_at", "closed_at"):
        if item.get(key) is not None:
            item[key] = item[key].isoformat(sep=" ", timespec="seconds")
    item["open"] = row.get("closed_at") is None
    return item
