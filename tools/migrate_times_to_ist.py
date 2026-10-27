"""One-off: move the timestamps this app wrote in UTC onto India time (+05:30).

Until this change the app wrote its times from the clock of whatever machine
it ran on, and scanned_at from SQL NOW(). On EC2 both of those are UTC, so a
form scanned at 10:00 in a school office was stored as 04:30. The app now
writes India time everywhere (server/services/clock.py). This script shifts
the rows written before that by +05:30, so old and new rows read the same way.

    python tools/migrate_times_to_ist.py
        dry run (the default): per table and column, how many rows would
        change, the three latest before/after values, and the first value
        left alone. Only ever runs SELECTs, in a read-only session.

    python tools/migrate_times_to_ist.py --before "2026-10-28 06:00:00"
        the same dry run, limited to rows before that cut-off.

    python tools/migrate_times_to_ist.py --apply --before "2026-10-28 06:00:00"
        shift every value earlier than the cut-off by +05:30, all tables in
        ONE transaction - either every column moves or none does. Asks you
        to type SHIFT first (--yes skips the question).

Run it from the project root with the app's virtualenv, on a machine whose
.env points at the database to fix.

READ THIS BEFORE USING --apply

* Run it ONCE. A second run cannot tell a row it already shifted from one it
  has not, and moves it again.

* Run it at the moment the EC2 deploy of this change goes live. The old EC2
  code keeps writing UTC until it is replaced, so rows written after an
  early run would be left in UTC.

* --before is the moment the new code went live, as the OLD clock (UTC)
  showed it - for example `date -u "+%Y-%m-%d %H:%M:%S"` on EC2 straight
  after the restart. Every old row is earlier than that, and the first new
  row is stored 5h30m later than it, so a cut-off anywhere in that gap
  separates them cleanly; the dry run prints the first value it would leave
  alone so you can see the gap.

* active_student_data is a shared warehouse table, not this app's own.
  Before shifting scanned_at, confirm no other system reads it as UTC or
  writes it itself. The dry run also warns when the table has a column that
  MySQL bumps on every update (ON UPDATE CURRENT_TIMESTAMP), because the
  shift would then touch that column on every row it moves.

* scanned_at was written by SQL NOW(), which follows the MySQL session time
  zone. The dry run checks that zone, and if it is not UTC the column was
  never in UTC and is left alone.

* Rows written by a machine that was ALREADY on India time - for example a
  laptop in India running the app, which wrote created_at, reviewed_at and
  the batch times from its own clock - cannot be told apart from UTC rows,
  and will be pushed 5h30m too late. If the app has only ever run in
  production on EC2 this does not arise; if not, check the samples first.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Run as "python tools/migrate_times_to_ist.py" from anywhere: put the project
# root on the path so the app's own database settings and connection are used.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.config import DB_HOST, DB_NAME  # noqa: E402
from server.services.student_lookup import get_connection  # noqa: E402

SHIFT_MINUTES = 5 * 60 + 30
CUTOFF_FORMAT = "%Y-%m-%d %H:%M:%S"

# (table, column, key column shown beside each sample, written by SQL NOW()).
# Only scanned_at came from NOW(); the rest came from the app machine's clock.
TARGETS = [
    ("active_student_data", "scanned_at", "student_id", True),
    ("ocr_scan_log", "created_at", "id", False),
    ("ocr_scan_log", "reviewed_at", "id", False),
    ("ocr_scan_batches", "opened_at", "id", False),
    ("ocr_scan_batches", "closed_at", "id", False),
    ("ocr_inbox_watcher", "last_seen", "id", False),
]


def _parse_cutoff(text: str) -> datetime:
    try:
        return datetime.strptime(text.strip(), CUTOFF_FORMAT)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'use the form "YYYY-MM-DD HH:MM:SS", not {text!r}'
        ) from None


def _session_offset_minutes(cur) -> int:
    """How far the MySQL session's NOW() is from UTC, in minutes."""
    cur.execute(
        "SELECT @@session.time_zone AS tz, "
        "TIMESTAMPDIFF(MINUTE, UTC_TIMESTAMP(), NOW()) AS offset_minutes"
    )
    row = cur.fetchone() or {}
    offset = int(round(float(row.get("offset_minutes") or 0)))
    sign = "-" if offset < 0 else "+"
    hours, minutes = divmod(abs(offset), 60)
    print(
        f"MySQL session time zone: {row.get('tz')!r} "
        f"(NOW() is UTC{sign}{hours:02d}:{minutes:02d})"
    )
    return offset


def _column_info(cur, table: str, column: str) -> dict | None:
    cur.execute(
        """
        SELECT c.DATA_TYPE AS data_type, t.ENGINE AS engine
        FROM information_schema.COLUMNS c
        JOIN information_schema.TABLES t
          ON t.TABLE_SCHEMA = c.TABLE_SCHEMA AND t.TABLE_NAME = c.TABLE_NAME
        WHERE c.TABLE_SCHEMA = DATABASE() AND c.TABLE_NAME = %s AND c.COLUMN_NAME = %s
        """,
        (table, column),
    )
    return cur.fetchone()


def _auto_update_columns(cur, table: str) -> list[str]:
    """Columns MySQL rewrites by itself on every UPDATE of a row."""
    cur.execute(
        """
        SELECT COLUMN_NAME AS name
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
          AND LOWER(EXTRA) LIKE %s
        """,
        (table, "%on update%"),
    )
    return [row["name"] for row in cur.fetchall() or []]


def _where(column: str, cutoff: datetime | None) -> tuple[str, tuple]:
    if cutoff is None:
        return f"`{column}` IS NOT NULL", ()
    return f"`{column}` IS NOT NULL AND `{column}` < %s", (cutoff,)


def _fmt(value) -> str:
    if isinstance(value, datetime):
        return value.strftime(CUTOFF_FORMAT)
    return "NULL" if value is None else str(value)


def plan(cur, cutoff: datetime | None) -> list[tuple[str, str, int]]:
    """Print what would change and return the (table, column, rows) to shift."""
    offset = _session_offset_minutes(cur)
    print()

    todo: list[tuple[str, str, int]] = []
    warned_tables: set[str] = set()
    for table, column, key, from_sql_now in TARGETS:
        label = f"{table}.{column}"
        info = _column_info(cur, table, column)
        if not info:
            print(f"{label}: not in this database - skipped\n")
            continue
        print(f"{label}  ({info['data_type']}, {info['engine']})")

        if from_sql_now and offset != 0:
            # NOW() was already off UTC, so these values were never UTC and
            # adding +05:30 would only make them wrong.
            print(
                "  skipped: written by SQL NOW(), and this database's session "
                "time zone is not UTC, so these values are not in UTC\n"
            )
            continue
        if (info.get("engine") or "").lower() != "innodb":
            print(
                "  ! not InnoDB: this table cannot take part in the transaction, "
                "so a failure part-way would leave it shifted"
            )
        if table not in warned_tables:
            warned_tables.add(table)
            bumped = [name for name in _auto_update_columns(cur, table) if name != column]
            if bumped:
                print(
                    f"  ! {', '.join(bumped)} is ON UPDATE CURRENT_TIMESTAMP: "
                    "shifting would also reset it on every row moved"
                )

        where, params = _where(column, cutoff)
        cur.execute(f"SELECT COUNT(*) AS c FROM `{table}` WHERE {where}", params)
        count = int((cur.fetchone() or {}).get("c") or 0)
        print(f"  {count} row(s) would change")

        if count:
            cur.execute(
                f"""
                SELECT `{key}` AS k, `{column}` AS before_value,
                       `{column}` + INTERVAL {SHIFT_MINUTES} MINUTE AS after_value
                FROM `{table}` WHERE {where}
                ORDER BY `{column}` DESC LIMIT 3
                """,
                params,
            )
            for row in cur.fetchall() or []:
                print(
                    f"    {key}={_fmt(row['k'])}  "
                    f"{_fmt(row['before_value'])} -> {_fmt(row['after_value'])}"
                )

        if cutoff is not None:
            # The first value on the new clock: it should sit at least 5h30m
            # after the newest row above, or the cut-off is in the wrong place.
            cur.execute(
                f"""
                SELECT `{key}` AS k, `{column}` AS v FROM `{table}`
                WHERE `{column}` >= %s ORDER BY `{column}` LIMIT 1
                """,
                (cutoff,),
            )
            first = cur.fetchone()
            if first:
                print(f"    first value left alone: {key}={_fmt(first['k'])}  {_fmt(first['v'])}")
        print()

        if count:
            todo.append((table, column, count))
    return todo


def apply(conn, todo: list[tuple[str, str, int]], cutoff: datetime) -> None:
    """Shift every planned column in one transaction."""
    conn.begin()
    try:
        with conn.cursor() as cur:
            for table, column, expected in todo:
                where, params = _where(column, cutoff)
                cur.execute(
                    f"UPDATE `{table}` "
                    f"SET `{column}` = `{column}` + INTERVAL {SHIFT_MINUTES} MINUTE "
                    f"WHERE {where}",
                    params,
                )
                print(f"  {table}.{column}: shifted {cur.rowcount} row(s) (planned {expected})")
        conn.commit()
    except Exception:
        conn.rollback()
        print("Failed - rolled back, nothing was changed.", file=sys.stderr)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Shift timestamps written in UTC by +05:30 (dry run unless --apply).",
    )
    parser.add_argument(
        "--before",
        type=_parse_cutoff,
        help='only rows earlier than this, "YYYY-MM-DD HH:MM:SS" as the OLD (UTC) '
        "clock showed it - required with --apply",
    )
    parser.add_argument(
        "--apply", action="store_true", help="actually write (one transaction)"
    )
    parser.add_argument(
        "--yes", action="store_true", help="with --apply, do not ask to confirm"
    )
    args = parser.parse_args(argv)

    if args.apply and args.before is None:
        parser.error("--apply needs --before \"YYYY-MM-DD HH:MM:SS\" (the go-live moment, UTC)")

    mode = "APPLY" if args.apply else "DRY RUN - nothing will be written"
    print(f"Shift UTC timestamps to India time (+05:30) - {mode}")
    print(f"Database: {DB_NAME} on {DB_HOST}")
    if args.before is None:
        print("Cut-off: none given - counting every value that is set")
    else:
        print(f"Cut-off: values before {_fmt(args.before)}")

    conn = get_connection()
    try:
        if not args.apply:
            with conn.cursor() as cur:
                # Belt and braces: the dry run only SELECTs, and a read-only
                # session makes sure a slip could never write either. A server
                # too old to know the setting still gets only SELECTs.
                try:
                    cur.execute("SET SESSION TRANSACTION READ ONLY")
                except Exception as exc:  # noqa: BLE001
                    print(f"(could not make the session read-only: {exc})")
                plan(cur, args.before)
            print("Dry run only. Nothing was written.")
            return 0

        with conn.cursor() as cur:
            todo = plan(cur, args.before)
        # The plan's SELECTs opened a transaction; end it so the shift runs
        # in a fresh one of its own.
        conn.rollback()
        if not todo:
            print("Nothing to shift.")
            return 0

        if not args.yes:
            total = sum(count for _table, _column, count in todo)
            answer = input(f"Type SHIFT to move {total} value(s) forward by 5h30m: ")
            if answer.strip() != "SHIFT":
                print("Not confirmed. Nothing was written.")
                return 1

        apply(conn, todo, args.before)
        print("Done. Do not run this again.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
