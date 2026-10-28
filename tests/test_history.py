"""History page: recheck scope, "Mark all checked", search races, delete scope.

Covers bugs #2, #4, #6, #18 and #21, and the recheck's duplicate-ID and
lost-race cases.

The database tests run against an in-memory SQLite stand-in for MySQL
rather than the recording fake in conftest: these bugs are about WHICH rows
a statement touches, and only a database that actually evaluates the SQL
can show that. The real database stays blocked by conftest throughout.

The page tests render the real history.html through Flask, pull out its
script and run it under Node with a small fake DOM and a fetch whose
replies the test hands back in any order.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any, Self

import pytest

# ------------------------------------------------------------ SQLite stand-in
SCHEMA = """
CREATE TABLE active_student_data (
    student_id    TEXT,
    student_name  TEXT,
    school_name   TEXT,
    grade_name    TEXT,
    division_name TEXT,
    academic_year TEXT,
    gender        TEXT,
    status        TEXT,
    scanned_at    TEXT
);
CREATE TABLE ocr_scan_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at           TEXT,
    confirmed_student_id TEXT NOT NULL,
    found_in_db          INTEGER NOT NULL,
    matched_student_id   TEXT,
    student_name         TEXT,
    school_name          TEXT,
    source               TEXT,
    message              TEXT,
    DrivelinkImage       TEXT,
    matched_via          TEXT,
    scan_school          TEXT,
    scan_location        TEXT,
    reviewed_at          TEXT,
    reviewed_by          TEXT
);
"""

# Columns PyMySQL would hand back as datetimes; SQLite returns text.
_DATETIME_KEYS = frozenset({"created_at", "scanned_at", "reviewed_at"})

# Which schools each city holds, standing in for the employees table.
CITY_SCHOOLS: dict[str, list[str]] = {
    "pune": ["ABMPS"],
    "nagpur": ["NAGSCH"],
    "mumbai": [],
}


def _to_sqlite(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    return value


def _from_sqlite(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for key in _DATETIME_KEYS & out.keys():
        if isinstance(out[key], str) and out[key]:
            out[key] = datetime.fromisoformat(out[key])
    return out


class SqliteCursor:
    """Just enough of a PyMySQL DictCursor over sqlite3 for these queries."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._cur = conn.cursor()
        self.rowcount = 0
        self.lastrowid = 0

    def execute(self, sql: str, params: Any = None) -> int:
        values = tuple(_to_sqlite(v) for v in (params or ()))
        self._cur.execute(sql.replace("%s", "?"), values)
        self.rowcount = self._cur.rowcount
        self.lastrowid = self._cur.lastrowid or 0
        return self.rowcount

    def fetchone(self) -> dict[str, Any] | None:
        row = self._cur.fetchone()
        return _from_sqlite(row) if row is not None else None

    def fetchall(self) -> list[dict[str, Any]]:
        return [_from_sqlite(row) for row in self._cur.fetchall()]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._cur.close()


class SqliteConnection:
    """A PyMySQL-shaped handle on one shared in-memory database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def cursor(self) -> SqliteCursor:
        return SqliteCursor(self._conn)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def begin(self) -> None:
        return None

    def close(self) -> None:
        # The app closes its connection after every call; the in-memory
        # database has to outlive that, so only the fixture really closes it.
        return None


@pytest.fixture
def sqlite_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[sqlite3.Connection]:
    """Point every module's get_connection at an in-memory SQLite database."""
    import server.blueprints.history_routes  # noqa: F401 - load before patching
    from server.services import history_store, student_lookup

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)

    def connect() -> SqliteConnection:
        return SqliteConnection(conn)

    for name, module in list(sys.modules.items()):
        if name.startswith(("server", "tools")) and hasattr(module, "get_connection"):
            monkeypatch.setattr(module, "get_connection", connect)
    monkeypatch.setattr(history_store, "_initialized", True)
    # The fuzzy-match candidate cache is per process; a list cached by an
    # earlier test would otherwise answer for this test's students.
    monkeypatch.setattr(student_lookup, "_id_cache", {})
    monkeypatch.setattr(student_lookup, "_id_cache_at", {})
    monkeypatch.setattr(
        student_lookup,
        "schools_in_location",
        lambda location: list(CITY_SCHOOLS.get((location or "").strip().lower(), [])),
    )
    yield conn
    conn.close()


def add_student(
    conn: sqlite3.Connection,
    student_id: str,
    school: str,
    *,
    name: str = "",
    scanned: bool = True,
) -> None:
    conn.execute(
        "INSERT INTO active_student_data (student_id, student_name, school_name, scanned_at)"
        " VALUES (?, ?, ?, ?)",
        (
            student_id,
            name or f"Child {student_id}",
            school,
            "2026-10-20 10:00:00" if scanned else None,
        ),
    )
    conn.commit()


def add_log(
    conn: sqlite3.Connection,
    read: str,
    *,
    found: bool,
    matched: str | None = None,
    via: str | None = None,
    reviewed: bool = False,
    school_name: str | None = None,
    scan_school: str | None = None,
    scan_location: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO ocr_scan_log (created_at, confirmed_student_id, found_in_db,"
        " matched_student_id, school_name, source, message, matched_via,"
        " scan_school, scan_location, reviewed_at, reviewed_by)"
        " VALUES (?, ?, ?, ?, ?, 'test', '', ?, ?, ?, ?, ?)",
        (
            "2026-10-20 10:00:00",
            read,
            1 if found else 0,
            matched,
            school_name,
            via,
            scan_school,
            scan_location,
            "2026-10-21 09:00:00" if reviewed else None,
            "earlier@akanksha.org" if reviewed else None,
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def scalar(conn: sqlite3.Connection, sql: str, *params: Any) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row is not None else None


# ===================================================== #2 recheck keeps scope
def seed_recheck(conn: sqlite3.Connection) -> None:
    """ABMPS (Pune) has four children; NAGSCH (Nagpur) has TARGUR290815."""
    for sid in ("ABCDEF111111", "GHIJKL222222", "MNOPQR333333", "STUVWX444444"):
        add_student(conn, sid, "ABMPS", scanned=False)
    add_student(conn, "TARGUR290815", "NAGSCH", scanned=False)


def test_get_attempt_returns_the_scan_school_and_location(sqlite_db: sqlite3.Connection) -> None:
    from server.services.history_store import get_attempt

    attempt_id = add_log(
        sqlite_db, "TARGUX290815", found=False, scan_school="ABMPS", scan_location="Pune"
    )

    attempt = get_attempt(attempt_id)

    assert attempt is not None
    assert attempt["scan_school"] == "ABMPS"
    assert attempt["scan_location"] == "Pune"


def test_recheck_does_not_reach_a_child_in_another_school(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    """The bug's own trigger: the near-twin lives in NAGSCH, the form in ABMPS."""
    seed_recheck(sqlite_db)
    attempt_id = add_log(
        sqlite_db, "TARGUX290815", found=False, scan_school="ABMPS", scan_location="Pune"
    )

    response = client.post(
        f"/api/not-found/{attempt_id}/recheck", json={"student_id": "TARGUX290815"}
    )

    assert response.status_code == 200
    assert response.get_json()["found"] is False
    assert "ABMPS" in response.get_json()["message"]
    # Nobody in Nagpur was marked scanned, and the entry is still not found.
    assert scalar(
        sqlite_db,
        "SELECT scanned_at FROM active_student_data WHERE student_id = 'TARGUR290815'",
    ) is None
    assert scalar(sqlite_db, "SELECT found_in_db FROM ocr_scan_log WHERE id = ?", attempt_id) == 0


def test_recheck_without_a_school_stays_inside_the_scan_location(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    """With only a city recorded, the search covers that city's schools only."""
    seed_recheck(sqlite_db)
    pune_row = add_log(sqlite_db, "TARGUR290815", found=False, scan_location="Pune")

    response = client.post(
        f"/api/not-found/{pune_row}/recheck", json={"student_id": "TARGUR290815"}
    )

    assert response.status_code == 200
    assert response.get_json()["found"] is False
    assert scalar(
        sqlite_db,
        "SELECT scanned_at FROM active_student_data WHERE student_id = 'TARGUR290815'",
    ) is None


def test_recheck_still_finds_the_child_inside_the_scanned_school(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    seed_recheck(sqlite_db)
    school_row = add_log(sqlite_db, "ABCDEF11111", found=False, scan_school="ABMPS")
    city_row = add_log(sqlite_db, "TARGUR29081", found=False, scan_location="Nagpur")

    by_school = client.post(
        f"/api/not-found/{school_row}/recheck", json={"student_id": "ABCDEF111111"}
    )
    by_city = client.post(
        f"/api/not-found/{city_row}/recheck", json={"student_id": "TARGUR290815"}
    )

    assert by_school.get_json()["found"] is True
    assert by_school.get_json()["student"]["school_name"] == "ABMPS"
    assert by_city.get_json()["found"] is True
    assert by_city.get_json()["student"]["school_name"] == "NAGSCH"
    assert scalar(sqlite_db, "SELECT found_in_db FROM ocr_scan_log WHERE id = ?", school_row) == 1


def scanned_at(conn: sqlite3.Connection, student_id: str, school: str) -> Any:
    return scalar(
        conn,
        "SELECT scanned_at FROM active_student_data WHERE student_id = ? AND school_name = ?",
        student_id,
        school,
    )


@pytest.mark.parametrize(
    "scan_location",
    [
        pytest.param(None, id="no-school-or-city-recorded"),
        pytest.param("Mumbai", id="city-with-no-schools"),
    ],
)
def test_recheck_with_no_usable_scope_does_not_guess(
    sqlite_db: sqlite3.Connection, client: Any, scan_location: str | None
) -> None:
    """Set-aside rows, old rows and All/All scans record no school to search.

    Unscoped, the fuzzy chain reaches every school, so TARGUX290815 was
    "repaired" to a child in Nagpur and that child marked as scanned.
    """
    seed_recheck(sqlite_db)
    attempt_id = add_log(sqlite_db, "TARGUX290815", found=False, scan_location=scan_location)

    response = client.post(
        f"/api/not-found/{attempt_id}/recheck", json={"student_id": "TARGUX290815"}
    )

    assert response.status_code == 200
    assert response.get_json()["found"] is False
    assert "exact" in response.get_json()["message"]
    assert scanned_at(sqlite_db, "TARGUR290815", "NAGSCH") is None
    assert scalar(sqlite_db, "SELECT found_in_db FROM ocr_scan_log WHERE id = ?", attempt_id) == 0


def test_recheck_with_no_scope_still_takes_an_exact_id(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    seed_recheck(sqlite_db)
    attempt_id = add_log(sqlite_db, "TARGUX290815", found=False)

    response = client.post(
        f"/api/not-found/{attempt_id}/recheck", json={"student_id": "TARGUR290815"}
    )

    assert response.get_json()["found"] is True
    assert response.get_json()["student"]["scanned_at"]
    assert scanned_at(sqlite_db, "TARGUR290815", "NAGSCH") is not None
    assert scalar(sqlite_db, "SELECT found_in_db FROM ocr_scan_log WHERE id = ?", attempt_id) == 1


def test_recheck_marks_only_the_child_in_the_scanned_school(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    """Two children share DUPLIC000001; the form came from ABMPS."""
    add_student(sqlite_db, "DUPLIC000001", "ABMPS", scanned=False)
    add_student(sqlite_db, "DUPLIC000001", "NAGSCH", scanned=False)
    attempt_id = add_log(sqlite_db, "DUPLIC00000I", found=False, scan_school="ABMPS")

    response = client.post(
        f"/api/not-found/{attempt_id}/recheck", json={"student_id": "DUPLIC000001"}
    )

    assert response.get_json()["found"] is True
    assert response.get_json()["student"]["school_name"] == "ABMPS"
    assert scanned_at(sqlite_db, "DUPLIC000001", "ABMPS") is not None
    assert scanned_at(sqlite_db, "DUPLIC000001", "NAGSCH") is None


def test_recheck_with_no_scope_will_not_pick_between_two_children(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    add_student(sqlite_db, "DUPLIC000001", "ABMPS", scanned=False)
    add_student(sqlite_db, "DUPLIC000001", "NAGSCH", scanned=False)
    attempt_id = add_log(sqlite_db, "DUPLIC00000I", found=False)

    response = client.post(
        f"/api/not-found/{attempt_id}/recheck", json={"student_id": "DUPLIC000001"}
    )

    assert response.get_json()["found"] is False
    assert "ABMPS" in response.get_json()["message"]
    assert "NAGSCH" in response.get_json()["message"]
    assert scanned_at(sqlite_db, "DUPLIC000001", "ABMPS") is None
    assert scanned_at(sqlite_db, "DUPLIC000001", "NAGSCH") is None
    assert scalar(sqlite_db, "SELECT found_in_db FROM ocr_scan_log WHERE id = ?", attempt_id) == 0


@pytest.mark.parametrize(
    "elsewhere",
    [
        pytest.param("DELETE FROM ocr_scan_log WHERE id = ? AND found_in_db = 0",
                     id="deleted-in-another-tab"),
        pytest.param("UPDATE ocr_scan_log SET found_in_db = 1 WHERE id = ?",
                     id="rechecked-in-another-tab"),
    ],
)
def test_recheck_that_loses_the_row_to_another_tab_marks_nobody(
    sqlite_db: sqlite3.Connection,
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
    elsewhere: str,
) -> None:
    """#21: the row goes between the recheck reading it and saving the match."""
    from server.blueprints import history_routes

    add_student(sqlite_db, "KIDAAA100001", "ABMPS", scanned=False)
    attempt_id = add_log(sqlite_db, "KIDAAA10000I", found=False, scan_school="ABMPS")
    real_lookup = history_routes.find_student_by_id

    def lookup_while_another_tab_acts(*args: Any, **kwargs: Any) -> Any:
        sqlite_db.execute(elsewhere, (attempt_id,))
        sqlite_db.commit()
        return real_lookup(*args, **kwargs)

    monkeypatch.setattr(history_routes, "find_student_by_id", lookup_while_another_tab_acts)

    response = client.post(
        f"/api/not-found/{attempt_id}/recheck", json={"student_id": "KIDAAA100001"}
    )

    assert response.status_code == 409
    assert response.get_json().get("found") is not True
    assert scanned_at(sqlite_db, "KIDAAA100001", "ABMPS") is None


# ================================================= #4 / #6 "Mark all checked"
PAGE_SIZE = 25
EXACT_WAYS = ("exact", "shape_fix")


def seed_review(conn: sqlite3.Connection) -> None:
    """60 unreviewed fuzzy matches across two schools, plus the edge cases."""
    for n in range(60):
        sid = f"KIDAAA1000{n:02d}"
        school = "ABMPS" if n % 2 == 0 else "NAGSCH"
        add_student(conn, sid, school)
        add_log(conn, sid, found=True, matched=sid, via="constrained_repair", scan_school=school)

    # Read exactly: never needs a check.
    add_student(conn, "EXACTA000001", "ABMPS")
    add_log(conn, "EXACTA000001", found=True, matched="EXACTA000001", via="exact")
    # Already signed off.
    add_student(conn, "REVWED000001", "ABMPS")
    add_log(conn, "REVWED000001", found=True, matched="REVWED000001", via="safe_variant",
            reviewed=True)
    # An old fuzzy row, but the newest save was exact: the list shows exact.
    add_student(conn, "NEWEXA000001", "ABMPS")
    add_log(conn, "NEWEXA00000I", found=True, matched="NEWEXA000001", via="safe_variant")
    add_log(conn, "NEWEXA000001", found=True, matched="NEWEXA000001", via="exact")
    # Signed off once, then saved again by a fuzzy match: the list shows pending.
    add_student(conn, "OLDREV000001", "NAGSCH")
    add_log(conn, "OLDREV00000I", found=True, matched="OLDREV000001", via="safe_variant",
            reviewed=True)
    add_log(conn, "OLDREV00000l", found=True, matched="OLDREV000001", via="digit_suffix")
    # A fuzzy match into another school's child: the list files it under the
    # child's school (NAGSCH), not the school that was being scanned.
    add_student(conn, "CROSSX000001", "NAGSCH")
    add_log(conn, "CROSSX00000I", found=True, matched="CROSSX000001",
            via="constrained_repair", scan_school="ABMPS", scan_location="Pune")
    # Not on the Scanned list at all (scanned_at was cleared).
    add_student(conn, "GHOSTX000001", "ABMPS", scanned=False)
    add_log(conn, "GHOSTX00000I", found=True, matched="GHOSTX000001", via="constrained_repair")


def pending_on_every_page(search: str, school: str, location: str) -> set[str]:
    """What the operator can reach: every page of the Scanned list, filtered."""
    from server.services.student_lookup import get_scanned_students

    pending: set[str] = set()
    offset = 0
    while True:
        items, total = get_scanned_students(
            limit=PAGE_SIZE, offset=offset, search=search, school=school, location=location
        )
        for row in items:
            via = (row["matched_via"] or "").lower()
            if via and via not in EXACT_WAYS and not row["reviewed_at"]:
                pending.add(row["student_id"])
        offset += PAGE_SIZE
        if offset >= total:
            return pending


FILTERS = [
    pytest.param("", "", "", id="all-all"),
    pytest.param("KIDAAA100059", "", "", id="one-id-searched"),
    pytest.param("KIDAAA10001", "", "", id="partial-id"),
    pytest.param("", "ABMPS", "", id="one-school"),
    pytest.param("", "", "Pune", id="pune"),
    pytest.param("child kidaaa1000", "", "Nagpur", id="name-search-in-nagpur"),
    pytest.param("", "", "Mumbai", id="city-with-no-schools"),
    pytest.param("", "NAGSCH", "Pune", id="school-beats-location"),
]


@pytest.mark.parametrize(("search", "school", "location"), FILTERS)
def test_mark_all_covers_exactly_the_pending_rows_of_the_list(
    sqlite_db: sqlite3.Connection, search: str, school: str, location: str
) -> None:
    """Also the guard on _scanned_list_where staying a copy of the list's filter."""
    from server.services.history_store import pending_review

    seed_review(sqlite_db)
    expected = pending_on_every_page(search, school, location)

    pending = pending_review(school=school, location=location, search=search)

    assert {student_id for student_id, _last_id in pending} == expected
    assert len(pending) == len(expected)


def test_scanned_list_reports_the_server_count_for_its_filter(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    seed_review(sqlite_db)

    data = client.get("/api/scanned?page=1&page_size=25&search=KIDAAA100059").get_json()
    everything = client.get("/api/scanned?page=1&page_size=25").get_json()

    assert data["needs_review"] == 1
    # 60 + OLDREV + CROSSX across every page, not the 25 rows on page 1.
    assert everything["needs_review"] == 62
    assert data["review_token"] and everything["review_token"]
    assert data["review_token"] != everything["review_token"]


def test_tab_count_preload_skips_the_review_count(fake_db: Any, client: Any) -> None:
    """The page's page_size=1 preload only wants the total, not a scan of the log."""
    response = client.get("/api/scanned?page=1&page_size=1&review=0")

    assert response.status_code == 200
    assert response.get_json()["needs_review"] is None
    assert fake_db.executed(r"GROUP BY matched_student_id") == []
    # The list itself still asks for it.
    client.get("/api/scanned?page=1&page_size=25")
    assert fake_db.executed(r"GROUP BY matched_student_id") != []


def unreviewed_fuzzy_students(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT DISTINCT matched_student_id FROM ocr_scan_log"
        " WHERE found_in_db = 1 AND reviewed_at IS NULL"
        " AND matched_via NOT IN ('exact', 'shape_fix')"
    ).fetchall()
    return {row[0] for row in rows}


def shown_list(client: Any, school: str = "", location: str = "", search: str = "") -> dict[str, Any]:
    """The Scanned list as the page loads it, review count and token included."""
    query = "&".join(
        f"{key}={value}"
        for key, value in (("school", school), ("location", location), ("search", search))
        if value
    )
    return dict(client.get(f"/api/scanned?page=1&page_size=25&{query}").get_json())


def mark_all(client: Any, shown: dict[str, Any], **scope: str) -> Any:
    """Press "Mark all checked" on a list loaded earlier, as the page does."""
    return client.post(
        "/api/scanned/review",
        json={
            "all": True,
            "school": scope.get("school", ""),
            "location": scope.get("location", ""),
            "search": scope.get("search", ""),
            "expected": shown["needs_review"],
            "token": shown.get("review_token"),
        },
    )


def test_mark_all_with_a_search_signs_off_only_the_searched_row(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    """#6's first trigger, inside one school: one ID searched, one row to check."""
    seed_review(sqlite_db)
    before = unreviewed_fuzzy_students(sqlite_db)
    scope = {"school": "NAGSCH", "search": "KIDAAA100059"}
    shown = shown_list(client, **scope)

    response = mark_all(client, shown, **scope)

    assert shown["needs_review"] == 1
    assert response.status_code == 200
    assert response.get_json()["student_ids"] == ["KIDAAA100059"]
    assert unreviewed_fuzzy_students(sqlite_db) == before - {"KIDAAA100059"}
    assert scalar(
        sqlite_db,
        "SELECT reviewed_by FROM ocr_scan_log WHERE matched_student_id = 'KIDAAA100059'",
    ) == "staff@akanksha.org"


def test_mark_all_covers_a_whole_city(sqlite_db: sqlite3.Connection, client: Any) -> None:
    seed_review(sqlite_db)
    shown = shown_list(client, location="Pune")

    response = mark_all(client, shown, location="Pune")

    assert response.status_code == 200
    assert response.get_json()["reviewed"] == 30
    assert not any(sid for sid in unreviewed_fuzzy_students(sqlite_db)
                   if sid.startswith("KIDAAA") and int(sid[-2:]) % 2 == 0)


@pytest.mark.parametrize(
    "search",
    [pytest.param("", id="all-all-defaults"), pytest.param("KIDAAA1000", id="all-all-searched")],
)
def test_mark_all_across_every_school_is_refused(
    sqlite_db: sqlite3.Connection, client: Any, search: str
) -> None:
    """#6's open trigger: All/All has 62 rows to check, most on pages never seen."""
    seed_review(sqlite_db)
    before = unreviewed_fuzzy_students(sqlite_db)
    shown = shown_list(client, search=search)

    response = mark_all(client, shown, search=search)

    assert shown["needs_review"] > PAGE_SIZE
    assert response.status_code == 400
    assert "school" in response.get_json()["error"]
    assert unreviewed_fuzzy_students(sqlite_db) == before


def test_mark_all_refuses_when_the_list_changed_since_it_was_counted(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    seed_review(sqlite_db)
    before = unreviewed_fuzzy_students(sqlite_db)
    shown = shown_list(client, school="ABMPS")
    add_student(sqlite_db, "LATEKD000001", "ABMPS")
    add_log(sqlite_db, "LATEKD00000I", found=True, matched="LATEKD000001",
            via="constrained_repair", scan_school="ABMPS")

    response = mark_all(client, shown, school="ABMPS")

    assert response.status_code == 409
    assert response.get_json()["needs_review"] == 31
    assert "31" in response.get_json()["error"]
    assert unreviewed_fuzzy_students(sqlite_db) == before | {"LATEKD000001"}


def guess_again(conn: sqlite3.Connection) -> None:
    """A new fuzzy save for a child already on the list: new reading, same count."""
    add_log(conn, "KIDAAA1000O1", found=True, matched="KIDAAA100001",
            via="safe_variant", scan_school="NAGSCH")


def one_off_one_on(conn: sqlite3.Connection) -> None:
    """Signed off in another tab, and a new child guessed: the count stays put."""
    from server.services.history_store import mark_reviewed

    mark_reviewed(["KIDAAA100003"], reviewed_by="other@akanksha.org")
    add_student(conn, "LATEKD000001", "NAGSCH")
    add_log(conn, "LATEKD00000I", found=True, matched="LATEKD000001",
            via="constrained_repair", scan_school="NAGSCH")


@pytest.mark.parametrize(
    "change",
    [pytest.param(one_off_one_on, id="one-off-one-on"),
     pytest.param(guess_again, id="same-child-guessed-again")],
)
def test_mark_all_refuses_when_the_rows_changed_but_the_count_did_not(
    sqlite_db: sqlite3.Connection, client: Any, change: Any
) -> None:
    """The count alone let LATEKD000001 be signed off without ever being shown."""
    seed_review(sqlite_db)
    shown = shown_list(client, school="NAGSCH")
    change(sqlite_db)
    before = unreviewed_fuzzy_students(sqlite_db)
    # The premise: the number on the bar would still be right.
    assert shown_list(client, school="NAGSCH")["needs_review"] == shown["needs_review"]

    response = mark_all(client, shown, school="NAGSCH")

    assert response.status_code == 409
    assert unreviewed_fuzzy_students(sqlite_db) == before
    assert scalar(
        sqlite_db,
        "SELECT COUNT(*) FROM ocr_scan_log WHERE reviewed_by = 'staff@akanksha.org'",
    ) == 0


def test_mark_all_leaves_a_save_that_lands_while_it_runs(
    sqlite_db: sqlite3.Connection, client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A guess saved between the check and the update was never on screen."""
    from server.blueprints import history_routes

    seed_review(sqlite_db)
    shown = shown_list(client, school="NAGSCH")
    real_pending = history_routes.pending_review
    late: list[int] = []

    def pending_then_a_new_save(**kwargs: str) -> Any:
        rows = real_pending(**kwargs)
        late.append(add_log(sqlite_db, "KIDAAA1000O1", found=True, matched="KIDAAA100001",
                            via="safe_variant", scan_school="NAGSCH"))
        return rows

    monkeypatch.setattr(history_routes, "pending_review", pending_then_a_new_save)

    response = mark_all(client, shown, school="NAGSCH")

    assert response.status_code == 200
    assert scalar(sqlite_db, "SELECT reviewed_at FROM ocr_scan_log WHERE id = ?", late[0]) is None
    assert "KIDAAA100001" in pending_on_every_page("", "NAGSCH", "")


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"all": True, "school": "", "location": ""}, id="old-page-no-count"),
        pytest.param({"all": True, "school": "ABMPS"}, id="old-page-with-a-school"),
        pytest.param({"all": True, "school": "ABMPS", "expected": "30", "token": "x"},
                     id="count-not-a-number"),
        pytest.param({"all": True, "school": "ABMPS", "expected": True, "token": "x"},
                     id="count-is-a-flag"),
        pytest.param({"all": True, "school": "ABMPS", "expected": 30}, id="no-token"),
    ],
)
def test_mark_all_without_a_confirmed_count_marks_nothing(
    sqlite_db: sqlite3.Connection, client: Any, payload: dict[str, Any]
) -> None:
    """An old copy of the page still open in a tab must not sign off the log."""
    seed_review(sqlite_db)
    before = unreviewed_fuzzy_students(sqlite_db)

    response = client.post("/api/scanned/review", json=payload)

    assert response.status_code == 400
    assert unreviewed_fuzzy_students(sqlite_db) == before


def test_mark_one_row_still_works(sqlite_db: sqlite3.Connection, client: Any) -> None:
    seed_review(sqlite_db)

    response = client.post("/api/scanned/review", json={"student_ids": ["KIDAAA100003"]})

    assert response.status_code == 200
    assert "KIDAAA100003" not in unreviewed_fuzzy_students(sqlite_db)
    assert "KIDAAA100004" in unreviewed_fuzzy_students(sqlite_db)


# ========================================== #21 delete only not-found rows
def test_delete_refuses_a_row_that_has_been_resolved(
    sqlite_db: sqlite3.Connection, client: Any
) -> None:
    """A stale Not found list must not be able to delete a resolved scan."""
    add_student(sqlite_db, "KIDAAA100001", "ABMPS")
    resolved = add_log(sqlite_db, "KIDAAA10000I", found=True, matched="KIDAAA100001",
                       via="constrained_repair")

    response = client.delete(f"/api/not-found/{resolved}")

    assert response.status_code == 404
    assert scalar(sqlite_db, "SELECT COUNT(*) FROM ocr_scan_log WHERE id = ?", resolved) == 1


def test_delete_removes_a_not_found_row(sqlite_db: sqlite3.Connection, client: Any) -> None:
    missing = add_log(sqlite_db, "NOSUCH000001", found=False, scan_school="ABMPS")

    response = client.delete(f"/api/not-found/{missing}")

    assert response.status_code == 200
    assert scalar(sqlite_db, "SELECT COUNT(*) FROM ocr_scan_log WHERE id = ?", missing) == 0


# ============================================ errors stay off the browser (C2)
def test_history_errors_do_not_echo_exception_text(
    fake_db: Any, client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server.blueprints import history_routes

    def explode(**_kwargs: Any) -> Any:
        raise RuntimeError("(2003, \"Can't connect to MySQL server on 'db.internal'\")")

    monkeypatch.setattr(history_routes, "get_scanned_students", explode)
    monkeypatch.setattr(history_routes, "get_not_found", explode)

    for url in ("/api/scanned", "/api/not-found"):
        response = client.get(url)
        assert response.status_code == 500
        assert "db.internal" not in response.get_data(as_text=True)
        assert "2003" not in response.get_data(as_text=True)


# ====================================================== the page (#4/#6, #18)
HARNESS_JS = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");

const source = fs.readFileSync(process.argv[2], "utf8");
const scenario = process.argv[3];

class FakeElement {
  constructor(id) {
    this.id = id || "";
    this.hidden = false;
    this.textContent = "";
    this.innerHTML = "";
    this.className = "";
    this.value = "";
    this.placeholder = "";
    this.disabled = false;
    this.dataset = {};
    this.style = {};
    this.children = [];
    this.listeners = {};
    this.classList = { add() {}, remove() {}, toggle() {}, contains() { return false; } };
  }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  fire(type) { for (const fn of this.listeners[type] || []) fn({ target: this }); }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  appendChild(child) { this.children.push(child); return child; }
  setAttribute() {}
  remove() {}
  closest() { return null; }
}

const elements = {};
const state = { location: "", confirmAnswer: true };
// Elements the markup starts with the hidden attribute on.
const STARTS_HIDDEN = new Set(["review-bar", "review-all-hint"]);
function el(id) {
  if (!elements[id]) {
    elements[id] = new FakeElement(id);
    elements[id].hidden = STARTS_HIDDEN.has(id);
    if (id === "hist-location") {
      elements[id].querySelector = () => ({ value: state.location });
    }
  }
  return elements[id];
}
const document = {
  getElementById: el,
  querySelectorAll() { return []; },
  createElement(tag) { return new FakeElement(tag); },
  body: new FakeElement("body"),
};

const requests = [];
function fetch(url, options) {
  return new Promise((resolve) => {
    requests.push({ url: String(url), options: options || {}, resolve, answered: false });
  });
}
function reply(request, status, data) {
  request.answered = true;
  request.resolve({ ok: status >= 200 && status < 300, status, json: async () => data });
}
const confirms = [];
function confirm(message) {
  confirms.push(String(message));
  return state.confirmAnswer;
}
async function settle() {
  for (let i = 0; i < 30; i += 1) await new Promise((r) => setImmediate(r));
}
function pending() { return requests.filter((r) => !r.answered); }

const context = {
  document, fetch, confirm, console, URLSearchParams,
  setTimeout, clearTimeout,
  setInterval: () => 0, clearInterval: () => {},
};
context.window = context;
vm.createContext(context);
vm.runInContext(source, context);
const page = (code) => vm.runInContext(code, context);

function emptyList(url) {
  if (url.includes("/api/history-schools")) return { schools: [] };
  return { items: [], total: 0, page: 1, page_size: 25, needs_review: 0 };
}
function row(id, via) {
  return { student_id: id, student_name: "Child " + id, school_name: "ABMPS",
           matched_via: via, reviewed_at: "", scanned_at: "2026-10-20 10:00:00" };
}

async function race() {
  await settle();
  for (const r of pending()) reply(r, 200, emptyList(r.url));
  await settle();

  page("searchInput.value = 'MOH'; loadTab();");
  page("searchInput.value = 'MOHSHA011021'; loadTab();");
  const [broad, narrow] = pending();
  reply(narrow, 200, { items: [row("MOHSHA011021", "exact")], total: 1, page: 1,
                       page_size: 25, needs_review: 0 });
  await settle();
  reply(broad, 200, { items: [row("MOHABC000001", "constrained_repair")], total: 312,
                      page: 1, page_size: 25, needs_review: 40 });
  await settle();

  // An older request that fails late must not wipe the newer table either.
  page("searchInput.value = 'MOHS'; loadTab();");
  page("searchInput.value = 'MOHSHA011021'; loadTab();");
  const [failing, fresh] = pending();
  reply(fresh, 200, { items: [row("MOHSHA011021", "exact")], total: 1, page: 1,
                      page_size: 25, needs_review: 0 });
  await settle();
  reply(failing, 500, { error: "Could not load scanned students." });
  await settle();

  return {
    urls: [broad.url, narrow.url],
    body: page("scannedBody.innerHTML"),
    pageLabel: page("pageLabel.textContent"),
    tabCount: String(page("scannedCount.textContent")),
    reviewBarHidden: page("reviewBar.hidden"),
    statusHidden: page("tableStatus.hidden"),
  };
}

async function review() {
  await settle();
  for (const r of pending()) reply(r, 200, emptyList(r.url));
  await settle();

  // School ABMPS, search "KID": page 1 has one orange row of 25, but the
  // server says seven rows in the whole filtered list need a check.
  page("histSchool.value = 'ABMPS'; searchInput.value = 'KID'; loadTab();");
  const [listing] = pending();
  reply(listing, 200, { items: [row("KIDAAA100001", "constrained_repair"),
                                row("KIDAAA100002", "exact")],
                        total: 40, page: 1, page_size: 25, needs_review: 7,
                        review_token: "tok-abmps-7" });
  await settle();
  const barCount = String(page("reviewCount.textContent"));
  const barHidden = page("reviewBar.hidden");
  const buttonHidden = page("reviewAllBtn.hidden");

  // The operator starts typing a new term but the list has not reloaded:
  // "Mark all" must still mean the list on screen.
  page("searchInput.value = 'KIDX';");

  state.confirmAnswer = false;
  page("reviewAllBtn").fire("click");
  await settle();
  const postsAfterCancel = requests.filter((r) => (r.options.method || "GET") === "POST").length;

  state.confirmAnswer = true;
  page("reviewAllBtn").fire("click");
  await settle();
  const post = requests.find((r) => (r.options.method || "GET") === "POST");
  return {
    barCount, barHidden, buttonHidden, confirms, postsAfterCancel,
    postUrl: post ? post.url : null,
    postBody: post ? JSON.parse(post.options.body) : null,
  };
}

async function everySchool() {
  await settle();
  for (const r of pending()) reply(r, 200, emptyList(r.url));
  await settle();

  // The defaults: Location All, School All, no search. 62 rows need a
  // check, most of them on pages nobody has opened.
  page("loadTab();");
  const [listing] = pending();
  reply(listing, 200, { items: [row("KIDAAA100001", "constrained_repair")],
                        total: 300, page: 1, page_size: 25, needs_review: 62,
                        review_token: "tok-all-62" });
  await settle();
  const shown = {
    barCount: String(page("reviewCount.textContent")),
    barHidden: page("reviewBar.hidden"),
    buttonHidden: page("reviewAllBtn.hidden"),
    hintHidden: page("document.getElementById('review-all-hint').hidden"),
  };
  page("reviewAllBtn").fire("click");
  await settle();
  const posts = requests.filter((r) => (r.options.method || "GET") === "POST").length;
  return { ...shown, confirms, posts };
}

async function preload() {
  await settle();
  const first = pending();
  const isPreload = (r) => r.url.includes("page_size=1&") || r.url.endsWith("page_size=1");
  const preloadScanned = first.find((r) => r.url.includes("/api/scanned") && isPreload(r));
  const preloadNotFound = first.find((r) => r.url.includes("/api/not-found") && isPreload(r));
  const list = first.find((r) => r.url.includes("/api/scanned") && !isPreload(r));
  for (const r of first.filter((r) => r.url.includes("/api/history-schools"))) {
    reply(r, 200, emptyList(r.url));
  }
  reply(list, 200, { items: [], total: 300, page: 1, page_size: 25, needs_review: 0 });
  await settle();

  // A search answers before the tab-count preload started at page load.
  page("searchInput.value = 'MOHSHA011021'; loadTab();");
  const search = pending().find((r) => r.url.includes("search="));
  reply(search, 200, { items: [row("MOHSHA011021", "exact")], total: 1, page: 1,
                       page_size: 25, needs_review: 0 });
  await settle();
  reply(preloadScanned, 200, { items: [], total: 300, page: 1, page_size: 1 });
  reply(preloadNotFound, 200, { items: [], total: 12, page: 1, page_size: 1 });
  await settle();
  return {
    preloadUrl: preloadScanned.url,
    scannedCount: String(page("scannedCount.textContent")),
    notFoundCount: String(page("notFoundCount.textContent")),
    pageLabel: page("pageLabel.textContent"),
  };
}

(async () => {
  const scenarios = { race, review, everySchool, preload };
  const result = await scenarios[scenario]();
  process.stdout.write(JSON.stringify(result) + "\n");
  process.exit(0);
})().catch((error) => {
  process.stderr.write(String(error && error.stack || error) + "\n");
  process.exit(1);
});
"""


def run_page_script(client: Any, tmp_path: Path, scenario: str) -> dict[str, Any]:
    """Run history.html's own script under Node through one scenario."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    html = client.get("/history").get_data(as_text=True)
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)
    page_script = next(s for s in scripts if "function loadTab" in s)
    script_path = tmp_path / "history_page.js"
    harness_path = tmp_path / "harness.js"
    script_path.write_text(page_script, encoding="utf-8")
    harness_path.write_text(HARNESS_JS, encoding="utf-8")
    result = subprocess.run(
        [node, str(harness_path), str(script_path), scenario],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_search_ignores_a_slower_answer_to_an_older_term(client: Any, tmp_path: Path) -> None:
    result = run_page_script(client, tmp_path, "race")

    assert "search=MOH&" in result["urls"][0] or result["urls"][0].endswith("search=MOH")
    assert "MOHSHA011021" in result["body"]
    assert "MOHABC000001" not in result["body"]
    assert "312" not in result["pageLabel"]
    assert result["tabCount"] == "1"
    assert result["reviewBarHidden"] is True
    assert result["statusHidden"] is True


def test_review_bar_shows_the_server_count_and_confirms_before_signing_off(
    client: Any, tmp_path: Path
) -> None:
    result = run_page_script(client, tmp_path, "review")

    assert result["barCount"] == "7"
    assert result["barHidden"] is False
    assert result["buttonHidden"] is False
    # Cancelled: asked with the number, and nothing was sent.
    assert result["postsAfterCancel"] == 0
    assert len(result["confirms"]) == 2
    assert "7 rows" in result["confirms"][0]
    assert "ABMPS" in result["confirms"][0]
    # Confirmed: the request carries the shown list's own filter, count and
    # the token for exactly the rows it counted.
    assert result["postUrl"].endswith("/api/scanned/review")
    assert result["postBody"] == {
        "all": True,
        "school": "ABMPS",
        "location": "",
        "search": "KID",
        "expected": 7,
        "token": "tok-abmps-7",
    }


def test_review_bar_offers_no_mark_all_across_every_school(
    client: Any, tmp_path: Path
) -> None:
    """#6: with the defaults the bar still counts, but the button is not offered."""
    result = run_page_script(client, tmp_path, "everySchool")

    assert result["barHidden"] is False
    assert result["barCount"] == "62"
    assert result["buttonHidden"] is True
    assert result["hintHidden"] is False
    assert result["confirms"] == []
    assert result["posts"] == 0


def test_tab_count_preload_does_not_overwrite_a_newer_search(
    client: Any, tmp_path: Path
) -> None:
    """#18: the page-load preload answered after a search and reset the count."""
    result = run_page_script(client, tmp_path, "preload")

    assert result["scannedCount"] == "1"
    assert result["notFoundCount"] == "12"
    assert "1 total" in result["pageLabel"]
    assert "review=0" in result["preloadUrl"]
