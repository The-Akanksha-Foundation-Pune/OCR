"""Shared test fixtures.

The local .env points at the PRODUCTION database, so the most important job
of this file is to make sure no test can ever reach it - or Google Vision,
Google Drive, or any other host. Every test runs with those blocked; a test
that needs a database gets the FakeDB below instead.
"""

from __future__ import annotations

import re
import socket
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BlockedInTests(RuntimeError):
    """A test tried to reach something real (database, Google, the network)."""


# ---------------------------------------------------------------- safety net
@pytest.fixture(autouse=True)
def _block_the_outside_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No real database, no Google APIs, no outbound sockets - ever.

    Also keeps tests out of the real logs/ and data/queue/: test lines in
    logs/ocr_debug.log make a live problem hard to read, and a stray sweep
    of data/queue could move a person's real queued pages. Tests that use
    the queue_dir fixture point the queue elsewhere again.
    """
    from server.services import debug_log, scan_queue

    monkeypatch.setattr(debug_log, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(debug_log, "LOG_FILE", tmp_path / "logs" / "ocr_debug.log")
    monkeypatch.setattr(scan_queue, "QUEUE_DIR", tmp_path / "queue")

    import pymysql

    def refuse_db(*_args: Any, **_kwargs: Any) -> Any:
        raise BlockedInTests("tests must use the fake_db fixture, not the real database")

    monkeypatch.setattr(pymysql, "connect", refuse_db)
    monkeypatch.setattr(pymysql, "Connect", refuse_db, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost") and not str(host).startswith("/"):
            raise BlockedInTests(f"tests may not open network connections (to {host!r})")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # The model warm-up thread would load RapidOCR/Vision on every app build.
    from server.services import warmup

    monkeypatch.setattr(warmup, "warm_models_in_background", lambda: None)


# -------------------------------------------------------------------- fake DB
@dataclass
class FakeCursor:
    db: FakeDB
    rowcount: int = 0
    lastrowid: int = 0
    _rows: list[dict[str, Any]] = field(default_factory=list)

    def execute(self, sql: str, params: Any = None) -> int:
        normalised = re.sub(r"\s+", " ", sql).strip()
        self.db.statements.append((normalised, params))
        rows, rowcount = self.db.handler(normalised, params)
        self._rows = list(rows or [])
        self.rowcount = rowcount if rowcount is not None else len(self._rows)
        self.lastrowid = self.db.next_id
        self.db.next_id += 1
        return self.rowcount

    def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


@dataclass
class FakeConnection:
    db: FakeDB

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.db)

    def commit(self) -> None:
        self.db.commits += 1

    def rollback(self) -> None:
        self.db.rollbacks += 1

    def begin(self) -> None:
        return None

    def close(self) -> None:
        return None


Handler = Callable[[str, Any], tuple[list[dict[str, Any]] | None, int | None]]


@dataclass
class FakeDB:
    """Records every statement and answers through `handler(sql, params)`.

    `handler` returns (rows, rowcount); rowcount None means len(rows). The
    default answers every statement with no rows.
    """

    handler: Handler = lambda _sql, _params: ([], 0)
    statements: list[tuple[str, Any]] = field(default_factory=list)
    commits: int = 0
    rollbacks: int = 0
    next_id: int = 1000

    def connect(self) -> FakeConnection:
        return FakeConnection(self)

    def executed(self, pattern: str) -> list[tuple[str, Any]]:
        """Statements whose SQL matches `pattern` (a regex, case-insensitive)."""
        return [s for s in self.statements if re.search(pattern, s[0], re.IGNORECASE)]


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDB]:
    """Point every module's get_connection at an in-memory fake."""
    db = FakeDB()
    import server.services.student_lookup  # noqa: F401 - make sure it is loaded

    for name, module in list(sys.modules.items()):
        if name.startswith(("server", "tools")) and hasattr(module, "get_connection"):
            monkeypatch.setattr(module, "get_connection", db.connect)
    # Schema set-up runs once per process; treat it as done so tests only see
    # the statements under test.
    for mod_name in ("server.services.history_store", "server.services.batch_store"):
        module = sys.modules.get(mod_name)
        if module is not None and hasattr(module, "_initialized"):
            monkeypatch.setattr(module, "_initialized", True)
    yield db


# ------------------------------------------------------------------ Flask app
@pytest.fixture
def app() -> Any:
    from server import create_app

    flask_app = create_app()
    flask_app.config.update(TESTING=True, SECRET_KEY="test-secret")
    flask_app.secret_key = "test-secret"
    return flask_app


@pytest.fixture
def client(app: Any) -> Any:
    """A test client already signed in as an allowlisted staff member."""
    test_client = app.test_client()
    with test_client.session_transaction() as session:
        session["user"] = {"email": "staff@akanksha.org", "name": "Staff", "picture": ""}
    return test_client


@contextmanager
def signed_out(app: Any) -> Iterator[Any]:
    yield app.test_client()
