"""The safety net itself: nothing in the test run may reach production."""

from __future__ import annotations

import socket

import pymysql
import pytest

from tests.conftest import BlockedInTests


def test_real_database_is_unreachable() -> None:
    with pytest.raises(BlockedInTests):
        pymysql.connect(host="example.invalid")


def test_network_is_unreachable() -> None:
    sock = socket.socket()
    try:
        with pytest.raises(BlockedInTests):
            sock.connect(("8.8.8.8", 53))
    finally:
        sock.close()


def test_fake_db_records_statements(fake_db) -> None:
    from server.services import student_lookup

    conn = student_lookup.get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT 1 AS one")
    assert fake_db.executed(r"select 1")


def test_app_builds_and_gates_login(app) -> None:
    response = app.test_client().get("/")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
