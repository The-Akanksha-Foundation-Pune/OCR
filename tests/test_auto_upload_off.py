"""Upload matched pages automatically, switched off on the server (config.AUTO_UPLOAD).

The app ships with it off: unless the server sets AUTO_UPLOAD=1, nothing is
confirmed without a person pressing Confirm. The page shows no switch and
never sends a document up by itself, and the server refuses an automatic
confirm with nothing saved whatever a page sends - so a page from before,
or one that says otherwise, cannot put anything in the database or Drive.

The database is the FakeDB, Drive the FakeDrive, OCR the stand-in readers
of tests/test_batch_queue.py; nothing leaves the test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient

from server import config
from tests import test_auto_upload as auto_tests
from tests import test_batch_queue as queue_tests
from tests.test_app import _run_page, needs_node
from tests.test_auto_upload import (
    _AUTO_JS,
    _auto_confirm,
    _document,
    _ready_exact,
    _refused_with_nothing_saved,
)
from tests.test_batch_queue import FakeDrive, _batches, _confirm, _writes

queue_dir = queue_tests.queue_dir
readers = queue_tests.readers
students = queue_tests.students
drive = queue_tests.drive
google = queue_tests.google
batch_codes = auto_tests.batch_codes


@pytest.fixture(autouse=True)
def auto_upload(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Automatic upload off, as the app ships."""
    monkeypatch.setattr(config, "AUTO_UPLOAD", False)
    return False


def test_it_is_off_unless_the_server_turns_it_on() -> None:
    assert config._env_flag("AUTO_UPLOAD") is not True or config.AUTO_UPLOAD is True
    assert config.auto_upload_on() is False


def test_a_form_that_would_go_up_by_itself_is_listed_for_a_person(client: FlaskClient) -> None:
    key = _ready_exact(client)

    [listed] = _batches(client)["batches"][0]["documents"]

    assert listed["key"] == key and listed["status"] == "ready"
    assert listed["auto_upload"] is False
    assert listed.get("auto_upload_pending", False) is False
    assert _document(client, key)["auto_upload"] is False


def test_an_automatic_confirm_is_refused_with_nothing_saved(
    client: FlaskClient, fake_db: Any, drive: FakeDrive
) -> None:
    key = _ready_exact(client)

    error = _refused_with_nothing_saved(_auto_confirm(client, key), fake_db, drive)

    assert "switched off" in error
    assert _document(client, key)["status"] == "ready"


def test_a_person_s_confirm_still_saves_it(
    google: FlaskClient, fake_db: Any, students: queue_tests.Students, drive: FakeDrive
) -> None:
    key = _ready_exact(google)

    response = _confirm(google, key=key)

    assert response.status_code == 200, response.get_json()
    assert _writes(fake_db) and drive.calls


def test_the_page_offers_no_switch_and_promises_nothing_goes_up_by_itself(client: FlaskClient) -> None:
    html = client.get("/").get_data(as_text=True)

    assert "const AUTO_UPLOAD = false;" in html
    assert 'id="auto-upload" disabled' in html and 'id="auto-upload" checked' not in html
    assert '<div class="school-label-spaced" hidden>' in html
    assert 'id="auto-upload-hint" hidden' in html
    assert "is confirmed and uploaded by itself" not in html
    assert "Nothing is saved until you do." in html


@needs_node
def test_the_page_never_confirms_by_itself_and_sends_every_scan_with_it_off(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Even when the browser remembers it on, and the server lists a document as eligible."""
    out = _run_page(client, tmp_path, _AUTO_JS + r"""
      const kept = new Map([["ocr.school", "School A"], ["ocr.location", "Pune"], ["ocr.autoUpload", "on"]]);
      const KEPT = {};
      const realStorage = makeStorage;
      makeStorage = (seed) => seed === KEPT
        ? { getItem: (k) => (kept.has(k) ? kept.get(k) : null), setItem: (k, v) => kept.set(k, String(v)), removeItem: (k) => kept.delete(k) }
        : realStorage(seed);
      const page = loadPage({ local: KEPT });
      await answer(page.calls, "/api/schools", { schools: [{ school: "School A", location: "Pune" }] });
      await answer(page.calls, "/api/inbox/status", { configured: false });
      await answerBatches(page, listing([one("p1", ASHA)]));
      await tick(50);
      const switchOn = page.byId["auto-upload"].checked;
      await uploadFile(page, "a.jpg");
      const sent = sentForm(await answerScan(page, ["p2"]));
      makeStorage = realStorage;
      return { switchOn, confirms: confirms(page).length, sent: sent.auto_upload };
    """)

    assert out["switchOn"] is False
    assert out["confirms"] == 0
    assert out["sent"] == "0"
