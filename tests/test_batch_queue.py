"""The scan queue: scanning stores, OCR reads in the background, a person confirms.

Covered here:

  - POST /api/scans stores and queues a file's pages and answers 202 without
    reading anything; a multi-page TIFF or PDF becomes one page each, every
    page normalised (greyscale, upright, at most 300 dpi).
  - Claiming a page is exclusive, even with two readers racing, and a claim
    whose reader died is handed back.
  - The worker, with every recogniser a stand-in, writes result.json and
    never writes to the database; an outage leaves the page queued.
  - Pages with the same batch code are one document; parts that disagree are
    a conflict; a page with no code is unsorted.
  - A page's image is served to its owner only.
  - Codes and Student IDs typed by hand.
  - Confirm: the only write. The happy path (scanned_at, ocr_scan_log, one
    PDF of both parts in the right Drive folders), and every way it refuses
    with nothing saved; a Drive failure leaves it confirmed with a retry.
  - The cross-site write guard still covers the new routes.

The database is tests/conftest.py's FakeDB; Google Drive is the FakeDrive
below; OCR, Vision and RapidOCR are never run. Nothing leaves the test.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import fitz
import numpy as np
import pymysql
import pytest
from flask import Flask
from flask.testing import FlaskClient
from PIL import Image, ImageDraw, features

from server.services import (
    drive_store,
    ocr_worker,
    scan_queue,
    student_lookup,
)
from server.services.batch_code import BatchCode, CodeReading
from server.services.extract_student_id import ExtractResult

OWNER = "staff@akanksha.org"
SCHOOL = "ABMPS"
AATIF = {
    "student_id": "AATKHA200618",
    "student_name": "Aatif Khan",
    "school_name": "ABMPS",
    "grade_name": "Grade 5",
    "division_name": "A",
    "academic_year": "2026-2027",
    "gender": "M",
    "status": "Active - Enrolled",
    "scanned_at": None,
}
MOHSHA = dict(AATIF, student_id="MOHSHA011021", student_name="Mohammad Shaikh")

needs_libtiff = pytest.mark.skipif(
    not features.check("libtiff"), reason="this Pillow was built without libtiff"
)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def queue_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The queue in the test's own folder; logs there too."""
    import server.services.debug_log as debug_module

    folder = tmp_path / "queue"
    monkeypatch.setattr(scan_queue, "QUEUE_DIR", folder)
    monkeypatch.setattr(debug_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(debug_module, "LOG_FILE", tmp_path / "logs" / "ocr_debug.log")
    # Fuzzy repair caches every ID it fetched; an earlier test's cache must
    # not turn a not-found here into a match.
    monkeypatch.setattr(student_lookup, "_id_cache", {})
    monkeypatch.setattr(student_lookup, "_id_cache_at", {})
    # The stale-claim sweep runs at most once a minute per process.
    monkeypatch.setattr(ocr_worker, "_last_reclaim", time.monotonic())
    return folder


@dataclass
class Students:
    """What the fake database knows, and an outage switch."""

    rows: dict[str, dict[str, Any]] = field(default_factory=lambda: {"AATKHA200618": dict(AATIF)})
    down: bool = False


@pytest.fixture
def students(fake_db: Any) -> Students:
    known = Students()

    def handler(sql: str, params: Any) -> tuple[list[dict[str, Any]] | None, int | None]:
        if known.down:
            raise pymysql.err.OperationalError(2003, "Can't connect to MySQL server")
        if sql.startswith("SELECT student_id, student_name") and "WHERE UPPER(TRIM(student_id)) = UPPER(%s)" in sql:
            row = known.rows.get(str(params[0]).upper())
            return ([dict(row)] if row else []), None
        if sql.startswith("UPDATE active_student_data SET scanned_at"):
            return [], 1 if str(params[1]).upper() in known.rows else 0
        if sql.startswith("SELECT scanned_at FROM"):
            return [{"scanned_at": None}], None
        if sql.startswith("UPDATE ocr_scan_log"):
            return [], 1
        return [], 0

    fake_db.handler = handler
    return known


def _writes(fake_db: Any) -> list[tuple[str, Any]]:
    return fake_db.executed(r"^(UPDATE|INSERT|DELETE|REPLACE|ALTER|CREATE)\b")


@dataclass
class Readers:
    """Stand-ins for the recognisers. Each read takes the next answer."""

    codes: list[str | None] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    code_calls: int = 0
    id_calls: int = 0
    code_rule: Callable[[np.ndarray], str | None] | None = None
    vision_allowed: list[bool] = field(default_factory=list)


@pytest.fixture
def readers(monkeypatch: pytest.MonkeyPatch) -> Readers:
    fake = Readers()

    def read_code(page: np.ndarray, *, allow_vision: bool = True) -> CodeReading:
        fake.code_calls += 1
        fake.vision_allowed.append(allow_vision)
        if fake.code_rule is not None:
            text = fake.code_rule(page)
        else:
            text = fake.codes.pop(0) if fake.codes else None
        if not text:
            return CodeReading(code=None)
        number = re.fullmatch(r"B(\d+)-(\d+)", text)
        assert number
        return CodeReading(
            code=BatchCode(int(number[1]), int(number[2])), confidence=0.9, raw=text, region="top-right"
        )

    def read_student_id(image_bgr: np.ndarray, source: str) -> ExtractResult:
        fake.id_calls += 1
        assert image_bgr.ndim == 3
        student_id = fake.ids.pop(0) if fake.ids else ""
        return ExtractResult(
            student_id=student_id,
            confidence=0.9 if student_id else 0.0,
            form_language="english" if student_id else "unknown",
            source=source,
            error=None if student_id else "No Student ID found on this page.",
        )

    monkeypatch.setattr(ocr_worker, "read_code", read_code)
    monkeypatch.setattr(ocr_worker, "read_student_id", read_student_id)
    monkeypatch.setattr(ocr_worker, "identify", lambda _image: None)
    return fake


class FakeResponse:
    def __init__(self, status: int, body: dict[str, Any] | None = None) -> None:
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self) -> dict[str, Any]:
        return self._body


class FakeDrive:
    """Google Drive's files API, as much of it as the upload uses."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.fail_uploads = False
        # Seconds each request takes, as a real round trip to Google does.
        self.latency = 0.0
        self._next = 0
        self._lock = threading.Lock()

    def _id(self) -> str:
        with self._lock:
            self._next += 1
            return f"id{self._next}"

    def folder(self, name: str, parent: str) -> str:
        new_id = self._id()
        self.files[new_id] = {"id": new_id, "name": name, "parents": [parent], "folder": True}
        return new_id

    def get(self, url: str, params: dict[str, Any], **_kwargs: Any) -> FakeResponse:
        self.calls.append("list")
        if self.latency:
            time.sleep(self.latency)
        query = params["q"]
        parent = re.search(r"'((?:\\.|[^'])*)' in parents", query)
        name = re.search(r"name='((?:\\.|[^'])*)'", query)
        found = []
        for entry in self.files.values():
            if parent and _unescape(parent[1]) not in entry["parents"]:
                continue
            if name and entry["name"] != _unescape(name[1]):
                continue
            if "mimeType='application/vnd.google-apps.folder'" in query and not entry["folder"]:
                continue
            found.append({"id": entry["id"], "name": entry["name"]})
        # Drive promises no order unless asked for one; newest first here,
        # so a caller that takes the first match without asking gets that.
        if params.get("orderBy") != "createdTime":
            found.reverse()
        return FakeResponse(200, {"files": found})

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        if self.latency:
            time.sleep(self.latency)
        if "upload" in url:
            self.calls.append("upload")
            if self.fail_uploads:
                return FakeResponse(503, {"error": "backendError"})
            boundary = kwargs["headers"]["Content-Type"].split("boundary=")[1]
            parts = kwargs["data"].split(f"--{boundary}".encode())
            metadata = json.loads(parts[1].split(b"\r\n\r\n", 1)[1])
            pdf = parts[2].split(b"\r\n\r\n", 1)[1][: -len(b"\r\n")]
            new_id = self._id()
            self.files[new_id] = {
                "id": new_id,
                "name": metadata["name"],
                "parents": metadata["parents"],
                "folder": False,
                "data": pdf,
            }
            return FakeResponse(200, {"id": new_id, "name": metadata["name"],
                                      "webViewLink": f"https://drive.example/{new_id}"})
        self.calls.append("create-folder")
        body = kwargs["json"]
        return FakeResponse(200, {"id": self.folder(body["name"], body["parents"][0])})

    def path_of(self, file_id: str) -> list[str]:
        names = []
        current = self.files[file_id]
        while True:
            parent = current["parents"][0]
            if parent == "root":
                return list(reversed(names))
            current = self.files[parent]
            names.append(current["name"])

    def uploaded(self) -> list[dict[str, Any]]:
        return [entry for entry in self.files.values() if not entry["folder"]]


def _unescape(text: str) -> str:
    return text.replace("\\'", "'").replace("\\\\", "\\")


@pytest.fixture
def drive(monkeypatch: pytest.MonkeyPatch) -> FakeDrive:
    fake = FakeDrive()
    monkeypatch.setattr(drive_store, "requests", SimpleNamespace(
        get=fake.get, post=fake.post, RequestException=OSError,
    ))
    return fake


@pytest.fixture
def google(client: FlaskClient) -> FlaskClient:
    """The signed-in staff member, signed in with Google (so a Drive token)."""
    with client.session_transaction() as session:
        session["google_token"] = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_at": time.time() + 3600,
        }
    return client


def _other_person(app: Flask) -> FlaskClient:
    other = app.test_client()
    with other.session_transaction() as session:
        session["user"] = {"email": "someone.else@akanksha.org", "name": "Else", "picture": ""}
    return other


# ------------------------------------------------------------------ makers
def _form_page(size: tuple[int, int] = (620, 874), mode: str = "L", mark: str = "top-right") -> Image.Image:
    """A page with enough ink not to be blank, and a mark in one corner."""
    width, height = size
    paper, ink = (1, 0) if mode == "1" else (255, 0)
    page = Image.new(mode, size, paper)
    draw = ImageDraw.Draw(page)
    draw.rectangle((width // 10, height // 4, width * 9 // 10, height // 2), fill=ink)
    if mark == "top-right":
        draw.rectangle((int(width * 0.8), 5, int(width * 0.95), int(height * 0.06)), fill=ink)
    elif mark == "bottom-left":
        draw.rectangle((int(width * 0.05), int(height * 0.94), int(width * 0.2), height - 5), fill=ink)
    return page


def _tiff(pages: list[Image.Image], **options: Any) -> bytes:
    buffer = io.BytesIO()
    pages[0].save(buffer, format="TIFF", save_all=True, append_images=pages[1:], **options)
    return buffer.getvalue()


def _pdf(pages: int) -> bytes:
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(60, 200, 535, 420), color=(0, 0, 0), fill=(0, 0, 0))
        page.insert_text((72, 72), f"page {index + 1}")
    data = document.tobytes()
    document.close()
    return data


def _jpeg(image: Image.Image, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", **options)
    return buffer.getvalue()


def _send(client: FlaskClient, name: str, data: bytes, **form: str) -> Any:
    fields = {"file": (io.BytesIO(data), name), "school": SCHOOL, "location": "Mumbai"}
    fields.update(form)
    return client.post("/api/scans", data=fields, content_type="multipart/form-data")


def _queue(client: FlaskClient, name: str = "Scan 1.tiff", data: bytes | None = None) -> list[str]:
    response = _send(client, name, data or _tiff([_form_page()]))
    assert response.status_code == 202, response.get_json()
    return [page["page_id"] for page in response.get_json()["pages"]]


def _read(page_id: str, code: str | None, student_id: str = "", matched: dict[str, Any] | None = None) -> None:
    """Mark a queued page read, as the worker would, with this reading."""
    (scan_queue.queue_root() / "_pending" / page_id).unlink()
    scan_queue.save_result(page_id, scan_queue.PageResult(code=code, student_id=student_id, matched_student=matched))
    meta = scan_queue.load_meta(page_id)
    assert meta is not None
    meta.status = scan_queue.STATUS_READ
    scan_queue.save_meta(meta)


def _batches(client: FlaskClient) -> dict[str, Any]:
    response = client.get("/api/batches")
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _confirm(
    client: FlaskClient,
    key: str = "B1-1",
    student_id: str = "AATKHA200618",
    page_ids: list[str] | None = None,
) -> Any:
    """Press Confirm and upload, sending the parts the page was showing.

    Without `page_ids`, the parts are the ones the document has right now,
    as the page would have them from its last poll.
    """
    if page_ids is None:
        shown = client.get(f"/api/documents/{key}").get_json() or {}
        page_ids = [page["page_id"] for page in (shown.get("document") or {}).get("pages", [])]
    return client.post(
        f"/api/documents/{key}/confirm", json={"student_id": student_id, "page_ids": page_ids}
    )


def _ready_document(client: FlaskClient) -> list[str]:
    """B1-1 in two parts, the first read and matched, the second read with its code."""
    first = _queue(client, "Scan 1.tiff")[0]
    second = _queue(client, "Scan 2.tiff")[0]
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))
    _read(second, "B1-1")
    return [first, second]


# ================================================================== intake
def test_a_scan_is_stored_and_queued_without_being_read(
    client: FlaskClient, fake_db: Any, readers: Readers, queue_dir: Path
) -> None:
    response = _send(client, "Scan 7.tiff", _tiff([_form_page()]))

    assert response.status_code == 202
    body = response.get_json()
    assert body["page_count"] == 1 and body["source_name"] == "Scan 7.tiff"
    [page] = body["pages"]
    folder = queue_dir / page["page_id"]
    assert sorted(p.name for p in folder.iterdir()) == ["meta.json", "original.tiff", "page.jpg"]
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["status"] == "queued" and meta["attempts"] == 0
    assert meta["owner_email"] == OWNER and meta["school"] == SCHOOL and meta["location"] == "Mumbai"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", meta["created_at"])
    assert (queue_dir / "_pending" / page["page_id"]).exists()
    # Nothing read, nothing looked up.
    assert readers.code_calls == readers.id_calls == 0
    assert fake_db.statements == []


@needs_libtiff
def test_a_multi_page_tiff_becomes_one_queued_page_each(client: FlaskClient, queue_dir: Path) -> None:
    pages = [_form_page((620 + 10 * i, 874), mode="1") for i in range(3)]

    response = _send(client, "stack.tif", _tiff(pages, compression="group4", dpi=(150, 150)))

    assert response.status_code == 202
    body = response.get_json()
    assert [p["source_page"] for p in body["pages"]] == [1, 2, 3]
    for index, page in enumerate(body["pages"]):
        folder = queue_dir / page["page_id"]
        with Image.open(folder / "original.tif") as original:
            assert getattr(original, "n_frames", 1) == 1
            assert original.size == (620 + 10 * index, 874)
        grey = cv2.imread(str(folder / "page.jpg"), cv2.IMREAD_UNCHANGED)
        assert grey.ndim == 2 and grey.shape == (874, 620 + 10 * index)
        meta = json.loads((folder / "meta.json").read_text())
        assert (meta["source_page"], meta["source_pages"], meta["dpi"]) == (index + 1, 3, 150.0)
    # Sorted by name, the ids are in page order: the queue reads them in order.
    ids = [p["page_id"] for p in body["pages"]]
    assert sorted(ids) == ids


def test_a_multi_page_pdf_becomes_one_queued_page_each(client: FlaskClient, queue_dir: Path) -> None:
    response = _send(client, "forms.pdf", _pdf(2))

    assert response.status_code == 202
    ids = [p["page_id"] for p in response.get_json()["pages"]]
    assert len(ids) == 2
    for page_id in ids:
        with fitz.open(queue_dir / page_id / "original.pdf") as original:
            assert original.page_count == 1
        grey = cv2.imread(str(queue_dir / page_id / "page.jpg"), cv2.IMREAD_UNCHANGED)
        # A4 at the 200 dpi the reader was tuned on, one channel.
        assert grey.ndim == 2 and grey.shape == (2339, 1653)


@needs_libtiff
def test_a_600_dpi_scan_is_stored_at_300_dpi_in_grey(client: FlaskClient, queue_dir: Path) -> None:
    big = _form_page((4960, 7016), mode="1")

    [page_id] = _queue(client, "fine.tiff", _tiff([big], compression="group4", dpi=(600, 600)))

    grey = cv2.imread(str(queue_dir / page_id / "page.jpg"), cv2.IMREAD_UNCHANGED)
    assert grey.ndim == 2
    assert grey.shape == (3508, 2480)
    assert json.loads((queue_dir / page_id / "meta.json").read_text())["dpi"] == 300.0


def test_a_huge_photo_is_shrunk_to_the_long_side_limit(client: FlaskClient, queue_dir: Path) -> None:
    photo = _form_page((8000, 6000))

    [page_id] = _queue(client, "photo.jpg", _jpeg(photo))

    grey = cv2.imread(str(queue_dir / page_id / "page.jpg"), cv2.IMREAD_UNCHANGED)
    assert grey.ndim == 2 and max(grey.shape) <= 3600


def test_a_photo_is_stored_the_way_up_its_exif_says(client: FlaskClient, queue_dir: Path) -> None:
    stored = Image.new("L", (300, 100), 255)
    ImageDraw.Draw(stored).rectangle((0, 0, 29, 29), fill=0)
    exif = Image.Exif()
    exif[274] = 6  # turn a quarter clockwise to view

    [page_id] = _queue(client, "phone.jpg", _jpeg(stored, exif=exif.tobytes()))

    grey = cv2.imread(str(queue_dir / page_id / "page.jpg"), cv2.IMREAD_UNCHANGED)
    assert grey.shape == (300, 100)
    assert grey[5, 90] < 60 and grey[5, 5] > 200  # the mark is now top right


def test_a_damaged_file_is_refused_and_leaves_nothing_queued(client: FlaskClient, queue_dir: Path) -> None:
    good = _tiff([_form_page(), _form_page()])

    response = _send(client, "stack.tif", good[:-200])

    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Could not decode")
    leftovers = [p for p in queue_dir.rglob("*") if p.is_file()] if queue_dir.exists() else []
    assert leftovers == []


@pytest.mark.parametrize("data", [b"not a pdf at all", b"%PDF-1.4\n%%EOF", _pdf(2)[:300]])
def test_a_damaged_pdf_is_refused_with_a_reason(client: FlaskClient, queue_dir: Path, data: bytes) -> None:
    response = _send(client, "forms.pdf", data)

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json()["error"].startswith("Could not decode")
    assert not queue_dir.exists() or not any(queue_dir.rglob("meta.json"))


@pytest.mark.parametrize("name", ["scan.bmp", "scan.gif", "scan.heic", "notes.txt"])
def test_other_file_types_are_refused(client: FlaskClient, queue_dir: Path, name: str) -> None:
    response = _send(client, name, b"not a scan")

    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Unsupported file type")
    assert not queue_dir.exists() or not any(queue_dir.rglob("meta.json"))


def test_a_file_only_named_tif_is_refused(client: FlaskClient) -> None:
    response = _send(client, "scan.tif", _jpeg(_form_page()))

    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Not a TIFF image")


def test_scanning_needs_a_signed_in_person(app: Flask, queue_dir: Path) -> None:
    response = app.test_client().post(
        "/api/scans", data={"file": (io.BytesIO(b"x"), "a.jpg")}, content_type="multipart/form-data"
    )

    assert response.status_code == 401
    assert not queue_dir.exists()


# ================================================================== claiming
def _make_pending(count: int) -> list[str]:
    pending = scan_queue.queue_root() / "_pending"
    pending.mkdir(parents=True)
    ids = [scan_queue.new_page_id(1_700_000_000_000 + i, 1) for i in range(count)]
    for page_id in ids:
        (pending / page_id).touch()
    return ids


def test_two_readers_never_claim_the_same_page() -> None:
    ids = _make_pending(300)
    claimed: list[list[str]] = [[], []]
    start = threading.Barrier(2)

    def reader(slot: int) -> None:
        start.wait()
        while (claim := scan_queue.claim_next()) is not None:
            claimed[slot].append(claim.page_id)

    threads = [threading.Thread(target=reader, args=(slot,)) for slot in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    everything = claimed[0] + claimed[1]
    assert len(everything) == len(set(everything)) == 300
    assert set(everything) == set(ids)


def test_the_oldest_page_is_claimed_first() -> None:
    ids = _make_pending(3)

    assert [scan_queue.claim_next().page_id for _ in ids] == ids  # type: ignore[union-attr]
    assert scan_queue.claim_next() is None


def test_a_stale_claim_is_handed_back_and_its_reader_loses_it() -> None:
    [page_id] = _make_pending(1)
    taken = 1_800_000_000_000
    claim = scan_queue.claim_next(now_ms=taken)
    assert claim is not None

    # Nine minutes on, the reader may still be working: left alone.
    assert scan_queue.reclaim_stale(now_ms=taken + 9 * 60_000) == []
    assert scan_queue.claim_next() is None
    # Past ten, it has died: queued again, for exactly one sweep.
    assert scan_queue.reclaim_stale(now_ms=taken + 11 * 60_000) == [page_id]
    assert scan_queue.reclaim_stale(now_ms=taken + 11 * 60_000) == []

    again = scan_queue.claim_next()
    assert again is not None and again.page_id == page_id
    # The old reader, waking up, can neither finish nor release the page.
    assert scan_queue.finish_claim(claim) is False
    assert scan_queue.release_claim(claim) is False
    assert scan_queue.still_held(again) and scan_queue.finish_claim(again)


# ================================================================== the worker
def test_the_worker_reads_a_page_and_writes_nothing_to_the_database(
    client: FlaskClient, fake_db: Any, students: Students, readers: Readers, queue_dir: Path
) -> None:
    [page_id] = _queue(client)
    readers.codes = ["B1-1"]
    readers.ids = ["AATKHA200618"]

    assert ocr_worker.process_one() == ocr_worker.READ

    result = json.loads((queue_dir / page_id / "result.json").read_text())
    assert result["code"] == "B1-1"
    assert result["code_region"] == "top-right"
    assert result["student_id"] == "AATKHA200618"
    assert result["student_raw_shape_ok"] is True
    assert result["candidates"] == ["AATKHA200618"]
    assert result["matched_student"]["student_name"] == "Aatif Khan"
    assert result["orientation"] == 0
    meta = json.loads((queue_dir / page_id / "meta.json").read_text())
    assert meta["status"] == "read" and meta["attempts"] == 1
    assert not (queue_dir / "_pending" / page_id).exists()
    assert list((queue_dir / "_reading").iterdir()) == []
    # Looked up, scoped to the page's school, and never written.
    lookups = fake_db.executed(r"FROM active_student_data")
    assert lookups and all("school_name" in sql for sql, _ in lookups)
    assert _writes(fake_db) == []
    assert ocr_worker.process_one() == ocr_worker.IDLE


def test_only_an_id_with_a_student_ids_shape_is_kept(
    client: FlaskClient, students: Students, readers: Readers, queue_dir: Path
) -> None:
    [page_id] = _queue(client)
    readers.ids = ["9876543210"]  # a phone number

    ocr_worker.process_one()

    result = json.loads((queue_dir / page_id / "result.json").read_text())
    assert result["student_id"] == "" and result["matched_student"] is None


def test_an_outage_leaves_the_page_queued_and_the_retry_only_looks_it_up(
    client: FlaskClient, fake_db: Any, students: Students, readers: Readers, queue_dir: Path
) -> None:
    [page_id] = _queue(client)
    readers.codes = ["B1-1"]
    readers.ids = ["AATKHA200618"]
    students.down = True

    assert ocr_worker.process_one() == ocr_worker.UNAVAILABLE

    assert (queue_dir / "_pending" / page_id).exists()
    assert not (queue_dir / page_id / "result.json").exists()
    meta = json.loads((queue_dir / page_id / "meta.json").read_text())
    assert meta["status"] == "queued" and meta["attempts"] == 0  # an outage is not a try
    assert ocr_worker.run_pass(budget_seconds=5) == 0  # stops at the outage, no spin

    students.down = False
    assert ocr_worker.process_one() == ocr_worker.READ
    assert readers.id_calls == 1 and readers.code_calls == 1  # OCR ran once in all
    assert json.loads((queue_dir / page_id / "result.json").read_text())["matched_student"]
    assert _writes(fake_db) == []


def test_a_page_that_keeps_failing_is_marked_failed_after_three_tries(
    client: FlaskClient, students: Students, readers: Readers, monkeypatch: pytest.MonkeyPatch,
    queue_dir: Path,
) -> None:
    [page_id] = _queue(client)

    def broken(_image: np.ndarray, _source: str) -> ExtractResult:
        raise MemoryError("out of memory")

    monkeypatch.setattr(ocr_worker, "read_student_id", broken)

    outcomes = [ocr_worker.process_one() for _ in range(4)]

    assert outcomes == [ocr_worker.REQUEUED, ocr_worker.REQUEUED, ocr_worker.FAILED, ocr_worker.IDLE]
    meta = json.loads((queue_dir / page_id / "meta.json").read_text())
    assert meta["status"] == "failed" and meta["attempts"] == 3
    assert "MemoryError" in meta["error"]


def test_a_page_whose_reader_died_three_times_is_failed_not_read_again(
    client: FlaskClient, students: Students, readers: Readers, queue_dir: Path
) -> None:
    [page_id] = _queue(client)
    meta = scan_queue.load_meta(page_id)
    assert meta is not None
    meta.attempts = 3
    scan_queue.save_meta(meta)

    assert ocr_worker.process_one() == ocr_worker.FAILED
    assert readers.id_calls == 0


def test_an_upside_down_page_is_turned_round(
    client: FlaskClient, students: Students, readers: Readers, queue_dir: Path
) -> None:
    # Scanned upside down: the corner mark is at the bottom left.
    [page_id] = _queue(client, data=_tiff([_form_page(mark="bottom-left")]))

    def code_if_marked_top_right(page: np.ndarray) -> str | None:
        height, width = page.shape[:2]
        corner = page[: int(height * 0.09), int(width * 0.72) :]
        return "B2-3" if float(np.mean(corner < 128)) > 0.05 else None

    readers.code_rule = code_if_marked_top_right

    ocr_worker.process_one()

    result = json.loads((queue_dir / page_id / "result.json").read_text())
    assert result["orientation"] == 180 and result["code"] == "B2-3"
    stored = cv2.imread(str(queue_dir / page_id / "page.jpg"), cv2.IMREAD_GRAYSCALE)
    assert code_if_marked_top_right(stored) == "B2-3"  # page.jpg is upright now


def test_the_layout_decides_which_way_up(
    client: FlaskClient, students: Students, readers: Readers, monkeypatch: pytest.MonkeyPatch,
    queue_dir: Path,
) -> None:
    [page_id] = _queue(client, data=_tiff([_form_page(mark="bottom-left")]))
    seen: list[np.ndarray] = []

    def identify(image: np.ndarray) -> Any:
        upright = float(np.mean(image[: image.shape[0] // 10, image.shape[1] * 3 // 4 :] < 128)) > 0.05
        return SimpleNamespace(error=0.001 if upright else 0.015)

    def code(page: np.ndarray) -> str:
        seen.append(page)
        return "B1-2"

    monkeypatch.setattr(ocr_worker, "identify", identify)
    readers.code_rule = code

    ocr_worker.process_one()

    assert json.loads((queue_dir / page_id / "result.json").read_text())["orientation"] == 180
    assert len(seen) == 1  # read once, the right way up


def test_a_blank_page_is_not_sent_to_the_student_id_reader(
    client: FlaskClient, students: Students, readers: Readers, queue_dir: Path
) -> None:
    [page_id] = _queue(client, data=_tiff([Image.new("L", (620, 874), 255)]))

    ocr_worker.process_one()

    result = json.loads((queue_dir / page_id / "result.json").read_text())
    assert result["blank"] is True and result["student_id"] == ""
    assert readers.id_calls == 0


def test_run_pass_reads_until_the_queue_is_empty(
    client: FlaskClient, students: Students, readers: Readers
) -> None:
    for name in ("a.tiff", "b.tiff", "c.tiff"):
        _queue(client, name)
    ticks: list[int] = []

    assert ocr_worker.run_pass(budget_seconds=30, progress=lambda: ticks.append(1)) == 3
    assert len(ticks) == 3


def test_the_worker_thread_follows_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_worker, "OCR_WORKER_IN_PROCESS", None)
    assert ocr_worker.should_run_in_process(True) is True
    assert ocr_worker.should_run_in_process(False) is False
    monkeypatch.setattr(ocr_worker, "OCR_WORKER_IN_PROCESS", False)
    assert ocr_worker.should_run_in_process(True) is False
    monkeypatch.setattr(ocr_worker, "OCR_WORKER_IN_PROCESS", True)
    assert ocr_worker.should_run_in_process(False) is True


def test_the_app_starts_one_worker_only_when_asked(app: Flask, client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[int] = []
    monkeypatch.setattr(ocr_worker, "in_process_worker_running", lambda: bool(started))
    monkeypatch.setattr(ocr_worker, "start_in_process_worker", lambda: started.append(1) or True)

    client.get("/api/batches")
    assert started == []  # not debug, not asked: production reads in the watcher

    monkeypatch.setattr(ocr_worker, "OCR_WORKER_IN_PROCESS", True)
    client.get("/api/batches")
    client.get("/api/batches")
    assert started == [1]


def test_the_worker_thread_runs_and_stops(students: Students, readers: Readers) -> None:
    stop = threading.Event()
    thread = threading.Thread(target=ocr_worker.run_forever, args=(stop,), kwargs={"idle_seconds": 0.01})
    thread.start()
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_the_inbox_watcher_reads_the_queue_and_survives_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server import inbox_watcher

    passes: list[float] = []
    monkeypatch.setattr(ocr_worker, "run_pass", lambda budget_seconds, progress=None: passes.append(budget_seconds) or 2)
    assert inbox_watcher.read_scan_queue() == 2
    assert passes == [inbox_watcher.QUEUE_PASS_SECONDS]

    def broken(**_kwargs: Any) -> int:
        raise OSError("disk gone")

    monkeypatch.setattr(ocr_worker, "run_pass", broken)
    assert inbox_watcher.read_scan_queue() == 0


# ================================================================== documents
def test_a_repeated_code_is_one_document_in_two_parts(client: FlaskClient, fake_db: Any) -> None:
    first, second = _ready_document(client)
    loose = _queue(client, "Scan 3.tiff")[0]
    _read(loose, None, "MOHSHA011021", None)

    body = _batches(client)

    [batch] = body["batches"]
    assert batch["batch"] == 1 and batch["label"] == "B1"
    [doc] = batch["documents"]
    assert doc["key"] == "B1-1" and doc["parts"] == 2
    assert [p["page_id"] for p in doc["pages"]] == [first, second]
    assert doc["status"] == "ready" and doc["can_confirm"] is True
    assert doc["student_id"] == "AATKHA200618"
    assert doc["matched_student"]["grade_name"] == "Grade 5"
    assert batch["counts"]["ready"] == 1 and batch["counts"]["pages"] == 2
    # The signature page has no ID of its own, and needs none: part 1 has it.
    assert [p["needs_person"] for p in doc["pages"]] == [False, False]
    assert [p["page_id"] for p in body["unsorted"]] == [loose]
    assert body["unsorted"][0]["needs_person"] is True  # no code: show its image
    assert body["queue"] == {"queued": 0, "reading": 0, "read": 3, "failed": 0}
    assert [p["page_id"] for p in body["pages"]] == [loose, second, first]  # newest first
    assert doc["pages"][0]["thumb_url"] == f"/api/pages/{first}/image?size=thumb"
    assert fake_db.statements == []


def test_parts_that_match_different_students_are_a_conflict(client: FlaskClient) -> None:
    first = _queue(client, "a.tiff")[0]
    second = _queue(client, "b.tiff")[0]
    _read(first, "B1-2", "AATKHA200618", dict(AATIF))
    _read(second, "B1-2", "MOHSHA011021", dict(MOHSHA))

    [doc] = _batches(client)["batches"][0]["documents"]

    assert doc["status"] == "needs_attention"
    assert "conflict" in doc["attention"]
    assert "AATKHA200618" in doc["message"] and "MOHSHA011021" in doc["message"]


def test_a_part_that_matched_beats_a_misread_part(client: FlaskClient) -> None:
    first = _queue(client, "a.tiff")[0]
    second = _queue(client, "b.tiff")[0]
    _read(first, "B1-3", "AATKHA200610", None)  # misread, no match
    _read(second, "B1-3", "AATKHA200618", dict(AATIF))

    [doc] = _batches(client)["batches"][0]["documents"]

    assert doc["status"] == "ready" and doc["student_id"] == "AATKHA200618"


@pytest.mark.parametrize(
    ("student_id", "matched", "reason"),
    [("", None, "no_id"), ("AATKHA200619", None, "no_match")],
)
def test_a_document_without_a_matched_id_needs_attention(
    client: FlaskClient, student_id: str, matched: dict[str, Any] | None, reason: str
) -> None:
    [page_id] = _queue(client)
    _read(page_id, "B4-1", student_id, matched)

    [doc] = _batches(client)["batches"][0]["documents"]

    assert doc["status"] == "needs_attention" and doc["attention"] == [reason]
    assert doc["can_confirm"] is True
    assert doc["pages"][0]["needs_person"] is (not student_id)


def test_a_document_waits_while_the_next_page_may_be_its_second_part(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    first = _queue(google, "a.tiff")[0]
    _queue(google, "b.tiff")  # not read yet
    _read(first, "B1-1", "AATKHA200618", dict(AATIF))

    [doc] = _batches(google)["batches"][0]["documents"]
    assert doc["status"] == "waiting" and doc["can_confirm"] is False

    response = _confirm(google, student_id="AATKHA200618")
    assert response.status_code == 409
    assert drive.calls == []


def test_batches_are_newest_first_and_only_the_owners(app: Flask, client: FlaskClient) -> None:
    old = _queue(client, "a.tiff")[0]
    new = _queue(client, "b.tiff")[0]
    _read(old, "B1-1")
    _read(new, "B2-1")
    theirs = _queue(_other_person(app), "c.tiff")[0]
    _read(theirs, "B3-1")

    assert [b["label"] for b in _batches(client)["batches"]] == ["B2", "B1"]
    assert [b["label"] for b in _batches(_other_person(app))["batches"]] == ["B3"]


def test_one_document_can_be_fetched_by_key(app: Flask, client: FlaskClient) -> None:
    _ready_document(client)

    response = client.get("/api/documents/B1-1")

    assert response.status_code == 200
    assert response.get_json()["document"]["parts"] == 2
    assert _other_person(app).get("/api/documents/B1-1").status_code == 404
    assert client.get("/api/documents/B9-9").status_code == 404
    assert client.get("/api/documents/not-a-key").status_code == 404


# ================================================================== the image
def test_a_page_image_is_served_to_its_owner_only(app: Flask, client: FlaskClient) -> None:
    [page_id] = _queue(client)

    full = client.get(f"/api/pages/{page_id}/image")
    thumb = client.get(f"/api/pages/{page_id}/image?size=thumb")

    assert full.status_code == 200 and full.mimetype == "image/jpeg"
    assert "private" in full.headers["Cache-Control"] and "no-cache" in full.headers["Cache-Control"]
    big = cv2.imdecode(np.frombuffer(full.data, np.uint8), cv2.IMREAD_UNCHANGED)
    small = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert big.shape == (874, 620)
    assert small.shape[1] == 400
    again = client.get(f"/api/pages/{page_id}/image", headers={"If-None-Match": full.headers["ETag"]})
    assert again.status_code == 304

    assert _other_person(app).get(f"/api/pages/{page_id}/image").status_code == 404
    assert app.test_client().get(f"/api/pages/{page_id}/image").status_code == 401
    assert client.get("/api/pages/" + "0" * 32 + "/image").status_code == 404
    assert client.get("/api/pages/..%2F..%2Fsecrets/image").status_code == 404


# ================================================================== corrections
def test_a_typed_code_regroups_the_page(app: Flask, client: FlaskClient) -> None:
    _ready_document(client)
    loose = _queue(client, "Scan 3.tiff")[0]
    _read(loose, None)

    response = client.post(f"/api/pages/{loose}/code", json={"code": " b1 - 1 "})

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["page"]["code"] == "B1-1" and body["page"]["code_source"] == "typed"
    assert body["document"]["key"] == "B1-1" and body["document"]["parts"] == 3
    assert _batches(client)["unsorted"] == []

    # Cleared, it goes back to what OCR read (nothing).
    client.post(f"/api/pages/{loose}/code", json={"code": ""})
    assert [p["page_id"] for p in _batches(client)["unsorted"]] == [loose]

    assert client.post(f"/api/pages/{loose}/code", json={"code": "C1-1"}).status_code == 400
    assert _other_person(app).post(f"/api/pages/{loose}/code", json={"code": "B1-1"}).status_code == 404


def test_a_typed_code_must_come_as_json(client: FlaskClient) -> None:
    [page_id] = _queue(client)

    response = client.post(f"/api/pages/{page_id}/code", data='{"code": "B1-1"}', content_type="text/plain")

    assert response.status_code == 415
    assert scan_queue.load_overrides(page_id).code is None


def test_a_typed_student_id_is_looked_up_without_writing(
    client: FlaskClient, fake_db: Any, students: Students
) -> None:
    [page_id] = _queue(client)
    _read(page_id, "B1-5", "", None)

    response = client.post(f"/api/pages/{page_id}/student-id", json={"student_id": "aatkha200618"})

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["page"]["student_id"] == "AATKHA200618" and body["page"]["student_id_source"] == "typed"
    assert body["page"]["matched_student"]["student_name"] == "Aatif Khan"
    assert body["document"]["status"] == "ready"
    assert fake_db.executed(r"FROM active_student_data") and _writes(fake_db) == []

    bad = client.post(f"/api/pages/{page_id}/student-id", json={"student_id": "9876543210"})
    assert bad.status_code == 400


def test_a_typed_student_id_during_an_outage_is_not_kept(client: FlaskClient, students: Students) -> None:
    [page_id] = _queue(client)
    students.down = True

    response = client.post(f"/api/pages/{page_id}/student-id", json={"student_id": "AATKHA200618"})

    assert response.status_code == 503
    assert scan_queue.load_overrides(page_id).student_id is None


# ================================================================== confirm
def test_confirm_marks_the_student_uploads_one_pdf_and_logs_it(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive, queue_dir: Path
) -> None:
    _ready_document(google)

    response = _confirm(google, student_id="AATKHA200618")

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["uploaded"] is True
    assert body["drive_link"].startswith("https://drive.example/")
    doc = body["document"]
    assert doc["status"] == "uploaded" and doc["key"].startswith("B1-1~")
    assert doc["drive_folder"] == "OCR/ABMPS/Grade 5/A/Aatif Khan_AATKHA200618"
    assert doc["drive_name"] == "Aatif Khan_AATKHA200618_B1-1.pdf"

    [pdf_file] = drive.uploaded()
    assert pdf_file["name"] == "Aatif Khan_AATKHA200618_B1-1.pdf"
    assert drive.path_of(pdf_file["id"]) == ["OCR", "ABMPS", "Grade 5", "A", "Aatif Khan_AATKHA200618"]
    with fitz.open(stream=pdf_file["data"], filetype="pdf") as pdf:
        assert pdf.page_count == 2
        # 620 x 874 px at 150 dpi as A4-ish points, not pixels.
        assert round(pdf[0].rect.width) == round(620 * 72 / (874 / 11.69))

    [update] = fake_db.executed(r"^UPDATE active_student_data SET scanned_at")
    assert update[1][1] == "AATKHA200618"
    [insert] = fake_db.executed(r"^INSERT INTO ocr_scan_log")
    params = insert[1]
    assert params[1] == "AATKHA200618" and params[2] == 1
    assert params[6] == "batch:B1-1"
    assert params[8] == body["drive_link"]  # DrivelinkImage
    assert params[10] == SCHOOL

    # The pages have left the open list; the document is listed as uploaded.
    assert scan_queue.open_page_ids(OWNER) == []
    [listed] = _batches(google)["batches"][0]["documents"]
    assert listed["status"] == "uploaded" and listed["drive_link"] == body["drive_link"]
    # A second confirm of the same code has nothing to confirm.
    again = _confirm(google, student_id="AATKHA200618")
    assert again.status_code == 404
    assert len(fake_db.executed(r"^UPDATE active_student_data")) == 1


def test_confirm_reuses_existing_folders_and_never_overwrites_a_pdf(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    ocr = drive.folder("OCR", "root")
    school = drive.folder("ABMPS", ocr)
    grade = drive.folder("Grade 5", school)
    division = drive.folder("A", grade)
    person = drive.folder("Aatif Khan_AATKHA200618", division)
    drive.files["old"] = {"id": "old", "name": "Aatif Khan_AATKHA200618_B1-1.pdf", "parents": [person], "folder": False}
    _ready_document(google)

    body = _confirm(google, student_id="AATKHA200618").get_json()

    assert body["document"]["drive_name"] == "Aatif Khan_AATKHA200618_B1-1 (2).pdf"
    assert "create-folder" not in drive.calls
    assert sorted(f["name"] for f in drive.uploaded()) == [
        "Aatif Khan_AATKHA200618_B1-1 (2).pdf", "Aatif Khan_AATKHA200618_B1-1.pdf",
    ]


def test_confirm_of_a_student_not_in_the_database_saves_nothing(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    _ready_document(google)

    response = _confirm(google, student_id="AATKHA200619")

    assert response.status_code == 409
    assert "Nothing was saved" in response.get_json()["error"]
    assert _writes(fake_db) == []
    assert drive.calls == []
    assert len(scan_queue.open_page_ids(OWNER)) == 2
    assert _batches(google)["batches"][0]["documents"][0]["status"] == "ready"


def test_confirm_during_an_outage_saves_nothing(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    _ready_document(google)
    students.down = True

    response = _confirm(google, student_id="AATKHA200618")

    assert response.status_code == 503
    assert "Nothing was saved" in response.get_json()["error"]
    assert _writes(fake_db) == []
    assert drive.calls == []
    assert len(scan_queue.open_page_ids(OWNER)) == 2


def test_confirm_signed_in_through_tether_saves_nothing(
    client: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    with client.session_transaction() as session:
        session["signed_in_via"] = "tether"
    _ready_document(client)

    response = _confirm(client, student_id="AATKHA200618")

    assert response.status_code == 409
    assert response.get_json()["error"].startswith("Sign in with Google to upload to Drive")
    assert fake_db.statements == []
    assert drive.calls == []


@pytest.mark.parametrize("student_id", ["", "9876543210", "Aatif Khan"])
def test_confirm_needs_a_student_id(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive, student_id: str
) -> None:
    _ready_document(google)

    response = _confirm(google, student_id=student_id)

    assert response.status_code == 400
    assert fake_db.statements == [] and drive.calls == []


def test_a_drive_failure_leaves_it_confirmed_and_the_retry_uploads(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    _ready_document(google)
    drive.fail_uploads = True

    response = _confirm(google, student_id="AATKHA200618")

    assert response.status_code == 200
    body = response.get_json()
    assert body["uploaded"] is False and body["drive_link"] is None
    doc = body["document"]
    assert doc["status"] == "confirmed" and doc["can_retry_upload"] is True
    assert doc["upload_error"] == "Google Drive refused the upload."
    assert "Retry upload" in body["message"]
    # The student is marked and History has its row - without a link yet.
    assert len(fake_db.executed(r"^UPDATE active_student_data SET scanned_at")) == 1
    [insert] = fake_db.executed(r"^INSERT INTO ocr_scan_log")
    assert insert[1][8] is None
    [listed] = _batches(google)["batches"][0]["documents"]
    assert listed["status"] == "confirmed"

    # Drive still down: the retry's whole answer is that failure.
    still = google.post(f"/api/documents/{doc['key']}/upload")
    assert still.status_code == 502
    assert still.get_json()["document"]["status"] == "confirmed"
    assert "try the upload again" in still.get_json()["error"]
    assert len(fake_db.executed(r"^INSERT INTO ocr_scan_log")) == 1

    drive.fail_uploads = False
    retried = google.post(f"/api/documents/{doc['key']}/upload")

    assert retried.status_code == 200, retried.get_json()
    after = retried.get_json()
    assert after["uploaded"] is True and after["document"]["status"] == "uploaded"
    # Only the upload was redone: no second mark, no second row - the link
    # goes onto the row already written.
    assert len(fake_db.executed(r"^UPDATE active_student_data")) == 1
    assert len(fake_db.executed(r"^INSERT INTO ocr_scan_log")) == 1
    [link_update] = fake_db.executed(r"^UPDATE ocr_scan_log SET DrivelinkImage")
    assert link_update[1][0] == after["drive_link"]
    assert len(drive.uploaded()) == 1
    # Folders made on the failed try are reused, not made twice.
    folder_names = [f["name"] for f in drive.files.values() if f["folder"]]
    assert sorted(folder_names) == sorted(set(folder_names))
    # Retrying an uploaded document does nothing more.
    assert google.post(f"/api/documents/{doc['key']}/upload").get_json()["uploaded"] is True
    assert len(drive.uploaded()) == 1


def test_retry_upload_is_for_the_owners_confirmed_documents_only(
    app: Flask, google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    _ready_document(google)
    drive.fail_uploads = True
    key = _confirm(google, student_id="AATKHA200618").get_json()["document"]["key"]

    assert _other_person(app).post(f"/api/documents/{key}/upload").status_code == 404
    assert google.post("/api/documents/B1-1/upload").status_code == 404


def test_a_code_written_again_after_its_document_was_confirmed_is_flagged(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    _ready_document(google)
    _confirm(google, student_id="AATKHA200618")
    late = _queue(google, "late.tiff")[0]
    _read(late, "B1-1", "AATKHA200618", dict(AATIF))

    docs = _batches(google)["batches"][0]["documents"]

    [open_doc] = [d for d in docs if d["key"] == "B1-1"]
    assert "code_reused" in open_doc["attention"] and open_doc["status"] == "needs_attention"
    assert "already confirmed for Aatif Khan" in open_doc["message"]


def test_a_confirmed_page_cannot_be_corrected(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    first, _second = _ready_document(google)
    _confirm(google, student_id="AATKHA200618")

    assert google.post(f"/api/pages/{first}/code", json={"code": "B2-2"}).status_code == 409
    assert google.post(f"/api/pages/{first}/student-id", json={"student_id": "MOHSHA011021"}).status_code == 409


def test_two_confirms_at_once_save_once(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    _ready_document(google)
    with scan_queue.confirm_lock(OWNER, "B1-1"):
        busy = _confirm(google, student_id="AATKHA200618")

    assert busy.status_code == 409
    assert _writes(fake_db) == []


def test_confirm_saves_only_the_parts_the_person_was_shown(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    shown = _ready_document(google)
    # Another page read as B1-1 after the page's last poll: a rescan, or
    # another form's corner misread. The person never saw it.
    third = _queue(google, "Scan 3.tiff")[0]
    _read(third, "B1-1")

    response = _confirm(google, page_ids=shown)

    assert response.status_code == 409
    error = response.get_json()["error"]
    assert error.startswith("This document changed") and "Nothing was saved" in error
    assert _writes(fake_db) == [] and drive.calls == []
    assert len(scan_queue.open_page_ids(OWNER)) == 3

    # Looked at again, all three parts are what goes into the PDF.
    again = _confirm(google, page_ids=[third, *shown])
    assert again.status_code == 200, again.get_json()
    assert again.get_json()["document"]["parts"] == 3
    with fitz.open(stream=drive.uploaded()[0]["data"], filetype="pdf") as pdf:
        assert pdf.page_count == 3


def test_confirm_checks_the_parts_again_once_it_holds_the_lock(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown = _ready_document(google)
    real_lock = scan_queue.confirm_lock

    def lock_while_a_page_is_read(owner: str, code: str) -> Any:
        # Another page arrives and the worker reads it with this code while
        # the confirm waits for its lock.
        [third] = scan_queue.intake(
            _tiff([_form_page()]), "Scan 3.tiff", source_name="Scan 3.tiff",
            owner_email=OWNER, school=SCHOOL, location="Mumbai",
        )
        _read(third.page_id, "B1-1")
        return real_lock(owner, code)

    monkeypatch.setattr(scan_queue, "confirm_lock", lock_while_a_page_is_read)

    response = _confirm(google, page_ids=shown)

    assert response.status_code == 409
    assert response.get_json()["error"].startswith("This document changed")
    assert _writes(fake_db) == [] and drive.calls == []


@pytest.mark.parametrize("page_ids", [None, "abc", [1, 2], []])
def test_confirm_needs_the_parts_the_page_showed(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive, page_ids: Any
) -> None:
    _ready_document(google)
    body: dict[str, Any] = {"student_id": "AATKHA200618"}
    if page_ids is not None:
        body["page_ids"] = page_ids

    response = google.post("/api/documents/B1-1/confirm", json=body)

    assert response.status_code in (400, 409)
    assert "Nothing was saved" in response.get_json()["error"]
    assert fake_db.statements == [] and drive.calls == []


def test_confirm_marks_only_the_child_at_the_documents_school(
    google: FlaskClient, fake_db: Any, drive: FakeDrive
) -> None:
    # One Student ID, two children: one at ABMPS, one at another school.
    children = {("AATKHA200618", "ABMPS"): None, ("AATKHA200618", "OTHER SCHOOL"): None}

    def handler(sql: str, params: Any) -> tuple[list[dict[str, Any]] | None, int | None]:
        if sql.startswith("SELECT student_id, student_name"):
            schools = {str(p).upper() for p in params[1:]}
            rows = [
                dict(AATIF, school_name=school)
                for (sid, school) in children
                if sid == str(params[0]).upper() and (not schools or school.upper() in schools)
            ]
            return rows[:1], None
        if sql.startswith("UPDATE active_student_data SET scanned_at"):
            when, sid, *school = params
            marked = 0
            for child_id, child_school in children:
                if child_id == str(sid).upper() and (
                    "school_name" not in sql or (school and child_school.upper() == str(school[0]).upper())
                ):
                    children[(child_id, child_school)] = when
                    marked += 1
            return [], marked
        if sql.startswith("SELECT scanned_at FROM"):
            return [{"scanned_at": None}], None
        return [], 0

    fake_db.handler = handler
    _ready_document(google)

    response = _confirm(google)

    assert response.status_code == 200, response.get_json()
    assert children[("AATKHA200618", "ABMPS")] is not None
    assert children[("AATKHA200618", "OTHER SCHOOL")] is None
    [update] = fake_db.executed(r"^UPDATE active_student_data SET scanned_at")
    assert "school_name" in update[0] and update[1][1:] == ("AATKHA200618", "ABMPS")
    [read_back] = fake_db.executed(r"^SELECT scanned_at FROM")
    assert "school_name" in read_back[0] and read_back[1] == ("AATKHA200618", "ABMPS")


def test_a_history_entry_that_failed_after_the_upload_is_written_by_the_retry(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive, queue_dir: Path
) -> None:
    answer = fake_db.handler
    database = {"dropped": True}

    def flaky(sql: str, params: Any) -> tuple[list[dict[str, Any]] | None, int | None]:
        if sql.startswith("INSERT INTO ocr_scan_log") and database["dropped"]:
            raise pymysql.err.OperationalError(2013, "Lost connection to MySQL server during query")
        return answer(sql, params)

    fake_db.handler = flaky
    _ready_document(google)

    response = _confirm(google)

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    # In Drive, but not finished: History has no row for it yet.
    assert body["uploaded"] is True and body["finished"] is False
    doc = body["document"]
    assert doc["status"] == "confirmed" and doc["can_retry_upload"] is True
    assert doc["history_pending"] is True and doc["drive_link"]
    assert "History" in doc["message"]
    assert len(drive.uploaded()) == 1
    [listed] = _batches(google)["batches"][0]["documents"]
    assert listed["can_retry_upload"] is True

    # Still down: the retry says so, and nothing is uploaded again.
    still = google.post(f"/api/documents/{doc['key']}/upload")
    assert still.status_code == 502
    assert len(drive.uploaded()) == 1

    database["dropped"] = False
    retried = google.post(f"/api/documents/{doc['key']}/upload")

    assert retried.status_code == 200, retried.get_json()
    after = retried.get_json()
    assert after["finished"] is True and after["document"]["status"] == "uploaded"
    assert after["document"]["can_retry_upload"] is False
    # Only the History row was written: no second upload, no second mark.
    assert len(drive.uploaded()) == 1
    assert len(fake_db.executed(r"^UPDATE active_student_data SET scanned_at")) == 1
    inserts = fake_db.executed(r"^INSERT INTO ocr_scan_log")
    assert len(inserts) == 3  # two refused, one written
    assert inserts[-1][1][8] == doc["drive_link"]
    record = scan_queue.load_document_record(OWNER, doc["key"])
    assert record is not None and record["log_pending"] is False and record["log_id"]


def test_an_unfinished_document_stays_listed_after_the_done_window(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive, queue_dir: Path
) -> None:
    answer = fake_db.handler

    def no_history(sql: str, params: Any) -> tuple[list[dict[str, Any]] | None, int | None]:
        if sql.startswith("INSERT INTO ocr_scan_log"):
            raise pymysql.err.OperationalError(2013, "Lost connection to MySQL server during query")
        return answer(sql, params)

    fake_db.handler = no_history
    _ready_document(google)
    key = _confirm(google).get_json()["document"]["key"]
    old = time.time() - 30 * 24 * 3600
    for record in (scan_queue.owner_dir(OWNER) / "docs").glob("*.json"):
        os.utime(record, (old, old))

    docs = [d for b in _batches(google)["batches"] for d in b["documents"]]

    assert [d["key"] for d in docs] == [key]
    assert docs[0]["can_retry_upload"] is True


def test_two_documents_of_one_class_confirmed_at_once_share_their_folders(
    app: Flask, google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    for code in ("B1-1", "B1-2"):
        page_id = _queue(google, f"{code}.tiff")[0]
        _read(page_id, code, "AATKHA200618", dict(AATIF))
    tabs = []
    for _ in range(2):
        tab = app.test_client()
        with tab.session_transaction() as session:
            session["user"] = {"email": OWNER, "name": "Staff", "picture": ""}
            session["google_token"] = {
                "access_token": "test-access-token",
                "refresh_token": "test-refresh-token",
                "expires_at": time.time() + 3600,
            }
        tabs.append(tab)
    shown = {code: [p["page_id"] for p in tabs[0].get(f"/api/documents/{code}").get_json()["document"]["pages"]]
             for code in ("B1-1", "B1-2")}
    drive.latency = 0.1
    start = threading.Barrier(2)
    answers: dict[str, int] = {}

    def press(tab: FlaskClient, code: str) -> None:
        start.wait()
        answers[code] = _confirm(tab, key=code, page_ids=shown[code]).status_code

    threads = [threading.Thread(target=press, args=(tab, code)) for tab, code in zip(tabs, ("B1-1", "B1-2"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert answers == {"B1-1": 200, "B1-2": 200}
    folders = [f["name"] for f in drive.files.values() if f["folder"]]
    assert sorted(folders) == ["A", "ABMPS", "Aatif Khan_AATKHA200618", "Grade 5", "OCR"]
    [first, second] = drive.uploaded()
    assert first["parents"] == second["parents"]


def test_confirm_files_into_the_oldest_of_two_folders_of_one_name(
    google: FlaskClient, students: Students, drive: FakeDrive
) -> None:
    oldest = drive.folder("OCR", "root")
    drive.folder("OCR", "root")  # a duplicate made later
    _ready_document(google)

    body = _confirm(google).get_json()

    assert body["uploaded"] is True
    [pdf_file] = drive.uploaded()
    top = pdf_file
    while top["parents"][0] != "root":
        top = drive.files[top["parents"][0]]
    assert top["id"] == oldest


# ================================================================== the bottom of the page stays here
def _bottom_inked_page(size: tuple[int, int] = (620, 874)) -> Image.Image:
    """A right-way-up signature page: nothing written at the top, ink in both bottom corners."""
    width, height = size
    page = _form_page(size, mark="bottom-left")
    ImageDraw.Draw(page).rectangle(
        (int(width * 0.8), int(height * 0.94), int(width * 0.95), height - 5), fill=0
    )
    return page


def test_the_upside_down_try_never_sends_the_bottom_of_a_page_to_google(
    client: FlaskClient, students: Students, readers: Readers, monkeypatch: pytest.MonkeyPatch,
    queue_dir: Path,
) -> None:
    from server.services import batch_code, vision_ocr

    sent: list[tuple[int, ...]] = []
    local: list[tuple[int, ...]] = []
    monkeypatch.setattr(vision_ocr, "is_configured", lambda: True)
    monkeypatch.setattr(vision_ocr, "recognize_text", lambda crop: sent.append(crop.shape) or [])
    monkeypatch.setattr(batch_code, "_rapid_texts", lambda crop: local.append(crop.shape) or [])
    monkeypatch.setattr(ocr_worker, "read_code", batch_code.read_batch_code)
    [page_id] = _queue(client, data=_tiff([_bottom_inked_page()]))

    assert ocr_worker.process_one() == ocr_worker.READ

    # The top corners are empty, so nothing was read there; the bottom
    # corners, turned round to be tried as the top, were read on this
    # machine only.
    assert sent == []
    assert len(local) == 2
    assert json.loads((queue_dir / page_id / "result.json").read_text())["orientation"] == 0


def test_the_upside_down_try_asks_for_the_local_reader(
    client: FlaskClient, students: Students, readers: Readers
) -> None:
    _queue(client, data=_tiff([_form_page(mark="bottom-left")]))
    readers.codes = [None, None]

    ocr_worker.process_one()

    assert readers.vision_allowed == [True, False]


def test_an_upside_down_code_read_on_this_machine_still_turns_the_page(
    client: FlaskClient, students: Students, readers: Readers, monkeypatch: pytest.MonkeyPatch,
    queue_dir: Path,
) -> None:
    from server.services import batch_code, vision_ocr

    monkeypatch.setattr(vision_ocr, "is_configured", lambda: True)
    monkeypatch.setattr(vision_ocr, "recognize_text", lambda _crop: pytest.fail("sent to Google"))
    monkeypatch.setattr(batch_code, "_rapid_texts", lambda _crop: [("B2-3", 0.8)])
    monkeypatch.setattr(ocr_worker, "read_code", batch_code.read_batch_code)
    [page_id] = _queue(client, data=_tiff([_form_page(mark="bottom-left")]))

    ocr_worker.process_one()

    result = json.loads((queue_dir / page_id / "result.json").read_text())
    assert result["orientation"] == 180 and result["code"] == "B2-3"


# ================================================================== a page read twice at once
def _claimed_page(client: FlaskClient) -> tuple[str, scan_queue.Claim]:
    """A page received long ago, released by a reader and at once claimed again."""
    [page_id] = _queue(client)
    meta = scan_queue.load_meta(page_id)
    assert meta is not None
    meta.received_ns = time.time_ns() - 3600 * 1_000_000_000
    scan_queue.save_meta(meta)
    claim = scan_queue.claim_next()
    assert claim is not None and claim.page_id == page_id
    return page_id, claim


def test_a_page_between_two_markers_is_not_queued_a_second_time(client: FlaskClient) -> None:
    from server.services import documents

    page_id, held = _claimed_page(client)
    # The listing looked at _pending and _reading a moment apart, and the
    # page moved between them: it was seen in neither.
    blind = scan_queue.QueueState(pending=frozenset(), reading=frozenset())

    documents.load_page_view(page_id, OWNER, blind)

    assert scan_queue.claim_next() is None
    assert scan_queue.still_held(held)


def test_a_page_quiet_past_the_claim_timeout_but_claimed_is_left_alone(client: FlaskClient) -> None:
    from server.services import documents

    page_id, held = _claimed_page(client)
    meta_path = scan_queue.page_dir(page_id) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["updated_at"] = "2020-01-01T00:00:00"
    meta_path.write_text(json.dumps(meta))
    blind = scan_queue.QueueState(pending=frozenset(), reading=frozenset())

    documents.load_page_view(page_id, OWNER, blind)

    assert scan_queue.claim_next() is None
    assert scan_queue.still_held(held)


def test_a_real_orphan_is_queued_again(client: FlaskClient, queue_dir: Path) -> None:
    from server.services import documents

    [page_id] = _queue(client)
    # Stored, but its marker never made (a crash part-way through intake),
    # and nothing has touched it since.
    (queue_dir / "_pending" / page_id).unlink()
    meta_path = scan_queue.page_dir(page_id) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["updated_at"] = "2020-01-01T00:00:00"
    meta_path.write_text(json.dumps(meta))

    view = documents.load_page_view(page_id, OWNER)

    assert view is not None and view.status == "queued"
    assert (queue_dir / "_pending" / page_id).exists()
    fresh = _queue(client, "fresh.tiff")[0]
    (queue_dir / "_pending" / fresh).unlink()
    documents.load_page_view(fresh, OWNER)
    # One just stored is not an orphan yet: it may still be on its way in.
    assert not (queue_dir / "_pending" / fresh).exists()


# ================================================================== the guards
@pytest.mark.parametrize(
    ("url", "kwargs"),
    [
        ("/api/scans", {"data": {"file": (io.BytesIO(b"x"), "a.jpg")}, "content_type": "multipart/form-data"}),
        ("/api/documents/B1-1/confirm", {"json": {"student_id": "AATKHA200618"}}),
        ("/api/documents/B1-1~0123abcd/upload", {}),
        ("/api/pages/" + "a" * 32 + "/code", {"json": {"code": "B1-1"}}),
        ("/api/pages/" + "a" * 32 + "/student-id", {"json": {"student_id": "AATKHA200618"}}),
    ],
)
def test_a_cross_site_write_to_the_queue_is_refused(
    google: FlaskClient, fake_db: Any, drive: FakeDrive, queue_dir: Path, url: str, kwargs: dict[str, Any]
) -> None:
    response = google.post(url, headers={"Origin": "https://evil.example"}, **kwargs)

    assert response.status_code == 403
    assert fake_db.statements == [] and drive.calls == []
    assert not queue_dir.exists()


def test_a_confirm_must_come_as_json(
    google: FlaskClient, fake_db: Any, students: Students, drive: FakeDrive
) -> None:
    _ready_document(google)

    response = google.post(
        "/api/documents/B1-1/confirm", data="student_id=AATKHA200618",
        content_type="application/x-www-form-urlencoded",
    )

    assert response.status_code == 415
    assert fake_db.statements == [] and drive.calls == []


def test_the_queue_routes_need_sign_in(app: Flask) -> None:
    anonymous = app.test_client()

    assert anonymous.get("/api/batches").status_code == 401
    assert anonymous.get("/api/documents/B1-1").status_code == 401
    assert anonymous.post("/api/documents/B1-1/confirm", json={"student_id": "X"}).status_code == 401
