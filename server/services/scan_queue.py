"""The scan queue: pages waiting for OCR, and what OCR made of them, on disk.

Scanning only stores a page here and returns; the OCR worker
(services/ocr_worker.py) reads it in the background, and nothing reaches the
database or Drive until the document is confirmed (services/documents.py) - by
a person, or by the scan page for an exact match sent with auto_upload on.
Everything is plain folders and JSON - no database tables - so a page that
was waiting survives a restart, and the web app and the watcher process can
share the queue on one disk.

    <QUEUE_DIR>/<page_id>/
        original.<ext>   the page as uploaded (one page of a PDF/TIFF split out)
        page.jpg         normalised: upright, greyscale, <= 300 dpi
        thumb.jpg        made on first request, ~400 px wide
        meta.json        who sent it, from which file, status, attempts
        ocr.json         what OCR read, kept so a lookup retry need not re-read
        result.json      the reading plus the read-only student match
        code.json        a batch code typed by a person (overrides the reading)
        student_id.json  a Student ID typed by a person (overrides the reading),
                         kept with who took it back once taken back
        declaration.json a person's word that the page is the Declaration page
                         of the form scanned straight before it, kept with who
                         took it back once taken back
    <QUEUE_DIR>/_pending/<page_id>                waiting for a reader
    <QUEUE_DIR>/_reading/<page_id>.<ms>.<token>   claimed by a reader at <ms>
    <QUEUE_DIR>/_owners/<owner>/open/<page_id>    not yet in a confirmed document
    <QUEUE_DIR>/_owners/<owner>/storing/<token>   a scan of theirs being stored now,
                                                  with its sender, school and city
    <QUEUE_DIR>/_owners/<owner>/docs/<key>.json   a confirmed document
    <QUEUE_DIR>/_owners/<owner>/locks/<key>.lock  a confirm in progress
    <QUEUE_DIR>/_owners/<owner>/locks/drive-folders.lock
                                                  finding or making Drive folders
    <QUEUE_DIR>/_worker_state.json                what the OCR worker is doing:
                                                  running, paused or idle

Claiming is one os.rename of the page's marker from _pending to _reading:
the rename succeeds for exactly one reader, whichever process it is in, so a
page is never read twice at once. The claim's name carries when it was taken,
so a claim left by a reader that died is spotted by its name alone and handed
back after QUEUE_CLAIM_STALE_SECONDS. Every JSON file is written whole to a
temporary name and moved into place, so a reader never sees half a file.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from server.config import (
    PAGE_MAX_DPI,
    PAGE_MAX_LONG_SIDE,
    PAGE_PDF_DPI,
    QUEUE_CLAIM_STALE_SECONDS,
    QUEUE_DIR,
)
from server.services.clock import now_ist
from server.services.debug_log import debug_log
from server.services.preprocess import (
    count_pages_from_bytes,
    load_page_grey,
    split_original_page,
)

STATUS_QUEUED = "queued"
STATUS_READING = "reading"
STATUS_READ = "read"
STATUS_FAILED = "failed"

PAGE_IMAGE = "page.jpg"
THUMB_IMAGE = "thumb.jpg"
THUMB_WIDTH = 400

# Beside the queue it describes, so the web app can show what the worker is
# doing whichever process the worker runs in (services/ocr_worker.py).
WORKER_STATE_FILE = "_worker_state.json"

# A confirm that has held its lock this long has died part-way; a real one
# takes seconds (a lookup and one Drive upload).
CONFIRM_LOCK_STALE_SECONDS = 5 * 60

# The lock around finding or making a person's Drive folders. Two confirms
# of one class at once would otherwise both find no "Grade 5" folder and
# both make one. A second confirm waits its turn rather than failing: the
# folder chain is a handful of Drive calls.
DRIVE_FOLDERS_LOCK = "drive-folders"
DRIVE_FOLDERS_WAIT_SECONDS = 90.0
_LOCK_POLL_SECONDS = 0.05

_PAGE_ID = re.compile(r"[0-9a-f]{32}")
_CLAIM_NAME = re.compile(r"(?P<page>[0-9a-f]{32})\.(?P<ms>\d+)\.(?P<token>[0-9a-f]+)")
# A document key: the code alone while open, the code and a token once
# confirmed (so the same code written again later is a new document). With
# batch codes off a document is one form, keyed "P" and its Student ID page's
# id - which no code can be - with the same token once confirmed.
_DOCUMENT_KEY = re.compile(r"(?:B\d{1,3}-\d{1,3}|P[0-9a-f]{32})(?:~[0-9a-f]{8})?")
# A scan page's id for one visit, as the page makes it: 32 hex digits, with
# "-1" after them on the files added by hand.
_SENDER = re.compile(r"[0-9a-f]{32}(?:-1)?")


class DocumentBusy(RuntimeError):
    """Another confirm of the same document is already running."""


@dataclass
class PageMeta:
    """meta.json: where a queued page came from and how far it has got.

    Attributes:
        page_id: 32 hex digits; the first 16 sort by when it was received.
        created_at: When it was received, India time ("YYYY-MM-DDTHH:MM:SS").
        owner_email: The signed-in person who sent it; only they see it.
        school: The school picked on the scan page, "" for none.
        location: The city picked on the scan page, "" for none.
        source_name: The uploaded file's name, as the person's computer had it.
        source_page: Which page of that file (1-based).
        source_pages: How many pages that file had.
        status: queued, reading, read or failed.
        attempts: How many times a reader has started on it.
        received_ns: When the file was received; with source_page, the scan order.
        dpi: The resolution of page.jpg, when known.
        original: The name of the stored original ("original.tiff").
        error: Why it failed, once it has.
        updated_at: When this file was last written, India time.
        auto_upload: Sent with Upload matched pages automatically on, so its
            document may be confirmed without anyone pressing Confirm
            (documents.auto_upload_refusal). Off for a page queued before
            the switch existed: its meta.json has no such key.
        sender: The id the scan page made for itself when it was loaded,
            sent with every scan (is_sender), with "-1" after it on the
            files added by hand: with batch codes off, only pages from one
            scan page are one stack (documents._runs), and only a form sent
            the one way may go up by itself. "" for a page sent by an older
            page or any other client.
    """

    page_id: str
    created_at: str
    owner_email: str
    school: str = ""
    location: str = ""
    source_name: str = ""
    source_page: int = 1
    source_pages: int = 1
    status: str = STATUS_QUEUED
    attempts: int = 0
    received_ns: int = 0
    dpi: float | None = None
    original: str = ""
    error: str = ""
    updated_at: str = ""
    auto_upload: bool = False
    sender: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageMeta:
        """Build from meta.json, ignoring keys this version does not know."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PageResult:
    """result.json (and ocr.json, without the match): what OCR read.

    Attributes:
        code: The batch code read ("B1-12"), or None.
        code_confidence: The recogniser's confidence in it.
        code_raw: The text the code was parsed from (or the first text read).
        code_region: Which corner it was in, or None.
        student_id: The Student ID read; only a reading with an ID's shape.
        student_raw_shape_ok: Whether that reading had the shape as read,
            before any shape correction.
        candidates: The Student ID readings considered.
        matched_student: The student it matched (read-only, within the page's
            school or city), or None.
        message: What a person should be told about the page.
        orientation: 0, or 180 when the page was scanned upside down (page.jpg
            has been turned upright since).
        blank: The page carries almost no ink.
        skip_reason: blank, declaration, or empty_id when the page was not read.
        form_language: The form's language, when its layout was recognised.
        confidence: The reader's confidence in the Student ID.
        read_at: When it was read, India time.
    """

    code: str | None = None
    code_confidence: float = 0.0
    code_raw: str = ""
    code_region: str | None = None
    student_id: str = ""
    student_raw_shape_ok: bool = False
    candidates: list[str] = field(default_factory=list)
    matched_student: dict[str, Any] | None = None
    message: str = ""
    orientation: int = 0
    blank: bool = False
    skip_reason: str = ""
    form_language: str = ""
    confidence: float = 0.0
    read_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageResult:
        """Build from result.json, ignoring keys this version does not know."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Overrides:
    """What a person typed for a page, which wins over what OCR read.

    Attributes:
        code: A batch code typed by hand, or None.
        code_by: Who typed it.
        student_id: A Student ID typed by hand, or None.
        matched_student: The read-only match for that ID, or None.
        student_id_by: Who typed it - or, with no ID now, who took it back.
        declaration_of: With batch codes off, the id of the page this one
            was attached to by hand as its Declaration page, or None.
        declaration_by: Who attached it - or, with none now, who took it off.
        declaration_at: When, India time.
        unreadable: One of the files these are kept in is there but could
            not be read - a file being replaced, or one a power cut left
            empty: a person said something about the page that is not known
            now.
    """

    code: str | None = None
    code_by: str = ""
    student_id: str | None = None
    matched_student: dict[str, Any] | None = None
    student_id_by: str = ""
    declaration_of: str | None = None
    declaration_by: str = ""
    declaration_at: str = ""
    unreadable: bool = False


@dataclass(frozen=True)
class Claim:
    """A reader's hold on one page.

    Attributes:
        page_id: The page.
        marker: The claim file in _reading/; gone once the claim is lost.
        claimed_ms: When it was taken (epoch milliseconds).
    """

    page_id: str
    marker: Path
    claimed_ms: int


@dataclass(frozen=True)
class Storing:
    """One of a person's scans being stored right now (storing), as intake was told it.

    Attributes:
        sender: The scan page that sent it (is_sender), or "" for none.
        school: The school picked for it, "" for none.
        location: The city picked for it, "" for none.
    """

    sender: str
    school: str
    location: str


@dataclass(frozen=True)
class QueueState:
    """Which pages are waiting and which are being read, at one moment."""

    pending: frozenset[str]
    reading: frozenset[str]


# ------------------------------------------------------------- paths
def queue_root() -> Path:
    """The queue's folder. Read on every call, so a test can point it elsewhere."""
    return Path(QUEUE_DIR)


def _pending_dir() -> Path:
    return queue_root() / "_pending"


def _reading_dir() -> Path:
    return queue_root() / "_reading"


def worker_state_path() -> Path:
    """Where the OCR worker writes what it is doing (ocr_worker.WorkerState)."""
    return queue_root() / WORKER_STATE_FILE


def is_page_id(text: str | None) -> bool:
    """Whether `text` is a page id (and so safe to put in a path)."""
    return bool(text) and bool(_PAGE_ID.fullmatch(str(text)))


def is_document_key(text: str | None) -> bool:
    """Whether `text` is a document key ("B1-12", "P<page id>", either with "~1a2b3c4d")."""
    return bool(text) and bool(_DOCUMENT_KEY.fullmatch(str(text)))


def is_sender(text: str | None) -> bool:
    """Whether `text` is a scan page's id as the page makes it (PageMeta.sender)."""
    return bool(text) and bool(_SENDER.fullmatch(str(text)))


def page_document_key(page_id: str) -> str:
    """The key of the form a page begins when batch codes are off: "P" and its id."""
    return f"P{page_id}"


def page_dir(page_id: str) -> Path:
    """A page's folder. The id must already have passed is_page_id."""
    if not is_page_id(page_id):
        raise ValueError("not a page id")
    return queue_root() / page_id


def normalise_owner(email: str | None) -> str:
    """An owner's email as stored: trimmed and lower-case."""
    return (email or "").strip().lower()


def owner_dir(owner_email: str) -> Path:
    """The folder of one person's index, named by a hash of their email.

    A hash rather than the address itself keeps every character an email
    may hold out of the path.
    """
    digest = hashlib.sha256(normalise_owner(owner_email).encode("utf-8")).hexdigest()[:24]
    return queue_root() / "_owners" / digest


def new_page_id(received_ms: int, index: int) -> str:
    """A page id that sorts by when it was received, then by page."""
    return f"{received_ms:012x}{index:04x}{secrets.token_hex(8)}"


# ------------------------------------------------------------- JSON on disk
def write_json_atomic(path: Path, data: dict[str, Any], durable: bool = False) -> None:
    """Write JSON whole to a temporary name, then move it into place.

    durable: flush it to the disk before the move as well, so that a power
    cut leaves the old file or the new one in place, never an empty one -
    for what a person typed, which nothing else can tell again.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with open(temporary, "w", encoding="utf-8") as out:
        out.write(json.dumps(data, ensure_ascii=False, default=str))
        if durable:
            out.flush()
            os.fsync(out.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    """A JSON file's contents, or None when it is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write bytes whole to a temporary name, then move them into place."""
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _stamp() -> str:
    return now_ist().isoformat(timespec="seconds")


# ------------------------------------------------------------- intake
def encode_page(grey: np.ndarray) -> bytes:
    """A greyscale page as the JPEG stored in page.jpg."""
    ok, encoded = cv2.imencode(".jpg", grey, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("Could not encode the page")
    return encoded.tobytes()


def intake(
    data: bytes,
    filename: str,
    *,
    source_name: str,
    owner_email: str,
    school: str = "",
    location: str = "",
    auto_upload: bool = False,
    sender: str = "",
) -> list[PageMeta]:
    """Store an upload in the queue, one queued page per page of the file.

    Nothing is read here - that is the worker's job - so this returns as
    soon as the pages are on disk. Every page is decoded and written to a
    staging folder first and only moved into the queue once the whole file
    has worked, so a damaged page refuses the file without leaving half of
    it queued. Its pages are received (received_ns) when it starts and
    listed only when it ends; in between, the person's storing marker says
    they are on their way (storing).

    Args:
        data: The uploaded file.
        filename: A safe name for it, with its real extension.
        source_name: The name as the person's computer had it, for display.
        owner_email: The signed-in person.
        school: The school picked on the page.
        location: The city picked on the page.
        auto_upload: Whether the page's Upload matched pages automatically
            switch was on when it sent the file.
        sender: The scan page's id (is_sender), or "" for none.

    Returns:
        The queued pages, in page order.

    Raises:
        ValueError: The file is empty or cannot be decoded (NotATiffError
            for a file only named .tif).
    """
    if not data:
        raise ValueError("Empty upload")
    total = count_pages_from_bytes(data, filename)
    if total < 1:
        raise ValueError("No pages found in upload.")

    owner = normalise_owner(owner_email)
    suffix = Path(filename).suffix.lower() or ".bin"
    root = queue_root()
    metas: list[PageMeta] = []
    # Marked before it is received, and until its pages are listed.
    with _being_stored(owner, Storing(sender=sender, school=school, location=location)) as marker:
        received_ns = time.time_ns()
        created = _stamp()
        staging = root / "_incoming" / secrets.token_hex(8)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            for number in range(1, total + 1):
                page_id = new_page_id(received_ns // 1_000_000, number)
                folder = staging / page_id
                folder.mkdir()
                page = load_page_grey(
                    data,
                    filename,
                    number,
                    max_dpi=PAGE_MAX_DPI,
                    max_long_side=PAGE_MAX_LONG_SIDE,
                    pdf_dpi=PAGE_PDF_DPI,
                )
                original = data if total == 1 else split_original_page(data, filename, number)
                (folder / f"original{suffix}").write_bytes(original)
                (folder / PAGE_IMAGE).write_bytes(encode_page(page.image))
                meta = PageMeta(
                    page_id=page_id,
                    created_at=created,
                    owner_email=owner,
                    school=school,
                    location=location,
                    source_name=source_name,
                    source_page=number,
                    source_pages=total,
                    received_ns=received_ns,
                    dpi=round(page.dpi, 1) if page.dpi else None,
                    original=f"original{suffix}",
                    updated_at=created,
                    auto_upload=auto_upload,
                    sender=sender,
                )
                write_json_atomic(folder / "meta.json", asdict(meta))
                metas.append(meta)
                # Still going, page by page, however big the file.
                marker.touch()

            index = owner_dir(owner) / "open"
            index.mkdir(parents=True, exist_ok=True)
            _pending_dir().mkdir(parents=True, exist_ok=True)
            _reading_dir().mkdir(parents=True, exist_ok=True)
            for meta in metas:
                os.rename(staging / meta.page_id, root / meta.page_id)
                (index / meta.page_id).touch()
                # Last: only now may a reader take it.
                (_pending_dir() / meta.page_id).touch()
        finally:
            # Our own staging copy of pages that were refused or already moved.
            shutil.rmtree(staging, ignore_errors=True)
    debug_log(
        f"[QUEUE] queued {len(metas)} page(s) from {source_name!r} for {owner!r} "
        f"school={school!r} location={location!r} auto_upload={auto_upload} sender={sender!r}"
    )
    return metas


# ------------------------------------------------------------- reading pages
def load_meta(page_id: str) -> PageMeta | None:
    """A page's meta.json, or None when there is no such page."""
    if not is_page_id(page_id):
        return None
    data = read_json(page_dir(page_id) / "meta.json")
    return PageMeta.from_dict(data) if data else None


def save_meta(meta: PageMeta) -> None:
    """Write a page's meta.json."""
    meta.updated_at = _stamp()
    write_json_atomic(page_dir(meta.page_id) / "meta.json", asdict(meta))


def load_result(page_id: str) -> PageResult | None:
    """What the worker read off a page, or None before it has."""
    data = read_json(page_dir(page_id) / "result.json")
    return PageResult.from_dict(data) if data else None


def save_result(page_id: str, result: PageResult) -> None:
    """Write a page's result.json."""
    write_json_atomic(page_dir(page_id) / "result.json", asdict(result))


def load_ocr(page_id: str) -> PageResult | None:
    """The OCR reading kept for a lookup retry, or None."""
    data = read_json(page_dir(page_id) / "ocr.json")
    return PageResult.from_dict(data) if data else None


def save_ocr(page_id: str, reading: PageResult) -> None:
    """Keep the OCR reading, so a retry after a database outage only looks it up."""
    write_json_atomic(page_dir(page_id) / "ocr.json", asdict(reading))


def load_page_grey_image(page_id: str) -> np.ndarray | None:
    """page.jpg as a greyscale array, or None when it is missing or unreadable."""
    path = page_dir(page_id) / PAGE_IMAGE
    if not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image if image is not None and image.size else None


def replace_page_image(page_id: str, grey: np.ndarray) -> None:
    """Store a new page.jpg (turned upright) and drop the old thumbnail."""
    folder = page_dir(page_id)
    write_bytes_atomic(folder / PAGE_IMAGE, encode_page(grey))
    with contextlib.suppress(FileNotFoundError):
        (folder / THUMB_IMAGE).unlink()


def page_image_path(page_id: str, size: str = "full") -> Path | None:
    """The JPEG to send for a page: page.jpg, or its ~400 px thumbnail.

    The thumbnail is made on first request and again whenever page.jpg is
    newer (the worker turns an upside-down page round).
    """
    if not is_page_id(page_id):
        return None
    folder = page_dir(page_id)
    full = folder / PAGE_IMAGE
    if not full.is_file():
        return None
    if size != "thumb":
        return full
    thumb = folder / THUMB_IMAGE
    try:
        if thumb.is_file() and thumb.stat().st_mtime_ns >= full.stat().st_mtime_ns:
            return thumb
    except OSError:
        pass
    grey = cv2.imread(str(full), cv2.IMREAD_GRAYSCALE)
    if grey is None:
        return full
    height, width = grey.shape[:2]
    if width > THUMB_WIDTH:
        grey = cv2.resize(
            grey,
            (THUMB_WIDTH, max(1, round(height * THUMB_WIDTH / width))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", grey, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return full
    write_bytes_atomic(thumb, encoded.tobytes())
    return thumb


# ------------------------------------------------------------- overrides
def load_overrides(page_id: str) -> Overrides:
    """What a person has typed for a page (empty when nothing).

    A file that is there but cannot be read is not taken as nothing typed:
    Overrides.unreadable says so, and what it held is left out.
    """
    folder = page_dir(page_id)
    code, code_lost = _read_override(folder / "code.json")
    student, student_lost = _read_override(folder / "student_id.json")
    declaration, declaration_lost = _read_override(folder / "declaration.json")
    of = declaration.get("of")
    return Overrides(
        code=code.get("code") or None,
        code_by=str(code.get("by") or ""),
        student_id=student.get("student_id") or None,
        matched_student=student.get("matched_student") or None,
        student_id_by=str(student.get("by") or ""),
        declaration_of=of if is_page_id(of) else None,
        declaration_by=str(declaration.get("by") or ""),
        declaration_at=str(declaration.get("at") or ""),
        unreadable=code_lost or student_lost or declaration_lost,
    )


def _read_override(path: Path) -> tuple[dict[str, Any], bool]:
    """An override file's contents ({} when there is none), and whether it is there but could not be read."""
    data = read_json(path)
    if data is not None:
        return data, False
    try:
        return {}, path.exists()
    except OSError:
        return {}, True


def set_code_override(page_id: str, code: str | None, by: str) -> None:
    """Record (or with None, clear) a batch code typed for a page.

    Kept in a file of its own, apart from the worker's result.json, so the
    worker finishing a read can never overwrite what a person typed.
    """
    path = page_dir(page_id) / "code.json"
    if code is None:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return
    write_json_atomic(path, {"code": code, "by": by, "at": _stamp()}, durable=True)


def set_student_id_override(
    page_id: str, student_id: str | None, matched_student: dict[str, Any] | None, by: str
) -> None:
    """Record (or with None, clear) a Student ID typed for a page, with its match.

    Cleared, the file stays with no ID in it and who took it back: with
    batch codes off a page a person has corrected never goes up by itself,
    nor does any form of its stack (documents._judge_stack).
    """
    path = page_dir(page_id) / "student_id.json"
    if student_id is None:
        if path.is_file():
            write_json_atomic(
                path, {"student_id": None, "matched_student": None, "by": by, "at": _stamp()}, durable=True
            )
        return
    write_json_atomic(
        path,
        {"student_id": student_id, "matched_student": matched_student, "by": by, "at": _stamp()},
        durable=True,
    )


def set_declaration_override(page_id: str, of_page_id: str | None, by: str) -> None:
    """Record (or with None, clear) a person's word that a page is the Declaration page of another.

    Cleared, the file stays with no page in it and who took it off, as for
    a Student ID taken back (set_student_id_override).

    Args:
        page_id: The page the reader did not recognise as a Declaration page.
        of_page_id: The page scanned straight before it, whose form it is
            the Declaration page of.
        by: Who said so.
    """
    path = page_dir(page_id) / "declaration.json"
    if of_page_id is None:
        if path.is_file():
            write_json_atomic(path, {"of": None, "by": by, "at": _stamp()}, durable=True)
        return
    write_json_atomic(path, {"of": of_page_id, "by": by, "at": _stamp()}, durable=True)


# ------------------------------------------------------------- claiming
def queue_state() -> QueueState:
    """Which pages are waiting and which are being read, right now."""
    try:
        pending = frozenset(n for n in os.listdir(_pending_dir()) if is_page_id(n))
    except FileNotFoundError:
        pending = frozenset()
    reading: set[str] = set()
    try:
        for name in os.listdir(_reading_dir()):
            match = _CLAIM_NAME.fullmatch(name)
            if match:
                reading.add(match["page"])
    except FileNotFoundError:
        pass
    return QueueState(pending=pending, reading=frozenset(reading))


def page_status(meta: PageMeta, state: QueueState) -> str:
    """A page's status, from the queue's markers first and meta.json after.

    The markers are what the readers act on, so they are the truth for
    queued and reading; meta.json may lag a moment behind them.
    """
    if meta.page_id in state.reading:
        return STATUS_READING
    if meta.page_id in state.pending:
        return STATUS_QUEUED
    if meta.status == STATUS_FAILED:
        return STATUS_FAILED
    if (page_dir(meta.page_id) / "result.json").is_file():
        return STATUS_READ
    return meta.status


def is_claimed(page_id: str) -> bool:
    """Whether a reader holds the page right now (a claim marker exists)."""
    try:
        names = os.listdir(_reading_dir())
    except FileNotFoundError:
        return False
    prefix = f"{page_id}."
    return any(name.startswith(prefix) and _CLAIM_NAME.fullmatch(name) for name in names)


def seconds_since_update(meta: PageMeta) -> float:
    """How long since meta.json last changed: every claim, release and result writes it.

    Read from updated_at (India time), else created_at, else received_ns.
    """
    for stamp in (meta.updated_at, meta.created_at):
        try:
            return (now_ist() - datetime.fromisoformat(stamp)).total_seconds()
        except (TypeError, ValueError):
            continue
    return (time.time_ns() - int(meta.received_ns or 0)) / 1_000_000_000


def requeue_orphan(page_id: str, seen_updated_at: str | None = None) -> bool:
    """Queue a page again that is neither waiting, being read, read nor failed.

    Only a crash part-way through intake (after the page was stored, before
    its marker was) leaves one like that; the listing spots it and calls
    this. The listing's view of the markers is two directory reads a moment
    apart, so a page a reader let go of between them looks like an orphan
    when it is not. Everything is checked again here, and the marker made
    with O_EXCL, so a page is never both claimed and waiting - which would
    have it read twice at once.

    Args:
        page_id: The page.
        seen_updated_at: meta.json's updated_at as the caller saw it; a
            reader that has touched the page since changed it.

    Returns:
        True when a marker was put back.
    """
    meta = load_meta(page_id)
    if (
        meta is None
        or meta.status not in (STATUS_QUEUED, STATUS_READING)
        or (seen_updated_at is not None and meta.updated_at != seen_updated_at)
        or (page_dir(page_id) / "result.json").is_file()
    ):
        return False
    marker = _pending_dir() / page_id
    if marker.exists() or is_claimed(page_id):
        return False
    _pending_dir().mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(handle)
    if is_claimed(page_id):
        # A reader took it between the checks and the marker: take the
        # marker back, unless a reader has already taken that too.
        with contextlib.suppress(FileNotFoundError):
            marker.unlink()
        return False
    debug_log(f"[QUEUE] re-queued orphaned page {page_id}")
    return True


def claim_next(now_ms: int | None = None) -> Claim | None:
    """Take the oldest waiting page for reading, or None when none is waiting.

    Args:
        now_ms: The claim's time (epoch ms); now, unless a test says.

    Returns:
        The claim, held until finish_claim or release_claim.
    """
    try:
        waiting = sorted(n for n in os.listdir(_pending_dir()) if is_page_id(n))
    except FileNotFoundError:
        return None
    if not waiting:
        return None
    _reading_dir().mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000) if now_ms is None else now_ms
    for page_id in waiting:
        marker = _reading_dir() / f"{page_id}.{stamp}.{_claim_token()}"
        try:
            os.rename(_pending_dir() / page_id, marker)
        except FileNotFoundError:
            continue  # another reader took it first
        return Claim(page_id=page_id, marker=marker, claimed_ms=stamp)
    return None


def _claim_token() -> str:
    """The claim's token: this process's id, then randomness.

    Carrying the pid lets the next worker on this machine hand a dead
    reader's page back at once (reclaim_orphaned) - a restart, or a deploy,
    would otherwise leave that page waiting out the full stale timeout.
    """
    return f"{os.getpid() & 0xFFFFFFFF:08x}{secrets.token_hex(4)}"


def _pid_alive(pid: int) -> bool:
    """Whether a process with this id is running on this machine (POSIX only)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists; it just is not ours to signal
    return True


def reclaim_orphaned() -> list[str]:
    """Hand back at once the pages whose reader process has gone.

    Only claims written with a pid (16-hex tokens from _claim_token) are
    judged; anything else waits for reclaim_stale. Skipped on Windows, where
    os.kill(pid, 0) would terminate the process rather than test for it.

    Returns:
        The ids of the pages put back in the queue.
    """
    if os.name != "posix":
        return []
    try:
        names = os.listdir(_reading_dir())
    except FileNotFoundError:
        return []
    returned: list[str] = []
    for name in names:
        match = _CLAIM_NAME.fullmatch(name)
        if not match or len(match["token"]) != 16:
            continue
        pid = int(match["token"][:8], 16)
        if pid == os.getpid() or _pid_alive(pid):
            continue
        _pending_dir().mkdir(parents=True, exist_ok=True)
        try:
            os.rename(_reading_dir() / name, _pending_dir() / match["page"])
        except FileNotFoundError:
            continue  # finished, or another sweep got there first
        returned.append(match["page"])
        debug_log(f"[QUEUE] page {match['page']}: its reader (pid {pid}) has stopped; queued again")
    return returned


def still_held(claim: Claim) -> bool:
    """Whether a claim is still this reader's (not handed on as stale)."""
    return claim.marker.exists()


def finish_claim(claim: Claim) -> bool:
    """Let go of a page that has been read or has failed.

    Returns:
        False when the claim had already been handed to another reader.
    """
    try:
        claim.marker.unlink()
    except FileNotFoundError:
        return False
    return True


def release_claim(claim: Claim) -> bool:
    """Put a claimed page back in the queue to be read again later.

    Returns:
        False when the claim had already been handed to another reader.
    """
    _pending_dir().mkdir(parents=True, exist_ok=True)
    try:
        os.rename(claim.marker, _pending_dir() / claim.page_id)
    except FileNotFoundError:
        return False
    return True


def reclaim_stale(
    max_age_seconds: float = QUEUE_CLAIM_STALE_SECONDS, now_ms: int | None = None
) -> list[str]:
    """Hand back pages whose reader has held them too long (it has died).

    The claim's own name says when it was taken - a rename keeps the file's
    old modification time, so that could not be trusted - and handing it
    back is itself a rename, so two processes sweeping at once cannot both
    do it.

    Args:
        max_age_seconds: How long a claim may be held.
        now_ms: The time now (epoch ms); now, unless a test says.

    Returns:
        The ids of the pages put back in the queue.
    """
    stamp = int(time.time() * 1000) if now_ms is None else now_ms
    try:
        names = os.listdir(_reading_dir())
    except FileNotFoundError:
        return []
    returned: list[str] = []
    for name in names:
        match = _CLAIM_NAME.fullmatch(name)
        if not match or stamp - int(match["ms"]) < max_age_seconds * 1000:
            continue
        _pending_dir().mkdir(parents=True, exist_ok=True)
        try:
            os.rename(_reading_dir() / name, _pending_dir() / match["page"])
        except FileNotFoundError:
            continue  # finished, or another sweep got there first
        returned.append(match["page"])
        debug_log(f"[QUEUE] page {match['page']} was held too long by its reader; queued again")
    return returned


# ------------------------------------------------------------- one person's pages
def open_page_ids(owner_email: str) -> list[str]:
    """A person's pages that are not in a confirmed document, oldest first."""
    try:
        names = os.listdir(owner_dir(owner_email) / "open")
    except FileNotFoundError:
        return []
    return sorted(name for name in names if is_page_id(name))


def close_pages(owner_email: str, page_ids: list[str]) -> None:
    """Take pages out of a person's open list once their document is confirmed."""
    index = owner_dir(owner_email) / "open"
    for page_id in page_ids:
        with contextlib.suppress(FileNotFoundError):
            (index / page_id).unlink()


@contextlib.contextmanager
def _being_stored(owner_email: str, scan: Storing) -> Iterator[Path]:
    """Mark one of a person's scans as being stored while the block runs (storing), yielding the marker.

    The marker says which scan page sent it and for which school and city,
    so only the stacks it may touch wait for it. It is in place, whole,
    before the block starts.
    """
    folder = owner_dir(owner_email) / "storing"
    marker = folder / secrets.token_hex(8)
    write_json_atomic(marker, asdict(scan))
    try:
        yield marker
    finally:
        with contextlib.suppress(FileNotFoundError):
            marker.unlink()


def storing(owner_email: str) -> list[Storing | None]:
    """The scans of this person's being stored right now: received, their pages not listed yet.

    intake marks a scan before it stamps its pages received and unmarks it
    only once they are on the open list. So a look that reads the clock,
    then this, then the open list either sees every page that had arrived by
    its clock or is told here that one is still on its way - from which scan
    page, for which school and city; None for one whose marker could not be
    read, which may be for anything. A marker nobody has touched for
    QUEUE_CLAIM_STALE_SECONDS was left by a process that died part-way
    (intake touches it after every page), and is cleared.
    """
    folder = owner_dir(owner_email) / "storing"
    try:
        entries = list(os.scandir(folder))
    except FileNotFoundError:
        return []
    scans: list[Storing | None] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue  # a marker still being written: its scan is not received yet
        try:
            age = time.time() - entry.stat().st_mtime
        except FileNotFoundError:
            continue  # stored meanwhile
        if age >= QUEUE_CLAIM_STALE_SECONDS:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(entry.path)
            continue
        data = read_json(Path(entry.path))
        if data is None:
            if os.path.exists(entry.path):
                # There, but not to be read: it may be for anything.
                scans.append(None)
            continue  # or stored meanwhile, so listed by now
        scans.append(
            Storing(
                sender=str(data.get("sender") or ""),
                school=str(data.get("school") or ""),
                location=str(data.get("location") or ""),
            )
        )
    return scans


# ------------------------------------------------------------- confirmed documents
def document_record_path(owner_email: str, key: str) -> Path:
    """Where a confirmed document's record is kept."""
    if not is_document_key(key):
        raise ValueError("not a document key")
    return owner_dir(owner_email) / "docs" / f"{key}.json"


def save_document_record(owner_email: str, record: dict[str, Any]) -> None:
    """Write a confirmed document's record."""
    record["updated_at"] = _stamp()
    write_json_atomic(document_record_path(owner_email, str(record["key"])), record)


def load_document_record(owner_email: str, key: str) -> dict[str, Any] | None:
    """One confirmed document's record, or None."""
    if not is_document_key(key):
        return None
    return read_json(document_record_path(owner_email, key))


def load_document_records(owner_email: str, since_hours: float | None = None) -> list[dict[str, Any]]:
    """A person's confirmed documents.

    Args:
        owner_email: Whose.
        since_hours: Only finished ones written in the last this-many
            hours; one still waiting for its upload or its History row is
            always included, so its Retry upload stays on the page.

    Returns:
        The records, newest first.
    """
    listed, _older = document_records(owner_email, since_hours)
    return listed


def document_records(
    owner_email: str, since_hours: float | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A person's confirmed documents: those to list, and the finished ones from before.

    Every record is read either way, so one read gives both. The listing
    shows only the first; with batch codes off the pairing needs the pages
    of every one, however old (documents._pages_beside), or a Declaration
    page would pair differently once the form before it dropped off the
    list.

    Args:
        owner_email: Whose.
        since_hours: As for load_document_records.

    Returns:
        The records load_document_records gives, and the finished ones
        written before `since_hours` - each newest first.
    """
    folder = owner_dir(owner_email) / "docs"
    try:
        entries = list(os.scandir(folder))
    except FileNotFoundError:
        return [], []
    cutoff = None if since_hours is None else time.time() - since_hours * 3600
    listed: list[tuple[float, dict[str, Any]]] = []
    older: list[tuple[float, dict[str, Any]]] = []
    for entry in entries:
        if not entry.name.endswith(".json") or entry.name.startswith("."):
            continue
        record = read_json(Path(entry.path))
        if not record:
            continue
        try:
            modified = entry.stat().st_mtime
        except OSError:
            continue
        if cutoff is not None and modified < cutoff and document_record_finished(record):
            older.append((modified, record))
        else:
            listed.append((modified, record))
    listed.sort(key=lambda item: item[0], reverse=True)
    older.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in listed], [record for _, record in older]


def document_record_history_pending(record: dict[str, Any]) -> bool:
    """Whether a confirmed document still owes History its row, or the row its link.

    The student was marked when it was confirmed; a database drop after
    that leaves the ocr_scan_log row unwritten (log_pending), or written
    before the upload without the Drive link it now has.
    """
    if record.get("log_pending"):
        return True
    link = (record.get("drive") or {}).get("link")
    return bool(link) and bool(record.get("log_id")) and not record.get("log_has_link")


def document_record_finished(record: dict[str, Any]) -> bool:
    """Whether a confirmed document is in Drive and in History: nothing left to retry."""
    return record.get("status") == "uploaded" and not document_record_history_pending(record)


def _take_lock(path: Path, stale_seconds: float) -> bool:
    """Make the lock file with O_EXCL, clearing one left by a holder that died.

    Returns:
        False when a live holder has it.
    """
    for _ in range(2):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < stale_seconds:
                return False
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            continue
        os.close(handle)
        return True
    return False


def _lock_path(owner_email: str, name: str) -> Path:
    folder = owner_dir(owner_email) / "locks"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.lock"


@contextlib.contextmanager
def confirm_lock(owner_email: str, code: str) -> Iterator[None]:
    """Hold the one-at-a-time lock for confirming a person's document.

    A double click, or two tabs, would otherwise mark the student twice,
    log two rows and upload two PDFs. Taken with O_EXCL, so it works across
    gunicorn workers.

    Raises:
        DocumentBusy: Another confirm of this document is running.
    """
    path = _lock_path(owner_email, code)
    if not _take_lock(path, CONFIRM_LOCK_STALE_SECONDS):
        raise DocumentBusy(code)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


@contextlib.contextmanager
def drive_folders_lock(
    owner_email: str, wait_seconds: float = DRIVE_FOLDERS_WAIT_SECONDS
) -> Iterator[None]:
    """Hold the lock for finding or making a person's Drive folders, waiting for it.

    Confirms of different documents run side by side (each has its own
    confirm_lock), and two for one class would each search for the class's
    folders, find none and make their own. Taken with O_EXCL like
    confirm_lock, so it holds across threads and gunicorn workers alike.

    Args:
        owner_email: Whose Drive.
        wait_seconds: How long to wait for another confirm to finish its
            folders.

    Raises:
        DocumentBusy: Still held by a live confirm after `wait_seconds`.
    """
    path = _lock_path(owner_email, DRIVE_FOLDERS_LOCK)
    deadline = time.monotonic() + wait_seconds
    while not _take_lock(path, CONFIRM_LOCK_STALE_SECONDS):
        if time.monotonic() >= deadline:
            raise DocumentBusy(DRIVE_FOLDERS_LOCK)
        time.sleep(_LOCK_POLL_SECONDS)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
