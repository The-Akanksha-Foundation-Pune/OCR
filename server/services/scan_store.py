"""Temporarily stash scanned uploads until the user confirms a Student ID."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import cv2

from server.config import UPLOAD_FOLDER
from server.services.debug_log import debug_log
from server.services.preprocess import load_pages_from_bytes

_META_SUFFIX = ".meta"


def stash_upload(data: bytes, filename: str) -> str:
    """Save upload bytes under a unique scan_id; return that id."""
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    scan_id = uuid.uuid4().hex
    safe_name = Path(filename).name or "scan.bin"
    path = UPLOAD_FOLDER / f"{scan_id}__{safe_name}"
    path.write_bytes(data)
    meta = UPLOAD_FOLDER / f"{scan_id}{_META_SUFFIX}"
    meta.write_text(safe_name, encoding="utf-8")
    debug_log(f"[SCAN] stashed scan_id={scan_id!r} file={safe_name!r} bytes={len(data)}")
    return scan_id


def load_stashed(scan_id: str) -> tuple[bytes, str] | None:
    """Return (bytes, original_filename) for a stashed scan, or None."""
    cleaned = (scan_id or "").strip()
    if not cleaned or not re.fullmatch(r"[0-9a-f]{32}", cleaned):
        return None

    matches = list(UPLOAD_FOLDER.glob(f"{cleaned}__*"))
    if not matches:
        return None
    path = matches[0]
    filename = path.name.split("__", 1)[-1]
    return path.read_bytes(), filename


def page_image_jpeg(scan_id: str, page_number: int = 1) -> tuple[bytes, str] | None:
    """Return JPEG bytes for one page of a stashed scan (1-based page index).

    Images become a single JPEG; PDFs render the requested page.
    """
    loaded = load_stashed(scan_id)
    if loaded is None:
        return None

    data, filename = loaded
    try:
        pages = load_pages_from_bytes(data, filename)
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[SCAN] could not decode scan_id={scan_id!r}: {exc}")
        return None

    if not pages:
        return None

    index = max(1, page_number) - 1
    if index >= len(pages):
        index = 0

    ok, encoded = cv2.imencode(".jpg", pages[index], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return None
    return encoded.tobytes(), "jpg"


def discard_stashed(scan_id: str) -> None:
    """Best-effort delete of a stashed scan (and its meta file)."""
    cleaned = (scan_id or "").strip()
    if not cleaned:
        return
    for path in UPLOAD_FOLDER.glob(f"{cleaned}*"):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            debug_log(f"[SCAN] could not delete {path}: {exc}")
