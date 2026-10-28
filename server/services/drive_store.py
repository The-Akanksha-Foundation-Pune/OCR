"""Save scanned uploads to the signed-in user's Google Drive OCR folder.

Required layout:
  My Drive / OCR / YYYY-MM-DD / StudentName_STUDENTID.jpg     (one page, /api/confirm)
  My Drive / OCR / <School> / <Grade> / <Division> / <StudentName>_<StudentID> /
      <StudentName>_<StudentID>_<Code>.pdf                    (a confirmed document)
"""

from __future__ import annotations

import contextlib
import json
import re
import secrets
import time
import unicodedata
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from flask import session

from server.config import (
    DRIVE_OCR_FOLDER_NAME,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
)
from server.services.clock import now_ist
from server.services.debug_log import debug_log

DRIVE_API = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pdf": "application/pdf",
}


def upload_scan_to_drive(
    data: bytes,
    filename: str,
    *,
    display_name: str | None = None,
) -> dict[str, Any] | None:
    """Upload into My Drive / OCR / <today> / <name>.jpg and enforce that path.

    Never raises — OCR must keep working even if Drive is unavailable.
    """
    if not data or not filename:
        return None

    try:
        access_token = _get_valid_access_token()
        if not access_token:
            debug_log("[DRIVE] skipped — no Google token in session (re-login required)")
            return None

        ocr_folder_id = _ensure_folder(
            access_token,
            name=DRIVE_OCR_FOLDER_NAME,
            parent_id="root",
            cache_key="drive_ocr_folder_id",
        )
        date_folder_name = now_ist().strftime("%Y-%m-%d")
        date_folder_id = _ensure_today_folder(
            access_token,
            parent_id=ocr_folder_id,
            date_name=date_folder_name,
        )

        mime_type = _mime_for_filename(filename)
        suffix = Path(filename).suffix.lower() or ".bin"
        if display_name:
            stem = _safe_drive_stem(display_name)
            # Only a real file extension counts as one already there. A name
            # with an initial ("Ramesh S. Patil") has a "." that Path() reads
            # as the start of a suffix, which used to leave the file with no
            # .jpg at all.
            if Path(stem).suffix.lower() in _MIME_BY_SUFFIX:
                drive_name = stem
            else:
                drive_name = f"{stem}{suffix}"
        else:
            stamp = now_ist().strftime("%H%M%S")
            drive_name = f"{stamp}_{filename}"

        # 1) Upload bytes first (lands in Drive root by default)
        created = requests.post(
            f"{DRIVE_UPLOAD_API}/files",
            params={"uploadType": "media", "fields": "id,name,parents,webViewLink"},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": mime_type,
            },
            data=data,
            timeout=60,
        )
        if not created.ok:
            debug_log(
                f"[DRIVE] upload failed status={created.status_code} body={created.text[:300]!r}"
            )
            return None

        file_meta = created.json()
        file_id = file_meta.get("id")
        if not file_id:
            debug_log("[DRIVE] upload returned no file id")
            return None

        # 2) Rename + move into OCR / YYYY-MM-DD (remove any other parents)
        old_parents = file_meta.get("parents") or ["root"]
        remove_parents = ",".join(p for p in old_parents if p != date_folder_id) or "root"

        moved = requests.patch(
            f"{DRIVE_API}/files/{file_id}",
            params={
                "addParents": date_folder_id,
                "removeParents": remove_parents,
                "fields": "id,name,parents,webViewLink",
            },
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"name": drive_name},
            timeout=30,
        )
        if not moved.ok:
            debug_log(
                f"[DRIVE] move into date folder failed status={moved.status_code} "
                f"body={moved.text[:300]!r}"
            )
            return file_meta

        payload = moved.json()
        parents = payload.get("parents") or []
        in_date_folder = date_folder_id in parents
        debug_log(
            f"[DRIVE] saved name={payload.get('name')!r} id={payload.get('id')!r} "
            f"path={DRIVE_OCR_FOLDER_NAME}/{date_folder_name} "
            f"parents={parents!r} in_date_folder={in_date_folder} "
            f"link={payload.get('webViewLink')!r}"
        )
        payload["folder_path"] = f"{DRIVE_OCR_FOLDER_NAME}/{date_folder_name}"
        payload["date_folder_id"] = date_folder_id
        payload["ocr_folder_id"] = ocr_folder_id
        return payload
    except Exception as exc:  # noqa: BLE001 — never break the OCR flow
        debug_log(f"[DRIVE] upload error: {exc}")
        return None


FOLDER_MIME = "application/vnd.google-apps.folder"


class DriveUploadError(RuntimeError):
    """A document could not be put in Drive. The message is safe to show."""


class NoDriveToken(DriveUploadError):
    """The session holds no Google token (signed in through Tether SSO)."""


@dataclass(frozen=True)
class DriveUpload:
    """Where a document's PDF went.

    Attributes:
        file_id: Its Drive id.
        name: Its name in Drive (with " (2)" when the name was taken).
        link: Its webViewLink.
        folder_path: The folders it is in, from My Drive down.
    """

    file_id: str
    name: str
    link: str
    folder_path: str


def has_drive_credentials() -> bool:
    """Whether this session can upload to the person's own Drive.

    Only a Google sign-in leaves a token in the session; a Tether SSO
    hand-off proves who the person is but carries no Google credentials.
    """
    token = session.get("google_token")
    return isinstance(token, dict) and bool(token.get("access_token") or token.get("refresh_token"))


def drive_safe_name(name: str, fallback: str) -> str:
    """A folder or file name Drive and every synced computer will take.

    Control characters go, slashes and the other characters Windows refuses
    in a file name become "-", runs of spaces become one, and trailing dots
    and spaces go (Windows drops them, so Drive for Desktop would not match).

    Args:
        name: The name as it came from the database.
        fallback: What to use when nothing is left.

    Returns:
        The cleaned name, at most 120 characters.
    """
    text = unicodedata.normalize("NFC", name or "")
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r'[\\/:*?"<>|]+', "-", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(". ")
    return text[:120].rstrip(". ") or fallback


def upload_document_pdf(
    pdf: bytes,
    folders: list[str],
    file_name: str,
    *,
    folder_lock: Callable[[], AbstractContextManager[Any]] | None = None,
) -> DriveUpload:
    """Put a document's PDF in the signed-in person's Drive, under `folders`.

    Each folder is found by name under the one before (starting at My
    Drive) and only created when it is not there, so a retry or the next
    form for the same class reuses the folders. A file of the same name
    already in the last folder is kept, and this one is named " (2)",
    " (3)" and so on. The PDF goes up in one request carrying its name and
    folder, so a failure never leaves a stray file in My Drive.

    Args:
        pdf: The PDF.
        folders: Folder names from the top (already drive_safe_name'd).
        file_name: The PDF's name, ending ".pdf".
        folder_lock: Makes the lock held while the folders are found or
            made. Two uploads for one class at once would otherwise both
            find no folder and both make one.

    Returns:
        Where it went.

    Raises:
        NoDriveToken: The session has no Google token.
        DriveUploadError: Drive refused or could not be reached.
    """
    if not has_drive_credentials():
        raise NoDriveToken("Sign in with Google to upload to Drive.")
    try:
        access_token = _get_valid_access_token()
    except requests.RequestException as exc:
        raise DriveUploadError("Could not reach Google to renew the sign-in.") from exc
    if not access_token:
        raise NoDriveToken("Sign in with Google to upload to Drive.")

    try:
        parent = "root"
        with folder_lock() if folder_lock is not None else contextlib.nullcontext():
            for name in folders:
                parent = _find_or_create_folder(access_token, name, parent)
        final_name = _free_name(access_token, parent, file_name)
        created = _multipart_upload(access_token, parent, final_name, pdf)
    except requests.RequestException as exc:
        debug_log(f"[DRIVE] document upload could not reach Drive: {type(exc).__name__}: {exc}")
        raise DriveUploadError("Could not reach Google Drive.") from exc

    path = "/".join(folders)
    debug_log(f"[DRIVE] document saved name={created.get('name')!r} path={path!r} id={created.get('id')!r}")
    return DriveUpload(
        file_id=str(created.get("id") or ""),
        name=str(created.get("name") or final_name),
        link=str(created.get("webViewLink") or ""),
        folder_path=path,
    )


def _drive_query_text(value: str) -> str:
    """A value quoted for a Drive search: backslashes and quotes escaped."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _list_files(access_token: str, query: str, order_by: str | None = None) -> list[dict[str, Any]]:
    """Every file a Drive search finds (following pages), in `order_by` order if given."""
    found: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "q": query,
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name)",
            "pageSize": 1000,
        }
        if order_by:
            params["orderBy"] = order_by
        if page_token:
            params["pageToken"] = page_token
        listed = requests.get(
            f"{DRIVE_API}/files",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=20,
        )
        if not listed.ok:
            debug_log(f"[DRIVE] search failed status={listed.status_code} body={listed.text[:200]!r}")
            raise DriveUploadError("Google Drive refused to list your folders.")
        body = listed.json() or {}
        found.extend(body.get("files") or [])
        page_token = body.get("nextPageToken")
        if not page_token:
            return found


def _find_or_create_folder(access_token: str, name: str, parent_id: str) -> str:
    """The id of folder `name` under `parent_id`, created only when missing.

    A failed search raises rather than creating: creating after a search
    that merely failed is how a second "OCR" folder appears. When there are
    two of one name already, the oldest is used, so every upload settles on
    the same one - Drive lists them in no set order otherwise.
    """
    query = (
        f"name='{_drive_query_text(name)}' and mimeType='{FOLDER_MIME}' "
        f"and trashed=false and '{_drive_query_text(parent_id)}' in parents"
    )
    existing = _list_files(access_token, query, order_by="createdTime")
    if existing:
        return str(existing[0]["id"])
    created = requests.post(
        f"{DRIVE_API}/files",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        params={"fields": "id,name"},
        json={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
        timeout=20,
    )
    if not created.ok:
        debug_log(f"[DRIVE] folder create failed status={created.status_code} body={created.text[:200]!r}")
        raise DriveUploadError("Google Drive refused to create a folder.")
    folder_id = str((created.json() or {}).get("id") or "")
    if not folder_id:
        raise DriveUploadError("Google Drive did not say which folder it created.")
    debug_log(f"[DRIVE] created folder {name!r} under {parent_id!r}")
    return folder_id


def _free_name(access_token: str, folder_id: str, file_name: str) -> str:
    """`file_name`, or "<stem> (2).pdf" and on, whichever the folder lacks."""
    taken = {
        str(entry.get("name") or "")
        for entry in _list_files(
            access_token,
            f"'{_drive_query_text(folder_id)}' in parents and trashed=false",
        )
    }
    stem, dot, suffix = file_name.rpartition(".")
    if not dot:
        stem, suffix = file_name, ""
    candidate = file_name
    number = 2
    while candidate in taken:
        candidate = f"{stem} ({number}).{suffix}" if suffix else f"{stem} ({number})"
        number += 1
    return candidate


def _multipart_upload(access_token: str, folder_id: str, name: str, pdf: bytes) -> dict[str, Any]:
    """Create the PDF, named and in its folder, in a single request."""
    boundary = f"ocr-{secrets.token_hex(12)}"
    metadata = json.dumps({"name": name, "parents": [folder_id], "mimeType": "application/pdf"})
    body = (
        f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{metadata}\r\n"
        f"--{boundary}\r\nContent-Type: application/pdf\r\n\r\n"
    ).encode() + pdf + f"\r\n--{boundary}--\r\n".encode("ascii")
    created = requests.post(
        f"{DRIVE_UPLOAD_API}/files",
        params={"uploadType": "multipart", "fields": "id,name,parents,webViewLink"},
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": f"multipart/related; boundary={boundary}",
        },
        data=body,
        timeout=120,
    )
    if not created.ok:
        debug_log(f"[DRIVE] document upload failed status={created.status_code} body={created.text[:300]!r}")
        raise DriveUploadError("Google Drive refused the upload.")
    payload = created.json() or {}
    if not payload.get("id"):
        raise DriveUploadError("Google Drive did not say which file it created.")
    return payload


def _safe_drive_stem(name: str) -> str:
    cleaned = re.sub(r"[^\w\s\-.]+", "", (name or "").strip(), flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    return cleaned[:120] or "scan"


def _get_valid_access_token() -> str | None:
    token = session.get("google_token")
    if not isinstance(token, dict):
        return None

    access_token = token.get("access_token")
    expires_at = float(token.get("expires_at") or 0)
    if access_token and expires_at and time.time() < (expires_at - 60):
        return access_token

    refresh_token = token.get("refresh_token")
    if not refresh_token:
        return access_token or None

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=20,
    )
    if not response.ok:
        debug_log(f"[DRIVE] token refresh failed: {response.status_code} {response.text[:200]!r}")
        return access_token or None

    refreshed = response.json()
    new_access = refreshed.get("access_token")
    if not new_access:
        return access_token or None

    expires_in = int(refreshed.get("expires_in") or 3600)
    token["access_token"] = new_access
    token["expires_at"] = time.time() + expires_in
    if refreshed.get("refresh_token"):
        token["refresh_token"] = refreshed["refresh_token"]
    session["google_token"] = token
    session.modified = True
    return new_access


def _ensure_today_folder(access_token: str, parent_id: str, date_name: str) -> str:
    """Return today's date subfolder under OCR, creating it if missing."""
    cached = session.get("drive_ocr_date_folder")
    if (
        isinstance(cached, dict)
        and cached.get("date") == date_name
        and cached.get("id")
        and cached.get("parent_id") == parent_id
    ):
        return cached["id"]

    folder_id = _ensure_folder(
        access_token,
        name=date_name,
        parent_id=parent_id,
        cache_key=None,
    )
    session["drive_ocr_date_folder"] = {
        "date": date_name,
        "id": folder_id,
        "parent_id": parent_id,
    }
    session.modified = True
    return folder_id


def _ensure_folder(
    access_token: str,
    *,
    name: str,
    parent_id: str,
    cache_key: str | None,
) -> str:
    """Find or create a Drive folder named `name` under `parent_id`."""
    if cache_key:
        cached = session.get(cache_key)
        if cached:
            return cached

    safe_name = name.replace("'", "\\'")
    query = (
        f"name='{safe_name}' "
        "and mimeType='application/vnd.google-apps.folder' "
        "and trashed=false "
        f"and '{parent_id}' in parents"
    )
    listed = requests.get(
        f"{DRIVE_API}/files",
        headers={"Authorization": f"Bearer {access_token}"},
        params={
            "q": query,
            "spaces": "drive",
            "fields": "files(id,name)",
            "pageSize": 1,
            "corpora": "user",
        },
        timeout=20,
    )
    if listed.ok:
        files = (listed.json() or {}).get("files") or []
        if files:
            folder_id = files[0]["id"]
            if cache_key:
                session[cache_key] = folder_id
                session.modified = True
            debug_log(f"[DRIVE] found folder {name!r} under parent={parent_id!r} id={folder_id!r}")
            return folder_id

    created = requests.post(
        f"{DRIVE_API}/files",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        },
        params={"fields": "id,name,parents,webViewLink"},
        timeout=20,
    )
    created.raise_for_status()
    body = created.json()
    folder_id = body["id"]
    if cache_key:
        session[cache_key] = folder_id
        session.modified = True
    debug_log(
        f"[DRIVE] created folder {name!r} under parent={parent_id!r} "
        f"id={folder_id!r} parents={body.get('parents')!r}"
    )
    return folder_id


def _mime_for_filename(filename: str) -> str:
    return _MIME_BY_SUFFIX.get(Path(filename).suffix.lower(), "application/octet-stream")
