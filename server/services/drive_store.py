"""Save scanned uploads to the signed-in user's Google Drive OCR folder.

Required layout:
  My Drive / OCR / YYYY-MM-DD / StudentName_STUDENTID.jpg
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from flask import session

from server.config import (
    DRIVE_OCR_FOLDER_NAME,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
)
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
        date_folder_name = datetime.now().strftime("%Y-%m-%d")
        date_folder_id = _ensure_today_folder(
            access_token,
            parent_id=ocr_folder_id,
            date_name=date_folder_name,
        )

        mime_type = _mime_for_filename(filename)
        suffix = Path(filename).suffix.lower() or ".bin"
        if display_name:
            stem = _safe_drive_stem(display_name)
            drive_name = f"{stem}{suffix}" if not Path(stem).suffix else stem
        else:
            stamp = datetime.now().strftime("%H%M%S")
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
