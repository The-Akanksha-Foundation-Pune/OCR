"""Read scanned pages out of each person's own Google Drive inbox folder.

The scanning PC never talks to this server. Its scanner app saves each page
into a folder called "OCR Inbox" in the person's own Drive, which Google Drive
for Desktop keeps in sync. The person shares that one folder with the server's
service account; the server reads it and nothing else. Nobody sees anyone
else's scans, and the route is the same on Windows and Mac because both sides
only need software they already have.

    My Drive/
        OCR Inbox/                          <- shared with the service account
            page-001.jpg                    <- lands here from the scanner app
            [saved] Aatif Khan_AATKHA200618.jpg   <- what it becomes once read
            [not found] AARSAY150821.jpg

Files are renamed in place rather than moved: the service account owns no
Drive storage, so it cannot create folders, but it can rename what it has
been given access to. The prefix in square brackets is also how a page that
has already been read is told apart from one that has not.

This module only knows Drive. What to do with a page is the watcher's job.
"""

from __future__ import annotations

import time
from typing import Any

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from server.config import DRIVE_INBOX_FOLDER_NAME, VISION_CREDENTIALS
from server.services.debug_log import debug_log

DRIVE_API = "https://www.googleapis.com/drive/v3"
FOLDER_MIME = "application/vnd.google-apps.folder"

# A processed page is renamed to start with one of these.
DONE_PREFIX = "["

# Same key file Cloud Vision uses. One credential to manage on the server.
_SCOPES = ["https://www.googleapis.com/auth/drive"]

# Shared Drives are invisible to the API unless every call opts in.
_ALL_DRIVES = {"supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}

_credentials: service_account.Credentials | None = None


class DriveInboxError(RuntimeError):
    """Drive said no, or is not configured. Message is fit to show."""


def configured() -> tuple[bool, str]:
    if not VISION_CREDENTIALS:
        return False, "GOOGLE_VISION_CREDENTIALS is not set (the Drive inbox uses the same key)."
    return True, ""


def service_account_email() -> str:
    """The address a person shares their folder with."""
    return _creds().service_account_email


def _creds() -> service_account.Credentials:
    global _credentials
    if _credentials is None:
        ok, why = configured()
        if not ok:
            raise DriveInboxError(why)
        _credentials = service_account.Credentials.from_service_account_file(
            VISION_CREDENTIALS, scopes=_SCOPES
        )
    return _credentials


def _token() -> str:
    creds = _creds()
    # Tokens last an hour; refresh a little early rather than racing expiry.
    if not creds.valid or (
        creds.expiry is not None and (creds.expiry.timestamp() - time.time()) < 120
    ):
        creds.refresh(Request())
    return creds.token


def _call(method: str, path: str, **kwargs: Any) -> requests.Response:
    params = dict(kwargs.pop("params", {}) or {})
    params.update(_ALL_DRIVES)
    headers = dict(kwargs.pop("headers", {}) or {})
    headers["Authorization"] = f"Bearer {_token()}"
    response = requests.request(
        method, f"{DRIVE_API}/{path}", params=params, headers=headers,
        timeout=kwargs.pop("timeout", 60), **kwargs,
    )
    if not response.ok:
        try:
            detail = response.json().get("error", {}).get("message", "")
        except ValueError:
            detail = response.text[:200]
        raise DriveInboxError(f"Drive {method} {path} -> {response.status_code}: {detail}")
    return response


def _list(query: str, fields: str, order_by: str | None = None) -> list[dict[str, Any]]:
    """Every file matching `query`, following pagination."""
    items: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        params: dict[str, Any] = {
            "q": query,
            "fields": f"nextPageToken, files({fields})",
            "pageSize": 200,
        }
        if order_by:
            params["orderBy"] = order_by
        if token:
            params["pageToken"] = token
        body = _call("GET", "files", params=params).json()
        items.extend(body.get("files", []))
        token = body.get("nextPageToken")
        if not token:
            return items


# ---------------------------------------------------------------- inboxes
def list_inboxes() -> list[dict[str, str]]:
    """Every "OCR Inbox" folder someone has shared with the service account,
    with who owns it. The owner is how a folder is tied to a signed-in user."""
    safe_name = DRIVE_INBOX_FOLDER_NAME.replace("'", "\\'")
    folders = _list(
        f"name = '{safe_name}' and mimeType = '{FOLDER_MIME}' "
        f"and sharedWithMe = true and trashed = false",
        fields="id, name, owners(emailAddress, displayName), sharedWithMeTime",
        order_by="sharedWithMeTime desc",
    )
    out = []
    for folder in folders:
        owner = (folder.get("owners") or [{}])[0]
        out.append(
            {
                "id": folder["id"],
                "name": folder["name"],
                "owner_email": (owner.get("emailAddress") or "").lower(),
                "owner_name": owner.get("displayName") or "",
                "shared_at": folder.get("sharedWithMeTime") or "",
            }
        )
    return out


def find_inbox(owner_email: str) -> dict[str, str] | None:
    """The inbox folder this person shared, or None if they have not yet."""
    wanted = (owner_email or "").strip().lower()
    if not wanted:
        return None
    for folder in list_inboxes():  # newest share first
        if folder["owner_email"] == wanted:
            return folder
    return None


# ------------------------------------------------------------------ pages
def list_new_pages(folder_id: str) -> list[dict[str, Any]]:
    """Scanned files waiting in an inbox, oldest first.

    Only images and PDFs - the scanner apps produce nothing else, and anything
    else that lands here (a stray .tmp, a desktop.ini) is not a page. Files
    already renamed with a [result] prefix are the ones we have read.
    """
    files = _list(
        f"'{folder_id}' in parents and trashed = false and "
        f"(mimeType contains 'image/' or mimeType = 'application/pdf')",
        fields="id, name, mimeType, size, createdTime, webViewLink",
        order_by="createdTime",
    )
    return [f for f in files if not (f.get("name") or "").startswith(DONE_PREFIX)]


def download(file_id: str) -> bytes:
    return _call("GET", f"files/{file_id}", params={"alt": "media"}, timeout=120).content


def file_link(file_id: str) -> str:
    """A view link that survives renaming the file."""
    return f"https://drive.google.com/file/d/{file_id}/view"


def mark_done(file_id: str, new_name: str) -> str:
    """Rename a page to its result, e.g. "[saved] Aatif Khan_AATKHA200618.jpg".
    Returns the file's link, which the History row keeps."""
    if not new_name.startswith(DONE_PREFIX):
        new_name = f"[done] {new_name}"
    _call("PATCH", f"files/{file_id}", params={"fields": "id"}, json={"name": new_name})
    debug_log(f"[INBOX] renamed {file_id} -> {new_name!r}")
    return file_link(file_id)
