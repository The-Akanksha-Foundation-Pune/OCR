"""The web app's own fixes: routes, sign-in, cross-site protection and the page.

Bugs covered here (numbers from the bug list): 7, 11, 17, 22, 23, 24, 25, 27
and the page side of 28. The page tests run the real Scanner page script,
rendered by Flask, under Node with a small stand-in DOM and a fetch whose
answers each test hands out - so they check what the browser would send,
not what the template happens to contain.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from flask import Flask, redirect, session
from flask.testing import FlaskClient

# A stashed upload's id: 32 hex digits, like scan_store makes.
SCAN_ID = "a1" * 16
DB_HOST = "secret-host.cluster-xyz.ap-south-1.rds.amazonaws.com"
PROXY_HEADERS = {
    "X-Forwarded-Proto": "https",
    "X-Forwarded-Host": "tether.akanksha.org",
    "X-Forwarded-Prefix": "/ocr",
}


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def debug_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Keep debug_log out of the repository's logs/ and let tests read it."""
    import server  # noqa: F401 - loads every module that logs
    import server.services.debug_log as debug_module

    lines: list[str] = []
    monkeypatch.setattr(debug_module, "LOG_DIR", tmp_path, raising=False)
    monkeypatch.setattr(debug_module, "LOG_FILE", tmp_path / "ocr_debug.log", raising=False)
    original = debug_module.debug_log
    for name, module in list(sys.modules.items()):
        if not name.startswith("server") or module is None:
            continue
        if getattr(module, "debug_log", None) is original:
            monkeypatch.setattr(module, "debug_log", lines.append)
    return lines


@pytest.fixture
def warm_ups(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count warm-ups, patched where conftest patches it: the warmup module."""
    from server.services import warmup

    calls: list[str] = []
    monkeypatch.setattr(warmup, "warm_models_in_background", lambda: calls.append("warm"))
    return calls


# ------------------------------------------- test hygiene: the model warm-up
def test_create_app_uses_the_warm_up_conftest_patches(warm_ups: list[str]) -> None:
    """Every app build goes through the module attribute conftest replaces.

    create_app used to call its own imported copy, so the suite-wide patch
    missed it: each app any other test file built started the real RapidOCR
    and Vision load, and wrote [WARMUP] lines into the repository's log.
    """
    from server import create_app

    create_app()

    assert warm_ups == ["warm"]


def test_create_app_still_honours_a_patch_of_its_own_name(
    monkeypatch: pytest.MonkeyPatch, warm_ups: list[str]
) -> None:
    """test_watcher.py patches server.warm_models_in_background; that keeps working."""
    import server

    mine: list[str] = []
    monkeypatch.setattr(server, "warm_models_in_background", lambda: mine.append("warm"))
    server.create_app()

    assert mine == ["warm"]
    assert warm_ups == []


@pytest.fixture
def stash(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for stash_upload: record the stored name, write nothing."""
    from server.blueprints import ocr_routes

    names: list[str] = []

    def fake_stash(_data: bytes, filename: str) -> str:
        names.append(filename)
        return SCAN_ID

    monkeypatch.setattr(ocr_routes, "stash_upload", fake_stash)
    return names


def _raiser(exc: BaseException) -> Callable[..., Any]:
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise exc

    return boom


def _upload(client: FlaskClient, url: str, filename: str, **kwargs: Any) -> Any:
    return client.post(
        url,
        data={"file": (BytesIO(b"\xff\xd8 not really a jpeg"), filename)},
        content_type="multipart/form-data",
        **kwargs,
    )


# ------------------------------------------- #7 / C4: unreadable pages keep scan_id
def test_extract_422_still_returns_scan_id(client: FlaskClient, stash: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from server.blueprints import ocr_routes
    from server.services.extract_student_id import ExtractError

    monkeypatch.setattr(
        ocr_routes,
        "extract_student_id_from_upload",
        _raiser(ExtractError("Could not read a valid Student ID.", status_code=422)),
    )
    response = _upload(client, "/api/extract", "page-001.jpg")

    assert response.status_code == 422
    body = response.get_json()
    assert body["error"] == "Could not read a valid Student ID."
    assert body["scan_id"] == SCAN_ID
    assert body["page_number"] == 1


def test_extract_crash_returns_scan_id_without_exception_text(
    client: FlaskClient, stash: list[str], monkeypatch: pytest.MonkeyPatch, debug_lines: list[str]
) -> None:
    from server.blueprints import ocr_routes

    monkeypatch.setattr(
        ocr_routes, "extract_student_id_from_upload", _raiser(RuntimeError(f"boom at {DB_HOST}"))
    )
    response = _upload(client, "/api/extract", "page-001.jpg")

    assert response.status_code == 500
    body = response.get_json()
    assert body["scan_id"] == SCAN_ID
    assert DB_HOST not in response.get_data(as_text=True)
    assert any(DB_HOST in line for line in debug_lines)


def test_extract_batch_error_returns_scan_id(client: FlaskClient, stash: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from server.blueprints import ocr_routes
    from server.services.extract_student_id import ExtractError

    monkeypatch.setattr(
        ocr_routes,
        "extract_student_ids_from_upload_batch",
        _raiser(ExtractError("No pages found in upload.", status_code=400)),
    )
    response = _upload(client, "/api/extract-batch", "stack.pdf")

    assert response.status_code == 400
    assert response.get_json()["scan_id"] == SCAN_ID


# ------------------------------------------------- #28 / C2: route side
def test_schools_error_hides_database_detail(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch, debug_lines: list[str]
) -> None:
    from server.blueprints import ocr_routes

    monkeypatch.setattr(
        ocr_routes,
        "list_schools_with_location",
        _raiser(RuntimeError(f"(2003, \"Can't connect to MySQL server on '{DB_HOST}'\")")),
    )
    response = client.get("/api/schools")

    assert response.status_code == 500
    assert DB_HOST not in response.get_data(as_text=True)
    assert response.get_json()["error"]
    assert any(DB_HOST in line for line in debug_lines)


def test_confirm_error_hides_database_detail(
    client: FlaskClient, monkeypatch: pytest.MonkeyPatch, debug_lines: list[str]
) -> None:
    from server.blueprints import ocr_routes

    monkeypatch.setattr(
        ocr_routes,
        "confirm_student_id",
        _raiser(RuntimeError(f"(1045, \"Access denied for user 'ocr'@'10.0.0.5' on {DB_HOST}\")")),
    )
    response = client.post("/api/confirm", json={"student_id": "MOHSHA011021"})

    assert response.status_code == 500
    text = response.get_data(as_text=True)
    assert DB_HOST not in text
    assert "Access denied" not in text
    assert any(DB_HOST in line for line in debug_lines)


# ------------------------------------------------- #11: next= after sign-in
@pytest.mark.parametrize(
    "value",
    [
        "/\t/evil.com",
        "/\r/evil.com",
        "/\n/evil.com",
        "/\x00/evil.com",
        "/\x1f/evil.com",
        "/\x7f/evil.com",
        "\t//evil.com",
        "/\\evil.com",
        "\\\\evil.com",
        "//evil.com",
        "///evil.com",
        "https://evil.com",
        "javascript:alert(1)",
        "evil.com",
        "",
        None,
    ],
)
def test_safe_next_path_refuses_anything_off_site(value: str | None) -> None:
    from server.blueprints.auth_routes import _safe_next_path

    assert _safe_next_path(value) is None


@pytest.mark.parametrize("value", ["/", "/history", "/history?tab=not-found&page=2", "/a/b/./c"])
def test_safe_next_path_keeps_paths_in_the_app(value: str) -> None:
    from server.blueprints.auth_routes import _safe_next_path

    assert _safe_next_path(value) == value


@pytest.fixture
def google_stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Google sign-in with nothing leaving the machine."""
    from server.blueprints import auth_routes
    from server.services.oauth import oauth

    monkeypatch.setattr(
        oauth.google, "authorize_redirect", lambda *_a, **_k: redirect("https://accounts.google.com/x")
    )
    monkeypatch.setattr(
        oauth.google,
        "authorize_access_token",
        lambda *_a, **_k: {
            "userinfo": {"email": "staff@akanksha.org", "email_verified": True},
            "access_token": "ya29.TEST",
            "refresh_token": "1//TEST",
            "expires_in": 3600,
        },
    )
    monkeypatch.setattr(auth_routes, "is_authorized_staff", lambda _email: True)
    monkeypatch.setattr(auth_routes, "ALLOWED_EMAIL_DOMAIN", "akanksha.org")


def test_google_login_drops_a_next_with_a_tab(app: Flask, google_stubbed: None) -> None:
    client = app.test_client()
    response = client.get("/login/google?next=/%09/evil.com")

    assert response.status_code == 302
    with client.session_transaction() as sess:
        assert sess.get("post_login_next") is None


def test_google_callback_never_redirects_off_site(app: Flask, google_stubbed: None) -> None:
    client = app.test_client()
    # A next stored before this fix (or by any other route) is checked again.
    with client.session_transaction() as sess:
        sess["post_login_next"] = "/\t/evil.com"
    response = client.get("/login/google/callback?code=x&state=y")

    assert response.status_code == 302
    location = response.headers["Location"]
    assert not location.startswith("//")
    assert not urlsplit(location).netloc or urlsplit(location).netloc == "localhost"


@pytest.mark.parametrize(
    ("headers", "stored", "expected"),
    [
        # The login gate stores request.path - the path inside the app - so
        # under the /ocr mount the way back needs the mount in front of it,
        # or "/" lands on the host's own home page: another app.
        (PROXY_HEADERS, "/", "/ocr/"),
        (PROXY_HEADERS, "/history?tab=not-found", "/ocr/history?tab=not-found"),
        ({}, "/history", "/history"),
    ],
)
def test_google_callback_returns_to_the_page_inside_the_mount(
    app: Flask, google_stubbed: None, headers: dict[str, str], stored: str, expected: str
) -> None:
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["post_login_next"] = stored
    response = client.get("/login/google/callback?code=x&state=y", headers=headers)

    assert response.status_code == 302
    assert response.headers["Location"] == expected


def test_google_callback_without_next_goes_to_the_scanner(app: Flask, google_stubbed: None) -> None:
    response = app.test_client().get("/login/google/callback?code=x&state=y", headers=PROXY_HEADERS)

    assert response.status_code == 302
    assert urlsplit(response.headers["Location"]).path == "/ocr/"


# ------------------------------------------------- #23: SSO token stripping
@pytest.mark.parametrize("path_info", ["/\t/evil.com", "//evil.com", "/\\/evil.com", "///evil.com"])
@pytest.mark.parametrize("prefix", [None, "/ocr"])
def test_sso_redirect_stays_on_this_site(app: Flask, path_info: str, prefix: str | None) -> None:
    headers = {"X-Forwarded-Prefix": prefix} if prefix else {}
    response = app.test_client().get(
        "/", query_string="token=x", environ_overrides={"PATH_INFO": path_info}, headers=headers
    )

    assert response.status_code == 302
    location = response.headers["Location"]
    assert location.startswith("/") and not location.startswith("//")
    assert not urlsplit(location).netloc
    assert "\t" not in location and "\\" not in location
    if prefix:
        assert location.startswith(prefix + "/")


def test_sso_redirect_keeps_the_page_and_other_arguments(app: Flask) -> None:
    response = app.test_client().get(
        "/history?tab=not-found&token=x", headers={"X-Forwarded-Prefix": "/ocr"}
    )

    assert response.status_code == 302
    assert response.headers["Location"] == "/ocr/history?tab=not-found"


# ------------------------------------------- #24: Devanagari file names
@pytest.mark.parametrize(
    ("url", "filename", "stored"),
    [
        ("/api/extract", "फॉर्म.jpg", "upload.jpg"),
        ("/api/extract-batch", "प्रवेश_फॉर्म.pdf", "upload.pdf"),
        ("/api/extract-batch", "प्रवेश.PNG", "upload.png"),
        ("/api/extract", "स्कैन 1.jpg", "1.jpg"),
    ],
)
def test_upload_with_a_non_latin_name_keeps_its_extension(
    client: FlaskClient,
    stash: list[str],
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    filename: str,
    stored: str,
) -> None:
    from server.blueprints import ocr_routes
    from server.services.extract_student_id import ExtractResult

    seen: list[str] = []

    def fake_extract(_data: bytes, name: str) -> Any:
        seen.append(name)
        result = ExtractResult(student_id="", confidence=0.0, form_language="english", source=name)
        return [result] if url.endswith("batch") else result

    monkeypatch.setattr(ocr_routes, "extract_student_id_from_upload", fake_extract)
    monkeypatch.setattr(ocr_routes, "extract_student_ids_from_upload_batch", fake_extract)
    response = _upload(client, url, filename)

    assert response.status_code == 200, response.get_json()
    assert stash == [stored]
    assert seen == [stored]


def test_upload_that_is_not_an_image_is_still_refused(client: FlaskClient, stash: list[str]) -> None:
    response = _upload(client, "/api/extract", "फॉर्म.exe")

    assert response.status_code == 400
    assert stash == []


# ------------------------------------------- #25: Drive name with a period
@pytest.mark.parametrize(
    ("display_name", "expected"),
    [
        ("Ramesh S. Patil_RAMPAT120314", "Ramesh_S._Patil_RAMPAT120314.jpg"),
        ("Aatif Khan_AATKHA200618", "Aatif_Khan_AATKHA200618.jpg"),
        ("NOTFOUND_MOHSHA011021", "NOTFOUND_MOHSHA011021.jpg"),
        ("Already Named_X.jpg", "Already_Named_X.jpg"),
    ],
)
def test_drive_name_keeps_jpg(
    app: Flask, monkeypatch: pytest.MonkeyPatch, display_name: str, expected: str
) -> None:
    from server.services import drive_store

    renamed: list[str] = []

    class Reply:
        ok = True
        status_code = 200
        text = ""

        def __init__(self, body: dict[str, Any]) -> None:
            self._body = body

        def json(self) -> dict[str, Any]:
            return self._body

    def fake_post(*_a: Any, **_k: Any) -> Reply:
        return Reply({"id": "FILE1", "parents": ["root"]})

    def fake_patch(*_a: Any, json: dict[str, Any], **_k: Any) -> Reply:
        renamed.append(json["name"])
        return Reply({"id": "FILE1", "name": json["name"], "parents": ["DATE"]})

    monkeypatch.setattr(drive_store, "_get_valid_access_token", lambda: "token")
    monkeypatch.setattr(drive_store, "_ensure_folder", lambda *_a, **_k: "OCR")
    monkeypatch.setattr(drive_store, "_ensure_today_folder", lambda *_a, **_k: "DATE")
    monkeypatch.setattr(drive_store.requests, "post", fake_post)
    monkeypatch.setattr(drive_store.requests, "patch", fake_patch)

    with app.test_request_context("/"):
        meta = drive_store.upload_scan_to_drive(b"jpeg", f"{display_name}.jpg", display_name=display_name)

    assert meta is not None
    assert renamed == [expected]


# ------------------------------------------- #22 / C5: cross-site writes
@pytest.fixture
def confirm_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record /api/confirm lookups instead of running them."""
    from server.blueprints import ocr_routes
    from server.services.extract_student_id import ExtractResult

    calls: list[str] = []

    def fake_confirm(student_id: str, **_kwargs: Any) -> ExtractResult:
        calls.append(student_id)
        return ExtractResult(student_id=student_id, confidence=1.0, form_language="manual", source="t")

    monkeypatch.setattr(ocr_routes, "confirm_student_id", fake_confirm)
    monkeypatch.setattr(ocr_routes, "log_confirm_result", lambda *_a, **_k: None)
    return calls


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        # Another app on a sibling host: same-site to the browser, but not us.
        {**PROXY_HEADERS, "Origin": "https://other.akanksha.org", "Sec-Fetch-Site": "same-site"},
        # Same host over plain http is a different origin.
        {**PROXY_HEADERS, "Origin": "http://tether.akanksha.org"},
    ],
)
def test_cross_site_write_is_refused(
    client: FlaskClient, confirm_calls: list[str], headers: dict[str, str]
) -> None:
    response = client.post("/api/confirm", json={"student_id": "MOHSHA011021"}, headers=headers)

    assert response.status_code == 403
    assert response.is_json and response.get_json()["error"]
    assert confirm_calls == []


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "http://localhost", "Sec-Fetch-Site": "same-origin"},
        {**PROXY_HEADERS, "Origin": "https://tether.akanksha.org", "Sec-Fetch-Site": "same-origin"},
        {**PROXY_HEADERS, "Origin": "https://Tether.Akanksha.org:443"},
    ],
)
def test_same_origin_write_goes_through(
    client: FlaskClient, confirm_calls: list[str], headers: dict[str, str]
) -> None:
    response = client.post("/api/confirm", json={"student_id": "MOHSHA011021"}, headers=headers)

    assert response.status_code == 200, response.get_json()
    assert confirm_calls == ["MOHSHA011021"]


@pytest.mark.parametrize(
    "headers",
    [
        # The browser says same-origin, but the proxy forwarded neither Host
        # nor X-Forwarded-Proto, so the app sees itself as http://localhost.
        {"Origin": "https://tether.akanksha.org", "Sec-Fetch-Site": "same-origin"},
        # Proto forwarded, Host not: the app thinks it is the upstream address.
        {"X-Forwarded-Proto": "https", "Origin": "https://tether.akanksha.org", "Sec-Fetch-Site": "same-origin"},
        # A page served with Referrer-Policy: no-referrer sends Origin: null
        # even on its own same-origin POSTs.
        {**PROXY_HEADERS, "Origin": "null", "Sec-Fetch-Site": "same-origin"},
    ],
)
def test_write_the_browser_calls_same_origin_goes_through(
    client: FlaskClient, confirm_calls: list[str], headers: dict[str, str]
) -> None:
    response = client.post("/api/confirm", json={"student_id": "MOHSHA011021"}, headers=headers)

    assert response.status_code == 200, response.get_json()
    assert confirm_calls == ["MOHSHA011021"]


@pytest.fixture
def inbox_spy(monkeypatch: pytest.MonkeyPatch, fake_db: Any) -> dict[str, list[Any]]:
    """Drive and the batch table stubbed; records what the inbox routes change."""
    from server.blueprints import inbox_routes

    changed: dict[str, list[Any]] = {"opened": [], "closed": []}

    def open_batch(**kwargs: Any) -> int:
        changed["opened"].append(kwargs)
        return 1

    def close_batch(batch_id: int) -> None:
        changed["closed"].append(batch_id)

    monkeypatch.setattr(inbox_routes.drive_inbox, "find_inbox", lambda _e: {"id": "F", "shared_at": None})
    monkeypatch.setattr(inbox_routes.drive_inbox, "list_new_pages", lambda _f: [])
    monkeypatch.setattr(inbox_routes.batch_store, "open_batch", open_batch)
    monkeypatch.setattr(inbox_routes.batch_store, "close_batch", close_batch)
    monkeypatch.setattr(
        inbox_routes.batch_store, "open_batch_for", lambda _e: {"id": 41, "folder_id": "F"}
    )
    monkeypatch.setattr(inbox_routes.batch_store, "get_batch", lambda _i: {"id": _i})
    monkeypatch.setattr(inbox_routes.batch_store, "watcher_last_seen", lambda: None)
    monkeypatch.setattr(inbox_routes, "school_progress", lambda *_a: None)
    return changed


@pytest.mark.parametrize("url", ["/api/inbox/start", "/api/inbox/stop"])
@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        ('{"school": ""}', "text/plain"),
        ("force=1", "application/x-www-form-urlencoded"),
        ("", "text/plain"),
    ],
)
def test_inbox_write_needs_a_json_body(
    client: FlaskClient, inbox_spy: dict[str, list[Any]], url: str, body: str, content_type: str
) -> None:
    response = client.post(url, data=body, content_type=content_type)

    assert response.status_code == 415
    assert response.is_json and response.get_json()["error"]
    assert inbox_spy == {"opened": [], "closed": []}


def test_inbox_write_with_json_still_works(client: FlaskClient, inbox_spy: dict[str, list[Any]]) -> None:
    started = client.post("/api/inbox/start", json={"school": "School A", "location": "Pune"})
    stopped = client.post("/api/inbox/stop", json={})

    assert started.status_code == 200, started.get_json()
    assert stopped.status_code == 200, stopped.get_json()
    assert len(inbox_spy["opened"]) == 1
    assert inbox_spy["closed"] == [41]


def test_upload_is_exempt_from_the_json_rule(
    client: FlaskClient, stash: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from server.blueprints import ocr_routes
    from server.services.extract_student_id import ExtractResult

    monkeypatch.setattr(
        ocr_routes,
        "extract_student_id_from_upload",
        lambda _d, name: ExtractResult(student_id="", confidence=0.0, form_language="english", source=name),
    )
    response = _upload(client, "/api/extract", "page-001.jpg", headers={"Origin": "http://localhost"})

    assert response.status_code == 200


def test_bodyless_delete_from_history_is_not_refused(client: FlaskClient, fake_db: Any) -> None:
    response = client.delete("/api/not-found/5", headers={"Origin": "http://localhost"})

    assert response.status_code not in (403, 415)


# ------------------------------------------- #27 / C5: the session cookie
def _session_cookie(app: Flask, **headers: str) -> str:
    @app.get("/_probe-session")
    def probe_session() -> str:
        session["probe"] = 1
        return "ok"

    response = app.test_client().get("/_probe-session", headers=headers)
    cookies = response.headers.getlist("Set-Cookie")
    assert len(cookies) == 1, cookies
    return cookies[0]


def test_session_cookie_is_scoped_to_the_ocr_mount(app: Flask) -> None:
    cookie = _session_cookie(app, **PROXY_HEADERS)
    attributes = {part.strip().split("=", 1)[0].lower(): part.strip() for part in cookie.split(";")}

    assert cookie.startswith("ocr_session=")
    assert attributes["path"] == "Path=/ocr"
    assert "secure" in attributes
    assert "httponly" in attributes
    assert attributes["samesite"].lower() == "samesite=lax"


def test_session_cookie_on_a_local_http_run(app: Flask) -> None:
    cookie = _session_cookie(app)
    attributes = {part.strip().split("=", 1)[0].lower(): part.strip() for part in cookie.split(";")}

    assert cookie.startswith("ocr_session=")
    assert attributes["path"] == "Path=/"
    # Not Secure over http, or a local run could never keep anyone signed in.
    assert "secure" not in attributes
    assert attributes["samesite"].lower() == "samesite=lax"


def test_session_cookie_secure_can_be_forced(app: Flask) -> None:
    app.config["SESSION_COOKIE_SECURE"] = True
    cookie = _session_cookie(app)

    assert "secure" in cookie.lower()


# ------------------------------------------------- the page itself (Node)
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node is needed to run the page script")

# A stand-in DOM with just enough surface for the Scanner page script, and a
# fetch whose answers the scenario hands out. Each scenario is the body of an
# async function; what it returns is printed as JSON for the test to check.
# For the scan folder there is also a clock the scenario moves by hand, an
# IndexedDB kept in memory (handed to a second page, it is a reload of the
# same browser) and a folder the scenario adds scans to, as the scanner's
# own software would - one the page may also sort into batch folders.
_PAGE_HARNESS_JS = r"""
const fs = require("fs");
const vm = require("vm");

class ClassList {
  constructor(el) { this.el = el; }
  _set() { return new Set((this.el.className || "").split(/\s+/).filter(Boolean)); }
  add(...c) { const s = this._set(); c.forEach((x) => s.add(x)); this.el.className = [...s].join(" "); }
  remove(...c) { const s = this._set(); c.forEach((x) => s.delete(x)); this.el.className = [...s].join(" "); }
  toggle(c, on) { const s = this._set(); const want = on === undefined ? !s.has(c) : on; want ? s.add(c) : s.delete(c); this.el.className = [...s].join(" "); return want; }
  contains(c) { return this._set().has(c); }
}
const allElements = [];
class El {
  constructor(tag, id) {
    this.tagName = (tag || "div").toUpperCase(); this.id = id || "";
    this.hidden = false; this.textContent = ""; this.className = ""; this.value = "";
    this.disabled = false; this.checked = false; this.type = ""; this.name = "";
    this.children = []; this.style = {}; this.dataset = {}; this.attrs = {};
    this.listeners = {}; this.parent = null; this._html = ""; this._lazy = {};
    this.classList = new ClassList(this); this.options = [];
    allElements.push(this);
  }
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = v; this.children = []; this._lazy = {}; }
  // As in a browser, a node put somewhere new leaves where it was.
  appendChild(c) { if (c && typeof c === "object") { this._detach(c); c.parent = this; this.children.push(c); } return c; }
  insertBefore(c, ref) {
    if (!ref) return this.appendChild(c);
    this._detach(c);
    c.parent = this;
    const at = this.children.indexOf(ref);
    this.children.splice(at === -1 ? this.children.length : at, 0, c);
    return c;
  }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); c.parent = null; return c; }
  _detach(c) { if (c.parent) c.parent.children = c.parent.children.filter((x) => x !== c); }
  append(...cs) { cs.forEach((c) => this.appendChild(typeof c === "string" ? Object.assign(new El("#text"), { textContent: c }) : c)); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this); }
  setAttribute(k, v) { this.attrs[k] = v; }
  getAttribute(k) { return this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(t, fn) { (this.listeners[t] ||= []).push(fn); }
  removeEventListener() {}
  dispatchEvent(ev) { ev.target ||= this; (this.listeners[ev.type] || []).forEach((fn) => fn(ev)); return true; }
  fire(type, extra) { const ev = Object.assign({ type, target: this, preventDefault() {} }, extra || {}); (this.listeners[type] || []).forEach((fn) => fn(ev)); return ev; }
  click() { this.fire("click"); }
  contains(other) { return other === this || this._all().includes(other); }
  focus() {} select() {} closest() { return null; } scrollIntoView() {}
  requestSubmit() { this.fire("submit"); }
  _all() {
    const out = [];
    const walk = (e) => [...e.children, ...Object.values(e._lazy)].forEach((c) => { out.push(c); walk(c); });
    walk(this);
    return out;
  }
  querySelectorAll(sel) {
    if (this.qsaHook) { const r = this.qsaHook(sel); if (r) return r; }
    if (sel.startsWith(".")) return this._all().filter((e) => e.classList.contains(sel.slice(1)));
    return [];
  }
  querySelector(sel) {
    if (this.qsHook) { const r = this.qsHook(sel); if (r !== undefined) return r; }
    const found = this.querySelectorAll(sel)[0];
    if (found) return found;
    // Children written as an innerHTML template are made on first use.
    if (sel.startsWith(".") && this._html.includes(sel.slice(1))) {
      if (!this._lazy[sel]) { const e = new El("span"); e.className = sel.slice(1); e.parent = this; this._lazy[sel] = e; }
      return this._lazy[sel];
    }
    return null;
  }
}

function makeStorage(seed) {
  const m = new Map(Object.entries(seed || {}));
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k) };
}

const tick = (ms = 0) => new Promise((r) => setTimeout(r, ms));
const settle = async () => { await tick(); await tick(); };

// A clock the scenario moves by hand: the page's timers and Date follow it,
// so two seconds between looks at the scan folder take no real time.
function makeClock() {
  const RealDate = Date;
  const clock = { now: RealDate.UTC(2026, 9, 27, 5, 0, 0), timers: new Map(), nextId: 1 };
  const add = (fn, ms, every) => {
    const id = clock.nextId++;
    clock.timers.set(id, { at: clock.now + Math.max(0, Number(ms) || 0), fn, every });
    return id;
  };
  clock.setTimeout = (fn, ms) => add(fn, ms, 0);
  clock.setInterval = (fn, ms) => add(fn, ms, Math.max(1, Number(ms) || 0));
  clock.clearTimeout = (id) => { clock.timers.delete(id); };
  clock.clearInterval = clock.clearTimeout;
  clock.advance = async (ms) => {
    const end = clock.now + ms;
    for (;;) {
      await settle();
      let next = null;
      for (const [id, t] of clock.timers) if (t.at <= end && (!next || t.at < next.t.at)) next = { id, t };
      if (!next) break;
      clock.now = Math.max(clock.now, next.t.at);
      if (next.t.every) next.t.at += next.t.every; else clock.timers.delete(next.id);
      next.t.fn();
    }
    clock.now = end;
    await settle();
  };
  clock.Date = class extends RealDate {
    constructor(...args) { if (args.length) super(...args); else super(clock.now); }
    static now() { return clock.now; }
  };
  return clock;
}

// IndexedDB as far as the page uses it.
function makeIndexedDB() {
  const dbs = new Map();
  const soon = (fn) => Promise.resolve().then(fn);
  return {
    dbs,
    open(name) {
      const request = {};
      soon(() => {
        let stores = dbs.get(name);
        const fresh = !stores;
        if (fresh) { stores = new Map(); dbs.set(name, stores); }
        request.result = {
          objectStoreNames: { contains: (store) => stores.has(store) },
          createObjectStore: (store) => { stores.set(store, new Map()); },
          transaction(storeName) {
            const store = stores.get(storeName);
            const work = [];
            const tx = {
              objectStore() {
                const op = (run) => { const r = {}; work.push(() => { r.result = run(); }); return r; };
                return {
                  put: (value, key) => op(() => { store.set(key, value); return key; }),
                  get: (key) => op(() => store.get(key)),
                  getAll: () => op(() => [...store.values()]),
                };
              },
            };
            soon(() => { work.forEach((run) => run()); soon(() => tx.oncomplete && tx.oncomplete()); });
            return tx;
          },
        };
        if (fresh && request.onupgradeneeded) request.onupgradeneeded();
        if (request.onsuccess) request.onsuccess();
      });
      return request;
    },
  };
}

// A folder handle the scenario adds scans to, grows and empties, as the
// scanner's software would. listings counts the page's looks into it. It
// holds folders too, which the page may make and move scans into: with
// options.move its file handles have move(), as Chrome's do; without, the
// page has to copy and remove. options.write is the answer to a request to
// edit it ("denied" for read-only), the read permission otherwise.
function notFound(what) { return Object.assign(new Error(what + " is not there"), { name: "NotFoundError" }); }
function makeFolder(name = "OCR Scans", options = {}) {
  const files = new Map();
  const dirs = new Map();
  let stamp = Date.UTC(2026, 9, 27, 4, 0, 0);
  const fileHandle = (fileName) => {
    const handle = {
      kind: "file", name: fileName,
      async getFile() { const f = files.get(fileName); if (!f) throw notFound(fileName); return { name: fileName, size: f.size, lastModified: f.lastModified }; },
      async createWritable() {
        return {
          async write(data) { folder.writes += 1; const f = files.get(fileName); f.size = data.size; f.lastModified = data.lastModified; },
          async close() {},
        };
      },
    };
    if (options.move) {
      handle.move = async (dir, newName) => {
        const f = files.get(fileName);
        if (!f) throw notFound(fileName);
        files.delete(fileName);
        dir.put(newName || fileName, f);
        folder.moves.push([fileName, dir.name, newName || fileName]);
      };
    }
    return handle;
  };
  const folder = {
    kind: "directory", name, permission: options.permission || "granted", write: options.write || null,
    listings: 0, moves: [], writes: 0, removed: [], made: [],
    add(file, size = 1000) { stamp += 1000; files.set(file, { size, lastModified: stamp }); },
    grow(file, by = 500) { const f = files.get(file); stamp += 1000; f.size += by; f.lastModified = stamp; },
    remove(file) { files.delete(file); },
    put(file, f) { files.set(file, { size: f.size, lastModified: f.lastModified }); },
    names() { return [...files.keys()].sort(); },
    size(file) { const f = files.get(file); return f ? f.size : null; },
    folders() { return [...dirs.keys()].sort(); },
    sub(dirName) { if (!dirs.has(dirName)) dirs.set(dirName, makeFolder(dirName, options)); return dirs.get(dirName); },
    async *values() {
      folder.listings += 1;
      for (const fileName of [...files.keys()]) yield fileHandle(fileName);
      for (const dir of [...dirs.values()]) yield dir;
    },
    async getFileHandle(fileName, opts = {}) {
      if (!files.has(fileName)) { if (!opts.create) throw notFound(fileName); files.set(fileName, { size: 0, lastModified: 0 }); }
      return fileHandle(fileName);
    },
    async getDirectoryHandle(dirName, opts = {}) {
      if (!dirs.has(dirName)) { if (!opts.create) throw notFound(dirName); folder.made.push(dirName); dirs.set(dirName, makeFolder(dirName, options)); }
      return dirs.get(dirName);
    },
    async removeEntry(fileName) { if (!files.has(fileName)) throw notFound(fileName); files.delete(fileName); folder.removed.push(fileName); },
    async queryPermission(opts = {}) { return opts.mode === "readwrite" ? (folder.write || folder.permission) : folder.permission; },
    async requestPermission(opts = {}) {
      if (opts.mode === "readwrite" && folder.write === "denied") return "denied";
      folder.permission = "granted";
      return "granted";
    },
  };
  return folder;
}

function loadPage(opts = {}) {
  const byId = {};
  const docListeners = {};
  const document = {
    body: new El("body"),
    getElementById(id) { return (byId[id] ||= new El("div", id)); },
    createElement(tag) { return new El(tag); },
    createTextNode(t) { return Object.assign(new El("#text"), { textContent: t }); },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    addEventListener(type, fn) { (docListeners[type] ||= []).push(fn); },
    fire(type) { (docListeners[type] || []).forEach((fn) => fn({ type })); },
    visibilityState: "visible",
  };
  const radios = ["Pune", "Mumbai"].map((v, i) => Object.assign(new El("input"), { type: "radio", name: "location", value: v, checked: i === 0 }));
  const locationRadios = document.getElementById("location-radios");
  radios.forEach((r) => locationRadios.appendChild(r));
  locationRadios.qsHook = (sel) => {
    if (sel.includes(":checked")) return radios.find((r) => r.checked) || null;
    const m = sel.match(/value="([^"]*)"/);
    if (m) return radios.find((r) => r.value === m[1]) || null;
    if (sel.startsWith("input[name=location]")) return radios[0];
    return undefined;
  };
  const schoolSelect = document.getElementById("school-select");
  schoolSelect.tagName = "SELECT";
  const appendOption = schoolSelect.appendChild.bind(schoolSelect);
  schoolSelect.appendChild = (o) => { appendOption(o); if (o.selected) schoolSelect.value = o.value; return o; };
  Object.defineProperty(schoolSelect, "innerHTML", { get() { return ""; }, set() { schoolSelect.children = []; schoolSelect.value = ""; } });
  document.getElementById("page-loader").hidden = true;

  // calls: the requests still waiting for an answer; log: every request made.
  const calls = [];
  const log = [];
  const fetch = (url, init) => new Promise((resolve, reject) => {
    const call = {
      url: String(url), opts: init || {}, reject,
      reply(data, status = 200) { resolve({ ok: status >= 200 && status < 300, status, json: async () => data }); },
    };
    calls.push(call);
    log.push(call);
  });
  const clock = opts.clock;
  const ctx = {
    document, fetch, console,
    setTimeout: clock ? clock.setTimeout : setTimeout,
    clearTimeout: clock ? clock.clearTimeout : clearTimeout,
    setInterval: clock ? clock.setInterval : setInterval,
    clearInterval: clock ? clock.clearInterval : clearInterval,
    Date: clock ? clock.Date : Date,
    URLSearchParams, URL, Math, JSON, Promise, String, Number, parseInt, Array, Set, Object, Error,
    TypeError, Boolean, RegExp, Uint8Array, Map, AbortController, AbortSignal,
    // The browser's own random numbers, for the page's id (SENDER).
    crypto: globalThis.crypto,
    localStorage: makeStorage(opts.local), sessionStorage: makeStorage(opts.session),
    navigator: opts.navigator || { userAgent: "Mozilla/5.0 (Macintosh)", platform: "MacIntel", maxTouchPoints: 0 },
    FormData: class { constructor() { this.parts = []; } append(k, v, n) { this.parts.push([k, v, n]); } },
    Blob: class { constructor(p, o) { this.p = p; this.o = o; } },
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    Event: class { constructor(type, o) { this.type = type; Object.assign(this, o || {}); } },
    confirm: opts.confirm || (() => true),
    location: { href: "http://localhost/", reload() {} },
  };
  // Chrome and Edge only: Firefox and Safari have no folder picker.
  if (opts.showDirectoryPicker) ctx.showDirectoryPicker = opts.showDirectoryPicker;
  if (opts.indexedDB) ctx.indexedDB = opts.indexedDB;
  ctx.window = Object.assign(ctx, { addEventListener() {}, removeEventListener() {} });
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
  return { byId, calls, log, schoolSelect, radios, document };
}

// Answer the first queued request that matches, waiting for it if need be.
async function answer(calls, match, data, status = 200) {
  for (let i = 0; i < 600; i += 1) {
    const at = calls.findIndex((c) => c.url.includes(match));
    if (at !== -1) { const [c] = calls.splice(at, 1); c.reply(data, status); await tick(); return c; }
    await tick(5);
  }
  throw new Error("no request for " + match + "; waiting: " + calls.map((c) => c.url).join(", "));
}
// The first queued request that matches gets no answer at all: Wi-Fi drops.
async function fail(calls, match) {
  for (let i = 0; i < 600; i += 1) {
    const at = calls.findIndex((c) => c.url.includes(match));
    if (at !== -1) { const [c] = calls.splice(at, 1); c.reject(new TypeError("Failed to fetch")); await tick(); return c; }
    await tick(5);
  }
  throw new Error("no request for " + match);
}
// Moves the clock on a second at a time for `seconds`, answering each send
// to /api/scans as it comes: reply(call, n) for the nth, or taken with a
// page of its own. Returns every file sent, in the order it was sent.
async function serveScans(page, clock, seconds, reply) {
  const names = [];
  for (let s = 0; s < seconds; s += 1) {
    await clock.advance(1000);
    for (const call of scanCalls(page)) {
      page.calls.splice(page.calls.indexOf(call), 1);
      names.push(sentName(call));
      const answer = reply ? reply(call, names.length) : null;
      if (answer) call.reply(answer[0], answer[1]);
      else call.reply({ pages: [{ page_id: "p" + names.length, source_page: 1 }] }, 202);
      await tick();
    }
  }
  return names;
}
async function answerSchools(page) {
  await answer(page.calls, "/api/schools", { schools: [
    { school: "School A", location: "Pune" }, { school: "School B", location: "Pune" },
  ] });
}
// The page with School A of Pune remembered, in a browser that cannot
// watch a folder (Firefox or Safari).
async function openPage(opts = {}) {
  const page = loadPage({ clock: opts.clock, local: { "ocr.school": "School A", "ocr.location": "Pune" } });
  await answerSchools(page);
  await answer(page.calls, "/api/inbox/status", { configured: false });
  return page;
}
// The same page in Chrome or Edge, which can: Connect scan folder hands it
// `folder`.
async function openScannerPage({ clock, idb, folder }) {
  const page = loadPage({
    clock, indexedDB: idb,
    local: { "ocr.school": "School A", "ocr.location": "Pune" },
    showDirectoryPicker: async (options) => { page.pickerOptions = options; return folder; },
  });
  await answerSchools(page);
  await answer(page.calls, "/api/inbox/status", { configured: false });
  await settle();
  return page;
}
async function connectFolder(page) {
  page.byId["scan-folder-connect"].fire("click");
  await settle();
}
// The sends to /api/scans still waiting for an answer, and every one made.
function scanCalls(page) {
  return page.calls.filter((c) => c.url.includes("/api/scans"));
}
function everyScan(page) {
  return page.log.filter((c) => c.url.includes("/api/scans"));
}
function sentName(call) {
  return call.opts.body.parts[0][1].name;
}
// What a send to /api/scans carried: the file's name and the school.
function sentForm(call) {
  const out = {};
  call.opts.body.parts.forEach(([k, v]) => { out[k] = k === "file" ? v.name : v; });
  return out;
}
async function answerScan(page, pageIds) {
  return answer(page.calls, "/api/scans", { pages: pageIds.map((id, i) => ({ page_id: id, source_page: i + 1 })) }, 202);
}

// /api/batches as the server gives it: pages carry their meta and result,
// documents their parts in scan order, and `pages` lists every page the
// person has, unread ones included.
const EMPTY = { batches: [], unsorted: [], queue: { queued: 0, reading: 0, failed: 0 } };
const ASHA = { student_id: "MOHSHA011021", student_name: "Asha Patil", school_name: "School A", grade_name: "5", division_name: "B" };
function pg(id, extra) {
  return Object.assign({
    page_id: id, status: "read", source_name: id + ".jpg", source_page: 1, created_at: "2026-10-27 10:30:00",
    code: null, student_id: "", matched_student: null, message: "",
  }, extra || {});
}
function doc(code, pages, extra) {
  return Object.assign({ key: "k-" + code, code, parts: pages.length, pages, student_id: "", matched_student: null, status: "ready" }, extra || {});
}
function batches(groups, unsorted = [], queue = { queued: 0, reading: 0, failed: 0 }, unread = []) {
  const pages = [...unread, ...unsorted];
  groups.forEach(([, documents]) => documents.forEach((d) => pages.push(...d.pages)));
  return { batches: groups.map(([batch, documents]) => ({ batch, label: "B" + batch, documents, counts: {} })), unsorted, pages, queue };
}
async function answerBatches(page, data) {
  return answer(page.calls, "/api/batches", data);
}

// Reading the OCR results back out of the stand-in DOM.
function within(el) { return [el, ...el._all()]; }
function textIn(el) { return within(el).map((e) => e.textContent).join(" "); }
function cards(page) {
  return [...page.byId["ocr-batches"]._all(), ...page.byId["ocr-unsorted-list"]._all()]
    .filter((e) => e.tagName === "LI" && e.classList.contains("ocr-doc"));
}
function card(page, code) {
  const found = cards(page).find((c) => c._all().some((e) => e.classList.contains("ocr-doc-code") && e.textContent === code));
  if (!found) throw new Error("no card for " + code + "; cards: " + cards(page).map(textIn).join(" | "));
  return found;
}
function unsortedCards(page) { return cards(page).filter((c) => c.classList.contains("ocr-page")); }
function fieldIn(el, name) { return el._all().find((e) => e.dataset && e.dataset.field === name); }
function buttonIn(el, text) { return el._all().find((e) => e.tagName === "BUTTON" && e.textContent === text); }
function imagesIn(el) { return el._all().filter((e) => e.tagName === "IMG").map((e) => e.src); }
function linksIn(el) { return el._all().filter((e) => e.tagName === "A").map((e) => e.href); }
function bigScans(el) { return el._all().filter((e) => e.classList.contains("ocr-scan")); }
async function uploadFile(page, ...names) {
  page.byId["file-input"].files = names.map((name) => ({ name }));
  page.byId["file-input"].fire("change");
}
function pageText() {
  return allElements.map((e) => `${e.textContent} ${e._html} ${e.value}`).join("\n");
}

async function run(scenario) {
  try {
    const out = await scenario();
    console.log("RESULT:" + JSON.stringify(out));
    process.exit(0);
  } catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    process.exit(1);
  }
}
"""


def _run_page(client: FlaskClient, tmp_path: Path, scenario: str) -> Any:
    """Run `scenario` (an async JS function body) against the rendered page."""
    html = client.get("/").get_data(as_text=True)
    scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    page_script = next(s for s in scripts if "const API_BASE" in s)
    (tmp_path / "page.js").write_text(page_script, encoding="utf-8")
    runner = tmp_path / "run.js"
    runner.write_text(f"{_PAGE_HARNESS_JS}\nrun(async () => {{\n{scenario}\n}});\n", encoding="utf-8")
    proc = subprocess.run(
        [str(NODE), str(runner), str(tmp_path / "page.js")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:"))
    return json.loads(line[len("RESULT:"):])


@needs_node
def test_page_unreadable_scan_folder_page_shows_its_scan_and_confirms_the_typed_id(
    client: FlaskClient, tmp_path: Path
) -> None:
    """#7: a page from the scan folder with no readable ID shows its scan; the ID typed from it is confirmed for its document."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      await clock.advance(4000);
      await answerScan(page, ["p1"]);
      await clock.advance(3000);
      await answerBatches(page, batches([[1, [doc("B1-1", [
        pg("p1", { code: "B1-1", source_name: "Scan 001.jpg", message: "Could not read a valid Student ID." }),
      ], { status: "needs_attention" })]]]));
      const c1 = card(page, "B1-1");
      const shown = { images: imagesIn(c1), big: bigScans(c1).length, text: textIn(c1) };
      fieldIn(c1, "id").value = "mohsha011021";
      buttonIn(c1, "Confirm and upload").fire("click");
      const c = await answer(page.calls, "/confirm", { document: { status: "uploaded", drive_link: "https://drive.google.com/file/d/abc/view" } });
      return { shown, url: c.url, method: c.opts.method, type: c.opts.headers["Content-Type"], body: JSON.parse(c.opts.body) };
    """)

    assert "/api/pages/p1/image?size=full" in out["shown"]["images"]
    assert "/api/pages/p1/image?size=thumb" in out["shown"]["images"]
    assert out["shown"]["big"] == 1
    assert "Could not read a valid Student ID." in out["shown"]["text"]
    assert out["url"] == "/api/documents/k-B1-1/confirm"
    assert (out["method"], out["type"]) == ("POST", "application/json")
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1"]}


@needs_node
def test_page_unreadable_file_shows_its_scan_and_takes_a_typed_code(client: FlaskClient, tmp_path: Path) -> None:
    """#7: a file whose page has no code and no ID read waits under Unsorted with its scan; the typed code makes it a document."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      await answerBatches(page, EMPTY);
      await uploadFile(page, "stack.pdf");
      await answerScan(page, ["p9"]);
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, batches([], [pg("p9", { source_name: "stack.pdf", message: "Could not read a valid Student ID." })]));
      const loose = unsortedCards(page);
      const first = {
        count: loose.length, hidden: page.byId["ocr-unsorted"].hidden,
        big: bigScans(loose[0]).map((f) => imagesIn(f)[0]),
        fields: ["code", "id"].map((name) => Boolean(fieldIn(loose[0], name))),
      };
      fieldIn(loose[0], "code").value = "b1-3";
      buttonIn(loose[0], "Set code").fire("click");
      const set = await answer(page.calls, "/api/pages/p9/code", { ok: true });
      await answerBatches(page, batches([[1, [doc("B1-3", [pg("p9", { code: "B1-3", source_name: "stack.pdf" })], { status: "needs_attention" })]]]));
      const c3 = card(page, "B1-3");
      fieldIn(c3, "id").value = "MOHSHA011021";
      buttonIn(c3, "Confirm and upload").fire("click");
      const c = await answer(page.calls, "/api/documents/k-B1-3/confirm", { status: "uploaded", drive_link: "https://drive.google.com/x" });
      return { first, set: [set.opts.headers["Content-Type"], JSON.parse(set.opts.body)], unsortedAfter: page.byId["ocr-unsorted"].hidden, body: JSON.parse(c.opts.body) };
    """)

    assert out["first"] == {
        "count": 1, "hidden": False, "big": ["/api/pages/p9/image?size=full"], "fields": [True, True],
    }
    assert out["set"] == ["application/json", {"code": "B1-3"}]
    assert out["unsortedAfter"] is True
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p9"]}


@needs_node
def test_page_typed_id_is_looked_up_before_anything_is_saved(client: FlaskClient, tmp_path: Path) -> None:
    """#7 and #17: Check ID only looks up; the confirm carries the ID alone, so a picker change cannot move the page."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      await answerBatches(page, EMPTY);
      await uploadFile(page, "form.jpg");
      const sent = await answerScan(page, ["p1"]);
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, batches([[1, [doc("B1-1", [
        pg("p1", { code: "B1-1" }), pg("p2", { code: "B1-1", source_name: "form-2.jpg" }),
      ], { status: "needs_attention" })]]]));
      let c1 = card(page, "B1-1");
      // The operator picks the next school before typing - the pages keep
      // the school they were sent for.
      page.schoolSelect.value = "School B";
      page.schoolSelect.fire("change");
      fieldIn(c1, "id").value = "mohsha011021";
      buttonIn(c1, "Check ID").fire("click");
      const looked = [];
      for (let i = 0; i < 2; i += 1) {
        const call = await answer(page.calls, "/student-id", { ok: true });
        looked.push([call.url, JSON.parse(call.opts.body)]);
      }
      const writesBeforeConfirm = page.log.filter((c) => c.url.endsWith("/confirm")).length;
      await answerBatches(page, batches([[1, [doc("B1-1", [
        pg("p1", { code: "B1-1", student_id: "MOHSHA011021", matched_student: ASHA }),
        pg("p2", { code: "B1-1", student_id: "MOHSHA011021" }),
      ], { status: "ready", student_id: "MOHSHA011021", matched_student: ASHA })]]]));
      c1 = card(page, "B1-1");
      const who = textIn(c1);
      const prefilled = fieldIn(c1, "id").value;
      buttonIn(c1, "Confirm and upload").fire("click");
      const c = await answer(page.calls, "/api/documents/k-B1-1/confirm", { status: "uploaded", drive_link: "https://drive.google.com/x" });
      return { form: sentForm(sent), looked, writesBeforeConfirm, who, prefilled, body: JSON.parse(c.opts.body) };
    """)

    assert out["form"] == {"file": "form.jpg", "school": "School A", "location": "Pune", "auto_upload": "1"}
    assert out["looked"] == [
        ["/api/pages/p1/student-id", {"student_id": "MOHSHA011021"}],
        ["/api/pages/p2/student-id", {"student_id": "MOHSHA011021"}],
    ]
    assert out["writesBeforeConfirm"] == 0
    assert "Asha Patil" in out["who"]
    assert out["prefilled"] == "MOHSHA011021"
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1", "p2"]}


@needs_node
def test_page_blank_page_shows_its_scan_but_offers_no_box(client: FlaskClient, tmp_path: Path) -> None:
    """#7 follow-up: an ID typed against a blank page would be saved with an empty image."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      await answerBatches(page, batches([], [
        pg("p1", { source_name: "stack.pdf", source_page: 2, blank: true, message: "This page looks blank. If it should have a form on it, check the sheet was fed the right way up." }),
        pg("p3", { source_name: "stack.pdf", source_page: 4, blank: true, message: "Nothing to read on this page." }),
        pg("p2", { source_name: "stack.pdf", source_page: 3, message: "No Student ID found on this page." }),
      ]));
      return unsortedCards(page).map((c) => ({
        offered: Boolean(fieldIn(c, "id") || fieldIn(c, "code")),
        big: bigScans(c).map((f) => imagesIn(f)[0]),
        text: textIn(c),
      }));
    """)

    blank, flagged, unread = out
    assert blank["offered"] is False
    assert blank["big"] == ["/api/pages/p1/image?size=full"]
    assert "looks blank" in blank["text"]
    # The server's blank flag is enough, whatever the message says.
    assert flagged["offered"] is False and "Looks blank" in flagged["text"]
    # A page that has a form on it, just not a readable ID, still gets one.
    assert unread["offered"] is True
    assert unread["big"] == ["/api/pages/p2/image?size=full"]


@needs_node
def test_page_confirmed_document_updates_the_results_summary(client: FlaskClient, tmp_path: Path) -> None:
    """#7 follow-up: the summary counts the document as uploaded, and an older answer does not take that back."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const ready = batches([[1, [doc("B1-1", [pg("p1", { code: "B1-1", student_id: "MOHSHA011021", matched_student: ASHA })],
        { student_id: "MOHSHA011021", matched_student: ASHA })]]]);
      await answerBatches(page, ready);
      const before = page.byId["ocr-status"].textContent;
      buttonIn(card(page, "B1-1"), "Confirm and upload").fire("click");
      await answer(page.calls, "/confirm", { document: { status: "uploaded", drive_link: "https://drive.google.com/file/d/abc/view" } });
      await tick(5);
      const after = page.byId["ocr-status"].textContent;
      // The look the confirm asks for, answered as the server stood before it caught up.
      await answerBatches(page, ready);
      return { before, after, later: page.byId["ocr-status"].textContent, live: page.byId["ocr-live"].textContent };
    """)

    assert out["before"] == "1 document ready to confirm."
    assert out["after"] == "1 uploaded."
    assert out["later"] == "1 uploaded."
    assert "B1-1 saved and uploaded" in out["live"]


@needs_node
def test_page_results_leave_the_manual_entry_line_alone(client: FlaskClient, tmp_path: Path) -> None:
    """Once the line under Enter Student ID directly says something, the OCR results keep off it."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      await answerBatches(page, EMPTY);
      page.byId["manual-input"].value = "AADMIS310714";
      page.byId["manual-form"].fire("submit");
      await answer(page.calls, "/api/confirm", { found_in_db: true, student_id: "AADMIS310714", student: {} });
      await tick(5);
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, batches([[1, [doc("B1-1", [pg("p1", { code: "B1-1" })], { status: "needs_attention" })]]]));
      return { line: page.byId["status-text"].textContent, ocr: page.byId["ocr-status"].textContent };
    """)

    assert out["line"].startswith("Saved") and "AADMIS310714" in out["line"]
    assert out["ocr"] == "1 document to check."


@needs_node
def test_page_each_added_file_keeps_the_school_it_was_added_for(client: FlaskClient, tmp_path: Path) -> None:
    """#17: changing the picker while a stack is still going out does not move the rest of that stack."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      await uploadFile(page, "a1.jpg", "a2.jpg");
      await tick(5);
      page.schoolSelect.value = "School B";
      page.schoolSelect.fire("change");
      await uploadFile(page, "b1.jpg");
      const forms = [];
      for (let i = 0; i < 3; i += 1) forms.push(sentForm(await answerScan(page, ["p" + i])));
      return { forms, saves: page.log.filter((c) => /confirm|extract/.test(c.url)).length };
    """)

    assert out["forms"] == [
        {"file": "a1.jpg", "school": "School A", "location": "Pune", "auto_upload": "1"},
        {"file": "a2.jpg", "school": "School A", "location": "Pune", "auto_upload": "1"},
        {"file": "b1.jpg", "school": "School B", "location": "Pune", "auto_upload": "1"},
    ]
    assert out["saves"] == 0


@needs_node
def test_page_scan_folder_keeps_each_scan_to_the_school_it_was_seen_under(
    client: FlaskClient, tmp_path: Path
) -> None:
    """#17 for the scan folder: a new school picked mid-stack does not take the old stack's pages.

    A scan seen before the change is sent for the school watching started
    for; one first seen after it is sent for the new school - straight
    after, as nothing waits for OCR any more.
    """
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      await clock.advance(2000);
      page.schoolSelect.value = "School B";
      page.schoolSelect.fire("change");
      folder.add("Scan 002.jpg");
      await clock.advance(2000);
      await clock.advance(2000);
      const sends = [];
      sends.push(sentForm(await answerScan(page, ["p1"])));
      sends.push(sentForm(await answerScan(page, ["p2"])));
      return { sends, held: page.byId["drop-message"].textContent, state: page.byId["scan-folder-state"].textContent };
    """)

    assert out["sends"] == [
        {"file": "Scan 001.jpg", "school": "School A", "location": "Pune", "auto_upload": "1"},
        {"file": "Scan 002.jpg", "school": "School B", "location": "Pune", "auto_upload": "1"},
    ]
    assert out["held"] == ""
    assert "for School B" in out["state"]


@needs_node
def test_page_never_shows_database_internals(client: FlaskClient, tmp_path: Path) -> None:
    """#28: whatever a response carries, the page prints no DB host, name or SQL."""
    out = _run_page(client, tmp_path, rf"""
      const debug = {{ db_host: "{DB_HOST}", db_name: "afdw_prod", sql: "SELECT * FROM active_student_data", error: "boom" }};
      const page = await openPage();
      page.byId["manual-input"].value = "MOHSHA011021";
      page.byId["manual-form"].fire("submit");
      await answer(page.calls, "/api/confirm", {{ found_in_db: false, student_id: "MOHSHA011021", db_debug: debug }});
      await answerBatches(page, Object.assign(batches([[1, [doc("B1-1", [pg("p1", {{ code: "B1-1" }})], {{ student_id: "MOHSHA011021", db_debug: debug }})]]]), {{ db_debug: debug }}));
      buttonIn(card(page, "B1-1"), "Confirm and upload").fire("click");
      await answer(page.calls, "/api/documents/k-B1-1/confirm", {{ error: "MOHSHA011021 is not among School A's students.", db_debug: debug }}, 409);
      await tick(5);
      const text = pageText();
      return {{ host: text.includes("{DB_HOST}"), name: text.includes("afdw_prod"), sql: text.includes("SELECT *"), said: text.includes("is not among School A") }};
    """)

    assert out == {"host": False, "name": False, "sql": False, "said": True}


# ------------------------------------------- scanning and OCR as two stages
@needs_node
def test_page_sends_each_file_to_api_scans_without_waiting_for_ocr(client: FlaskClient, tmp_path: Path) -> None:
    """Every file goes to /api/scans one after another; none waits for the OCR results."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      await uploadFile(page, "Scan 1.jpg", "Scan 2.jpg", "Scan 3.tif");
      const pendingAtOnce = [];
      const names = [];
      for (let i = 1; i <= 3; i += 1) {
        await tick(5);
        pendingAtOnce.push(scanCalls(page).length);
        names.push(sentName(await answerScan(page, ["p" + i])));
      }
      await tick(5);
      const out = {
        names, pendingAtOnce,
        // The first look at the results was never answered: sending did not wait on it.
        resultsUnanswered: page.calls.filter((c) => c.url.includes("/api/batches")).length,
        other: page.log.filter((c) => /extract|confirm/.test(c.url)).length,
        rows: page.byId["received-list"].children.map((li) => [li.children[0].textContent, li.children[2].textContent]),
        status: page.byId["received-status"].textContent,
      };
      // How far each page has got, as the server tells it - and a page from
      // an earlier visit that is still in the queue.
      await answerBatches(page, batches([], [pg("p3", { status: "failed", source_name: "Scan 3.tif" })],
        { queued: 2, reading: 1, failed: 1 }, [
          pg("p1", { status: "reading", source_name: "Scan 1.jpg" }),
          pg("p2", { status: "queued", source_name: "Scan 2.jpg" }),
          pg("p0", { status: "queued", source_name: "Yesterday.jpg", created_at: "2026-10-26 16:05:00" }),
        ]));
      out.later = page.byId["received-list"].children.map((li) => li.children.map((e) => e.textContent));
      out.laterStatus = page.byId["received-status"].textContent;
      return out;
    """)

    assert out["names"] == ["Scan 1.jpg", "Scan 2.jpg", "Scan 3.tif"]
    assert out["pendingAtOnce"] == [1, 1, 1]
    assert out["resultsUnanswered"] == 1
    assert out["other"] == 0
    assert out["rows"] == [
        ["Scan 3.tif", "Waiting for OCR"], ["Scan 2.jpg", "Waiting for OCR"], ["Scan 1.jpg", "Waiting for OCR"],
    ]
    assert out["status"] == "3 waiting for OCR."
    assert [row[0] for row in out["later"]] == ["Scan 3.tif", "Scan 2.jpg", "Scan 1.jpg", "Yesterday.jpg"]
    assert [row[2] for row in out["later"]] == ["Failed", "Waiting for OCR", "Reading", "Waiting for OCR"]
    assert out["later"][3][1] == "2026-10-26 16:05"
    assert out["laterStatus"] == "2 waiting for OCR · 1 being read · 1 failed."


@needs_node
def test_page_groups_results_by_batch_with_a_repeated_code_as_one_document(
    client: FlaskClient, tmp_path: Path
) -> None:
    """B1-1 on two pages is one document of two parts; each batch is its own group, then Unsorted."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const ravi = { student_id: "AAAONE000001", student_name: "Ravi Kale", school_name: "School A", grade_name: "Grade 3", division_name: "A" };
      await answerBatches(page, batches([
        [2, [doc("B2-1", [pg("p5", { code: "B2-1", source_name: "Scan 5.jpg", student_id: "AAAONE000001", matched_student: ravi })],
          { student_id: "AAAONE000001", matched_student: ravi })]],
        [1, [
          doc("B1-1", [
            pg("p1", { code: "B1-1", source_name: "Scan 1.jpg", student_id: "MOHSHA011021", matched_student: ASHA }),
            pg("p2", { code: "B1-1", source_name: "Scan 2.jpg" }),
          ], { student_id: "MOHSHA011021", matched_student: ASHA }),
          doc("B1-2", [pg("p3", { code: "B1-2", source_name: "Scan 3.jpg" })], { status: "needs_attention" }),
        ]],
      ], [pg("p4", { source_name: "Scan 4.jpg" })], { queued: 1, reading: 0, failed: 0 }));
      const groups = page.byId["ocr-batches"].children.map((section) => ({
        title: section.children[0].children[0].textContent,
        counts: section.children[0].children[1].textContent,
        codes: section.children[1].children.map((li) => li.children[0].children[0].textContent),
      }));
      const b11 = card(page, "B1-1");
      return {
        groups,
        parts: b11.children[0].children[1].textContent,
        who: textIn(b11),
        images: imagesIn(b11),
        links: linksIn(b11),
        status: page.byId["ocr-status"].textContent,
        unsorted: [page.byId["ocr-unsorted"].hidden, unsortedCards(page).length],
        received: page.byId["received-list"].children.map((li) => [li.children[0].textContent, li.children[2].textContent]),
      };
    """)

    assert out["groups"] == [
        {"title": "B2", "counts": "1 document", "codes": ["B2-1"]},
        {"title": "B1", "counts": "2 documents · 1 to check", "codes": ["B1-1", "B1-2"]},
    ]
    assert out["parts"] == "2 parts"
    for text in ("MOHSHA011021", "Asha Patil", "School A", "Grade 5", "Division B"):
        assert text in out["who"], text
    # A document that was read needs only its thumbnails, each opening the full page.
    assert out["images"] == ["/api/pages/p1/image?size=thumb", "/api/pages/p2/image?size=thumb"]
    assert out["links"] == ["/api/pages/p1/image?size=full", "/api/pages/p2/image?size=full"]
    assert out["status"] == (
        "2 documents ready to confirm · 1 document to check · 1 page without a batch code · "
        "1 page still waiting for OCR."
    )
    assert out["unsorted"] == [False, 1]
    assert sorted(out["received"]) == [[f"Scan {n}.jpg", "Read"] for n in range(1, 6)]


@needs_node
def test_page_shows_the_scan_itself_when_a_page_could_not_be_read(client: FlaskClient, tmp_path: Path) -> None:
    """Needs-attention documents and Unsorted pages show each scan full width, zoomable, with boxes to type into."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      await answerBatches(page, batches([[1, [
        doc("B1-1", [pg("p1", { code: "B1-1", student_id: "MOHSHA011021", matched_student: ASHA })], { student_id: "MOHSHA011021", matched_student: ASHA }),
        doc("B1-2", [pg("p2", { code: "B1-2" }), pg("p3", { code: "B1-2" })], { status: "needs_attention" }),
        doc("B1-3", [pg("p4", { code: "B1-3", student_id: "AAAONE000001" }), pg("p5", { code: "B1-3", student_id: "AAATWO000002" })], { status: "conflict" }),
      ]]], [pg("p6", { source_name: "Scan 6.jpg" })]));
      const describe = (c) => ({
        big: bigScans(c).map((f) => imagesIn(f)[0]),
        fields: c._all().filter((e) => e.dataset && e.dataset.field).map((e) => e.dataset.field),
        text: textIn(c),
      });
      const out = {
        ready: describe(card(page, "B1-1")), attention: describe(card(page, "B1-2")),
        conflict: describe(card(page, "B1-3")), loose: describe(unsortedCards(page)[0]),
      };
      const figure = bigScans(card(page, "B1-2"))[0];
      const zoom = buttonIn(figure, "Zoom in");
      zoom.fire("click");
      out.zoomed = [figure.classList.contains("is-zoomed"), zoom.textContent, zoom.getAttribute("aria-pressed")];
      return out;
    """)

    assert out["ready"]["big"] == []
    assert out["ready"]["fields"] == ["id"]
    assert out["attention"]["big"] == ["/api/pages/p2/image?size=full", "/api/pages/p3/image?size=full"]
    assert out["attention"]["fields"] == ["code-p2", "code-p3", "id"]
    assert "No Student ID could be read" in out["attention"]["text"]
    assert len(out["conflict"]["big"]) == 2
    assert "different Student IDs" in out["conflict"]["text"]
    assert out["loose"]["big"] == ["/api/pages/p6/image?size=full"]
    assert out["loose"]["fields"] == ["code", "id"]
    assert out["zoomed"] == [True, "Zoom out", "true"]


@needs_node
def test_page_confirm_posts_once_and_shows_the_drive_link(client: FlaskClient, tmp_path: Path) -> None:
    """A double press sends one confirm; the card then links to the PDF in Drive and offers no second confirm."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const ready = batches([[1, [doc("B1-1", [pg("p1", { code: "B1-1", student_id: "MOHSHA011021", matched_student: ASHA })],
        { student_id: "MOHSHA011021", matched_student: ASHA })]]]);
      await answerBatches(page, ready);
      const confirm = buttonIn(card(page, "B1-1"), "Confirm and upload");
      confirm.fire("click");
      confirm.fire("click");
      await tick(5);
      const posted = page.log.filter((c) => c.url.endsWith("/confirm")).length;
      const whileSaving = confirm.disabled;
      const c = await answer(page.calls, "/api/documents/k-B1-1/confirm", { document: { status: "uploaded", drive_link: "https://drive.google.com/file/d/abc/view" } });
      await tick(5);
      await answerBatches(page, ready);
      const done = card(page, "B1-1");
      const out = {
        posted, whileSaving, body: JSON.parse(c.opts.body), type: c.opts.headers["Content-Type"],
        links: linksIn(done), text: textIn(done), confirmLeft: Boolean(buttonIn(done, "Confirm and upload")),
        total: page.log.filter((call) => call.url.endsWith("/confirm")).length,
      };
      // Later the same code turns up on another form: an open document under
      // the code's key again, beside the one filed under its own key.
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, batches([[1, [
        doc("B1-1", [pg("p7", { code: "B1-1" })], { status: "needs_attention" }),
        doc("B1-1", [pg("p1", { code: "B1-1" })], { key: "k-B1-1~ab12", status: "uploaded", drive_link: "https://drive.google.com/file/d/abc/view" }),
      ]]]));
      out.again = cards(page).map((c) => [textIn(c).includes("Needs attention"), Boolean(buttonIn(c, "Confirm and upload"))]);
      return out;
    """)

    assert out["posted"] == 1
    assert out["whileSaving"] is True
    assert out["body"] == {"student_id": "MOHSHA011021", "page_ids": ["p1"]}
    assert out["type"] == "application/json"
    assert "https://drive.google.com/file/d/abc/view" in out["links"]
    assert "Uploaded" in out["text"] and "Open the PDF in Drive" in out["text"]
    assert out["confirmLeft"] is False
    assert out["total"] == 1
    assert out["again"] == [[True, True], [False, False]]


@needs_node
@pytest.mark.parametrize(
    ("status", "reply", "shown"),
    [
        (409, {"error": "MOHSHA011021 is not among School A's students - check it against the form."},
         "is not among School A's students"),
        (409, {"error": "Sign in with Google to upload to Drive"}, "Sign in with Google to upload to Drive"),
        (503, {"error": "The student list could not be reached, so nothing was saved."}, "could not be reached"),
        (503, {}, "The database could not be reached, so nothing was saved."),
    ],
)
def test_page_confirm_that_saved_nothing_says_why_and_can_be_tried_again(
    client: FlaskClient, tmp_path: Path, status: int, reply: dict[str, Any], shown: str
) -> None:
    """Not found, a Tether sign-in with no Drive, or no database: the card says so and keeps its Confirm."""
    out = _run_page(client, tmp_path, rf"""
      const page = await openPage();
      const data = batches([[1, [doc("B1-1", [pg("p1", {{ code: "B1-1", student_id: "MOHSHA011021" }})], {{ student_id: "MOHSHA011021", status: "needs_attention" }})]]]);
      await answerBatches(page, data);
      buttonIn(card(page, "B1-1"), "Confirm and upload").fire("click");
      await answer(page.calls, "/api/documents/k-B1-1/confirm", {json.dumps(reply)}, {status});
      await tick(5);
      await answerBatches(page, data);
      const c1 = card(page, "B1-1");
      return {{
        message: c1._all().find((e) => e.classList.contains("ocr-doc-message")).textContent,
        links: linksIn(c1), enabled: !buttonIn(c1, "Confirm and upload").disabled,
        text: textIn(c1),
      }};
    """)

    assert shown in out["message"]
    assert out["enabled"] is True
    assert "Needs attention" in out["text"] and "Uploaded" not in out["text"]
    if "Google" in shown:
        assert "/login/google" in out["links"]


@needs_node
def test_page_drive_failure_leaves_the_document_confirmed_with_a_retry(client: FlaskClient, tmp_path: Path) -> None:
    """Saved but not uploaded: Retry upload sends only the upload, under the key the confirm filed it as."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const parts = [pg("p1", { code: "B1-1", student_id: "MOHSHA011021", matched_student: ASHA })];
      await answerBatches(page, batches([[1, [doc("B1-1", parts, { student_id: "MOHSHA011021", matched_student: ASHA })]]]));
      buttonIn(card(page, "B1-1"), "Confirm and upload").fire("click");
      // As the server answers it: 200, the student saved, Drive not.
      await answer(page.calls, "/confirm", {
        document: { key: "B1-1~ab12", code: "B1-1", status: "confirmed", upload_error: "Drive is full." },
        uploaded: false, drive_link: null,
        message: "Saved, but the upload to Drive failed: Drive is full. Use Retry upload.",
      });
      await tick(5);
      const saved = card(page, "B1-1");
      const after = { text: textIn(saved), retry: Boolean(buttonIn(saved, "Retry upload")), confirm: Boolean(buttonIn(saved, "Confirm and upload")) };
      buttonIn(saved, "Retry upload").fire("click");
      const r = await answer(page.calls, "/upload", { document: {
        key: "B1-1~ab12", code: "B1-1", status: "uploaded", drive_link: "https://drive.google.com/file/d/abc/view",
        drive_folder: "OCR/School A/5/B/Asha Patil_MOHSHA011021",
      }, uploaded: true });
      await tick(5);
      const done = card(page, "B1-1");
      return {
        after, retry: [r.url, r.opts.method, r.opts.headers["Content-Type"], r.opts.body],
        links: linksIn(done), text: textIn(done),
        confirms: page.log.filter((c) => c.url.endsWith("/confirm")).length,
      };
    """)

    assert "Saved - not in Drive yet" in out["after"]["text"]
    assert "the upload to Drive failed: Drive is full." in out["after"]["text"]
    assert out["after"]["retry"] is True and out["after"]["confirm"] is False
    assert out["retry"] == ["/api/documents/B1-1~ab12/upload", "POST", "application/json", "{}"]
    assert "https://drive.google.com/file/d/abc/view" in out["links"]
    assert "Uploaded" in out["text"] and "OCR/School A/5/B/Asha Patil_MOHSHA011021" in out["text"]
    assert out["confirms"] == 1


@needs_node
def test_page_retry_upload_that_fails_again_says_so_and_stays_ready_to_retry(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Drive failing a second time (502) leaves the document saved, with Retry upload still there."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const parts = [pg("p1", { code: "B1-1", student_id: "MOHSHA011021", matched_student: ASHA })];
      await answerBatches(page, batches([[1, [doc("B1-1", parts, {
        key: "B1-1~ab12", status: "confirmed", student_id: "MOHSHA011021", matched_student: ASHA,
        message: "Saved, but the upload to Drive failed. Use Retry upload.", can_confirm: false,
      })]]]));
      const c1 = card(page, "B1-1");
      buttonIn(c1, "Retry upload").fire("click");
      await answer(page.calls, "/api/documents/B1-1~ab12/upload", { error: "Drive is full. The student is saved; try the upload again." }, 502);
      await tick(5);
      return {
        message: c1._all().find((e) => e.classList.contains("ocr-doc-message")).textContent,
        retry: !buttonIn(c1, "Retry upload").disabled, confirm: Boolean(buttonIn(c1, "Confirm and upload")),
      };
    """)

    assert "Drive is full" in out["message"]
    assert out["retry"] is True
    assert out["confirm"] is False


_SORT_SCAN_FOLDER_JS = r"""
  const clock = makeClock();
  const folder = makeFolder("OCR Scans", { move: %(move)s });
  folder.add("Old 001.jpg");
  // From an earlier day: a scan of the same name already in B1.
  folder.sub("B1").add("Scan 001.jpg", 999);
  const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
  await answerBatches(page, EMPTY);
  await connectFolder(page);
  page.byId["scan-folder-only-new"].fire("click");
  await settle();
  folder.add("Scan 001.jpg");
  folder.add("Scan 002.jpg");
  folder.add("Scan 003.jpg");
  await clock.advance(4000);
  for (const id of ["p1", "p2", "p3"]) await answerScan(page, [id]);
  await clock.advance(3000);
  await answerBatches(page, batches([
    [2, [doc("B2-1", [pg("p2", { code: "B2-1", source_name: "Scan 002.jpg" })])]],
    [1, [doc("B1-1", [pg("p1", { code: "B1-1", source_name: "Scan 001.jpg" })])]],
  ], [pg("p3", { source_name: "Scan 003.jpg" })]));
  await clock.advance(2000);
  const sentBefore = everyScan(page).length;
  await clock.advance(10000);
  return {
    top: folder.names(), folders: folder.folders(), b1: folder.sub("B1").names(), b2: folder.sub("B2").names(),
    kept: folder.sub("B1").size("Scan 001.jpg"), made: folder.made, moves: folder.moves.length, removed: folder.removed,
    writes: folder.sub("B1").writes + folder.sub("B2").writes,
    resent: everyScan(page).length - sentBefore, sent: everyScan(page).map(sentName),
    picker: page.pickerOptions, note: page.byId["scan-folder-sort"].textContent,
    error: page.byId["scan-folder-error"].textContent,
  };
"""


@needs_node
@pytest.mark.parametrize("how", ["move", "copy"])
def test_page_sorts_the_scans_it_sent_into_batch_folders(client: FlaskClient, tmp_path: Path, how: str) -> None:
    """Each sent scan whose code is read moves into B<n>/; nothing else in the folder is touched or sent again."""
    out = _run_page(client, tmp_path, _SORT_SCAN_FOLDER_JS % {"move": "true" if how == "move" else "false"})

    assert out["picker"] == {"id": "ocr-scan-folder", "mode": "readwrite"}
    # Never sent, and not read: the old scan and the one with no code stay.
    assert out["top"] == ["Old 001.jpg", "Scan 003.jpg"]
    assert out["folders"] == ["B1", "B2"]
    # Beside, never over, the scan already in B1.
    assert out["b1"] == ["Scan 001 (2).jpg", "Scan 001.jpg"]
    assert out["kept"] == 999
    assert out["b2"] == ["Scan 002.jpg"]
    assert out["made"] == ["B2"]
    if how == "move":
        assert (out["moves"], out["writes"], out["removed"]) == (2, 0, [])
    else:
        assert (out["moves"], out["writes"], out["removed"]) == (0, 2, ["Scan 001.jpg", "Scan 002.jpg"])
    # The watch reads only the folder's own files, not the batch folders.
    assert out["resent"] == 0
    assert out["sent"] == ["Scan 001.jpg", "Scan 002.jpg", "Scan 003.jpg"]
    assert "moved into a folder for its batch" in out["note"]
    assert out["error"] == ""


@needs_node
def test_page_read_only_scan_folder_keeps_sending_without_sorting(client: FlaskClient, tmp_path: Path) -> None:
    """Editing refused: the scans are still sent, nothing is moved, and the card says why."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder("OCR Scans", { move: true, write: "denied" });
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      await clock.advance(4000);
      await answerScan(page, ["p1"]);
      await clock.advance(3000);
      await answerBatches(page, batches([[1, [doc("B1-1", [pg("p1", { code: "B1-1" })])]]]));
      await clock.advance(4000);
      const note = page.byId["scan-folder-sort"];
      return {
        top: folder.names(), made: folder.made, moves: folder.moves.length,
        state: page.byId["scan-folder-state"].textContent,
        note: note.textContent, warn: note.classList.contains("is-warn"), hidden: note.hidden,
        error: page.byId["scan-folder-error"].textContent,
      };
    """)

    assert out["top"] == ["Scan 001.jpg"]
    assert out["made"] == [] and out["moves"] == 0
    assert "Watching - 1 page sent today" in out["state"]
    assert "may read" in out["note"] and "but not change it" in out["note"]
    assert out["warn"] is True and out["hidden"] is False
    assert out["error"] == ""


@needs_node
def test_page_asks_for_results_only_while_there_is_work_and_the_tab_is_visible(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Every 3 s while pages are still queued, one request at a time; none while hidden or idle."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const page = await openPage({ clock });
      const inFlight = () => page.calls.filter((c) => c.url.includes("/api/batches")).length;
      const asked = () => page.log.filter((c) => c.url.includes("/api/batches")).length;
      const out = { first: inFlight() };
      await answerBatches(page, batches([], [], { queued: 2, reading: 0, failed: 0 }));
      out.timersWhileWork = clock.timers.size;
      await clock.advance(2900);
      out.before3s = asked();
      await clock.advance(100);
      out.at3s = asked();
      // Asked for again while one is in the air: made once, after it.
      page.byId["ocr-refresh"].fire("click");
      page.byId["ocr-refresh"].fire("click");
      out.inFlightAtOnce = inFlight();
      await answerBatches(page, batches([], [], { queued: 1, reading: 0, failed: 0 }));
      await tick(5);
      out.afterAgain = asked();
      await answerBatches(page, batches([], [], { queued: 1, reading: 1, failed: 0 }));
      page.document.visibilityState = "hidden";
      page.document.fire("visibilitychange");
      out.timersWhileHidden = clock.timers.size;
      const hiddenFrom = asked();
      await clock.advance(15000);
      out.askedWhileHidden = asked() - hiddenFrom;
      page.document.visibilityState = "visible";
      page.document.fire("visibilitychange");
      page.document.fire("visibilitychange");
      await settle();
      out.onShow = inFlight();
      await answerBatches(page, batches([[1, [doc("B1-1", [pg("p1", { code: "B1-1" })])]]]));
      out.timersWhenIdle = clock.timers.size;
      const idleFrom = asked();
      await clock.advance(30000);
      out.askedWhenIdle = asked() - idleFrom;
      return out;
    """)

    assert out == {
        "first": 1, "timersWhileWork": 1, "before3s": 1, "at3s": 2, "inFlightAtOnce": 1, "afterAgain": 3,
        "timersWhileHidden": 0, "askedWhileHidden": 0, "onShow": 1, "timersWhenIdle": 0, "askedWhenIdle": 0,
    }


def test_page_has_no_automatic_save(client: FlaskClient) -> None:
    """The "Check each form automatically" mode is gone; the one /api/confirm left is the typed-in ID."""
    html = client.get("/").get_data(as_text=True)
    script = next(s for s in re.findall(r"<script>(.*?)</script>", html, re.DOTALL) if "const API_BASE" in s)

    for gone in ('id="auto-mode"', "Check each form automatically", "/api/extract-batch", "auto: true", "confirm-modal"):
        assert gone not in html, gone
    assert script.count("/api/confirm") == 1
    start = script.index('manualForm.addEventListener("submit"')
    assert "/api/confirm" in script[start:script.index("\n    });", start)]


@needs_node
def test_page_never_saves_a_document_nobody_confirmed(client: FlaskClient, tmp_path: Path) -> None:
    """A document read cleanly and matched still waits for a person: the page writes nothing by itself."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      await clock.advance(4000);
      await answerScan(page, ["p1"]);
      await clock.advance(3000);
      await answerBatches(page, batches([[1, [doc("B1-1", [pg("p1", { code: "B1-1", student_id: "MOHSHA011021", matched_student: ASHA })],
        { student_id: "MOHSHA011021", matched_student: ASHA })]]]));
      await clock.advance(60000);
      return {
        writes: page.log.filter((c) => c.opts.method === "POST" && !c.url.includes("/api/scans")).map((c) => c.url),
        waiting: Boolean(buttonIn(card(page, "B1-1"), "Confirm and upload")),
      };
    """)

    assert out == {"writes": [], "waiting": True}


# ------------------------------------------- scanning: the scan folder
def test_page_has_no_camera_and_no_scan_helper(client: FlaskClient) -> None:
    """Staff scan with the scanner; the page neither opens a camera nor calls a local helper."""
    html = client.get("/").get_data(as_text=True)

    for gone in ("getUserMedia", "capture=", "Use camera", "Take photo", "8765", "scan-helper", "scan helper"):
        assert gone not in html, gone
    assert 'id="scan-folder-connect"' in html and ">Connect scan folder<" in html
    assert 'accept=".jpg,.jpeg,.png,.pdf,.tif,.tiff"' in html
    assert re.search(r'<input[^>]*id="file-input"[^>]*\bmultiple\b', html, re.DOTALL)


@needs_node
def test_page_scan_folder_sends_a_scan_once_it_stops_changing(client: FlaskClient, tmp_path: Path) -> None:
    """A scan is sent only after its size and time are the same at two looks in a row."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await connectFolder(page);
      const sentAfter = [];
      folder.add("Scan 001.JPG");
      folder.add("notes.txt");
      // Saved and taken away again before a second look: never sent.
      folder.add("Scan 000.jpg");
      await clock.advance(2000);   // first seen
      sentAfter.push(scanCalls(page).length);
      folder.remove("Scan 000.jpg");
      folder.grow("Scan 001.JPG");
      await clock.advance(2000);   // still being written
      sentAfter.push(scanCalls(page).length);
      await clock.advance(2000);   // the same as last time: sent
      sentAfter.push(scanCalls(page).length);
      await clock.advance(4000);
      sentAfter.push(scanCalls(page).length);
      return {
        sentAfter,
        names: scanCalls(page).map(sentName),
        picker: page.pickerOptions,
        setupOpen: page.byId["scan-folder-setup"].open,
        state: page.byId["scan-folder-state"].textContent,
      };
    """)

    assert out["sentAfter"] == [0, 0, 1, 1]
    assert out["names"] == ["Scan 001.JPG"]
    # Asked with editing, so the page can sort what it sent into batch folders.
    assert out["picker"] == {"id": "ocr-scan-folder", "mode": "readwrite"}
    # The one-time setup folds away once a folder is connected.
    assert out["setupOpen"] is False
    assert '"OCR Scans"' in out["state"] and "Watching - 0 pages sent today" in out["state"]


@needs_node
def test_page_scan_folder_never_sends_a_scan_twice_even_after_a_reload(
    client: FlaskClient, tmp_path: Path
) -> None:
    """What was sent is remembered in IndexedDB by name, size and time, and a later visit reconnects."""
    out = _run_page(client, tmp_path, r"""
      const idb = makeIndexedDB();
      const folder = makeFolder();
      const clock1 = makeClock();
      const first = await openScannerPage({ clock: clock1, idb, folder });
      await connectFolder(first);
      folder.add("Scan 001.jpg");
      await clock1.advance(4000);
      await answerScan(first, ["p1"]);
      await clock1.advance(4000);
      const firstSent = scanCalls(first).length;
      const firstState = first.byId["scan-folder-state"].textContent;

      // A reload: the same browser, so the same IndexedDB, and Chrome has
      // let the permission lapse.
      folder.permission = "prompt";
      const clock2 = makeClock();
      const second = await openScannerPage({ clock: clock2, idb, folder });
      await settle();
      const connect = second.byId["scan-folder-connect"];
      const offered = { hidden: connect.hidden, label: connect.textContent, state: second.byId["scan-folder-state"].textContent };
      connect.fire("click");
      await settle();
      await clock2.advance(6000);
      const resent = scanCalls(second).map(sentName);
      folder.add("Scan 002.jpg");
      await clock2.advance(4000);
      return { firstSent, firstState, offered, resent, after: scanCalls(second).map(sentName) };
    """)

    # Answered, and not sent again on later looks in the same visit.
    assert out["firstSent"] == 0
    assert "1 page sent today" in out["firstState"]
    assert out["offered"]["hidden"] is False
    assert out["offered"]["label"] == "Reconnect scan folder"
    assert '"OCR Scans"' in out["offered"]["state"]
    assert out["resent"] == []
    assert out["after"] == ["Scan 002.jpg"]


_ASK_ABOUT_OLD_SCANS_JS = r"""
  const clock = makeClock();
  const idb = makeIndexedDB();
  const folder = makeFolder();
  folder.add("Old 001.jpg");
  folder.add("Old 002.jpg");
  folder.add("Old 003.tif");
  const page = await openScannerPage({ clock, idb, folder });
  await connectFolder(page);
  const ask = page.byId["scan-folder-ask"];
  const asked = { hidden: ask.hidden, text: page.byId["scan-folder-ask-text"].textContent };
  const looksBefore = folder.listings;
  await clock.advance(6000);
  const whileAsking = { looks: folder.listings - looksBefore, sent: scanCalls(page).length };
  page.byId["scan-folder-%(choice)s"].fire("click");
  await settle();
  await clock.advance(4000);
  const names = [];
  for (let i = 0; i < 3 && scanCalls(page).length; i += 1) {
    const c = await answerScan(page, ["old" + i]);
    names.push(sentName(c));
    await settle();
  }
  folder.add("New 001.jpg");
  await clock.advance(4000);
  const c = await answerScan(page, ["new1"]);
  names.push(sentName(c));
  // Asked once: the next visit remembers the answer.
  const again = await openScannerPage({ clock: makeClock(), idb, folder });
  return { asked, whileAsking, names, askedAgain: !again.byId["scan-folder-ask"].hidden && again.byId["scan-folder-ask-text"].textContent !== "" };
"""


@needs_node
@pytest.mark.parametrize("choice", ["only-new", "send-old"])
def test_page_scan_folder_asks_about_scans_already_in_it(
    client: FlaskClient, tmp_path: Path, choice: str
) -> None:
    """An old folder is not read by surprise: the person says whether its scans go too."""
    out = _run_page(client, tmp_path, _ASK_ABOUT_OLD_SCANS_JS % {"choice": choice})

    assert out["asked"]["hidden"] is False
    assert out["asked"]["text"] == "3 scans already in this folder - send them too, or only new ones?"
    assert out["whileAsking"] == {"looks": 0, "sent": 0}
    if choice == "only-new":
        assert out["names"] == ["New 001.jpg"]
    else:
        assert out["names"] == ["Old 001.jpg", "Old 002.jpg", "Old 003.tif", "New 001.jpg"]
    assert out["askedAgain"] is False


@needs_node
def test_page_scan_folder_looks_only_while_on_screen_and_never_twice_at_once(
    client: FlaskClient, tmp_path: Path
) -> None:
    """One look every 2 s while the tab is visible; none while hidden; stop/start never stacks them."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await connectFolder(page);
      const looks = (fn) => async () => { const before = folder.listings; await fn(); return folder.listings - before; };
      const out = {};
      out.visible = await looks(() => clock.advance(10000))();
      page.document.visibilityState = "hidden";
      page.document.fire("visibilitychange");
      out.timersWhileHidden = clock.timers.size;
      out.hidden = await looks(() => clock.advance(10000))();
      folder.add("Scan 001.jpg");
      await clock.advance(10000);
      out.sentWhileHidden = scanCalls(page).length;
      page.document.visibilityState = "visible";
      page.document.fire("visibilitychange");
      page.document.fire("visibilitychange");
      await settle();
      // Stop and start, quickly, again and again.
      const toggle = page.byId["scan-folder-toggle"];
      for (let i = 0; i < 4; i += 1) { toggle.fire("click"); await tick(); }
      await settle();
      out.watching = toggle.textContent;
      out.timersAfter = clock.timers.size;
      out.afterToggles = await looks(() => clock.advance(10000))();
      out.sentAfter = scanCalls(page).map(sentName);
      return out;
    """)

    assert out["visible"] == 5
    assert out["timersWhileHidden"] == 0
    assert out["hidden"] == 0
    assert out["sentWhileHidden"] == 0
    assert out["watching"] == "Stop watching"
    assert out["timersAfter"] <= 1
    assert out["afterToggles"] == 5
    assert out["sentAfter"] == ["Scan 001.jpg"]


@needs_node
def test_page_scan_folder_in_a_private_window_remembers_for_the_visit_and_says_so(
    client: FlaskClient, tmp_path: Path
) -> None:
    """No IndexedDB: an in-memory list keeps scans from going twice, and the page says to reconnect next time."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      folder.add("Old 001.jpg");
      const page = await openScannerPage({ clock, idb: undefined, folder });
      const noteOnLoad = page.byId["scan-folder-note"].textContent;
      await connectFolder(page);
      const asked = page.byId["scan-folder-ask-text"].textContent;
      page.byId["scan-folder-only-new"].fire("click");
      await settle();
      folder.add("Scan 001.jpg");
      await clock.advance(4000);
      await answer(page.calls, "/api/scans", { pages: [] }, 202);
      await clock.advance(6000);
      // The same folder chosen again this visit: nothing to ask, nothing resent.
      page.byId["scan-folder-change"].fire("click");
      await settle();
      await clock.advance(6000);
      return {
        noteOnLoad,
        asked,
        askedAgain: !page.byId["scan-folder-ask"].hidden,
        sent: scanCalls(page).map(sentName),
        note: page.byId["scan-folder-note"].textContent,
        state: page.byId["scan-folder-state"].textContent,
      };
    """)

    assert out["asked"] == "1 scan already in this folder - send it too, or only new ones?"
    assert out["askedAgain"] is False
    assert out["sent"] == []
    assert "connect the scan folder again on each visit" in out["noteOnLoad"]
    assert "connect the scan folder again on each visit" in out["note"]
    assert "Watching - 1 page sent today" in out["state"]


@needs_node
def test_page_without_a_folder_watch_says_so_and_offers_add_scanned_files(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Firefox and Safari: no folder to watch, so the card says so and points at Add scanned files."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const b = page.byId;
      return {
        state: b["scan-folder-state"].textContent,
        connectHidden: b["scan-folder-connect"].hidden,
        // Add scanned files is offered as it stands in the page: nothing hides it.
        tilesHidden: page.document.getElementById("upload-tiles").hidden,
        dropHidden: b["drop-zone"].hidden,
        steps: b["scan-folder-steps"].innerHTML,
      };
    """)

    assert "needs Chrome or Edge" in out["state"]
    assert "add the scanned files below" in out["state"]
    assert out["connectHidden"] is True
    assert out["tilesHidden"] is False and out["dropHidden"] is False
    assert "Choose files" in out["steps"] and "Connect scan folder" not in out["steps"]


@needs_node
def test_page_drop_zone_and_choose_files_take_the_same_way_in(client: FlaskClient, tmp_path: Path) -> None:
    """A drop and Choose files are both sent straight away, the same way, for the picker's school."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const b = page.byId;
      let chooserOpened = 0;
      b["file-input"].addEventListener("click", () => { chooserOpened += 1; });
      b["drop-zone"].fire("keydown", { key: "Enter" });
      b["drop-zone"].fire("keydown", { key: " " });
      b["drop-zone"].fire("keydown", { key: "a" });
      b["drop-zone"].fire("click");

      const out = { chooserOpened, ways: [] };
      b["drop-zone"].fire("drop", { dataTransfer: { types: ["Files"], files: [{ name: "notes.txt" }, { name: "dropped.jpg" }] } });
      out.dropMessage = b["drop-message"].textContent;
      for (const how of ["drop", "choose"]) {
        if (how === "choose") await uploadFile(page, "chosen.TIFF");
        const sent = await answerScan(page, ["p-" + how]);
        const form = sentForm(sent);
        out.ways.push([how, sent.opts.method, form.file, form.school, form.location]);
        await tick(5);
      }
      out.fileInputCleared = b["file-input"].value === "";
      out.saves = page.log.filter((c) => /confirm|extract/.test(c.url)).length;
      return out;
    """)

    assert out["chooserOpened"] == 3
    assert "1 item(s) left out" in out["dropMessage"]
    assert out["ways"] == [
        ["drop", "POST", "dropped.jpg", "School A", "Pune"],
        ["choose", "POST", "chosen.TIFF", "School A", "Pune"],
    ]
    assert out["fileInputCleared"] is True
    assert out["saves"] == 0


@needs_node
@pytest.mark.parametrize(
    ("navigator", "expected"),
    [
        (
            {"userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "platform": "Win32", "maxTouchPoints": 0},
            ["PaperStream IP (TWAIN)", "PaperStream Capture", "Assign to scanner button",
             "Documents\\OCR Scans", "100 dpi", "Black &amp; White", "simplex", "document feeder",
             "https://www.pfu.ricoh.com/global/scanners/fisp/dl/sp-1130n.html", "face down",
             "Connect scan folder"],
        ),
        (
            {"userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "platform": "MacIntel", "maxTouchPoints": 0},
            ["Image Scanner Driver for macOS (ICA)", "Image Capture", "Use Document Feeder",
             "100 dpi", "JPEG or TIFF", "Scan To", "OCR Scans", "face down"],
        ),
        (
            {"userAgent": "Mozilla/5.0 (X11; Linux x86_64)", "platform": "Linux x86_64", "maxTouchPoints": 0},
            ["Document Scanner", "simple-scan", "fujitsu", "OCR Scans", "face down"],
        ),
        (
            {"userAgent": "Mozilla/5.0 (X11; CrOS x86_64 14541.0.0)", "platform": "Linux x86_64", "maxTouchPoints": 0},
            ["<strong>Scan</strong> app", "OCR Scans", "face down"],
        ),
    ],
)
def test_page_scanner_setup_follows_the_visitors_computer(
    client: FlaskClient, tmp_path: Path, navigator: dict[str, Any], expected: list[str]
) -> None:
    """The one-time setup is for the computer the scanner is plugged into."""
    out = _run_page(client, tmp_path, rf"""
      const page = loadPage({{
        navigator: {json.dumps(navigator)},
        showDirectoryPicker: async () => makeFolder(),
        indexedDB: makeIndexedDB(),
      }});
      return page.byId["scan-folder-steps"].innerHTML;
    """)

    for text in expected:
        assert text in out, text


# ------------------------------------------- a send that did not get through
@needs_node
def test_page_scan_folder_tries_a_refused_send_again_until_the_server_takes_it(
    client: FlaskClient, tmp_path: Path
) -> None:
    """A 401 or a 500 says nothing about the file: it is sent again, not booked as sent and lost."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const idb = makeIndexedDB();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb, folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      folder.add("Scan 002.jpg");
      // Signed out in another tab, then the server's disk fills up; after
      // that every send is taken.
      const refusals = [
        [{ error: "Please sign in with Google to continue." }, 401],
        [{ error: "Could not store this scan on the server. Try again." }, 500],
      ];
      const refused = await serveScans(page, clock, 20, (_call, n) => refusals[n - 1] || null);
      const whileRefused = {
        state: page.byId["scan-folder-state"].textContent,
        error: page.byId["scan-folder-error"].textContent,
        rows: page.byId["received-list"].children.map((li) => li.children[2].textContent),
      };
      const taken = await serveScans(page, clock, 100);
      const out = {
        tries: [...refused, ...taken], taken, whileRefused,
        state: page.byId["scan-folder-state"].textContent,
        error: page.byId["scan-folder-error"].textContent,
      };
      // A reload: both are in the ledger as sent, so neither goes again.
      const clock2 = makeClock();
      const second = await openScannerPage({ clock: clock2, idb, folder });
      await answerBatches(second, EMPTY);
      out.resent = (await serveScans(second, clock2, 10)).length;
      return out;
    """)

    # One at a time and in scan order: Scan 002 waits until Scan 001 is in.
    assert out["tries"] == ["Scan 001.jpg", "Scan 001.jpg", "Scan 001.jpg", "Scan 002.jpg"]
    assert out["taken"] == ["Scan 001.jpg", "Scan 002.jpg"]
    assert "0 pages sent today" in out["whileRefused"]["state"]
    assert "Could not store this scan on the server" in out["whileRefused"]["error"]
    assert "trying again" in out["whileRefused"]["error"]
    assert out["whileRefused"]["rows"] == ["Not sent yet - trying again", "Not sent yet - trying again"]
    assert "2 pages sent today" in out["state"]
    assert out["error"] == ""
    assert out["resent"] == 0


@needs_node
def test_page_scan_folder_file_the_server_refuses_is_not_counted_as_sent(
    client: FlaskClient, tmp_path: Path
) -> None:
    """A 400 is about the file itself: it is not sent again, and not counted as a page sent."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      const tries = await serveScans(page, clock, 60, () => [{ error: "This file is damaged and could not be read." }, 400]);
      return {
        tries,
        state: page.byId["scan-folder-state"].textContent,
        rows: page.byId["received-list"].children.map((li) => li.children.map((e) => e.textContent)),
      };
    """)

    assert out["tries"] == ["Scan 001.jpg"]
    assert "0 pages sent today" in out["state"]
    assert out["rows"][0][0] == "Scan 001.jpg"
    assert out["rows"][0][2:] == ["Not received", "This file is damaged and could not be read."]


@needs_node
@pytest.mark.parametrize("way", ["folder", "files"])
def test_page_send_with_no_answer_holds_the_files_behind_it_in_scan_order(
    client: FlaskClient, tmp_path: Path, way: str
) -> None:
    """The server orders a document's parts by arrival, so a file that got no answer goes again before the next."""
    out = _run_page(client, tmp_path, rf"""
      const clock = makeClock();
      const folder = makeFolder();
      let page;
      if ("{way}" === "folder") {{
        page = await openScannerPage({{ clock, idb: makeIndexedDB(), folder }});
        await answerBatches(page, EMPTY);
        await connectFolder(page);
        folder.add("Scan 001.jpg");
        folder.add("Scan 002.jpg");
        await clock.advance(4000);
      }} else {{
        page = await openPage({{ clock }});
        await answerBatches(page, EMPTY);
        await uploadFile(page, "Scan 001.jpg", "Scan 002.jpg");
        await tick(5);
      }}
      // The ID page goes out as Wi-Fi drops for a moment.
      const first = sentName(await fail(page.calls, "/api/scans"));
      await tick(5);
      const held = {{
        waiting: scanCalls(page).map(sentName),
        rows: page.byId["received-list"].children.map((li) => [li.children[0].textContent, li.children[2].textContent]),
      }};
      const later = await serveScans(page, clock, 60);
      return {{ order: [first, ...later], held, status: page.byId["received-status"].textContent }};
    """)

    assert out["order"] == ["Scan 001.jpg", "Scan 001.jpg", "Scan 002.jpg"]
    assert out["held"]["waiting"] == []
    assert ["Scan 001.jpg", "Not sent yet - trying again"] in out["held"]["rows"]
    assert out["status"] == "2 waiting for OCR."


@needs_node
def test_page_scan_folder_drops_a_held_scan_that_was_deleted_and_sends_the_rest(
    client: FlaskClient, tmp_path: Path
) -> None:
    """A scan held for a retry that is then taken out of the folder does not hold up the scans behind it."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder();
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      folder.add("Scan 002.jpg");
      await clock.advance(4000);
      await fail(page.calls, "/api/scans");
      folder.remove("Scan 001.jpg");
      const later = await serveScans(page, clock, 60);
      return {
        later,
        rows: page.byId["received-list"].children.map((li) => li.children[0].textContent),
        state: page.byId["scan-folder-state"].textContent,
      };
    """)

    assert out["later"] == ["Scan 002.jpg"]
    assert out["rows"] == ["Scan 002.jpg"]
    assert "1 page sent today" in out["state"]


# ------------------------------------------- what a card says and offers
@needs_node
def test_page_conflicting_parts_name_no_student_and_leave_the_id_box_empty(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Parts matched to two children: the card names neither, so one press cannot file both under one."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const ravi = { student_id: "AAAONE000001", student_name: "Ravi Kale", school_name: "School A", grade_name: "3", division_name: "A" };
      // As documents._judge sends a conflict.
      await answerBatches(page, batches([[1, [doc("B1-3", [
        pg("p4", { code: "B1-3", student_id: "MOHSHA011021", matched_student: ASHA }),
        pg("p5", { code: "B1-3", student_id: "AAAONE000001", matched_student: ravi }),
      ], {
        status: "needs_attention", attention: ["conflict"], matched_student: null, student_id: "",
        message: "The parts match different students: AAAONE000001, MOHSHA011021.",
      })]]]));
      const c3 = card(page, "B1-3");
      const who = c3._all().find((e) => e.classList.contains("ocr-doc-student")).textContent;
      const box = fieldIn(c3, "id").value;
      buttonIn(c3, "Confirm and upload").fire("click");
      await tick(5);
      return {
        who, box, big: bigScans(c3).length, text: textIn(c3),
        confirms: page.log.filter((c) => c.url.endsWith("/confirm")).length,
      };
    """)

    assert "Asha" not in out["who"] and "Ravi" not in out["who"]
    assert "different students" in out["who"]
    assert out["box"] == ""
    assert out["big"] == 2
    assert "The parts match different students: AAAONE000001, MOHSHA011021." in out["text"]
    assert out["confirms"] == 0


@needs_node
def test_page_ready_card_looks_up_a_changed_id_before_it_confirms(client: FlaskClient, tmp_path: Path) -> None:
    """A Ready card whose ID is changed says it is not checked yet, offers Check ID, and saves nobody unseen."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const ravi = { student_id: "AAAONE000001", student_name: "Ravi Kale", school_name: "School A", grade_name: "3", division_name: "A" };
      await answerBatches(page, batches([[1, [doc("B1-1", [pg("p1", { code: "B1-1", student_id: "MOHSHA011021", matched_student: ASHA })],
        { student_id: "MOHSHA011021", matched_student: ASHA })]]]));
      let c1 = card(page, "B1-1");
      const student = (c) => c._all().find((e) => e.classList.contains("ocr-doc-student"));
      const offered = Boolean(buttonIn(c1, "Check ID"));
      const box = fieldIn(c1, "id");
      box.value = "aaaone000001";
      box.fire("input");
      const who = [student(c1).textContent, student(c1).classList.contains("is-warn")];
      buttonIn(c1, "Confirm and upload").fire("click");
      await tick(5);
      const early = {
        confirms: page.log.filter((c) => c.url.endsWith("/confirm")).length,
        message: c1._all().find((e) => e.classList.contains("ocr-doc-message")).textContent,
      };
      // Typed back as it was: the card names Asha again.
      box.value = "MOHSHA011021";
      box.fire("input");
      const back = student(c1).textContent;
      box.value = "AAAONE000001";
      box.fire("input");
      box.fire("keydown", { key: "Enter" });
      const looked = await answer(page.calls, "/student-id", { ok: true });
      await answerBatches(page, batches([[1, [doc("B1-1", [pg("p1", { code: "B1-1", student_id: "AAAONE000001", matched_student: ravi })],
        { student_id: "AAAONE000001", matched_student: ravi })]]]));
      c1 = card(page, "B1-1");
      const after = student(c1).textContent;
      buttonIn(c1, "Confirm and upload").fire("click");
      const c = await answer(page.calls, "/confirm", { document: { status: "uploaded", drive_link: "https://drive.google.com/x" } });
      return { offered, who, early, back, looked: [looked.url, JSON.parse(looked.opts.body)], after, body: JSON.parse(c.opts.body) };
    """)

    assert out["offered"] is True
    assert "AAAONE000001" in out["who"][0] and "not checked yet" in out["who"][0]
    assert "Asha" not in out["who"][0] and out["who"][1] is True
    assert out["early"]["confirms"] == 0
    assert "Check ID" in out["early"]["message"]
    assert "Asha Patil" in out["back"]
    assert out["looked"] == ["/api/pages/p1/student-id", {"student_id": "AAAONE000001"}]
    assert "Ravi Kale" in out["after"]
    assert out["body"] == {"student_id": "AAAONE000001", "page_ids": ["p1"]}


@needs_node
def test_page_scan_that_failed_to_load_can_be_tried_again(client: FlaskClient, tmp_path: Path) -> None:
    """An image that did not load keeps its place with a Try again, and Refresh asks for every broken one again."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      const data = batches([[1, [doc("B1-2", [pg("p2", { code: "B1-2" })], { status: "needs_attention" })]]]);
      await answerBatches(page, data);
      const c1 = card(page, "B1-2");
      const figure = bigScans(c1)[0];
      const img = figure._all().find((e) => e.tagName === "IMG");
      img.fire("error");
      const broken = { kept: figure._all().includes(img), text: textIn(figure), retry: Boolean(buttonIn(figure, "Try again")) };
      buttonIn(figure, "Try again").fire("click");
      const retried = img.src;
      img.fire("load");
      const loaded = { text: textIn(figure), retry: Boolean(buttonIn(figure, "Try again")) };
      // A thumbnail that did not load: Refresh draws the card again, so it is
      // asked for again - though not while someone is typing in the card.
      c1._all().find((e) => e.tagName === "IMG" && e.src.includes("size=thumb")).fire("error");
      const thumbText = textIn(c1);
      page.document.activeElement = fieldIn(c1, "id");
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, data);
      const whileTyping = card(page, "B1-2") === c1;
      page.document.activeElement = null;
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, data);
      const again = card(page, "B1-2");
      return { broken, retried, loaded, thumbText, whileTyping, rebuilt: again !== c1 };
    """)

    assert out["broken"]["kept"] is True
    assert "could not be loaded" in out["broken"]["text"]
    assert out["broken"]["retry"] is True
    assert out["retried"].startswith("/api/pages/p2/image?size=full&retry=")
    assert "could not be loaded" not in out["loaded"]["text"] and out["loaded"]["retry"] is False
    assert "Part 1 - not loaded, press Refresh" in out["thumbText"]
    assert out["whileTyping"] is True
    assert out["rebuilt"] is True


@needs_node
def test_page_zoomed_scan_can_be_scrolled_from_the_keyboard(client: FlaskClient, tmp_path: Path) -> None:
    """The scan's frame takes the focus, so arrow keys reach the corner where the code is written."""
    out = _run_page(client, tmp_path, r"""
      const page = await openPage();
      await answerBatches(page, batches([[1, [doc("B1-2", [pg("p2", { code: "B1-2" })], { status: "needs_attention" })]]]));
      const frame = bigScans(card(page, "B1-2"))[0]._all().find((e) => e.classList.contains("ocr-scan-frame"));
      return frame.attrs;
    """)
    css = (Path(__file__).resolve().parent.parent / "client" / "static" / "style.css").read_text(encoding="utf-8")

    assert out["tabindex"] == "0"
    assert out["role"] == "region"
    assert out["aria-label"] == "Scan of B1-2 - part 1 of 1"
    assert ".ocr-scan-frame:focus-visible" in css


@needs_node
def test_page_redraw_leaves_cards_that_did_not_change_where_they_are(client: FlaskClient, tmp_path: Path) -> None:
    """A card added or changed beside a zoomed scan does not take that scan out of the page and put it back."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const page = await openPage({ clock });
      const b11 = (extra) => doc("B1-1", [pg("p1", Object.assign({ code: "B1-1" }, extra))], extra);
      const b12 = doc("B1-2", [pg("p2", { code: "B1-2" })], { status: "needs_attention" });
      const b13 = doc("B1-3", [pg("p3", { code: "B1-3" })], { status: "needs_attention" });
      await answerBatches(page, batches([[1, [b11({ status: "waiting" }), b12]]], [], { queued: 2, reading: 0, failed: 0 }));
      const zoomed = card(page, "B1-2");
      const list = zoomed.parent;
      buttonIn(bigScans(zoomed)[0], "Zoom in").fire("click");
      const touched = { emptied: 0, reinserted: 0 };
      Object.defineProperty(list, "innerHTML", { get() { return ""; }, set() { touched.emptied += 1; list.children = []; } });
      for (const name of ["appendChild", "insertBefore"]) {
        const real = list[name].bind(list);
        list[name] = (node, ...rest) => { if (node === zoomed) touched.reinserted += 1; return real(node, ...rest); };
      }
      await clock.advance(3000);
      await answerBatches(page, batches([[1, [b11({ student_id: "MOHSHA011021", matched_student: ASHA }), b12]]], [], { queued: 1, reading: 0, failed: 0 }));
      await clock.advance(3000);
      await answerBatches(page, batches([[1, [b13, b11({ student_id: "MOHSHA011021", matched_student: ASHA }), b12]]]));
      return {
        touched,
        codes: list.children.map((li) => li.children[0].children[0].textContent),
        same: card(page, "B1-2") === zoomed, zoomedStill: bigScans(zoomed)[0].classList.contains("is-zoomed"),
        b11: textIn(card(page, "B1-1")).includes("Asha Patil"),
      };
    """)

    assert out["touched"] == {"emptied": 0, "reinserted": 0}
    assert out["codes"] == ["B1-3", "B1-1", "B1-2"]
    assert out["same"] is True and out["zoomedStill"] is True
    assert out["b11"] is True


# ------------------------------------------- sorting the scan folder: when
@needs_node
def test_page_scan_folder_sorts_scans_read_while_it_waited_for_reconnect(
    client: FlaskClient, tmp_path: Path
) -> None:
    """Scans read while the folder waited for Reconnect are sorted once it is reconnected, with nothing else to come."""
    out = _run_page(client, tmp_path, r"""
      const idb = makeIndexedDB();
      const folder = makeFolder("OCR Scans", { move: true });
      const clock1 = makeClock();
      const first = await openScannerPage({ clock: clock1, idb, folder });
      await answerBatches(first, EMPTY);
      await connectFolder(first);
      folder.add("Scan 001.jpg");
      folder.add("Scan 002.jpg");
      await clock1.advance(4000);
      await answerScan(first, ["p1"]);
      await answerScan(first, ["p2"]);
      // The tab is closed before OCR reads them; the next day Chrome asks
      // for the folder again.
      folder.permission = "prompt";
      const clock2 = makeClock();
      const second = await openScannerPage({ clock: clock2, idb, folder });
      await answerBatches(second, batches([[1, [doc("B1-1", [
        pg("p1", { code: "B1-1", source_name: "Scan 001.jpg" }), pg("p2", { code: "B1-1", source_name: "Scan 002.jpg" }),
      ])]]]));
      await clock2.advance(4000);
      const before = folder.names();
      second.byId["scan-folder-connect"].fire("click");
      await settle();
      await clock2.advance(60000);
      return { before, top: folder.names(), folders: folder.folders(), b1: folder.folders().includes("B1") ? folder.sub("B1").names() : [] };
    """)

    assert out["before"] == ["Scan 001.jpg", "Scan 002.jpg"]
    assert out["top"] == []
    assert out["b1"] == ["Scan 001.jpg", "Scan 002.jpg"]


@needs_node
def test_page_scan_folder_sorts_a_scan_read_while_an_earlier_move_was_under_way(
    client: FlaskClient, tmp_path: Path
) -> None:
    """An answer that lands while files are still being moved is sorted as soon as that move is done."""
    out = _run_page(client, tmp_path, r"""
      const clock = makeClock();
      const folder = makeFolder("OCR Scans", { move: true });
      const page = await openScannerPage({ clock, idb: makeIndexedDB(), folder });
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      folder.add("Scan 002.jpg");
      await clock.advance(4000);
      await answerScan(page, ["p1"]);
      await answerScan(page, ["p2"]);
      // Making a batch folder takes a while - a slow network drive.
      let open;
      const gate = new Promise((resolve) => { open = resolve; });
      const getDir = folder.getDirectoryHandle;
      folder.getDirectoryHandle = async (...args) => { await gate; return getDir(...args); };
      const p1 = pg("p1", { code: "B1-1", source_name: "Scan 001.jpg" });
      await clock.advance(3000);
      await answerBatches(page, batches([[1, [doc("B1-1", [p1])]]], [pg("p2", { source_name: "Scan 002.jpg" })]));
      // The person types Scan 002's code while Scan 001 is still being moved.
      page.byId["ocr-refresh"].fire("click");
      await answerBatches(page, batches([[2, [doc("B2-1", [pg("p2", { code: "B2-1", source_name: "Scan 002.jpg" })])]], [1, [doc("B1-1", [p1])]]]));
      open();
      await clock.advance(60000);
      return { top: folder.names(), folders: folder.folders(), b2: folder.folders().includes("B2") ? folder.sub("B2").names() : [] };
    """)

    assert out["top"] == []
    assert out["folders"] == ["B1", "B2"]
    assert out["b2"] == ["Scan 002.jpg"]


@needs_node
@pytest.mark.parametrize("how", ["move", "copy"])
def test_page_scan_folder_waits_for_a_checked_code_and_follows_a_corrected_one(
    client: FlaskClient, tmp_path: Path, how: str
) -> None:
    """A code that still needs a person is not sorted on; a corrected code moves the scan to its new batch."""
    out = _run_page(client, tmp_path, rf"""
      const clock = makeClock();
      const folder = makeFolder("OCR Scans", {{ move: {"true" if how == "move" else "false"} }});
      const page = await openScannerPage({{ clock, idb: makeIndexedDB(), folder }});
      await answerBatches(page, EMPTY);
      await connectFolder(page);
      folder.add("Scan 001.jpg");
      folder.add("Scan 002.jpg");
      await clock.advance(4000);
      await answerScan(page, ["p1"]);
      await answerScan(page, ["p2"]);
      await clock.advance(3000);
      // A handwritten B7-1 read as B1-1, with no Student ID read.
      await answerBatches(page, batches([
        [2, [doc("B2-1", [pg("p2", {{ code: "B2-1", source_name: "Scan 002.jpg" }})])]],
        [1, [doc("B1-1", [pg("p1", {{ code: "B1-1", source_name: "Scan 001.jpg" }})], {{ status: "needs_attention" }})]],
      ]));
      await clock.advance(4000);
      const misread = {{ top: folder.names(), folders: folder.folders() }};
      const c1 = card(page, "B1-1");
      fieldIn(c1, "code-p1").value = "B7-1";
      buttonIn(c1, "Set code").fire("click");
      await answer(page.calls, "/api/pages/p1/code", {{ ok: true }});
      // Meanwhile B2-1's page turns out to be B3-1, put right on another card.
      await answerBatches(page, batches([
        [7, [doc("B7-1", [pg("p1", {{ code: "B7-1", source_name: "Scan 001.jpg" }})], {{ status: "needs_attention" }})]],
        [3, [doc("B3-1", [pg("p2", {{ code: "B3-1", source_name: "Scan 002.jpg" }})])]],
      ]));
      await clock.advance(4000);
      const c7 = card(page, "B7-1");
      fieldIn(c7, "id").value = "MOHSHA011021";
      buttonIn(c7, "Confirm and upload").fire("click");
      await answer(page.calls, "/api/documents/k-B7-1/confirm", {{ document: {{ status: "uploaded", drive_link: "https://drive.google.com/x" }} }});
      await clock.advance(4000);
      const sub = (name) => folder.folders().includes(name) ? folder.sub(name).names() : null;
      return {{ misread, top: folder.names(), folders: folder.folders(), b2: sub("B2"), b3: sub("B3"), b7: sub("B7"), error: page.byId["scan-folder-error"].textContent }};
    """)

    # Ready: sorted at once. Needs attention: left where it is.
    assert out["misread"] == {"top": ["Scan 001.jpg"], "folders": ["B2"]}
    assert out["top"] == []
    assert out["folders"] == ["B2", "B3", "B7"]
    assert (out["b2"], out["b3"], out["b7"]) == ([], ["Scan 002.jpg"], ["Scan 001.jpg"])
    assert out["error"] == ""


# ------------------------------------------- #10: the Drive inbox's Stop
@needs_node
def test_page_inbox_stop_says_what_the_watcher_does_with_waiting_pages(
    client: FlaskClient, tmp_path: Path
) -> None:
    """#10: the waiting pages are read for this school - not left unread and unattributed."""
    from server.blueprints.inbox_routes import STOP_PENDING_MESSAGE

    out = _run_page(client, tmp_path, rf"""
      const asked = [];
      const page = loadPage({{
        local: {{ "ocr.school": "School A", "ocr.location": "Pune" }},
        confirm: (text) => {{ asked.push(text); return false; }},
      }});
      await answerSchools(page);
      await answer(page.calls, "/api/inbox/status", {{
        configured: true, inbox: {{ found: true }}, pending: 2, watcher: {{ alive: true }},
        batch: {{ open: true, id: 7, school: "School A", location: "Pune", pages_done: 2 }},
      }});
      for (const message of [{json.dumps(STOP_PENDING_MESSAGE)}, undefined]) {{
        page.byId["inbox-stop"].fire("click");
        await answer(page.calls, "/api/inbox/stop", {{ closed: false, reason: "pending", pending: 2, message }}, 409);
        await tick(5);
      }}
      return {{ asked, forced: page.calls.filter((c) => c.url.includes("/api/inbox/stop")).length }};
    """)

    with_message, without = out["asked"]
    assert STOP_PENDING_MESSAGE in with_message
    for text in (with_message, without):
        assert text.startswith("2 page(s) are still in your Drive folder")
        assert "still read for this school" in text
        assert "not attributed to a school" not in text
    # Cancel keeps the batch open: nothing forced it shut.
    assert out["forced"] == 0


def test_page_has_no_db_debug_block(client: FlaskClient) -> None:
    html = client.get("/").get_data(as_text=True)

    assert "db_debug" not in html
    assert "DB debug" not in html
    assert "db-debug" not in html
