"""Google OAuth2 sign-in, restricted to the configured email domain."""

from __future__ import annotations

import time
from urllib.parse import urlsplit

from flask import Blueprint, redirect, render_template, request, session, url_for
from werkzeug.wrappers.response import Response

from server.config import ALLOWED_EMAIL_DOMAIN, LOCAL_OPEN_ACCESS
from server.services.access_control import is_authorized_staff
from server.services.auth import current_user
from server.services.debug_log import debug_log
from server.services.oauth import oauth

auth_bp = Blueprint("auth", __name__)


def _safe_next_path(value: str | None) -> str | None:
    """Only ever redirect to a path within this app — never an external URL.

    Checked the way a browser will read it, not just as written. A browser
    (and Werkzeug, writing the Location header) drops tab, CR and LF from a
    URL, and reads a backslash as a slash - so "/\\t/evil.com" or "/\\evil.com"
    pass a plain "starts with one slash" test and still land on "//evil.com",
    another site. Anything with a control character or a backslash is
    refused outright rather than cleaned up: no real page of this app has one.
    """
    if not value:
        return None
    if "\\" in value or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return None
    if not value.startswith("/") or value.startswith("//"):
        return None
    # Belt and braces: whatever is left must parse as a bare path on this host.
    parts = urlsplit(value)
    if parts.scheme or parts.netloc or not parts.path.startswith("/") or parts.path.startswith("//"):
        return None
    return value


@auth_bp.get("/login")
def login():
    if current_user() is not None:
        return redirect(url_for("ocr.index"))
    return render_template(
        "login.html",
        error=request.args.get("error"),
        allowed_domain=ALLOWED_EMAIL_DOMAIN,
        next_path=_safe_next_path(request.args.get("next")) or "",
    )


@auth_bp.get("/login/google")
def google_login():
    if LOCAL_OPEN_ACCESS:
        return redirect(_safe_next_path(request.args.get("next")) or url_for("ocr.index"))
    session["post_login_next"] = _safe_next_path(request.args.get("next"))
    redirect_uri = url_for("auth.google_callback", _external=True)
    # access_type=offline + prompt=consent so we get a refresh token for Drive uploads
    return oauth.google.authorize_redirect(
        redirect_uri,
        access_type="offline",
        prompt="consent",
    )


@auth_bp.get("/login/google/callback")
def google_callback() -> Response:
    try:
        token = oauth.google.authorize_access_token()
    except Exception as exc:  # noqa: BLE001 — any OAuth failure lands on /login
        debug_log(f"[AUTH] Google OAuth callback failed: {exc}")
        return redirect(url_for("auth.login", error="oauth_failed"))

    userinfo = token.get("userinfo") or {}
    email = (userinfo.get("email") or "").strip().lower()

    if not email:
        debug_log("[AUTH] Google token had no email claim")
        return redirect(url_for("auth.login", error="oauth_failed"))

    if not userinfo.get("email_verified", True):
        debug_log(f"[AUTH] rejected unverified email: {email!r}")
        return redirect(url_for("auth.login", error="unverified"))

    email_domain = email.rsplit("@", 1)[-1] if "@" in email else ""
    if ALLOWED_EMAIL_DOMAIN and email_domain != ALLOWED_EMAIL_DOMAIN:
        debug_log(f"[AUTH] rejected domain mismatch: {email!r}")
        return redirect(url_for("auth.login", error="domain"))

    try:
        authorized = is_authorized_staff(email)
    except Exception as exc:  # noqa: BLE001 — fail closed, but don't crash the app
        debug_log(f"[AUTH] access check failed for {email!r}: {exc}")
        return redirect(url_for("auth.login", error="access_check_failed"))

    if not authorized:
        debug_log(f"[AUTH] rejected — not Developer/School Administration: {email!r}")
        return redirect(url_for("auth.login", error="not_authorized"))

    # Checked again on the way out: the session may hold a value stored
    # before this check was as strict as it is now.
    next_path = _safe_next_path(session.pop("post_login_next", None))
    session.clear()
    session.permanent = True
    session["user"] = {
        "email": email,
        "name": userinfo.get("name") or email,
        "picture": userinfo.get("picture") or "",
    }
    # Keep Google tokens so scanned files can be uploaded to Drive
    expires_in = int(token.get("expires_in") or 3600)
    session["google_token"] = {
        "access_token": token.get("access_token"),
        "refresh_token": token.get("refresh_token"),
        "expires_at": time.time() + expires_in,
        "token_type": token.get("token_type") or "Bearer",
    }
    debug_log(f"[AUTH] signed in: {email!r}")

    if not next_path:
        return redirect(url_for("ocr.index"))
    # next is a path inside the app (the login gate stores request.path), so
    # under the /ocr mount it needs the mount put back in front: "/" alone
    # would land on the host's own home page, another app. The leading
    # slashes are collapsed, as in sso._same_page_path, so whatever the
    # prefix the result can only be a path on this host.
    return redirect("/" + (request.script_root + next_path).lstrip("/"))


@auth_bp.get("/logout")
def logout():
    user = current_user()
    session.clear()
    if user:
        debug_log(f"[AUTH] signed out: {user.get('email')!r}")
    return redirect(url_for("auth.login"))
