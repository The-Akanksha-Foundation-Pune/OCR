"""Session-based login gate for Google OAuth2 sign-in."""

from __future__ import annotations

from typing import Any

from flask import jsonify, redirect, request, session, url_for

from server.config import LOCAL_OPEN_ACCESS

# Shown in the sidebar when Google sign-in is skipped for local development.
_LOCAL_USER = {
    "email": "local@localhost",
    "name": "Local",
    "picture": "",
}


def current_user() -> dict[str, Any] | None:
    """The signed-in user's {email, name, picture}, or None if not signed in."""
    user = session.get("user")
    if user is not None:
        return user
    if LOCAL_OPEN_ACCESS:
        return _LOCAL_USER
    return None


def require_login():
    """Blueprint-level gate — wire up via `some_bp.before_request(require_login)`.

    Page routes get redirected to /login (with a `next` back-link). API routes
    (path starting with /api/) get a 401 JSON body instead, since the page's
    JS expects JSON, not an HTML redirect, from a fetch() call.
    """
    if current_user() is not None:
        return None

    if request.path.startswith("/api/"):
        return jsonify({"error": "Please sign in with Google to continue."}), 401

    return redirect(url_for("auth.login", next=request.path))
