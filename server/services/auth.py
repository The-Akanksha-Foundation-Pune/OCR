"""Session-based login gate for Google OAuth2 sign-in."""

from __future__ import annotations

from typing import Any

from flask import jsonify, redirect, request, session, url_for


def current_user() -> dict[str, Any] | None:
    """The signed-in user's {email, name, picture}, or None if not signed in."""
    return session.get("user")


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
