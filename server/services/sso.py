"""Sign-in by way of a Tether SSO token.

Tether hands off to its partner portals - Kaleidoscope, Timetable, the IT
Visit Form - by signing a short-lived JWT and appending it to the destination
URL as `?token=`. This module verifies that token with the shared secret and
starts an ordinary session from it, so somebody already signed into Tether is
not asked to sign in a second time here.

The token proves *who* the visitor is. It is not by itself permission to use
this app: the same employees-table check the Google path runs is applied
afterwards, so a valid token for someone outside the allowlist still gets in
nowhere.
"""

from __future__ import annotations

from urllib.parse import urlencode

import jwt
from flask import redirect, request, session

from server.config import ALLOWED_EMAIL_DOMAIN, INTEGRATION_JWT_SECRET
from server.services.access_control import is_authorized_staff
from server.services.debug_log import debug_log

# Pin the algorithm. Accepting whatever the token names would let an attacker
# present `alg: none`, or swap HMAC for RSA and sign with the public key.
_ALGORITHMS = ["HS256"]

# Tether signs with expiresIn: "10m"; allow a little clock drift between hosts.
_LEEWAY_SECONDS = 30


def _redirect_without_token():
    """Bounce to the same page with `token` removed from the query string.

    Worth the extra redirect: it keeps the token out of the address bar, the
    browser history, and the Referer header sent to any third party.
    """
    remaining = [(k, v) for k, v in request.args.items(multi=True) if k != "token"]
    query = urlencode(remaining)
    # script_root, not just path. Mounted under a prefix (nginx sends
    # X-Forwarded-Prefix /ocr), request.path is the path *inside* the app, so
    # redirecting to it alone sends the browser out of the app to the host root.
    target = (request.script_root or "") + request.path
    return redirect(target + (f"?{query}" if query else ""))


def _claims_from(token: str) -> dict | None:
    if not INTEGRATION_JWT_SECRET:
        debug_log("[SSO] token presented but INTEGRATION_JWT_SECRET is not set")
        return None

    try:
        # decode() verifies the signature and the exp claim; require exp so a
        # token minted without one can never be replayed forever.
        claims = jwt.decode(
            token,
            INTEGRATION_JWT_SECRET,
            algorithms=_ALGORITHMS,
            leeway=_LEEWAY_SECONDS,
            options={"require": ["exp"]},
        )
    except jwt.InvalidTokenError as exc:
        # Bad signature, expired, malformed - all equally "not signed in".
        debug_log(f"[SSO] token rejected: {type(exc).__name__}: {exc}")
        return None

    return dict(claims)


def consume_sso_token():
    """Sign the visitor in from `?token=`, or return None to leave them alone.

    Registered as an app-wide before_request, so it runs ahead of the blueprint
    login gates and works on whichever page Tether points at.
    """
    if request.method != "GET":
        return None

    token = request.args.get("token")
    if not token:
        return None

    claims = _claims_from(token)
    if claims is None:
        return _redirect_without_token()

    email = str(claims.get("email") or "").strip().lower()
    if not email:
        debug_log("[SSO] token carried no email claim")
        return _redirect_without_token()

    domain = email.rsplit("@", 1)[-1] if "@" in email else ""
    if ALLOWED_EMAIL_DOMAIN and domain != ALLOWED_EMAIL_DOMAIN:
        debug_log(f"[SSO] rejected domain mismatch: {email!r}")
        return _redirect_without_token()

    try:
        authorized = is_authorized_staff(email)
    except Exception as exc:  # noqa: BLE001 - fail closed, but keep the app up
        debug_log(f"[SSO] access check failed for {email!r}: {exc}")
        return _redirect_without_token()

    if not authorized:
        debug_log(f"[SSO] rejected - not allowlisted in employees: {email!r}")
        return _redirect_without_token()

    already = (session.get("user") or {}).get("email")
    session.clear()
    session.permanent = True
    session["user"] = {
        "email": email,
        "name": claims.get("name") or email,
        "picture": claims.get("picture") or "",
    }
    # No Google token comes with an SSO hand-off, so Drive uploads have no
    # credentials. drive_store degrades to a no-op; this flag lets the UI say
    # so rather than looking broken.
    session["signed_in_via"] = "tether"

    if already and already != email:
        debug_log(f"[SSO] switched session from {already!r} to {email!r}")
    debug_log(f"[SSO] signed in via Tether: {email!r}")

    return _redirect_without_token()
