"""Application factory for the Student ID OCR Flask app."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit

from flask import Flask, Request, current_app, has_request_context, jsonify, request
from flask.sessions import SecureCookieSessionInterface
from flask.typing import ResponseReturnValue
from werkzeug.exceptions import UnsupportedMediaType
from werkzeug.middleware.proxy_fix import ProxyFix

from server.blueprints.auth_routes import auth_bp
from server.blueprints.batch_routes import batch_bp
from server.blueprints.history_routes import history_bp
from server.blueprints.inbox_routes import inbox_bp
from server.blueprints.ocr_routes import ocr_bp
from server.config import (
    BASE_DIR,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_OAUTH_SCOPES,
    MAX_CONTENT_LENGTH,
    SECRET_KEY,
    SESSION_COOKIE_SECURE,
    UPLOAD_FOLDER,
)
from server.services import ocr_worker, warmup
from server.services.auth import current_user
from server.services.debug_log import debug_log
from server.services.oauth import oauth
from server.services.sso import consume_sso_token

CLIENT_DIR = BASE_DIR / "client"

# Methods that change something. Only these are checked for cross-site use:
# a GET here never writes, and the sign-in hand-offs (Google's callback,
# Tether's ?token=) arrive as top-level GETs from another site by design.
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Renamed from Flask's default "session": tether.akanksha.org hosts other
# apps too, and one of them using "session" at Path=/ would overwrite ours.
SESSION_COOKIE_NAME = "ocr_session"

_DEFAULT_PORTS = {"http": 80, "https": 443}


class JsonBodyRequest(Request):
    """A request whose JSON body must arrive labelled as JSON.

    The JSON routes read their body with get_json(silent=True) or {}, so a
    text/plain or form-encoded body used to become {} and the route ran
    anyway - and those are exactly the bodies a plain HTML form on another
    site can post. A page script sends application/json, which no form can,
    and which another site's script cannot send here without a CORS
    preflight this app never answers. Upload routes read multipart files,
    never get_json, so this leaves them alone.
    """

    def get_json(self, force: bool = False, silent: bool = False, cache: bool = True) -> Any:
        """Parse the body as JSON, refusing a non-JSON body on a write with 415.

        Args:
            force: Parse even without a JSON Content-Type (reads only).
            silent: Return None instead of raising when the body is not
                valid JSON.
            cache: Keep the parsed body for later calls.

        Returns:
            The parsed body, or None when silent and it could not be parsed.

        Raises:
            UnsupportedMediaType: A POST/PUT/PATCH/DELETE without
                Content-Type application/json, whatever `silent` says.
        """
        if self.method in WRITE_METHODS and self.mimetype != "application/json":
            raise UnsupportedMediaType("Send this request as JSON.")
        return super().get_json(force=force, silent=silent, cache=cache)


class MountScopedSessionInterface(SecureCookieSessionInterface):
    """The signed session cookie, sent only to this app's own paths.

    Production serves the app at tether.akanksha.org/ocr, alongside other
    apps on the same host. Flask's default cookie goes out at Path=/, so
    every one of them - and their access logs - received it, Google token
    and all. The path here is the mount point (X-Forwarded-Prefix, through
    ProxyFix); at a host root that is just "/".
    """

    def get_cookie_path(self, app: Flask) -> str:
        """Path= for the cookie: SESSION_COOKIE_PATH if set, else the mount point."""
        configured = app.config.get("SESSION_COOKIE_PATH")
        if configured:
            return str(configured)
        if has_request_context():
            return request.script_root or "/"
        return super().get_cookie_path(app)

    def get_cookie_secure(self, app: Flask) -> bool:
        """Secure whenever the request came over https, or when configured to be.

        Following the request rather than always setting it keeps sign-in
        working on a local http run, where a Secure cookie would never be
        sent back.
        """
        if app.config.get("SESSION_COOKIE_SECURE"):
            return True
        return has_request_context() and request.is_secure


def _origin_key(url: str) -> tuple[str, str, int | None] | None:
    """(scheme, host, port) of an origin, spelt the way a browser compares them.

    Lowercased, with the scheme's default port left out, so that
    "https://Tether.akanksha.org:443" and "https://tether.akanksha.org" are
    the same origin. None for anything that is not an http(s) origin,
    including the "null" a sandboxed or privacy-redirected page sends.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    if port == _DEFAULT_PORTS[scheme]:
        port = None
    return scheme, parts.hostname.lower(), port


def refuse_cross_site_writes() -> ResponseReturnValue | None:
    """Turn away a write that another site's page sent, before any route runs.

    Two signals, either of which is enough: the browser's own Sec-Fetch-Site
    saying the request came from another site, or an Origin that is not this
    app's. Origin is what catches a sibling app on *.akanksha.org, which the
    browser counts as same-site. This app's origin is taken from the request
    after ProxyFix, so behind nginx it is https://tether.akanksha.org, the
    same address the Google sign-in callback is built from.

    When the browser itself says same-origin, Origin is not compared at all.
    The browser has already matched it against the page, and our own view
    of this app's origin is only as good as the proxy's forwarded headers:
    were Host or X-Forwarded-Proto ever lost, or the page served with
    Referrer-Policy: no-referrer (which makes Origin "null" even on its own
    POSTs), every write would be refused. Nothing a script on another site
    can do makes its request same-origin, so this lets no attack through.

    Returns:
        A 403 JSON response for a refused write, otherwise None.
    """
    if request.method not in WRITE_METHODS:
        return None

    fetch_site = (request.headers.get("Sec-Fetch-Site") or "").strip().lower()
    origin = request.headers.get("Origin")
    refused = fetch_site == "cross-site"
    if not refused and origin is not None and fetch_site != "same-origin":
        own = _origin_key(f"{request.scheme}://{request.host}")
        refused = own is None or _origin_key(origin) != own

    if not refused:
        return None
    debug_log(
        f"[SECURITY] refused cross-site {request.method} {request.path} "
        f"origin={origin!r} sec-fetch-site={fetch_site!r}"
    )
    return jsonify({"error": "That request came from another site, so it was refused."}), 403


def warm_models_in_background() -> None:
    """Start loading the OCR models in the background (services/warmup.py).

    A wrapper rather than the function imported by name: an imported copy
    is fixed at import time, so the test suite's patch of
    warmup.warm_models_in_background never reached create_app, and every
    app a test built loaded RapidOCR and a Vision client for real. This
    looks it up on each call, so a patch of either name takes effect.
    """
    warmup.warm_models_in_background()


def start_queue_worker_if_wanted() -> None:
    """Start the in-process OCR reader for the scan queue, when this app should run one.

    Checked on each request rather than only in create_app: run.py builds
    the app first and only turns debug on as it starts serving, and
    debug is what says "development" when OCR_WORKER_IN_PROCESS is unset.
    After the first start this is a single attribute check.
    """
    if (
        not ocr_worker.in_process_worker_running()
        and ocr_worker.should_run_in_process(current_app.debug)
        and ocr_worker.start_in_process_worker()
    ):
        debug_log("[WORKER] reading the scan queue in this process")


def _json_required(_error: UnsupportedMediaType) -> ResponseReturnValue:
    return jsonify({"error": "Send this request as JSON."}), 415


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(CLIENT_DIR / "templates"),
        static_folder=str(CLIENT_DIR / "static"),
    )
    # Trust X-Forwarded-* from ngrok / reverse proxies so OAuth redirect_uri
    # uses https://….ngrok-free.app instead of http://127.0.0.1:5000
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    app.secret_key = SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

    # The session cookie carries the user's Google token, so it is kept to
    # this app: its own name, only this app's paths (MountScopedSessionInterface),
    # never readable by page script, Secure over https, and not sent with a
    # cross-site POST. The new name signs everyone out once on deploy - the
    # old "session" cookie is simply ignored, and is left alone rather than
    # deleted because another app on the host may use that name too.
    app.config["SESSION_COOKIE_NAME"] = SESSION_COOKIE_NAME
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = SESSION_COOKIE_SECURE
    app.session_interface = MountScopedSessionInterface()
    app.request_class = JsonBodyRequest
    app.register_error_handler(UnsupportedMediaType, _json_required)

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    oauth.init_app(app)
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": GOOGLE_OAUTH_SCOPES},
    )

    # First of all, so a cross-site write is refused before anything -
    # sign-in included - has looked at it.
    app.before_request(refuse_cross_site_writes)
    # Runs ahead of every blueprint login gate, so a Tether hand-off works on
    # whichever page the token is aimed at.
    app.before_request(consume_sso_token)
    app.before_request(start_queue_worker_if_wanted)

    app.register_blueprint(auth_bp)
    app.register_blueprint(ocr_bp)
    app.register_blueprint(batch_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(inbox_bp)

    @app.context_processor
    def inject_current_user():
        return {"current_user": current_user()}

    # Under gunicorn each worker warms its own copy; that is the point.
    warm_models_in_background()

    return app
