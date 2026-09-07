"""Application factory for the Student ID OCR Flask app."""

from datetime import timedelta

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from server.blueprints.auth_routes import auth_bp
from server.blueprints.history_routes import history_bp
from server.blueprints.ocr_routes import ocr_bp
from server.config import (
    BASE_DIR,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_OAUTH_SCOPES,
    MAX_CONTENT_LENGTH,
    SECRET_KEY,
    UPLOAD_FOLDER,
)
from server.services.auth import current_user
from server.services.oauth import oauth
from server.services.sso import consume_sso_token

CLIENT_DIR = BASE_DIR / "client"


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

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    oauth.init_app(app)
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": GOOGLE_OAUTH_SCOPES},
    )

    # Runs ahead of every blueprint login gate, so a Tether hand-off works on
    # whichever page the token is aimed at.
    app.before_request(consume_sso_token)

    app.register_blueprint(auth_bp)
    app.register_blueprint(ocr_bp)
    app.register_blueprint(history_bp)

    @app.context_processor
    def inject_current_user():
        return {"current_user": current_user()}

    return app
