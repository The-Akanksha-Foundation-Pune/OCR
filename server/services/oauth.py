"""Google OAuth2 client (Authlib). Registered against the app in create_app()."""

from __future__ import annotations

from authlib.integrations.flask_client import OAuth

oauth = OAuth()
