"""Development server entrypoint. For production, point a WSGI server at run:app."""

import os

from server import create_app
from server.services.debug_log import LOG_FILE, debug_log

app = create_app()

if __name__ == "__main__":
    # use_reloader=False so debug prints stay in THIS terminal (not a hidden child)
    debug_log(f"Flask starting — debug log file: {LOG_FILE}")
    # PORT=5001 on a Mac, where AirPlay Receiver already holds 5000.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True, use_reloader=False)
