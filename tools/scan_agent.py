"""Local scan agent - lets the browser drive the scanner on this PC.

A browser cannot reach a USB scanner, so when the app runs on a server there
has to be something on the scanning machine that can. This is that something:
a small HTTP server on 127.0.0.1 which runs the scan script for this machine -
tools/scan.ps1 on Windows, tools/scan_mac.py on a Mac (which needs NAPS2:
brew install --cask naps2) - and hands the pages back to the page that asked.

It deliberately holds no credentials and never contacts the server. The browser
uploads the pages itself, using the session the person already signed in with.
So a copy of this agent taken off a laptop is worth nothing on its own - it can
only work a scanner, and only for a browser on the same machine.

Run it:

    python tools/scan_agent.py        (Windows)
    python3 tools/scan_agent.py       (Mac)

Then from the app, Scan now calls it. To check it by hand, open
https://tether.akanksha.org/ocr/ and run this in the browser console:

    await (await fetch("http://127.0.0.1:8765/health")).json()
"""

from __future__ import annotations

import base64
import errno
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = 8765
TOOLS = Path(__file__).resolve().parent
# Both scripts honour the same contract: page-NNN.jpg files in the folder they
# are given, then done.txt or failed.txt. Everything below is shared.
SCAN_SCRIPTS = {"win32": TOOLS / "scan.ps1", "darwin": TOOLS / "scan_mac.py"}
SCAN_SCRIPT = SCAN_SCRIPTS.get(sys.platform)
DEFAULT_SCANNER = "SP-1130N"

# Only these pages may talk to the agent. A browser will refuse anything else,
# but say it explicitly rather than leaving it to the browser to enforce.
ALLOWED_ORIGINS = {
    "https://tether.akanksha.org",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    # Local runs on a Mac, where AirPlay Receiver holds port 5000.
    "http://localhost:5001",
    "http://127.0.0.1:5001",
}

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _scan_command(folder: Path, scanner: str) -> list[str]:
    if sys.platform == "darwin":
        return [
            sys.executable,
            str(SCAN_SCRIPT),
            "--stream-dir",
            str(folder),
            "--scanner",
            scanner,
        ]
    return [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCAN_SCRIPT),
        "-StreamDir",
        str(folder),
        "-Scanner",
        scanner,
    ]


def _start_scan(scanner: str) -> dict:
    """Kick off the scan script writing into a fresh folder, and return the job."""
    folder = Path(tempfile.mkdtemp(prefix="scan-agent-"))
    command = _scan_command(folder, scanner)
    log = open(folder / "scan.log", "w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job = {"id": uuid.uuid4().hex, "folder": folder, "process": process, "log": log}
    with _jobs_lock:
        _jobs[job["id"]] = job
    return job


def _pages_after(job: dict, after: int) -> list[dict]:
    """Pages captured since page `after`, as base64 JPEGs.

    Both scan scripts write each page under another name and rename it once
    complete, so anything matching page-*.jpg is safe to read - no
    half-written images.
    """
    out = []
    for path in sorted(job["folder"].glob("page-*.jpg")):
        number = int(path.stem.split("-")[1])
        if number <= after:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue  # still being renamed; it will appear on the next poll
        out.append(
            {
                "page": number,
                "jpeg_base64": base64.b64encode(data).decode("ascii"),
            }
        )
    return out


def _job_finished(job: dict) -> tuple[bool, str | None]:
    folder = job["folder"]
    if (folder / "done.txt").exists():
        return True, None
    failed = folder / "failed.txt"
    if failed.exists():
        return True, failed.read_text(encoding="utf-8-sig", errors="replace").strip()
    if job["process"].poll() is not None:
        return True, "The scanner stopped unexpectedly. " + _last_words(job)
    return False, None


def _last_words(job: dict) -> str:
    """The tail of the scan script's output, for when it dies without a marker."""
    try:
        job["log"].flush()
        text = (job["folder"] / "scan.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # PowerShell prints the message first, then several lines of
    # CategoryInfo / FullyQualifiedErrorId noise - keep the message.
    for ln in reversed(lines):
        noise = (
            ln.startswith("+")
            or ln.startswith("At ")
            or "CategoryInfo" in ln
            or "FullyQualifiedErrorId" in ln
        )
        if not noise:
            return ln
    return ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ---- plumbing -------------------------------------------------------
    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        # Chrome's Private Network Access check: a public page reaching a
        # loopback address must be told explicitly that this is intended.
        if self.headers.get("Access-Control-Request-Private-Network"):
            self.send_header("Access-Control-Allow-Private-Network", "true")

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self._cors()
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}", flush=True)

    # ---- routes ---------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        query = parse_qs(route.query)

        if route.path == "/health":
            return self._json(
                {
                    "ok": True,
                    "agent": "akanksha-scan-agent",
                    "version": 1,
                    "platform": sys.platform,
                    "scan_script": bool(SCAN_SCRIPT and SCAN_SCRIPT.exists()),
                }
            )

        if route.path == "/pages":
            job_id = (query.get("job") or [""])[0]
            after = int((query.get("after") or ["0"])[0])
            with _jobs_lock:
                job = _jobs.get(job_id)
            if job is None:
                return self._json({"error": "No such scan."}, 404)
            finished, error = _job_finished(job)
            return self._json(
                {
                    "pages": _pages_after(job, after),
                    "finished": finished,
                    "error": error,
                }
            )

        return self._json({"error": "Not found."}, 404)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path)

        if route.path == "/scan":
            if SCAN_SCRIPT is None:
                return self._json(
                    {"error": "The scan agent only runs on Windows and macOS."}, 503
                )
            if not SCAN_SCRIPT.exists():
                return self._json(
                    {"error": f"{SCAN_SCRIPT.name} is missing: {SCAN_SCRIPT}"}, 500
                )
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or "{}") if length else {}
            scanner = (body.get("scanner") or DEFAULT_SCANNER).strip()
            try:
                job = _start_scan(scanner)
            except OSError as exc:
                return self._json({"error": f"Could not start: {exc}"}, 502)
            print(f"  scan started: job={job['id']} scanner={scanner}", flush=True)
            return self._json({"job": job["id"]})

        if route.path == "/finish":
            # The browser has everything it needs; drop the temporary images.
            job_id = (parse_qs(route.query).get("job") or [""])[0]
            with _jobs_lock:
                job = _jobs.pop(job_id, None)
            if job is not None:
                if job["process"].poll() is None:
                    job["process"].terminate()
                try:
                    job["log"].close()
                except OSError:
                    pass
                shutil.rmtree(job["folder"], ignore_errors=True)
            return self._json({"ok": True})

        return self._json({"error": "Not found."}, 404)


def main() -> None:
    # Bind before printing anything, so a second copy says why it cannot
    # start instead of announcing itself and then dying with a traceback.
    # Loopback only. Binding 0.0.0.0 would let the rest of the network drive
    # this person's scanner.
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        sys.exit(
            f"Port {PORT} is already in use - the scan agent is probably "
            "already running in another window. Use that one."
        )

    print("Akanksha scan agent", flush=True)
    print(f"  listening on  http://127.0.0.1:{PORT}", flush=True)
    if SCAN_SCRIPT is None:
        print(f"  scan script   none for {sys.platform} - scanning will be refused", flush=True)
    else:
        print(f"  scan script   {SCAN_SCRIPT}  ({'found' if SCAN_SCRIPT.exists() else 'MISSING'})", flush=True)
    print(f"  accepts       {', '.join(sorted(ALLOWED_ORIGINS))}", flush=True)
    print("\n  Leave this window open while scanning. Ctrl+C to stop.\n", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
        time.sleep(0)
