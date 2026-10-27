"""Local scan helper - lets the browser drive a scanner on this computer.

A browser cannot reach a scanner plugged into someone's laptop, so when the
app runs on a server there has to be something on the scanning machine that
can. This is that something: a small HTTP server on 127.0.0.1 that finds the
scanners this computer can drive, runs a scan on the one the page picks, and
hands the pages back to the page that asked.

It works on Windows, macOS and Linux, through two scan engines:

  scan.ps1   Windows WIA scanners - built into Windows, nothing to install.
             Pages come back one by one while the stack is still feeding.
  NAPS2      everything else (https://www.naps2.com/download): TWAIN on
             Windows, Apple's Image Capture drivers on a Mac, SANE on Linux,
             and network (eSCL / AirScan) scanners everywhere. NAPS2 hands
             the stack over in one go once it is through.

scanner_devices.py finds the scanners and scan_naps2.py runs NAPS2. Keep all
four files - those two, this one and scan.ps1 - together in one folder.

It deliberately holds no credentials and never contacts the server. The
browser uploads the pages itself, using the session the person already
signed in with. So a copy of this helper taken off a laptop is worth nothing
on its own - it can only work a scanner, and only for a browser on the same
machine.

Run it:

    python tools/scan_agent.py        (Windows)
    python3 tools/scan_agent.py       (Mac, Linux)

To let another copy of the app use it as well, name its address:

    python3 tools/scan_agent.py --origin https://staging.example.org
    (or set SCAN_AGENT_ORIGINS to a comma-separated list)

A scan stuck for 15 minutes is stopped; SCAN_AGENT_TIMEOUT_SECONDS changes
that. NAPS2_PATH points the helper at a NAPS2 it cannot find by itself.

Then from the app, Check and Scan call it. To check it by hand, open the app
and run this in the browser console:

    await (await fetch("http://127.0.0.1:8765/health")).json()
    await (await fetch("http://127.0.0.1:8765/devices")).json()
"""

from __future__ import annotations

import argparse
import base64
import errno
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import scanner_devices

PORT = 8765
VERSION = 2
AGENT_NAME = "akanksha-scan-agent"
TOOLS = Path(__file__).resolve().parent
SCAN_PS1 = TOOLS / "scan.ps1"
SCAN_NAPS2 = TOOLS / "scan_naps2.py"

# The page in production before version 2 posts "{}" and expects the
# office's Fujitsu, so a scan without a device_id looks for it first.
OLD_PAGE_SCANNER = "SP-1130N"

SOURCES = ("feeder", "duplex", "flatbed")
DPIS = (150, 200, 300, 400, 600)
COLOURS = ("gray", "color")


def _timeout_seconds() -> float:
    try:
        seconds = float(os.environ.get("SCAN_AGENT_TIMEOUT_SECONDS") or 900)
    except ValueError:
        seconds = 900.0
    return seconds if seconds > 0 else 900.0


SCAN_TIMEOUT = _timeout_seconds()
# A device list older than this is checked again before a scan starts on it;
# a younger one is trusted, so Scan does not sit through another Check.
DEVICES_MAX_AGE_FOR_SCAN = 600
# NAPS2 hands over a whole stack at once. Send it in slices of about this
# many bytes per poll rather than as one response of a hundred pages.
PAGE_BYTES_PER_POLL = 24 * 1024 * 1024
# A finished scan nobody came back for (the tab was closed) is cleared away
# after this long.
ABANDONED_SECONDS = 3600

# Only these pages may talk to the helper. Any other Origin is turned away
# before a route runs: CORS alone only stops another site reading the
# answer, and a plain form post from it would still start a scan.
BUILT_IN_ORIGINS = {
    "https://tether.akanksha.org",
    "https://ocr.akanksha.org",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    # Local runs on a Mac, where AirPlay Receiver holds port 5000.
    "http://localhost:5001",
    "http://127.0.0.1:5001",
}
ALLOWED_ORIGINS: set[str] = set(BUILT_IN_ORIGINS)
# The names the helper answers to. A site that points its own domain at
# 127.0.0.1 (DNS rebinding) is same-origin as far as the browser knows, so
# the Host it sends is what gives it away.
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
# Every request body is read off the connection before it is answered; past
# this the request is refused and the connection closed instead.
MAX_BODY_BYTES = 64 * 1024

_PAGE_FILE = re.compile(r"^page-(\d+)\.jpg$")


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second copy bind the same port and
    # share the requests with the first. Without it the second copy fails
    # to start and says why; elsewhere it only skips the wait after a
    # restart, which is worth keeping.
    allow_reuse_address = os.name != "nt"


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class Refused(Exception):
    """A request the helper turns down, with the status and message to send."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# ---- scans -------------------------------------------------------------


def _running_job() -> dict | None:
    with _jobs_lock:
        for job in _jobs.values():
            if job["process"].poll() is None:
                return job
    return None


def _scan_options(body: dict) -> dict:
    """Check the POST /scan body and fill in the defaults."""
    source = body.get("source") or "feeder"
    if source not in SOURCES:
        raise Refused(400, 'source must be "feeder", "duplex" or "flatbed".')

    dpi = body.get("dpi")
    if dpi is None or dpi == "":
        dpi = 300
    elif isinstance(dpi, str) and dpi.strip().isdigit():
        dpi = int(dpi.strip())
    elif isinstance(dpi, float) and dpi.is_integer():
        dpi = int(dpi)
    if isinstance(dpi, bool) or dpi not in DPIS:
        raise Refused(400, "dpi must be one of " + ", ".join(str(d) for d in DPIS) + ".")

    colour = body.get("color") or "gray"
    if colour not in COLOURS:
        raise Refused(400, 'color must be "gray" or "color".')

    device_id = body.get("device_id")
    if device_id is not None and not isinstance(device_id, str):
        raise Refused(400, "device_id must be a string.")
    return {
        "device_id": (device_id or "").strip(),
        "source": source,
        "dpi": dpi,
        "color": colour,
    }


def _choose_device(device_id: str) -> dict:
    listing = scanner_devices.get_devices(max_age=DEVICES_MAX_AGE_FOR_SCAN)
    ready = listing["devices"]
    if device_id:
        device = next((d for d in ready if d["id"] == device_id), None)
        if device is None:
            raise Refused(404, "That scanner is no longer available. Press Check again.")
    else:
        # No device_id: the page from before version 2. The list is already
        # in driver preference order, so the first match is the best way to it.
        named = [d for d in ready if OLD_PAGE_SCANNER.lower() in d["name"].lower()]
        if named:
            device = named[0]
        elif len(ready) == 1:
            device = ready[0]
        else:
            raise Refused(409, "Choose a scanner first.")
    _refuse_twins(device, ready)
    return device


def _refuse_twins(device: dict, ready: list[dict]) -> None:
    """Turn down a scanner the engine cannot tell from its twin.

    Two scanners with one name under one driver get ids of their own, but
    NAPS2 is told which to use by name alone and would take the first one
    every time. scan.ps1 is also given WIA's own id for the scanner, so a
    WIA twin is fine as long as that id is known.
    """
    twins = [d for d in ready if d["driver"] == device["driver"] and d["name"] == device["name"]]
    if len(twins) < 2:
        return
    if device["engine"] == "scan.ps1" and scanner_devices.wia_device_id(device["id"]):
        return
    raise Refused(
        409,
        f"Two scanners share the name {device['name']}, so the helper cannot tell "
        "which one to use. Unplug the one you are not scanning with and press Check again.",
    )


def _scan_command(device: dict, folder: Path, options: dict) -> list[str]:
    """The engine's command line for this scan. Always a list - a device
    name never passes through a shell."""
    source, dpi, colour = options["source"], options["dpi"], options["color"]

    if device["engine"] == "scan.ps1":
        if sys.platform != "win32":
            raise Refused(503, "scan.ps1 only runs on Windows.")
        if not SCAN_PS1.exists():
            raise Refused(
                503,
                f"scan.ps1 is missing from {TOOLS}. Download the scan helper again.",
            )
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCAN_PS1),
            "-StreamDir",
            str(folder),
            "-Scanner",
            device["name"],
            "-Dpi",
            str(dpi),
            "-Mode",
            "Color" if colour == "color" else "Grayscale",
        ]
        # WIA's own id picks the scanner out from another with the same
        # name; -Scanner stays for when the id has changed since the check.
        wia_id = scanner_devices.wia_device_id(device["id"])
        if wia_id:
            command += ["-DeviceId", wia_id]
        if source == "duplex":
            command.append("-Duplex")
        elif source == "flatbed":
            command.append("-Flatbed")
        return command

    if device["engine"] == "naps2":
        naps2 = scanner_devices.find_naps2()
        if not naps2:
            raise Refused(
                503,
                "NAPS2 is not installed on this computer. "
                + scanner_devices.install_hint(),
            )
        if not SCAN_NAPS2.exists():
            raise Refused(
                503,
                f"scan_naps2.py is missing from {TOOLS}. Download the scan helper again.",
            )
        return [
            sys.executable,
            str(SCAN_NAPS2),
            "--stream-dir",
            str(folder),
            "--naps2",
            naps2,
            "--driver",
            device["driver"],
            "--device",
            device["name"],
            "--source",
            "glass" if source == "flatbed" else source,
            "--dpi",
            str(dpi),
            "--bitdepth",
            colour,
            "--timeout",
            f"{SCAN_TIMEOUT:g}",
        ]

    raise Refused(503, f"No scan engine on this computer can drive {device['name']}.")


def _start_scan(device: dict, options: dict) -> dict:
    """Start the engine writing into a fresh folder, and return the job.

    The engine gets a process group of its own, so stopping a scan reaches
    NAPS2 or PowerShell and everything they started, not just the first
    process.
    """
    folder = Path(tempfile.mkdtemp(prefix="scan-agent-"))
    try:
        command = _scan_command(device, folder, options)
    except Refused:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with _jobs_lock:
        # Checked again under the lock: two Scan presses at once must not
        # both get through.
        if any(job["process"].poll() is None for job in _jobs.values()):
            shutil.rmtree(folder, ignore_errors=True)
            raise Refused(409, "A scan is already running on this computer.")
        log = open(folder / "scan.log", "wb")
        try:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                **group,
            )
        except OSError as exc:
            log.close()
            shutil.rmtree(folder, ignore_errors=True)
            raise Refused(502, f"Could not start the scan: {exc}") from exc
        job = {
            "id": uuid.uuid4().hex,
            "folder": folder,
            "process": process,
            "log": log,
            "device": device,
            "started": time.monotonic(),
            "seen": time.monotonic(),
            "ended": None,
            "error": None,
        }
        _jobs[job["id"]] = job
    return job


def _stop(job: dict) -> None:
    """Kill a job's whole process tree if it is still going."""
    scanner_devices.kill_process_tree(job["process"], own_group=True)


def _discard(job: dict) -> None:
    _stop(job)
    try:
        job["log"].close()
    except OSError:
        pass
    # Windows can hold a file for a moment after the process that had it
    # open is killed.
    for _ in range(5):
        shutil.rmtree(job["folder"], ignore_errors=True)
        if not job["folder"].exists():
            return
        time.sleep(0.5)


def _enforce_timeout(job: dict) -> None:
    if job["error"] is None and job["process"].poll() is None:
        if time.monotonic() - job["started"] > SCAN_TIMEOUT:
            job["error"] = scanner_devices.timeout_message(SCAN_TIMEOUT)
            print(f"  scan timed out: job={job['id']}", flush=True)
            _stop(job)


def _job_state(job: dict) -> tuple[bool, str | None]:
    """(finished, error) for a job.

    Whether the process has exited is read before the marker files: an
    engine writes its marker and then exits, so a process seen gone with no
    marker really did die without saying why.
    """
    _enforce_timeout(job)
    if job["error"]:
        return True, job["error"]
    exited = job["process"].poll() is not None
    folder = job["folder"]
    failed = folder / "failed.txt"
    if failed.exists():
        try:
            text = failed.read_text(encoding="utf-8-sig", errors="replace").strip()
        except OSError:
            text = ""
        return True, text or "The scan failed."
    if (folder / "done.txt").exists():
        return True, None
    if exited:
        return True, ("The scanner stopped unexpectedly. " + _last_words(job)).strip()
    return False, None


def _pages_after(job: dict, after: int) -> tuple[list[dict], bool]:
    """Pages captured since page `after`, as base64 JPEGs, in page order.

    Both engines write each page under another name and rename it once
    complete, so a page-NNN.jpg is safe to read. Returns the pages and
    whether more are waiting beyond them. A page that cannot be read yet
    ends the batch there: the page asks for "after" the last one it got,
    so skipping one would lose it for good.
    """
    numbered = []
    for path in job["folder"].glob("page-*.jpg"):
        found = _PAGE_FILE.match(path.name)
        if found and int(found.group(1)) > after:
            numbered.append((int(found.group(1)), path))
    numbered.sort()
    out: list[dict] = []
    used = 0
    for number, path in numbered:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue
        except OSError:
            return out, True  # still being renamed; it comes on the next poll
        if out and used + len(data) > PAGE_BYTES_PER_POLL:
            return out, True
        out.append(
            {"page": number, "jpeg_base64": base64.b64encode(data).decode("ascii")}
        )
        used += len(data)
    return out, False


def _last_words(job: dict) -> str:
    """The tail of the engine's output, for when it dies without a marker."""
    try:
        job["log"].flush()
        text = (job["folder"] / "scan.log").read_bytes().decode("utf-8", errors="replace")
    except (OSError, ValueError):
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


def _watch_jobs() -> None:
    """Stop scans that run past the timeout even when no page is polling,
    and clear away finished ones nobody came back for."""
    while True:
        time.sleep(5)
        with _jobs_lock:
            jobs = list(_jobs.values())
        now = time.monotonic()
        for job in jobs:
            _enforce_timeout(job)
            if job["process"].poll() is None:
                continue
            if job["ended"] is None:
                job["ended"] = now
            # A page still polling is still reading - only silence counts.
            elif now - max(job["ended"], job["seen"]) > ABANDONED_SECONDS:
                with _jobs_lock:
                    gone = _jobs.pop(job["id"], None)
                if gone is not None:
                    _discard(gone)


def _health() -> dict:
    naps2 = scanner_devices.find_naps2()
    wia_script = sys.platform == "win32" and SCAN_PS1.exists()
    return {
        "ok": True,
        "agent": AGENT_NAME,
        "version": VERSION,
        "platform": sys.platform,
        # The page from before version 2 shows Scan only when this is true,
        # so it stays: true when either engine can run here.
        "scan_script": wia_script or (bool(naps2) and SCAN_NAPS2.exists()),
        "engines": {
            "naps2": bool(naps2),
            "naps2_path": naps2,
            "wia_script": wia_script,
        },
    }


# ---- HTTP --------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Per request; set again by _handle before a route runs.
    _body = b""
    _close_after = False

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
        self.send_header("Cache-Control", "no-store")
        if self._close_after:
            # Also sets close_connection, so nothing more is read from it.
            self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _take_body(self) -> None:
        """Read the request body off the connection before anything else.

        The connection stays open for the browser's next request, so a body
        left unread - by a route that has no use for it, or by a refusal -
        would be read as the start of that request. A body that cannot be
        read in full is refused and the connection closed behind it.
        """
        if self.headers.get("Transfer-Encoding"):
            self._close_after = True
            raise Refused(411, "Send the request with a Content-Length.")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            self._close_after = True
            raise Refused(400, "Bad Content-Length.")
        if length > MAX_BODY_BYTES:
            self._close_after = True
            raise Refused(413, "The request is too large.")
        self._body = self.rfile.read(length) if length > 0 else b""

    def _guard(self) -> None:
        """Turn away anything but the allowed pages, before a route runs.

        A browser names the page in Origin on every cross-site request that
        could do harm here (any POST, any fetch), so a request without one is
        a person typing the address or a tool on this machine.
        """
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in ALLOWED_HOSTS:
            raise Refused(403, f"The scan helper only answers at http://127.0.0.1:{PORT}.")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in ALLOWED_ORIGINS:
            raise Refused(403, "This page is not allowed to use the scan helper.")

    def _read_body(self, require_json: bool = False) -> dict:
        # A cross-site text/plain or form post needs no preflight; asking for
        # JSON puts every scan from another page through one.
        kind = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if require_json and kind != "application/json":
            raise Refused(415, "Send the scan settings as JSON (Content-Type: application/json).")
        raw = self._body
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise Refused(400, "The request body must be JSON.") from None
        if not isinstance(body, dict):
            raise Refused(400, "The request body must be a JSON object.")
        return body

    def _handle(self, route) -> None:
        self._body = b""
        self._close_after = False
        try:
            self._take_body()
            self._guard()
            route()
        except Refused as exc:
            self._json({"error": str(exc)}, exc.status)
        except Exception as exc:  # noqa: BLE001 - answer rather than drop the connection
            print(f"  error: {exc!r}", flush=True)
            self._json({"error": f"The scan helper hit an error: {exc}"}, 500)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self._cors()
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802
        self._handle(self._get)

    def do_POST(self) -> None:  # noqa: N802
        self._handle(self._post)

    # ---- routes ---------------------------------------------------------
    def _get(self) -> None:
        route = urlparse(self.path)
        query = parse_qs(route.query)

        if route.path == "/health":
            return self._json(_health())

        if route.path == "/devices":
            refresh = (query.get("refresh") or [""])[0].lower() in ("1", "true", "yes")
            cached = scanner_devices.cached_devices()
            if cached is not None and _running_job() is not None:
                # Listing asks every driver for its scanners, which can
                # disturb one that is mid-stack. The last answer will do
                # until the scan is over.
                return self._json(cached)
            return self._json(scanner_devices.get_devices(refresh=refresh))

        if route.path == "/pages":
            job_id = (query.get("job") or [""])[0]
            try:
                after = int((query.get("after") or ["0"])[0] or 0)
            except ValueError:
                raise Refused(400, "after must be a whole number.") from None
            with _jobs_lock:
                job = _jobs.get(job_id)
            if job is None:
                raise Refused(404, "No such scan.")
            job["seen"] = time.monotonic()
            # Finished is read before the pages are listed: an engine writes
            # its last page before its marker, so a finished job's pages are
            # all on disk by the time they are listed below.
            finished, error = _job_state(job)
            pages, more = _pages_after(job, after)
            finished = finished and not more
            return self._json(
                {
                    "pages": pages,
                    "finished": finished,
                    # Comes with the last batch, so pages already delivered
                    # from a failed scan never read as a whole stack.
                    "error": error if finished else None,
                }
            )

        raise Refused(404, "Not found.")

    def _post(self) -> None:
        route = urlparse(self.path)

        if route.path == "/scan":
            options = _scan_options(self._read_body(require_json=True))
            if _running_job() is not None:
                raise Refused(409, "A scan is already running on this computer.")
            device = _choose_device(options["device_id"])
            job = _start_scan(device, options)
            print(
                f"  scan started: job={job['id']} scanner={device['name']} "
                f"({device['driver']} via {device['engine']}) {options['source']} "
                f"{options['dpi']} DPI {options['color']}",
                flush=True,
            )
            return self._json({"job": job["id"], "device": device})

        if route.path == "/finish":
            # The browser has everything it needs: stop the scan if it is
            # still going and drop the temporary images.
            job_id = (parse_qs(route.query).get("job") or [""])[0]
            with _jobs_lock:
                job = _jobs.pop(job_id, None)
            if job is not None:
                _discard(job)
            return self._json({"ok": True})

        raise Refused(404, "Not found.")


def _describe_devices() -> None:
    """Look for scanners at start-up, so the first Scan from the older page
    does not wait on a Check, and say what was found."""
    try:
        listing = scanner_devices.get_devices(refresh=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  scanners      could not check: {exc}", flush=True)
        return
    devices = listing["devices"]
    if devices:
        for device in devices:
            print(
                f"  scanner       {device['name']}  ({device['driver']}, "
                f"{device['connection']}, via {device['engine']})",
                flush=True,
            )
    else:
        print("  scanner       none found yet - connect one and press Check", flush=True)
    for device in listing["unready"]:
        print(f"  no driver     {device['name']} - {device['fix']}", flush=True)
    for problem in listing["errors"]:
        print(f"  note          {problem}", flush=True)


def _stop_on_signal(signum, frame) -> None:
    raise KeyboardInterrupt


def _exit_cleanly_on_kill() -> None:
    """Make `kill` and a closed terminal stop the helper the way Ctrl+C does.

    A scan has its own process group, so it outlives a helper that simply
    dies - the scanner keeps feeding into a folder nobody reads. Routed
    through KeyboardInterrupt, both run the clean-up in main(). A signal
    someone chose to ignore (nohup ignores SIGHUP) is left ignored.
    """
    for name in ("SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is not None and signal.getsignal(number) == signal.SIG_DFL:
            signal.signal(number, _stop_on_signal)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--origin",
        action="append",
        default=[],
        metavar="URL",
        help="another page address allowed to use the helper (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    extra = os.environ.get("SCAN_AGENT_ORIGINS", "").split(",") + args.origin
    # An origin never ends in a slash; a copied address often does.
    ALLOWED_ORIGINS.update(o.strip().rstrip("/") for o in extra if o.strip())

    # Bind before printing anything, so a second copy says why it cannot
    # start instead of announcing itself and then dying with a traceback.
    # Loopback only. Binding 0.0.0.0 would let the rest of the network drive
    # this person's scanner.
    try:
        server = Server(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        if exc.errno not in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", -1)):
            raise
        sys.exit(
            f"Port {PORT} is already in use - the scan helper is probably "
            "already running in another window. Use that one."
        )

    health = _health()
    print(f"Akanksha scan helper (version {VERSION})", flush=True)
    print(f"  listening on  http://127.0.0.1:{PORT}", flush=True)
    naps2 = health["engines"]["naps2_path"]
    print(
        f"  NAPS2         {naps2 or 'not installed - ' + scanner_devices.install_hint()}",
        flush=True,
    )
    if sys.platform == "win32":
        print(f"  WIA script    {SCAN_PS1}  ({'found' if SCAN_PS1.exists() else 'MISSING'})", flush=True)
    print(f"  accepts       {', '.join(sorted(ALLOWED_ORIGINS))}", flush=True)
    print(f"  stops a scan  after {SCAN_TIMEOUT:g} seconds", flush=True)
    print("\n  Leave this window open while scanning. Ctrl+C to stop.\n", flush=True)

    _exit_cleanly_on_kill()
    threading.Thread(target=_watch_jobs, daemon=True).start()
    threading.Thread(target=_describe_devices, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        # A scan has its own process group, so Ctrl+C here does not reach
        # it - stop it explicitly rather than leave the scanner running.
        with _jobs_lock:
            jobs = list(_jobs.values())
            _jobs.clear()
        for job in jobs:
            _discard(job)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
