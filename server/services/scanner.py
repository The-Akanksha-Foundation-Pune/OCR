"""Drive the document scanner directly, so a scan never has to be uploaded.

Wraps tools/scan.ps1, which talks to whatever scanner Windows exposes over
WIA. Keeping the scanning logic in that one script means the command line
and the web page behave identically - the same settings, the same duplex
side-picking, the same failure messages.
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

from server.config import (
    SCAN_SCRIPT,
    SCAN_TIMEOUT_SECONDS,
    SCANNER_NAME,
)
from server.services.debug_log import debug_log


class ScannerError(Exception):
    """Raised with a message worth showing the operator."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def scan_to_pdf(scanner_name: str | None = None) -> tuple[bytes, str]:
    """
    Pull whatever is in the feeder and return (pdf_bytes, filename).

    Raises ScannerError with something the operator can act on - an empty
    feeder and a missing driver need different responses, and "scan failed"
    tells them neither.
    """
    if not SCAN_SCRIPT.exists():
        raise ScannerError(f"Scan script is missing: {SCAN_SCRIPT}", status_code=500)

    name = (scanner_name or SCANNER_NAME).strip()
    out_dir = Path(tempfile.gettempdir()) / "ocr-webscan"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"scan-{uuid.uuid4().hex}.pdf"

    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCAN_SCRIPT),
        "-OutFile",
        str(out_file),
    ]
    if name:
        command += ["-Scanner", name]

    debug_log(f"[SCAN] running {' '.join(command)}")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=SCAN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise ScannerError(
            f"The scanner did not finish within {SCAN_TIMEOUT_SECONDS}s. "
            "Check for a paper jam, then try again."
        ) from None
    except FileNotFoundError:
        raise ScannerError(
            "PowerShell could not be launched, so the scanner cannot be driven "
            "from this machine.",
            status_code=500,
        ) from None

    output = f"{completed.stdout}\n{completed.stderr}"
    debug_log(f"[SCAN] exit={completed.returncode} output={output.strip()[:800]!r}")

    if completed.returncode != 0 or not out_file.exists():
        raise ScannerError(_friendly_error(output, name))

    data = out_file.read_bytes()
    try:
        out_file.unlink()
    except OSError:
        pass

    if not data:
        raise ScannerError("The scanner produced an empty file. Try scanning again.")

    debug_log(f"[SCAN] captured {len(data)} bytes from {name!r}")
    return data, out_file.name


def _friendly_error(output: str, scanner_name: str) -> str:
    """Turn the script's stderr into something an operator can act on."""
    lowered = output.lower()
    if "feeder is empty" in lowered or "nothing was scanned" in lowered:
        return (
            "The feeder is empty. Load the forms face down into the ADF chute "
            "at the back, pushing until the rollers grip, then scan again."
        )
    if "jam" in lowered:
        return (
            "The scanner reported a paper jam. Open the cover, clear the sheet, "
            "reload the stack and try again."
        )
    if "no scanner found" in lowered or "no scanner matched" in lowered:
        return (
            f"No scanner named {scanner_name!r} is connected. Check it is powered "
            "on and plugged in."
        )
    if "more than one scanner" in lowered:
        return (
            "Several scanners are attached and none was selected. Set SCANNER_NAME "
            "in .env to the one you want to use."
        )
    tail = [line.strip() for line in output.splitlines() if line.strip()]
    return tail[-1] if tail else "The scan failed. Check the scanner and try again."
