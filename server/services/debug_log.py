"""Always-visible OCR/DB debug logging (terminal + log file)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "ocr_debug.log"


def _safe_print(line: str) -> None:
    """
    Print without ever raising.

    The Windows console is cp1252, so printing the Devanagari this pipeline
    routinely reads raises UnicodeEncodeError - and that is a subclass of
    ValueError, so it used to be swallowed by callers catching ValueError
    around a crop, silently costing them the reading. Logging must never be
    able to change what the caller does.
    """
    try:
        print(line, flush=True)
        return
    except UnicodeEncodeError:
        pass
    except Exception:
        return

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        print(line.encode(encoding, "replace").decode(encoding, "replace"), flush=True)
    except Exception:
        pass


def debug_log(message: str) -> None:
    """Print and append to logs/ocr_debug.log so debug is always findable."""
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    _safe_print(line)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        # A logging failure must never break a scan.
        pass
