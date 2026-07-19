"""Always-visible OCR/DB debug logging (terminal + log file)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "ocr_debug.log"


def debug_log(message: str) -> None:
    """Print and append to logs/ocr_debug.log so debug is always findable."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
