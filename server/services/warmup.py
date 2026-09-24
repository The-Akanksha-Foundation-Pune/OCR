"""Load the OCR models before the first page needs them.

RapidOCR and the Cloud Vision client are built lazily, on first use. After a
restart that first use lands on whoever scans next: the first full-page OCR
of the day took 38 seconds where a warm one takes five. Loading them in the
background at startup moves that cost to the seconds after launch, when
nobody is waiting on it.

Failures are logged and swallowed - a warm-up must never stop the app.
"""

from __future__ import annotations

import threading
import time

from server.services.debug_log import debug_log


def _warm() -> None:
    started = time.perf_counter()
    try:
        from server.services.field_detector import get_rapid_ocr

        get_rapid_ocr()
        debug_log(f"[WARMUP] RapidOCR ready in {time.perf_counter() - started:.1f}s")
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[WARMUP] RapidOCR failed: {exc}")

    try:
        from server.services import vision_ocr

        if vision_ocr.is_configured():
            vision_ocr.get_client()
            debug_log(f"[WARMUP] Vision client ready at {time.perf_counter() - started:.1f}s")
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[WARMUP] Vision failed: {exc}")


def warm_models_in_background() -> None:
    threading.Thread(target=_warm, name="ocr-warmup", daemon=True).start()
