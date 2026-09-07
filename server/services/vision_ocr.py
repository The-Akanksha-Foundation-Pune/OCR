"""
Read the handwritten Student ID with Google Cloud Vision.

The local recognisers read only about 31% of forms exactly; the rest are
recovered by fuzzy-matching against the database, which is what forces a
human to confirm. Vision's handwriting model is far stronger, so the point
of this is not speed - it is getting enough readings exact that confirmation
becomes rare instead of routine.

Only the cropped Student ID field is ever sent, never the whole form. The
crop is under 2% of the page and contains a 12-character code: no child's
name, no parent's name, no contact details, no signature.

Everything here fails soft. If the network is down, the credentials are
missing, or the quota is spent, this returns nothing and the local
recognisers carry on as before - a scan in progress must never stop because
a cloud service is unavailable.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from server.config import VISION_CREDENTIALS, VISION_ENABLED, VISION_TIMEOUT_SECONDS
from server.services.debug_log import debug_log

_client = None
_client_failed = False
_lock = threading.Lock()


def is_configured() -> bool:
    """True when Vision has credentials and has not been turned off."""
    return bool(VISION_ENABLED and VISION_CREDENTIALS)


def get_client():
    """
    Build the Vision client once, and remember if it cannot be built.

    A missing or malformed key file should cost one failed attempt and a log
    line, not a retry on every page of a 500-form batch.
    """
    global _client, _client_failed

    if _client is not None or _client_failed:
        return _client

    with _lock:
        if _client is not None or _client_failed:
            return _client

        if not is_configured():
            _client_failed = True
            debug_log("[VISION] not configured - GOOGLE_VISION_CREDENTIALS unset")
            return None

        try:
            from google.cloud import vision
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                VISION_CREDENTIALS
            )
            _client = vision.ImageAnnotatorClient(credentials=credentials)
            debug_log(f"[VISION] client ready ({VISION_CREDENTIALS})")
        except Exception as exc:  # noqa: BLE001 — never break the local path
            _client_failed = True
            debug_log(f"[VISION] unavailable: {type(exc).__name__}: {exc}")
            return None

    return _client


def recognize_with_vision(crop_bgr: np.ndarray) -> list[tuple[str, float]]:
    """
    Read a Student ID crop with Cloud Vision.

    Returns (text, confidence) pairs, or an empty list if Vision is
    unavailable for any reason.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []

    client = get_client()
    if client is None:
        return []

    try:
        from google.cloud import vision

        ok, encoded = cv2.imencode(".png", crop_bgr)
        if not ok:
            debug_log("[VISION] could not encode crop")
            return []

        image = vision.Image(content=encoded.tobytes())
        # DOCUMENT_TEXT_DETECTION is the handwriting-capable model; the
        # language hints keep it from reaching for scripts that cannot
        # appear in an ID, while still allowing Devanagari digits.
        context = vision.ImageContext(language_hints=["en", "hi", "mr"])
        response = client.document_text_detection(
            image=image,
            image_context=context,
            timeout=VISION_TIMEOUT_SECONDS,
        )

        if response.error.message:
            debug_log(f"[VISION] api error: {response.error.message}")
            return []
    except Exception as exc:  # noqa: BLE001 — fall back to local readers
        debug_log(f"[VISION] request failed: {type(exc).__name__}: {exc}")
        return []

    return _collect_readings(response)


def _collect_readings(response) -> list[tuple[str, float]]:
    """Pull whole-crop text and per-line text out of a Vision response."""
    from server.services.handwriting_ocr import clean_student_id

    found: list[tuple[str, float]] = []

    full = (response.full_text_annotation.text or "").strip()
    if full:
        cleaned = clean_student_id(full.replace("\n", ""))
        confidence = _page_confidence(response)
        debug_log(f"[VISION] raw={full!r} cleaned={cleaned!r} conf={confidence:.2f}")
        if cleaned:
            found.append((cleaned, confidence))

    # A crop occasionally comes back split across lines; offer each line as
    # its own candidate so a stray mark on one line cannot spoil the ID.
    for line in full.splitlines():
        cleaned = clean_student_id(line)
        if cleaned and all(cleaned != text for text, _ in found):
            found.append((cleaned, _page_confidence(response) * 0.95))

    return found


def _page_confidence(response) -> float:
    """Vision's own confidence for the page, defaulting high when absent."""
    try:
        pages = response.full_text_annotation.pages
        if pages and pages[0].confidence:
            return float(pages[0].confidence)
    except Exception:  # noqa: BLE001
        pass
    return 0.90
