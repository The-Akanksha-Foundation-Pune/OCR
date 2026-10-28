"""
Read the handwritten Student ID with Google Cloud Vision.

The local recognisers read only about 31% of forms exactly; the rest are
recovered by fuzzy-matching against the database, which is what forces a
human to confirm. Vision's handwriting model is far stronger, so the point
of this is not speed - it is getting enough readings exact that confirmation
becomes rare instead of routine.

Only a crop that this page itself places on the Student ID field is ever
sent, never the whole form, and the caller has to vouch for it:
extract_student_id sends the field located from the form's ruled lines when
they matched closely (error within LAYOUT_TRUST_ERROR), or the crop beside a
line that really is the Student ID label - each no taller than
VISION_MAX_CROP_HEIGHT, about two form rows. A loosely matched layout (on
page-006 its crop sat on the parent's mobile number), every template box
(a fixed page fraction, which drifts onto the parent's details on a real
scan), a crop beside any other line (the email line can win the label
search), and anything taller are read by the local recognisers only.

The one other crop sent is a page's top corner, for the handwritten batch
code (recognize_text): a strip 9% of the page tall holding the code and the
printed serial, above every line of the form.

What Vision sends back is not written to the log unless it is shaped like
a Student ID (id_shape.loggable_reading): a crop that strays still must not
leave a parent's name, email or phone number in logs/ocr_debug.log.

Everything here fails soft. If the network is down, the credentials are
missing, or the quota is spent, this returns nothing and the local
recognisers carry on as before - a scan in progress must never stop because
a cloud service is unavailable.
"""

from __future__ import annotations

import threading
from typing import Any

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


def recognize_text(crop_bgr: np.ndarray) -> list[tuple[str, float]]:
    """Read a page corner's handwritten batch code ("B1-12") with Cloud Vision.

    Unlike recognize_with_vision the text comes back as written - the dash
    and the spaces are the point of a batch code, and the Student ID clean-up
    would strip them. Only a corner crop is ever sent: it holds the code and
    the printed serial, never a child's or a parent's details. The text is
    not logged here; the caller logs only a code that parses.

    Args:
        crop_bgr: The corner crop.

    Returns:
        (text, confidence) per line, or an empty list if Vision is
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
            return []
        response = client.document_text_detection(
            image=vision.Image(content=encoded.tobytes()),
            image_context=vision.ImageContext(language_hints=["en"]),
            timeout=VISION_TIMEOUT_SECONDS,
        )
        if response.error.message:
            debug_log(f"[VISION] corner read api error: {response.error.message}")
            return []
    except Exception as exc:  # noqa: BLE001 - fall back to the local reader
        debug_log(f"[VISION] corner read failed: {type(exc).__name__}: {exc}")
        return []

    full = (response.full_text_annotation.text or "").strip()
    if not full:
        return []
    confidence = _page_confidence(response)
    return [(line.strip(), confidence) for line in full.splitlines() if line.strip()]


def _collect_readings(response: Any) -> list[tuple[str, float]]:
    """Pull whole-crop text and per-line text out of a Vision response."""
    from server.services.handwriting_ocr import clean_student_id
    from server.services.id_shape import loggable_reading

    found: list[tuple[str, float]] = []

    full = (response.full_text_annotation.text or "").strip()
    if full:
        cleaned = clean_student_id(full.replace("\n", ""))
        confidence = _page_confidence(response)
        # Never the raw text: if the crop caught anything but the ID, that
        # is a name or contact details. A reading shaped like an ID is the
        # only thing worth keeping in the log.
        debug_log(
            f"[VISION] read {len(full.splitlines())} line(s) -> "
            f"{loggable_reading(cleaned)} conf={confidence:.2f}"
        )
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
    except Exception as exc:  # noqa: BLE001 - a missing confidence is not an error
        debug_log(f"[VISION] no page confidence in the response ({exc}); assuming 0.90")
    return 0.90
