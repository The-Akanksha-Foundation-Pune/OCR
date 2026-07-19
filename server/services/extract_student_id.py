"""Orchestrate Student ID extraction from a scanned consent form."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from server.config import BASE_DIR
from server.services.debug_log import debug_log
from server.services.field_detector import (
    crop_student_id_value,
    crop_template_value,
    find_student_id_label,
    get_rapid_ocr,
    iter_template_languages,
    parse_student_id_from_text,
)
from server.services.handwriting_ocr import (
    looks_like_student_id,
    recognize_handwritten_id,
    recognize_with_easyocr,
)
from server.services.history_store import log_scan_attempt
from server.services.preprocess import (
    load_image_from_bytes,
    load_image_from_path,
    load_pages_from_bytes,
    preprocess_for_ocr,
)
from server.services.student_lookup import find_student_by_id

DEBUG_CROP_DIR = BASE_DIR / "logs" / "crops"


@dataclass
class ExtractResult:
    student_id: str
    confidence: float
    form_language: str
    source: str
    label_text: str = ""
    message: str = "ok"
    found_in_db: bool = False
    student: dict[str, Any] | None = None
    db_error: str | None = None
    db_debug: dict[str, Any] | None = None
    error: str | None = None
    page_number: int = 1
    page_count: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class ExtractError(Exception):
    """Raised when extraction fails with a user-facing message."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def extract_student_id_from_path(file_path: str | Path) -> ExtractResult:
    path = Path(file_path)
    image = load_image_from_path(path)
    return _extract(image, source=path.name)


def extract_student_id_from_upload(data: bytes, filename: str) -> ExtractResult:
    """
    OCR ONLY — reads the raw handwritten value off the scan and returns it
    exactly as read, with NO database contact at all. This is what the
    confirmation modal shows the user to compare against the physical
    paper. The DB (with its fuzzy-repair safety net) is only consulted
    later, in confirm_student_id(), once the user has
    reviewed or corrected this raw reading.
    """
    if not data:
        raise ExtractError("Empty upload", status_code=400)
    image = load_image_from_bytes(data, filename)
    return _extract(image, source=filename)


def extract_student_ids_from_upload_batch(data: bytes, filename: str) -> list[ExtractResult]:
    """
    OCR ONLY, for EVERY page — a printer/scanner machine often batch-scans a
    whole stack of consent forms into one multi-page PDF. Returns one
    ExtractResult per page (in order) so the UI can walk the user through
    confirming each Student ID one at a time. A page that fails OCR gets a
    result with student_id="" and `error` set, instead of blowing up the
    whole batch.
    """
    if not data:
        raise ExtractError("Empty upload", status_code=400)

    pages = load_pages_from_bytes(data, filename)
    if not pages:
        raise ExtractError("No pages found in upload.", status_code=400)

    total = len(pages)
    results: list[ExtractResult] = []
    for index, image in enumerate(pages, start=1):
        page_source = filename if total == 1 else f"{filename} (page {index}/{total})"
        try:
            result = _extract(image, source=page_source)
        except ExtractError as exc:
            result = ExtractResult(
                student_id="",
                confidence=0.0,
                form_language="unknown",
                source=page_source,
                message=exc.message,
                error=exc.message,
            )
        result.page_number = index
        result.page_count = total
        results.append(result)

    return results


def confirm_student_id(
    student_id: str,
    source: str = "manual-confirm",
    *,
    image_link: str | None = None,
    defer_log: bool = False,
) -> ExtractResult:
    """
    Final step after the user reviews (and optionally corrects) the OCR
    preview in the UI. Looks up the given Student ID and, if found, marks
    scanned_at — this is the only place scanned_at gets written.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", student_id or "").upper()
    if not cleaned:
        raise ExtractError("Student ID cannot be empty.", status_code=400)

    result = ExtractResult(
        student_id=cleaned,
        confidence=1.0,
        form_language="manual",
        source=source,
        label_text="user-confirmed",
        message="ok",
    )
    result = _with_db_lookup(result, mark_scanned=True)
    if not defer_log:
        _log_confirm_attempt(result, image_link=image_link)
    return result


def log_confirm_result(
    result: ExtractResult, *, image_link: str | None = None
) -> None:
    """Public wrapper used by the confirm API after Drive upload."""
    _log_confirm_attempt(result, image_link=image_link)


def _log_confirm_attempt(
    result: ExtractResult, *, image_link: str | None = None
) -> None:
    """Record this confirm attempt for the History page's Not Found list.
    Never lets logging problems break the confirm flow itself."""
    try:
        student = result.student or {}
        log_scan_attempt(
            confirmed_student_id=result.student_id,
            found_in_db=result.found_in_db,
            matched_student_id=student.get("student_id"),
            student_name=student.get("student_name"),
            school_name=student.get("school_name"),
            source=result.source,
            message=result.message,
            image_link=image_link,
        )
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[HISTORY] failed to log scan attempt: {exc}")


def _extract(image_bgr: np.ndarray, source: str) -> ExtractResult:
    """Pure OCR — never touches the database. Returns the single best raw
    reading so the confirmation modal shows exactly what was scanned."""
    prepared = preprocess_for_ocr(image_bgr)
    label = find_student_id_label(prepared)
    candidates: list[tuple[str, float, str, str]] = []

    if label is not None:
        debug_log(
            f"[DETECT] label={label.text!r} language={label.language} "
            f"source={label.source} parsed={label.parsed_student_id!r}"
        )

        # Keep noisy RapidOCR tokens too — better to show the user a rough
        # reading to correct than nothing at all.
        if label.parsed_student_id:
            token = re.sub(r"[^A-Za-z0-9]", "", label.parsed_student_id).upper()
            if looks_like_student_id(token):
                candidates.append((token, 0.92, label.language, f"rapidocr:{label.text}"))
            elif len(token) >= 8 and token[:4].isalpha() and any(c.isdigit() for c in token):
                debug_log(f"[DETECT] noisy OCR token kept: {token!r}")
                candidates.append((token, 0.70, label.language, f"noisy:{label.text}"))

        if label.source == "ocr":
            try:
                crop = crop_student_id_value(prepared, label)
                _save_debug_crop(crop, "label")
                candidates.extend(_read_id_from_crop(crop, label.language, label.text))
            except ValueError as exc:
                debug_log(f"[DETECT] OCR crop failed: {exc}")

    best = _best_ocr_candidate(candidates)

    # Only fall back to the slower multi-language template crops when the
    # label-based reading found nothing at all — most real scans resolve
    # above without needing this.
    if best is None:
        languages = iter_template_languages()
        if label is not None and label.language in languages:
            languages = [label.language] + [lang for lang in languages if lang != label.language]

        fallback_candidates: list[tuple[str, float, str, str]] = []
        for language in languages:
            try:
                crop = crop_template_value(prepared, language)
                _save_debug_crop(crop, f"template_{language}")
                fallback_candidates.extend(
                    _read_id_from_crop(crop, language, f"template:{language}")
                )
            except ValueError:
                continue

        best = _best_ocr_candidate(fallback_candidates)

    if best is None:
        raise ExtractError(
            "Could not read a valid Student ID (letters+digits, e.g. AADMIS310714). "
            "OCR may have misread handwriting. Retake with Take photo on the FULL page, "
            "close and sharp on the Student ID line.",
            status_code=422,
        )

    student_id, confidence, language, label_text = best
    return ExtractResult(
        student_id=student_id,
        confidence=min(confidence, 0.99),
        form_language=language,
        source=source,
        label_text=label_text,
        message="ocr_read — not yet checked against the database",
        found_in_db=False,
        student=None,
    )


def _best_ocr_candidate(
    candidates: list[tuple[str, float, str, str]]
) -> tuple[str, float, str, str] | None:
    """Pick the single most plausible raw OCR reading — strictly-formatted
    IDs (letters-then-digits) are preferred over noisy ones, then by
    confidence. No database is touched here."""
    ranked = [
        row
        for row in candidates
        if looks_like_student_id(row[0]) or _is_noisy_id_like(row[0])
    ]
    if not ranked:
        return None

    ranked.sort(key=lambda row: (looks_like_student_id(row[0]), row[1]), reverse=True)
    debug_log(f"[OCR] candidates={ranked} -> best={ranked[0]}")
    return ranked[0]


def _is_noisy_id_like(token: str) -> bool:
    """Loose shape check so near-miss OCR still reaches DB fuzzy repair."""
    return (
        8 <= len(token) <= 20
        and token[:4].isalpha()
        and any(c.isdigit() for c in token)
    )


def _read_id_from_crop(
    crop_bgr: np.ndarray, language: str, label_text: str
) -> list[tuple[str, float, str, str]]:
    """Run three independent recognizers on a crop and keep every plausible
    reading (strict + noisy) so DB fuzzy repair gets the best shot at
    recovering the real Student ID from messy handwriting."""
    found: list[tuple[str, float, str, str]] = []

    # 1) RapidOCR on the crop alone (often best for blocky handwritten caps)
    try:
        ocr = get_rapid_ocr()
        result, _ = ocr(crop_bgr)
        for item in result or []:
            text = str(item[1])
            conf = float(item[2])
            parsed = parse_student_id_from_text(text) or re.sub(
                r"[^A-Za-z0-9]", "", text
            ).upper()
            debug_log(f"[RapidOCR-crop] text={text!r} parsed={parsed!r} conf={conf}")
            if looks_like_student_id(parsed):
                found.append((parsed, conf + 0.05, language, f"crop-rapid:{label_text}"))
            elif _is_noisy_id_like(parsed):
                found.append((parsed, min(conf, 0.65), language, f"crop-rapid-noisy:{label_text}"))
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[RapidOCR-crop] failed: {exc}")

    # 2) TrOCR handwriting model
    student_id, confidence = recognize_handwritten_id(crop_bgr)
    if student_id and looks_like_student_id(student_id):
        found.append((student_id, confidence, language, f"crop-trocr:{label_text}"))

    # 3) EasyOCR — independently trained recognizer, a second opinion on the
    # same crop; often catches characters the other two miss.
    for text, conf in recognize_with_easyocr(crop_bgr):
        if looks_like_student_id(text):
            found.append((text, conf + 0.05, language, f"crop-easyocr:{label_text}"))
        elif _is_noisy_id_like(text):
            found.append((text, min(conf, 0.65), language, f"crop-easyocr-noisy:{label_text}"))

    return found


def _save_debug_crop(crop_bgr: np.ndarray, tag: str) -> None:
    try:
        DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%H%M%S")
        path = DEBUG_CROP_DIR / f"{stamp}_{tag}.jpg"
        cv2.imwrite(str(path), crop_bgr)
        debug_log(f"[DEBUG] saved crop {path}")
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[DEBUG] could not save crop: {exc}")


def _with_db_lookup(result: ExtractResult, mark_scanned: bool = False) -> ExtractResult:
    """Attach active_student_data match for the OCR Student ID.

    mark_scanned=False (the default, used during OCR preview) never writes
    scanned_at — only confirm_student_id() does that, once the user has
    reviewed/corrected the ID in the confirmation modal.
    """
    from server.config import DB_HOST, DB_NAME, STUDENT_TABLE

    result.db_debug = {
        "ocr_student_id": result.student_id,
        "db_host": DB_HOST,
        "db_name": DB_NAME,
        "table": STUDENT_TABLE,
        "column": "student_id",
        "sql": (
            f"SELECT * FROM {STUDENT_TABLE} "
            f"WHERE UPPER(TRIM(student_id)) = UPPER(%s) "
            f"(exact, safe variant, or unique constrained repair)"
        ),
        "sql_param": result.student_id,
        "lookup_result": "pending",
    }

    debug_log(f"[OCR] Student ID found: {result.student_id!r} (confidence={result.confidence})")
    try:
        student = find_student_by_id(result.student_id, mark_scanned=mark_scanned)
        if student is None:
            debug_log(
                f"[DB] No match in active_student_data.student_id for: {result.student_id!r}"
            )
            result.found_in_db = False
            result.student = None
            result.message = "student_id_not_found_in_db"
            result.db_debug["lookup_result"] = "not_found"
            return result

        result.student_id = student.get("student_id") or result.student_id
        result.found_in_db = True
        result.student = student
        result.message = "ok"
        result.db_debug["lookup_result"] = "found"
        result.db_debug["matched_student_id"] = student.get("student_id")
        result.db_debug["ocr_student_id"] = result.db_debug["sql_param"]
        return result
    except Exception as exc:  # noqa: BLE001 — keep OCR result even if DB fails
        debug_log(f"[DB] Lookup failed for {result.student_id!r}: {exc}")
        result.found_in_db = False
        result.student = None
        result.db_error = str(exc)
        result.message = "db_lookup_failed"
        result.db_debug["lookup_result"] = "error"
        result.db_debug["error"] = str(exc)
        return result
