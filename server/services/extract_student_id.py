"""Orchestrate Student ID extraction from a scanned consent form."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from server.config import (
    BASE_DIR,
    BLANK_PAGE_INK_RATIO,
    FAST_MODE,
    VISION_CONFIDENCE_BONUS,
    LAYOUT_TRUST_CONFIDENCE,
    USE_TROCR,
)
from server.services.debug_log import debug_log
from server.services.field_detector import (
    crop_student_id_value,
    crop_template_value,
    crop_value_roi,
    find_student_id_label,
    get_rapid_ocr,
    iter_template_languages,
    parse_student_id_from_text,
)
from server.services.form_layout import identify_form
from server.services.vision_ocr import is_configured as vision_configured
from server.services.vision_ocr import recognize_with_vision
from server.services.handwriting_ocr import (
    looks_like_student_id,
    recognize_devanagari_id,
    recognize_handwritten_id,
    recognize_with_easyocr,
)
from server.services.history_store import log_scan_attempt
from server.services.preprocess import (
    count_pages_from_bytes,
    load_image_from_bytes,
    load_image_from_path,
    load_page_from_bytes,
    load_pages_from_bytes,
    preprocess_for_ocr,
)
from server.services.student_lookup import find_student_by_id

DEBUG_CROP_DIR = BASE_DIR / "logs" / "crops"

# How much a strict letters-then-digits reading is worth against a noisy one.
# Enough to break a near-tie, not enough to beat a much more confident read.
STRICT_FORMAT_BONUS = 0.25

# A token parsed out of the whole label line, before any crop enhancement.
# Scored below the crop readers because they work on an upscaled, contrast
# boosted image of just the value - the line parse routinely mangles the
# handwriting it happens to overlap.
LABEL_LINE_CONFIDENCE = 0.55


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
    # What OCR actually read, kept even after a database match replaces
    # student_id. Without it the UI cannot show "read X, saved Y", which is
    # exactly the comparison someone needs to spot a wrong match.
    ocr_student_id: str = ""

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


def extract_one_page_from_stash(scan_id: str, page_number: int) -> ExtractResult:
    """
    OCR a single page of an already-stashed scan.

    Reading a whole stack inside one request meant a 12-minute HTTP call
    that browsers simply drop, so the page is fetched one at a time: the
    scan happens once, then each page is read in its own short request.
    """
    from server.services.scan_store import load_stashed

    loaded = load_stashed(scan_id)
    if loaded is None:
        raise ExtractError("That scan is no longer available. Scan again.", status_code=404)

    data, filename = loaded
    total = count_pages_from_bytes(data, filename)
    if total < 1:
        raise ExtractError("No pages found in that scan.", status_code=400)
    if page_number < 1 or page_number > total:
        raise ExtractError(
            f"Page {page_number} is outside this scan (it has {total}).",
            status_code=400,
        )

    # Render only the page asked for - see load_page_from_bytes.
    image = load_page_from_bytes(data, filename, page_number)
    source = filename if total == 1 else f"{filename} (page {page_number}/{total})"
    try:
        result = _extract(image, source=source)
    except ExtractError as exc:
        result = ExtractResult(
            student_id="",
            confidence=0.0,
            form_language="unknown",
            source=source,
            message=exc.message,
            error=exc.message,
        )
    result.page_number = page_number
    result.page_count = total
    return result


def extract_one_image(path, page_number: int, total: int) -> ExtractResult:
    """
    OCR a single already-captured page image.

    Used by the streaming scan, where pages arrive one at a time and there
    is no PDF to index into yet.
    """
    from pathlib import Path as _Path

    image = load_image_from_path(_Path(path))
    source = f"scan (page {page_number}/{total})" if total > 1 else "scan"
    try:
        result = _extract(image, source=source)
    except ExtractError as exc:
        result = ExtractResult(
            student_id="",
            confidence=0.0,
            form_language="unknown",
            source=source,
            message=exc.message,
            error=exc.message,
        )
    result.page_number = page_number
    result.page_count = total
    return result


def confirm_student_id(
    student_id: str,
    source: str = "manual-confirm",
    *,
    image_link: str | None = None,
    defer_log: bool = False,
    school: str | None = None,
    location: str | None = None,
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
    result = _with_db_lookup(
        result, mark_scanned=True, school=school, location=location
    )
    if not defer_log:
        _log_confirm_attempt(result, image_link=image_link)
    return result


def log_confirm_result(
    result: ExtractResult,
    *,
    image_link: str | None = None,
    scan_school: str | None = None,
    scan_location: str | None = None,
) -> None:
    """Public wrapper used by the confirm API after Drive upload."""
    _log_confirm_attempt(
        result,
        image_link=image_link,
        scan_school=scan_school,
        scan_location=scan_location,
    )


def _log_confirm_attempt(
    result: ExtractResult,
    *,
    image_link: str | None = None,
    scan_school: str | None = None,
    scan_location: str | None = None,
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
            matched_via=student.get("matched_via"),
            scan_school=scan_school,
            scan_location=scan_location,
        )
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[HISTORY] failed to log scan attempt: {exc}")


def _extract(image_bgr: np.ndarray, source: str) -> ExtractResult:
    """Pure OCR — never touches the database. Returns the single best raw
    reading so the confirmation modal shows exactly what was scanned."""
    prepared = preprocess_for_ocr(image_bgr)

    # A page with almost no ink is a blank reverse or a separator sheet.
    # Running four recognisers and three template fallbacks over it costs
    # ~27s to conclude what a pixel count answers instantly, and in a
    # 500-form batch those add up to hours.
    if _is_blank_page(prepared):
        debug_log("[FAST] page is blank - skipped OCR")
        raise ExtractError(
            "This page looks blank. If it should have a form on it, check the "
            "sheet was fed the right way up.",
            status_code=422,
        )

    # Which form is this, and where is its value box? Matching the page's
    # ruled lines works on Hindi/Marathi scans, where the printed label is
    # unreadable to an English-only OCR model and the label path below
    # comes back empty.
    layout = identify_form(prepared)
    # Whether to spend a Devanagari pass on this page. The layout is the
    # only trustworthy signal here: _score_label calls a bare ID token
    # "english" even on a Hindi form, so the per-candidate language would
    # switch the pass off exactly where it is needed.
    devanagari_form = layout is not None and layout.language in ("hindi", "marathi")

    # find_student_id_label runs RapidOCR over the entire page (~3.7s) purely
    # to locate the value field. When the rule ladder has already located it
    # with confidence, that is work for an answer we have.
    trust_layout = (
        FAST_MODE
        and layout is not None
        and layout.confidence >= LAYOUT_TRUST_CONFIDENCE
    )
    label = None if trust_layout else find_student_id_label(prepared)
    if trust_layout:
        debug_log(
            f"[FAST] layout confident ({layout.confidence:.2f}) - "
            "skipped full-page label OCR"
        )
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
                candidates.append(
                    (token, LABEL_LINE_CONFIDENCE, label.language, f"noisy:{label.text}")
                )

        if label.source == "ocr":
            try:
                crop = crop_student_id_value(prepared, label)
                _save_debug_crop(crop, "label")
                candidates.extend(
                    _read_id_from_crop(
                        crop,
                        label.language,
                        label.text,
                        allow_devanagari=devanagari_form,
                    )
                )
            except ValueError as exc:
                debug_log(f"[DETECT] OCR crop failed: {exc}")

    # The layout crop is a primary source, not a fallback: on Hindi and
    # Marathi forms it is the only one that produces anything.
    #
    # But only when the label path came up short. Reading a second region we
    # do not need is the single most expensive thing this function can do:
    # if that crop lands on a heading rather than the value line, every
    # recogniser runs over the wrong pixels and the page takes ~30s instead
    # of ~10s to reach an answer we already had.
    have_clean = FAST_MODE and any(
        looks_like_student_id(row[0]) for row in candidates
    )
    if have_clean:
        debug_log("[FAST] clean read from the label crop - skipped layout crop")

    if layout is not None and not have_clean:
        try:
            crop = crop_value_roi(prepared, layout.student_id_roi)
            _save_debug_crop(crop, f"layout_{layout.language}")
            candidates.extend(
                _read_id_from_crop(
                    crop,
                    layout.language,
                    f"layout:{layout.language}",
                    allow_devanagari=devanagari_form,
                )
            )
        except Exception as exc:  # noqa: BLE001 — a bad crop must not sink the page
            debug_log(f"[LAYOUT] crop failed: {exc}")

    best = _best_ocr_candidate(candidates)

    # Only fall back to the slower multi-language template crops when the
    # label-based reading found nothing at all — most real scans resolve
    # above without needing this.
    if best is None:
        languages = iter_template_languages()
        preferred = layout.language if layout is not None else (
            label.language if label is not None else None
        )
        if preferred in languages:
            languages = [preferred] + [lang for lang in languages if lang != preferred]

        fallback_candidates: list[tuple[str, float, str, str]] = []
        for language in languages:
            try:
                crop = crop_template_value(prepared, language)
                _save_debug_crop(crop, f"template_{language}")
                fallback_candidates.extend(
                    _read_id_from_crop(
                        crop,
                        language,
                        f"template:{language}",
                        allow_devanagari=language in ("hindi", "marathi"),
                    )
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
    # Report the language of the *form*, which the layout match knows, not
    # the winning candidate's. _score_label tags a bare ID token "english"
    # whatever the form says, so on a Hindi or Marathi page the candidate
    # language is routinely wrong.
    form_language = layout.language if layout is not None else language
    return ExtractResult(
        student_id=student_id,
        confidence=min(confidence, 0.99),
        form_language=form_language,
        source=source,
        label_text=label_text,
        message="ocr_read — not yet checked against the database",
        found_in_db=False,
        student=None,
    )


def _is_blank_page(image_bgr: np.ndarray) -> bool:
    """
    True for a page carrying no printed form.

    A scanned consent form measures around 4-6% dark pixels; a blank reverse
    showing only bleed-through sits under 1%. The gap is wide enough that a
    single threshold separates them reliably.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray < 160)) < BLANK_PAGE_INK_RATIO


def _best_ocr_candidate(
    candidates: list[tuple[str, float, str, str]]
) -> tuple[str, float, str, str] | None:
    """Pick the single most plausible raw OCR reading — strictly-formatted
    IDs (letters-then-digits) are preferred over noisy ones, then by
    confidence. No database is touched here."""
    ranked = [
        row
        for row in candidates
        if looks_like_student_id(row[0])
        or _is_noisy_id_like(row[0])
        or _is_digit_anchored(row[0])
    ]
    if not ranked:
        return None

    # Score additively rather than ranking format above everything. Sorting
    # by (is_strict, confidence) let a 0.02-confidence reading that merely
    # had the right shape beat a 0.65 one that was actually correct - the
    # shape is evidence, not a trump card.
    def score(row: tuple[str, float, str, str]) -> float:
        return row[1] + (STRICT_FORMAT_BONUS if looks_like_student_id(row[0]) else 0.0)

    ranked.sort(key=score, reverse=True)
    debug_log(
        "[OCR] candidates="
        + repr([(r[0], round(score(r), 3), r[3]) for r in ranked])
        + f" -> best={ranked[0][0]!r}"
    )
    return ranked[0]


def _is_noisy_id_like(token: str) -> bool:
    """Loose shape check so near-miss OCR still reaches DB fuzzy repair."""
    return (
        8 <= len(token) <= 20
        and token[:4].isalpha()
        and any(c.isdigit() for c in token)
    )


def _is_digit_anchored(token: str) -> bool:
    """
    Admit a reading whose six-digit tail is intact even though its letters
    are a mess (L0NDN011021, MH2011021).

    Devanagari OCR scatters junk through the letter half, which the check
    above rejects outright for not starting with four letters. But the
    digits are the half student_lookup can actually resolve against, so a
    clean tail is worth keeping and letting the digit-suffix lookup finish.
    """
    return bool(re.search(r"\d{6}$", token or "")) and any(
        c.isalpha() for c in token
    )


def _read_id_from_crop(
    crop_bgr: np.ndarray,
    language: str,
    label_text: str,
    allow_devanagari: bool = False,
) -> list[tuple[str, float, str, str]]:
    """Run three independent recognizers on a crop and keep every plausible
    reading (strict + noisy) so DB fuzzy repair gets the best shot at
    recovering the real Student ID from messy handwriting."""
    found: list[tuple[str, float, str, str]] = []

    # 0) Cloud Vision first when configured - it reads handwriting far
    # better than anything local, so a clean result here means the rest of
    # this function can be skipped entirely.
    if vision_configured():
        for text, conf in recognize_with_vision(crop_bgr):
            score = min(conf + VISION_CONFIDENCE_BONUS, 0.99)
            if looks_like_student_id(text):
                found.append((text, score, language, f"crop-vision:{label_text}"))
            elif _is_noisy_id_like(text) or _is_digit_anchored(text):
                found.append(
                    (text, min(score, 0.70), language, f"crop-vision-noisy:{label_text}")
                )
        if FAST_MODE and any(looks_like_student_id(row[0]) for row in found):
            debug_log("[VISION] clean read - skipped local recognizers")
            return found

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

    # 2) TrOCR handwriting model - off by default, see USE_TROCR in config
    if USE_TROCR:
        student_id, confidence = recognize_handwritten_id(crop_bgr)
        if student_id and looks_like_student_id(student_id):
            found.append((student_id, confidence, language, f"crop-trocr:{label_text}"))

    # 3) EasyOCR — independently trained recognizer, a second opinion on the
    # same crop; often catches characters the other two miss. It is also the
    # slowest, so in fast mode it only runs when nothing above produced a
    # cleanly-formatted ID and a second opinion would actually change things.
    have_clean = any(looks_like_student_id(row[0]) for row in found)
    if not (FAST_MODE and have_clean):
        for text, conf in recognize_with_easyocr(crop_bgr):
            if looks_like_student_id(text):
                found.append(
                    (text, conf + 0.05, language, f"crop-easyocr:{label_text}")
                )
            elif _is_noisy_id_like(text):
                found.append(
                    (text, min(conf, 0.65), language, f"crop-easyocr-noisy:{label_text}")
                )

    # 4) Devanagari — on a Hindi or Marathi form the parent may have written
    # the ID in Devanagari, which the three Latin-only readers above cannot
    # see at all. Only the digits survive transliteration cleanly, so these
    # readings are scored below the Latin ones and lean on the digit-suffix
    # lookup in student_lookup to resolve the letters.
    # Only worth the extra pass when the Latin readers came back with
    # nothing usable. Most IDs are written in Latin even on Hindi and
    # Marathi forms, and the Devanagari model is the slowest of the four.
    if allow_devanagari and not found:
        for text, conf in recognize_devanagari_id(crop_bgr):
            if looks_like_student_id(text):
                found.append(
                    (text, min(conf, 0.75), language, f"crop-deva:{label_text}")
                )
            elif _is_noisy_id_like(text) or _is_digit_anchored(text):
                found.append(
                    (text, min(conf, 0.55), language, f"crop-deva-noisy:{label_text}")
                )

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


def _with_db_lookup(
    result: ExtractResult,
    mark_scanned: bool = False,
    school: str | None = None,
    location: str | None = None,
) -> ExtractResult:
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
        student = find_student_by_id(
            result.student_id,
            mark_scanned=mark_scanned,
            school=school,
            location=location,
        )
        if student is None:
            debug_log(
                f"[DB] No match in active_student_data.student_id for: {result.student_id!r}"
            )
            result.found_in_db = False
            result.student = None
            result.ocr_student_id = result.student_id
            result.message = "student_id_not_found_in_db"
            result.db_debug["lookup_result"] = "not_found"
            return result

        result.ocr_student_id = result.student_id
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
