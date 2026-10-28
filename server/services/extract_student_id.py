"""Orchestrate Student ID extraction from a scanned consent form."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from server.config import (
    BASE_DIR,
    BLANK_PAGE_INK_RATIO,
    FAST_MODE,
    LAYOUT_TRUST_CONFIDENCE,
    LAYOUT_TRUST_ERROR,
    USE_TROCR,
    VISION_CONFIDENCE_BONUS,
)
from server.services.clock import now_ist
from server.services.debug_log import debug_log
from server.services.deskew import deskew
from server.services.field_detector import (
    LabelHit,
    _score_label,
    crop_student_id_value,
    crop_template_value,
    crop_value_roi,
    find_student_id_label,
    get_rapid_ocr,
    iter_template_languages,
    parse_student_id_from_text,
)
from server.services.form_layout import identify_form
from server.services.handwriting_ocr import (
    looks_like_student_id,
    recognize_devanagari_id,
    recognize_handwritten_id,
    recognize_with_easyocr,
)
from server.services.history_store import log_scan_attempt
from server.services.id_shape import as_student_id, loggable_reading
from server.services.id_shape import looks_like_student_id as has_student_id_shape
from server.services.preprocess import (
    count_pages_from_bytes,
    load_image_from_bytes,
    load_image_from_path,
    load_page_from_bytes,
    load_pages_from_bytes,
    preprocess_for_ocr,
)
from server.services.student_lookup import DB_ERRORS, find_student_by_id
from server.services.vision_ocr import is_configured as vision_configured
from server.services.vision_ocr import recognize_with_vision

DEBUG_CROP_DIR = BASE_DIR / "logs" / "crops"

# How much a strict letters-then-digits reading is worth against a noisy one.
# Enough to break a near-tie, not enough to beat a much more confident read.
STRICT_FORMAT_BONUS = 0.25

# A token parsed out of the whole label line, before any crop enhancement.
# Scored below the crop readers because they work on an upscaled, contrast
# boosted image of just the value - the line parse routinely mangles the
# handwriting it happens to overlap.
LABEL_LINE_CONFIDENCE = 0.55

# What every reading path says when nothing on the page has the shape of a
# Student ID. Worded so it cannot be mistaken for the blank-page message,
# which the page tells apart by its "looks blank".
NO_STUDENT_ID_MESSAGE = (
    "No Student ID found on this page (letters then digits, e.g. "
    "AADMIS310714). The handwriting may be unclear - type the ID from the "
    "paper, or rescan with the Student ID line sharp."
)

# The tallest crop Cloud Vision may see, as a fraction of page height.
# The value-field crops measure 0.030-0.052 of the page with their padding
# (template ROI 0.030-0.035, rule-located layout ROI 0.046-0.052), about
# one and a half form rows. A label crop is 3.2x the height of whatever line
# was taken for the label, and the one that sent a parent's name, email and
# phone number to Vision was 0.079 - three rows of contact details.
VISION_MAX_CROP_HEIGHT = 0.065

# How sure _score_label must be that a line is the Student ID label before
# the crop beside it may go to Vision. 2.5 and up means the line names the
# student ("Student ID", "विद्यार्थी आयडी"); a bare "ID" scores 2.0, and that
# is also how "ईमेल आईडी" - the email line - would score.
STUDENT_LABEL_SCORE = 2.5

# Server-only fields: never sent to the browser.
_INTERNAL_FIELDS = ("db_error",)


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
    # Kept so older callers that read it still work. A failed lookup now
    # raises LookupUnavailable instead, so this stays None, and it is never
    # sent to the browser either way.
    db_error: str | None = None
    error: str | None = None
    page_number: int = 1
    page_count: int = 1
    # What OCR actually read, kept even after a database match replaces
    # student_id. Without it the UI cannot show "read X, saved Y", which is
    # exactly the comparison someone needs to spot a wrong match.
    ocr_student_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The result as API JSON, without anything about the server's internals."""
        data = asdict(self)
        for name in _INTERNAL_FIELDS:
            data.pop(name, None)
        return data


class ExtractError(Exception):
    """Raised when extraction fails with a user-facing message."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class LookupUnavailable(ExtractError):
    """The student database could not be asked, so there is no answer yet.

    Not the same thing as "not found". Treating an outage as a miss filed
    real students' forms as NOTFOUND_ in Drive, logged them under Not found
    and renamed inbox files so they were never read again. Raising instead
    leaves nothing saved, logged or renamed, so the page can simply be tried
    again once the database is back.
    """

    MESSAGE = "The student database could not be reached. Nothing was saved - try again."

    def __init__(self, message: str = MESSAGE) -> None:
        super().__init__(message, status_code=503)


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


def extract_from_image(image_bgr: np.ndarray, source: str) -> ExtractResult:
    """OCR one page image already in memory, never raising for an unreadable page.

    The scan queue's worker reads pages it has stored itself, so there is
    no file to decode. A page OCR cannot read comes back with an empty
    student_id and `error` set, as extract_one_image does. No database is
    touched.

    Args:
        image_bgr: The upright page, BGR.
        source: How to name the page in the log and in the result.

    Returns:
        The reading.
    """
    try:
        return _extract(image_bgr, source=source)
    except LookupUnavailable:
        raise
    except ExtractError as exc:
        return ExtractResult(
            student_id="",
            confidence=0.0,
            form_language="unknown",
            source=source,
            message=exc.message,
            error=exc.message,
        )


def lookup_student_read_only(
    student_id: str, *, school: str | None = None, location: str | None = None
) -> dict[str, Any] | None:
    """Look a reading up within a school or city, writing nothing.

    What the scan queue's worker records for each page, and what a typed
    Student ID is checked against, before a person confirms the document.
    Only the confirm marks scanned_at.

    Args:
        student_id: The reading.
        school: Only this school's students.
        location: Otherwise only this city's.

    Returns:
        The matched student, or None when there is none.

    Raises:
        LookupUnavailable: The database could not be asked.
    """
    try:
        return find_student_by_id(
            student_id, mark_scanned=False, school=school, location=location
        )
    except DB_ERRORS as exc:
        debug_log(f"[DB] read-only lookup failed: {type(exc).__name__}: {exc}")
        raise LookupUnavailable() from exc


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

    Raises ExtractError (400) for something that is not shaped like a
    Student ID, and LookupUnavailable (503) when the database cannot be
    asked. Neither is logged: a parent's phone number must never land in
    History > Not found, and an outage is not a "not found".
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", student_id or "").upper()
    if not cleaned:
        raise ExtractError("Student ID cannot be empty.", status_code=400)

    # Auto mode and the Drive inbox send every reading straight here with
    # nobody looking, so this is the last place a name, an email address or
    # a phone number read off the form can be stopped before it is saved,
    # uploaded as NOTFOUND_<it> and listed for every member of staff.
    shaped = as_student_id(cleaned)
    if not shaped:
        raise ExtractError(
            "That does not look like a Student ID (letters then digits, e.g. "
            "AADMIS310714). Nothing was saved.",
            status_code=400,
        )
    cleaned = shaped

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
            # The reading, not the match: matched_student_id already holds
            # the match, and History's "read as X" (and any audit of a
            # fuzzy save) needs what was actually read to compare against.
            confirmed_student_id=result.ocr_student_id or result.student_id,
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
    if layout is None:
        # ADF-fed pages often land a degree or two off square, which hides
        # every horizontal line from the detector - measured 0 rules on a
        # page that was only tilted -1.5 deg. Straighten it and try once more,
        # rather than fall through to the 26-second template-guess fallback.
        straightened, angle = deskew(prepared)
        if angle != 0.0:
            debug_log(f"[LAYOUT] first pass empty, retrying after deskew {angle:+.2f} deg")
            layout = identify_form(straightened)
            if layout is not None:
                prepared = straightened
    # Whether to spend a Devanagari pass on this page. The layout is the
    # only trustworthy signal here: _score_label calls a bare ID token
    # "english" even on a Hindi form, so the per-candidate language would
    # switch the pass off exactly where it is needed.
    devanagari_languages = ("hindi", "marathi")
    language_settled = layout is not None and layout.confidence >= LAYOUT_TRUST_CONFIDENCE
    # Whether the rule ladder has pinned the form's lines precisely. The
    # Student ID is on the same rule in every template, so this locates the
    # field even when the language call is a near tie.
    field_located = layout is not None and layout.error <= LAYOUT_TRUST_ERROR
    devanagari_form = layout is not None and (
        layout.language in devanagari_languages
        # A near tie with a Devanagari template: keep that recogniser on,
        # or a Marathi form read as "english" loses its Devanagari IDs.
        or (not language_settled and layout.runner_up_language in devanagari_languages)
    )

    # find_student_id_label runs RapidOCR over the entire page (5-38s) purely
    # to locate the value field. When the rule ladder has already located it,
    # that is work for an answer we have.
    trust_layout = FAST_MODE and layout is not None and (language_settled or field_located)
    label = None if trust_layout else find_student_id_label(prepared)
    if trust_layout:
        debug_log(
            f"[FAST] layout {'confident' if language_settled else 'located the field'} "
            f"(confidence={layout.confidence:.2f}, error={layout.error:.4f}"
            f"{'' if language_settled else ', runner-up=' + str(layout.runner_up_language)}) "
            "- skipped full-page label OCR"
        )
    candidates: list[tuple[str, float, str, str]] = []

    if label is not None:
        label_tag = _label_tag(label)
        debug_log(
            f"[DETECT] label={label_tag!r} language={label.language} "
            f"source={label.source} parsed={loggable_reading(label.parsed_student_id)}"
        )

        # Keep noisy RapidOCR tokens too — better to show the user a rough
        # reading to correct than nothing at all.
        if label.parsed_student_id:
            token = re.sub(r"[^A-Za-z0-9]", "", label.parsed_student_id).upper()
            if looks_like_student_id(token):
                candidates.append((token, 0.92, label.language, f"rapidocr:{label_tag}"))
            elif len(token) >= 8 and token[:4].isalpha() and any(c.isdigit() for c in token):
                debug_log(f"[DETECT] noisy OCR token kept: {loggable_reading(token)}")
                candidates.append(
                    (token, LABEL_LINE_CONFIDENCE, label.language, f"noisy:{label_tag}")
                )

        if label.source == "ocr":
            try:
                crop = crop_student_id_value(prepared, label)
                _save_debug_crop(crop, "label")
                candidates.extend(
                    _read_id_from_crop(
                        crop,
                        label.language,
                        label_tag,
                        allow_devanagari=devanagari_form,
                        # The "label" is whichever line scored best, and an
                        # email line can win. Only a crop beside the Student
                        # ID line itself may leave for Vision.
                        allow_vision=(
                            _label_is_student_id_line(label)
                            and _vision_may_read(crop, prepared, "label")
                        ),
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
    have_clean = FAST_MODE and any(_is_clean_read(row[0]) for row in candidates)
    if have_clean:
        debug_log("[FAST] clean read from the label crop - skipped layout crop")

    if layout is not None and not have_clean:
        try:
            crop = crop_value_roi(prepared, layout.student_id_roi)
            _save_debug_crop(crop, f"layout_{layout.language}")
            # Only a close rule match has put this crop on the Student ID
            # rule. A loose one lands it rows away: page-006 matched English
            # at error 0.0064 (scale 1.08, where every other page matched at
            # 0.85-0.86) and its crop held the parent's mobile number. A
            # clear language call does not count - confidence measures how
            # far the runner-up trails, not where the lines are.
            if not field_located and vision_configured():
                debug_log(
                    f"[VISION] layout match too loose to vouch for its crop "
                    f"(error={layout.error:.4f} > {LAYOUT_TRUST_ERROR}) - local recognisers only"
                )
            candidates.extend(
                _read_id_from_crop(
                    crop,
                    layout.language,
                    f"layout:{layout.language}",
                    allow_devanagari=devanagari_form,
                    allow_vision=field_located and _vision_may_read(crop, prepared, "layout"),
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
                # Read on this machine only, never by Vision. A template box
                # is a fixed page fraction, not a place found on this page.
                # The other languages' boxes fall on other fields by
                # construction - on an English form the Hindi and Marathi
                # ones sit on the parent's email and phone lines - and even
                # the matched form's own box drifts: on the scans in the log
                # (scale 0.85-0.86) the English box starts 0.07-0.10 of the
                # page below the located field, down among the parent's
                # details. Where it does line up, the layout crop above has
                # already been sent.
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
        raise ExtractError(NO_STUDENT_ID_MESSAGE, status_code=422)

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
    confidence. No database is touched here.

    Only a reading with the shape of a Student ID can win (see
    id_shape.as_student_id). A crop that strays onto the parent's details
    reads back as a name, an email address or a mobile number, and each of
    those used to qualify here simply by ending in six digits."""

    # Score additively rather than ranking format above everything. Sorting
    # by (is_strict, confidence) let a 0.02-confidence reading that merely
    # had the right shape beat a 0.65 one that was actually correct - the
    # shape is evidence, not a trump card.
    def score(row: tuple[str, float, str, str]) -> float:
        return row[1] + (STRICT_FORMAT_BONUS if looks_like_student_id(row[0]) else 0.0)

    ranked: list[tuple[float, tuple[str, float, str, str]]] = []
    for row in candidates:
        reading = as_student_id(row[0])
        if reading:
            ranked.append((score(row), (reading, row[1], row[2], row[3])))
    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0], reverse=True)
    debug_log(
        "[OCR] candidates="
        + repr([(row[0], round(points, 3), row[3]) for points, row in ranked])
        + f" -> best={ranked[0][1][0]!r}"
    )
    return ranked[0][1]


def _is_clean_read(text: str) -> bool:
    """A strict letters-then-digits reading that also has a Student ID's shape.

    The fast paths stop looking once they have one of these. A name run into
    a ten-digit mobile number passes the strict check alone, and stopping on
    it would skip the readers that might have found the real ID.
    """
    return looks_like_student_id(text) and has_student_id_shape(text)


def _label_is_student_id_line(label: LabelHit) -> bool:
    """Whether the line the label path settled on is the Student ID line."""
    _language, score = _score_label(label.text)
    return score >= STUDENT_LABEL_SCORE or has_student_id_shape(label.parsed_student_id)


def _label_tag(label: LabelHit) -> str:
    """The label line's text as logs and label_text may carry it.

    The label path settles on whichever line scored best, and that can be
    the email line, or a name under the Grade fallback. label_text goes to
    the browser and into the [OCR] candidates log line, so only a line that
    names the Student ID, a bare ID-shaped line or the synthetic template
    label is quoted; any other is described by its length.
    """
    text = label.text or ""
    _language, score = _score_label(text)
    if label.source == "template" or score >= STUDENT_LABEL_SCORE or as_student_id(text):
        return text
    return f"<not the Student ID line: {len(text)} chars>"


def _vision_may_read(crop_bgr: np.ndarray, page_bgr: np.ndarray, tag: str) -> bool:
    """Whether a crop is small enough to hold only the Student ID line.

    Anything taller has swallowed neighbouring lines - names, an email, a
    phone number - and those must not be sent to Google. The crop is still
    read by the local recognisers either way.
    """
    page_height = max(int(page_bgr.shape[0]), 1)
    share = crop_bgr.shape[0] / page_height
    if share > VISION_MAX_CROP_HEIGHT:
        debug_log(
            f"[VISION] {tag} crop is {share:.3f} of the page height "
            f"(over {VISION_MAX_CROP_HEIGHT}) - local recognisers only"
        )
        return False
    return True


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
    allow_vision: bool = False,
) -> list[tuple[str, float, str, str]]:
    """Run three independent recognizers on a crop and keep every plausible
    reading (strict + noisy) so DB fuzzy repair gets the best shot at
    recovering the real Student ID from messy handwriting.

    allow_vision is the caller vouching that the crop holds the Student ID
    field and nothing else; only then is it sent to Cloud Vision."""
    found: list[tuple[str, float, str, str]] = []

    # 0) Cloud Vision first when configured - it reads handwriting far
    # better than anything local, so a clean result here means the rest of
    # this function can be skipped entirely.
    if allow_vision and vision_configured():
        for text, conf in recognize_with_vision(crop_bgr):
            score = min(conf + VISION_CONFIDENCE_BONUS, 0.99)
            if looks_like_student_id(text):
                found.append((text, score, language, f"crop-vision:{label_text}"))
            elif _is_noisy_id_like(text) or _is_digit_anchored(text):
                found.append(
                    (text, min(score, 0.70), language, f"crop-vision-noisy:{label_text}")
                )
        if FAST_MODE and any(_is_clean_read(row[0]) for row in found):
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
            # The crop may have strayed onto the parent's details.
            debug_log(
                f"[RapidOCR-crop] text={loggable_reading(text)} "
                f"parsed={loggable_reading(parsed)} conf={conf}"
            )
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
    have_clean = any(_is_clean_read(row[0]) for row in found)
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
    # "Usable" means shaped like an ID: junk the shape check will throw
    # away later must not stand in for a reading here.
    if allow_devanagari and not any(as_student_id(row[0]) for row in found):
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
        stamp = now_ist().strftime("%H%M%S")
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

    Raises LookupUnavailable when the database cannot be asked. Reporting
    that as "not found" is what filed real students' forms as not found.
    """
    debug_log(f"[OCR] Student ID found: {result.student_id!r} (confidence={result.confidence})")
    try:
        student = find_student_by_id(
            result.student_id,
            mark_scanned=mark_scanned,
            school=school,
            location=location,
        )
    except DB_ERRORS as exc:
        # The detail stays in the server log. pymysql's message names the
        # database host, user and client address - nothing for a browser.
        debug_log(
            f"[DB] Lookup failed for {result.student_id!r}: {type(exc).__name__}: {exc}"
        )
        raise LookupUnavailable() from exc

    if student is None:
        debug_log(
            f"[DB] No match in active_student_data.student_id for: {result.student_id!r}"
        )
        result.found_in_db = False
        result.student = None
        result.ocr_student_id = result.student_id
        result.message = "student_id_not_found_in_db"
        return result

    result.ocr_student_id = result.student_id
    result.student_id = student.get("student_id") or result.student_id
    result.found_in_db = True
    result.student = student
    result.message = "ok"
    return result
