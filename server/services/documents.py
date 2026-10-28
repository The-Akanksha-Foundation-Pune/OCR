"""Documents: queued pages grouped by their batch code, and confirming them.

A document is every read page one person sent with the same batch code, in
the order they were scanned - usually two parts, the Student ID page and the
signature page. Open documents are not stored anywhere: they are worked out
from the queue each time, so a code typed by hand regroups the pages at
once. Only confirming one writes anything, and confirming is the one place
the scan queue reaches the database or Drive:

  1. the Student ID is looked up within the document's school and the
     student marked scanned (scanned_at),
  2. the document is recorded as confirmed (its pages leave the open list),
  3. its parts become one PDF in the person's own Drive, at
     OCR / <School> / <Grade> / <Division> / <Name>_<ID> / <Name>_<ID>_<Code>.pdf,
  4. the attempt is logged to ocr_scan_log for History, with the PDF's link.

A Drive failure after step 1 leaves the document confirmed, with Retry
upload; the retry repeats only steps 3 and 4. So does a database drop at
step 4 after the upload worked: the document is listed as confirmed, not
uploaded, until its History row is written, and the retry writes only that.

The person confirms what they were shown: the confirm names the parts the
page was showing, and is refused with nothing saved when the document has
gained or lost a part since.
"""

from __future__ import annotations

import io
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import fitz
from PIL import Image

from server.config import (
    DONE_DOCUMENTS_HOURS,
    DRIVE_OCR_FOLDER_NAME,
    QUEUE_CLAIM_STALE_SECONDS,
)
from server.services import drive_store, scan_queue
from server.services.batch_code import parse_batch_code
from server.services.clock import now_ist
from server.services.debug_log import debug_log
from server.services.extract_student_id import (
    ExtractError,
    confirm_student_id,
    lookup_student_read_only,
)
from server.services.history_store import log_scan_attempt, set_attempt_image_link
from server.services.id_shape import as_student_id
from server.services.student_lookup import DB_ERRORS

WAITING = "waiting"
READY = "ready"
NEEDS_ATTENTION = "needs_attention"
CONFIRMED = "confirmed"
UPLOADED = "uploaded"
FAILED = "failed"

# Why a document needs a person, in the order the page should mention them.
NO_ID = "no_id"
NO_MATCH = "no_match"
CONFLICT = "conflict"
CODE_REUSED = "code_reused"

# A page with no marker is only "lost" once nothing has touched its
# meta.json for as long as a reader may hold a claim: every claim, release
# and result rewrites it, so a page a reader is passing between markers is
# always more recent than this. (Measured from when the page arrived, a page
# released and claimed again between the listing's two directory reads was
# queued a second time and read twice at once.)
_ORPHAN_AFTER_SECONDS = QUEUE_CLAIM_STALE_SECONDS

# What the page is told when the parts it showed are not the document's now.
CHANGED_MESSAGE = (
    "This document changed - check it again. A part has been added or taken "
    "away since it was shown. Nothing was saved."
)

# A4's long side, for the PDF page size of a scan that did not say its dpi.
_A4_LONG_SIDE_INCHES = 11.69

# The parts of a matched student that the page and the record need.
_STUDENT_FIELDS = (
    "student_id",
    "student_name",
    "school_name",
    "grade_name",
    "division_name",
    "academic_year",
    "matched_via",
)

ImageUrl = Callable[[str, str], str]


class DocumentError(Exception):
    """A request about a document or page that cannot be done as asked.

    Attributes:
        message: What to tell the person.
        status_code: The HTTP status to answer with.
    """

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class PageView:
    """One queued page as the scan page shows it.

    The code, Student ID and match are what a person typed when they typed
    something, and what OCR read otherwise; `*_source` says which.
    """

    page_id: str
    created_at: str
    received_ns: int
    source_name: str
    source_page: int
    source_pages: int
    school: str
    location: str
    status: str
    attempts: int
    code: str | None = None
    code_source: str | None = None
    code_read: str | None = None
    code_raw: str = ""
    student_id: str = ""
    student_id_source: str | None = None
    student_id_read: str = ""
    matched_student: dict[str, Any] | None = None
    message: str = ""
    orientation: int = 0
    blank: bool = False
    error: str = ""
    in_document: str | None = None
    needs_person: bool = False

    @property
    def order(self) -> tuple[int, int, str]:
        """Scan order: when the file arrived, then the page within it."""
        return (self.received_ns, self.source_page, self.page_id)

    @property
    def unread(self) -> bool:
        """Whether OCR has still to finish with this page."""
        return self.status in (scan_queue.STATUS_QUEUED, scan_queue.STATUS_READING)

    def to_dict(self, image_url: ImageUrl | None = None) -> dict[str, Any]:
        """The page as API JSON."""
        data: dict[str, Any] = {
            "page_id": self.page_id,
            "created_at": self.created_at,
            "source_name": self.source_name,
            "source_page": self.source_page,
            "source_pages": self.source_pages,
            "school": self.school,
            "location": self.location,
            "status": self.status,
            "attempts": self.attempts,
            "code": self.code,
            "code_source": self.code_source,
            "code_read": self.code_read,
            "code_raw": self.code_raw,
            "student_id": self.student_id,
            "student_id_source": self.student_id_source,
            "student_id_read": self.student_id_read,
            "matched_student": self.matched_student,
            "message": self.message,
            "orientation": self.orientation,
            "blank": self.blank,
            "error": self.error,
            "document_key": self.in_document,
            # Unreadable: the page should be shown, full size, for a person
            # to read its code or Student ID off it.
            "needs_person": self.needs_person,
        }
        if image_url is not None:
            data["thumb_url"] = image_url(self.page_id, "thumb")
            data["image_url"] = image_url(self.page_id, "full")
        return data


@dataclass
class Document:
    """Pages sharing one batch code, and where they have got to."""

    key: str
    code: str
    batch: int
    number: int
    pages: list[PageView]
    status: str = READY
    attention: list[str] = field(default_factory=list)
    message: str = ""
    student_id: str = ""
    student_id_source: str | None = None
    matched_student: dict[str, Any] | None = None
    school: str = ""
    location: str = ""
    record: dict[str, Any] | None = None

    @property
    def is_open(self) -> bool:
        """Not yet confirmed."""
        return self.record is None

    @property
    def can_confirm(self) -> bool:
        """Whether Confirm and upload may be pressed now."""
        return self.is_open and self.status in (READY, NEEDS_ATTENTION, FAILED)

    @property
    def history_pending(self) -> bool:
        """Confirmed, but History has not got its row (or the row its link) yet."""
        return self.record is not None and scan_queue.document_record_history_pending(self.record)

    @property
    def latest(self) -> int:
        """When its newest part arrived (for newest-first lists)."""
        return max((page.received_ns for page in self.pages), default=0)

    def to_dict(self, image_url: ImageUrl | None = None) -> dict[str, Any]:
        """The document as API JSON."""
        record = self.record or {}
        drive = record.get("drive") or {}
        return {
            "key": self.key,
            "code": self.code,
            "batch": self.batch,
            "number": self.number,
            "parts": len(self.pages),
            "status": self.status,
            "attention": list(self.attention),
            "message": self.message,
            "student_id": self.student_id,
            "student_id_source": self.student_id_source,
            "matched_student": self.matched_student,
            "school": self.school,
            "location": self.location,
            "created_at": self.pages[0].created_at if self.pages else "",
            "can_confirm": self.can_confirm,
            # Confirmed and not finished: the upload, or only the History
            # row, is still to do. Retry upload does whichever it is.
            "can_retry_upload": self.status == CONFIRMED,
            "history_pending": self.history_pending,
            "confirmed_at": record.get("confirmed_at"),
            "confirmed_by": record.get("confirmed_by"),
            "drive_link": drive.get("link"),
            "drive_name": drive.get("name"),
            "drive_folder": drive.get("folder_path"),
            "upload_error": record.get("upload_error") or "",
            "pages": [page.to_dict(image_url) for page in self.pages],
        }


@dataclass
class Snapshot:
    """Everything one person has in the queue right now."""

    pages: list[PageView]
    documents: list[Document]
    unsorted: list[PageView]

    def find(self, key: str) -> Document | None:
        """A document by key, open or confirmed."""
        return next((doc for doc in self.documents if doc.key == key), None)

    def batches(self, image_url: ImageUrl | None = None) -> list[dict[str, Any]]:
        """Documents by batch, newest batch first, each batch in number order."""
        grouped: dict[int, list[Document]] = {}
        for doc in self.documents:
            grouped.setdefault(doc.batch, []).append(doc)
        out: list[dict[str, Any]] = []
        for batch, docs in sorted(
            grouped.items(), key=lambda item: max(d.latest for d in item[1]), reverse=True
        ):
            docs.sort(key=lambda d: (d.number, not d.is_open, d.latest))
            counts = {name: 0 for name in (WAITING, READY, NEEDS_ATTENTION, CONFIRMED, UPLOADED, FAILED)}
            for doc in docs:
                counts[doc.status] += 1
            counts["documents"] = len(docs)
            counts["pages"] = sum(len(doc.pages) for doc in docs)
            out.append(
                {
                    "batch": batch,
                    "label": f"B{batch}",
                    "documents": [doc.to_dict(image_url) for doc in docs],
                    "counts": counts,
                }
            )
        return out

    def queue_counts(self) -> dict[str, int]:
        """How many of the person's pages are at each stage of OCR."""
        counts = {status: 0 for status in (
            scan_queue.STATUS_QUEUED, scan_queue.STATUS_READING, scan_queue.STATUS_READ, scan_queue.STATUS_FAILED
        )}
        for page in self.pages:
            counts[page.status] = counts.get(page.status, 0) + 1
        return counts


# ------------------------------------------------------------- reading the queue
def load_page_view(
    page_id: str, owner_email: str, state: scan_queue.QueueState | None = None
) -> PageView | None:
    """One page as its owner sees it, or None when it is not theirs."""
    meta = scan_queue.load_meta(page_id)
    owner = scan_queue.normalise_owner(owner_email)
    if meta is None or meta.owner_email != owner:
        return None
    state = state or scan_queue.queue_state()
    status = scan_queue.page_status(meta, state)
    if (
        status in (scan_queue.STATUS_QUEUED, scan_queue.STATUS_READING)
        and page_id not in state.pending
        and page_id not in state.reading
        and scan_queue.seconds_since_update(meta) > _ORPHAN_AFTER_SECONDS
        and scan_queue.requeue_orphan(page_id, seen_updated_at=meta.updated_at)
    ):
        status = scan_queue.STATUS_QUEUED

    result = scan_queue.load_result(page_id) if status == scan_queue.STATUS_READ else None
    overrides = scan_queue.load_overrides(page_id)
    view = PageView(
        page_id=page_id,
        created_at=meta.created_at,
        received_ns=meta.received_ns,
        source_name=meta.source_name,
        source_page=meta.source_page,
        source_pages=meta.source_pages,
        school=meta.school,
        location=meta.location,
        status=status,
        attempts=meta.attempts,
        error=meta.error if status == scan_queue.STATUS_FAILED else "",
    )
    if result is not None:
        view.code_read = result.code
        view.code_raw = result.code_raw
        view.student_id_read = result.student_id
        view.message = result.message
        view.orientation = result.orientation
        view.blank = result.blank
    if status == scan_queue.STATUS_FAILED:
        view.message = (
            f"Could not read this page after {meta.attempts} tries. "
            "Type its code and Student ID from the image."
        )

    if overrides.code:
        view.code, view.code_source = overrides.code, "typed"
    elif result is not None and result.code:
        view.code, view.code_source = result.code, "read"
    if overrides.student_id:
        view.student_id, view.student_id_source = overrides.student_id, "typed"
        view.matched_student = _compact(overrides.matched_student)
    elif result is not None and result.student_id:
        view.student_id, view.student_id_source = result.student_id, "read"
        view.matched_student = _compact(result.matched_student)
    # Narrowed in _judge: a signature page with no ID of its own is fine
    # once another part of its document has one.
    view.needs_person = status == scan_queue.STATUS_FAILED or (
        status == scan_queue.STATUS_READ and (not view.code or not view.student_id)
    )
    return view


def snapshot(owner_email: str) -> Snapshot:
    """One person's pages, documents and unsorted pages, as of now."""
    owner = scan_queue.normalise_owner(owner_email)
    state = scan_queue.queue_state()
    records = scan_queue.load_document_records(owner, since_hours=DONE_DOCUMENTS_HOURS)
    in_records = {pid for record in records for pid in record.get("page_ids") or []}

    open_ids = scan_queue.open_page_ids(owner)
    # A confirm that stopped between recording the document and taking its
    # pages off the open list: finish that step now.
    stray = [pid for pid in open_ids if pid in in_records]
    if stray:
        scan_queue.close_pages(owner, stray)

    open_views = [
        view
        for pid in open_ids
        if pid not in in_records and (view := load_page_view(pid, owner, state)) is not None
    ]
    done = [
        doc for record in records if (doc := _document_from_record(record, owner, state)) is not None
    ]
    every_page = sorted(open_views + [p for doc in done for p in doc.pages], key=lambda p: p.order)

    by_code: dict[str, list[PageView]] = {}
    unsorted: list[PageView] = []
    for view in open_views:
        if view.code:
            by_code.setdefault(view.code, []).append(view)
        elif view.status in (scan_queue.STATUS_READ, scan_queue.STATUS_FAILED):
            unsorted.append(view)

    position = {page.page_id: index for index, page in enumerate(every_page)}
    open_docs = []
    for code, parts in by_code.items():
        parts.sort(key=lambda p: p.order)
        doc = _open_document(code, parts)
        for part in parts:
            part.in_document = doc.key
        _judge(doc, every_page, position, done)
        open_docs.append(doc)

    unsorted.sort(key=lambda p: p.order, reverse=True)
    return Snapshot(
        pages=sorted(every_page, key=lambda p: p.order, reverse=True),
        documents=open_docs + done,
        unsorted=unsorted,
    )


def get_document(owner_email: str, key: str) -> Document | None:
    """One document by key - open, or confirmed at any time."""
    if not scan_queue.is_document_key(key):
        return None
    found = snapshot(owner_email).find(key)
    if found is not None or "~" not in key:
        return found
    record = scan_queue.load_document_record(owner_email, key)
    if record is None:
        return None
    return _document_from_record(record, scan_queue.normalise_owner(owner_email), scan_queue.queue_state())


def _open_document(code: str, parts: list[PageView]) -> Document:
    parsed = parse_batch_code(code)
    batch, number = (parsed.batch, parsed.number) if parsed else (0, 0)
    first = parts[0]
    return Document(
        key=code,
        code=code,
        batch=batch,
        number=number,
        pages=parts,
        school=first.school,
        location=first.location,
    )


def _judge(
    doc: Document, every_page: list[PageView], position: dict[str, int], done: list[Document]
) -> None:
    """Work out an open document's Student ID, status and message."""
    matched = {
        part.matched_student["student_id"]: part
        for part in doc.pages
        if part.matched_student and part.matched_student.get("student_id")
    }
    read_ids = {part.student_id for part in doc.pages if part.student_id}
    attention: list[str] = []
    if len(matched) > 1:
        attention.append(CONFLICT)
        doc.message = "The parts match different students: " + ", ".join(sorted(matched)) + "."
    elif matched:
        # A part whose ID matched beats one whose reading did not: the
        # unmatched one is the misread.
        part = next(iter(matched.values()))
        doc.student_id = part.matched_student["student_id"] if part.matched_student else part.student_id
        doc.student_id_source = part.student_id_source
        doc.matched_student = part.matched_student
    elif len(read_ids) > 1:
        attention.append(CONFLICT)
        doc.message = "The parts were read as different Student IDs: " + ", ".join(sorted(read_ids)) + "."
    elif read_ids:
        doc.student_id = next(iter(read_ids))
        source = next(p for p in doc.pages if p.student_id == doc.student_id)
        doc.student_id_source = source.student_id_source
        attention.append(NO_MATCH)
        where = doc.school or (f"{doc.location} schools" if doc.location else "the student list")
        doc.message = f"No student with ID {doc.student_id} at {where}. Check the ID against the paper."
    else:
        attention.append(NO_ID)
        doc.message = "No Student ID was read. Type it from the image."

    earlier = [d for d in done if d.code == doc.code]
    if earlier:
        attention.append(CODE_REUSED)
        who = (earlier[0].matched_student or {}).get("student_name") or earlier[0].student_id
        doc.message = (
            f"{doc.code} was already confirmed for {who}. Check the code on the paper"
            " - this may be a part scanned after that one was confirmed."
        )
    doc.attention = attention

    if doc.student_id and CONFLICT not in attention:
        for part in doc.pages:
            if part.status == scan_queue.STATUS_READ and part.code:
                part.needs_person = False

    last = max(position[p.page_id] for p in doc.pages)
    following = every_page[last + 1] if last + 1 < len(every_page) else None
    waiting_parts = [p for p in doc.pages if p.unread]
    if waiting_parts or (following is not None and following.unread and not following.code):
        # Parts of one form are scanned one after the other, so until the
        # page after this document has been read it may yet be another part.
        doc.status = WAITING
        doc.message = "Waiting for the next page to be read - it may be another part of this form."
    elif any(p.status == scan_queue.STATUS_FAILED for p in doc.pages):
        doc.status = FAILED
        doc.message = "A part could not be read. Check its image and type the Student ID."
    elif attention:
        doc.status = NEEDS_ATTENTION
    else:
        doc.status = READY
        student = doc.matched_student or {}
        doc.message = f"Matched {student.get('student_name') or doc.student_id}."


def _document_from_record(
    record: dict[str, Any], owner: str, state: scan_queue.QueueState
) -> Document | None:
    key = str(record.get("key") or "")
    code = str(record.get("code") or "")
    parsed = parse_batch_code(code)
    if not scan_queue.is_document_key(key) or parsed is None:
        return None
    pages = [
        view
        for pid in record.get("page_ids") or []
        if (view := load_page_view(pid, owner, state)) is not None
    ]
    for page in pages:
        page.in_document = key
    in_drive = record.get("status") == UPLOADED
    # Listed as uploaded only once History has its row as well; until then
    # it is confirmed, with Retry upload to finish it.
    uploaded = scan_queue.document_record_finished(record)
    student = _compact(record.get("student"))
    who = (student or {}).get("student_name") or record.get("student_id")
    if uploaded:
        message = f"Uploaded to Drive for {who}."
    elif in_drive:
        message = (
            f"Uploaded to Drive for {who}, but its History entry could not be written yet. "
            "Use Retry upload to finish."
        )
    else:
        message = (
            "Saved, but the upload to Drive failed"
            + (f": {record['upload_error']}" if record.get("upload_error") else ".")
            + " Use Retry upload."
        )
    return Document(
        key=key,
        code=code,
        batch=parsed.batch,
        number=parsed.number,
        pages=pages,
        status=UPLOADED if uploaded else CONFIRMED,
        message=message,
        student_id=str(record.get("student_id") or ""),
        student_id_source="confirmed",
        matched_student=student,
        school=str(record.get("school") or ""),
        location=str(record.get("location") or ""),
        record=record,
    )


def _compact(student: dict[str, Any] | None) -> dict[str, Any] | None:
    if not student:
        return None
    return {name: student.get(name) or "" for name in _STUDENT_FIELDS}


# ------------------------------------------------------------- a person's corrections
def _owned_open_page(owner_email: str, page_id: str) -> PageView:
    view = load_page_view(page_id, owner_email) if scan_queue.is_page_id(page_id) else None
    if view is None:
        raise DocumentError("No such page.", 404)
    if page_id not in scan_queue.open_page_ids(owner_email):
        raise DocumentError("This page is in a document that has already been confirmed.", 409)
    return view


def set_page_code(owner_email: str, page_id: str, code_text: str, by: str) -> PageView:
    """Set a page's batch code by hand ("" puts back what OCR read).

    Raises:
        DocumentError: 404 not theirs, 409 already confirmed, 400 not a code.
    """
    _owned_open_page(owner_email, page_id)
    text = (code_text or "").strip()
    if not text:
        scan_queue.set_code_override(page_id, None, by)
    else:
        code = parse_batch_code(text)
        if code is None:
            raise DocumentError("That is not a batch code. Write it as B, the batch, a dash and the number, e.g. B1-12.", 400)
        scan_queue.set_code_override(page_id, code.text, by)
    view = load_page_view(page_id, owner_email)
    assert view is not None
    return view


def set_page_student_id(owner_email: str, page_id: str, student_id: str, by: str) -> PageView:
    """Set a page's Student ID by hand ("" puts back what OCR read).

    The ID is looked up straight away, read-only and within the page's
    school, so the person sees whom it matches before confirming.

    Raises:
        DocumentError: 404, 409 or 400 as for set_page_code.
        LookupUnavailable: The database could not be asked; nothing is kept.
    """
    view = _owned_open_page(owner_email, page_id)
    cleaned = re.sub(r"[^A-Za-z0-9]", "", student_id or "").upper()
    if not cleaned:
        scan_queue.set_student_id_override(page_id, None, None, by)
    else:
        shaped = as_student_id(cleaned)
        if not shaped:
            raise DocumentError(
                "That does not look like a Student ID (letters then digits, e.g. AADMIS310714).", 400
            )
        matched = lookup_student_read_only(
            shaped, school=view.school or None, location=view.location or None
        )
        scan_queue.set_student_id_override(page_id, shaped, matched, by)
    updated = load_page_view(page_id, owner_email)
    assert updated is not None
    return updated


# ------------------------------------------------------------- confirming
@dataclass
class ConfirmOutcome:
    """What a confirm (or a retry of its upload) did.

    Attributes:
        document: The document afterwards.
        uploaded: Whether its PDF is in Drive now.
        message: What to tell the person.
        finished: Whether it is in Drive and History both, with nothing
            left for Retry upload to do.
    """

    document: Document
    uploaded: bool
    message: str
    finished: bool = False


def shown_page_ids(value: Any) -> list[str]:
    """The parts a confirm says the person was shown, checked.

    Args:
        value: The request's page_ids: a list of page ids.

    Returns:
        The ids.

    Raises:
        DocumentError: 400 when it is missing, empty, or not page ids.
    """
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and scan_queue.is_page_id(item) for item in value)
        or len(set(value)) != len(value)
    ):
        raise DocumentError(
            "The confirm did not say which parts it was for. Reload the page and "
            "check this document again. Nothing was saved.",
            400,
        )
    return list(value)


def _same_parts(doc: Document, shown: list[str]) -> bool:
    """Whether a document's parts now are exactly the ones the person was shown.

    Order is not compared: the server keeps the parts in scan order whatever
    order the page sent them in, and that order goes into the PDF.
    """
    return sorted(page.page_id for page in doc.pages) == sorted(shown)


def confirm_document(
    owner_email: str, key: str, student_id: str, by: str, page_ids: list[str]
) -> ConfirmOutcome:
    """Confirm an open document: the one place the queue writes anything.

    Args:
        owner_email: The signed-in person, whose document it must be.
        key: The open document's key (its code).
        student_id: The Student ID the person confirmed.
        by: Who is confirming, for the record.
        page_ids: The parts the page was showing when Confirm was pressed.
            Only exactly those are confirmed: a part read since the page's
            last look (a rescan, or another form's corner misread as this
            code) refuses the confirm rather than going into the record and
            the PDF unseen.

    Returns:
        What happened. A Drive failure is not an error here: the student is
        marked and the document left confirmed with Retry upload.

    Raises:
        DocumentError: 404 no such document; 409 already confirmed, still
            being read, changed since it was shown, being confirmed in
            another tab, no Google sign-in, or the student is not in the
            database (nothing saved); 400 not a Student ID, or no parts.
        LookupUnavailable: The database could not be asked (nothing saved).
    """
    owner = scan_queue.normalise_owner(owner_email)
    doc = _open_document_for(owner, key)
    shown = shown_page_ids(page_ids)
    if not _same_parts(doc, shown):
        raise DocumentError(CHANGED_MESSAGE, 409)
    shaped = as_student_id(re.sub(r"[^A-Za-z0-9]", "", student_id or "").upper())
    if not shaped:
        raise DocumentError(
            "That does not look like a Student ID (letters then digits, e.g. AADMIS310714). Nothing was saved.",
            400,
        )
    if not drive_store.has_drive_credentials():
        # Checked before the lookup: marking the student and then being
        # unable to file the form would leave a scanned child with no form.
        raise DocumentError("Sign in with Google to upload to Drive. Nothing was saved.", 409)

    try:
        with scan_queue.confirm_lock(owner, doc.code):
            # Again inside the lock: another tab may have just confirmed it,
            # or the worker read another part with this code meanwhile.
            doc = _open_document_for(owner, key)
            if not _same_parts(doc, shown):
                raise DocumentError(CHANGED_MESSAGE, 409)
            source = f"batch:{doc.code}"
            try:
                result = confirm_student_id(
                    shaped,
                    source=source,
                    defer_log=True,
                    school=doc.school or None,
                    location=doc.location or None,
                )
            except ExtractError as exc:
                # LookupUnavailable is one of these; it goes up as it is.
                if exc.status_code == 503:
                    raise
                raise DocumentError(exc.message, exc.status_code) from exc
            if not result.found_in_db:
                where = doc.school or (f"{doc.location} schools" if doc.location else "the student list")
                raise DocumentError(
                    f"{shaped} is not a student at {where}. Nothing was saved - "
                    "check the ID against the paper and correct it.",
                    409,
                )

            student = result.student or {}
            record: dict[str, Any] = {
                "key": f"{doc.code}~{secrets.token_hex(4)}",
                "code": doc.code,
                "owner_email": owner,
                "page_ids": [page.page_id for page in doc.pages],
                "status": CONFIRMED,
                "student_id": result.student_id,
                "read_as": result.ocr_student_id or shaped,
                "student": student,
                "school": doc.school,
                "location": doc.location,
                "source": source,
                "message": result.message,
                "confirmed_at": now_ist().isoformat(timespec="seconds"),
                "confirmed_by": by,
                "drive": None,
                "upload_error": "",
                "log_id": None,
                "log_pending": True,
            }
            # Recorded the moment the student is marked, before Drive is
            # tried: whatever happens next, this confirm is not lost.
            scan_queue.save_document_record(owner, record)
            scan_queue.close_pages(owner, record["page_ids"])
            _upload_and_log(owner, record)
    except scan_queue.DocumentBusy as exc:
        raise DocumentError("This document is already being confirmed.", 409) from exc

    return _outcome(owner, record)


def retry_upload(owner_email: str, key: str) -> ConfirmOutcome:
    """Finish a confirmed document: its upload, or only its History row.

    Nothing else is redone. A document whose PDF is already in Drive is not
    uploaded again; only its History row (or the link on it) is written.

    Raises:
        DocumentError: 404 no such confirmed document; 409 no Google
            sign-in (when the upload is still to do), or a retry already
            running.
    """
    owner = scan_queue.normalise_owner(owner_email)
    record = scan_queue.load_document_record(owner, key) if "~" in (key or "") else None
    if record is None:
        raise DocumentError("No such confirmed document.", 404)
    if scan_queue.document_record_finished(record):
        return _outcome(owner, record)
    if record.get("status") != UPLOADED and not drive_store.has_drive_credentials():
        raise DocumentError("Sign in with Google to upload to Drive.", 409)
    try:
        with scan_queue.confirm_lock(owner, str(record["code"])):
            record = scan_queue.load_document_record(owner, key) or record
            if not scan_queue.document_record_finished(record):
                _upload_and_log(owner, record)
    except scan_queue.DocumentBusy as exc:
        raise DocumentError("This document is already being uploaded.", 409) from exc
    return _outcome(owner, record)


def _open_document_for(owner: str, key: str) -> Document:
    if not scan_queue.is_document_key(key):
        raise DocumentError("No such document.", 404)
    found = snapshot(owner).find(key)
    if found is None:
        raise DocumentError("No such document. It may have just been confirmed.", 404)
    if not found.is_open:
        raise DocumentError("This document has already been confirmed.", 409)
    if found.status == WAITING:
        raise DocumentError("Some of this document is still being read. Try again in a moment.", 409)
    return found


def _outcome(owner: str, record: dict[str, Any]) -> ConfirmOutcome:
    doc = _document_from_record(record, owner, scan_queue.queue_state())
    assert doc is not None
    return ConfirmOutcome(
        document=doc,
        uploaded=record.get("status") == UPLOADED,
        message=doc.message,
        finished=scan_queue.document_record_finished(record),
    )


def drive_location(record: dict[str, Any]) -> tuple[list[str], str]:
    """The folders and file name a confirmed document goes to in Drive.

    OCR / <School> / <Grade> / <Division> / <Name>_<ID> / <Name>_<ID>_<Code>.pdf,
    from the matched student's own row in active_student_data.
    """
    student = record.get("student") or {}
    student_id = str(record.get("student_id") or student.get("student_id") or "")
    name = drive_store.drive_safe_name(str(student.get("student_name") or ""), "Unnamed")
    person = drive_store.drive_safe_name(f"{name}_{student_id}", student_id or "Unnamed")
    folders = [
        drive_store.drive_safe_name(DRIVE_OCR_FOLDER_NAME, "OCR"),
        drive_store.drive_safe_name(str(student.get("school_name") or record.get("school") or ""), "Unknown school"),
        drive_store.drive_safe_name(str(student.get("grade_name") or ""), "Unknown grade"),
        drive_store.drive_safe_name(str(student.get("division_name") or ""), "Unknown division"),
        person,
    ]
    file_name = drive_store.drive_safe_name(f"{person}_{record.get('code')}", person) + ".pdf"
    return folders, file_name


def build_pdf(page_ids: list[str]) -> bytes:
    """One PDF of a document's parts, in order, each page at its scanned size.

    page.jpg goes in as it is (no re-encoding). A page that did not say its
    resolution is sized as A4.

    Raises:
        ValueError: A part's image is missing.
    """
    pdf = fitz.open()
    try:
        for page_id in page_ids:
            path = scan_queue.page_dir(page_id) / scan_queue.PAGE_IMAGE
            if not path.is_file():
                raise ValueError(f"page {page_id} has no image")
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
            meta = scan_queue.load_meta(page_id)
            dpi = (meta.dpi if meta and meta.dpi else None) or max(width, height) / _A4_LONG_SIDE_INCHES
            page = pdf.new_page(width=width * 72.0 / dpi, height=height * 72.0 / dpi)
            page.insert_image(page.rect, stream=data)
        return pdf.tobytes(garbage=3, deflate=True)
    finally:
        pdf.close()


def _upload_and_log(owner: str, record: dict[str, Any]) -> None:
    """Finish a confirmed document: upload its PDF unless that is done, then log it.

    Logging writes its History row, or adds the link to the row written
    before the upload worked. Whatever fails is left for Retry upload.
    """
    if record.get("status") != UPLOADED:
        _upload(owner, record)

    link = (record.get("drive") or {}).get("link") or None
    try:
        if record.get("log_pending"):
            student = record.get("student") or {}
            record["log_id"] = log_scan_attempt(
                confirmed_student_id=str(record.get("read_as") or record.get("student_id") or ""),
                found_in_db=True,
                matched_student_id=student.get("student_id"),
                student_name=student.get("student_name"),
                school_name=student.get("school_name"),
                source=str(record.get("source") or f"batch:{record.get('code')}"),
                message=str(record.get("message") or "ok"),
                image_link=link,
                matched_via=student.get("matched_via"),
                scan_school=record.get("school") or None,
                scan_location=record.get("location") or None,
            )
            record["log_pending"] = False
            record["log_has_link"] = bool(link)
        elif link and record.get("log_id") and not record.get("log_has_link"):
            set_attempt_image_link(int(record["log_id"]), link)
            record["log_has_link"] = True
    except DB_ERRORS as exc:
        # The student is marked and the PDF (perhaps) filed; the History row
        # is written by Retry upload rather than failing the confirm - the
        # document stays confirmed, not uploaded, until it is.
        record["log_error"] = "Could not write the History entry yet."
        debug_log(f"[DOCS] {record['key']}: could not log to History yet: {type(exc).__name__}: {exc}")
    else:
        record["log_error"] = ""
    scan_queue.save_document_record(owner, record)


def _upload(owner: str, record: dict[str, Any]) -> None:
    """Upload a confirmed document's PDF to the signed-in person's Drive.

    The folder chain is found or made under the person's Drive-folders lock,
    so two confirms for one class at once share one set of folders.
    """
    try:
        folders, file_name = drive_location(record)
        pdf = build_pdf(list(record.get("page_ids") or []))
        upload = drive_store.upload_document_pdf(
            pdf, folders, file_name, folder_lock=lambda: scan_queue.drive_folders_lock(owner)
        )
    except drive_store.DriveUploadError as exc:
        record["upload_error"] = str(exc)
        debug_log(f"[DOCS] {record['key']}: upload failed: {exc}")
    except scan_queue.DocumentBusy:
        record["upload_error"] = "Another upload to your Drive took too long. Try again."
        debug_log(f"[DOCS] {record['key']}: Drive folders lock not free in time")
    except (OSError, ValueError, RuntimeError) as exc:
        record["upload_error"] = "Could not make the PDF of this document."
        debug_log(f"[DOCS] {record['key']}: PDF failed: {type(exc).__name__}: {exc}")
    else:
        record["drive"] = {
            "file_id": upload.file_id,
            "name": upload.name,
            "link": upload.link,
            "folder_path": upload.folder_path,
        }
        record["status"] = UPLOADED
        record["upload_error"] = ""
