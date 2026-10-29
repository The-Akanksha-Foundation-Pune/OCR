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

With batch codes off (config.batch_codes_on) the pages are gathered by scan
order instead, inside runs: the pages one scan page sent (PageMeta.sender),
from its scan folder or added by hand, each within RUN_GAP_MINUTES of the
one before (_runs). Each form is two sheets: the page with its Student ID -
read by OCR or typed by a person - and the Declaration page scanned
straight after it, which has none (_pair). The two are one document, keyed
"P" and the Student ID page's id, and go to Drive as one PDF,
<Name>_<ID>.pdf, the Student ID page first. A form whose first page has no
Student ID waits under Unsorted, its Declaration page with it, until one is
typed. A Declaration page that joins no form waits under Unsorted on its
own, and holds every form of its run for a person: the stack may be out of
order or missing a page. So does a form's first page with no Declaration
page between it and the next form's (_check_steps), and a held run holds
the run that may be the rest of its stack (_carry_on). A Student ID page
with nothing after it yet waits a little for its Declaration page before it
can be confirmed alone. A Declaration page the reader did not recognise can
be attached to the form straight before it by a person
(set_page_declaration). A code read while codes were on is not looked at,
and documents confirmed with a code are still listed.

With Upload matched pages automatically on, the scan page confirms some
documents itself, nobody pressing Confirm: those listed as auto_upload -
Ready, with batch codes off its one Declaration page with it, recognised by
the reader and scanned straight after it from the same scan page and the
same way, in a run with nothing out of place, once every page scanned
before it has been read and the page after its Declaration page has too
and is the next form's Student ID page - or nothing has come after it for
DECLARATION_WAIT_SECONDS (_unsettled); the Student ID read or typed exactly
the matched student's, scanned for one school rather than a whole city,
every page sent with the switch on, no other document for the same student
and the student not yet marked scanned (auto_upload_refusal). It is the
same confirm, made in the person's own session because the PDF goes to
their Drive. It checks all of that again, is refused with nothing saved if
anything has changed, looks the student up by the exact ID only and marks
only one not yet marked scanned. Its record and History row say it was
automatic. Everything else still waits for a person.
"""

from __future__ import annotations

import io
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import fitz
from PIL import Image

from server.config import (
    DECLARATION_WAIT_SECONDS,
    DONE_DOCUMENTS_HOURS,
    DRIVE_OCR_FOLDER_NAME,
    QUEUE_CLAIM_STALE_SECONDS,
    RUN_GAP_MINUTES,
    auto_upload_on,
    batch_codes_on,
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
# Batch codes off: the student matched already has a form confirmed lately.
SAME_STUDENT = "same_student"
# Batch codes off: no Declaration page came after the Student ID page.
NO_DECLARATION = "no_declaration"
# Batch codes off: why every form of a run is held for a person, as its
# forms may be paired wrongly (_pair). Its run has a Declaration page that is
# not with its form; or a form's first page with no Declaration page between
# it and the next form's (_check_steps); or it may be the rest of a stack
# whose run before it is held (_carry_on).
ORPHAN_IN_STACK = "orphan_in_stack"
GAP_IN_STACK = "gap_in_stack"
AFTER_HELD_STACK = "after_held_stack"

# Batch codes off: what a Declaration page of no form says under Unsorted
# (_no_form) - only what is known, never which form it might be.
_NOTHING_BEFORE = "A Declaration page with no Student ID page right before it."
_AFTER_DECLARATION = (
    "A Declaration page right after another Declaration page - a form has one,"
    " so the Student ID page before this one may be missing."
)
_AFTER_CONFIRMED = "A Declaration page that came after a form that was already confirmed."
_OTHER_SCHOOL = "A Declaration page sent for another school or city than the page before it."
# ... what every form of a held run says, by why it is held ...
_HELD_MESSAGES = {
    ORPHAN_IN_STACK: (
        "This stack has a Declaration page without its Student ID page, so its forms"
        " may be paired wrongly - check them."
    ),
    GAP_IN_STACK: (
        "This stack has a page with no Declaration page between it and the next form,"
        " so its forms may be paired wrongly - check them."
    ),
    AFTER_HELD_STACK: (
        "This stack may go on from one scanned before it that has a page out of place,"
        " so its forms may be paired wrongly - check them."
    ),
}
# ... and why none of them goes up by itself (auto_upload_refusal).
_HELD_REASONS = {
    ORPHAN_IN_STACK: "its stack has a Declaration page without its Student ID page",
    GAP_IN_STACK: "its stack has a page with no Declaration page between it and the next form",
    AFTER_HELD_STACK: "its stack may go on from one scanned before it that has a page out of place",
}
_RUN_GAP_NS = RUN_GAP_MINUTES * 60 * 1_000_000_000
_DECLARATION_WAIT_NS = DECLARATION_WAIT_SECONDS * 1_000_000_000

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

# The parts of a matched student that the page and the record need. scanned_at
# is as the lookup found it: a child already marked scanned never goes up by
# itself (auto_upload_refusal).
_STUDENT_FIELDS = (
    "student_id",
    "student_name",
    "school_name",
    "grade_name",
    "division_name",
    "academic_year",
    "matched_via",
    "scanned_at",
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
    skip_reason: str = ""
    error: str = ""
    in_document: str | None = None
    needs_person: bool = False
    # Sent with Upload matched pages automatically on (PageMeta.auto_upload).
    auto_upload: bool = False
    # The scan page that sent it (PageMeta.sender); "" for none.
    sender: str = ""
    # Batch codes off: the page a person attached this one to as its
    # Declaration page, and who (set_page_declaration); "" for none.
    declaration_of: str = ""
    declaration_by: str = ""
    # Batch codes off, under Unsorted: the Declaration page of this page's
    # form, shown with it until its Student ID is typed (_documents_by_page).
    declaration_pages: list[PageView] = field(default_factory=list)
    # Batch codes off, under Unsorted: the page scanned straight before this
    # one, whose form's Declaration page this may be - one the reader missed
    # - and that form's student (_offer); "" for none.
    attach_to: str = ""
    attach_name: str = ""

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
        if not batch_codes_on():
            # Without batch codes the page is told why a page was skipped:
            # a Declaration page is shown as its form's second sheet, and a
            # skipped page of no form offers the Student ID box - all but a
            # blank one - in case the skip was wrong.
            data["skip_reason"] = self.skip_reason
            # Attached by a person as its form's Declaration page: that form
            # never goes up by itself. A Student ID typed on it since makes
            # it a page of its own again.
            data["declaration_by"] = self.declaration_by if self.student_id_source != "typed" else ""
            # A page waiting under Unsorted for its Student ID comes with
            # its form's Declaration page: once the ID is typed, both are
            # one document.
            data["declaration_pages"] = [page.to_dict(image_url) for page in self.declaration_pages]
            # Under Unsorted, a page that may be the Declaration page of the
            # form straight before it: the page offers to attach it.
            data["attach_to"] = (
                {"page_id": self.attach_to, "student_name": self.attach_name} if self.attach_to else None
            )
        if image_url is not None:
            data["thumb_url"] = image_url(self.page_id, "thumb")
            data["image_url"] = image_url(self.page_id, "full")
        return data


@dataclass
class Document:
    """Pages sharing one batch code (one form, with codes off), and where they have got to."""

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
    # What only the person's other documents and pages show, set by snapshot
    # for auto_upload_refusal: another document, open or confirmed lately,
    # matched the same student; with batch codes on, some pages are still to
    # be read and may be more of its parts.
    shares_student: bool = False
    pages_to_read: bool = False
    # Batch codes off, set by _documents_by_page: why its run is held, if it
    # is (ORPHAN_IN_STACK, GAP_IN_STACK or AFTER_HELD_STACK; _pair); how many
    # pages of any kind came between its Student ID page and its Declaration
    # page in its run; whether a page scanned before it - in its run, or in
    # a run it may go on from - is still to be read; and the page straight
    # after its Declaration page: the next in its run, or with none there the
    # first page of each run that may go on from it (_carry_on) - empty while
    # nothing has come after it.
    held: str = ""
    between: int = 0
    earlier_unread: bool = False
    after: list[PageView] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        """Not yet confirmed."""
        return self.record is None

    @property
    def can_confirm(self) -> bool:
        """Whether Confirm and upload may be pressed now."""
        return self.is_open and self.status in (READY, NEEDS_ATTENTION, FAILED)

    @property
    def auto_upload(self) -> bool:
        """Whether the scan page may confirm it by itself (auto_upload_refusal)."""
        return not auto_upload_refusal(self)

    @property
    def auto_upload_pending(self) -> bool:
        """Not auto_upload yet, but it may be once its stack has settled (batch codes off, _unsettled)."""
        return not batch_codes_on() and not _lasting_refusal(self) and bool(_unsettled(self))

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
        data: dict[str, Any] = {
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
            # With its switch on, the scan page confirms this one by itself,
            # once; everything else waits for a person.
            "auto_upload": self.auto_upload,
            # Confirmed by the scan page itself, not by someone pressing
            # Confirm - the list says it went up automatically.
            "confirmed_automatically": record.get("automatic") is True,
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
        if not batch_codes_on():
            # Not going up by itself yet only because its stack has still to
            # settle - the page after its Declaration page to come or be
            # read, say: the scan page keeps looking until it is.
            data["auto_upload_pending"] = self.auto_upload_pending
        return data


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
        """Documents by batch, newest batch first, each batch in number order.

        With batch codes off there are no batches: every document is in the
        one list, with no batch or label, newest scan first. With them on, a
        document confirmed while they were off has no code, so no batch
        either: those are listed apart, with no batch or label.
        """
        if not batch_codes_on():
            docs = sorted(
                self.documents,
                key=lambda d: max((page.order for page in d.pages), default=(0, 0, "")),
                reverse=True,
            )
            return [_listing(None, "", docs, image_url)] if docs else []
        grouped: dict[int | None, list[Document]] = {}
        for doc in self.documents:
            grouped.setdefault(doc.batch if doc.code else None, []).append(doc)
        out: list[dict[str, Any]] = []
        for batch, docs in sorted(
            grouped.items(), key=lambda item: max(d.latest for d in item[1]), reverse=True
        ):
            docs.sort(key=lambda d: (d.number, not d.is_open, d.latest))
            out.append(_listing(batch, "" if batch is None else f"B{batch}", docs, image_url))
        return out

    def queue_counts(self) -> dict[str, int]:
        """How many of the person's pages are at each stage of OCR."""
        counts = {status: 0 for status in (
            scan_queue.STATUS_QUEUED, scan_queue.STATUS_READING, scan_queue.STATUS_READ, scan_queue.STATUS_FAILED
        )}
        for page in self.pages:
            counts[page.status] = counts.get(page.status, 0) + 1
        return counts


def _listing(
    batch: int | None, label: str, docs: list[Document], image_url: ImageUrl | None
) -> dict[str, Any]:
    """One group of documents as GET /api/batches lists it, with its counts."""
    counts = {name: 0 for name in (WAITING, READY, NEEDS_ATTENTION, CONFIRMED, UPLOADED, FAILED)}
    for doc in docs:
        counts[doc.status] += 1
    counts["documents"] = len(docs)
    counts["pages"] = sum(len(doc.pages) for doc in docs)
    return {
        "batch": batch,
        "label": label,
        "documents": [doc.to_dict(image_url) for doc in docs],
        "counts": counts,
    }


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
        auto_upload=meta.auto_upload is True,
        sender=str(meta.sender) if scan_queue.is_sender(meta.sender) else "",
    )
    if overrides.declaration_of:
        view.declaration_of, view.declaration_by = overrides.declaration_of, overrides.declaration_by
    codes = batch_codes_on()
    if result is not None:
        view.code_read = result.code
        view.code_raw = result.code_raw
        view.student_id_read = result.student_id
        view.message = result.message
        view.orientation = result.orientation
        view.blank = result.blank
        view.skip_reason = result.skip_reason
    if status == scan_queue.STATUS_FAILED:
        view.message = f"Could not read this page after {meta.attempts} tries. " + (
            "Type its code and Student ID from the image." if codes else "Type its Student ID from the image."
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
    # once another part of its document has one. Without batch codes only
    # the Student ID is wanted, and a Declaration page in a form never has
    # one (_documents_by_page).
    view.needs_person = status == scan_queue.STATUS_FAILED or (
        status == scan_queue.STATUS_READ and (not view.student_id or (codes and not view.code))
    )
    return view


def snapshot(owner_email: str) -> Snapshot:
    """One person's pages, documents and unsorted pages, as of now."""
    owner = scan_queue.normalise_owner(owner_email)
    state = scan_queue.queue_state()
    # The open list before the records: a confirm saves its record and only
    # then takes its pages off the list, so a document being confirmed now
    # is seen either open or confirmed - never neither, which would let an
    # automatic confirm of another document for the same student through.
    open_ids = scan_queue.open_page_ids(owner)
    records, older = scan_queue.document_records(owner, since_hours=DONE_DOCUMENTS_HOURS)
    in_records = {pid for record in records + older for pid in record.get("page_ids") or []}

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

    if batch_codes_on():
        open_docs, unsorted = _documents_by_code(open_views, every_page, done)
    else:
        # The pairing needs every page that may share a run with an open
        # one, however long ago its document was confirmed.
        beside = _pages_beside(open_views, every_page, older, owner, state)
        firsts = {str(ids[0]) for record in records + older if (ids := record.get("page_ids"))}
        open_docs, unsorted = _documents_by_page(
            open_views, sorted(every_page + beside, key=lambda p: p.order), done, firsts
        )
    _note_others(open_docs, done, open_views)

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


def _documents_by_code(
    open_views: list[PageView], every_page: list[PageView], done: list[Document]
) -> tuple[list[Document], list[PageView]]:
    """The open documents, one per batch code, and the pages with no code."""
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
    return open_docs, unsorted


def _documents_by_page(
    open_views: list[PageView], every_page: list[PageView], done: list[Document], firsts: set[str]
) -> tuple[list[Document], list[PageView]]:
    """With batch codes off: each form with a Student ID is a document of its own.

    A form is its first page and the Declaration page paired with it in its
    run (_pair): one document, keyed by that first page, its parts in scan
    order. It is one once OCR is done with the first page (read, or failed)
    and the page has a Student ID, read or typed. A skipped page - a blank
    page, a declaration page or an empty Student ID line - has none read off
    it, so it is one only once a person types its ID from the scan: the skip
    may have been wrong. Every other first page OCR is done with waits under
    Unsorted for its Student ID, its Declaration page with it rather than on
    a card of its own, or offered as the Declaration page of the form
    before it (_offer). A Declaration page of no form waits under Unsorted
    by itself and says why; one whose page before is still to be read is
    listed with that page's form once it is. A code read while batch codes
    were on is not looked at.

    Args:
        open_views: The person's open pages.
        every_page: Those and every confirmed page their runs may hold, in
            scan order.
        done: The person's confirmed documents still listed.
        firsts: The first page of each of the person's confirmed documents.
    """
    open_ids = {view.page_id for view in open_views}
    runs = _runs(every_page)
    pairing = _pair(runs, open_ids, firsts)
    now = time.time_ns()
    open_docs: list[Document] = []
    unsorted: list[PageView] = []
    for view in open_views:
        if view.status not in (scan_queue.STATUS_READ, scan_queue.STATUS_FAILED):
            continue
        if view.page_id in pairing.form_of or view.page_id in pairing.pending:
            # Shown with its form - once the page before it is read, for one
            # still waiting for that.
            continue
        if view.page_id in pairing.orphans:
            view.message = pairing.orphans[view.page_id]
            unsorted.append(view)
            continue
        number = pairing.run_of[view.page_id]
        run = runs[number]
        at = pairing.index_of[view.page_id]
        declaration = pairing.declaration.get(view.page_id)
        parts = [declaration] if declaration else []
        for part in parts:
            # Its form's Student ID is the first page's to give.
            part.needs_person = False
        if view.student_id and (not view.skip_reason or view.student_id_source == "typed"):
            doc = _page_document([view, *parts])
            for page in doc.pages:
                page.in_document = doc.key
            doc.held = pairing.held.get(number, "")
            doc.earlier_unread = number in pairing.unsettled or any(page.unread for page in run[:at])
            if declaration is not None:
                end = pairing.index_of[declaration.page_id]
                doc.between = end - at - 1
                doc.after = run[end + 1 : end + 2] or pairing.later.get(number, [])
            _judge_page(doc, done, "" if parts else _waiting_for_declaration(view, run, at, now))
            open_docs.append(doc)
        else:
            view.declaration_pages = parts
            target = pairing.offered.get(view.page_id)
            if target is not None:
                view.attach_to = target.page_id
                view.attach_name = str((target.matched_student or {}).get("student_name") or "")
            unsorted.append(view)
    return open_docs, unsorted


def _pages_beside(
    open_views: list[PageView],
    known: list[PageView],
    older: list[dict[str, Any]],
    owner: str,
    state: scan_queue.QueueState,
) -> list[PageView]:
    """With batch codes off: the confirmed pages no longer listed that may share a run with an open one.

    A run never spans RUN_GAP_MINUTES with no page of the person's arriving,
    whoever sent it, so only a page in such a stretch of time with an open
    page in it can be in that page's run. A page id begins with when the
    page arrived (scan_queue.new_page_id), so the stretches are found from
    the ids alone; only the pages in them are read.

    Args:
        open_views: The person's open pages.
        known: Every page already read for the snapshot, open ones too.
        older: The confirmed documents no longer listed.
        owner: Whose.
        state: The queue's markers, as the snapshot read them.
    """
    known_ids = {page.page_id for page in known}
    unlisted = {
        pid for record in older for pid in record.get("page_ids") or [] if scan_queue.is_page_id(pid)
    } - known_ids
    if not unlisted or not open_views:
        return []
    open_ids = {view.page_id for view in open_views}
    gap_ms = RUN_GAP_MINUTES * 60 * 1000
    wanted: list[str] = []
    stretch: list[str] = []
    with_open = False
    last = 0
    for arrived, pid in sorted((int(pid[:12], 16), pid) for pid in unlisted | known_ids):
        if stretch and arrived - last > gap_ms:
            if with_open:
                wanted += stretch
            stretch, with_open = [], False
        stretch.append(pid)
        with_open = with_open or pid in open_ids
        last = arrived
    if with_open:
        wanted += stretch
    return [
        view for pid in wanted if pid in unlisted and (view := load_page_view(pid, owner, state)) is not None
    ]


def _runs(pages: list[PageView]) -> list[list[PageView]]:
    """With batch codes off: a person's pages cut into runs, each in scan order.

    A run is one stack as it came in: the pages one scan page sent - from
    its scan folder or added by hand, which it marks (_scan_page); those
    sent with none count as one sender - each reaching the server within
    RUN_GAP_MINUTES of the one before it. The pages of one file arrive
    together, so they are always in one run.

    Args:
        pages: The pages, in scan order.
    """
    runs: list[list[PageView]] = []
    current: dict[str, list[PageView]] = {}
    for page in pages:
        sender = _scan_page(page.sender)
        run = current.get(sender)
        if run is None or page.received_ns - run[-1].received_ns > _RUN_GAP_NS:
            run = []
            runs.append(run)
            current[sender] = run
        run.append(page)
    return runs


def _scan_page(sender: str) -> str:
    """The scan page a sender id is from: a file added by hand carries its id with "-1" after it."""
    return sender.split("-", 1)[0]


@dataclass
class _Pairing:
    """How a person's pages pair up in their runs (batch codes off, _pair)."""

    # A form's first page id -> its Declaration page.
    declaration: dict[str, PageView] = field(default_factory=dict)
    # A Declaration page id -> its form's first page.
    form_of: dict[str, PageView] = field(default_factory=dict)
    # An open Declaration page of no form -> what it says under Unsorted.
    orphans: dict[str, str] = field(default_factory=dict)
    # The open Declaration pages whose page before is still to be read.
    pending: set[str] = field(default_factory=set)
    # An open page under Unsorted that may be the Declaration page the
    # reader missed -> the page straight before it, whose form it may be of.
    offered: dict[str, PageView] = field(default_factory=dict)
    # Each page id -> which run it is in, and where in that run.
    run_of: dict[str, int] = field(default_factory=dict)
    index_of: dict[str, int] = field(default_factory=dict)
    # The runs that are held, each with why: ORPHAN_IN_STACK, GAP_IN_STACK
    # or AFTER_HELD_STACK. Every form in one may be paired wrongly.
    held: dict[int, str] = field(default_factory=dict)
    # The runs that may go on from one with a page still to be read, and for
    # each run the first page of every run that may go on from it (_carry_on).
    unsettled: set[int] = field(default_factory=set)
    later: dict[int, list[PageView]] = field(default_factory=dict)


def _pair(runs: list[list[PageView]], open_ids: set[str], firsts: set[str]) -> _Pairing:
    """With batch codes off: pair each Declaration page with the form before it, inside its run.

    Each form is two sheets scanned one straight after the other: the page
    with its Student ID, then its Declaration page, which has none. So an
    open Declaration page joins the page before it in its run - blank pages
    passed over - when that page is open, not a Declaration page itself and
    sent for the same school and city. A form has one: a second Declaration
    page in a row joins nothing. One whose page before is still being read
    waits for it. Any other is an orphan and holds its run, because a stack
    fed Declaration-first, or missing a Student ID page, pairs every form
    after that wrongly with nothing else to show it. So does a page the
    reader found to be a Declaration page that is, or went up as, a form of
    its own - a Student ID typed on it: the stack is out of place there
    too, whatever was typed. A run is held, too, for a form's first page
    with no Declaration page between it and the next form's (_check_steps),
    and for a held run it may be the rest of (_carry_on).

    Args:
        runs: The person's runs (_runs).
        open_ids: The open pages; every other is confirmed.
        firsts: The first page of each confirmed document.
    """
    pairing = _Pairing()
    for number, run in enumerate(runs):
        _drop_stale_attachments(run, open_ids)
        before: PageView | None = None  # the last page that is not blank
        for at, page in enumerate(run):
            pairing.run_of[page.page_id] = number
            pairing.index_of[page.page_id] = at
            if page.unread:
                # It may yet be a form's first page.
                before = page
                continue
            if _is_blank(page):
                continue
            if not _is_declaration(page):
                # Found by the reader itself to be a Declaration page, with a
                # Student ID typed on it since (one attached by a person is
                # theirs to take back).
                if page.skip_reason == "declaration" and (page.page_id in open_ids or page.page_id in firsts):
                    pairing.held[number] = ORPHAN_IN_STACK
                before = page
                continue
            last, before = before, page
            if page.page_id not in open_ids:
                # Confirmed: it went with its form, unless it went first.
                if page.page_id in firsts:
                    pairing.held[number] = ORPHAN_IN_STACK
                continue
            if last is not None and last.unread:
                pairing.pending.add(page.page_id)
                continue
            why = _no_form(page, last, open_ids)
            if why:
                pairing.orphans[page.page_id] = why
                pairing.held[number] = ORPHAN_IN_STACK
            elif last is not None:
                pairing.declaration[last.page_id] = page
                pairing.form_of[page.page_id] = last
        _offer(pairing, run, open_ids)
        _check_steps(pairing, number, run, open_ids, firsts)
    _carry_on(pairing, runs)
    return pairing


def _drop_stale_attachments(run: list[PageView], open_ids: set[str]) -> None:
    """For this look, set aside an attachment by hand whose form is no longer straight before it (batch codes off).

    A person attaches a page to the form scanned straight before it
    (set_page_declaration). Should the page before it stop beginning a
    form - a Student ID typed on a blank page, then taken off again - the
    attached page is a page of its own again, rather than joining whatever
    is before it now. A confirmed page went with its form as it was.
    """
    for at, page in enumerate(run):
        if page.declaration_of and page.page_id in open_ids and not (
            at and run[at - 1].page_id == page.declaration_of and _may_take(run[at - 1], page, open_ids)
        ):
            page.declaration_of = page.declaration_by = ""


def _may_take(before: PageView, page: PageView, open_ids: set[str]) -> bool:
    """Whether `page` may be the Declaration page of the open form `before` begins (batch codes off).

    `before` is open, OCR is done with it, it is neither blank nor a
    Declaration page, and it was sent for the same school and city.
    """
    return (
        before.page_id in open_ids
        and before.status in (scan_queue.STATUS_READ, scan_queue.STATUS_FAILED)
        and not _is_blank(before)
        and not _is_declaration(before)
        and (before.school, before.location) == (page.school, page.location)
    )


def _offer(pairing: _Pairing, run: list[PageView], open_ids: set[str]) -> None:
    """Offer an Unsorted page as the Declaration page of the form straight before it (batch codes off).

    The reader can miss a Declaration page's heading on a crooked or faint
    scan and take it for a page whose Student ID could not be read. When the
    page scanned straight before it in its run - nothing between, blank
    included - begins an open form that has no Declaration page, sent for
    the same school and city, a person may attach it to that form
    (set_page_declaration). A page with a Student ID, a blank page, a
    Declaration page, or one with a Declaration page of its own is never
    offered; nor is one straight after a page that is offered itself, which
    is the likelier Declaration page of the two.
    """
    for at in range(1, len(run)):
        page, before = run[at], run[at - 1]
        if (
            page.page_id in open_ids
            and page.status in (scan_queue.STATUS_READ, scan_queue.STATUS_FAILED)
            and not page.student_id
            and not _is_blank(page)
            and not _found_declaration(page)
            and page.page_id not in pairing.declaration
            and _may_take(before, page, open_ids)
            and before.page_id not in pairing.declaration
            and before.page_id not in pairing.offered
        ):
            pairing.offered[page.page_id] = before


def _check_steps(
    pairing: _Pairing, number: int, run: list[PageView], open_ids: set[str], firsts: set[str]
) -> None:
    """Hold a run whose sheets do not go a form's first page, its Declaration page, and on (batch codes off).

    Two forms' first pages one after the other - blank pages passed over -
    leave the first with no Declaration page: it may have lost it, or the
    stack has slipped a sheet there, and every form after it then takes the
    next one's Declaration page, exactly one, straight after it. A blank
    page where a form should begin, with a form straight after it, may be a
    Declaration page fed face down, or fed first. Either way every form of
    the run waits for a person, as for a Declaration page out of place
    (_pair). A page offered as the Declaration page the reader missed
    (_offer) counts as one here, and so does every page but the first of a
    confirmed document. A page still to be read may be anything, so nothing
    is said across it yet.
    """
    if number in pairing.held:
        return
    before: PageView | None = None  # the last page that is not blank
    blank = False  # a blank page since then
    for page in run:
        if page.unread:
            before, blank = page, False
            continue
        if _is_blank(page):
            blank = True
            continue
        if not _second_sheet(page, pairing, open_ids, firsts):
            # A form begins here: out of step after a blank page where one
            # should have begun, or straight after another form's first page.
            if before is None or _second_sheet(before, pairing, open_ids, firsts):
                out_of_step = blank
            else:
                out_of_step = not before.unread
            if out_of_step:
                pairing.held[number] = GAP_IN_STACK
                return
        before, blank = page, False


def _second_sheet(page: PageView, pairing: _Pairing, open_ids: set[str], firsts: set[str]) -> bool:
    """Whether a page is in a form's second place (_check_steps).

    A Declaration page, found or attached, one offered as the Declaration
    page the reader missed, or any page but the first of a confirmed
    document.
    """
    if page.page_id not in open_ids:
        return page.page_id not in firsts
    return _is_declaration(page) or page.page_id in pairing.offered


def _carry_on(pairing: _Pairing, runs: list[list[PageView]]) -> None:
    """Carry a run's hold, and its pages still to be read, into the runs that may go on from it (batch codes off).

    One stack can come in as two runs - a pause longer than RUN_GAP_MINUTES,
    or the scan page loaded again, which gives it a new id - and the second
    then begins with a Student ID page like any other, whatever was out of
    place in the first: a stack fed Declaration page first pairs every form
    after the cut wrongly too. So a run that may go on from an earlier one
    (_goes_on) is held while that one is, and none of its forms goes up by
    itself while that one still has a page to be read.
    """
    reading = {number for number, run in enumerate(runs) if any(page.unread for page in run)}
    for number in range(1, len(runs)):
        first = runs[number][0]
        for earlier in range(number):
            if not _goes_on(runs[earlier][-1], first):
                continue
            pairing.later.setdefault(earlier, []).append(first)
            if earlier in pairing.held:
                pairing.held.setdefault(number, AFTER_HELD_STACK)
            if earlier in reading or earlier in pairing.unsettled:
                pairing.unsettled.add(number)


def _goes_on(last: PageView, first: PageView) -> bool:
    """Whether a run beginning with `first` may be the rest of the stack of a run that ended with `last`.

    It began after that run ended, for the same school and city, and from
    the same scan page or within RUN_GAP_MINUTES of it.
    """
    return (
        last.received_ns < first.received_ns
        and (last.school, last.location) == (first.school, first.location)
        and (
            first.received_ns - last.received_ns <= _RUN_GAP_NS
            or (bool(last.sender) and _scan_page(last.sender) == _scan_page(first.sender))
        )
    )


def _no_form(page: PageView, before: PageView | None, open_ids: set[str]) -> str:
    """Why a Declaration page joins no form, as it says so under Unsorted; "" when it joins `before`."""
    if before is None:
        return _NOTHING_BEFORE
    if _is_declaration(before):
        return _AFTER_DECLARATION
    if before.page_id not in open_ids:
        return _AFTER_CONFIRMED
    if (before.school, before.location) != (page.school, page.location):
        return _OTHER_SCHOOL
    return ""


def _found_declaration(page: PageView) -> bool:
    """Whether the reader found the page to be a Declaration page, or a person attached it as one."""
    return page.skip_reason == "declaration" or bool(page.declaration_of)


def _is_declaration(page: PageView) -> bool:
    """A Declaration page, found or attached (_found_declaration), that nobody has typed a Student ID on."""
    return _found_declaration(page) and page.student_id_source != "typed"


def _is_blank(page: PageView) -> bool:
    """A blank page, as OCR found it and nobody has typed a Student ID on."""
    return (page.blank or page.skip_reason == "blank") and page.student_id_source != "typed"


def _waiting_for_declaration(first: PageView, run: list[PageView], at: int, now_ns: int) -> str:
    """Why a form with no Declaration page is waiting for one, or "" when it is not (batch codes off).

    It waits while the next page in its run that is not blank is still being
    read - unless that page was sent for another school or city, so is never
    its Declaration page - and while nothing but blank pages has come after
    it yet and it reached the server less than DECLARATION_WAIT_SECONDS
    ago: the scanner may still be saving the next sheet.

    Args:
        first: The form's first page.
        run: Its run, and `at` its place in it.
        now_ns: The time now (epoch ns).
    """
    after = next((page for page in run[at + 1 :] if not _is_blank(page)), None)
    if after is not None:
        if after.unread and (after.school, after.location) == (first.school, first.location):
            return "Waiting for its Declaration page - the page after it is still being read."
        return ""
    if now_ns - first.received_ns < _DECLARATION_WAIT_NS:
        return "Waiting for its Declaration page - nothing has come after it yet."
    return ""


def _note_others(open_docs: list[Document], done: list[Document], open_views: list[PageView]) -> None:
    """Mark the open documents that may not go up by themselves because of the others.

    Two documents for one student - the same form scanned twice, or two
    pages of one form that each carry the ID - both wait for a person while
    either is open, as one confirmed lately for the student does. A confirm
    still running is one of them: its pages stay open until its record is
    saved. With batch codes on, a page still to be read may be a later part
    of any open document.
    """
    holders: dict[str, set[str]] = {}
    for doc in open_docs + done:
        # An open document's parts may match different students; a
        # confirmed one is for the student it was confirmed for.
        students = [page.matched_student for page in doc.pages] if doc.is_open else [doc.matched_student]
        for student_id in filter(None, map(_matched_id, students)):
            holders.setdefault(student_id, set()).add(doc.key)
    for doc in open_docs:
        doc.shares_student = any(
            len(holders.get(_matched_id(page.matched_student), ())) > 1 for page in doc.pages
        )
    if batch_codes_on() and any(view.unread for view in open_views):
        for doc in open_docs:
            doc.pages_to_read = True


def _matched_id(student: dict[str, Any] | None) -> str:
    """A matched student's ID, to compare with another's ("" for none)."""
    return str((student or {}).get("student_id") or "").strip().upper()


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


def _page_document(pages: list[PageView]) -> Document:
    """A form as a document of its own (batch codes off), keyed by its first page."""
    first = pages[0]
    return Document(
        key=scan_queue.page_document_key(first.page_id),
        code="",
        batch=0,
        number=0,
        pages=pages,
        school=first.school,
        location=first.location,
    )


def _judge_page(doc: Document, done: list[Document], waiting: str) -> None:
    """Work out a form's Student ID, status and message (batch codes off).

    Its Student ID is its first page's. A form whose Declaration page has
    not joined it waits for it (_waiting_for_declaration), with no Confirm
    to press. Once it stops waiting it needs a person's look: it can still
    be confirmed without its Declaration page, its PDF the Student ID page
    alone. So does every form of a held run (_pair), which may be paired
    wrongly, and says why it is held, and a matched student who already has
    a confirmed document on the scan page (those of the last
    DONE_DOCUMENTS_HOURS) - it may be the same form scanned twice.

    Args:
        doc: The form, its first page first; `held` set.
        done: The person's confirmed documents.
        waiting: For a form with no Declaration page, why it is still
            waiting for one; "" when it is not.
    """
    page = doc.pages[0]
    match = page.matched_student
    # Why it needs a person, by attention code, in the order to say them.
    said: dict[str, str] = {}
    if match and match.get("student_id"):
        doc.student_id = match["student_id"]
        doc.student_id_source = page.student_id_source
        doc.matched_student = match
        earlier = [d for d in done if d.student_id.upper() == doc.student_id.upper()]
        if earlier:
            name = match.get("student_name")
            who = f"{name} ({doc.student_id})" if name else doc.student_id
            said[SAME_STUDENT] = (
                f"A form for {who} was already confirmed. Check the paper"
                " - this may be the same form scanned twice."
            )
    else:
        doc.student_id = page.student_id
        doc.student_id_source = page.student_id_source
        where = doc.school or (f"{doc.location} schools" if doc.location else "the student list")
        said[NO_MATCH] = f"No student with ID {doc.student_id} at {where}. Check the ID against the paper."
    if len(doc.pages) == 1 and not waiting:
        said[NO_DECLARATION] = (
            "No Declaration page came after this Student ID page - check the form."
            " It can still be confirmed without one."
        )
    if doc.held:
        said[doc.held] = _HELD_MESSAGES[doc.held]
    doc.attention = list(said)
    doc.message = " ".join(said.values())

    if waiting:
        doc.status = WAITING
        doc.message = waiting
    elif page.status == scan_queue.STATUS_FAILED:
        doc.status = FAILED
        # A form already confirmed for this student is still said, or the
        # same form could be filed twice; so are a missing Declaration page
        # and a held run.
        doc.message = " ".join(
            ["This page could not be read. Check the Student ID against its image."]
            + [text for code, text in said.items() if code != NO_MATCH]
        )
    elif said:
        doc.status = NEEDS_ATTENTION
    else:
        doc.status = READY
        doc.message = f"Matched {(match or {}).get('student_name') or doc.student_id}."


def _document_from_record(
    record: dict[str, Any], owner: str, state: scan_queue.QueueState
) -> Document | None:
    key = str(record.get("key") or "")
    code = str(record.get("code") or "")
    parsed = parse_batch_code(code)
    # No code: a form, confirmed while batch codes were off.
    if not scan_queue.is_document_key(key) or (code and parsed is None):
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
    automatic = record.get("automatic") is True
    if uploaded:
        message = f"Uploaded automatically to Drive for {who}." if automatic else f"Uploaded to Drive for {who}."
    elif in_drive:
        message = (
            f"Uploaded to Drive for {who}, but its History entry could not be written yet. "
            "Use Retry upload to finish."
        )
    else:
        message = (
            ("Saved automatically" if automatic else "Saved")
            + ", but the upload to Drive failed"
            + (f": {record['upload_error']}" if record.get("upload_error") else ".")
            + " Use Retry upload."
        )
    return Document(
        key=key,
        code=code,
        batch=parsed.batch if parsed else 0,
        number=parsed.number if parsed else 0,
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
        DocumentError: 404 not theirs, 409 already confirmed or batch codes
            are off, 400 not a code.
    """
    _owned_open_page(owner_email, page_id)
    if not batch_codes_on():
        # A page left open from before would otherwise take a code that
        # nothing reads.
        raise DocumentError(
            "Batch codes are switched off, so a page needs only its Student ID. Reload the page.", 409
        )
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


def set_page_declaration(owner_email: str, page_id: str, of_page_id: str, by: str) -> PageView:
    """Attach a page by hand as the Declaration page of the form straight before it ("" takes it off).

    For a Declaration page the reader did not recognise, waiting under
    Unsorted as a page with no Student ID (_offer). The person's word
    is kept like a typed Student ID, with who and when, and the pairing
    then counts the page as that form's Declaration page - while that
    form's page is still straight before it and begins a form
    (_drop_stale_attachments). Such a form never goes up by itself: a
    person presses Confirm. Taken under the form's confirm lock, so a
    confirm of the form cannot run in between.

    Args:
        owner_email: The signed-in person, whose page it must be.
        page_id: The page to attach.
        of_page_id: The page scanned straight before it, the form's Student
            ID page, as the page offered it; "" to take the page off again.
        by: Who is attaching it, for the record.

    Raises:
        DocumentError: 404 not theirs; 409 already confirmed, batch codes
            are on, the form being confirmed right now, or the page no
            longer straight after that form's page with no Declaration page
            (nothing saved); 400 not a page id.
    """
    _owned_open_page(owner_email, page_id)
    if batch_codes_on():
        raise DocumentError(
            "Batch codes are switched on, so a page joins its document by its code. Reload the page.", 409
        )
    owner = scan_queue.normalise_owner(owner_email)
    if not of_page_id:
        scan_queue.set_declaration_override(page_id, None, by)
    elif not scan_queue.is_page_id(of_page_id):
        raise DocumentError("The page to attach it to was not named. Reload the page. Nothing was saved.", 400)
    else:
        try:
            with scan_queue.confirm_lock(owner, scan_queue.page_document_key(of_page_id)):
                # Asked again now, inside the lock: the form may have been
                # confirmed, or have its Declaration page, since the offer.
                page = next((p for p in snapshot(owner).unsorted if p.page_id == page_id), None)
                if page is None or page.attach_to != of_page_id:
                    raise DocumentError(
                        "This page is no longer straight after a form waiting for its Declaration "
                        "page, so it was not attached. Nothing was saved - reload the page.",
                        409,
                    )
                scan_queue.set_declaration_override(page_id, of_page_id, by)
        except scan_queue.DocumentBusy as exc:
            raise DocumentError(
                "That form is being confirmed right now, so the page was not attached. Nothing was saved.", 409
            ) from exc
    view = load_page_view(page_id, owner)
    assert view is not None
    return view


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


def _shaped_id(text: str) -> str:
    """A Student ID as confirm compares it: letters and digits only, upper-case, as_student_id."""
    return as_student_id(re.sub(r"[^A-Za-z0-9]", "", text or "").upper())


def auto_upload_refusal(doc: Document) -> str:
    """Why the scan page may not confirm a document by itself, or "" when it may.

    With Upload matched pages automatically on, a document goes up without
    anyone pressing Confirm only when all four hold:

      1. It is Ready: its Student ID matched a student, with nothing to
         check - no parts that disagree, no code used before, no part
         failed or still being read. With batch codes off it is a whole
         form, put together by the reader alone: its Student ID page and
         exactly one Declaration page, which the reader recognised (not
         one a person attached), scanned straight after it - no page of
         any kind between, a blank one included - within RUN_GAP_MINUTES,
         from the same scan page (PageMeta.sender, which an older page
         does not send) and the same way, both from its scan folder or
         both added by hand, in a run with nothing out of place (_pair);
         and the page after its Declaration page, once read, is the next
         form's Student ID page. It waits, too, while its stack settles
         (_unsettled): until every page scanned before it and the page
         after its Declaration page have been read, or with nothing after
         that page, for DECLARATION_WAIT_SECONDS - either may turn out to
         be a page out of place. A form never goes up as its Student ID
         page alone. Nor anything only the other documents show
         (snapshot): no other document, open or confirmed lately, for the
         same student, and with batch codes on none of the person's pages
         still to be read. And the student was not already marked scanned
         (scanned_at) when looked up - by anyone, at any time.
      2. The match is exact: every Student ID on its parts, read by OCR or
         typed by a person, is the matched student's own after the clean-up
         every Student ID gets - trimmed, upper-cased, and as_student_id,
         which for a reading not shaped like an ID as read puts a letter
         right where the shape needs a digit (Z for 2, O for 0) or a digit
         where it needs a letter, before the reading is stored. A student
         the lookup reached only by changing that ID towards a near one -
         student_lookup's shape fix, safe variant, constrained repair or
         digit suffix - is not.
      3. One school was picked for every page. With only a city ("All Pune
         schools") the lookup searched every school in it.
      4. Every page - a form's Declaration page too - was sent with the
         switch on (PageMeta.auto_upload), so a page queued before the
         switch existed never goes up by itself.
    """
    reason = _lasting_refusal(doc)
    if not reason and not batch_codes_on():
        reason = _unsettled(doc)
    return reason


def _lasting_refusal(doc: Document) -> str:
    """Why a document may not go up by itself however long its stack settles (auto_upload_refusal), or ""."""
    if not auto_upload_on():
        return "automatic upload is switched off on this server"
    if not doc.is_open:
        return "it has already been confirmed"
    if not batch_codes_on():
        reason = _form_refusal(doc)
        if reason:
            return reason
    if doc.status != READY:
        return "it needs checking first"
    if doc.pages_to_read:
        return "some of your pages are still being read, and may be more parts of it"
    if doc.shares_student:
        return "another document - open, or confirmed lately - is for the same student"
    if any((page.matched_student or {}).get("scanned_at") for page in doc.pages):
        return "the student is already marked scanned"
    matched = _shaped_id(str((doc.matched_student or {}).get("student_id") or ""))
    read = [_shaped_id(page.student_id) for page in doc.pages if page.student_id]
    if not matched or not read or any(found != matched for found in read):
        return "its Student ID is not an exact match"
    if not doc.school or any(page.school != doc.school for page in doc.pages):
        return "no single school was picked for it"
    if not all(page.auto_upload for page in doc.pages):
        return "it was sent with automatic upload off"
    return ""


def _form_refusal(doc: Document) -> str:
    """With batch codes off: why a form may not go up by itself as it was put together, or "".

    Asked before its status, so a refused automatic confirm says which.
    """
    if not any(_is_declaration(page) for page in doc.pages[1:]):
        return "it has no Declaration page"
    if len(doc.pages) != 2:
        return "it has more than one Declaration page"
    first, declaration = doc.pages
    if declaration.declaration_of:
        return "its Declaration page was attached by hand"
    if doc.between:
        return "a page came between its Student ID page and its Declaration page"
    if declaration.received_ns - first.received_ns > _RUN_GAP_NS:
        return f"its Declaration page came more than {RUN_GAP_MINUTES} minutes after its Student ID page"
    if doc.held:
        return _HELD_REASONS[doc.held]
    if any(not page.unread and not page.student_id for page in doc.after):
        # Blank, failed, a Declaration page or one with no Student ID read:
        # this form's own Declaration page may be that one.
        return "the page after its Declaration page is not a Student ID page"
    if not first.sender or _scan_page(declaration.sender) != _scan_page(first.sender):
        return "it is not known that one computer sent both its pages"
    if declaration.sender != first.sender:
        # Added while the scanner was between its two sheets, perhaps.
        return "one of its pages was added by hand and the other came from the scan folder"
    return ""


def _unsettled(doc: Document) -> str:
    """With batch codes off: why a form may not go up by itself yet, though it may once its stack settles, or "".

    Until every page scanned before it has been read, and the page after
    its Declaration page has come and been read - or, with nothing after
    that page, for DECLARATION_WAIT_SECONDS: any of them may turn out to be
    a page out of place, which holds the form (_pair). Asked last, once
    nothing else stands in the way (auto_upload_pending).
    """
    if doc.earlier_unread:
        return "a page scanned before it is still being read"
    if any(page.unread for page in doc.after):
        return "the page after its Declaration page is still being read"
    if not doc.after and time.time_ns() - doc.pages[-1].received_ns < _DECLARATION_WAIT_NS:
        return "nothing has come after its Declaration page yet"
    return ""


def _refuse_unless_automatic(doc: Document, shaped: str) -> None:
    """Refuse an automatic confirm of a document that may not go up by itself now.

    The page asks from its last look, which may be stale: a part read since,
    an ID typed in another tab, or a page that was never eligible.

    Raises:
        DocumentError: 409, with nothing saved.
    """
    reason = auto_upload_refusal(doc)
    if not reason and shaped != _shaped_id(doc.student_id):
        reason = "the Student ID sent is not the one it matched"
    if reason:
        raise DocumentError(
            f"{reason[0].upper()}{reason[1:]}, so it was not uploaded automatically. "
            "Nothing was saved - check it and press Confirm and upload.",
            409,
        )


def confirm_document(
    owner_email: str,
    key: str,
    student_id: str,
    by: str,
    page_ids: list[str],
    automatic: bool = False,
) -> ConfirmOutcome:
    """Confirm an open document: the one place the queue writes anything.

    Args:
        owner_email: The signed-in person, whose document it must be.
        key: The open document's key (its code, or with batch codes off
            its Student ID page's key).
        student_id: The Student ID the person confirmed.
        by: Who is confirming, for the record.
        page_ids: The parts the page was showing when Confirm was pressed.
            Only exactly those are confirmed: a part read since the page's
            last look (a rescan, another form's corner misread as this code,
            or with batch codes off the form's Declaration page) refuses the
            confirm rather than going into the record and the PDF unseen.
        automatic: The scan page is confirming it by itself (Upload matched
            pages automatically), nobody having pressed Confirm. Done only
            while auto_upload_refusal has nothing to say, for the student it
            matched, with a Google sign-in still in date; the lookup that
            marks the student takes the exact ID only, and leaves one
            already marked scanned as they are. The record and History's
            source ("auto:...") say so; `by` is still the signed-in person,
            in whose session it ran.

    Returns:
        What happened. A Drive failure is not an error here: the student is
        marked and the document left confirmed with Retry upload.

    Raises:
        DocumentError: 404 no such document; 409 already confirmed, still
            being read, changed since it was shown, being confirmed in
            another tab, no Google sign-in, the student is not in the
            database, or an automatic confirm the document no longer
            qualifies for - its sign-in run out, or the student marked
            scanned since (nothing saved); 400 not a Student ID, or no parts.
        LookupUnavailable: The database could not be asked (nothing saved).
    """
    owner = scan_queue.normalise_owner(owner_email)
    doc = _open_document_for(owner, key)
    shown = shown_page_ids(page_ids)
    if not _same_parts(doc, shown):
        raise DocumentError(CHANGED_MESSAGE, 409)
    shaped = _shaped_id(student_id)
    if not shaped:
        raise DocumentError(
            "That does not look like a Student ID (letters then digits, e.g. AADMIS310714). Nothing was saved.",
            400,
        )
    if automatic:
        _refuse_unless_automatic(doc, shaped)
    if not drive_store.has_drive_credentials():
        # Checked before the lookup: marking the student and then being
        # unable to file the form would leave a scanned child with no form.
        raise DocumentError("Sign in with Google to upload to Drive. Nothing was saved.", 409)
    if automatic and not drive_store.drive_sign_in_current():
        # Nobody is watching this one: a sign-in that has run out would mark
        # the child and then fail the upload, one form after another.
        raise DocumentError(
            "Your Google sign-in has run out, so it was not uploaded automatically. "
            "Nothing was saved - sign in with Google again, then press Confirm and upload.",
            409,
        )

    try:
        with scan_queue.confirm_lock(owner, doc.key):
            # Again inside the lock: another tab may have just confirmed it,
            # or the worker read another part with this code meanwhile.
            doc = _open_document_for(owner, key)
            if not _same_parts(doc, shown):
                raise DocumentError(CHANGED_MESSAGE, 409)
            if automatic:
                _refuse_unless_automatic(doc, shaped)
            source = f"batch:{doc.code}" if doc.code else _scan_source(doc.pages[0])
            if automatic:
                source = f"auto:{source}"
            try:
                result = confirm_student_id(
                    shaped,
                    source=source,
                    defer_log=True,
                    school=doc.school or None,
                    location=doc.location or None,
                    exact_only=automatic,
                    unscanned_only=automatic,
                )
            except ExtractError as exc:
                # LookupUnavailable is one of these; it goes up as it is.
                if exc.status_code == 503:
                    raise
                raise DocumentError(exc.message, exc.status_code) from exc
            if not result.found_in_db:
                where = doc.school or (f"{doc.location} schools" if doc.location else "the student list")
                if automatic:
                    # Gone from the school since the page was read, or there
                    # only as a near ID: a guess is for a person to make.
                    raise DocumentError(
                        f"{shaped} is no longer exactly a student at {where}, so it was not "
                        "uploaded automatically. Nothing was saved - check it and press "
                        "Confirm and upload.",
                        409,
                    )
                raise DocumentError(
                    f"{shaped} is not a student at {where}. Nothing was saved - "
                    "check the ID against the paper and correct it.",
                    409,
                )

            student = result.student or {}
            if automatic and student.get("already_scanned"):
                # Marked since the page was read - another form for the child,
                # perhaps another person's, or a confirm at the same moment.
                name = student.get("student_name")
                who = f"{name} ({result.student_id})" if name else result.student_id
                raise DocumentError(
                    f"{who} is already marked scanned, so it was not uploaded automatically. "
                    "Nothing was saved - check the paper and press Confirm and upload.",
                    409,
                )
            record: dict[str, Any] = {
                "key": f"{doc.key}~{secrets.token_hex(4)}",
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
                # Confirmed by the scan page itself, nobody pressing Confirm.
                "automatic": automatic,
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
        # The lock its confirm held: the code, or its Student ID page's key.
        with scan_queue.confirm_lock(owner, str(record.get("code") or key.split("~", 1)[0])):
            record = scan_queue.load_document_record(owner, key) or record
            if not scan_queue.document_record_finished(record):
                _upload_and_log(owner, record)
    except scan_queue.DocumentBusy as exc:
        raise DocumentError("This document is already being uploaded.", 409) from exc
    return _outcome(owner, record)


def _scan_source(page: PageView) -> str:
    """History's source for a document with no code: the scan its first page came from."""
    name = page.source_name or "scan"
    if page.source_pages > 1:
        name = f"{name} (page {page.source_page}/{page.source_pages})"
    return f"scan:{name}"


def _open_document_for(owner: str, key: str) -> Document:
    if not scan_queue.is_document_key(key):
        raise DocumentError("No such document.", 404)
    found = snapshot(owner).find(key)
    if found is None:
        raise DocumentError("No such document. It may have just been confirmed.", 404)
    if not found.is_open:
        raise DocumentError("This document has already been confirmed.", 409)
    if found.status == WAITING:
        if not batch_codes_on() and found.message:
            # A form waiting for its Declaration page says what it waits for.
            raise DocumentError(f"{found.message} Try again in a moment.", 409)
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
    from the matched student's own row in active_student_data. A document
    with no code (batch codes off) is <Name>_<ID>.pdf.
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
    code = str(record.get("code") or "")
    file_name = drive_store.drive_safe_name(f"{person}_{code}" if code else person, person) + ".pdf"
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
