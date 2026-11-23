"""Which paper form a page is - the scan page's form choice - and so which rules read and pair it.

There are two consent forms:

  - the two-sheet form, used by almost every school: a Student ID page, then
    a Declaration page on a sheet of its own, its printed "घोषणा" heading at
    the top left;
  - the one-sheet form: one sheet printed on both sides, its Student ID side
    on the front and its Declaration side on the back, the heading about half
    way down the back's left margin.

The scan page asks which one is being scanned before it sends anything, and
every page it sends carries the answer (scan_queue.PageMeta.form). A page
that comes with none - from the Drive inbox, the /api/extract routes, an
older scan page, or queued before the choice existed - is the two-sheet
form, read and paired exactly as before the one-sheet form was known. The
rules that differ are each decided by one_sheet, in one place each:

  - the reader looks for the heading down the whole left margin, as the page
    came and turned round, and so knows a Declaration side, only on the
    one-sheet form; the two-sheet form's Declaration page is known by the
    heading at the top left alone (extract_student_id._extract);
  - the worker turns a page round by its ruled layout whenever that matches
    better turned round on the two-sheet form, but on the one-sheet form only
    on a close match (ocr_worker._layout_orientation);
  - the two sides of each sheet pair by their places in their file, or their
    numbers in their scanner session, only on the one-sheet form; the
    two-sheet form pairs by scan order (documents._sheets).

Everything else - the empty Student ID skip, batch codes, automatic upload -
is the same for both.
"""

from __future__ import annotations

TWO_SHEET = "two_sheet"
ONE_SHEET = "one_sheet"
FORMS = (TWO_SHEET, ONE_SHEET)


def parse(value: str | None) -> str | None:
    """A form choice as a scan sends it, or None for one this server does not know.

    Nothing at all is TWO_SHEET: a page that knows nothing of the choice is
    read as every page was before it. Case and spaces around it do not
    count.

    Args:
        value: The scan's "form" field, or None without one.

    Returns:
        TWO_SHEET or ONE_SHEET; None for any other value.
    """
    text = (value or "").strip().lower()
    if not text:
        return TWO_SHEET
    return text if text in FORMS else None


def one_sheet(form: str | None) -> bool:
    """Whether a page is the one-sheet form, read and paired by its rules; any other value is the two-sheet form."""
    return form == ONE_SHEET
