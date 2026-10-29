# Student ID OCR

Extract the handwritten English **Student ID** from Akanksha Parent Consent Forms
and mark the student as scanned. Staff scan with a sheet-fed scanner such as
the Fujitsu / Ricoh **SP-1130N**, started from the scanner itself (its Scan
button or its own software). **Nothing of ours is installed** on the scanning
computer: the page watches the folder the scanner's software saves into, takes
scanned files added by hand, or the server reads them from a **Drive inbox**
folder.

Supports English, Hindi, and Marathi form labels. Only the Student ID value is read.

## How it works

1. A form arrives - from the watched scan folder or scanned files added on the
   page - as a JPG / PNG / PDF / TIFF. The server only stores and queues it
   (one queued page per page of the file) and answers at once.
2. In the background the OCR worker reads each queued page: it turns it the
   right way up, reads the handwritten **batch code** (`B1-12`) in its top-right
   corner when batch codes are on (they are off by default - see
   [Batch codes and documents](#batch-codes-and-documents)), finds the printed
   Student ID line (EN / HI / MR) from the form's ruled layout, and reads the
   handwritten value with Google Cloud Vision, falling back to the local
   RapidOCR readers.
3. It looks that Student ID up in MySQL `active_student_data`, read-only and
   within the school you picked, repairing a near-miss reading only when it is
   safely unambiguous.
4. Each form is one **document**: its Student ID page and the Declaration
   page scanned straight after it, which has no Student ID - or, with batch
   codes on, the pages with the same batch code. The page shows each
   document with its parts, the Student ID read and who it matched; a page
   that could not be read is shown as its scanned image, so you can type its
   Student ID (or code) from it.
5. Only when a document is confirmed does anything get saved: it marks
   `scanned_at`, puts one PDF of all its parts in your own Google Drive and
   logs the attempt for the History page. **You press Confirm and upload** -
   or, with **Upload matched pages automatically** on, the page does it by
   itself for a Ready document scanned for one school whose Student ID is
   exactly a student there, not yet marked scanned, once its Declaration page,
   scanned straight after it on the same computer, and the page after that
   have been read; everything else waits for you.

```text
Physical form → scanner (its Scan button or its own software)
  ├─ saves into "OCR Scans" on the computer → watched by the page (Chrome / Edge) ┐→ queue → OCR worker
  ├─ saved files → added on the page with "Add scanned files" (any browser)      ┘   → you confirm → Drive + DB
  └─ saves into "OCR Inbox" in your Drive → inbox watcher on the server → OCR → lookup → saved
```

Neither the server nor the page talks to the scanner: the scanner's own
software does the scanning and the page only uploads the files it saves. So the
same page works with any scanner that has its own software, on Windows, macOS,
Linux and ChromeOS.

## Setup (Windows)

```powershell
cd d:\NewOCRRealForm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Open http://127.0.0.1:5000

On a Mac, AirPlay Receiver already holds port 5000: run `PORT=5001 .venv/bin/python run.py`
and open http://127.0.0.1:5001 instead.

First extract downloads the RapidOCR models (can take a few minutes).

## Sign-in (Google OAuth2)

The whole app (Scanner + History) requires signing in with a Google account
on the `ALLOWED_EMAIL_DOMAIN` from `.env` (defaults to `akanksha.org`).

1. In [Google Cloud Console](https://console.cloud.google.com/apis/credentials),
   create an **OAuth 2.0 Client ID** (Web application).
2. Add this **Authorized redirect URI**: `http://127.0.0.1:5000/login/google/callback`
   (add your production URL's equivalent too, once deployed).
3. Put the Client ID/Secret in `.env`:
   ```env
   GOOGLE_CLIENT_ID=...
   GOOGLE_CLIENT_SECRET=...
   ALLOWED_EMAIL_DOMAIN=akanksha.org
   ```
4. `FLASK_SECRET_KEY` (signs the session cookie) is already generated in `.env` — keep it secret, don't commit it.

Sign-in is session-only — no `users` table. Anyone outside `ALLOWED_EMAIL_DOMAIN`,
or with an unverified Google email, gets bounced back to `/login` with an error.

## Web UI

1. Sign in and open the Scanner page.
2. Pick the school whose forms you are scanning - it keeps a misread ID from
   matching a child at another school.
3. Bring the forms in (see below): connect the scan folder once and press the
   scanner's Scan button, add scanned files by hand, or press **Start
   scanning** on the Drive inbox panel.
4. With batch codes on, write the batch code on every page before scanning
   (see [Batch codes and documents](#batch-codes-and-documents)).
5. Check each document under **OCR results**, correct the Student ID (or code)
   if needed, and press **Confirm and upload**. Nothing is saved until you do -
   except that, with **Upload matched pages automatically** on (the default),
   a Ready document scanned for one school whose Student ID is exactly a
   student there, not yet marked scanned, is confirmed and uploaded by itself
   once its Declaration page, scanned straight after it, and the page after
   that have been read (see
   [Uploading matched pages automatically](#uploading-matched-pages-automatically)).

## Scanning - nothing of ours to install

Browsers cannot talk to a scanner, so the scanner's own software does the
scanning and the page picks up the files it saves. Every file goes through the
same upload, `POST /api/scans`, which only stores and queues it: the page never
waits for OCR, and moves straight on to the next scan. The **Scanned documents**
list shows every page received and whether it is waiting for OCR, being read,
read or failed.

Accepted files: `.jpg`, `.jpeg`, `.png`, `.pdf`, `.tif`, `.tiff`. A multi-page
PDF or TIFF is read page by page; 1-bit (black and white), greyscale and colour
TIFFs all work. A file that is only *named* `.tif` but is something else is
refused with "Not a TIFF image" rather than read as whatever it really is.

### 1. Scanner + watched scan folder (Chrome or Edge on a computer)

The scanner saves each scan into one folder, `OCR Scans`, and the page watches
that folder.

1. Set the scanner up once (per-OS steps below; the page shows the steps for
   the OS it is opened on, in a setup guide that stays open until a folder is
   connected).
2. On the Scanner card press **Connect scan folder** and pick `OCR Scans`.
   The browser asks once for permission to view its files. With batch codes
   on it asks to change them too: once a scan's batch code has been read, the
   page moves that file into a subfolder named after its batch (`B1`, `B2`,
   ...), so the folder ends up sorted the way the paper is. It only ever moves
   files it sent itself, never deletes anything else, and leaves the
   subfolders alone. Allow viewing only and everything still works, without
   the sorting. With batch codes off nothing is moved.
3. If the folder already holds scans, the page asks once whether to send them
   too or only new ones, so an old folder is not re-read by surprise.
4. Load the forms and press the scanner's Scan button. While the Scanner tab
   is open and visible the page looks in the folder every 2 seconds, waits
   until a new file's size and time have stopped changing (the scanner has
   finished writing it), and sends it in - with the school and location chosen
   when watching started. The card shows the
   folder, "Watching - N pages sent today", the last error, and **Stop
   watching** / **Start watching** and **Change folder**.

The page remembers which files it has sent from each folder (by name, size and
time), so a reload or another day does not send one twice; the files
themselves are left where they are. On the next visit the folder is shown
again, and if the browser wants its permission back the page offers
**Reconnect scan folder** - one click. In a private window the browser keeps
nothing, so the folder has to be connected again on each visit.

The folder watch needs Chrome or Edge on a computer, and a secure page: open
the app over `https://`, or on `http://127.0.0.1` / `http://localhost` for a
local run. Firefox and Safari do not offer it; the page says so and offers
**Add scanned files** instead.

#### Setting up the scanner (once per computer)

Written for the Fujitsu / Ricoh SP-1130N; any scanner with its own software
works the same way - only the save folder and the scan settings change.

**Windows**

1. Download Ricoh's free **PaperStream IP (TWAIN)** driver and **PaperStream
   Capture** from the SP-1130N download page,
   https://www.pfu.ricoh.com/global/scanners/fisp/dl/sp-1130n.html, and
   install both.
2. In PaperStream Capture create a profile: output folder `Documents\OCR Scans`,
   file type JPEG (or PDF), 100 dpi, Black & White, one side (simplex),
   document feeder.
3. Link the profile to the scanner's Scan button (PaperStream Capture's
   "Assign to scanner button").
4. Press the Scan button on the SP-1130N.

**macOS**

1. Install Ricoh's **Image Scanner Driver for macOS (ICA)** for the SP-1130N
   (it works over USB).
2. Open Image Capture and choose the SP-1130N: Use Document Feeder, Kind
   Black & White, Resolution 100 dpi, Format JPEG or TIFF, Scan To the
   `OCR Scans` folder.
3. Press Scan.

**Linux**

Use Document Scanner (simple-scan), which drives the SP-1130N through SANE's
`fujitsu` backend, and save each scan into the `OCR Scans` folder.

**ChromeOS**

Use the Scan app and set it to save to the `OCR Scans` folder.

**Every OS:** load page 1 of each form face down in the feeder. The page picks
each scan up within a few seconds of the scanner saving it.

### 2. Add scanned files (every browser)

Drop scans onto the drop zone (or focus it and press Enter or Space), or press
**Choose files** and pick several at once - `.jpg`, `.jpeg`, `.png`, `.pdf`,
`.tif`, `.tiff`. They go through exactly the same upload as the watched folder,
marked as added by hand (see [Batch codes and documents](#batch-codes-and-documents)).
This is the way in on Firefox and Safari, and for scans saved somewhere else.

### 3. Drive inbox

See [Scanning through the Drive inbox](#scanning-through-the-drive-inbox): the
scanner's software saves into a Google Drive folder and the server reads the
pages from there.

## Batch codes and documents

**Batch codes are off by default.** Without them the code corner is not read,
and the pages are put together by the order they were scanned in. Each form is
two sheets, scanned one straight after the other: the **Student ID page**, then
its **Declaration page** (the signature page, headed "घोषणा"), which has no
Student ID. The two are one document: one Confirm, one PDF with the Student ID
page first, named `<StudentName>_<StudentID>.pdf` (same folders as below).

Pages are only put together inside a **run**: the pages one scan page sent -
the page makes an id for itself each time it is loaded and sends it with every
scan, so two computers (or two tabs) on one sign-in never mix their stacks -
each reaching the server within `RUN_GAP_MINUTES` (10) of the one before. Files
added by hand are in the same run as what the scan folder sends, so a form
added a sheet at a time is still one document; they carry the page's id with
`-1` after it, and a form with one sheet from the scan folder and one added by
hand - one added while the scanner was between its two sheets, say - waits
for you. Scans sent without an id, by an older page, make runs by the time
alone.

A Declaration page joins the page right before it in its run, a blank page
between them passed over, when that page is not a Declaration page itself, was
sent for the same school and city, is not yet confirmed and has no Declaration
page yet: a form has exactly one. Any other Declaration page joins nothing and
waits under **Unsorted** on its own, saying only what is true of it - no
Student ID page right before it, right after another Declaration page, after
a form that was already confirmed, or sent for another school or city - and
**holds its run**: every form in that run says "This stack has a Declaration
page without its Student ID page, so its forms may be paired wrongly - check
them", needs your look and does not go up by itself. That is what catches a
stack fed Declaration page first, and a missing or face-down Student ID page.
A form goes up by itself only once every page scanned before it, and the page
after its Declaration page, have been read, so the forms such a page may put
out of step - those after it, and the one straight before it - are held before
they can go up; a form further back may have gone up already. A page the
reader took for a Declaration page and that you then typed a Student ID on
holds its run the same way. How a page pairs never changes once the form
before it drops off the scan page after `DONE_DOCUMENTS_HOURS`.

A run is held the same way - "This stack has a page with no Declaration page
between it and the next form" - when a form begins straight after a page that
has no Declaration page after it: two Student ID pages in a row, a page with
nothing read on it and then a Student ID page (a Declaration page fed first
whose heading was missed, or that could not be read), or a blank page where a
form should begin with a form straight after it (one fed face down). Every
form after such a place may have the next one's Declaration page, exactly one,
straight after it. A Declaration page you can attach (below) counts as one
here. And a stack cut in two runs - a pause longer than `RUN_GAP_MINUTES`, or
the scan page loaded again - holds the second run too when the first is held
("This stack may go on from one scanned before it that has a page out of
place"): a run that began after a held one ended, for the same school and
city, from the same scan page or within `RUN_GAP_MINUTES` of it. None of its
forms goes up by itself while that earlier run still has a page to be read.

Statuses are **Ready** (the ID matched a student at the school and the
Declaration page is with it), **Waiting** (no Declaration page yet: the page
after the Student ID page is still being read, or nothing has come after it
and it arrived less than `DECLARATION_WAIT_SECONDS` (120) ago - there is no
Confirm until then), **Needs attention** (no Declaration page came after it,
it matched nobody there, its run is held, or the same student already has a
form confirmed in the last `DONE_DOCUMENTS_HOURS` - perhaps the same form
scanned twice; you can still confirm it as it is, and without a Declaration
page its PDF is the Student ID page alone), **Failed**, **Confirmed** and
**Uploaded**. A form whose Student ID could not be read waits under
**Unsorted** as one card, with its scans and a box to type the ID; once typed,
it becomes a document with both pages. Blank and empty-ID pages are skipped and
wait under Unsorted too; one becomes a document only if you type its Student
ID, in case the skip was wrong (a blank page offers no box).

When the reader misses a Declaration page's heading - a crooked or faint scan -
the page waits under Unsorted as a page with no Student ID. If it came
straight after a form with no Declaration page, nothing between them, its
card offers **Attach as the Declaration page of** that form's student (or "the
form before it" when no student matched), kept as your word (who and when,
like a typed Student ID). The two then become one document - once its Student
ID is typed, for a form whose Student ID was not read, which shows the page
with it under Unsorted until then - and that form waits for you to press
Confirm: it never goes up by itself. A page straight after one that is offered
itself is not offered too: the first is the likelier Declaration page. **Take
off the Declaration page**, on the form's card or under Unsorted, undoes it,
and an attached page goes back under Unsorted by itself if the page before it
no longer begins a form (a Student ID typed on a blank page, taken off
again). Pages read with a code before the switch are put together like any
other, and documents already confirmed with a code are still listed.

Set `BATCH_CODES=1` in `.env` to bring batch codes back; everything below then
applies.

Before scanning, write a **batch code** in the top-right corner of every page,
where the forms already carry a circled serial: `B`, the batch number, a dash,
and the form's number within the batch - `B1-1`, `B1-12`, `B2-1`. Write the
**same code on every page of one form** (the Student ID page and the signature
page): that is how the app knows they belong together. Save every scan into
the one scan folder; the app sorts them into batches itself.

The reader forgives what handwriting and OCR usually do to a code - a `B` read
as `8` or `3`, a `1` read as `I`, `l` or `|`, an `O` for a zero, any kind of
dash, a space for the dash, spaces inside it - and refuses anything else. A
page whose code or Student ID could not be read is shown as its scanned image,
with boxes to type them from it.

Under **OCR results** each batch lists its documents: the code, how many parts,
the Student ID read, the student it matched (with school, grade and division)
and where it stands:

- **Waiting** - a part, or the page scanned after it (which may be another
  part), is still being read.
- **Ready** - every part is read and the Student ID matched a student.
- **Needs attention** - no Student ID was read, it matched nobody at the school,
  the parts disagree, or the code was already confirmed for another form.
- **Confirmed** - saved, but not finished: the upload to Drive failed, or the
  History entry could not be written after it. Press **Retry upload**, which
  does only what is left.
- **Uploaded** - in Drive.

Pages with no code are listed under **Unsorted** until you type one.

**Confirm and upload** is the only thing that saves anything - pressed by you,
or made by the page itself for an exact match (see
[Uploading matched pages automatically](#uploading-matched-pages-automatically)). It looks the
Student ID up within the school (nothing is saved if the student is not found
there, or if the database cannot be reached), marks `scanned_at`, makes one
PDF of all the parts in scan order, and puts it in **your own** Google Drive at

```text
OCR / <School> / <Grade> / <Division> / <StudentName>_<StudentID> / <StudentName>_<StudentID>_<Code>.pdf
```

creating any folder that is missing and reusing those already there; a PDF of
the same name already in the folder is kept, and the new one is named `... (2)`.
School, grade and division are the matched student's own, from
`active_student_data`. The attempt is logged for History with the PDF's link.
Drive needs a Google sign-in: signed in through Tether there is no Google token,
so Confirm says so and saves nothing.

### Uploading matched pages automatically

Under the school picker, **Upload matched pages automatically** is on unless
you switch it off, and the browser remembers your choice. Every scan is sent
with the switch as it is at that moment (`auto_upload` on `POST /api/scans`),
and each of its pages keeps it.

With the switch on, the page confirms and uploads a document by itself - the
same Confirm and upload, made from your browser because the PDF goes to your
own Drive - only when all four hold:

1. It is **Ready**: the Student ID matched a student and nothing needs
   checking - no parts that disagree, no batch code used before, no part
   failed or still being read. With batch codes off it is a whole form as the
   reader put it together: exactly one Declaration page, recognised by the
   reader (not attached by you), scanned straight after its Student ID page -
   no page of any kind between them, a blank one included - within
   `RUN_GAP_MINUTES`, from the same scan page and the same way (both from the
   scan folder, or both added by hand), in a run with nothing out of place.
   It goes up only once its stack has settled: every page scanned before it
   has been read, and so has the page after its Declaration page, which must
   be the next form's Student ID page - or, at the end of a stack, nothing has
   come after it for `DECLARATION_WAIT_SECONDS` (the page keeps looking until
   then). A Declaration page out of place straight after it holds it in
   time. A Student ID page never goes up by itself alone. Nor:
   - any other document for the same student, open or confirmed by you in the
     last `DONE_DOCUMENTS_HOURS` - the same form scanned twice, or two pages of
     one form that both carry the ID. Both wait for you;
   - a student already marked scanned (`scanned_at` set), by anyone at any
     time - a second form for one child is for you to look at;
   - any of your scans still to send or to read: a form's last part - or,
     with batch codes off, the sheet after its Declaration page - may still
     be on its way.
2. The match is **exact**: the Student ID read by OCR, or typed by you, is the
   student's own ID after the clean-up every Student ID gets - spaces, dashes
   and case set aside and, for a reading that is not shaped like a Student ID
   as read, a letter put right where the shape needs a digit (`Z` for `2`, `O`
   for `0` in the date of birth) or a digit where it needs a letter. That is
   done to the reading before it is looked up, so it counts as exact. A match
   the lookup reaches only by changing the ID it was given - to a near stored
   ID by the ID's shape, or by a fuzzy repair of a misread letter - is not
   exact.
3. **One school** was picked when its pages were sent. With only a city
   ("All Pune schools") it waits.
4. **Every page** of it was sent with the switch on. Pages queued before the
   switch existed count as sent with it off, so they never go up by themselves.

Everything else waits for you to press **Confirm and upload**, as before, and
nothing goes up by itself while the switch is off. The server checks all of
this again before saving anything, looks the student up by the exact ID only,
and marks the student only if nobody has marked them scanned meanwhile, so a
page that is out of date cannot upload the wrong document: it is refused, with
nothing saved. Each document is tried once per visit. If it is refused or the
database is down, it stays on its card for you to confirm, and is not tried
again by itself. If your Google sign-in has run out or there is none (Tether),
nothing is saved; if Drive fails, the document is saved and waits for **Retry
upload**. Either way the page stops uploading by itself for the rest of the
visit - it says so under the switch - since every other document would go the
same way; reload the page to start again. A document that went up this way is
listed as **Uploaded automatically**; its record names you as the person who
confirmed it, and its History entry's source starts with `auto:`.

### The scan queue and the OCR worker

The queue is plain folders on the server, in `data/queue/` (`QUEUE_DIR`,
gitignored): for each page its original, a normalised `page.jpg` (upright,
greyscale, at most 300 dpi), `meta.json` and, once read, `result.json`. A page
is claimed for reading with one atomic rename, so two readers never read the
same page; a reader that dies has its page handed on after 10 minutes, and a
page that fails three times is marked failed and left to a person.

- **Development** (`python run.py`): the web app reads the queue in a thread of
  its own. `OCR_WORKER_IN_PROCESS=1` / `0` in `.env` forces it on or off.
- **Production** (gunicorn): the inbox watcher reads it, one pass per loop,
  so there is no new service to run - just keep `python -m server.inbox_watcher`
  running as before. `python -m server.inbox_watcher --queue-once` reads it
  once and exits. Without a Drive inbox configured, the watcher reads only the
  queue.

## Scanning through the Drive inbox

For a scanner whose own software can save to a folder that Google Drive for
Desktop syncs: each person keeps a folder called **OCR Inbox**
(`DRIVE_INBOX_FOLDER_NAME`) in their Google Drive, shares it with the Vision
service account, and points the scanner's software at it. On the page, pick the
school and press **Start scanning**; the server reads every page that reaches
Drive against that school, and renames each file to say what became of it
(`[saved] Name_ID.jpg`, `[not found] ID.jpg`).

**Stop** decides by when each page reached Drive:

- Pages that reached Drive before Stop are still read for that school, even
  if the server gets to them after Stop (it keeps going back to the folder for
  up to 6 hours).
- Pages that reach Drive after Stop and before the next **Start scanning** are
  not read for any school: they are renamed `[unread] ...` and listed as unread
  in History. Scan them again while a batch is open.
- A page that reaches Drive within 10 minutes of the next Start, for a
  different school, may be the last sheet of the stack before (still
  uploading when the school was switched); if the new school has no match for
  it, it is also looked for in the school before.

The reading is done by the inbox watcher, a separate process on the server:

```bash
python -m server.inbox_watcher            # keep watching (production)
python -m server.inbox_watcher --once     # one pass, then exit
python -m server.inbox_watcher --status   # show open batches
```

## Times are India time

Every time the app writes - `scanned_at`, the History log, scan batches, the
watcher heartbeat, Drive date folders, the debug log - is India Standard Time
(UTC+05:30), whatever machine the code runs on. They all come from
`server/services/clock.py`; nothing uses the server's own clock or SQL `NOW()`.

Rows written before this change are in UTC. `tools/migrate_times_to_ist.py`
shifts them, once, at the moment the deploy goes live - read its docstring first.
It is a dry run unless given `--apply --before "YYYY-MM-DD HH:MM:SS"`.

## API

Every route needs a signed-in session, and every write is refused when it
comes from another site. POSTs with a body send it as `application/json`.

| Route | What it does |
| --- | --- |
| `POST /api/scans` (multipart `file`, `school`, `location`, `auto_upload`, `sender`) | Store and queue a scan, one page per page of the file; each page keeps `auto_upload` (`1` when Upload matched pages automatically is on, off when missing) and `sender`, the scan page's id for itself (32 hex digits, with `-1` after them for a file added by hand; anything else is kept as none). `202 {pages: [{page_id, source_page}]}`. No OCR. |
| `GET /api/batches` | Your documents by batch (newest first) - with batch codes off, one group with no batch, newest scan first - `unsorted` pages (with batch codes off, each with the `declaration_pages` of its form and, for a page that may be the Declaration page the reader missed, `attach_to {page_id, student_name}`), every page with its status (`pages`) and the `queue` counts. |
| `GET /api/documents/<key>` | One document (`B1-12`, or with batch codes off `P` and its Student ID page's id; `~<id>` added once confirmed) with its parts. `auto_upload: true` on an open one the page may confirm by itself - with batch codes off, `auto_upload_pending: true` on one it may once its stack has settled, which the page keeps looking at; `confirmed_automatically` on one it did. |
| `GET /api/pages/<page_id>/image?size=thumb\|full` | The page's scanned image, to its owner only. |
| `POST /api/pages/<page_id>/code {code}` | Set the page's batch code by hand (`""` clears it); `409` with batch codes off. |
| `POST /api/pages/<page_id>/student-id {student_id}` | Set the page's Student ID by hand; looked up read-only. |
| `POST /api/pages/<page_id>/declaration {of}` | With batch codes off: attach the page by hand as the Declaration page of the form straight before it (`of`: that form's first page, as `attach_to` offered it; `""` takes it off). `409` with nothing saved when it is no longer straight after that form's page with no Declaration page, or with batch codes on. |
| `POST /api/documents/<key>/confirm {student_id, page_ids}` | The only write: mark scanned, upload the PDF, log it. `page_ids` are the parts the page was showing; only exactly those are confirmed. `409` (nothing saved) when the document has gained or lost a part since, when not found, still being read, or no Google sign-in; `400` without `page_ids`; `503` when the database is down. A Drive failure answers `200` with `uploaded: false`; a History entry not written after the upload, `uploaded: true, finished: false`. With `automatic: true` (the page confirming by itself) it is done only while the document is still `auto_upload`, for the student it matched, found by the exact ID and not marked scanned meanwhile, with a Google sign-in still in date - otherwise `409` with nothing saved. |
| `POST /api/documents/<key>/upload` | Finish a confirmed document: retry only its Drive upload, or only its History entry (`502` if it is still not finished). |

The older one-shot routes still answer, for scripts: `POST /api/extract` and
`/api/extract-batch` read a file at once, and `POST /api/confirm` saves one
typed Student ID.

```powershell
curl -X POST http://127.0.0.1:5000/api/extract -F "file=@C:\path\to\scan.jpg"
```

Success response (DB match):

```json
{
  "student_id": "MOHSHA011021",
  "confidence": 0.87,
  "form_language": "english",
  "source": "scan.jpg",
  "label_text": "Student ID:",
  "message": "ok",
  "found_in_db": true,
  "student": {
    "student_id": "MOHSHA011021",
    "student_name": "Mohammad Zayn Shahrukh Shaikh",
    "school_name": "SBP",
    "grade_name": "JR.KG",
    "division_name": "A",
    "academic_year": "2026-2027",
    "status": "Active - Enrolled",
    "scanned_at": "2026-07-13 16:30:00"
  }
}
```

Configure DB in `.env` (`DB_HOST`, `DB_USER`, `DB_PASS`, `DB_NAME`, `DB_PORT`).

## Tips for accuracy

- Scan the whole first page, straight through the feeder, so the Student ID
  line is visible.
- The scanner setup above uses 100 dpi black and white, which keeps files
  small and is enough to find the form's layout; if many readings need
  correcting, try 200 dpi.
- Student ID handwriting should be English letters/digits.
- Blank template PDFs in `data/forms/` have no ID value — use filled scans in `data/samples/`.

## Project layout

```text
run.py                     # entrypoint: python run.py (exposes run:app for WSGI)
requirements.txt
server/                     # backend — Flask app package (never touches a scanner)
  __init__.py                 # create_app() factory (points Flask at client/ for templates+static)
  config.py
  inbox_watcher.py            # separate process: reads pages landing in Drive inboxes
  blueprints/                 # route handlers
    ocr_routes.py                # Scanner page, /api/extract, /api/confirm
    batch_routes.py              # the scan queue: /api/scans, /api/batches, confirm
    inbox_routes.py              # Drive inbox Start / Stop / results
    history_routes.py
    auth_routes.py               # /login, /login/google, /login/google/callback, /logout
  services/                   # business logic (OCR pipeline, DB lookup, logging)
    preprocess.py
    field_detector.py
    form_layout.py
    handwriting_ocr.py
    vision_ocr.py
    extract_student_id.py
    student_lookup.py
    history_store.py
    batch_store.py               # Drive inbox batches + watcher heartbeat
    scan_queue.py                # the scan queue on disk (data/queue)
    ocr_worker.py                # reads queued pages in the background
    batch_code.py                # reads and parses the handwritten B1-12 code
    documents.py                 # groups pages into documents; confirm + upload
    drive_inbox.py               # reads the shared "OCR Inbox" folders
    drive_store.py               # saves confirmed pages to the user's Drive
    scan_store.py                # stashes an upload until it is confirmed
    clock.py                     # now_ist(): the one clock every write uses
    debug_log.py
    auth.py                      # login-gate (current_user, require_login)
    oauth.py                     # Authlib Google OAuth2 client
    sso.py                       # Tether single sign-on
client/                     # frontend — server-rendered HTML + CSS
  templates/
    index.html
    history.html
    login.html
    _nav.html
  static/
    style.css
tools/                      # run by hand
  migrate_times_to_ist.py     # one-off: shift old UTC rows to India time
data/
  forms/          # blank EN/HI/MR templates
  samples/        # put real filled scans here
  uploads/        # runtime uploads (gitignored)
  queue/          # the scan queue (gitignored)
logs/             # runtime debug logs + debug crops (gitignored)
```

## Notes

- Max upload size: 100 MB (a full feeder stack as one PDF).
- Accepted types: `.jpg`, `.jpeg`, `.png`, `.pdf`, `.tif`, `.tiff`.
- Engine: form-layout matching + RapidOCR to find the field, Google Cloud Vision for the handwritten value (RapidOCR / EasyOCR fallback; TrOCR off unless `USE_TROCR=1`).
- Blank templates in `data/forms/` correctly return "could not read" until a real filled scan is uploaded.
- Put real scanner scans in `data/samples/` to tune accuracy.
- Requires Python 3.10+ for the server (tested on 3.14). Use the project `.venv`. Scanning computers need only a browser and the scanner's own software.
