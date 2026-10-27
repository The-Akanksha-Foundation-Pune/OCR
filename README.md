# Student ID OCR

Extract the handwritten English **Student ID** from Akanksha Parent Consent Forms
and mark the student as scanned. Forms come in three ways: straight from a
**scanner plugged into your own computer** (Windows, macOS or Linux), through a
**Drive inbox** folder your scanner app saves into, or as a **phone photo /
uploaded file** (CamScanner, a PDF from a printer).

Supports English, Hindi, and Marathi form labels. Only the Student ID value is read.

## How it works

1. A form arrives - from the scanner through the scan helper, from your Drive
   inbox, or as an uploaded JPG / PNG / PDF.
2. The app finds the printed Student ID line (EN / HI / MR) from the form's ruled layout.
3. It crops the handwritten value and reads it with Google Cloud Vision, falling
   back to the local RapidOCR readers.
4. It looks that Student ID up in MySQL `active_student_data`, within the school
   you picked, and repairs a near-miss reading only when it is safely unambiguous.
5. Once confirmed, it marks `scanned_at`, saves the page to Google Drive and logs
   the attempt for the History page.

```text
Physical form
  ├─ Scanner on your computer → scan helper (127.0.0.1:8765) → the page in your browser ┐
  ├─ Scanner app → "OCR Inbox" in your Drive → inbox watcher on the server              ├→ OCR → lookup → confirm
  └─ Phone photo / PDF        → upload on the page                                      ┘
```

The server never talks to a scanner, so it runs the same on any OS.

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
3. Bring the forms in by whichever route suits the machine you are at:
   the **Scanner card** (a scanner on this computer), **Start scanning** on the
   Drive inbox panel, or **upload** a photo / PDF.
4. Check each reading, correct it if needed, and confirm.

## Scanning from a scanner on your own computer

A browser cannot reach a USB scanner, and the server is nowhere near one, so a
small **scan helper** runs on the computer the scanner is plugged into. The
Scanner card on the page talks to it at `http://127.0.0.1:8765`, lists the
scanners it can use, and sends each scanned page through `/api/extract` exactly
as an uploaded file - so reading, confirming and saving are unchanged.

**Which scanners work:** any scanner that has a driver for that computer's OS,
or that speaks AirScan (eSCL) over the network:

| OS      | Driver types the helper uses                     | Needs                                        |
|---------|--------------------------------------------------|----------------------------------------------|
| Windows | WIA (built in), TWAIN and AirScan through NAPS2  | Python 3; NAPS2 only for TWAIN / AirScan     |
| macOS   | Image Capture (ICA) and AirScan, through NAPS2   | Python 3, NAPS2, the scanner's Mac driver    |
| Linux   | SANE and AirScan, through NAPS2                  | Python 3, NAPS2, the scanner's SANE driver   |

The card shows each scanner that is ready, and also any USB scanner from a known
maker that is plugged in but has no driver yet, with a link to the maker's
driver page. Press **Check** again after installing a driver.

The Fujitsu / Ricoh **SP-1130N** works over USB on all three: on Windows install
PaperStream IP (it includes the WIA driver), on macOS Ricoh's ICA driver, on
Linux Ricoh's Linux driver. It has no AirScan, so it is never found over the network.

On Windows a WIA scanner hands over each page as it is fed, so reading starts
on page 1 while the rest of the stack is still going through. Through NAPS2
(macOS, Linux, TWAIN, AirScan) the pages arrive together once the whole stack
has been fed.

### Setting up the helper

1. On the Scanner card, download the helper (`/scan-helper.zip`, sign-in
   required). It holds `scan_agent.py`, `scanner_devices.py`, `scan_naps2.py`,
   `scan.ps1` and a `HOW-TO-RUN.txt` with the steps for each OS.
2. Install what the table above says the OS needs. NAPS2 is at
   https://www.naps2.com/ (macOS also: `brew install --cask naps2`).
3. Unzip it anywhere and run, from that folder:
   - Windows: `python scan_agent.py` (or `py scan_agent.py`)
   - macOS / Linux: `python3 scan_agent.py`
4. Leave that window open while scanning. Use Chrome or Edge, and allow access
   to devices on the local network if the browser asks.

From a checkout of this repository, `python tools/scan_agent.py` runs the same helper.

The helper answers only the production addresses and local runs on ports 5000 /
5001. For the app at any other address (an ngrok tunnel, a LAN address such as
`http://192.168.0.10:5000`), start it with `--origin <that address>` - the run
lines in `HOW-TO-RUN.txt` already carry it when the zip was downloaded from such
an address. Without it the page cannot see the helper and says it is not running.

The helper listens on 127.0.0.1 only, answers only the app's own web addresses,
holds no credentials and never contacts the server - the page uploads the pages
with the signed-in person's own session. To check it by hand:

```bash
curl http://127.0.0.1:8765/health
curl "http://127.0.0.1:8765/devices?refresh=1"
```

Optional settings, read by the helper on the scanning computer (not by the server):

| Variable                     | What it does                                                 |
|------------------------------|--------------------------------------------------------------|
| `NAPS2_PATH`                 | Where NAPS2 is, when it is not in a standard place           |
| `SCAN_AGENT_ORIGINS`         | Extra web addresses allowed to use the helper, comma-separated |
| `SCAN_AGENT_TIMEOUT_SECONDS` | Stop a scan that runs longer than this (default 900)         |

## Scanning through the Drive inbox

For a scanner whose own app can save to a folder: each person keeps a folder
called **OCR Inbox** (`DRIVE_INBOX_FOLDER_NAME`) in their Google Drive, synced by
Google Drive for Desktop, shares it with the Vision service account, and points
the scanner app at it. On the page, pick the school and press **Start scanning**;
the server reads every page that lands there against that school until **Stop**,
and renames each file to say what became of it (`[saved] Name_ID.jpg`,
`[not found] ID.jpg`).

The reading is done by the inbox watcher, a separate process on the server:

```bash
python -m server.inbox_watcher            # keep watching (production)
python -m server.inbox_watcher --once     # one pass, then exit
python -m server.inbox_watcher --status   # show open batches
```

## Phone photo or upload

On any device, photograph the form (CamScanner document mode works well) or take
a PDF from a printer, and upload it on the Scanner page. A multi-page PDF is
read page by page.

## Times are India time

Every time the app writes - `scanned_at`, the History log, scan batches, the
watcher heartbeat, Drive date folders, the debug log - is India Standard Time
(UTC+05:30), whatever machine the code runs on. They all come from
`server/services/clock.py`; nothing uses the server's own clock or SQL `NOW()`.

Rows written before this change are in UTC. `tools/migrate_times_to_ist.py`
shifts them, once, at the moment the deploy goes live - read its docstring first.
It is a dry run unless given `--apply --before "YYYY-MM-DD HH:MM:SS"`.

## API

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

- Capture the full first page so the Student ID line is visible.
- Prefer CamScanner document mode or a flat printer scan.
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
    ocr_routes.py                # Scanner page, /api/extract, /api/confirm, /scan-helper.zip
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
tools/                      # run on the scanning computer, or by hand
  scan_agent.py               # the scan helper (HTTP on 127.0.0.1:8765)
  scanner_devices.py          # finds scanners, and USB scanners missing a driver
  scan_naps2.py               # one scan through NAPS2 (macOS, Linux, TWAIN, AirScan)
  scan.ps1                    # one scan through WIA on Windows, page by page
  find-scanner.ps1            # Windows: what scanners and drivers are present
  topdf.py
  migrate_times_to_ist.py     # one-off: shift old UTC rows to India time
data/
  forms/          # blank EN/HI/MR templates
  samples/        # put real filled scans here
  uploads/        # runtime uploads (gitignored)
logs/             # runtime debug logs + debug crops (gitignored)
```

## Notes

- Max upload size: 100 MB (a full feeder stack as one PDF).
- Accepted types: `.jpg`, `.jpeg`, `.png`, `.pdf`.
- Engine: form-layout matching + RapidOCR to find the field, Google Cloud Vision for the handwritten value (RapidOCR / EasyOCR fallback; TrOCR off unless `USE_TROCR=1`).
- Blank templates in `data/forms/` correctly return "could not read" until a real filled scan is uploaded.
- Put real CamScanner / printer scans in `data/samples/` to tune accuracy.
- Requires Python 3.10+ for the server (tested on 3.14). Use the project `.venv`. The scan helper needs only Python 3.9+ and nothing beyond the standard library.
