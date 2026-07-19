# Student ID OCR

Extract the handwritten English **Student ID** from Akanksha Parent Consent Forms scanned with **CamScanner (phone)** or a **printer / flatbed scanner**.

Supports English, Hindi, and Marathi form labels. Only the Student ID value is read.

## How it works

1. User scans the physical form (CamScanner or printer).
2. Upload the JPG / PNG / PDF to the web page or API.
3. App finds the printed Student ID label (EN / HI / MR).
4. Crops the handwritten value and reads it with TrOCR (handwriting model).
5. Looks up that Student ID in MySQL `active_student_data`.
6. Returns Student ID + student details as JSON (and marks `scanned_at`).

```text
Physical form
  ├─ Phone CamScanner  → JPG/PDF
  └─ Printer scanner   → JPG/PDF
           ↓
     Upload to this app
           ↓
   student_id in response
```

## Setup (Windows)

```powershell
cd d:\NewOCRRealForm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Open http://127.0.0.1:5000

First extract downloads RapidOCR + TrOCR models (can take a few minutes).

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

1. Open the home page.
2. Choose a CamScanner export or printer scan.
3. Click **Extract Student ID**.

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
server/                     # backend — Flask app package
  __init__.py                 # create_app() factory (points Flask at client/ for templates+static)
  config.py
  blueprints/                 # route handlers
    ocr_routes.py
    history_routes.py
    auth_routes.py               # /login, /login/google, /login/google/callback, /logout
  services/                   # business logic (OCR pipeline, DB lookup, logging)
    preprocess.py
    field_detector.py
    handwriting_ocr.py
    extract_student_id.py
    student_lookup.py
    history_store.py
    debug_log.py
    auth.py                      # login-gate (current_user, require_login)
    oauth.py                     # Authlib Google OAuth2 client
client/                     # frontend — server-rendered HTML + CSS
  templates/
    index.html
    history.html
    login.html
    _nav.html
  static/
    style.css
data/
  forms/          # blank EN/HI/MR templates
  samples/        # put real filled scans here
  uploads/        # runtime uploads (gitignored)
logs/             # runtime debug logs + debug crops (gitignored)
```

## Notes

- Max upload size: 15 MB.
- Accepted types: `.jpg`, `.jpeg`, `.png`, `.pdf` (first page).
- Engine: RapidOCR (printed EN labels) + form template ROIs (Hindi/Marathi) + TrOCR handwritten English (value).
- Blank templates in `data/forms/` correctly return "could not read" until a real filled scan is uploaded.
- Put real CamScanner / printer scans in `data/samples/` to tune accuracy.
- Requires Python 3.10+ (tested on 3.14). Use the project `.venv`.
