"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_FOLDER = DATA_DIR / "uploads"


def _env_path(name: str, default: Path) -> Path:
    """A folder from .env, relative paths taken from the repository root."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def _env_flag(name: str) -> bool | None:
    """A yes/no setting from .env, or None when it is not set at all."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes")


# The scan queue (services/scan_queue.py). Scanning only stores pages here;
# the OCR worker reads them in the background and a person confirms each
# document before anything reaches the database or Drive. Plain folders and
# JSON, no database tables, so a page waiting for OCR survives a restart.
QUEUE_DIR = _env_path("QUEUE_DIR", DATA_DIR / "queue")
# Where the OCR worker runs. In development a thread inside the web app
# reads the queue; unset, that follows Flask's debug mode (on under run.py).
# In production gunicorn starts several workers, so the inbox watcher reads
# the queue instead and this stays off.
OCR_WORKER_IN_PROCESS = _env_flag("OCR_WORKER_IN_PROCESS")
# A page whose reader stopped answering (a crash, a killed process) is
# handed to another reader after this long.
QUEUE_CLAIM_STALE_SECONDS = 10 * 60
# Tries at reading one page before it is marked failed and left to a person.
QUEUE_MAX_ATTEMPTS = 3
# How large a stored page may be. A 600 dpi colour scan is ~100 MB as a
# three-channel array; nothing in the reader needs more than 300 dpi, and a
# greyscale page that size is a few megabytes.
PAGE_MAX_DPI = 300
PAGE_MAX_LONG_SIDE = 3600
# PDF pages are rendered at this resolution, the one the reader was tuned on.
PAGE_PDF_DPI = 200
# The reader itself works on at most this many pixels down the long side
# (about 200 dpi on A4): the layout match and the crops are all fractions of
# the page, and the full-page passes cost in proportion to its area.
OCR_MAX_LONG_SIDE = 2400
# Where the handwritten batch code ("B1-12") is written, as fractions
# (x0, y0, x1, y1) of the upright page: the top-right corner, where the forms
# carry their circled serial, and the top-left, where some prints put it.
BATCH_CODE_REGION = (0.72, 0.0, 1.0, 0.09)
BATCH_CODE_FALLBACK_REGION = (0.0, 0.0, 0.30, 0.09)
# Confirmed and uploaded documents stay on the scan page for this long, so
# the person can see what went to Drive; History keeps the full record.
DONE_DOCUMENTS_HOURS = int(os.getenv("DONE_DOCUMENTS_HOURS") or "72")
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB — a full ADF stack of scans
# TIFF because macOS Image Capture and many scanners' own software save it
# by default; a multi-page TIFF is read page by page, like a PDF.
# BMP and WebP are what some phone and Windows tools save.
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".bmp", ".tif", ".tiff", ".webp"}

# MySQL RDS (AFDW) — from .env
DB_HOST = os.getenv("DB_HOST", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
STUDENT_TABLE = "active_student_data"
MARK_SCANNED_ON_LOOKUP = True

# Staff/employee directory — used to gate app access beyond the email domain.
# Comma-separated allowlists from .env (OR logic: either column may match).
EMPLOYEES_TABLE = os.getenv("EMPLOYEES_TABLE", "employees")
EMPLOYEE_EMAIL_COLUMN = "work_email"


def _csv_set(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


ALLOWED_SCHOOL_ACRONYMS = _csv_set(
    os.getenv("ALLOWED_SCHOOL_ACRONYMS", "Developer")
)
ALLOWED_DESIGNATIONS = _csv_set(
    os.getenv("ALLOWED_DESIGNATIONS", "School Administration")
)

# Flask session signing — required for the login cookie to work
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "")
# The session cookie is marked Secure whenever the request itself came over
# https (behind nginx that is X-Forwarded-Proto). Set this to 1 to insist on
# it regardless - only on a host that is never reached over plain http, or
# nobody there could stay signed in.
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Google OAuth2 sign-in (Google Cloud Console > APIs & Services > Credentials)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
# Only Google accounts on this email domain may sign in — leave blank to allow any
ALLOWED_EMAIL_DOMAIN = os.getenv("ALLOWED_EMAIL_DOMAIN", "").strip().lower()

# Local development only. When set, the login gate is skipped and pages open
# without Google. Leave unset on the live server — the OAuth client there
# only accepts the production redirect URI.
LOCAL_OPEN_ACCESS = os.getenv("LOCAL_OPEN_ACCESS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Tether SSO — must equal JWT_SECRET in the Tether server .env. Tether signs a
# 10-minute token and appends it to this app's URL; see services/sso.py.
# Unset means no token is ever accepted, and only Google sign-in works.
INTEGRATION_JWT_SECRET = os.getenv("INTEGRATION_JWT_SECRET", "").strip()

# Google Drive — scanned uploads land in this folder (created if missing)
DRIVE_OCR_FOLDER_NAME = os.getenv("DRIVE_OCR_FOLDER_NAME", "OCR").strip() or "OCR"

# Scan-to-Drive inbox. Each person keeps a folder of this name in their own
# Drive, shares it with the Vision service account, and points their scanner
# app at it. The watcher (server/inbox_watcher.py) reads pages from there.
DRIVE_INBOX_FOLDER_NAME = os.getenv("DRIVE_INBOX_FOLDER_NAME", "OCR Inbox").strip() or "OCR Inbox"
INBOX_POLL_SECONDS = int(os.getenv("INBOX_POLL_SECONDS", "5"))
GOOGLE_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
GOOGLE_OAUTH_SCOPES = f"openid email profile {GOOGLE_DRIVE_SCOPE}"

# TrOCR model for handwritten English alphanumeric IDs.
#
# Off by default: on real Fujitsu scans it was the slowest recogniser by a
# wide margin (~4s per crop, plus a ~90s first load) while contributing
# nothing the others did not already find - on one page it read "a member
# of the House of Representatives" off a Student ID field. RapidOCR plus
# the database repair carried every successful read. Set USE_TROCR=1 in
# .env to bring it back.
TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"
USE_TROCR = os.getenv("USE_TROCR", "0").strip().lower() in ("1", "true", "yes")

# ROI crop padding relative to detected label box
ROI_RIGHT_MULTIPLIER = 8.0
ROI_HEIGHT_MULTIPLIER = 1.8
ROI_LEFT_PADDING_RATIO = 0.05

# Parents on Hindi/Marathi forms sometimes write the Student ID in
# Devanagari rather than Latin. The digits map one-to-one, so they can be
# folded to ASCII with no ambiguity; the letters are a name transliteration
# (MOHSHA = MOHammad SHAikh) and only ever produce a rough guess, which is
# why DEVANAGARI_SUFFIX_LOOKUP leans on the digits to shortlist and uses
# the letters merely to rank.
DEVANAGARI_DIGITS = "०१२३४५६७८९"
DEVANAGARI_SUFFIX_LOOKUP = True
# Ranked candidates to offer when a Devanagari ID is resolved by its digits
MAX_DEVANAGARI_CANDIDATES = 5

# Google Cloud Vision for the handwritten value. The local recognisers read
# ~31% of forms exactly; the rest need fuzzy repair, which is what forces a
# human to confirm. Vision's handwriting model is much stronger, and only
# the cropped ID field is ever sent - under 2% of the page, carrying a
# 12-character code and no personal details.
#
# Leave GOOGLE_VISION_CREDENTIALS unset and nothing changes: the local
# readers run exactly as they do today.
VISION_CREDENTIALS = os.getenv("GOOGLE_VISION_CREDENTIALS", "").strip()
VISION_ENABLED = os.getenv("VISION_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
VISION_TIMEOUT_SECONDS = int(os.getenv("VISION_TIMEOUT_SECONDS", "20"))
# A Vision reading is worth more than a local one, but not so much that a
# confident local read is ignored outright.
VISION_CONFIDENCE_BONUS = 0.15

# Throughput. At a few hundred forms a batch the per-page cost dominates
# everything else, and most of it is spent running recognisers that add
# nothing once an earlier one has produced a clean reading.
#
# FAST_MODE: stop as soon as a reading is good enough rather than always
#   running every recogniser on every crop.
# LAYOUT_TRUST_CONFIDENCE: above this the rule-ladder match is reliable
#   enough to skip the full-page label OCR, which costs ~3.7s a page and is
#   only there to locate a field the layout already found.
FAST_MODE = os.getenv("FAST_MODE", "1").strip().lower() in ("1", "true", "yes")
# Scanner JPEGs arrive near 600 DPI (about 5000x7000). PDFs are already
# rendered at 200 DPI, which is what the field crop and recognisers were
# tuned on. Anything taller than this is shrunk before OCR. The Student ID
# crop stays sharp enough, and EasyOCR stops spending most of the page on
# pixels it does not use.
MAX_OCR_LONG_SIDE = int(os.getenv("MAX_OCR_LONG_SIDE", "2400"))
# Debug crops under logs/crops. Off by default: writing one JPEG per
# recogniser pass is pure overhead on a batch.
OCR_SAVE_CROPS = os.getenv("OCR_SAVE_CROPS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Below this fraction of dark pixels a page carries no form at all.
# Real forms measure 4-6%; a blank reverse with bleed-through is under 1%.
BLANK_PAGE_INK_RATIO = 0.02
# Student ID box with the printed underline removed. Measured on real scans
# (28 Oct 2026): an empty box is about 0.006, the faintest written ID about
# 0.076. Anything below this is empty. At or above it, including a faint
# pencil line, the page is read normally.
BLANK_ID_INK_RATIO = float(os.getenv("BLANK_ID_INK_RATIO", "0.02"))
# A horizontal run this wide, as a fraction of the crop, is the printed rule
# rather than a letter stroke.
BLANK_ID_RULE_WIDTH_FRACTION = 0.25
# Declaration pages ("घोषणा") are recognised from the top of the page only.
DECLARATION_TOP_FRACTION = 0.15
DECLARATION_WORK_WIDTH = 900
# matchTemplate score and how far from the left edge the heading may sit.
# On the sample set, declaration pages scored 0.75-0.97 at the left;
# form pages scored at most 0.62.
DECLARATION_MATCH_MIN = float(os.getenv("DECLARATION_MATCH_MIN", "0.72"))
DECLARATION_MATCH_MAX_X_FRACTION = 0.25
# Width of the page the heading image was cropped from, so a 600 DPI scan
# and a 200 DPI PDF scale the same template.
DECLARATION_TEMPLATE_PAGE_WIDTH = 1655
DECLARATION_HEADING_PATH = DATA_DIR / "labels" / "ghoshana.png"
# Used only when the heading image does not match and the page has no form
# ladder. RapidOCR is run on the top strip, never the whole page.
DECLARATION_PHRASES = (
    "घोषणा",
    "संमतीची पुष्टी",
    "सहमति की पुष्टि",
)
LAYOUT_TRUST_CONFIDENCE = 0.80
# LAYOUT_TRUST_ERROR: a rule-ladder match at or under this mean row error has
# found the form's lines precisely, whichever language it turns out to be.
# The Student ID sits on the same rule in every template, so the field is
# located even when English and Marathi are a near tie - and a near tie is
# what sent three pages in four through a 5-38s full-page OCR for nothing.
# Typical clean scans match at 0.0003-0.0011; rejection is at 0.020.
LAYOUT_TRUST_ERROR = 0.0025

# Fuzzy-repair safety. A repaired match is a guess at which of 16,000+ real
# students a misread ID belongs to, and a wrong guess silently marks the
# wrong child as scanned - so these are deliberately strict.
#
# MIN_REPAIR_SCORE: how similar the reading must be to the winning ID.
# MIN_REPAIR_MARGIN: how far clear of the runner-up it must be. Ties are the
#   dangerous case - AARSAY150821 and another ID once tied at 0.783.
# MAX_REPAIR_LENGTH_DELTA: IDs are 6 letters + 6 digits, so a reading of the
#   wrong length lost or invented characters and is not safe to repair from.
MIN_REPAIR_SCORE = 0.80
MIN_REPAIR_MARGIN = 0.08
MAX_REPAIR_LENGTH_DELTA = 1

# Student ID post-process: keep common ID characters
STUDENT_ID_PATTERN = r"[^A-Za-z0-9\-_/]"

# Real IDs look like MOHSHA011021 (letters + digits). Reject form-word OCR junk.
MIN_INK_RATIO = 0.015
MIN_OCR_CONFIDENCE = 0.40
MIN_STUDENT_ID_LENGTH = 8
MAX_STUDENT_ID_LENGTH = 20
# letters then digits, optional separators — matches AFDW active_student_data samples
STUDENT_ID_FORMAT = r"^[A-Z]{3,12}\d{4,12}$"

# Relative value-field ROIs (x0, y0, x1, y1) as fractions of page
# width/height, covering the handwritten value underscores only.
# Measured off the blank template PDFs' own text layer (page 1, the
# underscore run after the Student ID label) rather than eyeballed, so the
# crop lands on the value line and not the Grade line above it.
TEMPLATE_VALUE_ROIS = {
    "english": (0.200, 0.3041, 0.450, 0.3209),
    "hindi": (0.242, 0.4266, 0.600, 0.4462),
    "marathi": (0.243, 0.4439, 0.601, 0.4635),
}

# Vertical breathing room around the ROI, as a fraction of its height, so
# handwriting that overshoots the underscore still gets caught. Rows sit
# ~0.029 of page height apart, so anything much above 0.4 swallows the
# neighbouring line.
TEMPLATE_ROI_PAD_RATIO = 0.40

# Fill-in rule ladders measured off the blank template PDFs (page 1),
# as fractions of page height. The Student ID is the 4th rule from the top
# in every language, which is what makes the ladder worth matching: it
# identifies the form and locates the value box in one pass, and it holds
# up under the framing drift of a real scan in a way fixed page fractions
# do not. `value_height` is the nominal height of a handwritten value.
FORM_RULE_FINGERPRINTS = {
    "english": {
        "rows": [0.1683, 0.2465, 0.2754, 0.3044, 0.3826, 0.4115, 0.4403],
        "student_id_index": 3,
        "value_height": 0.0175,
    },
    "hindi": {
        "rows": [0.2515, 0.3588, 0.3941, 0.4294, 0.5369, 0.5722, 0.6075, 0.6940],
        "student_id_index": 3,
        "value_height": 0.0200,
    },
    "marathi": {
        "rows": [0.3040, 0.3863, 0.4164, 0.4467, 0.5272, 0.5575, 0.5876],
        "student_id_index": 3,
        "value_height": 0.0200,
    },
}

# Pages per printed form, by language — the Student ID is only ever on
# page 1, so a batch-scanned stack has (pages_per_form - 1) continuation
# pages after each form that carry no ID at all.
FORM_PAGE_COUNTS = {
    "english": 2,
    "hindi": 3,
    "marathi": 3,
}
