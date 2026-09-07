"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_FOLDER = DATA_DIR / "uploads"
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB — a full ADF stack of scans
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

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

# Google OAuth2 sign-in (Google Cloud Console > APIs & Services > Credentials)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
# Only Google accounts on this email domain may sign in — leave blank to allow any
ALLOWED_EMAIL_DOMAIN = os.getenv("ALLOWED_EMAIL_DOMAIN", "").strip().lower()

# Google Drive — scanned uploads land in this folder (created if missing)
DRIVE_OCR_FOLDER_NAME = os.getenv("DRIVE_OCR_FOLDER_NAME", "OCR").strip() or "OCR"
GOOGLE_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
GOOGLE_OAUTH_SCOPES = f"openid email profile {GOOGLE_DRIVE_SCOPE}"

# Scanning straight from the app. The name is matched against what WIA
# reports, and comes from config rather than the request so a browser can
# never choose what gets executed.
SCANNER_NAME = os.getenv("SCANNER_NAME", "SP-1130N")
SCAN_SCRIPT = BASE_DIR / "tools" / "scan.ps1"
# A full ADF stack takes a while to feed; well past that we assume a jam
# or a wedged driver rather than a slow scan.
SCAN_TIMEOUT_SECONDS = int(os.getenv("SCAN_TIMEOUT_SECONDS", "600"))

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
# Below this fraction of dark pixels a page carries no form at all.
# Real forms measure 4-6%; a blank reverse with bleed-through is under 1%.
BLANK_PAGE_INK_RATIO = 0.02
LAYOUT_TRUST_CONFIDENCE = 0.80

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

# Auto mode saves exact matches without asking. A repaired match is only a
# best guess, so it waits for a human unless this is turned on.
AUTO_COMMIT_REPAIRED = os.getenv("AUTO_COMMIT_REPAIRED", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

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
