"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_FOLDER = DATA_DIR / "uploads"
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15 MB
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

# TrOCR model for handwritten English alphanumeric IDs
TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"

# ROI crop padding relative to detected label box
ROI_RIGHT_MULTIPLIER = 8.0
ROI_HEIGHT_MULTIPLIER = 1.8
ROI_LEFT_PADDING_RATIO = 0.05

# Student ID post-process: keep common ID characters
STUDENT_ID_PATTERN = r"[^A-Za-z0-9\-_/]"

# Real IDs look like MOHSHA011021 (letters + digits). Reject form-word OCR junk.
MIN_INK_RATIO = 0.015
MIN_OCR_CONFIDENCE = 0.40
MIN_STUDENT_ID_LENGTH = 8
MAX_STUDENT_ID_LENGTH = 20
# letters then digits, optional separators — matches AFDW active_student_data samples
STUDENT_ID_FORMAT = r"^[A-Z]{3,12}\d{4,12}$"

# Relative value-field ROIs from blank Akanksha Parent Consent Form PDFs
# (x0, y0, x1, y1) as fractions of page width/height — value underscores only
TEMPLATE_VALUE_ROIS = {
    "english": (0.24, 0.300, 0.62, 0.330),
    "hindi": (0.24, 0.425, 0.62, 0.455),
    "marathi": (0.24, 0.442, 0.62, 0.472),
}
