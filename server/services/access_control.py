"""Gate app access to specific staff, on top of the email-domain check.

A signed-in Google account is only let in if its matching employees row
has school_acronym in ALLOWED_SCHOOL_ACRONYMS or designation in
ALLOWED_DESIGNATIONS (from .env) — either condition is enough.
"""

from __future__ import annotations

from server.config import (
    ALLOWED_DESIGNATIONS,
    ALLOWED_SCHOOL_ACRONYMS,
    EMPLOYEE_EMAIL_COLUMN,
    EMPLOYEES_TABLE,
)
from server.services.debug_log import debug_log
from server.services.student_lookup import get_connection


def is_authorized_staff(email: str) -> bool:
    """True if email matches an allowed school_acronym or designation."""
    cleaned = (email or "").strip().lower()
    if not cleaned:
        return False

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT school_acronym, designation
                FROM {EMPLOYEES_TABLE}
                WHERE LOWER(TRIM({EMPLOYEE_EMAIL_COLUMN})) = %s
                LIMIT 1
                """,
                (cleaned,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        debug_log(f"[ACCESS] no employee record for {cleaned!r} — denied")
        return False

    school_acronym = (row.get("school_acronym") or "").strip()
    designation = (row.get("designation") or "").strip()

    authorized = (
        school_acronym in ALLOWED_SCHOOL_ACRONYMS
        or designation in ALLOWED_DESIGNATIONS
    )
    if not authorized:
        debug_log(
            f"[ACCESS] {cleaned!r} denied — school_acronym={school_acronym!r} "
            f"designation={designation!r}"
        )
    return authorized
