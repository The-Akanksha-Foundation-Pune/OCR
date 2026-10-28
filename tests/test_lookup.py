"""Student lookup, ID shape and Vision privacy fixes (bugs 3, 12, 15, 16, 26, 28).

Nothing here touches the real database or Google: the lookup runs against
conftest's FakeDB (or, where the SQL itself has to be evaluated, an
in-memory SQLite stand-in), and every recogniser (Vision included) is
replaced by a stand-in that records what it was shown.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pymysql  # type: ignore[import-untyped]
import pytest

from server.services import extract_student_id as ex
from server.services import id_shape, student_lookup, vision_ocr
from server.services.field_detector import LabelHit
from server.services.form_layout import FormMatch

if TYPE_CHECKING:
    from flask.testing import FlaskClient

    from tests.conftest import FakeDB

# A parent's name, email and mobile run together, as Vision returned them
# off a crop that caught the contact lines (synthetic values).
CONTACT_LINES = "RIYASANJAYPATILRIYAPATIL12GMAILCOM9876543210"

STUDENT_COLUMNS = (
    "student_id",
    "student_name",
    "school_name",
    "grade_name",
    "division_name",
    "academic_year",
    "gender",
    "status",
    "scanned_at",
)


# ------------------------------------------------------------------ helpers
def _student(student_id: str, school: str) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(STUDENT_COLUMNS)
    row.update(student_id=student_id, student_name=f"Child {student_id}", school_name=school)
    return row


def _roster_handler(
    roster: dict[str, str],
) -> Callable[[str, Any], tuple[list[dict[str, Any]], int | None]]:
    """Answer student_lookup's SQL from {student_id: school}.

    Every SELECT that names schools is held to them, exactly as MySQL would,
    so a test can see a lookup reach outside its scope.
    """

    def in_scope(school: str, params: tuple[Any, ...], first: int) -> bool:
        wanted = {str(p).upper() for p in params[first:]}
        return not wanted or school.upper() in wanted

    def handler(sql: str, params: Any) -> tuple[list[dict[str, Any]], int | None]:
        params = tuple(params or ())
        if re.match(r"SELECT student_id FROM active_student_data", sql):
            ids = [sid for sid, school in roster.items() if in_scope(school, params, 0)]
            return [{"student_id": sid} for sid in ids], None
        if "WHERE UPPER(TRIM(student_id)) = UPPER(%s)" in sql and sql.startswith("SELECT student_id,"):
            wanted = str(params[0]).upper()
            school = roster.get(wanted)
            if school is not None and in_scope(school, params, 1):
                return [_student(wanted, school)], None
            return [], None
        if sql.startswith("UPDATE active_student_data SET scanned_at"):
            return [], 1
        if sql.startswith("SELECT scanned_at FROM"):
            return [{"scanned_at": None}], None
        return [], 0

    return handler


@pytest.fixture(autouse=True)
def _private_debug_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Keep this module's synthetic contact details out of logs/ocr_debug.log.

    debug_log appends to the real log, which is the person's own debugging
    history; these tests feed it fake names, emails and phone numbers.
    """
    from server.services import debug_log

    log_file = tmp_path / "ocr_debug.log"
    monkeypatch.setattr(debug_log, "LOG_DIR", tmp_path)
    monkeypatch.setattr(debug_log, "LOG_FILE", log_file)
    return log_file


@pytest.fixture
def fresh_id_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """student_lookup caches the candidate list per scope for five minutes."""
    monkeypatch.setattr(student_lookup, "_id_cache", {})
    monkeypatch.setattr(student_lookup, "_id_cache_at", {})


# ABMPS has several children born in October 2021. MOHSHA011021 - the
# child whose form it really is - is in another school.
ABMPS_ROSTER = {
    "MOHSIN151021": "ABMPS",
    "PRIYAD041021": "ABMPS",
    "ROHANK221021": "ABMPS",
    "SANIKA301021": "ABMPS",
    "MOHSHA011021": "OTHER",
}


# ------------------------------------------------- bug 3: digit-suffix lookup
def test_digit_suffix_ignores_a_reading_missing_part_of_the_date(
    fake_db: FakeDB,
    fresh_id_cache: None,
) -> None:
    fake_db.handler = _roster_handler(ABMPS_ROSTER)

    found = student_lookup.find_student_by_id("MOHSHA1021", mark_scanned=True, school="ABMPS")

    assert found is None
    assert not fake_db.executed(r"^UPDATE active_student_data SET scanned_at")


def test_digit_suffix_refuses_a_reading_too_short_to_guess_from(
    fake_db: FakeDB,
    fresh_id_cache: None,
) -> None:
    # Six digits, but half the letters gone: MOH matches MOHSIN perfectly on
    # partial_ratio, which is exactly the guess constrained repair refuses.
    fake_db.handler = _roster_handler(ABMPS_ROSTER)
    with student_lookup.get_connection().cursor() as cur:
        assert student_lookup._digit_suffix_lookup(cur, "MOH151021", scope=["ABMPS"]) is None


def test_digit_suffix_still_matches_within_the_school_only(
    fake_db: FakeDB,
    fresh_id_cache: None,
) -> None:
    roster = dict(ABMPS_ROSTER, LONDNA011021="ABMPS", LONDNA011022="OTHER")
    fake_db.handler = _roster_handler(roster)

    with student_lookup.get_connection().cursor() as cur:
        row = student_lookup._digit_suffix_lookup(cur, "L0NDN011021", scope=["ABMPS"])

    assert row is not None and row["student_id"] == "LONDNA011021"
    # The shortlist and the final fetch were both held to the school.
    shortlist = fake_db.executed(r"^SELECT student_id FROM active_student_data WHERE")
    assert shortlist and shortlist[-1][1] == ("ABMPS",)
    final = fake_db.executed(r"^SELECT student_id, .* AND UPPER\(TRIM\(school_name\)\) IN")
    assert final and final[-1][1] == ("LONDNA011021", "ABMPS")


def test_digit_suffix_cannot_reach_a_child_in_another_school(
    fake_db: FakeDB,
    fresh_id_cache: None,
) -> None:
    fake_db.handler = _roster_handler({"LONDNA011021": "OTHER", "PRIYAD041021": "ABMPS"})
    with student_lookup.get_connection().cursor() as cur:
        assert student_lookup._digit_suffix_lookup(cur, "L0NDN011021", scope=["ABMPS"]) is None


# ----------------------------------------------------- bug 12: the ID shape
@pytest.mark.parametrize(
    "reading",
    [
        "AADMIS310714",
        "aadmis.310714",
        "AADMIS310714_A1",
        "AADMI 310714",
        "TAROUR29081S",
        "ABCD12345",  # the shortest the contract allows
        "ABCDEFG1234567A1",  # and the longest
    ],
)
def test_real_id_shapes_pass(reading: str) -> None:
    assert id_shape.looks_like_student_id(reading)


@pytest.mark.parametrize(
    "reading",
    [
        CONTACT_LINES,
        "PRIYA9876543210",
        "RIYA12GMAILCOM",
        "9876543210",
        "SANJAYPATIL",
        "",
        "ABC123456",  # three letters
        "MOHSHA1021",  # four digits: a month and a year, not a date of birth
        "ABCDEFGH123456",  # eight letters
        "AADMIS310714AB",  # two trailing letters
    ],
)
def test_names_emails_and_phone_numbers_fail(reading: str) -> None:
    assert not id_shape.looks_like_student_id(reading)


@pytest.mark.parametrize("reading", ["AADMIS310714", "M0HSHA011021", "aadmis.310714"])
def test_id_shaped_readings_are_logged_as_read(reading: str) -> None:
    assert id_shape.loggable_reading(reading) == repr(reading)


@pytest.mark.parametrize(
    "reading",
    [CONTACT_LINES, "Riya Patil 9876543210", "Email: riya12@gmail.com", "9876543210"],
)
def test_anything_else_is_logged_by_length_only(reading: str) -> None:
    shown = id_shape.loggable_reading(reading)

    assert str(len(reading)) in shown
    for fragment in ("riya", "patil", "gmail", "9876543210", "543210"):
        assert fragment not in shown.lower()


def test_best_candidate_never_picks_contact_details() -> None:
    picked = ex._best_ocr_candidate([(CONTACT_LINES, 0.7, "english", "crop-vision-noisy:x")])
    assert picked is None


def test_best_candidate_rescues_a_shape_confusion() -> None:
    picked = ex._best_ocr_candidate([("M0HSHA011021", 0.6, "english", "crop-vision-noisy:x")])
    assert picked is not None and picked[0] == "MOHSHA011021"


def test_confirm_refuses_contact_details_and_logs_nothing(fake_db: FakeDB) -> None:
    with pytest.raises(ex.ExtractError) as caught:
        ex.confirm_student_id(CONTACT_LINES, source="inbox:7")

    assert caught.value.status_code == 400
    assert fake_db.statements == []  # no lookup, no ocr_scan_log row


def test_confirm_api_refuses_contact_details(
    client: FlaskClient,
    fake_db: FakeDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads: list[Any] = []
    monkeypatch.setattr(
        "server.blueprints.ocr_routes.upload_scan_to_drive", lambda *a, **k: uploads.append(a)
    )
    response = client.post("/api/confirm", json={"student_id": "Priya 9876543210"})

    assert response.status_code == 400
    assert not fake_db.executed(r"INSERT INTO ocr_scan_log")
    assert uploads == []


# ---- bug 12 + 26: a page whose only reading is the parent's contact lines
class FakeReaders:
    """Stand-ins for every recogniser, recording which crops each one saw."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, page: np.ndarray) -> None:
        self.page = page
        self.tags: dict[int, str] = {}
        self.vision_saw: list[str] = []
        self.local_saw: list[str] = []
        self.vision_reads: dict[str, list[tuple[str, float]]] = {}
        self.rapid_reads: dict[str, list[tuple[str, float]]] = {}

        patch = monkeypatch.setattr
        patch(ex, "FAST_MODE", True)
        patch(ex, "USE_TROCR", False)
        patch(ex, "preprocess_for_ocr", lambda image: image)
        patch(ex, "_is_blank_page", lambda image: False)
        patch(ex, "deskew", lambda image: (image, 0.0))
        patch(ex, "_save_debug_crop", lambda crop, tag: None)
        patch(ex, "vision_configured", lambda: True)
        patch(ex, "recognize_with_vision", self._vision)
        patch(ex, "get_rapid_ocr", lambda: self._rapid)
        patch(ex, "recognize_with_easyocr", lambda crop: [])
        patch(ex, "recognize_devanagari_id", lambda crop: [])
        patch(ex, "crop_value_roi", lambda image, roi: self.crop("layout"))
        patch(ex, "crop_template_value", lambda image, lang: self.crop(f"template_{lang}"))
        patch(ex, "crop_student_id_value", lambda image, label: self.crop("label"))
        self.label_height = 0.04
        self.layout_height = 0.035

    def crop(self, tag: str) -> np.ndarray:
        share = {"label": self.label_height, "layout": self.layout_height}.get(tag, 0.035)
        height = max(1, int(self.page.shape[0] * share))
        array = np.zeros((height, 200, 3), dtype=np.uint8)
        self.tags[id(array)] = tag
        return array

    def _vision(self, crop: np.ndarray) -> list[tuple[str, float]]:
        tag = self.tags[id(crop)]
        self.vision_saw.append(tag)
        return list(self.vision_reads.get(tag, []))

    def _rapid(self, crop: np.ndarray) -> tuple[list[Any], None]:
        tag = self.tags[id(crop)]
        self.local_saw.append(tag)
        return [[None, text, conf] for text, conf in self.rapid_reads.get(tag, [])], None


def _page() -> np.ndarray:
    return np.full((1000, 700, 3), 255, dtype=np.uint8)


def _english_layout() -> FormMatch:
    return FormMatch(
        language="english",
        confidence=0.95,
        student_id_roi=(0.2, 0.29, 0.45, 0.32),
        rule_count=7,
        error=0.001,
        source="rules",
        runner_up_language="hindi",
    )


def test_contact_lines_from_vision_are_not_a_student_id(monkeypatch: pytest.MonkeyPatch) -> None:
    readers = FakeReaders(monkeypatch, _page())
    monkeypatch.setattr(ex, "identify_form", lambda image: _english_layout())
    readers.vision_reads["layout"] = [(CONTACT_LINES, 0.9)]

    with pytest.raises(ex.ExtractError) as caught:
        ex._extract(readers.page, source="page-006.jpg")

    assert caught.value.status_code == 422
    assert "No Student ID found on this page" in caught.value.message
    assert "looks blank" not in caught.value.message


def test_batch_page_with_only_contact_lines_has_no_student_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readers = FakeReaders(monkeypatch, _page())
    monkeypatch.setattr(ex, "identify_form", lambda image: _english_layout())
    monkeypatch.setattr(ex, "load_pages_from_bytes", lambda data, name: [readers.page])
    readers.vision_reads["layout"] = [(CONTACT_LINES, 0.9)]

    [page] = ex.extract_student_ids_from_upload_batch(b"%PDF", "stack.pdf")

    assert page.student_id == ""
    assert page.error and "No Student ID found on this page" in page.error
    assert CONTACT_LINES not in json.dumps(page.to_dict())


# ------------------------------------------ bug 26: what reaches Cloud Vision
def test_template_crops_never_go_to_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    # A template box is a fixed page fraction. On a real scan (scale 0.85)
    # even the identified form's box sits rows below the located field, on
    # the parent's lines; the located field itself has already been sent.
    readers = FakeReaders(monkeypatch, _page())
    monkeypatch.setattr(ex, "identify_form", lambda image: _english_layout())

    with pytest.raises(ex.ExtractError):
        ex._extract(readers.page, source="scan")

    assert readers.vision_saw == ["layout"]
    # Every box is still read - on this machine only.
    assert {"template_english", "template_hindi", "template_marathi"} <= set(readers.local_saw)


def _email_line() -> LabelHit:
    return LabelHit(
        text="Email: riya12@gmail.com",
        language="english",
        box=[[300, 600], [600, 600], [600, 612], [300, 612]],
        confidence=0.9,
        source="ocr",
        parsed_student_id="RIYA12GMAILCOM",
    )


def _page006_layout(confidence: float) -> FormMatch:
    """The layout match page-006 really got: English, but a loose fit.

    Its rules matched at error 0.0064 (scale 1.08, where every other page in
    the log matched at 0.85-0.86 and error <= 0.0022), so the "Student ID"
    ROI landed on the parent's mobile number.
    """
    return FormMatch(
        language="english",
        confidence=confidence,
        student_id_roi=(0.139, 0.3368, 0.491, 0.3642),
        rule_count=12,
        error=0.0064,
        source="rules",
        runner_up_language="marathi",
    )


def test_page006_loose_layout_crop_is_not_sent_to_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug's own trigger page, replayed: a tied match and the email line."""
    readers = FakeReaders(monkeypatch, _page())
    readers.layout_height = 0.049  # under VISION_MAX_CROP_HEIGHT, as it was
    monkeypatch.setattr(ex, "identify_form", lambda image: _page006_layout(confidence=0.3))
    monkeypatch.setattr(ex, "find_student_id_label", lambda image: _email_line())
    readers.vision_reads["layout"] = [("9876543210", 0.93)]

    with pytest.raises(ex.ExtractError):
        ex._extract(readers.page, source="page-006.jpg")

    assert readers.vision_saw == []
    assert {"label", "layout", "template_english"} <= set(readers.local_saw)


def test_a_clear_language_call_does_not_vouch_for_a_loose_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Confidence only measures how far the runner-up language trails; it
    # says nothing about whether the rules were matched in the right place.
    readers = FakeReaders(monkeypatch, _page())
    monkeypatch.setattr(ex, "identify_form", lambda image: _page006_layout(confidence=0.9))
    readers.vision_reads["layout"] = [("9876543210", 0.93)]

    with pytest.raises(ex.ExtractError):
        ex._extract(readers.page, source="scan")

    assert readers.vision_saw == []
    assert "layout" in readers.local_saw


def test_a_closely_matched_layout_still_goes_to_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    readers = FakeReaders(monkeypatch, _page())
    monkeypatch.setattr(ex, "identify_form", lambda image: _english_layout())
    readers.vision_reads["layout"] = [("AADMIS310714", 0.9)]

    result = ex._extract(readers.page, source="scan")

    assert readers.vision_saw == ["layout"]
    assert result.student_id == "AADMIS310714"


def test_unidentified_page_keeps_every_template_crop_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readers = FakeReaders(monkeypatch, _page())
    monkeypatch.setattr(ex, "identify_form", lambda image: None)
    monkeypatch.setattr(ex, "find_student_id_label", lambda image: None)
    readers.rapid_reads["template_english"] = [("AADMIS310714", 0.8)]

    result = ex._extract(readers.page, source="scan")

    assert readers.vision_saw == []
    assert result.student_id == "AADMIS310714"  # the local readers still work
    assert {"template_english", "template_hindi", "template_marathi"} <= set(readers.local_saw)


def test_label_on_the_email_line_is_not_sent_to_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    readers = FakeReaders(monkeypatch, _page())
    monkeypatch.setattr(ex, "identify_form", lambda image: None)
    email_line = LabelHit(
        text="Email: riya12@gmail.com",
        language="english",
        box=[[300, 600], [600, 600], [600, 612], [300, 612]],
        confidence=0.9,
        source="ocr",
        parsed_student_id="RIYA12GMAILCOM",
    )
    monkeypatch.setattr(ex, "find_student_id_label", lambda image: email_line)

    with pytest.raises(ex.ExtractError):
        ex._extract(readers.page, source="scan")

    assert "label" not in readers.vision_saw
    assert "label" in readers.local_saw


def _student_id_label() -> LabelHit:
    return LabelHit(
        text="Student ID:",
        language="english",
        box=[[100, 300], [250, 300], [250, 312], [100, 312]],
        confidence=0.95,
        source="ocr",
    )


def test_tall_label_crop_is_not_sent_to_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    readers = FakeReaders(monkeypatch, _page())
    readers.label_height = 0.079  # the crop that held three contact lines
    monkeypatch.setattr(ex, "identify_form", lambda image: None)
    monkeypatch.setattr(ex, "find_student_id_label", lambda image: _student_id_label())

    with pytest.raises(ex.ExtractError):
        ex._extract(readers.page, source="scan")

    assert "label" not in readers.vision_saw
    assert "label" in readers.local_saw


def test_student_id_label_crop_still_goes_to_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    readers = FakeReaders(monkeypatch, _page())
    monkeypatch.setattr(ex, "identify_form", lambda image: None)
    monkeypatch.setattr(ex, "find_student_id_label", lambda image: _student_id_label())
    readers.vision_reads["label"] = [("AADMIS310714", 0.9)]

    result = ex._extract(readers.page, source="scan")

    assert readers.vision_saw == ["label"]
    assert result.student_id == "AADMIS310714"


def test_vision_raw_text_is_not_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[str] = []
    monkeypatch.setattr(vision_ocr, "debug_log", logged.append)
    response = SimpleNamespace(
        full_text_annotation=SimpleNamespace(
            text="Riya Patil\nriya12@gmail.com\n9876543210",
            pages=[SimpleNamespace(confidence=0.9)],
        )
    )

    vision_ocr._collect_readings(response)

    joined = "\n".join(logged).lower()
    assert logged
    for secret in ("riya", "gmail", "9876543210"):
        assert secret not in joined


def test_local_readers_keep_contact_details_out_of_the_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readers = FakeReaders(monkeypatch, _page())
    logged: list[str] = []
    monkeypatch.setattr(ex, "debug_log", logged.append)
    monkeypatch.setattr(ex, "identify_form", lambda image: None)
    monkeypatch.setattr(ex, "find_student_id_label", lambda image: _email_line())
    readers.rapid_reads["label"] = [("Riya Patil 9876543210", 0.8)]
    readers.rapid_reads["template_english"] = [("riya12@gmail.com", 0.7)]

    with pytest.raises(ex.ExtractError):
        ex._extract(readers.page, source="scan")

    joined = "\n".join(logged)
    # The lines are still written - just without what the crop caught.
    assert "[DETECT] label=" in joined
    assert "[RapidOCR-crop] text=" in joined
    for secret in ("riya", "patil", "gmail", "9876543210"):
        assert secret not in joined.lower()


def test_an_id_read_beside_the_email_line_does_not_carry_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readers = FakeReaders(monkeypatch, _page())
    logged: list[str] = []
    monkeypatch.setattr(ex, "debug_log", logged.append)
    monkeypatch.setattr(ex, "identify_form", lambda image: None)
    monkeypatch.setattr(ex, "find_student_id_label", lambda image: _email_line())
    readers.rapid_reads["label"] = [("AADMIS310714", 0.8)]

    result = ex._extract(readers.page, source="scan")

    assert result.student_id == "AADMIS310714"
    # label_text reaches the browser and the [OCR] candidates log line.
    assert "riya" not in json.dumps(result.to_dict()).lower()
    assert "riya" not in "\n".join(logged).lower()
    assert "AADMIS310714" in "\n".join(logged)  # an ID is still logged as read


def test_the_student_id_label_line_is_still_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    readers = FakeReaders(monkeypatch, _page())
    logged: list[str] = []
    monkeypatch.setattr(ex, "debug_log", logged.append)
    monkeypatch.setattr(ex, "identify_form", lambda image: None)
    monkeypatch.setattr(ex, "find_student_id_label", lambda image: _student_id_label())
    readers.rapid_reads["label"] = [("AADMIS310714", 0.8)]

    result = ex._extract(readers.page, source="scan")

    assert result.label_text.endswith("Student ID:")
    assert "[DETECT] label='Student ID:'" in "\n".join(logged)


# --------------------------------- bug 15: log the reading, not the match
def test_confirm_logs_what_was_read_alongside_the_match(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(ex, "log_scan_attempt", lambda **row: logged.append(row))
    monkeypatch.setattr(
        ex,
        "find_student_by_id",
        lambda *a, **k: {
            "student_id": "TARGUR290815",
            "student_name": "Tara",
            "school_name": "ABMPS",
            "matched_via": "constrained_repair",
        },
    )

    result = ex.confirm_student_id("TAROUR29081S", source="manual-confirm", school="ABMPS")

    assert result.student_id == "TARGUR290815"
    assert result.ocr_student_id == "TAROUR29081S"
    [row] = logged
    assert row["confirmed_student_id"] == "TAROUR29081S"
    assert row["matched_student_id"] == "TARGUR290815"


def test_confirm_logs_the_reading_through_the_real_lookup(
    fake_db: FakeDB,
    fresh_id_cache: None,
) -> None:
    fake_db.handler = _roster_handler({"TARGUR290815": "ABMPS", "PRIYAD041021": "ABMPS"})

    result = ex.confirm_student_id("TAROUR29081S", source="inbox:3", school="ABMPS")

    assert result.found_in_db and result.student is not None
    assert result.student["matched_via"] == "constrained_repair"
    [(_sql, params)] = fake_db.executed(r"^INSERT INTO ocr_scan_log")
    assert params[1] == "TAROUR29081S"  # confirmed_student_id: the reading
    assert params[3] == "TARGUR290815"  # matched_student_id: the match


# The two tables the confirm -> History round trip touches, as SQLite. The
# "read as" bug lives in which log row's column the Scanned list picks up,
# so only a database that really runs the subqueries can show it.
_SQLITE_SCHEMA = """
CREATE TABLE active_student_data (
    student_id TEXT, student_name TEXT, school_name TEXT, grade_name TEXT,
    division_name TEXT, academic_year TEXT, gender TEXT, status TEXT,
    scanned_at TEXT
);
CREATE TABLE ocr_scan_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
    confirmed_student_id TEXT NOT NULL, found_in_db INTEGER NOT NULL,
    matched_student_id TEXT, student_name TEXT, school_name TEXT, source TEXT,
    message TEXT, DrivelinkImage TEXT, matched_via TEXT, scan_school TEXT,
    scan_location TEXT, reviewed_at TEXT, reviewed_by TEXT
);
"""
_DATETIME_COLUMNS = ("created_at", "scanned_at", "reviewed_at")


class _SqliteCursor:
    """Just enough of a PyMySQL DictCursor over sqlite3 for these queries."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._cur = conn.cursor()
        self.rowcount = 0

    def execute(self, sql: str, params: Any = None) -> int:
        values = tuple(
            v.isoformat(sep=" ") if isinstance(v, datetime) else v for v in (params or ())
        )
        self._cur.execute(sql.replace("%s", "?"), values)
        self.rowcount = self._cur.rowcount
        return self.rowcount

    @staticmethod
    def _as_mysql(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        for key in _DATETIME_COLUMNS:
            if isinstance(out.get(key), str):
                out[key] = datetime.fromisoformat(out[key])
        return out

    def fetchone(self) -> dict[str, Any] | None:
        row = self._cur.fetchone()
        return self._as_mysql(row) if row is not None else None

    def fetchall(self) -> list[dict[str, Any]]:
        return [self._as_mysql(row) for row in self._cur.fetchall()]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._cur.close()


class _SqliteConnection:
    """A PyMySQL-shaped handle that outlives the app's close() calls."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def cursor(self) -> _SqliteCursor:
        return _SqliteCursor(self._conn)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        return None


@pytest.fixture
def sqlite_db(monkeypatch: pytest.MonkeyPatch, fresh_id_cache: None) -> Iterator[sqlite3.Connection]:
    """Every module's get_connection pointed at one in-memory SQLite database."""
    from server.services import history_store

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SQLITE_SCHEMA)
    for name, module in list(sys.modules.items()):
        if name.startswith(("server", "tools")) and hasattr(module, "get_connection"):
            monkeypatch.setattr(module, "get_connection", lambda: _SqliteConnection(conn))
    monkeypatch.setattr(history_store, "_initialized", True)
    yield conn
    conn.close()


def test_scanned_list_shows_read_as_for_a_repaired_save(sqlite_db: sqlite3.Connection) -> None:
    """The whole trip: a fuzzy save, then History's Scanned list for it."""
    for student_id in ("TARGUR290815", "PRIYAD041021"):
        sqlite_db.execute(
            "INSERT INTO active_student_data (student_id, student_name, school_name)"
            " VALUES (?, ?, 'ABMPS')",
            (student_id, f"Child {student_id}"),
        )
    sqlite_db.commit()

    ex.confirm_student_id("TAROUR29081S", source="inbox:3", school="ABMPS")
    items, total = student_lookup.get_scanned_students(school="ABMPS")

    assert total == 1
    [row] = items
    assert row["student_id"] == "TARGUR290815"
    assert row["matched_via"] == "constrained_repair"
    # What the page needs for "read as TAROUR29081S": the reading, which is
    # not the saved ID.
    assert row["ocr_read"] == "TAROUR29081S"


# ---------------------------------- bug 16: a database outage is not a miss
def _db_down(_sql: str, _params: Any) -> tuple[list[dict[str, Any]], int | None]:
    raise pymysql.err.OperationalError(
        2003, "Can't connect to MySQL server on 'db-host.sentinel.invalid' (timed out)"
    )


def test_lookup_outage_raises_instead_of_not_found(fake_db: FakeDB) -> None:
    fake_db.handler = _db_down

    with pytest.raises(ex.LookupUnavailable) as caught:
        ex.confirm_student_id("AATKHA200618", source="inbox:7", school="ABMPS")

    assert isinstance(caught.value, ex.ExtractError)
    assert caught.value.status_code == 503
    assert caught.value.message == (
        "The student database could not be reached. Nothing was saved - try again."
    )
    assert not fake_db.executed(r"INSERT INTO ocr_scan_log")


def test_lookup_outage_logs_nothing_even_when_logging_is_immediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(ex, "log_scan_attempt", lambda **row: logged.append(row))

    def lost(*_a: Any, **_k: Any) -> None:
        raise pymysql.err.OperationalError(2013, "Lost connection to MySQL server during query")

    monkeypatch.setattr(ex, "find_student_by_id", lost)

    with pytest.raises(ex.LookupUnavailable):
        ex.confirm_student_id("AATKHA200618")
    assert logged == []


@pytest.mark.parametrize(
    "outage",
    [
        pymysql.err.OperationalError(2003, "Can't connect to MySQL server"),
        pymysql.err.InterfaceError(0, ""),
        TimeoutError("timed out"),
    ],
    ids=["refused", "closed", "timeout"],
)
def test_every_kind_of_outage_is_unavailable_not_missing(
    monkeypatch: pytest.MonkeyPatch, outage: BaseException
) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(ex, "log_scan_attempt", lambda **row: logged.append(row))

    def no_database() -> None:
        raise outage

    monkeypatch.setattr(student_lookup, "get_connection", no_database)

    with pytest.raises(ex.LookupUnavailable):
        ex.confirm_student_id("AATKHA200618", source="inbox:7", school="ABMPS")
    assert logged == []


def test_an_unconfigured_database_is_unavailable_not_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(ex, "log_scan_attempt", lambda **row: logged.append(row))
    monkeypatch.setattr(student_lookup, "DB_HOST", "")  # the real get_connection refuses

    with pytest.raises(ex.LookupUnavailable):
        ex.confirm_student_id("AATKHA200618", source="inbox:7", school="ABMPS")
    assert logged == []


@pytest.mark.parametrize(
    "bug",
    [NotImplementedError("unfinished code path"), RecursionError("maximum recursion depth")],
    ids=["not-implemented", "recursion"],
)
def test_a_bug_in_the_lookup_is_not_reported_as_an_outage(
    monkeypatch: pytest.MonkeyPatch, bug: BaseException
) -> None:
    # Both are RuntimeErrors. Called an outage, the inbox would retry the
    # same file on every pass for ever and never park it as failed.
    def broken() -> None:
        raise bug

    monkeypatch.setattr(student_lookup, "get_connection", broken)

    with pytest.raises(type(bug)):
        ex.confirm_student_id("AATKHA200618", source="inbox:7", school="ABMPS")


def test_outage_while_resolving_the_city_is_unavailable_too(fake_db: FakeDB) -> None:
    # A city scope is looked up in the database before the student is, so
    # an outage can strike there first. It must not widen the search or
    # come back as "not found".
    fake_db.handler = _db_down

    with pytest.raises(ex.LookupUnavailable):
        ex.confirm_student_id("AATKHA200618", source="inbox:7", location="Pune")
    assert not fake_db.executed(r"INSERT INTO ocr_scan_log")


def test_confirm_api_answers_503_on_an_outage(client: FlaskClient, fake_db: FakeDB) -> None:
    fake_db.handler = _db_down

    response = client.post("/api/confirm", json={"student_id": "AATKHA200618", "school": "ABMPS"})

    assert response.status_code == 503
    body = response.get_json()
    assert body["error"] == (
        "The student database could not be reached. Nothing was saved - try again."
    )
    assert "sentinel" not in response.get_data(as_text=True)
    assert not fake_db.executed(r"INSERT INTO ocr_scan_log")


# --------------------------------- bug 28: no server internals in the JSON
def test_result_json_has_no_database_internals() -> None:
    data = ex.ExtractResult(
        student_id="AADMIS310714", confidence=1.0, form_language="manual", source="x"
    ).to_dict()
    assert "db_debug" not in data
    assert "db_error" not in data


def test_confirm_api_does_not_reveal_the_database(
    client: FlaskClient,
    fake_db: FakeDB,
    monkeypatch: pytest.MonkeyPatch,
    fresh_id_cache: None,
) -> None:
    from server import config

    monkeypatch.setattr(config, "DB_HOST", "db-host.sentinel.invalid")
    monkeypatch.setattr(config, "DB_NAME", "sentinel_schema")
    fake_db.handler = _roster_handler({"AADMIS310714": "ABMPS"})

    response = client.post("/api/confirm", json={"student_id": "AADMIS310714", "school": "ABMPS"})

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert response.get_json()["found_in_db"] is True
    for internal in ("db_debug", "sentinel", "active_student_data", "SELECT"):
        assert internal not in body
