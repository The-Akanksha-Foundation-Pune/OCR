"""The handwritten batch code in a page's corner: parsing it and reading it.

Staff write "B<batch>-<number>" (B1-1, B1-12, B2-1) in the top-right corner of
every page; the same code on two pages makes them one document. These tests
cover the parser's tolerance for what handwriting and OCR do to a code - and
its refusal of anything else - and which recogniser reads the corner. Google
Vision and RapidOCR are stand-ins: nothing leaves the test.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from server.services import batch_code, vision_ocr
from server.services.batch_code import BatchCode, CodeReading, parse_batch_code


@pytest.fixture(autouse=True)
def _logs_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import server.services.debug_log as debug_module

    monkeypatch.setattr(debug_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(debug_module, "LOG_FILE", tmp_path / "logs" / "ocr_debug.log")


# ------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    ("text", "batch", "number"),
    [
        ("B1-1", 1, 1),
        ("B1-12", 1, 12),
        ("B2-1", 2, 1),
        ("B3-5", 3, 5),
        ("b1-1", 1, 1),
        ("  B1-1  ", 1, 1),
        ("B999-999", 999, 999),
        ("B01-07", 1, 7),
        # B read as 8 or 3 at the start
        ("81-1", 1, 1),
        ("812-3", 12, 3),
        ("31-5", 1, 5),
        ("888-2", 88, 2),
        # 1 read as I, l or |
        ("BI-I", 1, 1),
        ("Bl-12", 1, 12),
        ("B|-1|", 1, 11),
        ("BI-l2", 1, 12),
        # 0 read as O
        ("B1O-1", 10, 1),
        ("B1-2O", 1, 20),
        ("Bo-1", None, None),  # a lone O is a zero, and batch 0 does not exist
        # every dash OCR gives back
        ("B1–1", 1, 1),  # en dash
        ("B1—1", 1, 1),  # em dash
        ("B1−1", 1, 1),  # minus sign
        ("B1~1", 1, 1),
        ("B1_1", 1, 1),
        ("B1.1", 1, 1),
        ("B1 1", 1, 1),  # a space where the dash was
        ("B1--1", 1, 1),  # a doubled dash is still one
        # spaces anywhere
        ("B 1 - 1", 1, 1),
        ("B1 -12", 1, 12),
        ("B 1 2-3", 12, 3),
        ("B 1 1", 1, 1),
        ("8 1 - 1 2", 1, 12),
        # full-width digits from a phone keyboard
        ("Ｂ１－１", 1, 1),
    ],
)
def test_codes_as_written_or_misread_parse(text: str, batch: int | None, number: int | None) -> None:
    code = parse_batch_code(text)

    if batch is None:
        assert code is None
    else:
        assert code == BatchCode(batch=batch, number=number or 0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "   ",
        "B",
        "B1",
        "B11",  # no separator: B1-1 or B11-? cannot be told apart
        "1-1",  # no B
        "A1-1",
        "X1-1",
        "B0-1",
        "B1-0",
        "B1000-1",
        "B1-1000",
        "B1-2-3",
        "B-1-1",
        "B1-1.",
        "B1-",
        "B-1",
        "B1 2 3",  # two gaps and no dash: where does the batch end?
        "B1-1A",
        "BS-1",  # S for 5 would make almost any word parse
        "B1-Z",
        "MOHSHA011021",
        "9876543210",
        "3",  # a printed serial, circled or not
        "③",  # circled 3, as printed on the forms
        "Student ID",
        "B١-١",  # Arabic-Indic digits are not ASCII
    ],
)
def test_anything_else_is_refused(text: str | None) -> None:
    assert parse_batch_code(text) is None


def test_a_code_knows_its_text_and_folder() -> None:
    code = BatchCode(batch=2, number=14)

    assert code.text == "B2-14"
    assert str(code) == "B2-14"
    assert code.folder == "B2"
    assert parse_batch_code("82~14") == code


# ------------------------------------------------------------- reading a corner
def _page_with_ink(*regions: tuple[float, float, float, float]) -> np.ndarray:
    """A white 1000 x 1400 greyscale page with dark marks in `regions` (fractions)."""
    page = np.full((1400, 1000), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in regions:
        page[int(y0 * 1400) : int(y1 * 1400), int(x0 * 1000) : int(x1 * 1000)] = 0
    return page


TOP_RIGHT_MARK = (0.80, 0.02, 0.90, 0.06)
TOP_LEFT_MARK = (0.05, 0.02, 0.12, 0.06)


def _fake_recogniser(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[list[tuple[str, float]]],
    vision_allowed: list[bool] | None = None,
) -> list[np.ndarray]:
    seen: list[np.ndarray] = []

    def recognise(crop: np.ndarray, *, allow_vision: bool = True) -> list[tuple[str, float]]:
        seen.append(crop)
        if vision_allowed is not None:
            vision_allowed.append(allow_vision)
        return answers.pop(0) if answers else []

    monkeypatch.setattr(batch_code, "recognise_corner_text", recognise)
    return seen


def test_the_top_right_corner_is_read_first(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_recogniser(monkeypatch, [[("noise", 0.4), ("B1-12", 0.9)]])

    reading = batch_code.read_batch_code(_page_with_ink(TOP_RIGHT_MARK, TOP_LEFT_MARK))

    assert reading == CodeReading(code=BatchCode(1, 12), confidence=0.9, raw="B1-12", region="top-right")
    assert reading.code is not None and reading.code.text == "B1-12"
    # The crop is the corner region [0.72, 0.0, 1.0, 0.09] of the page.
    assert len(seen) == 1 and seen[0].shape == (126, 280)


def test_the_top_left_corner_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_recogniser(monkeypatch, [[("3", 0.8)], [("B2-1", 0.7)]])

    reading = batch_code.read_batch_code(_page_with_ink(TOP_RIGHT_MARK, TOP_LEFT_MARK))

    assert reading.code == BatchCode(2, 1)
    assert reading.region == "top-left"
    assert [crop.shape for crop in seen] == [(126, 280), (126, 300)]


@pytest.mark.parametrize(
    ("readings", "expected"),
    [
        # The printed serial beside the code, as a plain digit, circled, or
        # in brackets: never part of the number.
        ([("B1-12 3", 0.9)], "B1-12"),
        ([("B1-12 ③", 0.9)], "B1-12"),
        ([("(3) B1-4", 0.9)], "B1-4"),
        ([("③", 0.9), ("B2-1", 0.8)], "B2-1"),
        # A code read with gaps, on one line.
        ([("B 1 - 1 2", 0.9)], "B1-12"),
        # A code split across boxes, with the serial after it.
        ([("B1", 0.9), ("-12", 0.8), ("3", 0.7)], "B1-12"),
        ([("81", 0.9), ("~", 0.5), ("5", 0.8)], "B1-5"),
        ([("hello", 0.9), ("world", 0.9)], None),
    ],
)
def test_the_code_is_found_among_what_the_corner_holds(
    monkeypatch: pytest.MonkeyPatch, readings: list[tuple[str, float]], expected: str | None
) -> None:
    _fake_recogniser(monkeypatch, [readings, []])

    reading = batch_code.read_batch_code(_page_with_ink(TOP_RIGHT_MARK))

    assert (reading.code.text if reading.code else None) == expected


def test_an_empty_corner_is_not_sent_to_any_recogniser(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_recogniser(monkeypatch, [[("B1-1", 0.9)]])

    reading = batch_code.read_batch_code(_page_with_ink((0.4, 0.5, 0.6, 0.6)))

    assert reading.code is None
    assert seen == []


def test_nothing_that_parses_means_no_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_recogniser(monkeypatch, [[("hello", 0.9)], [("1", 0.9)]])

    reading = batch_code.read_batch_code(_page_with_ink(TOP_RIGHT_MARK, TOP_LEFT_MARK))

    assert reading.code is None
    assert reading.region is None
    assert reading.raw == "hello"


class _FakeRapid:
    def __init__(self, boxes: list[Any]) -> None:
        self.boxes = boxes
        self.calls = 0

    def __call__(self, _crop: np.ndarray) -> tuple[list[Any], float]:
        self.calls += 1
        return self.boxes, 0.1


def test_vision_reads_the_corner_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    rapid = _FakeRapid([])
    sent: list[np.ndarray] = []
    monkeypatch.setattr(vision_ocr, "is_configured", lambda: True)
    monkeypatch.setattr(vision_ocr, "recognize_text", lambda crop: sent.append(crop) or [("B3-5", 0.95)])
    from server.services import field_detector

    monkeypatch.setattr(field_detector, "get_rapid_ocr", lambda: rapid)

    readings = batch_code.recognise_corner_text(np.full((120, 280), 255, dtype=np.uint8))

    assert readings == [("B3-5", 0.95)]
    assert rapid.calls == 0
    # Sent as three-channel, and doubled in size for small handwriting.
    assert sent[0].shape == (240, 560, 3)


def test_rapidocr_reads_the_corner_without_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two boxes out of order: "B1" to the left of "-12".
    rapid = _FakeRapid([
        [[[60, 0], [90, 0], [90, 20], [60, 20]], "-12", 0.8],
        [[[0, 0], [50, 0], [50, 20], [0, 20]], "B1", 0.9],
    ])
    monkeypatch.setattr(vision_ocr, "is_configured", lambda: False)
    from server.services import field_detector

    monkeypatch.setattr(field_detector, "get_rapid_ocr", lambda: rapid)

    readings = batch_code.recognise_corner_text(np.full((120, 280, 3), 255, dtype=np.uint8))

    assert readings == [("B1", 0.9), ("-12", 0.8)]
    first = next(t for t, _ in batch_code.candidate_texts(readings) if parse_batch_code(t))
    assert parse_batch_code(first) == BatchCode(1, 12)


def test_rapidocr_stands_in_when_vision_returns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    rapid = _FakeRapid([[[[0, 0], [50, 0], [50, 20], [0, 20]], "B2-3", 0.9]])
    monkeypatch.setattr(vision_ocr, "is_configured", lambda: True)
    monkeypatch.setattr(vision_ocr, "recognize_text", lambda _crop: [])
    from server.services import field_detector

    monkeypatch.setattr(field_detector, "get_rapid_ocr", lambda: rapid)

    assert batch_code.recognise_corner_text(np.full((120, 280), 255, dtype=np.uint8)) == [("B2-3", 0.9)]


def test_a_corner_kept_off_vision_is_read_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    rapid = _FakeRapid([[[[0, 0], [50, 0], [50, 20], [0, 20]], "B2-3", 0.9]])
    monkeypatch.setattr(vision_ocr, "is_configured", lambda: True)
    monkeypatch.setattr(vision_ocr, "recognize_text", lambda _crop: pytest.fail("sent to Google"))
    from server.services import field_detector

    monkeypatch.setattr(field_detector, "get_rapid_ocr", lambda: rapid)

    crop = np.full((120, 280), 255, dtype=np.uint8)
    assert batch_code.recognise_corner_text(crop, allow_vision=False) == [("B2-3", 0.9)]
    assert rapid.calls == 1


def test_reading_a_page_passes_the_vision_switch_to_each_corner(monkeypatch: pytest.MonkeyPatch) -> None:
    allowed: list[bool] = []
    _fake_recogniser(monkeypatch, [[("noise", 0.4)], [("noise", 0.4)]], allowed)
    page = _page_with_ink(TOP_RIGHT_MARK, TOP_LEFT_MARK)

    batch_code.read_batch_code(page, allow_vision=False)
    batch_code.read_batch_code(page)

    assert allowed == [False, False, True, True]


def test_a_recogniser_crash_reads_as_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Any:
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(vision_ocr, "is_configured", lambda: False)
    from server.services import field_detector

    monkeypatch.setattr(field_detector, "get_rapid_ocr", broken)

    assert batch_code.recognise_corner_text(np.full((120, 280), 255, dtype=np.uint8)) == []


def test_vision_corner_text_keeps_the_dash(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Student ID clean-up strips dashes; the corner reader must not."""
    asked: list[Any] = []

    class FakeClient:
        def document_text_detection(self, **kwargs: Any) -> Any:
            asked.append(kwargs)
            return SimpleNamespace(
                error=SimpleNamespace(message=""),
                full_text_annotation=SimpleNamespace(
                    text="B1 - 12\n③\n", pages=[SimpleNamespace(confidence=0.9)]
                ),
            )

    monkeypatch.setattr(vision_ocr, "get_client", lambda: FakeClient())

    readings = vision_ocr.recognize_text(np.full((60, 140, 3), 255, dtype=np.uint8))

    assert readings == [("B1 - 12", 0.9), ("③", 0.9)]
    assert parse_batch_code(readings[0][0]) == BatchCode(1, 12)
    assert len(asked) == 1


def test_vision_corner_text_fails_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenClient:
        def document_text_detection(self, **_kwargs: Any) -> Any:
            raise TimeoutError("deadline exceeded")

    monkeypatch.setattr(vision_ocr, "get_client", lambda: BrokenClient())

    assert vision_ocr.recognize_text(np.full((60, 140, 3), 255, dtype=np.uint8)) == []
    monkeypatch.setattr(vision_ocr, "get_client", lambda: None)
    assert vision_ocr.recognize_text(np.full((60, 140, 3), 255, dtype=np.uint8)) == []
