"""Handwritten English Student ID recognition with TrOCR."""

from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image

from server.config import (
    MAX_STUDENT_ID_LENGTH,
    MIN_INK_RATIO,
    MIN_OCR_CONFIDENCE,
    MIN_STUDENT_ID_LENGTH,
    STUDENT_ID_FORMAT,
    STUDENT_ID_PATTERN,
    TROCR_MODEL_NAME,
)
from server.services.debug_log import debug_log
from server.services.preprocess import to_pil_rgb

_processor = None
_model = None
_easyocr_reader = None

# Form words / fragments TrOCR often reads instead of the ID
_REJECT_FRAGMENTS = (
    "MATION",
    "INFORMATION",
    "STUDENT",
    "GRADE",
    "NAME",
    "SCHOOL",
    "PARENT",
    "GUARDIAN",
    "CONSENT",
    "AKANKSHA",
    "CHILDREN",
    "DIVISION",
)


def get_trocr():
    """Lazy-load TrOCR processor and model (Python 3.14 / transformers 5 safe)."""
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    from transformers import (
        AutoImageProcessor,
        RobertaTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    image_processor = AutoImageProcessor.from_pretrained(TROCR_MODEL_NAME)
    tokenizer = RobertaTokenizer.from_pretrained(TROCR_MODEL_NAME)
    _processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    _model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
    _model.eval()
    return _processor, _model


def get_easyocr_reader():
    """Lazy-load EasyOCR — a second, independently-trained recognizer used
    to give the pipeline another guess at messy handwritten IDs."""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader

    import easyocr

    _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def recognize_with_easyocr(crop_bgr: np.ndarray) -> list[tuple[str, float]]:
    """Run EasyOCR on a crop; returns list of (cleaned_text, confidence)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    if not has_handwriting_ink(crop_bgr):
        return []

    try:
        reader = get_easyocr_reader()
        enhanced = enhance_id_crop(crop_bgr)
        results = reader.readtext(
            enhanced,
            detail=1,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[EasyOCR] failed: {exc}")
        return []

    found: list[tuple[str, float]] = []
    for _box, text, conf in results:
        cleaned = clean_student_id(str(text))
        debug_log(f"[EasyOCR] raw={text!r} cleaned={cleaned!r} conf={conf}")
        if cleaned:
            found.append((cleaned, float(conf)))
    return found


def has_handwriting_ink(crop_bgr: np.ndarray) -> bool:
    """Return True when the crop has enough dark pixels to be filled handwriting."""
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ink_ratio = float(np.mean(gray < 150))
    return ink_ratio >= MIN_INK_RATIO


def looks_like_student_id(text: str) -> bool:
    """
    True only for AFDW-style IDs: letters then digits (e.g. AADMIS310714).

    Rejects OCR junk like AADMES3OZY / MATION.
    """
    if not text:
        return False
    cleaned = clean_student_id(text)
    if len(cleaned) < MIN_STUDENT_ID_LENGTH or len(cleaned) > MAX_STUDENT_ID_LENGTH:
        return False
    if any(fragment in cleaned for fragment in _REJECT_FRAGMENTS):
        return False
    return bool(re.fullmatch(STUDENT_ID_FORMAT, cleaned))


def enhance_id_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Upscale + contrast boost for small handwritten ID crops."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    scale = max(2.5, 120 / max(h, 1))
    enlarged = cv2.resize(
        crop_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    boosted = clahe.apply(gray)
    # Keep as 3-channel for OCR engines that expect color
    return cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR)


def recognize_handwritten_id(crop_bgr: np.ndarray) -> tuple[str, float]:
    """
    Run TrOCR on a cropped Student ID region.

    Returns (cleaned_student_id, confidence_proxy).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    if not has_handwriting_ink(crop_bgr):
        return "", 0.0

    try:
        processor, model = get_trocr()
        enhanced = enhance_id_crop(crop_bgr)
        image: Image.Image = to_pil_rgb(enhanced)

        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated = model.generate(
            pixel_values,
            max_new_tokens=32,
            return_dict_in_generate=True,
            output_scores=True,
        )
        raw_text = processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )[0]
    except OSError as exc:
        # WinError 4551 / Application Control can block torch DLLs on Windows
        debug_log(f"[TrOCR] skipped (torch blocked or unavailable): {exc}")
        return "", 0.0
    except Exception as exc:  # noqa: BLE001 — keep RapidOCR/EasyOCR path alive
        debug_log(f"[TrOCR] failed: {exc}")
        return "", 0.0

    cleaned = clean_student_id(raw_text)
    confidence = _estimate_confidence(generated)
    debug_log(f"[TrOCR] raw={raw_text!r} cleaned={cleaned!r} conf={confidence}")

    if confidence < MIN_OCR_CONFIDENCE:
        return "", confidence
    if not looks_like_student_id(cleaned):
        debug_log(f"[TrOCR] rejected (not a valid Student ID format): {cleaned!r}")
        return "", confidence

    return cleaned, confidence


def clean_student_id(raw_text: str) -> str:
    """Normalize OCR output to an English alphanumeric Student ID."""
    if not raw_text:
        return ""

    text = raw_text.strip().upper()
    text = re.sub(STUDENT_ID_PATTERN, "", text)
    text = text.replace(" ", "")
    return text


def _estimate_confidence(generated) -> float:
    """Approximate confidence from sequence scores when available."""
    try:
        import torch

        if not getattr(generated, "scores", None):
            return 0.75

        probs = []
        for score_tensor in generated.scores:
            token_probs = torch.softmax(score_tensor[0], dim=-1)
            probs.append(float(token_probs.max().item()))
        if not probs:
            return 0.75
        return round(sum(probs) / len(probs), 4)
    except Exception:
        return 0.75
