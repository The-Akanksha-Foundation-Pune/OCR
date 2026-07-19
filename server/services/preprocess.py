"""Image / PDF loading and scan preprocessing for CamScanner and printer scans."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image


def load_image_from_path(file_path: str | Path) -> np.ndarray:
    """Load first page of a PDF or an image file as a BGR OpenCV array."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _pdf_first_page_to_bgr(path)

    if suffix in {".jpg", ".jpeg", ".png"}:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path.name}")
        return image

    raise ValueError(f"Unsupported file type: {suffix}")


def load_image_from_bytes(data: bytes, filename: str) -> np.ndarray:
    """Load image/PDF bytes into a BGR OpenCV array."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _pdf_bytes_to_bgr(data)

    if suffix in {".jpg", ".jpeg", ".png"}:
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {filename}")
        return image

    raise ValueError(f"Unsupported file type: {suffix}")


def load_pages_from_bytes(data: bytes, filename: str) -> list[np.ndarray]:
    """Load EVERY page as a BGR OpenCV array — a printer/scanner machine
    often batch-scans a whole stack of forms into one multi-page PDF.
    Image files (JPG/PNG) are always a single "page"."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _pdf_bytes_to_bgr_pages(data)

    return [load_image_from_bytes(data, filename)]


def preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """Deskew and enhance a scan for more reliable label detection."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")

    deskewed = _deskew(image_bgr)
    enhanced = _enhance_contrast(deskewed)
    return enhanced


def to_pil_rgb(image_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pdf_first_page_to_bgr(path: Path, dpi: int = 200) -> np.ndarray:
    with fitz.open(path) as document:
        if document.page_count < 1:
            raise ValueError(f"PDF has no pages: {path.name}")
        return _render_page(document[0], dpi)


def _pdf_bytes_to_bgr(data: bytes, dpi: int = 200) -> np.ndarray:
    with fitz.open(stream=data, filetype="pdf") as document:
        if document.page_count < 1:
            raise ValueError("PDF has no pages")
        return _render_page(document[0], dpi)


def _pdf_bytes_to_bgr_pages(data: bytes, dpi: int = 200) -> list[np.ndarray]:
    with fitz.open(stream=data, filetype="pdf") as document:
        if document.page_count < 1:
            raise ValueError("PDF has no pages")
        return [_render_page(page, dpi) for page in document]


def _render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    )
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _enhance_contrast(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(lightness)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _deskew(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min(image_bgr.shape[:2]) // 4,
        maxLineGap=20,
    )

    if lines is None:
        return image_bgr

    angles: list[float] = []
    # OpenCV 4 returns shape (N, 1, 4); OpenCV 5 may return (N, 4)
    line_rows = lines.reshape(-1, 4)
    for x1, y1, x2, y2 in line_rows:
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if -45.0 < angle < 45.0:
            angles.append(angle)

    if not angles:
        return image_bgr

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return image_bgr

    height, width = image_bgr.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
