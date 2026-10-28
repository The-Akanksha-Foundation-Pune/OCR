"""Image / PDF loading and scan preprocessing for CamScanner and printer scans."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

# Windows WIA often saves a BMP even when the file is named .jpg, and some
# drivers write grayscale or 4-channel bitmaps. The rest of the pipeline
# assumes a 3-channel BGR image.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image_from_path(file_path: str | Path) -> np.ndarray:
    """Load first page of a PDF or an image file as a BGR OpenCV array."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _pdf_first_page_to_bgr(path)

    if suffix in _IMAGE_SUFFIXES:
        return _decode_image_bytes(path.read_bytes(), path.name)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_image_from_bytes(data: bytes, filename: str) -> np.ndarray:
    """Load image/PDF bytes into a BGR OpenCV array."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _pdf_bytes_to_bgr(data)

    if suffix in _IMAGE_SUFFIXES:
        return _decode_image_bytes(data, filename)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_pages_from_bytes(data: bytes, filename: str) -> list[np.ndarray]:
    """Load EVERY page as a BGR OpenCV array — a printer/scanner machine
    often batch-scans a whole stack of forms into one multi-page PDF.
    Image files (JPG/PNG) are always a single "page"."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _pdf_bytes_to_bgr_pages(data)

    return [load_image_from_bytes(data, filename)]


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Turn grayscale, BGRA, or BGR into a 3-channel BGR image."""
    if image is None or image.size == 0:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[2]
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _decode_image_bytes(data: bytes, filename: str) -> np.ndarray:
    """Decode a scan. The suffix is only a hint.

    A Windows scanner names the file .jpg and then writes a BMP, which
    OpenCV still recognises from the file header. If that fails, Pillow
    reads the formats OpenCV refuses (some 16-bit and palette BMPs).
    """
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        try:
            with Image.open(io.BytesIO(data)) as pil_image:
                rgb = np.asarray(pil_image.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"Could not decode image: {filename}") from exc
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image = ensure_bgr(image)
    if image is None or image.size == 0:
        raise ValueError(f"Could not decode image: {filename}")
    return image


def count_pages_from_bytes(data: bytes, filename: str) -> int:
    """How many pages a scan holds, without rendering any of them."""
    if Path(filename).suffix.lower() != ".pdf":
        return 1
    with fitz.open(stream=data, filetype="pdf") as document:
        return document.page_count


def load_page_from_bytes(data: bytes, filename: str, page_number: int) -> np.ndarray:
    """
    Render ONE page (1-based) and nothing else.

    load_pages_from_bytes renders the whole document, which is fine for a
    handful of forms but quadratic when each page is fetched in its own
    request - a 500-page batch would render 250,000 pages and hold gigabytes
    of them in memory at once.
    """
    if Path(filename).suffix.lower() != ".pdf":
        return load_image_from_bytes(data, filename)

    with fitz.open(stream=data, filetype="pdf") as document:
        if document.page_count < 1:
            raise ValueError("PDF has no pages")
        index = page_number - 1
        if index < 0 or index >= document.page_count:
            raise ValueError(
                f"Page {page_number} is outside this scan "
                f"(it has {document.page_count})"
            )
        return _render_page(document[index], 200)


def limit_working_size(image_bgr: np.ndarray, max_long_side: int) -> np.ndarray:
    """Shrink a scan so its long side is at most `max_long_side` pixels.

    INTER_AREA is the right filter when reducing: it averages pixels instead
    of picking one, so a 600 DPI scanner JPEG becomes the same kind of page
    a 200 DPI PDF render already is.
    """
    if image_bgr is None or image_bgr.size == 0 or max_long_side <= 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    long_side = max(height, width)
    if long_side <= max_long_side:
        return image_bgr
    scale = max_long_side / float(long_side)
    return cv2.resize(
        image_bgr,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """Deskew and enhance a scan for more reliable label detection."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")

    from server.config import MAX_OCR_LONG_SIDE

    sized = limit_working_size(ensure_bgr(image_bgr), MAX_OCR_LONG_SIDE)
    deskewed = _deskew(sized)
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
