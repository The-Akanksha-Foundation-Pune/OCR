"""Image / PDF / TIFF loading and preprocessing for scanned forms."""

from __future__ import annotations

import contextlib
import io
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageOps

# macOS Image Capture and many scanners' own software save TIFF by default,
# often 1-bit (CCITT G4) or 16-bit greyscale. Pillow reads those reliably
# where OpenCV's decoder may not, and a TIFF holds a whole stack as pages.
TIFF_SUFFIXES = frozenset({".tif", ".tiff"})

# The first four bytes of every TIFF: little-endian ("II") or big-endian
# ("MM"), classic TIFF (42) or BigTIFF (43). Checked before Pillow sees the
# file, because Pillow fails a file cut off in its header and a file that was
# never a TIFF in the same way, and the person needs to hear which it is.
_TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")

# Pillow reports a damaged TIFF through several exception types: a cut-off
# or corrupted file can raise TypeError, SyntaxError ("unknown data
# organization") or KeyError while its pages are counted or decoded. Each is
# turned into the same ValueError a bad JPG or PNG gives, so the person is
# told the file cannot be read rather than to try again.
_TIFF_DECODE_ERRORS = (
    OSError,
    EOFError,
    ValueError,
    TypeError,
    SyntaxError,
    KeyError,
    IndexError,
    struct.error,
    Image.DecompressionBombError,
)


# PyMuPDF reports a file that is not a PDF, or a damaged one, as
# FileDataError (a RuntimeError) and some damage found while rendering as
# MuPDF's own FzErrorBase, which is neither a RuntimeError nor a ValueError.
# Each becomes the ValueError a damaged image gives, so the person is told
# the file cannot be read instead of getting a server error.
_PDF_DECODE_ERRORS: tuple[type[BaseException], ...] = tuple(
    kind
    for kind in (RuntimeError, getattr(getattr(fitz, "mupdf", None), "FzErrorBase", None))
    if isinstance(kind, type)
)


@contextlib.contextmanager
def _decoding_pdf(filename: str) -> Iterator[None]:
    """Turn PyMuPDF's failures on a damaged PDF into "Could not decode" ValueErrors."""
    try:
        yield
    except _PDF_DECODE_ERRORS as exc:
        raise ValueError(f"Could not decode PDF: {filename}") from exc


class NotATiffError(ValueError):
    """A file named .tif or .tiff whose contents are not a TIFF at all.

    Kept apart from a damaged TIFF so the person is told the file is some
    other kind - a JPEG or PDF given the wrong name, say - rather than that
    the scan is broken.
    """

# Windows WIA often saves a BMP even when the file is named .jpg, and some
# drivers write grayscale or 4-channel bitmaps. The rest of the pipeline
# assumes a 3-channel BGR image.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image_from_path(file_path: str | Path) -> np.ndarray:
    """Load first page of a PDF, a TIFF or an image file as a BGR OpenCV array."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _pdf_first_page_to_bgr(path)

    if suffix in TIFF_SUFFIXES:
        return _tiff_bytes_to_bgr_pages(path.read_bytes(), path.name, page_number=1)[0]

    if suffix in _IMAGE_SUFFIXES:
        return _decode_image_bytes(path.read_bytes(), path.name)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_image_from_bytes(data: bytes, filename: str) -> np.ndarray:
    """Load image/PDF/TIFF bytes (the first page) into a BGR OpenCV array."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _pdf_bytes_to_bgr(data)

    if suffix in TIFF_SUFFIXES:
        return _tiff_bytes_to_bgr_pages(data, filename, page_number=1)[0]

    if suffix in _IMAGE_SUFFIXES:
        return _decode_image_bytes(data, filename)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_pages_from_bytes(data: bytes, filename: str) -> list[np.ndarray]:
    """Load EVERY page as a BGR OpenCV array — a printer/scanner machine
    often batch-scans a whole stack of forms into one multi-page PDF or
    TIFF. Image files (JPG/PNG) are always a single "page"."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _pdf_bytes_to_bgr_pages(data)

    if suffix in TIFF_SUFFIXES:
        return _tiff_bytes_to_bgr_pages(data, filename)

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
    """How many pages a scan holds, without rendering any of them.

    Raises:
        ValueError: The file cannot be decoded (a damaged or non-PDF .pdf,
            a damaged TIFF).
    """
    suffix = Path(filename).suffix.lower()
    if suffix in TIFF_SUFFIXES:
        return _tiff_page_count(data, filename)
    if suffix != ".pdf":
        return 1
    with _decoding_pdf(filename), fitz.open(stream=data, filetype="pdf") as document:
        return document.page_count


def load_page_from_bytes(data: bytes, filename: str, page_number: int) -> np.ndarray:
    """
    Render ONE page (1-based) and nothing else.

    load_pages_from_bytes renders the whole document, which is fine for a
    handful of forms but quadratic when each page is fetched in its own
    request - a 500-page batch would render 250,000 pages and hold gigabytes
    of them in memory at once.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in TIFF_SUFFIXES:
        return _tiff_bytes_to_bgr_pages(data, filename, page_number=page_number)[0]
    if suffix != ".pdf":
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


@dataclass
class GreyPage:
    """One page of a scan, normalised for the queue.

    Attributes:
        image: The page as a 2-D uint8 greyscale array, upright by the file's
            own orientation tag, at most `max_dpi` and `max_long_side`.
        dpi: Its resolution, when the file said (always, for a PDF page).
    """

    image: np.ndarray
    dpi: float | None


# Readers for a file named .jpg / .jpeg / .png. Either kind under the other's
# name is read, as cv2.imdecode always has; nothing else is - a file only
# named .jpg must not reach Pillow's other decoders.
_RASTER_FORMATS = ["JPEG", "MPO", "PNG"]


def load_page_grey(
    data: bytes,
    filename: str,
    page_number: int = 1,
    *,
    max_dpi: float = 300,
    max_long_side: int = 3600,
    pdf_dpi: float = 200,
) -> GreyPage:
    """Decode ONE page (1-based) as greyscale, no larger than it needs to be.

    The scan queue stores every page this way. Decoding straight to one
    channel, and shrinking with INTER_AREA before anything else sees the
    page, keeps a 600 dpi colour scan from ever becoming a ~100 MB
    three-channel array in OpenCV.

    Args:
        data: The uploaded file.
        filename: Its name; the suffix picks the decoder.
        page_number: Which page of a PDF or TIFF.
        max_dpi: Shrink a page scanned finer than this.
        max_long_side: Shrink a page longer than this many pixels.
        pdf_dpi: Resolution to render a PDF page at.

    Returns:
        The page and its resolution.

    Raises:
        NotATiffError: A .tif / .tiff that is not a TIFF.
        ValueError: The page cannot be decoded, or is not in the file.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        with _decoding_pdf(filename):
            grey, dpi = _pdf_page_grey(data, page_number, pdf_dpi, max_long_side)
    elif suffix in TIFF_SUFFIXES:
        grey, dpi = _tiff_page_grey(data, filename, page_number)
    elif suffix in {".jpg", ".jpeg", ".png"}:
        grey, dpi = _raster_grey(data, filename, max_long_side)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return GreyPage(*_shrink(grey, dpi, max_dpi, max_long_side))


def split_original_page(data: bytes, filename: str, page_number: int) -> bytes:
    """One page of an upload in its own format, to keep beside the queued page.

    A single-page file is returned exactly as uploaded. A page of a PDF
    becomes a one-page PDF, and a page of a TIFF a one-page TIFF, so each
    queued page holds its own original and not a copy of the whole stack.

    Args:
        data: The uploaded file.
        filename: Its name.
        page_number: Which page (1-based).

    Returns:
        The page's bytes.

    Raises:
        ValueError: The page cannot be split out.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        with _decoding_pdf(filename), fitz.open(stream=data, filetype="pdf") as document:
            if document.page_count == 1:
                return data
            single = fitz.open()
            try:
                single.insert_pdf(document, from_page=page_number - 1, to_page=page_number - 1)
                return single.tobytes()
            finally:
                single.close()
    if suffix in TIFF_SUFFIXES:
        with _open_tiff(data, filename) as tiff:
            if _tiff_frame_count(tiff, filename) == 1:
                return data
            try:
                tiff.seek(page_number - 1)
                return _tiff_frame_bytes(tiff)
            except _TIFF_DECODE_ERRORS as exc:
                raise ValueError(f"Could not split page {page_number} of {filename}") from exc
    return data


def _tiff_frame_bytes(tiff: Image.Image) -> bytes:
    """The frame a TIFF is positioned on, saved as a TIFF of its own.

    Its own compression where Pillow can write it (Group 4 needs libtiff),
    else uncompressed - a 1-bit A4 page is still only ~300 KB that way.
    """
    dpi = tiff.info.get("dpi")
    options: dict[str, object] = {"dpi": dpi} if dpi else {}
    for compression in (tiff.info.get("compression"), None):
        buffer = io.BytesIO()
        try:
            extra = dict(options, compression=compression) if compression else options
            tiff.save(buffer, format="TIFF", **extra)
        except (OSError, ValueError, KeyError) as exc:
            if compression is None:
                raise ValueError("Could not save the TIFF page") from exc
            continue
        return buffer.getvalue()
    raise ValueError("Could not save the TIFF page")


def _pdf_page_grey(
    data: bytes, page_number: int, pdf_dpi: float, max_long_side: int
) -> tuple[np.ndarray, float]:
    with fitz.open(stream=data, filetype="pdf") as document:
        if document.page_count < 1:
            raise ValueError("PDF has no pages")
        index = page_number - 1
        if index < 0 or index >= document.page_count:
            raise ValueError(
                f"Page {page_number} is outside this scan (it has {document.page_count})"
            )
        page = document[index]
        longest_points = max(page.rect.width, page.rect.height, 1.0)
        # Render no larger than the cap, rather than render big and shrink.
        zoom = min(pdf_dpi / 72.0, max_long_side / longest_points)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
        grey = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.stride
        )[:, : pixmap.width]
        return np.ascontiguousarray(grey), zoom * 72.0


def _tiff_page_grey(data: bytes, filename: str, page_number: int) -> tuple[np.ndarray, float | None]:
    with _open_tiff(data, filename) as tiff:
        total = _tiff_frame_count(tiff, filename)
        if page_number < 1 or page_number > total:
            raise ValueError(f"Page {page_number} is outside this scan (it has {total})")
        try:
            tiff.seek(page_number - 1)
            dpi = _dpi_of(tiff)
            if tiff.mode.startswith("I;16"):
                # As in _tiff_page_to_bgr: scale 16-bit grey, never clip it.
                grey = (np.asarray(tiff, dtype=np.uint16) >> 8).astype(np.uint8)
            else:
                grey = np.asarray(tiff.convert("L"))
        except _TIFF_DECODE_ERRORS as exc:
            raise ValueError(f"Could not decode page {page_number} of {filename}") from exc
        return np.ascontiguousarray(grey), dpi


def _raster_grey(data: bytes, filename: str, max_long_side: int) -> tuple[np.ndarray, float | None]:
    try:
        with Image.open(io.BytesIO(data), formats=_RASTER_FORMATS) as image:
            dpi = _dpi_of(image)
            if image.format in ("JPEG", "MPO") and max(image.size) > 2 * max_long_side:
                # Let the JPEG decoder skip the detail we would throw away:
                # it decodes at 1/2, 1/4 or 1/8 scale, no smaller than asked.
                full_width = image.width
                scale = max_long_side / max(image.size)
                image.draft("L", (int(image.width * scale), int(image.height * scale)))
                if dpi:
                    dpi = dpi * image.width / max(full_width, 1)
            # OpenCV turns a JPEG upright by its EXIF tag; so must this.
            upright = ImageOps.exif_transpose(image)
            if upright.mode.startswith("I;16"):
                grey = (np.asarray(upright, dtype=np.uint16) >> 8).astype(np.uint8)
            elif upright.mode in ("RGBA", "LA", "PA") or "transparency" in upright.info:
                # Transparent margins are paper, not ink.
                rgba = upright.convert("RGBA")
                paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                grey = np.asarray(Image.alpha_composite(paper, rgba).convert("L"))
            else:
                grey = np.asarray(upright.convert("L"))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Could not decode image: {filename}") from exc
    return np.ascontiguousarray(grey), dpi


def _dpi_of(image: Image.Image) -> float | None:
    """The resolution an image file declares, or None when it does not."""
    raw = image.info.get("dpi")
    try:
        value = float(raw[0]) if isinstance(raw, tuple) else float(raw)
    except (TypeError, ValueError, IndexError):
        return None
    # 1 and 72 are what software writes when it does not know.
    return value if value > 72 else None


def _shrink(
    grey: np.ndarray, dpi: float | None, max_dpi: float, max_long_side: int
) -> tuple[np.ndarray, float | None]:
    """Scale a page down (never up) to max_dpi and max_long_side."""
    height, width = grey.shape[:2]
    scale = 1.0
    if dpi and dpi > max_dpi:
        scale = max_dpi / dpi
    longest = max(height, width, 1)
    if longest * scale > max_long_side:
        scale = max_long_side / longest
    if scale >= 1.0:
        return grey, dpi
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    return small, (dpi * scale if dpi else None)


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


def _open_tiff(data: bytes, filename: str) -> Image.Image:
    """Open TIFF bytes with Pillow's TIFF reader and no other.

    Raises:
        NotATiffError: The bytes do not start like any TIFF.
        ValueError: They do, but Pillow cannot read the file's header.
    """
    if data[:4] not in _TIFF_SIGNATURES:
        suffix = Path(filename).suffix.lower() or ".tif"
        raise NotATiffError(
            f"Not a TIFF image, although its name ends in {suffix}. "
            "Save the scan again as TIFF, JPEG or PDF."
        )
    try:
        # formats= keeps a file that is only named .tif away from Pillow's
        # other readers - the EPS one would hand it to Ghostscript.
        return Image.open(io.BytesIO(data), formats=["TIFF"])
    except _TIFF_DECODE_ERRORS as exc:
        raise ValueError(f"Could not decode image: {filename}") from exc


def _tiff_page_count(data: bytes, filename: str) -> int:
    with _open_tiff(data, filename) as tiff:
        return _tiff_frame_count(tiff, filename)


def _tiff_frame_count(tiff: Image.Image, filename: str) -> int:
    # Counting walks every page's header, so a cut-off file fails here.
    try:
        return int(getattr(tiff, "n_frames", 1))
    except _TIFF_DECODE_ERRORS as exc:
        raise ValueError(f"Could not decode image: {filename}") from exc


def _tiff_bytes_to_bgr_pages(
    data: bytes, filename: str, page_number: int | None = None
) -> list[np.ndarray]:
    """Decode every page of a TIFF, or only `page_number` (1-based), as BGR.

    Each TIFF page is one sheet, the same as a page of a PDF, so a stack fed
    through the scanner into one file is read form by form.
    """
    with _open_tiff(data, filename) as tiff:
        total = _tiff_frame_count(tiff, filename)
        if total < 1:
            raise ValueError("TIFF has no pages")
        if page_number is None:
            indexes = range(total)
        else:
            if page_number < 1 or page_number > total:
                raise ValueError(
                    f"Page {page_number} is outside this scan (it has {total})"
                )
            indexes = range(page_number - 1, page_number)

        pages: list[np.ndarray] = []
        for index in indexes:
            try:
                tiff.seek(index)
                pages.append(_tiff_page_to_bgr(tiff))
            except _TIFF_DECODE_ERRORS as exc:
                raise ValueError(
                    f"Could not decode page {index + 1} of {filename}"
                ) from exc
        return pages


def _tiff_page_to_bgr(page: Image.Image) -> np.ndarray:
    """The TIFF page the file is positioned on, as 8-bit BGR.

    Pillow already turns each page upright by its own Orientation tag as it
    loads it - as OpenCV does for a JPEG's EXIF - so that is not redone here.
    """
    if page.mode.startswith("I;16"):
        # Pillow clips 16-bit grey to 8 bits rather than scaling it, which
        # turns all but the darkest ink white - a blank page to the reader.
        grey = (np.asarray(page, dtype=np.uint16) >> 8).astype(np.uint8)
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    # 1-bit, greyscale, palette and CMYK pages all come out as plain RGB.
    rgb = np.asarray(page.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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
