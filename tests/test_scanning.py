"""Zero-install browser scanning: the server side.

The page no longer reaches a scan helper on the scanning computer; it uploads
whatever the scanner's own software saves - and macOS Image Capture and many
scanners save TIFF, often 1-bit. Covered here:

  - /scan-helper.zip is gone (404), and so is every route that served it.
  - .tif / .tiff uploads go through /api/extract, /api/extract-batch and
    /api/confirm like a PDF: a multi-page TIFF is one page each, 1-bit,
    greyscale, 16-bit and colour TIFFs all decode, and the page saved to
    Drive is a JPEG of the page asked for.
  - A file only named .tif / .tiff that is not a TIFF is refused as "Not a
    TIFF image" before any image reader sees it, so Pillow never hands it to
    another one (Ghostscript for an EPS, say).
  - JPG, PNG and PDF uploads behave exactly as before, and other types are
    still refused.

Every TIFF, PNG, JPEG and PDF is made here with Pillow, OpenCV or PyMuPDF.
The OCR itself is a stand-in that records the page image it was handed, and
Drive and the database lookup are stand-ins too - nothing leaves the test.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient
from PIL import Image, ImageDraw, TiffImagePlugin, features

from server.services import preprocess, scan_store
from server.services.extract_student_id import ExtractResult

# (width, height) of each page of the test stack - all different, so a test
# can tell from its size which page it was handed.
PAGE_SIZES = [(300, 200), (240, 360), (180, 120)]
INK = (20, 20, 60, 40)  # x0, y0, x1, y1 of a black mark on each test page

needs_libtiff = pytest.mark.skipif(
    not features.check("libtiff"), reason="this Pillow was built without libtiff"
)


# ------------------------------------------------------------------ makers
def _tiff(pages: list[Image.Image], **options: Any) -> bytes:
    """The pages saved as one TIFF, the first page first."""
    buffer = io.BytesIO()
    pages[0].save(buffer, format="TIFF", save_all=True, append_images=pages[1:], **options)
    return buffer.getvalue()


def _page(size: tuple[int, int], mode: str = "L", paper: Any = 255, ink: Any = 0) -> Image.Image:
    """A blank page with one black mark at INK."""
    page = Image.new(mode, size, paper)
    ImageDraw.Draw(page).rectangle(INK, fill=ink)
    return page


def _stack() -> bytes:
    """A three-page greyscale TIFF, each page its own size."""
    return _tiff([_page(size) for size in PAGE_SIZES])


def _one_bit(compression: str | None = None) -> bytes:
    """A 1-bit black-and-white TIFF, as a sheet-fed scanner saves one."""
    options = {"compression": compression} if compression else {}
    return _tiff([_page((200, 100), mode="1", paper=1, ink=0)], **options)


def _shape(size: tuple[int, int]) -> tuple[int, int, int]:
    """The BGR array shape of a page of `size` (width, height)."""
    return (size[1], size[0], 3)


def _png(size: tuple[int, int] = (160, 90)) -> tuple[bytes, np.ndarray]:
    image = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    image[10:30, 10:50] = (0, 0, 200)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes(), image


def _jpeg(size: tuple[int, int] = (160, 90)) -> bytes:
    _data, image = _png(size)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _pdf(pages: int = 2) -> bytes:
    document = fitz.open()
    for _ in range(pages):
        document.new_page(width=200, height=300)
    data = document.tobytes()
    document.close()
    return data


def _is_ink(pixel: np.ndarray) -> bool:
    return bool(int(pixel.max()) < 60)


def _is_paper(pixel: np.ndarray) -> bool:
    return bool(int(pixel.min()) > 200)


# ---------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _logs_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep debug_log's lines out of the repository's logs/ folder."""
    import server.services.debug_log as debug_module

    monkeypatch.setattr(debug_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(debug_module, "LOG_FILE", tmp_path / "logs" / "ocr_debug.log")


@pytest.fixture
def stash_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stash uploads under the test's own folder, not data/uploads."""
    folder = tmp_path / "uploads"
    monkeypatch.setattr(scan_store, "UPLOAD_FOLDER", folder)
    return folder


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Stand in for the OCR: record each page image it is handed."""
    from server.services import extract_student_id as extract_module

    seen: list[np.ndarray] = []

    def fake_extract(image_bgr: np.ndarray, source: str) -> ExtractResult:
        seen.append(image_bgr)
        return ExtractResult(student_id="", confidence=0.0, form_language="english", source=source)

    monkeypatch.setattr(extract_module, "_extract", fake_extract)
    return seen


@pytest.fixture
def drive(monkeypatch: pytest.MonkeyPatch) -> list[tuple[bytes, str]]:
    """Stand in for Drive and the lookup in /api/confirm; record what is saved."""
    from server.blueprints import ocr_routes

    saved: list[tuple[bytes, str]] = []

    def fake_upload(data: bytes, name: str, **_kwargs: Any) -> dict[str, str]:
        saved.append((data, name))
        return {"id": "FILE1", "name": name, "webViewLink": "https://drive.example/FILE1"}

    def fake_confirm(student_id: str, **_kwargs: Any) -> ExtractResult:
        return ExtractResult(student_id=student_id, confidence=1.0, form_language="manual", source="t")

    monkeypatch.setattr(ocr_routes, "upload_scan_to_drive", fake_upload)
    monkeypatch.setattr(ocr_routes, "confirm_student_id", fake_confirm)
    monkeypatch.setattr(ocr_routes, "log_confirm_result", lambda *_a, **_k: None)
    return saved


def _upload(client: FlaskClient, url: str, filename: str, data: bytes) -> Any:
    return client.post(
        url,
        data={"file": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def _decode_jpeg(data: bytes) -> np.ndarray:
    assert data[:2] == b"\xff\xd8", "Drive was not handed a JPEG"
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


# ------------------------------------------------------- the helper is gone
def test_scan_helper_download_is_gone(client: FlaskClient) -> None:
    response = client.get("/scan-helper.zip")

    assert response.status_code == 404


def test_no_route_serves_the_scan_helper(app: Flask) -> None:
    rules = [rule.rule for rule in app.url_map.iter_rules()]

    assert not [rule for rule in rules if "helper" in rule], rules


# ------------------------------------------------------ decoding a TIFF
def test_single_page_greyscale_tiff_decodes() -> None:
    image = preprocess.load_image_from_bytes(_tiff([_page((300, 200), paper=200, ink=30)]), "scan.tif")

    assert image.shape == _shape((300, 200))
    assert image.dtype == np.uint8
    assert image[30, 40].tolist() == [30, 30, 30]
    assert image[150, 250].tolist() == [200, 200, 200]


@pytest.mark.parametrize("compression", [None, "packbits", pytest.param("group4", marks=needs_libtiff)])
def test_one_bit_tiff_keeps_black_on_white(compression: str | None) -> None:
    image = preprocess.load_image_from_bytes(_one_bit(compression), "scan.tif")

    assert image.shape == _shape((200, 100))
    assert image[30, 40].tolist() == [0, 0, 0]
    assert image[80, 150].tolist() == [255, 255, 255]


def test_sixteen_bit_grey_tiff_is_scaled_not_clipped_to_white() -> None:
    # 0x8080 is mid-grey; clipped to 8 bits it would come out white.
    page = Image.fromarray(np.full((50, 80), 0x8080, dtype=np.uint16))
    assert page.mode.startswith("I;16")

    image = preprocess.load_image_from_bytes(_tiff([page]), "scan.tif")

    assert image.shape == _shape((80, 50))
    assert image[25, 40].tolist() == [128, 128, 128]


def test_colour_tiff_keeps_its_colours() -> None:
    page = Image.new("RGB", (60, 40), (200, 100, 50))

    image = preprocess.load_image_from_bytes(_tiff([page], compression="tiff_lzw"), "scan.tiff")

    assert image[20, 30].tolist() == [50, 100, 200]  # BGR


def test_multi_page_tiff_is_one_page_each_like_a_pdf() -> None:
    data = _stack()

    assert preprocess.count_pages_from_bytes(data, "stack.tif") == 3
    pages = preprocess.load_pages_from_bytes(data, "stack.tif")
    assert [page.shape for page in pages] == [_shape(size) for size in PAGE_SIZES]
    assert all(_is_ink(page[30, 40]) for page in pages)


def test_one_page_of_a_tiff_is_rendered_alone() -> None:
    data = _stack()

    assert preprocess.load_page_from_bytes(data, "stack.tif", 2).shape == _shape(PAGE_SIZES[1])
    assert preprocess.load_page_from_bytes(data, "stack.tif", 3).shape == _shape(PAGE_SIZES[2])
    # /api/extract reads the first page only, the same as for a PDF.
    assert preprocess.load_image_from_bytes(data, "stack.tif").shape == _shape(PAGE_SIZES[0])


@pytest.mark.parametrize("page_number", [0, 4])
def test_a_page_outside_the_tiff_is_refused(page_number: int) -> None:
    with pytest.raises(ValueError, match="outside this scan"):
        preprocess.load_page_from_bytes(_stack(), "stack.tif", page_number)


def test_upper_case_tiff_suffix_is_read() -> None:
    assert preprocess.count_pages_from_bytes(_stack(), "STACK.TIFF") == 3


def test_tiff_on_disk_is_read(tmp_path: Path) -> None:
    path = tmp_path / "scan.tif"
    path.write_bytes(_stack())

    assert preprocess.load_image_from_path(path).shape == _shape(PAGE_SIZES[0])


def test_tiff_page_turned_by_its_orientation_tag_comes_out_upright() -> None:
    # Orientation 6: the stored page must be turned a quarter clockwise, so
    # the mark stored at the top left belongs at the top right.
    stored = Image.new("L", (300, 100), 255)
    ImageDraw.Draw(stored).rectangle((0, 0, 9, 9), fill=0)
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[274] = 6

    image = preprocess.load_image_from_bytes(_tiff([stored], tiffinfo=tags), "scan.tif")

    assert image.shape == _shape((100, 300))
    assert _is_ink(image[2, 97])
    assert _is_paper(image[2, 2])


# Pillow warns about the cut-off header on its way to failing; expected here.
@pytest.mark.filterwarnings("ignore:Corrupt EXIF data:UserWarning")
@pytest.mark.parametrize("cut", [8, 60, 400, -40])
def test_cut_off_tiff_is_a_decode_error(cut: int) -> None:
    with pytest.raises(ValueError, match="Could not decode"):
        preprocess.load_pages_from_bytes(_stack()[:cut], "stack.tif")


# Files given a .tif name that are something else. Left to pick its own
# reader, Pillow would hand the EPS to Ghostscript and quietly read the JPEG,
# PNG or PDF; none of them may reach any reader at all.
EPS = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n"
NOT_A_TIFF = [
    pytest.param(lambda: EPS, id="eps"),
    pytest.param(_jpeg, id="jpeg"),
    pytest.param(lambda: _png()[0], id="png"),
    pytest.param(lambda: _pdf(1), id="pdf"),
    pytest.param(lambda: b"not a scan", id="text"),
]
LOADERS = [
    pytest.param(preprocess.load_image_from_bytes, id="first-page"),
    pytest.param(preprocess.load_pages_from_bytes, id="every-page"),
    pytest.param(preprocess.count_pages_from_bytes, id="page-count"),
    pytest.param(lambda data, name: preprocess.load_page_from_bytes(data, name, 1), id="one-page"),
]


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[list[str] | None]:
    """Record every Image.open call by the readers it allowed (None = all)."""
    calls: list[list[str] | None] = []
    real_open = Image.open

    def spy(fp: Any, mode: str = "r", formats: Any = None) -> Any:
        calls.append(None if formats is None else list(formats))
        return real_open(fp, mode, formats)

    monkeypatch.setattr(Image, "open", spy)
    return calls


@pytest.mark.parametrize("make", NOT_A_TIFF)
@pytest.mark.parametrize("load", LOADERS)
@pytest.mark.parametrize("filename", ["scan.tif", "SCAN.TIFF"])
def test_file_only_named_tif_is_refused_before_any_reader(
    opened: list[list[str] | None], make: Any, load: Any, filename: str
) -> None:
    suffix = Path(filename).suffix.lower()
    expected = rf"^Not a TIFF image, although its name ends in \{suffix}\."

    with pytest.raises(preprocess.NotATiffError, match=expected):
        load(make(), filename)

    assert opened == []


def test_file_only_named_tif_on_disk_is_refused_before_any_reader(
    opened: list[list[str] | None], tmp_path: Path
) -> None:
    path = tmp_path / "scan.tif"
    path.write_bytes(EPS)

    with pytest.raises(preprocess.NotATiffError):
        preprocess.load_image_from_path(path)

    assert opened == []


@pytest.mark.filterwarnings("ignore:Corrupt EXIF data:UserWarning")
def test_tiff_header_on_something_else_goes_to_the_tiff_reader_alone(
    opened: list[list[str] | None],
) -> None:
    # Starts like a little-endian TIFF, so only the TIFF reader may judge it.
    with pytest.raises(ValueError, match="Could not decode image") as raised:
        preprocess.load_pages_from_bytes(b"II*\x00" + EPS, "scan.tif")

    assert not isinstance(raised.value, preprocess.NotATiffError)
    assert opened and all(formats == ["TIFF"] for formats in opened)


@pytest.mark.parametrize("url", ["/api/extract", "/api/extract-batch"])
def test_upload_only_named_tif_is_refused_with_its_scan_id(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray], url: str
) -> None:
    response = _upload(client, url, "scan.tif", EPS)

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"].startswith("Not a TIFF image, although its name ends in .tif.")
    assert len(body["scan_id"]) == 32
    assert reader == []


def test_page_image_jpeg_renders_the_tiff_page_asked_for(stash_dir: Path) -> None:
    scan_id = scan_store.stash_upload(_stack(), "stack.tif")

    rendered = scan_store.page_image_jpeg(scan_id, 3)

    assert rendered is not None
    jpeg, ext = rendered
    assert ext == "jpg"
    assert _decode_jpeg(jpeg).shape == _shape(PAGE_SIZES[2])


# ------------------------------------------------------- TIFF over the API
def test_extract_reads_the_first_page_of_a_tiff(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray]
) -> None:
    response = _upload(client, "/api/extract", "stack.tif", _stack())

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert len(body["scan_id"]) == 32
    assert body["page_number"] == 1
    assert [image.shape for image in reader] == [_shape(PAGE_SIZES[0])]


def test_extract_batch_reads_every_page_of_a_tiff(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray]
) -> None:
    response = _upload(client, "/api/extract-batch", "stack.tiff", _stack())

    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["page_count"] == 3
    assert [page["page_number"] for page in body["pages"]] == [1, 2, 3]
    assert {page["scan_id"] for page in body["pages"]} == {body["scan_id"]}
    assert [image.shape for image in reader] == [_shape(size) for size in PAGE_SIZES]


@pytest.mark.parametrize("compression", [None, pytest.param("group4", marks=needs_libtiff)])
def test_extract_batch_reads_a_one_bit_tiff(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray], compression: str | None
) -> None:
    response = _upload(client, "/api/extract-batch", "scan.tif", _one_bit(compression))

    assert response.status_code == 200, response.get_json()
    [image] = reader
    assert _is_ink(image[30, 40])
    assert _is_paper(image[80, 150])


def test_confirm_saves_the_tiff_page_it_was_given_to_drive(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray], drive: list[tuple[bytes, str]]
) -> None:
    batch = _upload(client, "/api/extract-batch", "stack.tif", _stack()).get_json()

    response = client.post(
        "/api/confirm",
        json={"student_id": "MOHSHA011021", "scan_id": batch["scan_id"], "page_number": 2},
    )

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["image_link"] == "https://drive.example/FILE1"
    [(jpeg, name)] = drive
    assert name == "NOTFOUND_MOHSHA011021.jpg"
    page = _decode_jpeg(jpeg)
    assert page.shape == _shape(PAGE_SIZES[1])
    assert _is_ink(page[30, 40])


def test_tiff_with_a_non_latin_name_keeps_its_extension(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray]
) -> None:
    response = _upload(client, "/api/extract-batch", "प्रवेश_फॉर्म.tif", _stack())

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["filename"] == "upload.tif"


@pytest.mark.filterwarnings("ignore:Corrupt EXIF data:UserWarning")
def test_damaged_tiff_is_refused_but_keeps_its_scan_id(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray]
) -> None:
    response = _upload(client, "/api/extract-batch", "stack.tif", _stack()[:400])

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"].startswith("Could not decode")
    assert len(body["scan_id"]) == 32
    assert reader == []


# -------------------------------------------- other types are unchanged
@pytest.mark.parametrize(
    ("filename", "make", "shapes"),
    [
        ("form.jpg", _jpeg, [(90, 160, 3)]),
        ("form.jpeg", _jpeg, [(90, 160, 3)]),
        ("form.png", lambda: _png()[0], [(90, 160, 3)]),
        # 200 x 300 pt at the 200 dpi a PDF page is rendered at.
        ("forms.pdf", _pdf, [(834, 556, 3), (834, 556, 3)]),
    ],
)
def test_jpg_png_and_pdf_uploads_read_as_before(
    client: FlaskClient,
    stash_dir: Path,
    reader: list[np.ndarray],
    filename: str,
    make: Any,
    shapes: list[tuple[int, int, int]],
) -> None:
    response = _upload(client, "/api/extract-batch", filename, make())

    assert response.status_code == 200, response.get_json()
    assert response.get_json()["page_count"] == len(shapes)
    assert [image.shape for image in reader] == shapes


def test_png_upload_reaches_the_reader_unchanged(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray]
) -> None:
    data, original = _png()

    response = _upload(client, "/api/extract", "form.png", data)

    assert response.status_code == 200, response.get_json()
    [image] = reader
    assert np.array_equal(image, original)


def test_confirm_still_saves_a_png_page_to_drive_as_jpeg(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray], drive: list[tuple[bytes, str]]
) -> None:
    scan_id = _upload(client, "/api/extract", "form.png", _png()[0]).get_json()["scan_id"]

    response = client.post("/api/confirm", json={"student_id": "MOHSHA011021", "scan_id": scan_id})

    assert response.status_code == 200, response.get_json()
    [(jpeg, _name)] = drive
    assert _decode_jpeg(jpeg).shape == (90, 160, 3)


@pytest.mark.parametrize("filename", ["scan.bmp", "scan.gif", "scan.webp", "scan.heic", "scan.exe"])
@pytest.mark.parametrize("url", ["/api/extract", "/api/extract-batch"])
def test_other_types_are_still_refused(
    client: FlaskClient, stash_dir: Path, reader: list[np.ndarray], url: str, filename: str
) -> None:
    response = _upload(client, url, filename, b"not a scan")

    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Unsupported file type")
    assert not stash_dir.exists() or not any(stash_dir.iterdir())
    assert reader == []
