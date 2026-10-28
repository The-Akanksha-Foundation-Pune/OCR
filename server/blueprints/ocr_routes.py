"""OCR upload routes and simple web UI."""

from __future__ import annotations

import ast
import io
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from flask import Blueprint, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from server.config import ALLOWED_EXTENSIONS, BASE_DIR
from server.services.auth import require_login
from server.services.debug_log import debug_log
from server.services.drive_store import upload_scan_to_drive
from server.services.extract_student_id import (
    ExtractError,
    confirm_student_id,
    extract_student_id_from_upload,
    extract_student_ids_from_upload_batch,
    log_confirm_result,
)
from server.services.scan_store import page_image_jpeg, stash_upload
from server.services.student_lookup import (
    find_student_by_id,
    list_schools_with_location,
)

ocr_bp = Blueprint("ocr", __name__)
ocr_bp.before_request(require_login)

# The scan helper the Scanner card offers to download. It runs on the
# computer the scanner is plugged into, never on this server - a browser
# cannot reach a USB scanner, and the server may be on another continent.
# scan_agent.py is the HTTP side, scanner_devices.py finds scanners and
# missing drivers, scan_naps2.py drives NAPS2, and scan.ps1 drives a WIA
# scanner on Windows (kept because it hands over each page as it lands).
HELPER_DIR = BASE_DIR / "tools"
HELPER_FILES = ("scan_agent.py", "scanner_devices.py", "scan_naps2.py", "scan.ps1")
HELPER_ZIP_NAME = "akanksha-scan-helper.zip"


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@ocr_bp.get("/")
def index():
    # Scanning happens on the person's own computer through the scan helper,
    # so nothing on this page depends on which OS the server runs on.
    return render_template("index.html", active_page="scanner")


def _page_origin(page_url: str) -> str:
    """scheme://host[:port] of the page, spelt the way a browser sends Origin.

    A browser leaves out the default port and lowercases the host, and the
    helper compares Origin as a plain string, so this must do the same.
    """
    parts = urlsplit(page_url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"  # an IPv6 address keeps its brackets in Origin
    try:
        port = parts.port
    except ValueError:
        port = None
    if port and port != {"http": 80, "https": 443}.get(parts.scheme):
        host = f"{host}:{port}"
    return f"{parts.scheme}://{host}"


def _helper_built_in_origins() -> set[str]:
    """The page addresses tools/scan_agent.py answers without --origin.

    Read from the helper's own source rather than copied here, so the read-me
    cannot drift from the helper it ships with. Empty when it cannot be read,
    which only costs an --origin the helper did not need.
    """
    try:
        tree = ast.parse((HELPER_DIR / "scan_agent.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "BUILT_IN_ORIGINS" for t in targets):
            try:
                return {str(o) for o in ast.literal_eval(node.value)}
            except (ValueError, TypeError, SyntaxError):
                return set()
    return set()


def _how_to_run(page_url: str) -> str:
    """The read-me that goes in the helper download.

    Kept to what a school office needs to get from a fresh download to a
    working Scan button; the page itself explains the rest. CRLF line ends
    because it is mostly opened in Windows Notepad.
    """
    # The helper only answers pages at addresses it knows (production and
    # local runs). Reached any other way - an ngrok tunnel, a LAN address, a
    # staging host - the page could not see the helper and would say it is
    # not running, so the run lines name this page's address themselves.
    origin = _page_origin(page_url)
    flag = "" if origin in _helper_built_in_origins() else f" --origin {origin}"
    lines = [
        "Akanksha scan helper",
        "====================",
        "",
        "This small program lets the Scanner page use the scanner plugged into",
        "THIS computer. It only talks to a browser on this same computer, holds",
        "no passwords, and never contacts the server itself - the page uploads",
        "the pages using your own sign-in.",
        "",
        "Leave its window open while you scan. Closing the window stops it.",
        "",
        "Windows",
        "-------",
        "1. Install Python 3 from https://www.python.org/downloads/ - tick",
        '   "Add python.exe to PATH" on the first screen of the installer.',
        "2. Install the scanner's own Windows driver (for a Fujitsu / Ricoh",
        "   SP-1130N that is PaperStream IP, which includes the WIA driver).",
        "3. Optional: install NAPS2 from https://www.naps2.com/ - only needed for",
        "   scanners with a TWAIN-only driver, or network (AirScan) scanners.",
        "4. Open this folder, type cmd in the address bar, press Enter, then run:",
        f"       python scan_agent.py{flag}",
        f'   (if "python" is not found, try:  py scan_agent.py{flag})',
        "",
        "macOS",
        "-----",
        "1. Install NAPS2 from https://www.naps2.com/ (or: brew install --cask naps2).",
        "2. Install the scanner's Mac driver from the maker's website, and check",
        "   the scanner shows up in Image Capture. For a Fujitsu / Ricoh SP-1130N",
        "   that is Ricoh's ICA driver, and it works over USB only.",
        "3. Open Terminal, type  cd  and a space, drag this folder onto the",
        "   Terminal window, press Return, then run:",
        f"       python3 scan_agent.py{flag}",
        "   If macOS offers to install the command line developer tools, accept,",
        "   then run it again.",
        "",
        "Linux",
        "-----",
        "1. Install NAPS2 from https://www.naps2.com/ (the .deb / .rpm package or",
        "   the Flatpak). NAPS2 scans through SANE, so a USB scanner also needs",
        "   its SANE / Linux driver (sane-utils, or the maker's own Linux driver).",
        "   Network (AirScan) scanners need no extra driver.",
        "2. In a terminal in this folder run:",
        f"       python3 scan_agent.py{flag}",
        "",
        "Then",
        "----",
        f"Open {page_url} in Chrome or Edge and choose your scanner on the",
        "Scanner card. If the browser asks to allow access to devices on your",
        "local network, allow it - that is how the page reaches this helper.",
        "",
        "Optional settings, as environment variables on THIS computer:",
        "  NAPS2_PATH                  where NAPS2 is, if it is not found by itself",
        "  SCAN_AGENT_ORIGINS          extra web addresses allowed to use the helper",
        "                              (comma-separated)",
        "  SCAN_AGENT_TIMEOUT_SECONDS  stop a scan that runs longer (default 900)",
        "",
    ]
    return "\r\n".join(lines)


@ocr_bp.get("/scan-helper.zip")
def scan_helper_zip():
    """
    The scan helper, zipped fresh on every request.

    Built from tools/ rather than kept as a prebuilt zip so the download can
    never drift from the code in the repository - deploying a helper fix is
    the same as deploying anything else.
    """
    buffer = io.BytesIO()
    included = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in HELPER_FILES:
            path = HELPER_DIR / name
            if not path.is_file():
                # Say so in the log rather than failing the download: the
                # rest of the helper is still worth having, and a half-deployed
                # tools/ folder should be noticed, not hidden behind a 500.
                debug_log(f"[HELPER] {path} is missing - left out of {HELPER_ZIP_NAME}")
                continue
            bundle.write(path, arcname=name)
            included.append(name)
        bundle.writestr("HOW-TO-RUN.txt", _how_to_run(request.url_root))
    buffer.seek(0)
    debug_log(f"[HELPER] download {HELPER_ZIP_NAME} files={included!r}")

    response = send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=HELPER_ZIP_NAME,
    )
    # Always fetch a fresh copy: an old helper cached in the browser is
    # exactly what the "helper is out of date" message is there to fix.
    response.headers["Cache-Control"] = "no-store"
    return response


@ocr_bp.get("/api/schools")
def schools_api():
    """Schools to choose from before scanning a stack, grouped by city."""
    try:
        return jsonify({"schools": list_schools_with_location()}), 200
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Could not load schools: {exc}"}), 500


@ocr_bp.post("/api/extract")
def extract_api():
    if "file" not in request.files:
        return jsonify({"error": "Missing file field. Upload as multipart with key 'file'."}), 400

    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(uploaded.filename)
    if not _is_allowed(filename):
        return jsonify(
            {
                "error": "Unsupported file type. Use JPG, PNG, BMP, TIFF, or PDF."
            }
        ), 400

    try:
        data = uploaded.read()
        scan_id = stash_upload(data, filename)
        result = extract_student_id_from_upload(data, filename)
        payload = result.to_dict()
        payload["scan_id"] = scan_id
        payload["page_number"] = payload.get("page_number") or 1
        debug_log(
            f"[API] PREVIEW file={filename!r} student_id={payload.get('student_id')!r} "
            f"scan_id={scan_id!r} found_in_db={payload.get('found_in_db')} "
            f"message={payload.get('message')!r}"
        )
        return jsonify(payload), 200
    except ExtractError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Extraction failed: {exc}"}), 500


@ocr_bp.post("/api/extract-batch")
def extract_batch_api():
    """
    Same OCR-only preview as /api/extract, but reads EVERY page — a
    printer/scanner machine often batch-scans a whole stack of forms into
    one multi-page PDF. Returns one preview per page for the UI to walk
    through and confirm one by one. Single-page uploads (incl. phone
    photos) just come back as a 1-page batch.
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing file field. Upload as multipart with key 'file'."}), 400

    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(uploaded.filename)
    if not _is_allowed(filename):
        return jsonify(
            {
                "error": "Unsupported file type. Use JPG, PNG, BMP, TIFF, or PDF."
            }
        ), 400

    try:
        data = uploaded.read()
        scan_id = stash_upload(data, filename)
        results = extract_student_ids_from_upload_batch(data, filename)
        pages = []
        for result in results:
            page = result.to_dict()
            page["scan_id"] = scan_id
            pages.append(page)
        payload = {"filename": filename, "page_count": len(pages), "pages": pages, "scan_id": scan_id}
        debug_log(
            f"[API] BATCH PREVIEW file={filename!r} page_count={len(pages)} "
            f"scan_id={scan_id!r} student_ids={[p.get('student_id') for p in pages]!r}"
        )
        return jsonify(payload), 200
    except ExtractError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Batch extraction failed: {exc}"}), 500


@ocr_bp.post("/api/confirm")
def confirm_api():
    """
    Final step: user confirms Student ID → DB lookup first → upload scan
    to Drive (found or not) → log DrivelinkImage.
    """
    payload = request.get_json(silent=True) or {}
    student_id = str(payload.get("student_id", "")).strip()
    if not student_id:
        return jsonify({"error": "Student ID cannot be empty."}), 400

    source = str(payload.get("source") or "manual-confirm").strip() or "manual-confirm"
    scan_id = str(payload.get("scan_id") or "").strip()
    page_number = int(payload.get("page_number") or 1)
    # Which school's forms are being scanned. Narrowing the candidate list
    # is what makes unattended saving safe enough to do: it removes every
    # cross-school near-twin, and it disambiguates the Student IDs that are
    # duplicated across two schools.
    school = str(payload.get("school") or "").strip()
    # Location is the fallback scope: with the school left on "All Pune
    # schools" it still keeps the search inside Pune.
    location = str(payload.get("location") or "").strip()

    try:
        # 1) Search DB (and mark scanned_at when found)
        result = confirm_student_id(
            student_id,
            source=source,
            defer_log=True,
            school=school or None,
            location=location or None,
        )
        data = result.to_dict()
        image_link: str | None = None

        # 2) Always save the scan image to Drive when we have the file
        #    (found → StudentName_ID; not found → NOTFOUND_ID)
        if scan_id:
            student = result.student or {}
            if result.found_in_db:
                student_name = (student.get("student_name") or "").strip() or result.student_id
                display_name = f"{student_name}_{result.student_id}"
            else:
                display_name = f"NOTFOUND_{result.student_id}"

            page_bytes = page_image_jpeg(scan_id, page_number)
            if page_bytes:
                jpeg_data, _ext = page_bytes
                drive_meta = upload_scan_to_drive(
                    jpeg_data,
                    f"{display_name}.jpg",
                    display_name=display_name,
                )
                if drive_meta:
                    image_link = drive_meta.get("webViewLink") or None
                    data["drive"] = {
                        "id": drive_meta.get("id"),
                        "name": drive_meta.get("name"),
                        "webViewLink": image_link,
                    }

        # 3) Log attempt with Drive link in DrivelinkImage
        log_confirm_result(
            result,
            image_link=image_link,
            scan_school=school or None,
            scan_location=location or None,
        )
        data["image_link"] = image_link
        data["DrivelinkImage"] = image_link

        debug_log(
            f"[API] CONFIRM student_id={data.get('student_id')!r} "
            f"found_in_db={data.get('found_in_db')} DrivelinkImage={image_link!r} "
            f"message={data.get('message')!r}"
        )
        return jsonify(data), 200
    except ExtractError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception as exc:  # noqa: BLE001 — return safe API error
        return jsonify({"error": f"Lookup failed: {exc}"}), 500
