"""
Run the scanner in the background so reading can start on page 1 while the
feeder is still pulling page 2.

The old flow fed the whole stack, then read it - so nothing appeared on
screen until every sheet had been through the rollers. Feeding dominates the
wall clock, so that idle reading time was pure waste, and worse, the operator
watched a spinner with nothing to show for it.

Here scan.ps1 writes each page into a job folder as it captures it (renaming
from a .part file so a half-written image is never picked up), and the web
layer polls that folder. Pages get OCR'd as they land.
"""

from __future__ import annotations

import re
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from server.config import SCAN_SCRIPT, SCANNER_NAME, UPLOAD_FOLDER
from server.services.debug_log import debug_log

JOB_ROOT = UPLOAD_FOLDER / "scan-jobs"
_PAGE_RE = re.compile(r"^page-(\d{3})\.jpg$")


@dataclass
class ScanJob:
    job_id: str
    folder: Path
    process: subprocess.Popen | None = None
    error: str | None = None
    logged_output: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


_jobs: dict[str, ScanJob] = {}
_jobs_lock = threading.Lock()


def start_scan(scanner_name: str | None = None) -> ScanJob:
    """Launch a scan and return immediately; pages appear in job.folder."""
    if not SCAN_SCRIPT.exists():
        raise FileNotFoundError(f"Scan script is missing: {SCAN_SCRIPT}")

    job_id = uuid.uuid4().hex
    folder = JOB_ROOT / job_id
    folder.mkdir(parents=True, exist_ok=True)

    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(SCAN_SCRIPT),
        "-StreamDir",
        str(folder),
    ]
    name = (scanner_name or SCANNER_NAME).strip()
    if name:
        command += ["-Scanner", name]

    debug_log(f"[SCANJOB] {job_id} starting: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job = ScanJob(job_id=job_id, folder=folder, process=process)
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> ScanJob | None:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id or ""):
        return None
    with _jobs_lock:
        return _jobs.get(job_id)


def page_paths(job: ScanJob) -> list[Path]:
    """Pages captured so far, in order. Only fully-written files appear."""
    try:
        found = [
            path
            for path in job.folder.iterdir()
            if _PAGE_RE.match(path.name)
        ]
    except OSError:
        return []
    return sorted(found, key=lambda p: p.name)


def job_status(job: ScanJob) -> dict:
    """
    Where the scan has got to.

    `finished` means the scanner has stopped, not that every page has been
    read - the caller keeps reading whatever is already on disk.
    """
    done_marker = job.folder / "done.txt"
    failed_marker = job.folder / "failed.txt"

    error = job.error
    if failed_marker.exists() and not error:
        try:
            error = failed_marker.read_text(encoding="utf-8").strip()
        except OSError:
            error = "The scan failed."

    finished = done_marker.exists() or bool(error)

    # Once the scanner stops, drain its output into the debug log. It carries
    # the per-page transfer timings, which are the only way to tell a slow
    # feeder apart from slow saving - and nobody should have to run the
    # script by hand to see them.
    if finished and not job.logged_output:
        with job.lock:
            if not job.logged_output:
                job.logged_output = True
                output = ""
                if job.process is not None:
                    try:
                        job.process.wait(timeout=5)
                        output = (job.process.stdout.read() or "").strip()
                    except Exception:  # noqa: BLE001
                        pass
                for line in output.splitlines():
                    if line.strip():
                        debug_log(f"[SCANJOB] {job.job_id[:8]} | {line.strip()}")

    # A process that exited without writing either marker died unexpectedly;
    # without this the poller would wait for pages that will never arrive.
    if not finished and job.process is not None:
        code = job.process.poll()
        if code is not None:
            finished = True
            if not error:
                output = ""
                try:
                    if not job.logged_output:
                        output = (job.process.stdout.read() or "").strip()
                except Exception:  # noqa: BLE001
                    pass
                error = _friendly_error(output) if output else (
                    f"The scanner stopped unexpectedly (exit {code})."
                )
                debug_log(f"[SCANJOB] {job.job_id} died: {error}")

    pages = page_paths(job)
    return {
        "job_id": job.job_id,
        "pages_ready": len(pages),
        "finished": finished,
        "error": error,
    }


def discard_job(job_id: str) -> None:
    """Delete a finished job's images."""
    job = get_job(job_id)
    if job is None:
        return
    with _jobs_lock:
        _jobs.pop(job_id, None)
    try:
        for path in job.folder.iterdir():
            path.unlink(missing_ok=True)
        job.folder.rmdir()
    except OSError as exc:
        debug_log(f"[SCANJOB] could not clean {job.folder}: {exc}")


def _friendly_error(output: str) -> str:
    """Same wording as the blocking path, so operators see one vocabulary."""
    lowered = output.lower()
    if "feeder is empty" in lowered or "nothing was scanned" in lowered:
        return (
            "The feeder is empty. Load the forms face down into the ADF chute "
            "at the back, pushing until the rollers grip, then scan again."
        )
    if "jam" in lowered:
        return (
            "The scanner reported a paper jam. Open the cover, clear the sheet, "
            "reload the stack and try again."
        )
    if "no scanner found" in lowered or "no scanner matched" in lowered:
        return "No scanner is connected. Check it is powered on and plugged in."
    tail = [line.strip() for line in output.splitlines() if line.strip()]
    return tail[-1] if tail else "The scan failed. Check the scanner and try again."
