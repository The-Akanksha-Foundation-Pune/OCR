"""A page claimed by a reader that has since stopped goes back at once.

Without this, restarting the dev server or deploying left the page that
was being read waiting out the whole stale-claim timeout (10 minutes).
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import pytest

from server.services import debug_log, scan_queue


def _queue_page() -> str:
    page_id = secrets.token_hex(16)
    pending = scan_queue.queue_root() / "_pending"
    pending.mkdir(parents=True, exist_ok=True)
    (pending / page_id).touch()
    return page_id


def _claim_as(pid: int, page_id: str, token_tail: str = "abcd1234") -> Path:
    """Put a claim on disk as if process `pid` had taken it."""
    reading = scan_queue.queue_root() / "_reading"
    reading.mkdir(parents=True, exist_ok=True)
    (scan_queue.queue_root() / "_pending" / page_id).unlink(missing_ok=True)
    marker = reading / f"{page_id}.{int(time.time() * 1000)}.{pid:08x}{token_tail}"
    marker.touch()
    return marker


def _dead_pid() -> int:
    """The id of a process that has already exited."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _pending_ids() -> set[str]:
    folder = scan_queue.queue_root() / "_pending"
    return set(os.listdir(folder)) if folder.exists() else set()


def test_a_claim_carries_the_readers_pid() -> None:
    page_id = _queue_page()
    claim = scan_queue.claim_next()
    assert claim is not None and claim.page_id == page_id
    token = claim.marker.name.rsplit(".", 1)[1]
    assert len(token) == 16 and int(token[:8], 16) == os.getpid()


@pytest.mark.skipif(os.name != "posix", reason="pid checks run on POSIX only")
def test_a_dead_readers_page_goes_back_at_once() -> None:
    page_id = _queue_page()
    marker = _claim_as(_dead_pid(), page_id)

    assert scan_queue.reclaim_orphaned() == [page_id]
    assert not marker.exists() and page_id in _pending_ids()
    # The ordinary stale sweep would still have held on to it.
    assert scan_queue.reclaim_stale() == []


@pytest.mark.skipif(os.name != "posix", reason="pid checks run on POSIX only")
def test_a_live_readers_page_and_our_own_stay_claimed() -> None:
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        theirs = _claim_as(sleeper.pid, _queue_page())
        ours = _claim_as(os.getpid(), _queue_page())
        assert scan_queue.reclaim_orphaned() == []
        assert theirs.exists() and ours.exists()
    finally:
        sleeper.kill()
        sleeper.wait()


def test_old_format_claims_wait_for_the_stale_timeout() -> None:
    page_id = _queue_page()
    reading = scan_queue.queue_root() / "_reading"
    reading.mkdir(parents=True, exist_ok=True)
    (scan_queue.queue_root() / "_pending" / page_id).unlink()
    old = reading / f"{page_id}.{int(time.time() * 1000)}.{secrets.token_hex(6)}"  # no pid
    old.touch()

    assert scan_queue.reclaim_orphaned() == []
    assert old.exists()


def test_windows_never_signals_a_process(monkeypatch: pytest.MonkeyPatch) -> None:
    page_id = _queue_page()
    marker = _claim_as(_dead_pid(), page_id)

    def must_not_run(*_args: object) -> None:
        raise AssertionError("os.kill(pid, 0) terminates the process on Windows")

    monkeypatch.setattr(scan_queue.os, "name", "nt")
    monkeypatch.setattr(scan_queue.os, "kill", must_not_run)
    assert scan_queue.reclaim_orphaned() == []
    assert marker.exists()


def test_tests_never_write_to_the_real_log_or_queue(tmp_path: Path) -> None:
    real_root = Path(__file__).resolve().parent.parent
    assert debug_log.LOG_FILE.is_relative_to(tmp_path)
    assert scan_queue.queue_root().is_relative_to(tmp_path)
    assert not debug_log.LOG_FILE.is_relative_to(real_root)
    debug_log.debug_log("[TEST] this line belongs in the temporary log")
    assert "[TEST]" in debug_log.LOG_FILE.read_text(encoding="utf-8")
