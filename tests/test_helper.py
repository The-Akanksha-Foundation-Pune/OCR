"""The local scan helper (tools/scan_agent.py).

Nothing here starts a real process or touches a real scanner: the scan
engine, the scanner check and the process killer are all stand-ins, every
job folder lives under the test's own tmp_path, and the helper listens on a
port the operating system picks rather than 8765, where a real helper may be
running.

Bugs covered:
  #1   the defences the v1 helper lacked - a foreign Origin, a rebinding
       Host and a no-preflight text/plain POST are refused, and only one
       scan runs at a time - with the page's own round trip still working.
  #13  a request without an Origin (another site's <img> or <script>) cannot
       make the helper re-run its full scanner check again and again, while
       the README's curl check by hand still gets a fresh one.
  #14  job folders left by an earlier run are swept at start-up - also once
       Finder or Explorer has left its own file in one, and the pages a
       Flatpak NAPS2 staged in its own folder - and on Windows closing the
       console window still clears away held jobs.
"""

from __future__ import annotations

import base64
import ctypes
import errno
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    # The helper runs as a script from tools/ and imports its neighbours
    # by their bare names, so the tests do the same.
    sys.path.insert(0, str(TOOLS))

import scan_agent
import scanner_devices

PAGE = "https://tether.akanksha.org"
EVIL = "https://evil.example"
HELPER_HOST = f"127.0.0.1:{scan_agent.PORT}"
REBOUND_HOST = f"rebind.evil.example:{scan_agent.PORT}"

DEVICE = {
    "id": "escl:Fujitsu SP-1130N",
    "name": "Fujitsu SP-1130N",
    "driver": "escl",
    "connection": "network",
    "engine": "naps2",
    "status": "ready",
}
OPTIONS = {"device_id": DEVICE["id"], "source": "feeder", "dpi": 300, "color": "gray"}

# What a cross-site <img src="http://127.0.0.1:8765/..."> sends: no Origin.
CROSS_SITE_IMAGE = {
    "Referer": f"{EVIL}/",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Dest": "image",
}

# What a tool on the scanning computer sends when someone checks the helper
# by hand: no Origin, no Sec-Fetch-*, and a User-Agent of its own. In
# Windows PowerShell "curl" is Invoke-WebRequest, which borrows "Mozilla/".
BY_HAND = {
    "curl": "curl/8.7.1",
    "windows-powershell": (
        "Mozilla/5.0 (Windows NT; Windows NT 10.0; en-GB) WindowsPowerShell/5.1.19041.4648"
    ),
    "powershell-7": (
        "Mozilla/5.0 (Windows NT 10.0; Microsoft Windows 10.0.19045; en-GB) PowerShell/7.4.6"
    ),
    "python": "Python-urllib/3.9",
}
BROWSER = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/15.6.1 Safari/605.1.15"
)

_NO_BODY = object()


# ------------------------------------------------------------- stand-ins
class FakeEngine:
    """Stands in for scan.ps1 or NAPS2: records the command, runs nothing."""

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.returncode: int | None = None
        self.stopped = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode

    def stream_dir(self) -> Path:
        return Path(self.command[self.command.index("--stream-dir") + 1])


@dataclass
class Engines:
    """Every engine the helper "started", in order."""

    started: list[FakeEngine] = field(default_factory=list)

    def popen(self, command: list[str], **kwargs: Any) -> FakeEngine:
        engine = FakeEngine(command, **kwargs)
        self.started.append(engine)
        return engine


@dataclass
class Checks:
    """Counts full scanner checks (check_devices) instead of running them."""

    count: int = 0

    def check(self, platform: str | None = None) -> dict[str, Any]:
        self.count += 1
        return {
            "checked_at": "2026-10-27T10:00:00+05:30",
            "platform": sys.platform,
            "naps2": {"installed": True, "path": "/fake/naps2", "install_hint": ""},
            "devices": [dict(DEVICE)],
            "unready": [],
            "errors": [],
        }


@dataclass
class Rig:
    """The helper's world with every outside effect replaced."""

    engines: Engines
    checks: Checks
    temp: Path
    home: Path


@dataclass
class Reply:
    status: int
    body: dict[str, Any]
    headers: Message


@dataclass
class Helper:
    """A real helper Handler listening on a spare loopback port."""

    port: int
    rig: Rig

    def call(
        self,
        method: str,
        path: str,
        *,
        origin: str | None = PAGE,
        host: str = HELPER_HOST,
        json_body: Any = _NO_BODY,
        content_type: str | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Reply:
        sent = {"Host": host}
        if origin is not None:
            sent["Origin"] = origin
        if json_body is not _NO_BODY:
            body = json.dumps(json_body).encode("utf-8")
            content_type = content_type or "application/json"
        if content_type is not None:
            sent["Content-Type"] = content_type
        sent.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=sent)
            response = connection.getresponse()
            raw = response.read()
            return Reply(
                response.status, json.loads(raw) if raw else {}, response.headers
            )
        finally:
            connection.close()


def _refuse_real_process(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("the helper tests must never start a real process")


def _fake_kill(process: FakeEngine, own_group: bool = False) -> None:
    process.stopped = True
    if process.returncode is None:
        process.returncode = -15


# -------------------------------------------------------------- fixtures
@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Rig]:
    """Stub the engine, the scanner check and the killer; jobs go to tmp_path."""
    engines, checks = Engines(), Checks()
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    # The start-up sweep also looks in ~/.var/app; keep it off the real one.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    # The helper's own view of subprocess gets the fake engine; the real
    # module refuses outright, so nothing else can slip a process through.
    monkeypatch.setattr(
        scan_agent,
        "subprocess",
        types.SimpleNamespace(
            Popen=engines.popen,
            STDOUT=subprocess.STDOUT,
            DEVNULL=subprocess.DEVNULL,
            CREATE_NEW_PROCESS_GROUP=0x200,
        ),
    )
    monkeypatch.setattr(subprocess, "Popen", _refuse_real_process)
    monkeypatch.setattr(scanner_devices, "check_devices", checks.check)
    monkeypatch.setattr(scanner_devices, "_devices_cache", {"result": None, "at": None})
    monkeypatch.setattr(scanner_devices, "find_naps2", lambda *_a, **_k: "/fake/naps2")
    monkeypatch.setattr(scanner_devices, "kill_process_tree", _fake_kill)
    monkeypatch.setattr(scan_agent, "_jobs", {})
    monkeypatch.setattr(scan_agent, "ALLOWED_ORIGINS", set(scan_agent.BUILT_IN_ORIGINS))
    yield Rig(engines, checks, temp, home)
    scan_agent._discard_all()


@pytest.fixture
def helper(rig: Rig) -> Iterator[Helper]:
    server = scan_agent.Server(("127.0.0.1", 0), scan_agent.Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield Helper(port=server.server_address[1], rig=rig)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _start(helper: Helper) -> str:
    reply = helper.call("POST", "/scan", json_body=OPTIONS)
    assert reply.status == 200, reply.body
    return str(reply.body["job"])


# ----------------------------------------------- #1: the v2 helper's defences
def test_foreign_origin_cannot_start_a_scan(helper: Helper) -> None:
    reply = helper.call("POST", "/scan", origin=EVIL, json_body={})

    assert reply.status == 403
    assert "not allowed" in reply.body["error"]
    assert reply.headers.get("Access-Control-Allow-Origin") is None
    assert helper.rig.engines.started == []
    assert scan_agent._jobs == {}


@pytest.mark.parametrize(
    ("origin", "content_type", "status"),
    [
        # fetch(..., {method: "POST", mode: "no-cors", body: "{}"}) from
        # another site: a simple request, so no preflight stops it.
        (EVIL, "text/plain;charset=UTF-8", 403),
        (EVIL, "application/x-www-form-urlencoded", 403),
        # Even the app's own page must send JSON, so every scan from a page
        # goes through a preflight first.
        (PAGE, "text/plain;charset=UTF-8", 415),
        (PAGE, "multipart/form-data; boundary=x", 415),
    ],
)
def test_post_without_json_cannot_start_a_scan(
    helper: Helper, origin: str, content_type: str, status: int
) -> None:
    reply = helper.call(
        "POST", "/scan", origin=origin, content_type=content_type, body=b"{}"
    )

    assert reply.status == status
    assert helper.rig.engines.started == []
    assert scan_agent._jobs == {}


def test_rebound_host_can_neither_start_a_scan_nor_read_one(helper: Helper) -> None:
    job = _start(helper)
    folder = helper.rig.engines.started[0].stream_dir()
    (folder / "page-001.jpg").write_bytes(b"CONSENT-FORM")

    # A site that pointed its own name at 127.0.0.1 is same-origin as far
    # as the browser knows, so its GETs carry no Origin - only the Host
    # gives it away.
    start = helper.call(
        "POST",
        "/scan",
        host=REBOUND_HOST,
        origin=f"http://{REBOUND_HOST}",
        json_body={},
    )
    read = helper.call("GET", f"/pages?job={job}", host=REBOUND_HOST, origin=None)
    health = helper.call("GET", "/health", host=REBOUND_HOST, origin=None)

    assert start.status == 403
    assert read.status == 403
    assert "pages" not in read.body
    assert health.status == 403
    assert len(helper.rig.engines.started) == 1


def test_only_one_scan_runs_at_a_time(helper: Helper) -> None:
    _start(helper)

    second = helper.call("POST", "/scan", json_body=OPTIONS)

    assert second.status == 409
    assert "already running" in second.body["error"]
    assert len(helper.rig.engines.started) == 1

    # Once that scan is over the next one may start.
    helper.rig.engines.started[0].returncode = 0
    _start(helper)
    assert len(helper.rig.engines.started) == 2


def test_two_presses_at_once_cannot_both_start_a_scan(rig: Rig) -> None:
    # Two requests can both pass the route's check before either has
    # started; _start_scan checks again under the lock.
    scan_agent._start_scan(DEVICE, OPTIONS)

    with pytest.raises(scan_agent.Refused) as refused:
        scan_agent._start_scan(DEVICE, OPTIONS)

    assert refused.value.status == 409
    assert len(rig.engines.started) == 1
    # The refused one's folder does not linger.
    assert len(list(rig.temp.glob("scan-agent-*"))) == 1


def test_the_pages_own_round_trip_still_works(helper: Helper) -> None:
    preflight = helper.call(
        "OPTIONS",
        "/scan",
        headers={
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert preflight.status == 204
    assert preflight.headers.get("Access-Control-Allow-Origin") == PAGE
    assert preflight.headers.get("Access-Control-Allow-Private-Network") == "true"

    health = helper.call("GET", "/health")
    assert health.status == 200
    assert health.body["version"] == scan_agent.VERSION
    assert health.headers.get("Access-Control-Allow-Origin") == PAGE

    devices = helper.call(
        "GET", "/devices?refresh=1", host=f"localhost:{scan_agent.PORT}"
    )
    assert devices.status == 200
    assert [d["id"] for d in devices.body["devices"]] == [DEVICE["id"]]

    job = _start(helper)
    engine = helper.rig.engines.started[0]
    folder = engine.stream_dir()
    assert folder.parent == helper.rig.temp
    assert engine.command[engine.command.index("--device") + 1] == DEVICE["name"]
    (folder / "page-001.jpg").write_bytes(b"PAGE-ONE")
    (folder / "page-002.jpg").write_bytes(b"PAGE-TWO")
    (folder / "done.txt").write_text("2", encoding="utf-8")
    engine.returncode = 0

    pages = helper.call("GET", f"/pages?job={job}&after=0")
    assert pages.status == 200
    assert pages.body["finished"] is True
    assert pages.body["error"] is None
    assert [base64.b64decode(p["jpeg_base64"]) for p in pages.body["pages"]] == [
        b"PAGE-ONE",
        b"PAGE-TWO",
    ]

    finish = helper.call("POST", f"/finish?job={job}")
    assert finish.status == 200
    assert not folder.exists()
    assert scan_agent._jobs == {}


# ------------------------------------- #13: no endless checks for other sites
def test_requests_without_origin_cannot_rerun_the_scanner_check(helper: Helper) -> None:
    # new Image().src = "http://127.0.0.1:8765/devices?refresh=1&" + Math.random()
    for n in range(10):
        reply = helper.call(
            "GET", f"/devices?refresh=1&n={n}", origin=None, headers=CROSS_SITE_IMAGE
        )
        assert reply.status == 200

    # Only the first, when there was no answer at all yet, checked.
    assert helper.rig.checks.count == 1


def test_an_old_answer_is_not_checked_again_without_origin(helper: Helper) -> None:
    assert helper.call("GET", "/devices").status == 200
    assert helper.rig.checks.count == 1
    scanner_devices._devices_cache["at"] -= 3600  # an hour later

    reply = helper.call("GET", "/devices", origin=None, headers=CROSS_SITE_IMAGE)

    assert reply.status == 200
    assert reply.body["devices"][0]["id"] == DEVICE["id"]
    assert helper.rig.checks.count == 1


def test_the_page_can_still_ask_for_a_fresh_check(helper: Helper) -> None:
    helper.call("GET", "/devices", origin=None)
    assert helper.rig.checks.count == 1

    assert helper.call("GET", "/devices?refresh=1").status == 200
    assert helper.rig.checks.count == 2
    # A young answer still does for a plain Check.
    assert helper.call("GET", "/devices").status == 200
    assert helper.rig.checks.count == 2
    # Another site naming itself is turned away before any check.
    assert helper.call("GET", "/devices?refresh=1", origin=EVIL).status == 403
    assert helper.rig.checks.count == 2


def test_no_fresh_check_while_a_scan_is_running(helper: Helper) -> None:
    helper.call("GET", "/devices")
    _start(helper)

    assert helper.call("GET", "/devices?refresh=1").status == 200
    assert helper.rig.checks.count == 1


@pytest.mark.parametrize("agent", list(BY_HAND.values()), ids=list(BY_HAND))
def test_a_check_by_hand_still_gets_a_fresh_answer(helper: Helper, agent: str) -> None:
    # The README's check: curl "http://127.0.0.1:8765/devices?refresh=1"
    by_hand = {"User-Agent": agent, "Accept": "*/*"}
    assert helper.call("GET", "/devices").status == 200
    assert helper.rig.checks.count == 1

    reply = helper.call("GET", "/devices?refresh=1", origin=None, headers=by_hand)
    assert reply.status == 200
    assert helper.rig.checks.count == 2

    # And a plain one on an old answer checks again, as before #13.
    scanner_devices._devices_cache["at"] -= 3600
    assert helper.call("GET", "/devices", origin=None, headers=by_hand).status == 200
    assert helper.rig.checks.count == 3


@pytest.mark.parametrize(
    "headers",
    [
        # A browser from before Sec-Fetch-* (Safari 15 on an older Mac).
        {"User-Agent": BROWSER},
        # A browser whose User-Agent an extension changed still says where
        # the request came from.
        {**CROSS_SITE_IMAGE, "User-Agent": BY_HAND["curl"]},
        # The address typed into the browser.
        {
            "User-Agent": BROWSER,
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
        },
    ],
    ids=["browser-without-sec-fetch", "changed-user-agent", "address-bar"],
)
def test_a_browser_never_passes_for_a_check_by_hand(
    helper: Helper, headers: dict[str, str]
) -> None:
    assert helper.call("GET", "/devices").status == 200
    scanner_devices._devices_cache["at"] -= 3600  # an hour later

    for n in range(5):
        reply = helper.call(
            "GET", f"/devices?refresh=1&n={n}", origin=None, headers=headers
        )
        assert reply.status == 200

    assert helper.rig.checks.count == 1


# ------------------------------------------- #14: no pages left in %TEMP%
def _job_folder(base: Path, name: str, files: dict[str, bytes]) -> Path:
    folder = base / name
    folder.mkdir()
    for relative, data in files.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return folder


def test_sweep_removes_job_folders_an_earlier_run_left(tmp_path: Path) -> None:
    left = _job_folder(
        tmp_path,
        "scan-agent-k3x_9q2a",
        {
            "scan.log": b"scanning",
            "page-001.jpg": b"CONSENT-FORM",
            "page-002.jpg.part": b"HALF",
            "failed.txt": b"Paper jam",
            ".staging-ab12cd34/0003.jpg": b"STAGED",
        },
    )
    # The helper died between making the folder and starting the engine.
    empty = _job_folder(tmp_path, "scan-agent-00000000", {})

    assert scan_agent._sweep_old_jobs(tmp_path) == 2
    assert not left.exists()
    assert not empty.exists()


def test_sweep_leaves_anything_that_is_not_a_job_folder(tmp_path: Path) -> None:
    kept = [
        # Not a name mkdtemp gives a job.
        _job_folder(tmp_path, "scan-agent-backup", {"scan.log": b""}),
        _job_folder(tmp_path, "wiascan-0123456789abcdef", {"page-001.jpg": b"x"}),
        # The right name, but something the helper never writes is inside.
        _job_folder(
            tmp_path, "scan-agent-zz99yy88", {"page-001.jpg": b"x", "notes.docx": b"x"}
        ),
        _job_folder(tmp_path, "scan-agent-yy88xx77", {"other/page-001.jpg": b"x"}),
    ]
    not_a_folder = tmp_path / "scan-agent-abcdefgh"
    not_a_folder.write_bytes(b"a file")

    assert scan_agent._sweep_old_jobs(tmp_path) == 0
    assert all(path.exists() for path in [*kept, not_a_folder])


def test_sweep_never_follows_a_link(tmp_path: Path) -> None:
    elsewhere = _job_folder(tmp_path, "elsewhere", {"page-001.jpg": b"KEEP"})
    temp = tmp_path / "temp"
    temp.mkdir()
    try:
        (temp / "scan-agent-l1nk0000").symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("this account cannot make symbolic links")

    assert scan_agent._sweep_old_jobs(temp) == 0
    assert (elsewhere / "page-001.jpg").read_bytes() == b"KEEP"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="folder owners are POSIX-only")
def test_sweep_leaves_another_users_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    theirs = _job_folder(tmp_path, "scan-agent-0th3rus3", {"page-001.jpg": b"x"})
    me = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: me + 1)

    assert scan_agent._sweep_old_jobs(tmp_path) == 0
    assert theirs.exists()


@pytest.mark.parametrize("litter", [".DS_Store", "Thumbs.db", "desktop.ini"])
def test_sweep_removes_a_job_folder_someone_opened(tmp_path: Path, litter: str) -> None:
    # Opening the folder in Finder or Explorer to look at the pages leaves
    # a file of the system's own behind.
    opened = _job_folder(
        tmp_path, "scan-agent-0p3n3d00", {"page-001.jpg": b"CONSENT-FORM", litter: b""}
    )

    assert scan_agent._sweep_old_jobs(tmp_path) == 1
    assert not opened.exists()


def test_sweep_still_leaves_a_folder_named_like_the_systems_files(
    tmp_path: Path,
) -> None:
    kept = _job_folder(
        tmp_path, "scan-agent-d1rs0000", {"page-001.jpg": b"x", "desktop.ini/a.txt": b"x"}
    )

    assert scan_agent._sweep_old_jobs(tmp_path) == 0
    assert kept.exists()


def _flatpak_staging(apps: Path, name: str, files: dict[str, bytes]) -> Path:
    """A folder where scan_naps2.py stages a Flatpak NAPS2's pages."""
    base = apps / "com.naps2.Naps2" / "cache" / "akanksha-scan"
    base.mkdir(parents=True, exist_ok=True)
    return _job_folder(base, name, files)


def test_sweep_removes_pages_a_flatpak_naps2_staged(tmp_path: Path) -> None:
    # A Flatpak NAPS2 has a /tmp of its own, so its pages wait in the app's
    # folder in the home folder, outside the job folder, until handed over.
    staged = _flatpak_staging(
        tmp_path, "staging-q7w8e9r0", {"0001.jpg": b"CONSENT-FORM", "0002.jpg": b"MORE"}
    )
    empty = _flatpak_staging(tmp_path, "staging-00000000", {})

    assert scan_agent._sweep_flatpak_staging(tmp_path) == 2
    assert not staged.exists()
    assert not empty.exists()
    # The folder scan_naps2.py makes them in is left for the next scan.
    assert staged.parent.is_dir()


def test_sweep_leaves_anything_in_flatpak_folders_it_did_not_stage(
    tmp_path: Path,
) -> None:
    kept = [
        # Not a name mkdtemp gives a staging folder.
        _flatpak_staging(tmp_path, "staging-backup", {"0001.jpg": b"x"}),
        _flatpak_staging(tmp_path, "pages-q7w8e9r0", {"0001.jpg": b"x"}),
        # NAPS2 writes only pages straight into it, never a folder.
        _flatpak_staging(tmp_path, "staging-n3st3d00", {"keep/0001.jpg": b"x"}),
    ]
    # Another app's own cache rather than the akanksha-scan folder.
    other_app = tmp_path / "org.other.App" / "cache"
    other_app.mkdir(parents=True)
    kept.append(_job_folder(other_app, "staging-q7w8e9r0", {"0001.jpg": b"x"}))
    elsewhere = _job_folder(tmp_path, "elsewhere", {"0001.jpg": b"KEEP"})
    link = kept[0].parent / "staging-l1nk0000"
    try:
        link.symlink_to(elsewhere, target_is_directory=True)
        linked = True
    except OSError:
        linked = False  # this account cannot make symbolic links

    assert scan_agent._sweep_flatpak_staging(tmp_path) == 0
    assert all(path.exists() for path in kept)
    assert (elsewhere / "0001.jpg").read_bytes() == b"KEEP"
    assert link.is_symlink() == linked


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="folder owners are POSIX-only")
def test_sweep_leaves_another_users_flatpak_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    theirs = _flatpak_staging(tmp_path, "staging-0th3rus3", {"0001.jpg": b"x"})
    me = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: me + 1)

    assert scan_agent._sweep_flatpak_staging(tmp_path) == 0
    assert theirs.exists()


class _ServerStoppedAtOnce:
    """Binds nothing; serve_forever ends as Ctrl+C would."""

    def __init__(self, address: tuple[str, int], handler: type) -> None:
        self.address = address

    def serve_forever(self) -> None:
        raise KeyboardInterrupt


class _PortTaken:
    def __init__(self, address: tuple[str, int], handler: type) -> None:
        raise OSError(errno.EADDRINUSE, "Address already in use")


@pytest.fixture
def quiet_main(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """main() with no signals, threads or console handler of its own."""
    calls: list[str] = []
    monkeypatch.setattr(sys, "argv", ["scan_agent.py"])
    monkeypatch.delenv("SCAN_AGENT_ORIGINS", raising=False)
    monkeypatch.setattr(scan_agent, "_exit_cleanly_on_kill", lambda: None)
    monkeypatch.setattr(scan_agent, "_watch_jobs", lambda: None)
    monkeypatch.setattr(scan_agent, "_describe_devices", lambda: None)

    def console_handler() -> bool:
        calls.append("console handler")
        return True

    monkeypatch.setattr(scan_agent, "_clean_up_on_console_close", console_handler)
    return calls


def test_start_up_sweeps_what_an_earlier_run_left(
    rig: Rig,
    quiet_main: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left = _job_folder(
        rig.temp, "scan-agent-a1b2c3d4", {"page-001.jpg": b"CONSENT-FORM"}
    )
    monkeypatch.setattr(scan_agent, "Server", _ServerStoppedAtOnce)

    with pytest.raises(KeyboardInterrupt):
        scan_agent.main()

    assert not left.exists()
    assert (
        "cleared       1 scan folder an earlier run left behind"
        in capsys.readouterr().out
    )
    assert quiet_main == ["console handler"]


def test_start_up_sweeps_what_a_flatpak_naps2_staged(
    rig: Rig,
    quiet_main: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A power cut mid-scan: neither the helper nor scan_naps2.py cleared up.
    left = _job_folder(rig.temp, "scan-agent-a1b2c3d4", {"scan.log": b"scanning"})
    staged = _flatpak_staging(
        rig.home / ".var" / "app", "staging-q7w8e9r0", {"0001.jpg": b"CONSENT-FORM"}
    )
    monkeypatch.setattr(scan_agent, "Server", _ServerStoppedAtOnce)

    with pytest.raises(KeyboardInterrupt):
        scan_agent.main()

    assert not left.exists()
    assert not staged.exists()
    assert (
        "cleared       2 scan folders an earlier run left behind"
        in capsys.readouterr().out
    )


def test_a_second_copy_leaves_the_running_copys_folders_alone(
    rig: Rig, quiet_main: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _job_folder(rig.temp, "scan-agent-a1b2c3d4", {"page-001.jpg": b"MID-SCAN"})
    staging = _flatpak_staging(
        rig.home / ".var" / "app", "staging-q7w8e9r0", {"0001.jpg": b"MID-SCAN"}
    )
    monkeypatch.setattr(scan_agent, "Server", _PortTaken)

    with pytest.raises(SystemExit):
        scan_agent.main()

    assert live.exists()
    assert staging.exists()


class _FakeKernel32:
    def __init__(self, result: int = 1) -> None:
        self.result = result
        self.calls: list[tuple[Callable[[int], bool], bool]] = []

    def SetConsoleCtrlHandler(  # the Windows API's own name
        self, handler: Callable[[int], bool], add: bool
    ) -> int:
        self.calls.append((handler, add))
        return self.result


@pytest.fixture
def windows_ctypes(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], _FakeKernel32]:
    """Swap in a ctypes that looks like Windows' for the helper to import."""

    def install(result: int = 1) -> _FakeKernel32:
        kernel32 = _FakeKernel32(result)

        def winfunctype(restype: Any, *argtypes: Any) -> Callable[[Any], Any]:
            assert (restype, argtypes) == (ctypes.c_long, (ctypes.c_ulong,))
            return lambda function: function  # the "C callback" is the function

        fake = types.SimpleNamespace(
            windll=types.SimpleNamespace(kernel32=kernel32),
            WINFUNCTYPE=winfunctype,
            c_long=ctypes.c_long,
            c_ulong=ctypes.c_ulong,
        )
        monkeypatch.setitem(sys.modules, "ctypes", fake)
        monkeypatch.setattr(scan_agent, "_console_handler", None)
        return kernel32

    return install


def test_console_handler_is_registered_and_kept(
    windows_ctypes: Callable[[int], _FakeKernel32],
) -> None:
    kernel32 = windows_ctypes(1)

    assert scan_agent._clean_up_on_console_close() is True

    [(handler, add)] = kernel32.calls
    assert add is True
    # Held by the module, so ctypes never frees what Windows will call.
    assert scan_agent._console_handler is handler


def test_console_handler_failing_to_register_is_not_fatal(
    windows_ctypes: Callable[[int], _FakeKernel32],
) -> None:
    windows_ctypes(0)

    assert scan_agent._clean_up_on_console_close() is False
    assert scan_agent._console_handler is None


@pytest.mark.skipif(
    sys.platform == "win32", reason="would register with the real console"
)
def test_console_handler_is_windows_only() -> None:
    assert scan_agent._clean_up_on_console_close() is False


@pytest.mark.parametrize(
    "event",
    [
        scan_agent.CTRL_CLOSE_EVENT,
        scan_agent.CTRL_BREAK_EVENT,
        scan_agent.CTRL_LOGOFF_EVENT,
        scan_agent.CTRL_SHUTDOWN_EVENT,
    ],
)
def test_closing_the_window_clears_away_held_jobs(
    rig: Rig, windows_ctypes: Callable[[int], _FakeKernel32], event: int
) -> None:
    kernel32 = windows_ctypes(1)
    assert scan_agent._clean_up_on_console_close() is True
    [(handler, _add)] = kernel32.calls
    job = scan_agent._start_scan(DEVICE, OPTIONS)
    (job["folder"] / "page-001.jpg").write_bytes(b"CONSENT-FORM")

    # Ctrl+C is Python's: it becomes KeyboardInterrupt and main() cleans up.
    assert handler(scan_agent.CTRL_C_EVENT) is False
    assert job["folder"].exists()
    assert scan_agent._jobs

    # The event is passed on (False) so Windows still ends the process,
    # but only once the scan is stopped and its pages are gone.
    assert handler(event) is False
    assert scan_agent._jobs == {}
    assert not job["folder"].exists()
    assert rig.engines.started[0].stopped
