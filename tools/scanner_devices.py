"""Which scanners this computer can drive - and which it cannot, yet.

scan_agent.py asks this module whenever the page presses Check. It answers
two questions:

  ready     scanners a scan engine on this computer can drive right now,
            each tagged with the driver that sees it and the engine that will
            run the scan;
  unready   scanners plugged in by USB that no driver here can see - nearly
            always because the maker's driver is not installed - with what
            to install.

The two engines:

  scan.ps1  Windows only. Talks to WIA directly and hands pages over one at a
            time while the stack is still feeding, so it is preferred whenever
            Windows has a WIA driver for the scanner. Nothing to install.
  NAPS2     every OS (https://www.naps2.com/download). Reaches what scan.ps1
            cannot: TWAIN on Windows, Apple's Image Capture drivers on a Mac,
            SANE on Linux, and network (eSCL / AirScan) scanners everywhere.

Everything that touches the machine is a thin wrapper round a pure parser or
matcher, so those can be tested without a scanner attached.

    python3 tools/scanner_devices.py        (prints what GET /devices returns)
"""

from __future__ import annotations

import base64
import json
import os
import plistlib
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

NAPS2_DOWNLOAD = "https://www.naps2.com/download"
NAPS2_FLATPAK = "com.naps2.Naps2"

# One stuck driver must not hold up Check for every other scanner, so each
# listing is stopped after this long and reported.
LIST_CAP_SECONDS = 25
# NAPS2's eSCL listing prints each network scanner as Bonjour finds it and
# then never exits on its own (measured on a Mac: the printer at 1s, then
# nothing for as long as it was left). Network scanners answer within a
# second or two, so stop once it has run this long and gone quiet.
ESCL_MIN_SECONDS = 6
ESCL_QUIET_SECONDS = 3
# Listing takes several seconds, so the page's Check reuses a recent answer.
CACHE_SECONDS = 60

# The drivers asked on each OS, best first. When one scanner shows up under
# two of them the earlier is kept: on Windows WIA streams page by page
# through scan.ps1, and on a Mac Apple's driver is the one makers support.
DRIVERS = {
    "win32": ("wia", "twain", "escl"),
    "darwin": ("apple", "sane", "escl"),
    "linux": ("sane", "escl"),
}
ENGINES = {
    "wia": "scan.ps1",
    "twain": "naps2",
    "apple": "naps2",
    "sane": "naps2",
    "escl": "naps2",
}
OS_NAMES = {"win32": "Windows", "darwin": "macOS", "linux": "Linux"}
# The listings that would find a USB scanner whose driver is installed. When
# one of them fails, a USB scanner missing from the list says nothing about
# its driver, so it is not reported as needing one.
USB_DRIVERS = {
    "win32": ("wia", "twain"),
    "darwin": ("apple",),
    "linux": ("sane",),
}

# USB makers whose devices are worth a look when no driver claims them.
# "printers" marks makers that also sell plain printers: from those only a
# device whose name says scanner or multifunction is reported, so a printer
# on the same desk is not listed as a scanner without a driver.
VENDORS = {
    0x04C5: {
        "maker": "Fujitsu / PFU / Ricoh",
        "url": "https://www.pfu.ricoh.com/global/scanners/downloads/",
        "printers": False,
    },
    0x04A9: {"maker": "Canon", "url": "https://global.canon/en/support/", "printers": True},
    0x04B8: {"maker": "Epson", "url": "https://epson.com/support", "printers": True},
    0x03F0: {"maker": "HP", "url": "https://support.hp.com/drivers", "printers": True},
    0x04F9: {"maker": "Brother", "url": "https://support.brother.com/", "printers": True},
    0x07B3: {"maker": "Plustek", "url": "https://plustek.com/", "printers": False},
    0x0638: {"maker": "Avision", "url": "https://www.avision.com/", "printers": False},
    0x1083: {
        "maker": "Canon Electronics",
        "url": "https://global.canon/en/support/",
        "printers": False,
    },
    # Kodak's vendor id also covers its cameras and printers.
    0x040A: {"maker": "Kodak", "url": "https://www.alarisworld.com/", "printers": True},
}

# Names that mark a scanner or multifunction device among printer makers'
# products: ScanJet / CanoScan / ScanSnap, "MFP", Brother MFC and DCP, Canon
# LiDE and imageFORMULA, Epson Perfection, HP Envy and Smart Tank, Kodak
# Alaris i1150 / S2050 / E1025, and the model prefixes those makers give
# their scanners and all-in-ones. A line that also has print-only models -
# Canon TS and TR, Epson WorkForce, EcoTank and XP, HP OfficeJet - is left
# out: missing a driver hint is better than calling a printer a scanner.
_SCANNERISH = re.compile(
    r"scan|mfp|mfc|dcp|all.?in.?one|multi.?function|\baio\b|lide|perfection"
    r"|imageformula|envy|smart ?tank"
    r"|\b(?:ads|pds|ds|es|dr|mf|mg|mx)-?\d|\b[ise]\d{4}\b",
    re.IGNORECASE,
)

# Words that say nothing about which scanner it is - the maker, the software
# layer, the bus. Dropped before names are compared, so "FUJITSU SP-1130N",
# "PaperStream IP SP-1130N" and "SP-1130N" all count as the same scanner.
_NOISE_WORDS = frozenset(
    """
    fujitsu pfu ricoh hp hewlett packard canon epson seiko brother plustek
    avision kodak alaris inc ltd co corp corporation limited
    paperstream ip twain wia escl airscan driver usb series
    """.split()
)
# "(192.168.0.24)" from NAPS2's eSCL listing, "[C8B242]" from Bonjour: the
# address, not the model, so they are left out of the comparison.
_BRACKETED = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_JOINERS = re.compile(r"[-_.]")
_LEADING_JOINED = re.compile(r"\s*([a-z]+)[-_.]+")
_SPLIT = re.compile(r"[^0-9a-z]+")
# Windows names the parts of a composite USB device generically; those say
# nothing about what is plugged in.
_GENERIC_USB_NAME = re.compile(
    r"^(?:usb\b|generic\b|unknown\b)|composite|root hub|mass storage"
    r"|input device|printing support",
    re.IGNORECASE,
)
# NAPS2 reports problems on stdout, among the device names, and still exits
# 0 - 'The "Twain" driver is not supported on this platform.' - so a line
# that reads like a sentence is a message, not a scanner.
_NAPS2_PROBLEM = re.compile(
    r"\b(?:error|exception|not supported|not valid|could not|cannot|unable)\b",
    re.IGNORECASE,
)
_NAPS2_BANNER = re.compile(r"^(?:NAPS2 \d|Copyright\b)")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def platform_key(platform: str | None = None) -> str:
    platform = platform or sys.platform
    if platform == "win32":
        return "win32"
    if platform == "darwin":
        return "darwin"
    # NAPS2 ships for Linux only, but any other Unix gets the same treatment.
    return "linux"


def timeout_message(seconds: float) -> str:
    """Shared by scan_agent.py and scan_naps2.py, so whichever stops a stuck
    scan first, the person reads the same thing."""
    minutes = seconds / 60
    unit = "minute" if minutes == 1 else "minutes"
    return (
        f"The scan took longer than {minutes:g} {unit} and was stopped. "
        "Check the scanner for a jam."
    )


# ---- processes ---------------------------------------------------------


def hidden_window_flags() -> int:
    """Keep a console window from flashing up for a background listing."""
    return _NO_WINDOW if os.name == "nt" else 0


def kill_process_tree(process: subprocess.Popen, own_group: bool = False) -> None:
    """Stop a process and everything it started.

    On Windows taskkill /T walks the tree - NAPS2 and PowerShell can both
    start helpers of their own. Elsewhere a process started with
    start_new_session=True leads its own group, and the group is what gets
    the signal; anything else is killed on its own.
    """
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                creationflags=_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    elif own_group:
        # TERM first so a scanner driver can let go of the device cleanly,
        # then KILL whatever is still there.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    else:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_lines(
    argv: list[str],
    cap: float,
    settle: tuple[float, float] | None = None,
) -> tuple[list[str], list[str], int | None, bool]:
    """Run a listing command and collect its output lines.

    Stops it after `cap` seconds, or - with settle=(min, quiet) - once it has
    run `min` seconds and printed nothing for `quiet` seconds, for a command
    that never finishes by itself. Returns (stdout lines, stderr lines, exit
    code or None if it was stopped, whether the cap was hit).
    """
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        creationflags=hidden_window_flags(),
    )
    lines: queue.Queue = queue.Queue()

    def pump(stream, tag: str) -> None:
        for raw in iter(stream.readline, b""):
            lines.put((tag, raw.decode("utf-8", errors="replace").rstrip("\r\n")))
        lines.put((tag, None))

    for stream, tag in ((process.stdout, "out"), (process.stderr, "err")):
        threading.Thread(target=pump, args=(stream, tag), daemon=True).start()

    out: list[str] = []
    err: list[str] = []
    open_streams = 2
    started = last_line = time.monotonic()
    capped = False
    while open_streams:
        now = time.monotonic()
        if now - started >= cap:
            capped = True
            break
        if settle and now - started >= settle[0] and now - last_line >= settle[1]:
            break
        try:
            tag, line = lines.get(timeout=0.2)
        except queue.Empty:
            continue
        if line is None:
            open_streams -= 1
            continue
        (out if tag == "out" else err).append(line)
        if tag == "out" and line.strip():
            last_line = time.monotonic()

    if open_streams:
        kill_process_tree(process)
        return out, err, None, capped
    try:
        return out, err, process.wait(timeout=5), False
    except subprocess.TimeoutExpired:
        kill_process_tree(process)
        return out, err, None, False


def run_powershell(script: str, timeout: float) -> subprocess.CompletedProcess:
    """Run a PowerShell snippet. -EncodedCommand sidesteps every quoting rule
    between Python, the Windows command line and PowerShell."""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ],
        capture_output=True,
        timeout=timeout,
        creationflags=_NO_WINDOW,
    )


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ---- NAPS2 -------------------------------------------------------------


def is_flatpak(naps2: str) -> bool:
    words = naps2.split()
    return bool(words) and Path(words[0]).name == "flatpak" and not os.path.isfile(naps2)


def flatpak_app_id(naps2: str) -> str | None:
    """The app id in "flatpak run [options] com.naps2.Naps2", or None."""
    if not is_flatpak(naps2):
        return None
    words = shlex.split(naps2)
    if "run" not in words:
        return None
    for word in words[words.index("run") + 1 :]:
        if not word.startswith("-"):
            return word
    return None


def naps2_command(naps2: str) -> list[str]:
    """The argv that starts NAPS2's command line.

    `naps2` is what find_naps2() returns: a path to the program, or a flatpak
    command as one string. Windows ships a separate NAPS2.Console.exe; the
    Mac and Linux builds take a leading "console" instead.
    """
    argv = shlex.split(naps2) if is_flatpak(naps2) else [naps2]
    # Split on both kinds of slash, so this reads a Windows path the same
    # whichever OS runs it.
    program = re.split(r"[\\/]", argv[-1])[-1].lower()
    if program in ("naps2.console.exe", "naps2.console"):
        return argv
    if argv[-1] == "console":
        return argv
    return argv + ["console"]


def _resolve_naps2_override(value: str) -> str | None:
    value = value.strip().strip('"')
    if not value:
        return None
    if is_flatpak(value):
        return value
    path = Path(value).expanduser()
    if path.is_dir():
        # A folder is fine too: the install folder on Windows, NAPS2.app on
        # a Mac.
        for inner in ("NAPS2.Console.exe", "Contents/MacOS/NAPS2", "naps2"):
            if (path / inner).is_file():
                return str(path / inner)
        return None
    if path.is_file():
        console = path.parent / "NAPS2.Console.exe"
        if path.name.lower() == "naps2.exe" and console.is_file():
            return str(console)
        return str(path)
    return shutil.which(value)


def _naps2_candidates(key: str) -> list[str]:
    if key == "win32":
        roots = [os.environ.get(name) for name in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)")]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.append(os.path.join(local, "Programs"))
        found = [os.path.join(root, "NAPS2", "NAPS2.Console.exe") for root in roots if root]
        found.append(shutil.which("NAPS2.Console.exe") or "")
        return found
    if key == "darwin":
        # Per-user installs land in ~/Applications - the machine this was
        # built on has NAPS2 there and not in /Applications.
        return [
            "/Applications/NAPS2.app/Contents/MacOS/NAPS2",
            str(Path.home() / "Applications/NAPS2.app/Contents/MacOS/NAPS2"),
            shutil.which("naps2") or "",
        ]
    return [shutil.which("naps2") or "", "/usr/bin/naps2", "/usr/local/bin/naps2"]


def _flatpak_naps2() -> str | None:
    if not shutil.which("flatpak"):
        return None
    for base in (Path("/var/lib/flatpak/app"), Path.home() / ".local/share/flatpak/app"):
        if (base / NAPS2_FLATPAK).is_dir():
            return f"flatpak run {NAPS2_FLATPAK}"
    try:
        info = subprocess.run(
            ["flatpak", "info", NAPS2_FLATPAK],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return f"flatpak run {NAPS2_FLATPAK}" if info.returncode == 0 else None


_naps2_cache: dict = {"path": None, "at": None}
_naps2_lock = threading.Lock()


def find_naps2(refresh: bool = False, platform: str | None = None) -> str | None:
    """Where NAPS2 is on this computer, or None.

    NAPS2_PATH wins when it points at something real; then the usual install
    places; then PATH, and on Linux the Flatpak. Cached for a minute -
    /health asks on every page load and must answer at once.
    """
    with _naps2_lock:
        at = _naps2_cache["at"]
        if not refresh and at is not None and time.monotonic() - at < CACHE_SECONDS:
            return _naps2_cache["path"]
        found = None
        override = os.environ.get("NAPS2_PATH", "")
        if override:
            found = _resolve_naps2_override(override)
        key = platform_key(platform)
        if not found:
            found = next((p for p in _naps2_candidates(key) if p and os.path.isfile(p)), None)
        if not found and key == "linux":
            found = _flatpak_naps2()
        _naps2_cache["path"], _naps2_cache["at"] = found, time.monotonic()
        return found


def install_hint(platform: str | None = None) -> str:
    key = platform_key(platform)
    if key == "win32":
        return (
            f"Install NAPS2 from {NAPS2_DOWNLOAD} to add TWAIN and network (eSCL) "
            "scanners - WIA scanners work without it."
        )
    if key == "darwin":
        return (
            f"Install NAPS2 from {NAPS2_DOWNLOAD} (or run: brew install --cask naps2), "
            "then press Check again."
        )
    return (
        f"Install NAPS2 from {NAPS2_DOWNLOAD} (the .deb, .rpm or Flatpak), "
        "then press Check again."
    )


def parse_naps2_listing(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split NAPS2 --listdevices output into (device names, messages)."""
    names: list[str] = []
    messages: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or _NAPS2_BANNER.match(line):
            continue
        sentence = line.endswith((".", "!")) and len(line.split()) >= 4
        if sentence or _NAPS2_PROBLEM.search(line):
            messages.append(line)
        elif line not in names:
            names.append(line)
    return names, messages


def list_naps2_devices(
    naps2: str, driver: str, cap: float = LIST_CAP_SECONDS
) -> tuple[list[str], str | None]:
    """Device names NAPS2 sees through one driver, and a problem if any."""
    argv = naps2_command(naps2) + ["--listdevices", "--driver", driver]
    settle = (ESCL_MIN_SECONDS, ESCL_QUIET_SECONDS) if driver == "escl" else None
    try:
        out, err, code, capped = run_lines(argv, cap, settle)
    except OSError as exc:
        return [], f"NAPS2 could not be started to list {driver} scanners: {exc}"
    names, messages = parse_naps2_listing(out)
    if messages:
        return names, f"NAPS2 ({driver} driver): {' '.join(messages)}"
    if capped and driver != "escl":
        return names, (
            f"The {driver} driver did not finish listing scanners within "
            f"{cap:g} seconds, so it was stopped."
        )
    if code not in (0, None) and not names:
        detail = _last_line("\n".join(err))
        return names, (
            f"NAPS2 stopped with exit code {code} while listing {driver} scanners"
            + (f": {detail}" if detail else ".")
        )
    return names, None


# ---- Windows WIA -------------------------------------------------------

# Every WIA scanner Windows knows, as JSON. Type 1 is a scanner (2 is a
# camera). Non-ASCII is escaped so a name with an accent survives whatever
# code page the console happens to use.
WIA_LIST_PS = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try {
    $found = @()
    $manager = New-Object -ComObject WIA.DeviceManager
    foreach ($info in $manager.DeviceInfos) {
        if ($info.Type -ne 1) { continue }
        $props = @{}
        foreach ($p in $info.Properties) {
            $key = [string]$p.Name
            try { $props[$key] = [string]$p.Value } catch {}
        }
        $found += New-Object PSObject -Property @{
            name = $props['Name']
            description = $props['Description']
            port = $props['Port']
            id = [string]$info.DeviceID
        }
    }
    $json = ConvertTo-Json -InputObject @($found) -Compress
    $json = [regex]::Replace($json, '[^\x00-\x7F]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
    Write-Output $json
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
"""


def _json_list(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    return data if isinstance(data, list) else []


def parse_wia_json(text: str) -> list[dict]:
    """[{"name": ..., "usb": bool, "wia_id": ...}] from WIA_LIST_PS output."""
    out = []
    for item in _json_list(text):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("description") or "").strip()
        if not name:
            continue
        # A USB scanner's WIA port is \\.\Usbscan0 and the like.
        port = str(item.get("port") or "").lower()
        out.append(
            {
                "name": name,
                "usb": "usbscan" in port,
                # Unique where the name may not be: two of the same model
                # side by side share a name but never a DeviceID.
                "wia_id": str(item.get("id") or "").strip(),
            }
        )
    return out


def list_wia_devices() -> tuple[list[dict], str | None]:
    try:
        done = run_powershell(WIA_LIST_PS, timeout=LIST_CAP_SECONDS)
    except subprocess.TimeoutExpired:
        return [], f"Windows did not list its WIA scanners within {LIST_CAP_SECONDS} seconds."
    except OSError as exc:
        return [], f"PowerShell could not be started to list WIA scanners: {exc}"
    stdout = done.stdout.decode("utf-8-sig", errors="replace")
    if done.returncode != 0:
        detail = _last_line(done.stderr.decode("utf-8", errors="replace"))
        return [], "Windows could not list its WIA scanners" + (f": {detail}" if detail else ".")
    try:
        return parse_wia_json(stdout), None
    except ValueError:
        return [], "Windows listed its WIA scanners in a form this helper could not read."


# ---- USB ---------------------------------------------------------------


def _usb_entry(
    vendor_id, product_id, name, maker="", flagged=True, has_driver=False
) -> dict | None:
    try:
        vendor = int(vendor_id)
    except (TypeError, ValueError):
        return None
    try:
        product = int(product_id) if product_id is not None else None
    except (TypeError, ValueError):
        product = None
    return {
        "vendor_id": vendor,
        "product_id": product,
        "name": str(name or "").strip(),
        "maker": str(maker or "").strip(),
        # Only Windows can say whether a driver has claimed the device; on
        # the others every device is a candidate until a ready scanner
        # matches it by name.
        "flagged": bool(flagged),
        # Windows only: some driver has claimed it as a scanner, so a driver
        # is there even if it is not one WIA can list.
        "has_driver": bool(has_driver),
    }


def parse_ioreg_plist(data: bytes) -> list[dict]:
    """USB devices from `ioreg -p IOUSB -l -w0 -a` (the plist form)."""
    root = plistlib.loads(data)
    out = []
    stack = [root] if isinstance(root, dict) else list(root or [])
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if "idVendor" in node:
            entry = _usb_entry(
                node.get("idVendor"),
                node.get("idProduct"),
                node.get("USB Product Name") or node.get("kUSBProductString"),
                node.get("USB Vendor Name") or node.get("kUSBVendorString"),
            )
            if entry:
                out.append(entry)
        stack.extend(reversed(node.get("IORegistryEntryChildren") or []))
    return out


def parse_ioreg_text(text: str) -> list[dict]:
    """USB devices from plain `ioreg -p IOUSB -l -w0`, should -a ever fail."""
    nodes: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        if "+-o " in line:
            current = {}
            nodes.append(current)
            continue
        found = re.search(r'"([^"]+)" = (.*)$', line)
        if found and current is not None:
            value = found.group(2).strip()
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                value = value[1:-1]
            current[found.group(1)] = value
    out = []
    for node in nodes:
        if "idVendor" not in node:
            continue
        try:
            vendor = int(node["idVendor"], 0)
            product = int(node["idProduct"], 0) if "idProduct" in node else None
        except ValueError:
            continue
        entry = _usb_entry(
            vendor,
            product,
            node.get("USB Product Name") or node.get("kUSBProductString"),
            node.get("USB Vendor Name") or node.get("kUSBVendorString"),
        )
        if entry:
            out.append(entry)
    return out


def usb_devices_macos() -> list[dict]:
    # system_profiler SPUSBDataType returns nothing on recent macOS, so ask
    # the I/O Registry, which does list a scanner on the bus.
    try:
        done = subprocess.run(
            ["ioreg", "-p", "IOUSB", "-l", "-w0", "-a"], capture_output=True, timeout=15
        )
        if done.returncode == 0 and done.stdout.strip():
            return parse_ioreg_plist(done.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    done = subprocess.run(
        ["ioreg", "-p", "IOUSB", "-l", "-w0"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
    )
    return parse_ioreg_text(done.stdout)


# Present USB devices with their class and driver state, as JSON. A device
# with no driver shows as Status Error with a problem code (28 = no driver
# installed); one with a scanner driver shows as class Image.
PNP_USB_PS = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try {
    $found = @()
    foreach ($d in @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue)) {
        $instance = [string]$d.InstanceId
        if ($instance -notmatch '^USB\\VID_[0-9A-Fa-f]{4}&PID_[0-9A-Fa-f]{4}') { continue }
        $found += New-Object PSObject -Property @{
            name = [string]$d.FriendlyName
            instance = $instance
            class = [string]$d.Class
            status = [string]$d.Status
            problem = [int]$d.ConfigManagerErrorCode
        }
    }
    $json = ConvertTo-Json -InputObject @($found) -Compress
    $json = [regex]::Replace($json, '[^\x00-\x7F]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
    Write-Output $json
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
"""


def parse_pnp_json(text: str) -> list[dict]:
    """USB devices from PNP_USB_PS output, one per vendor:product.

    A multifunction device shows as several entries - the composite parent
    and one per interface (printer, scanner, storage) - so they are folded
    together: flagged if any part has a scanner class or a driver problem,
    named after the most telling part.
    """
    groups: dict[tuple[int, int], dict] = {}
    for item in _json_list(text):
        if not isinstance(item, dict):
            continue
        found = re.match(
            r"^USB\\VID_([0-9A-F]{4})&PID_([0-9A-F]{4})", str(item.get("instance") or ""), re.I
        )
        if not found:
            continue
        key = (int(found.group(1), 16), int(found.group(2), 16))
        group = groups.setdefault(key, {"names": [], "flagged": False, "has_driver": False})
        device_class = str(item.get("class") or "").strip().lower()
        status = str(item.get("status") or "").strip().upper()
        try:
            problem = int(item.get("problem") or 0)
        except (TypeError, ValueError):
            problem = 0
        if device_class in ("image", "scanner") or (status and status != "OK") or problem:
            group["flagged"] = True
        if device_class in ("image", "scanner") and status == "OK" and not problem:
            group["has_driver"] = True
        name = str(item.get("name") or "").strip()
        if name and not _GENERIC_USB_NAME.search(name) and name not in group["names"]:
            group["names"].append(name)
    out = []
    for (vendor, product), group in groups.items():
        names = group["names"]
        name = next((n for n in names if _SCANNERISH.search(n)), names[0] if names else "")
        entry = _usb_entry(
            vendor, product, name, flagged=group["flagged"], has_driver=group["has_driver"]
        )
        if entry:
            out.append(entry)
    return out


def usb_devices_windows() -> list[dict]:
    done = run_powershell(PNP_USB_PS, timeout=LIST_CAP_SECONDS)
    if done.returncode != 0:
        detail = _last_line(done.stderr.decode("utf-8", errors="replace"))
        raise OSError(detail or f"PowerShell exited with code {done.returncode}")
    return parse_pnp_json(done.stdout.decode("utf-8-sig", errors="replace"))


def read_sysfs_usb(root: Path = Path("/sys/bus/usb/devices")) -> list[dict]:
    """USB devices from Linux sysfs - the strings the device itself reports,
    which lsusb's name database may not know for a newer model."""

    def read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    out = []
    if not root.is_dir():
        return out
    for device in sorted(root.iterdir()):
        vendor = read(device / "idVendor")
        if not vendor:
            continue  # an interface, not a device
        product = read(device / "idProduct")
        try:
            entry = _usb_entry(
                int(vendor, 16),
                int(product, 16) if product else None,
                read(device / "product"),
                read(device / "manufacturer"),
            )
        except ValueError:
            continue
        if entry:
            out.append(entry)
    return out


def parse_lsusb(text: str) -> list[dict]:
    """USB devices from `lsusb`: "Bus 001 Device 004: ID 04c5:1627 Fujitsu, Ltd SP-1130N"."""
    out = []
    for line in text.splitlines():
        found = re.search(r"\bID ([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\s*(.*)$", line)
        if not found:
            continue
        entry = _usb_entry(int(found.group(1), 16), int(found.group(2), 16), found.group(3))
        if entry:
            out.append(entry)
    return out


def usb_devices_linux() -> list[dict]:
    devices = read_sysfs_usb()
    if devices:
        return devices
    done = subprocess.run(
        ["lsusb"], capture_output=True, text=True, errors="replace", timeout=15
    )
    return parse_lsusb(done.stdout)


def list_usb_devices(key: str) -> list[dict]:
    if key == "win32":
        return usb_devices_windows()
    if key == "darwin":
        return usb_devices_macos()
    return usb_devices_linux()


# ---- matching ----------------------------------------------------------


def name_tokens(name: str) -> frozenset[str]:
    """The words of a device name that identify the model.

    Hyphens and dots join rather than split, so "SP-1130N" stays one token
    ("sp1130n") and cannot be confused with an "SP-1130".
    """
    text = _BRACKETED.sub(" ", name.lower())
    # Windows offers every WIA scanner to TWAIN again as "WIA-<name>". A
    # leading word that says nothing is split off before the joiners run,
    # or "WIA-FUJITSU SP-1130N" would keep a "wiafujitsu" no other name has.
    while True:
        found = _LEADING_JOINED.match(text)
        if not found or found.group(1) not in _NOISE_WORDS:
            break
        text = text[found.end() :]
    text = _JOINERS.sub("", text)
    return frozenset(t for t in _SPLIT.split(text) if t and t not in _NOISE_WORDS)


def dedupe_key(name: str) -> str:
    tokens = name_tokens(name)
    return " ".join(sorted(tokens)) if tokens else name.strip().lower()


def _is_model_number(token: str) -> bool:
    return (
        len(token) >= 4
        and any(c.isdigit() for c in token)
        and any(c.isalpha() for c in token)
    )


def names_match(a: str, b: str) -> bool:
    """Whether two names plausibly belong to the same scanner.

    Drivers name one device differently ("SP-1130N" over USB, "FUJITSU
    SP-1130N" in a driver list), so one name's words being all inside the
    other's counts, and so does a shared model number like "sp1130n".
    """
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    return any(_is_model_number(t) for t in ta & tb)


def looks_like_scanner(name: str) -> bool:
    return bool(_SCANNERISH.search(name or ""))


def dedupe(found: list[dict], order: tuple[str, ...]) -> tuple[list[dict], set[str]]:
    """Keep one entry per scanner seen under several drivers.

    Two entries are the same scanner when their names normalise the same;
    the one under the earliest driver in `order` stays. Entries under one
    driver are never merged - two identical scanners side by side are two
    scanners. Also returns the names (as dedupe keys) eSCL saw, which mark a
    scanner as networked whichever driver ends up driving it.
    """
    rank = {driver: index for index, driver in enumerate(order)}
    best: dict[str, int] = {}
    network: set[str] = set()
    for device in found:
        key = dedupe_key(device["name"])
        position = rank.get(device["driver"], len(order))
        best[key] = min(best.get(key, position), position)
        if device["driver"] == "escl":
            network.add(key)
    kept = [
        device
        for device in found
        if rank.get(device["driver"], len(order)) == best[dedupe_key(device["name"])]
    ]
    kept.sort(key=lambda device: rank.get(device["driver"], len(order)))
    return kept, network


def classify_connection(device: dict, usb_names: list[str], network: set[str]) -> str:
    if device["driver"] == "escl":
        return "network"
    if device.get("usb_port") or any(names_match(device["name"], n) for n in usb_names):
        return "usb"
    if dedupe_key(device["name"]) in network:
        return "network"
    return "unknown"


def _driver_phrase(vendor_id: int, key: str) -> str:
    if vendor_id == 0x04C5:
        return {
            "win32": (
                "PaperStream IP for this model from Ricoh's scanner downloads page "
                "(it includes the WIA driver)"
            ),
            "darwin": "Ricoh's macOS driver (ICA) for this model from its scanner downloads page",
            "linux": "Ricoh's Linux driver for this model from its scanner downloads page",
        }[key]
    maker = VENDORS[vendor_id]["maker"]
    return f"{maker}'s {OS_NAMES[key]} driver for this scanner from {maker}'s support site"


def twain_only(has_driver: bool, key: str, naps2_installed: bool) -> bool:
    """A Windows scanner with a driver that WIA does not list - a TWAIN-only
    driver, which only NAPS2 can reach."""
    return key == "win32" and has_driver and not naps2_installed


def driver_fix(
    vendor_id: int, key: str, naps2_installed: bool, has_driver: bool = False
) -> str:
    if twain_only(has_driver, key, naps2_installed):
        return (
            "Windows has a driver for this scanner, but not a WIA one - install "
            f"NAPS2 from {NAPS2_DOWNLOAD} to scan through its TWAIN driver, then "
            "press Check again."
        )
    phrase = _driver_phrase(vendor_id, key)
    if key != "win32" and not naps2_installed:
        # Without NAPS2 nothing on a Mac or Linux can see any scanner, so
        # the maker's driver may well be fine - NAPS2 comes first.
        return (
            "Install NAPS2 first, and if the scanner still does not appear after "
            f"Check again, install {phrase}."
        )
    return f"Install {phrase}, then press Check again."


def find_unready(
    usb: list[dict], ready: list[dict], key: str, naps2_installed: bool
) -> list[dict]:
    """USB scanners from known makers that no ready scanner accounts for.

    Only meaningful when the listings that would have found them finished -
    check_devices holds these back when one did not.
    """
    out = []
    seen = set()
    for device in usb:
        vendor = VENDORS.get(device["vendor_id"])
        if vendor is None or not device.get("flagged", True):
            continue
        name = device.get("name") or ""
        if vendor["printers"] and not looks_like_scanner(name):
            continue
        if name and any(names_match(name, r["name"]) for r in ready):
            continue
        if not name:
            product = device.get("product_id")
            name = f"{vendor['maker']} USB device ({device['vendor_id']:04x}:" + (
                f"{product:04x})" if product is not None else "????)"
            )
        marker = (device["vendor_id"], device.get("product_id"), name)
        if marker in seen:
            continue
        seen.add(marker)
        has_driver = bool(device.get("has_driver"))
        out.append(
            {
                "name": name,
                "vendor": vendor["maker"],
                "connection": "usb",
                "status": "driver_missing",
                "fix": driver_fix(device["vendor_id"], key, naps2_installed, has_driver),
                "driver_url": (
                    NAPS2_DOWNLOAD if twain_only(has_driver, key, naps2_installed) else vendor["url"]
                ),
            }
        )
    return out


# ---- the whole check ---------------------------------------------------

# WIA's own DeviceID for each ready WIA scanner, by the id the page sees.
# Kept out of the device objects the page is sent; scan_agent.py hands it to
# scan.ps1 so two scanners with one name are told apart. Each check rewrites
# the entry for every id it lists, and only a listed id is ever looked up.
_wia_ids: dict[str, str] = {}


def wia_device_id(device_id: str) -> str | None:
    return _wia_ids.get(device_id)


def check_devices(platform: str | None = None) -> dict:
    """Ask every driver at once and put together the GET /devices answer."""
    key = platform_key(platform)
    naps2 = find_naps2(refresh=True, platform=platform)
    order = DRIVERS[key]
    errors: list[str] = []
    failed: set[str] = set()

    with ThreadPoolExecutor(max_workers=len(order) + 1) as pool:
        usb_job = pool.submit(list_usb_devices, key)
        listings = {}
        for driver in order:
            if driver == "wia":
                listings[driver] = pool.submit(list_wia_devices)
            elif naps2:
                listings[driver] = pool.submit(list_naps2_devices, naps2, driver)

        found: list[dict] = []
        for driver in order:
            if driver not in listings:
                continue
            try:
                entries, problem = listings[driver].result()
            except Exception as exc:  # noqa: BLE001 - one driver must not sink the rest
                entries, problem = [], f"Listing {driver} scanners failed: {exc}"
            if problem:
                errors.append(problem)
                failed.add(driver)
            for entry in entries:
                if isinstance(entry, str):
                    entry = {"name": entry}
                found.append(
                    {
                        "name": entry["name"],
                        "driver": driver,
                        "engine": ENGINES[driver],
                        "usb_port": bool(entry.get("usb")),
                        "wia_id": entry.get("wia_id") or None,
                    }
                )
        try:
            usb = usb_job.result()
        except Exception as exc:  # noqa: BLE001
            usb = []
            errors.append(f"Could not check the USB devices on this computer: {exc}")

    kept, network = dedupe(found, order)
    usb_names = [d["name"] for d in usb if d.get("name")]
    devices = []
    ids: set[str] = set()
    for device in kept:
        # Stable across checks so the page can remember a choice. Only two
        # identically named scanners under one driver need the suffix.
        base = f"{device['driver']}:{device['name']}"
        device_id, n = base, 1
        while device_id in ids:
            n += 1
            device_id = f"{base} #{n}"
        ids.add(device_id)
        if device["wia_id"]:
            _wia_ids[device_id] = device["wia_id"]
        else:
            _wia_ids.pop(device_id, None)
        devices.append(
            {
                "id": device_id,
                "name": device["name"],
                "driver": device["driver"],
                "connection": classify_connection(device, usb_names, network),
                "engine": device["engine"],
                "status": "ready",
            }
        )

    unready = find_unready(usb, devices, key, bool(naps2))
    if unready and failed.intersection(USB_DRIVERS[key]):
        # The listing that would have found them did not finish, so telling
        # the person to install a driver they may well have would mislead.
        # Say they were seen instead; the failure itself is already above.
        names = ", ".join(row["name"] for row in unready)
        verb = "is" if len(unready) == 1 else "are"
        errors.append(
            f"{names} {verb} plugged in, but the check did not finish, so whether "
            "a driver is missing is not known yet. Press Check again."
        )
        unready = []

    return {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform or sys.platform,
        "naps2": {
            "installed": bool(naps2),
            "path": naps2,
            "install_hint": install_hint(platform),
        },
        "devices": devices,
        "unready": unready,
        "errors": errors,
    }


_devices_cache: dict = {"result": None, "at": None}
_devices_lock = threading.Lock()


def get_devices(refresh: bool = False, max_age: float = CACHE_SECONDS) -> dict:
    """check_devices(), reusing an answer younger than max_age.

    One check at a time: a request that arrives while another is checking
    waits for it and takes its answer rather than starting a second round of
    listings, even with refresh - that answer is newer than the request.
    """
    asked = time.monotonic()
    with _devices_lock:
        result, at = _devices_cache["result"], _devices_cache["at"]
        if result is not None and at is not None:
            if at >= asked or (not refresh and asked - at < max_age):
                return result
        result = check_devices()
        _devices_cache["result"], _devices_cache["at"] = result, time.monotonic()
        return result


def cached_devices() -> dict | None:
    """The last answer, however old, without checking again."""
    return _devices_cache["result"]


if __name__ == "__main__":
    print(json.dumps(check_devices(), indent=2))
