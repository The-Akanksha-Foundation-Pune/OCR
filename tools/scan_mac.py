"""Scan consent forms on a Mac - the macOS counterpart of tools/scan.ps1.

scan.ps1 drives the scanner through Windows Image Acquisition, which a Mac
does not have. Here NAPS2's command line does the driving instead, through
Apple's own Image Capture driver - the same one Image Capture.app uses, so
any scanner that shows up there works here.

Install NAPS2 once (free):

    brew install --cask naps2

Same settings as scan.ps1: 300 DPI grayscale from the document feeder, page 1
of each form only, stack loaded FACE DOWN. Pass --duplex to capture both
sides instead.

It writes what scan_agent.py reads, exactly as scan.ps1 does: page-001.jpg,
page-002.jpg, ... into --stream-dir, then done.txt (the page count) or
failed.txt (why). Unlike scan.ps1 the pages land together when the stack is
through rather than one by one - NAPS2 only writes its files at the end.

    python3 tools/scan_mac.py --list
    python3 tools/scan_mac.py --stream-dir /tmp/scan --scanner SP-1130N
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NAPS2_CANDIDATES = [
    Path("/Applications/NAPS2.app/Contents/MacOS/NAPS2"),
    Path.home() / "Applications/NAPS2.app/Contents/MacOS/NAPS2",
]
DEFAULT_SCANNER = "SP-1130N"
# Apple's Image Capture driver. "sane" and "escl" are NAPS2's other Mac
# drivers, for a scanner that has no Mac driver of its own.
DEFAULT_DRIVER = "apple"


class ScanError(Exception):
    pass


def naps2() -> Path:
    for path in NAPS2_CANDIDATES:
        if path.exists():
            return path
    raise ScanError(
        "NAPS2 is not installed. Install it with: brew install --cask naps2"
    )


def list_devices(driver: str) -> list[str]:
    done = subprocess.run(
        [str(naps2()), "console", "--listdevices", "--driver", driver],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return [ln.strip() for ln in done.stdout.splitlines() if ln.strip()]


def pick_device(wanted: str, driver: str) -> str:
    """The full name NAPS2 knows the scanner by.

    The Image Capture name usually carries the maker as well ("FUJITSU
    SP-1130N"), so match on the part we were given. With only one scanner
    attached, take it whatever it is called.
    """
    devices = list_devices(driver)
    if not devices:
        raise ScanError(
            "No scanner found. Is it switched on and connected, and does it "
            "show up in Image Capture?"
        )
    matches = [d for d in devices if wanted.lower() in d.lower()]
    if len(matches) >= 1:
        return matches[0]
    if len(devices) == 1:
        return devices[0]
    raise ScanError(
        f"No scanner called {wanted!r}. Found: {', '.join(devices)}"
    )


def _page_order(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[-1]) if numbers else 0, path.name)


def scan(stream_dir: Path, scanner: str, driver: str, duplex: bool, dpi: int) -> int:
    device = pick_device(scanner, driver)
    print(f"Scanner : {device} ({driver} driver)", flush=True)
    print(f"Source  : document feeder, {'both sides' if duplex else 'one side - load face DOWN'}", flush=True)
    print(f"Settings: {dpi} DPI, grayscale", flush=True)

    # NAPS2 writes into a scratch folder first: scan_agent.py treats any
    # page-*.jpg in stream_dir as finished, so nothing may appear there
    # until it is whole.
    staging = Path(tempfile.mkdtemp(prefix="naps2-"))
    try:
        done = subprocess.run(
            [
                str(naps2()),
                "console",
                # Our settings only - not whatever was last used in the NAPS2 app.
                "--noprofile",
                "--driver", driver,
                "--device", device,
                "--source", "duplex" if duplex else "feeder",
                "--dpi", str(dpi),
                "--bitdepth", "gray",
                "--split",
                "--force",
                "-o", str(staging / "$(nnnn).jpg"),
            ],
            capture_output=True,
            text=True,
        )
        output = (done.stdout + done.stderr).strip()
        if output:
            print(output, flush=True)

        scanned = sorted(staging.glob("*.jpg"), key=_page_order)
        if not scanned:
            last = output.splitlines()[-1] if output else ""
            raise ScanError(
                "Nothing was scanned. Is there paper in the feeder?"
                + (f" ({last})" if last else "")
            )
        for number, path in enumerate(scanned, start=1):
            shutil.move(str(path), stream_dir / f"page-{number:03d}.jpg")
        return len(scanned)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stream-dir", type=Path, help="where page-NNN.jpg files go")
    parser.add_argument("--scanner", default=DEFAULT_SCANNER)
    parser.add_argument("--driver", default=DEFAULT_DRIVER, choices=["apple", "sane", "escl"])
    parser.add_argument("--duplex", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--list", action="store_true", help="list scanners and exit")
    args = parser.parse_args()

    if args.list:
        try:
            for name in list_devices(args.driver):
                print(name)
        except ScanError as exc:
            sys.exit(str(exc))
        return

    if not args.stream_dir:
        parser.error("--stream-dir is required (or --list)")
    args.stream_dir.mkdir(parents=True, exist_ok=True)

    try:
        count = scan(args.stream_dir, args.scanner, args.driver, args.duplex, args.dpi)
    except (ScanError, OSError, subprocess.SubprocessError) as exc:
        (args.stream_dir / "failed.txt").write_text(str(exc), encoding="utf-8")
        print(f"failed: {exc}", flush=True)
        sys.exit(1)
    (args.stream_dir / "done.txt").write_text(str(count), encoding="ascii")
    print(f"Streamed {count} page(s)", flush=True)


if __name__ == "__main__":
    main()
