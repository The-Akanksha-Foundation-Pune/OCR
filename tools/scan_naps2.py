"""Run one NAPS2 scan into a folder the scan helper is watching.

NAPS2 (https://www.naps2.com/download) drives scanners on every OS: TWAIN on
Windows, Apple's Image Capture drivers on a Mac (the same ones Image
Capture.app uses), SANE on Linux, and network (eSCL / AirScan) scanners
everywhere. scan_agent.py runs this for any scanner it found through NAPS2;
Windows WIA scanners go through scan.ps1 instead, which hands pages over one
at a time.

It writes what scan_agent.py reads, exactly as scan.ps1 does: page-001.jpg,
page-002.jpg, ... into --stream-dir, each only once it is whole, then
done.txt (the page count) or failed.txt (why - pages already written stay).
Unlike scan.ps1 the pages land together when the stack is through: NAPS2
only writes its files at the end.

For a one-sided feeder scan load the stack FACE DOWN; --source duplex
captures both sides instead.

    python3 tools/scan_naps2.py --list --driver apple
    python3 tools/scan_naps2.py --stream-dir /tmp/scan --driver apple --device "SP-1130N"
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Shipped alongside this file in the scan helper download.
import scanner_devices

DEFAULT_DRIVERS = {"win32": "wia", "darwin": "apple", "linux": "sane"}
SOURCE_LABELS = {
    "glass": "flatbed glass",
    "feeder": "document feeder, one side - load the stack face DOWN",
    "duplex": "document feeder, both sides",
}


class ScanError(Exception):
    pass


def write_marker(stream_dir: Path, name: str, text: str) -> None:
    """Write done.txt / failed.txt in one step, so the helper never reads a
    half-written one. Plain UTF-8, no byte-order mark."""
    partial = stream_dir / f"{name}.part"
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, stream_dir / name)


def _page_order(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[-1]) if numbers else 0, path.name)


def _staging_dir(naps2: str, stream_dir: Path) -> Path:
    """Where NAPS2 writes before the pages are handed over.

    Never the stream folder itself: the helper takes any page-*.jpg there as
    finished. A Flatpak NAPS2 has its own private /tmp and may not see the
    stream folder at all, but it can always write to its own app folder,
    which sits at the same path inside and outside the sandbox.
    """
    app = scanner_devices.flatpak_app_id(naps2)
    if app:
        base = Path.home() / ".var" / "app" / app / "cache" / "akanksha-scan"
        base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="staging-", dir=str(base)))
    return Path(tempfile.mkdtemp(prefix=".staging-", dir=str(stream_dir)))


def _hand_over(staged: list[Path], stream_dir: Path) -> int:
    """Move pages into the stream folder in page order, each under a .part
    name first and renamed once whole - the staging folder may be on another
    disk, where a move is a copy."""
    for number, source in enumerate(staged, start=1):
        final = stream_dir / f"page-{number:03d}.jpg"
        partial = stream_dir / f"page-{number:03d}.jpg.part"
        shutil.move(str(source), str(partial))
        os.replace(partial, final)
    return len(staged)


def _naps2_says(stdout: str) -> str:
    """NAPS2 prints nothing on a good scan without --verbose, so whatever it
    did print is the reason it went wrong."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return " ".join(lines)[:500]


def scan(args: argparse.Namespace, naps2: str) -> int:
    stream_dir: Path = args.stream_dir
    print(f"Scanner : {args.device} ({args.driver} driver)", flush=True)
    print(f"NAPS2   : {naps2}", flush=True)
    print(f"Source  : {SOURCE_LABELS[args.source]}", flush=True)
    print(f"Settings: {args.dpi} DPI, {'colour' if args.bitdepth == 'color' else 'grayscale'}", flush=True)

    staging = _staging_dir(naps2, stream_dir)
    argv = scanner_devices.naps2_command(naps2) + [
        # Our settings only - not whatever was last used in the NAPS2 app.
        "--noprofile",
        "--driver", args.driver,
        "--device", args.device,
        "--source", args.source,
        "--dpi", str(args.dpi),
        "--bitdepth", args.bitdepth,
        "--split",
        "--force",
        "-o", str(staging / "$(nnnn).jpg"),
    ]
    try:
        # Output to files, not pipes: a pipe held open by anything NAPS2
        # starts would leave this waiting after NAPS2 itself is gone.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            process = subprocess.Popen(
                argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL
            )
            try:
                code = process.wait(timeout=args.timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                scanner_devices.kill_process_tree(process)
                code, timed_out = None, True
            out.seek(0)
            err.seek(0)
            stdout = out.read().decode("utf-8", errors="replace")
            stderr = err.read().decode("utf-8", errors="replace")
        for text in (stdout, stderr):
            if text.strip():
                print(text.strip(), flush=True)

        if timed_out:
            # Killed mid-run, NAPS2 may have been half way through writing a
            # file - hand nothing over rather than a broken page.
            raise ScanError(scanner_devices.timeout_message(args.timeout))

        count = _hand_over(sorted(staging.glob("*.jpg"), key=_page_order), stream_dir)
        said = _naps2_says(stdout)
        kept = f" The {count} page(s) scanned before that were kept." if count else ""
        if code != 0:
            raise ScanError(
                f"NAPS2 stopped with an error (exit code {code})"
                + (f": {said}" if said else ".")
                + kept
            )
        if said and count:
            # NAPS2 still exits 0 after a jam part-way through; a partial
            # stack must not pass for a whole one.
            raise ScanError(f"The scan stopped part-way - NAPS2 said: {said}{kept}")
        if said:
            raise ScanError(f"Nothing was scanned - NAPS2 said: {said}")
        if not count:
            if args.source == "glass":
                raise ScanError("Nothing was scanned. Is the scanner switched on?")
            raise ScanError("Nothing was scanned. Is there paper in the feeder?")
        return count
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _naps2_from_args(value: list[str] | None) -> str | None:
    if not value:
        return scanner_devices.find_naps2()
    if len(value) > 1:
        # The argv form: --naps2 flatpak run com.naps2.Naps2
        return " ".join(shlex.quote(word) for word in value)
    return value[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stream-dir", type=Path, help="where page-NNN.jpg files go")
    parser.add_argument(
        "--naps2",
        nargs="+",
        metavar="PATH",
        help='NAPS2 program, or a Flatpak command ("flatpak run com.naps2.Naps2"); found automatically if left out',
    )
    parser.add_argument(
        "--driver",
        default=DEFAULT_DRIVERS[scanner_devices.platform_key()],
        choices=["wia", "twain", "apple", "sane", "escl"],
    )
    parser.add_argument("--device", help="the scanner's name exactly as --list prints it")
    parser.add_argument("--source", default="feeder", choices=["glass", "feeder", "duplex"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bitdepth", default="gray", choices=["gray", "color"])
    parser.add_argument("--timeout", type=float, default=900, help="seconds before a stuck scan is stopped")
    parser.add_argument("--list", action="store_true", help="list scanners for --driver and exit")
    args = parser.parse_args()

    naps2 = _naps2_from_args(args.naps2)

    if args.list:
        if not naps2:
            sys.exit("NAPS2 is not installed. " + scanner_devices.install_hint())
        names, problem = scanner_devices.list_naps2_devices(naps2, args.driver)
        for name in names:
            print(name)
        if problem:
            print(problem, file=sys.stderr)
            if not names:
                sys.exit(1)
        return

    if not args.stream_dir:
        parser.error("--stream-dir is required (or --list)")
    args.stream_dir.mkdir(parents=True, exist_ok=True)

    # Every way out below leaves exactly one marker, so the helper is never
    # left polling a folder that will not finish.
    try:
        if not naps2:
            raise ScanError("NAPS2 is not installed on this computer. " + scanner_devices.install_hint())
        if not args.device:
            raise ScanError("No scanner was named - pass --device as --list prints it.")
        count = scan(args, naps2)
    except ScanError as exc:
        message = str(exc)
    except (OSError, subprocess.SubprocessError) as exc:
        message = f"The scan could not run: {exc}"
    except KeyboardInterrupt:
        message = "The scan was stopped."
    except Exception as exc:  # noqa: BLE001 - any failure must still leave failed.txt
        message = f"The scan failed unexpectedly: {exc}"
    else:
        write_marker(args.stream_dir, "done.txt", str(count))
        print(f"Streamed {count} page(s)", flush=True)
        return
    write_marker(args.stream_dir, "failed.txt", message)
    print(f"failed: {message}", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
