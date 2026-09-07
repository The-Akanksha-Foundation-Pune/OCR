"""Combine scanned page images into one PDF (called by scan.ps1)."""

import sys
from pathlib import Path

import cv2
import fitz
import numpy as np


def ink_ratio(path: Path) -> float:
    """Fraction of the page that is actual ink, ignoring paper tone."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return float(np.mean(image < 160))


# A printed form measures around 5% ink; a blank reverse showing only
# bleed-through sits under 1%. Anything above this is real content.
BLANK_INK_CEILING = 0.02


def pick_printed_sides(pages: list[Path]) -> tuple[list[Path], int]:
    """
    From a duplex scan, keep the printed face of each sheet.

    Duplex yields front and back adjacent, and normally only one carries
    the form, so the two are compared against each other rather than to a
    fixed cutoff - that works whichever way the stack was loaded.

    But a driver may quietly ignore the duplex request and return one image
    per sheet. Blind pairing would then throw away every second *form*, so
    when both sides of a pair carry real ink we keep both: a stray blank
    page costs the operator one click, a discarded form loses a student.
    """
    kept: list[Path] = []
    dropped = 0
    for index in range(0, len(pages), 2):
        pair = pages[index : index + 2]
        if len(pair) == 1:
            kept.append(pair[0])
            continue

        front, back = pair
        front_ink, back_ink = ink_ratio(front), ink_ratio(back)

        if min(front_ink, back_ink) > BLANK_INK_CEILING:
            kept.extend(pair)
            continue

        kept.append(front if front_ink >= back_ink else back)
        dropped += 1
    return kept, dropped


def main() -> int:
    list_file, out_file = Path(sys.argv[1]), Path(sys.argv[2])
    duplex = "--duplex" in sys.argv[3:]

    pages = [
        Path(line.strip())
        # utf-8-sig: PowerShell 5.1's Set-Content -Encoding utf8 writes a BOM,
        # which would otherwise glue ﻿ onto the first path and make it
        # unopenable.
        for line in list_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not pages:
        print("no pages to combine", file=sys.stderr)
        return 1

    if duplex:
        pages, dropped = pick_printed_sides(pages)
        if dropped:
            print(f"dropped {dropped} blank reverse side(s)")

    doc = fitz.open()
    for page_path in pages:
        with fitz.open(page_path) as image:
            pdf_bytes = image.convert_to_pdf()
        with fitz.open("pdf", pdf_bytes) as as_pdf:
            doc.insert_pdf(as_pdf)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_file, garbage=3, deflate=True)
    doc.close()
    print(f"wrote {len(pages)} page(s) to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
