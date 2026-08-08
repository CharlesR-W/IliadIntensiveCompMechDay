#!/usr/bin/env python3
"""Validate slide sources, offline HTML bundles, and rendered PDFs."""

from __future__ import annotations

import re
from pathlib import Path
import subprocess


SLIDES = Path(__file__).resolve().parent
DIST = SLIDES / "dist"
ASSETS = SLIDES / "assets" / "generated"

EXPECTED_SLIDES = {
    "01_hmm_primer": 5,
    "02_hmm_msp_synthesis": 10,
    "03_transformer_primer": 6,
    "04_transformer_debrief": 4,
    "05_hankel_psr_primer": 7,
    "06_final_synthesis": 10,
}

EXPECTED_ASSETS = {
    "z1r_generator.png",
    "z1r_belief_geometry.png",
    "z1r_future_comparison.png",
    "mess3_one_symbol.png",
    "mess3_recursive_geometry.png",
    "transformer_probe.png",
    "observable_route.png",
}

FORBIDDEN_SOURCE = {
    "01_hmm_primer": ("T^{(0)}=", "T^{(1)}=", "mess3", "fractal"),
    "03_transformer_primer": ("R^2=0.95", "R^2=0.31", "concatenated activations"),
    "05_hankel_psr_primer": ("H_{\\cdot,\\epsilon}", "rank exactly two", "A_0=", "A_1=", "B^{-1}"),
}

REQUIRED_SOURCE = {
    "06_final_synthesis": (
        "longum iter est per praecepta, breve et efficax per exempla",
        "multum interest utrum non velit an nesciat",
        "velle non discitur",
        "Seneca the Younger",
        "Moral Letters to Lucilius",
    ),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def pdf_pages(path: Path) -> tuple[int, float]:
    result = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, check=True
    ).stdout
    page_match = re.search(r"^Pages:\s+(\d+)$", result, flags=re.MULTILINE)
    size_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", result, flags=re.MULTILINE
    )
    if page_match is None or size_match is None:
        fail(f"Could not read page metadata from {path}")
    width, height = float(size_match.group(1)), float(size_match.group(2))
    return int(page_match.group(1)), width / height


def main() -> None:
    actual_assets = {path.name for path in ASSETS.glob("*.png")}
    if actual_assets != EXPECTED_ASSETS:
        fail(
            f"Generated asset inventory mismatch: expected {sorted(EXPECTED_ASSETS)}, "
            f"found {sorted(actual_assets)}"
        )

    expected_top_level = {
        f"{deck}.{suffix}"
        for deck in EXPECTED_SLIDES
        for suffix in ("html", "pdf")
    }
    actual_top_level = {
        path.name for path in DIST.iterdir() if path.is_file() and path.suffix in {".html", ".pdf"}
    }
    if actual_top_level != expected_top_level:
        fail(
            f"Rendered deck inventory mismatch: expected {sorted(expected_top_level)}, "
            f"found {sorted(actual_top_level)}"
        )

    for deck, expected_pages in EXPECTED_SLIDES.items():
        qmd = SLIDES / f"{deck}.qmd"
        html = DIST / f"{deck}.html"
        pdf = DIST / f"{deck}.pdf"
        source = qmd.read_text(encoding="utf-8")

        slide_headings = len(re.findall(r"^## ", source, flags=re.MULTILINE))
        if slide_headings + 1 != expected_pages:
            fail(
                f"{qmd.name}: title + headings imply {slide_headings + 1} slides, "
                f"expected {expected_pages}"
            )

        note_blocks = len(re.findall(r"^::: \{\.notes\}", source, flags=re.MULTILINE))
        if note_blocks < slide_headings:
            fail(f"{qmd.name}: {note_blocks} note blocks for {slide_headings} content slides")

        for forbidden in FORBIDDEN_SOURCE.get(deck, ()):
            if forbidden.casefold() in source.casefold():
                fail(f"{qmd.name}: spoiler guard found {forbidden!r}")

        for required in REQUIRED_SOURCE.get(deck, ()):
            if required.casefold() not in source.casefold():
                fail(f"{qmd.name}: missing required attribution text {required!r}")

        html_text = html.read_text(encoding="utf-8")
        remote_resources = re.findall(
            r"(?:src|data-src)=[\"']https?://|@import\s+(?:url\()?['\"]?https?://",
            html_text,
            flags=re.IGNORECASE,
        )
        if remote_resources:
            fail(f"{html.name}: contains remote runtime resources")

        pages, aspect = pdf_pages(pdf)
        if pages != expected_pages:
            fail(f"{pdf.name}: {pages} pages, expected {expected_pages}")
        if not 1.75 <= aspect <= 1.80:
            fail(f"{pdf.name}: page aspect ratio {aspect:.3f}, expected 16:9")

        subprocess.run(["qpdf", "--check", str(pdf)], check=True, capture_output=True)
        print(f"PASS {deck}: {pages} slides, aspect {aspect:.3f}, notes and offline assets present")


if __name__ == "__main__":
    main()
