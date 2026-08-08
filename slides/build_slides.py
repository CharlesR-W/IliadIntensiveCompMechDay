#!/usr/bin/env python3
"""Build the offline Reveal.js micro-decks and matching PDFs."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import Thread

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides"
ASSETS = SLIDES / "assets" / "generated"
DIST = SLIDES / "dist"

DECKS = (
    "01_hmm_primer",
    "02_hmm_msp_synthesis",
    "03_transformer_primer",
    "04_transformer_debrief",
    "05_hankel_psr_primer",
    "06_final_synthesis",
)

# Each entry is (notebook path, code-cell ID, image-within-cell index, output name).
FIGURES = (
    ("notebooks/01_hmms_msps_solutions.ipynb", "03808fe8311a", 0, "z1r_generator.png"),
    ("notebooks/01_hmms_msps_solutions.ipynb", "82a9e5c96d83", 0, "z1r_belief_geometry.png"),
    ("notebooks/01_hmms_msps_solutions.ipynb", "614e8f0c3946", 0, "z1r_future_comparison.png"),
    ("notebooks/01_hmms_msps_solutions.ipynb", "c347e659da9b", 0, "mess3_one_symbol.png"),
    ("notebooks/01_hmms_msps_solutions.ipynb", "39dc3db3167a", 0, "mess3_recursive_geometry.png"),
    ("notebooks/02_transformer_belief_geometry_solutions.ipynb", "998c22f73eeb", 0, "transformer_probe.png"),
    ("notebooks/03_hankel_psr_wfa_solutions.ipynb", "2f2f07b5d19b", 0, "observable_route.png"),
)


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def notebook_image(notebook_path: Path, cell_id: str, image_index: int) -> bytes:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    matches = [cell for cell in notebook["cells"] if cell.get("id") == cell_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell {cell_id} in {notebook_path}, found {len(matches)}")

    images: list[str] = []
    for output in matches[0].get("outputs", []):
        encoded = output.get("data", {}).get("image/png")
        if encoded is None:
            continue
        images.append("".join(encoded) if isinstance(encoded, list) else encoded)

    if image_index >= len(images):
        raise RuntimeError(
            f"Cell {cell_id} in {notebook_path} has {len(images)} PNG output(s), "
            f"not index {image_index}"
        )
    return base64.b64decode(images[image_index])


def extract_assets() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    expected = {name for *_, name in FIGURES}
    unexpected = {path.name for path in ASSETS.iterdir() if path.is_file()} - expected
    if unexpected:
        raise RuntimeError(f"Unexpected generated slide assets: {sorted(unexpected)}")

    for relative_notebook, cell_id, image_index, name in FIGURES:
        payload = notebook_image(ROOT / relative_notebook, cell_id, image_index)
        target = ASSETS / name
        target.write_bytes(payload)
        print(f"extracted {target.relative_to(ROOT)} ({len(payload):,} bytes)")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def expected_pages(deck: str) -> int:
    source = (SLIDES / f"{deck}.qmd").read_text(encoding="utf-8")
    return 1 + len(re.findall(r"^## ", source, flags=re.MULTILINE))


def rendered_pages(pdf_path: Path) -> int | None:
    if not pdf_path.exists():
        return None
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=False
    )
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def capture_slide(
    chromium: str,
    url: str,
    screenshot_path: Path,
    profile_prefix: str,
    env: dict[str, str],
) -> None:
    """Capture one live Reveal canvas, retrying only invalid browser output."""
    for attempt in range(1, 4):
        screenshot_path.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"{profile_prefix}-{attempt}-", dir="/tmp"
        ) as profile:
            result = subprocess.run(
                [
                    chromium,
                    "--headless=new",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--disable-background-networking",
                    "--no-sandbox",
                    f"--user-data-dir={profile}",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=1",
                    "--window-size=1600,900",
                    "--virtual-time-budget=5000",
                    f"--screenshot={screenshot_path}",
                    url,
                ],
                cwd=SLIDES,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode == 0 and screenshot_path.exists():
            with Image.open(screenshot_path) as image:
                extrema = image.convert("RGB").getextrema()
                if image.size == (1600, 900) and not all(
                    low == high for low, high in extrema
                ):
                    return
        print(
            f"retrying screenshot {screenshot_path.name}: attempt {attempt} "
            f"returned {result.returncode}"
        )
    raise RuntimeError(f"Could not capture a valid frame for {url}")


def export_pdfs(
    chromium: str,
    img2pdf: str,
    env: dict[str, str],
    decks: tuple[str, ...] = DECKS,
) -> None:
    """Export lossless 16:9 PDF handouts from the live Reveal canvases."""
    handler = partial(QuietHandler, directory=str(DIST))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        for deck in decks:
            html_path = (DIST / f"{deck}.html").resolve()
            pdf_path = (DIST / f"{deck}.pdf").resolve()
            if not html_path.exists():
                raise RuntimeError(f"Quarto did not produce {html_path}")

            target_pages = expected_pages(deck)
            with tempfile.TemporaryDirectory(
                prefix=f"iliad-{deck}-frames-", dir="/tmp"
            ) as frame_directory_name:
                frame_directory = Path(frame_directory_name)
                frame_paths = [
                    frame_directory / f"{slide_index + 1:02d}.png"
                    for slide_index in range(target_pages)
                ]
                with ThreadPoolExecutor(max_workers=min(4, target_pages)) as executor:
                    futures = {
                        executor.submit(
                            capture_slide,
                            chromium,
                            f"http://127.0.0.1:{port}/{deck}.html?handout=1#/"
                            f"{slide_index}",
                            frame_path,
                            f"iliad-{deck}-{slide_index + 1}",
                            env,
                        ): slide_index
                        for slide_index, frame_path in enumerate(frame_paths)
                    }
                    for future in as_completed(futures):
                        future.result()

                with tempfile.NamedTemporaryFile(
                    prefix=f".{deck}-", suffix=".pdf", dir=DIST, delete=False
                ) as handle:
                    candidate_pdf = Path(handle.name)
                candidate_pdf.unlink()
                try:
                    run(
                        [
                            img2pdf,
                            "--nodate",
                            "--pagesize",
                            "13.333333inx7.5in",
                            "--imgsize",
                            "13.333333inx7.5in",
                            "--title",
                            deck.replace("_", " "),
                            "--creator",
                            "ILIAD Intensive slide builder",
                            "--output",
                            str(candidate_pdf),
                            *(str(path) for path in frame_paths),
                        ],
                        cwd=SLIDES,
                        env=env,
                    )
                    if rendered_pages(candidate_pdf) != target_pages:
                        raise RuntimeError(
                            f"{candidate_pdf.name} did not contain {target_pages} pages"
                        )
                    os.replace(candidate_pdf, pdf_path)
                finally:
                    candidate_pdf.unlink(missing_ok=True)
                print(f"exported {pdf_path.name} ({target_pages} slides)")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-notebook-validation", action="store_true")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument(
        "--deck",
        choices=DECKS,
        help="print one deck while still rendering and validating the full suite",
    )
    args = parser.parse_args()

    quarto = shutil.which("quarto")
    chromium = "/usr/bin/chromium"
    img2pdf = shutil.which("img2pdf")
    if quarto is None:
        raise RuntimeError("quarto is required to build the slide deck")
    if not args.skip_pdf and not Path(chromium).exists():
        raise RuntimeError(f"Chromium not found at {chromium}")
    if not args.skip_pdf and img2pdf is None:
        raise RuntimeError("img2pdf is required to build PDF handouts")

    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = "/tmp/iliad-quarto-cache"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = "/tmp/iliad-matplotlib"

    if not args.skip_notebook_validation:
        run(["python", "notebooks/build_notebooks.py", "--check"], cwd=ROOT, env=env)
        run(["python", "notebooks/validate_notebooks.py"], cwd=ROOT, env=env)

    extract_assets()
    run([quarto, "render", "."], cwd=SLIDES, env=env)

    if not args.skip_pdf:
        export_pdfs(
            chromium,
            img2pdf,
            env,
            (args.deck,) if args.deck else DECKS,
        )

    run(["python", "validate_slides.py"], cwd=SLIDES, env=env)


if __name__ == "__main__":
    main()
