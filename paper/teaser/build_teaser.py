"""Render the authored HTML teaser (``overview_teaser.pdf``) with headless Chrome, then crop it.

System requirements: Chrome or Chromium (``$MULLIGAN_CHROME`` or ``google-chrome`` /
``chromium`` on ``PATH``) and Ghostscript (``gs``) for the bounding-box crop. The bucket
chart's counts are regenerated from the pinned simulation evidence into the staged copy
of ``assets/bucket_counts.js``; the committed copy documents the same numbers.

    python -m paper.teaser.build_teaser
"""

from __future__ import annotations

import hashlib
import os
import re
import selectors
import shutil
import signal
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from mulligan.plotting import paper

HERE = Path(__file__).resolve().parent
CHROME_ENV = "MULLIGAN_CHROME"
CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
OUTPUT = "overview_teaser.pdf"
# pdfcrop --margins 1: one point of white margin around the ink bounding box.
MARGIN_PT = 1.0


class ChromeNotFound(RuntimeError):
    pass


def chrome_binary() -> str:
    """The Chrome/Chromium executable, from ``$MULLIGAN_CHROME`` or ``PATH``."""
    configured = os.environ.get(CHROME_ENV)
    if configured:
        if not (Path(configured).is_file() and os.access(configured, os.X_OK)):
            raise ChromeNotFound(f"${CHROME_ENV}={configured} is not an executable file")
        return configured
    for name in CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise ChromeNotFound(
        f"Chrome/Chromium not found: set ${CHROME_ENV} or put one of {CHROME_NAMES} on PATH"
    )


def _print_to_pdf(chrome: str, html: Path, raw_pdf: Path, profile: str) -> None:
    command = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={raw_pdf}",
        html.as_uri(),
    ]
    # Some Chrome builds write the PDF but leave background services running.
    # Wait for the explicit write-completion message, then close this isolated
    # instance rather than waiting indefinitely for exit.
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True
    )
    log = b""
    deadline = time.monotonic() + 60
    with selectors.DefaultSelector() as selector:
        selector.register(process.stderr, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            if not selector.select(timeout=max(0, deadline - time.monotonic())):
                break
            chunk = os.read(process.stderr.fileno(), 65536)
            if not chunk:
                break
            log += chunk
            if f"bytes written to file {raw_pdf}".encode() in log:
                break
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        _, tail = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        _, tail = process.communicate(timeout=10)
    log += tail
    if not (raw_pdf.is_file() and raw_pdf.stat().st_size > 0):
        raise RuntimeError(f"Chrome did not write {raw_pdf}:\n{log.decode(errors='replace')}")


def _clean_id(pdf: Path) -> None:
    """Replace the trailer ``/ID`` (Ghostscript derives it from the clock) by a hash of the
    file's other bytes. Same length, so the cross-reference offsets stay valid."""
    data = pdf.read_bytes()
    ids = re.findall(rb"/ID \[<([0-9A-F]{32})><\1>\]", data)
    if len(ids) != 1:
        raise RuntimeError(f"{pdf}: expected one trailer /ID, found {len(ids)}")
    digest = hashlib.md5(data.replace(ids[0], b"0" * 32)).hexdigest().upper().encode()
    pdf.write_bytes(data.replace(ids[0], digest))


def _crop(raw_pdf: Path, out_pdf: Path) -> None:
    """Crop to the ink bounding box plus ``MARGIN_PT`` (what ``pdfcrop --margins 1`` does).

    Ghostscript writes the cropped page into a new PDF: no input file names (unlike
    ``pdfcrop``'s ``PTEX.FileName``), dates fixed by ``SOURCE_DATE_EPOCH=0``, and a content-hash
    document id.
    """
    gs = shutil.which("gs")
    if gs is None:
        raise RuntimeError("Ghostscript (gs) not found; it is required to crop the teaser")
    bbox = subprocess.run(
        [gs, "-q", "-dBATCH", "-dNOPAUSE", "-dSAFER", "-sDEVICE=bbox", str(raw_pdf)],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    # pdfcrop uses the integer %%BoundingBox, not the high-resolution one.
    boxes = re.findall(r"%%BoundingBox: (-?\d+) (-?\d+) (-?\d+) (-?\d+)", bbox)
    if len(boxes) != 1:
        raise RuntimeError(f"expected a one-page teaser, got bounding boxes {boxes}")
    x0, y0, x1, y1 = (float(v) for v in boxes[0])
    x0, y0, x1, y1 = x0 - MARGIN_PT, y0 - MARGIN_PT, x1 + MARGIN_PT, y1 + MARGIN_PT
    subprocess.run(
        [
            gs,
            "-q",
            "-dBATCH",
            "-dNOPAUSE",
            "-dSAFER",
            "-sDEVICE=pdfwrite",
            # Keep the embedded photo lossless (Flate) at its resolution.
            "-dPassThroughJPEGImages=true",
            "-dAutoFilterColorImages=false",
            "-dColorImageFilter=/FlateEncode",
            "-dAutoFilterGrayImages=false",
            "-dGrayImageFilter=/FlateEncode",
            "-dDownsampleColorImages=false",
            "-dDownsampleGrayImages=false",
            "-dDownsampleMonoImages=false",
            "-dFIXEDMEDIA",
            f"-dDEVICEWIDTHPOINTS={x1 - x0:.3f}",
            f"-dDEVICEHEIGHTPOINTS={y1 - y0:.3f}",
            f"-sOutputFile={out_pdf}",
            "-c",
            f"<</PageOffset [{-x0:.3f} {-y0:.3f}]>> setpagedevice",
            "-f",
            str(raw_pdf),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "SOURCE_DATE_EPOCH": "0"},
    )
    _clean_id(out_pdf)


def build(out_dir: Path = paper.FIGS_DIR) -> Path:
    """Stage the teaser, print it with Chrome, and write the cropped ``overview_teaser.pdf``."""
    from paper.appendix.productivity.teaser import bucket_counts_js

    chrome = chrome_binary()
    stage = paper.BUILD_DIR / "teaser"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "assets").mkdir(parents=True)
    shutil.copyfile(HERE / "teaser.html", stage / "teaser.html")
    shutil.copyfile(
        HERE / "assets/routing_init_overlay.png", stage / "assets/routing_init_overlay.png"
    )
    counts = bucket_counts_js()
    committed = (HERE / "assets/bucket_counts.js").read_text()
    if _data_lines(counts) != _data_lines(committed):
        raise RuntimeError("paper/teaser/assets/bucket_counts.js differs from the pinned evidence")
    (stage / "assets/bucket_counts.js").write_text(counts)
    raw_pdf = stage / "teaser_raw.pdf"
    with TemporaryDirectory(prefix="mulligan-teaser-chrome-") as profile:
        _print_to_pdf(chrome, stage / "teaser.html", raw_pdf, profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / OUTPUT
    _crop(raw_pdf, out_pdf)
    print(f"wrote {out_pdf}")
    return out_pdf


def _data_lines(js: str) -> list[str]:
    return [line for line in js.splitlines() if not line.startswith("//")]


def main() -> None:
    build()


if __name__ == "__main__":
    main()
