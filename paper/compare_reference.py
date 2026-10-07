"""Visual comparison of the rebuilt figures with the frozen manuscript's (``paper/reference/figs``).

Rasterizes every PDF at 150 dpi (poppler's ``pdftoppm``), compares pixels with the
reference, writes side-by-side PNGs (reference | rebuilt | amplified difference) and
prints a Markdown table. Byte equality is reported separately: matplotlib and font
drift can change bytes without a visible difference.

    python -m paper.compare_reference --out /tmp/fig-compare
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mulligan.plotting import paper

REFERENCE = Path(__file__).resolve().parent / "reference/figs"
DPI = 150
# A pixel "differs" when any channel moves by more than this (0-255); smaller changes
# are antialiasing and resampling noise.
PIXEL_TOLERANCE = 16


@dataclass(frozen=True)
class Comparison:
    name: str
    identical_bytes: bool
    size_match: bool
    mean_abs_diff: float
    differing_fraction: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rasterize(path: Path, scratch: Path) -> Image.Image:
    """The first page of a PDF at ``DPI``, or a raster image as-is, as RGB."""
    if path.suffix != ".pdf":
        return Image.open(path).convert("RGB")
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise RuntimeError("pdftoppm (poppler) is required to rasterize PDFs")
    stem = scratch / path.stem
    subprocess.run(
        [pdftoppm, "-r", str(DPI), "-png", "-singlefile", str(path), str(stem)], check=True
    )
    return Image.open(stem.with_suffix(".png")).convert("RGB")


def compare(built: Path, reference: Path, out_dir: Path | None, scratch: Path) -> Comparison:
    a = np.asarray(rasterize(reference, scratch / "ref"), dtype=np.int16)
    b_image = rasterize(built, scratch / "new")
    size_match = b_image.size == (a.shape[1], a.shape[0])
    if not size_match:
        b_image = b_image.resize((a.shape[1], a.shape[0]), Image.Resampling.LANCZOS)
    b = np.asarray(b_image, dtype=np.int16)
    diff = np.abs(a - b)
    comparison = Comparison(
        name=built.name,
        identical_bytes=_sha256(built) == _sha256(reference),
        size_match=size_match,
        mean_abs_diff=float(diff.mean()),
        differing_fraction=float((diff.max(axis=2) > PIXEL_TOLERANCE).mean()),
    )
    if out_dir is not None:
        amplified = np.clip(255 - 8 * diff.max(axis=2), 0, 255).astype(np.uint8)
        panels = [
            Image.fromarray(a.astype(np.uint8)),
            Image.fromarray(b.astype(np.uint8)),
            Image.fromarray(amplified).convert("RGB"),
        ]
        width, height = panels[0].size
        sheet = Image.new("RGB", (3 * width + 20, height), "white")
        for index, panel in enumerate(panels):
            sheet.paste(panel, (index * (width + 10), 0))
        out_dir.mkdir(parents=True, exist_ok=True)
        sheet.save(out_dir / f"{built.stem}.compare.png")
    return comparison


def compare_all(out_dir: Path | None) -> list[Comparison]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="paper-compare-") as scratch:
        root = Path(scratch)
        (root / "ref").mkdir()
        (root / "new").mkdir()
        for reference in sorted(REFERENCE.iterdir()):
            built = paper.FIGS_DIR / reference.name
            if not built.is_file():
                raise FileNotFoundError(f"{built} is missing; run python -m paper.figures")
            rows.append(compare(built, reference, out_dir, root))
    return rows


def markdown(rows: list[Comparison]) -> str:
    lines = [
        f"| Figure | Bytes identical | Size match | Mean abs diff (0-255) | Pixels > {PIXEL_TOLERANCE} |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.name}` | {'yes' if row.identical_bytes else 'no'} | "
            f"{'yes' if row.size_match else 'no'} | {row.mean_abs_diff:.4f} | "
            f"{100 * row.differing_fraction:.3f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="write side-by-side PNGs here")
    args = parser.parse_args()
    print(markdown(compare_all(args.out)))


if __name__ == "__main__":
    main()
