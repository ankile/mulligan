"""Shared plotting helpers for real-world lifecycle figures: deterministic SVG output and
repo-relative display paths."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]


def display_path(path: Path) -> str:
    """Repo-relative display form of ``path`` (falls back to the raw path outside the repo)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_svg(fig: plt.Figure, path: Path) -> None:
    """Write an SVG with no timestamp and no trailing whitespace (stable diffs)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")
