"""Single pinned files from an appendix package's paper-evidence lock (``inputs.json``)."""

from __future__ import annotations

import json
from pathlib import Path

from paper.appendix.artifacts import fetch

APPENDIX = Path(__file__).resolve().parents[1] / "appendix"


def pinned_input(package: str, path: str) -> Path:
    """Materialize the input ``path`` of ``paper/appendix/<package>/inputs.json``, hash-checked."""
    lock = json.loads((APPENDIX / package / "inputs.json").read_text())
    (row,) = (row for row in lock["files"] if row["path"] == path)
    return fetch(package, row)
