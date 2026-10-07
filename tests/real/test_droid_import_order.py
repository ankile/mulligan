"""Every DROID import in mulligan comes after mulligan.real.robot.droid_compat.

droid.misc.parameters calls the OpenCV aruco API that opencv 4.7 removed, and droid_compat puts it
back. A droid import that runs first (for example after an import sorter moved it) fails with
``AttributeError: module 'cv2.aruco' has no attribute 'Dictionary_get'`` on the robot workstation
only, where droid is installed. Parsed from source; droid is not needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
COMPAT = "mulligan.real.robot.droid_compat"
# station_env has no OpenCV dependency; station.py reads it lazily to validate the station file.
EXEMPT_DROID_MODULES = {"droid.misc.station_env"}


def _imports(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return sorted(found)


DROID_IMPORTERS = sorted(
    path
    for path in (REPO / "mulligan").rglob("*.py")
    if path.name != "droid_compat.py"
    and any(
        name == "droid" or name.startswith("droid.")
        for _, name in _imports(ast.parse(path.read_text()))
    )
)


def test_droid_importers_found():
    assert DROID_IMPORTERS, "no module imports droid; the scan is broken"


@pytest.mark.parametrize("path", DROID_IMPORTERS, ids=lambda p: str(p.relative_to(REPO)))
def test_droid_compat_imported_before_droid(path):
    imports = _imports(ast.parse(path.read_text()))
    droid_lines = [
        line
        for line, name in imports
        if (name == "droid" or name.startswith("droid.")) and name not in EXEMPT_DROID_MODULES
    ]
    if not droid_lines:
        return
    compat_lines = [
        line
        for line, name in imports
        if name == COMPAT
        or name.startswith(COMPAT + ".")
        or name == "mulligan.real.robot.camera_config"
    ]
    assert compat_lines and min(compat_lines) < min(droid_lines), (
        f"{path.relative_to(REPO)}: droid is imported on line {min(droid_lines)} before "
        f"{COMPAT} (OpenCV aruco shim)"
    )
