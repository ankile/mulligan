"""Third-party code in the release tree carries its upstream license notice (THIRD_PARTY_NOTICES.md).

A file needs a notice in its leading comment block when it is one of the vendored RLPD/EXPO files
(``VENDORED_DIRS``, ``VENDORED_FILES``), when its leading comments or module docstring say it is
vendored or adapted from another project, or when it is one of the known adapted files below.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

SCANNED = ("mulligan", "paper", "scripts", "hardware", "robot")
# The RLPD agent, dataset and training code vendored from EXPO / RLPD.
VENDORED_DIRS = (
    "mulligan/baselines/rlpd/agent",
    "mulligan/baselines/rlpd/data",
)
VENDORED_FILES = (
    "mulligan/baselines/rlpd/configs.py",
    "mulligan/baselines/rlpd/datasets.py",
    "mulligan/baselines/rlpd/evaluation.py",
    "mulligan/baselines/rlpd/train.py",
)
CLAIM = re.compile(r"\b(?:vendored|adapted)\b(?: in spirit)? from\b", re.IGNORECASE)
# Files adapted from third-party code whose docstring may not say so.
KNOWN_ADAPTED = (
    "mulligan/baselines/hilserl/replay.py",
    "mulligan/data/fast_lerobot_reader.py",
    "mulligan/real/policy/lerobot_patches.py",
    "mulligan/sim/_patches.py",
    "mulligan/sim/render_cgl.py",
    "mulligan/teleop/spacemouse.py",
    "mulligan/utils/lerobot_patches.py",
)
# Files that mention vendored code but contain none.
EXEMPT = {
    "mulligan/baselines/hilserl/__init__.py": "names the vendored RLPD agent; no upstream code",
    "mulligan/baselines/rlpd/__init__.py": "docstring-only package init (EXPO's is empty)",
}
PERMISSION = ("Permission is hereby granted", "Licensed under the Apache License")
# Upstream named in the header -> copyright lines its notice must contain.
COPYRIGHTS = {
    "EXPO": ("Copyright (c) 2025 pd-perry", "Copyright (c) 2022 Ilya Kostrikov"),
    "robosuite": ("Copyright (c) 2022 Stanford Vision and Learning Lab",),
    "dm_control": ("Copyright 2017 The dm_control Authors",),
    "LeRobot": ("Copyright 2024", "The HuggingFace Inc. team"),
}


def _leading_comments(source: str) -> str:
    lines = []
    for line in source.splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line.lstrip("# "))
        else:
            break
    return "\n".join(lines)


def _claims_third_party(source: str) -> bool:
    docstring = ast.get_docstring(ast.parse(source)) or ""
    return bool(CLAIM.search(_leading_comments(source)) or CLAIM.search(docstring))


def vendor_targets() -> set[str]:
    files = {
        path.relative_to(REPO).as_posix()
        for directory in VENDORED_DIRS
        for path in (REPO / directory).rglob("*.py")
    }
    return files | set(VENDORED_FILES)


def files_needing_notice() -> set[str]:
    claimed = {
        path.relative_to(REPO).as_posix()
        for top in SCANNED
        for path in (REPO / top).rglob("*.py")
        if "node_modules" not in path.parts and _claims_third_party(path.read_text())
    }
    return (vendor_targets() | claimed | set(KNOWN_ADAPTED)) - set(EXEMPT)


def header_problems(source: str) -> list[str]:
    header = _leading_comments(source)
    problems = []
    if not any(text in header for text in PERMISSION):
        problems.append("no license permission notice in the leading comments")
    upstreams = [name for name in COPYRIGHTS if name in header]
    if not upstreams:
        problems.append(f"leading comments name none of {sorted(COPYRIGHTS)}")
    for name in upstreams:
        problems += [f"missing {line!r}" for line in COPYRIGHTS[name] if line not in header]
    return problems


def test_vendor_targets_exist():
    missing = sorted(t for t in VENDORED_FILES if not (REPO / t).is_file())
    assert not missing, missing
    for directory in VENDORED_DIRS:
        assert (REPO / directory).is_dir(), directory


def test_exemptions_are_current():
    for path in EXEMPT:
        assert (REPO / path).is_file(), path


@pytest.mark.parametrize("path", sorted(files_needing_notice()))
def test_third_party_file_has_license_header(path):
    problems = header_problems((REPO / path).read_text())
    assert not problems, f"{path}: {problems}"


def test_detector():
    assert _claims_third_party('"""Vendored in spirit from X."""\n')
    assert _claims_third_party("# Adapted from Y.\nimport os\n")
    assert not _claims_third_party('"""Episodes are copied from the source."""\n')
    assert header_problems("import os\n")
    assert header_problems("# robosuite\n# Permission is hereby granted\nimport os\n") == [
        f"missing {COPYRIGHTS['robosuite'][0]!r}"
    ]
