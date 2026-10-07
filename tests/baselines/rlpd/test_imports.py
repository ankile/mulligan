"""The RLPD code needs no legacy stack: importing it (and building its env) loads no gym,
robomimic, d4rl or ml_collections, and its own modules import neither those nor cv2 directly
(robosuite itself imports cv2)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FORBIDDEN = ("gym", "robomimic", "d4rl", "dmcgym", "mujoco_py", "ml_collections")
PORT_FILES = sorted((REPO / "mulligan" / "baselines" / "rlpd").rglob("*.py"))

PROBE = f"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")
import mulligan.baselines.rlpd.agent, mulligan.baselines.rlpd.train
import mulligan.baselines.rlpd.evaluation, mulligan.baselines.rlpd.datasets
from mulligan.baselines.rlpd.env import RLPDEnv
from mulligan.baselines.rlpd.datasets import PH_ENV_META
RLPDEnv(PH_ENV_META).close()
bad = sorted({{m.split(".")[0] for m in sys.modules}} & set({FORBIDDEN!r}))
print("FORBIDDEN", bad)
"""


def test_import_closure_has_no_legacy_stack():
    pytest.importorskip("jax")
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=REPO, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "FORBIDDEN []" in proc.stdout, proc.stdout


@pytest.mark.parametrize("path", PORT_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_direct_imports(path):
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    assert not names & (set(FORBIDDEN) | {"cv2"}), names
