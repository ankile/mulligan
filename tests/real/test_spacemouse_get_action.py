"""Characterization + behavioral tests for the SpaceMouse ``get_action`` helper.

Context
-------
``mulligan.real.collect.teleop`` imports the canonical ``get_action`` (SpaceMouse state ->
7D Franka action) from ``mulligan.real.collect.dagger``.

These modules pull in real robot hardware deps (``droid``) and cannot be
imported in CI, so we parse/exec the function *source* via ``ast`` rather than
importing it. The behavioral test pins the exact 7-vector math: the x/y flip,
the roll<->pitch swap, per-axis sensitivity scaling, [-1, 1] clipping, and the
gripper 0/1 -> -1/+1 mapping.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TELEOP_PATH = ROOT / "mulligan" / "real" / "collect" / "teleop.py"
DAGGER_PATH = ROOT / "mulligan" / "real" / "collect" / "dagger.py"


def _module_source(path: Path) -> str:
    return path.read_text()


def _find_get_action(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_action":
            return node
    raise AssertionError("get_action FunctionDef not found")


def _strip_docstring(fn: ast.FunctionDef) -> ast.FunctionDef:
    fn = copy.deepcopy(fn)
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        fn.body = fn.body[1:]
    return fn


def _extract_and_compile_get_action(source: str):
    """Exec the standalone get_action source in a numpy-only namespace."""
    fn_node = _find_get_action(source)
    fn_src = ast.get_source_segment(source, fn_node)
    assert fn_src is not None
    namespace: dict = {"np": np}
    exec(compile(fn_src, "<get_action>", "exec"), namespace)  # noqa: S102
    return namespace["get_action"]


class _FakeSpaceMouse:
    """Minimal stand-in exposing exactly the attributes get_action reads."""

    def __init__(self, control, pos_sensitivity, rot_sensitivity, control_gripper):
        self.control = np.asarray(control, dtype=float)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.control_gripper = control_gripper


# --------------------------------------------------------------------------- #
# teleop does not define get_action locally; it imports the canonical dagger
# implementation.
# --------------------------------------------------------------------------- #


def test_teleop_imports_canonical_get_action():
    teleop_src = _module_source(TELEOP_PATH)
    # No local definition remains.
    teleop_tree = ast.parse(teleop_src)
    local_defs = [
        n
        for n in ast.walk(teleop_tree)
        if isinstance(n, ast.FunctionDef) and n.name == "get_action"
    ]
    assert local_defs == [], "teleop must not define get_action locally"
    # The import line is present.
    assert "from mulligan.real.collect.dagger import get_action" in teleop_src
    # The canonical definition still lives in dagger.
    _find_get_action(_module_source(DAGGER_PATH))


# --------------------------------------------------------------------------- #
# Behavioral: pin the exact 7-vector math against hand-computed expectations.
# The canonical implementation lives in mulligan/real/collect/dagger.py.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "control, pos_s, rot_s, gripper, expected",
    [
        # x/y flip, z passthrough, roll<->pitch swap (action[3]=-control[4],
        # action[4]=-control[3]), yaw flip, unit sensitivity, gripper closed.
        (
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            1.0,
            1.0,
            1,
            [-0.1, -0.2, 0.3, -0.5, -0.4, -0.6, 1.0],
        ),
        # Clipping to [-1, 1] on translation; gripper open -> -1.0.
        (
            [2.0, -3.0, 0.5, 0.1, 0.2, 0.3],
            1.0,
            1.0,
            0,
            [-1.0, 1.0, 0.5, -0.2, -0.1, -0.3, -1.0],
        ),
        # Sensitivity scaling then clipping; rot scaled by 0.5.
        (
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            3.0,
            0.5,
            1,
            [-1.0, -1.0, 1.0, -0.25, -0.25, -0.25, 1.0],
        ),
    ],
)
def test_get_action_behavior(control, pos_s, rot_s, gripper, expected):
    get_action = _extract_and_compile_get_action(_module_source(DAGGER_PATH))
    device = _FakeSpaceMouse(control, pos_s, rot_s, gripper)
    action = get_action(device)
    assert action.shape == (7,)
    np.testing.assert_allclose(action, expected, rtol=0, atol=1e-12)


def test_get_action_gripper_nonclosed_maps_to_negative_one():
    """Any gripper value != 1 maps to -1.0 (only ==1 -> +1.0)."""
    get_action = _extract_and_compile_get_action(_module_source(DAGGER_PATH))
    for g in (0, -1, 2):
        device = _FakeSpaceMouse([0, 0, 0, 0, 0, 0], 1.0, 1.0, g)
        assert get_action(device)[6] == -1.0
    device = _FakeSpaceMouse([0, 0, 0, 0, 0, 0], 1.0, 1.0, 1)
    assert get_action(device)[6] == 1.0
