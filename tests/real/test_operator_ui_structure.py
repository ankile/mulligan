"""Structural guards over the real-robot entrypoints' operator UI.

These parse source (importing the entrypoints pulls in lerobot/torch and costs minutes) and
pin the invariants that only ever broke on the robot, not in CI:

1. HighGUI is prewarmed before lerobot/av are imported (the Qt xcb deadlock); the
   parse-first entrypoints parse their arguments before that, and their ``--help`` (run in a
   subprocess) loads no torch/lerobot/av.
2. No entrypoint re-implements key reading, display checks, or windows;
   ``mulligan.real.operator_ui`` is the single owner.
3. Every operator key-polling loop is preceded by a drain, so a keystroke buffered during a
   reset can never be read as the next decision.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "mulligan" / "real"
# Every module that builds an OperatorUI from flags is an operator entrypoint; derived, so a
# new eval or collector is fenced the moment it adopts the package.
ENTRYPOINTS = sorted(
    str(path.relative_to(REAL))
    for path in REAL.rglob("*.py")
    if "OperatorUI.from_args(" in path.read_text()
)
# Every mulligan.real module outside the package must leave HighGUI to operator_ui. The one
# standalone diagnostic viewer keeps its own loop deliberately (no robot, no gates).
HIGHGUI_FENCED = sorted(
    path
    for path in REAL.rglob("*.py")
    if "operator_ui" not in path.parts and path.name != "view_cameras.py"
)
# Anything that transitively loads lerobot/av/torch: the libraries themselves and every
# mulligan module except the torch-free ones below (operator_ui, initial_states, and what
# the parse-first entrypoints' parsers import; test_help_exits_before_the_robot_stack
# checks that these stay light).
HEAVY_PREFIXES = ("lerobot", "torch", "av", "droid", "mulligan.")
LIGHT_PREFIXES = (
    "mulligan.real.operator_ui",
    "mulligan.real.collect.initial_states",
    "mulligan.real.collect.hf_utils",
    "mulligan.real.eval.inference_server",
    "mulligan.real.lifecycle.tasks",
    "mulligan.real.policy.dp",
    "mulligan.real.robot.cameras",
    "mulligan.real.robot.cli",
    "mulligan.sim.collect.quota",
)
# Entrypoints that parse their arguments before the prewarm, so --help and argument errors
# exit without the robot stack.
PARSE_FIRST = ["collect/blind_dagger.py", "eval/manifest_eval.py"]
FORBIDDEN_DEFS = {
    "read_opencv_key",
    "_read_opencv_key",
    "read_operator_key",
    "_read_operator_key",
    "_has_display",
    "_probe_x11_display",
    "_show_initial_state_opencv_window",
    "wait_for_keypress",
    "drain_operator_keys",
    "TerminalKeyboardListener",
}
FORBIDDEN_CV2_CALLS = {
    "waitKey",
    "waitKeyEx",
    "pollKey",
    "imshow",
    "namedWindow",
    "resizeWindow",
    "moveWindow",
    "destroyWindow",
    "destroyAllWindows",
    "setWindowProperty",
    "setMouseCallback",
}
KEY_READS = {"read_key", "read_operator_key"}
DRAIN_CALLS = {"drain_keys", "drain_operator_keys"}


def _tree(name: str) -> ast.Module:
    return ast.parse((REAL / name).read_text(), filename=name)


def _module_of(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    return []


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_highgui_prewarm_precedes_the_robot_stack(name):
    tree = _tree(name)
    prewarm_import_idx = prewarm_call_idx = first_heavy_idx = None
    for idx, node in enumerate(tree.body):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "mulligan.real.operator_ui.display"
            and any(a.name == "prewarm_highgui" for a in node.names)
        ):
            prewarm_import_idx = idx
        if isinstance(node, ast.If) and any(
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and getattr(s.value.func, "id", None) == "prewarm_highgui"
            for s in node.body
        ):
            prewarm_call_idx = idx
        if first_heavy_idx is None and any(
            m.startswith(HEAVY_PREFIXES) and not m.startswith(LIGHT_PREFIXES)
            for m in _module_of(node)
        ):
            first_heavy_idx = idx
    assert prewarm_import_idx is not None, f"{name}: no prewarm_highgui import"
    assert prewarm_call_idx is not None, (
        f"{name}: no `if __name__ == '__main__': prewarm_highgui()`"
    )
    assert first_heavy_idx is not None, f"{name}: expected a lerobot/torch import"
    assert prewarm_import_idx < prewarm_call_idx < first_heavy_idx, (
        f"{name}: prewarm must run before the first heavy import (statement {first_heavy_idx})"
    )


_HELP_PROBE = """
import runpy, sys
module = sys.argv[1]
sys.argv = [module, "--help"]
try:
    runpy.run_module(module, run_name="__main__", alter_sys=True)
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
print("HEAVY=" + ",".join(m for m in ("torch", "lerobot", "av") if m in sys.modules))
"""


@pytest.mark.parametrize("name", PARSE_FIRST)
def test_help_exits_before_the_robot_stack(name):
    module = "mulligan.real." + name.removesuffix(".py").replace("/", ".")
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", _HELP_PROBE, module],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert result.stdout.splitlines()[-1] == "HEAVY=", result.stdout.splitlines()[-1]


@pytest.mark.parametrize("name", PARSE_FIRST)
def test_parse_first_entrypoints_parse_before_the_prewarm(name):
    tree = _tree(name)
    main_blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and getattr(node.test.left, "id", None) == "__name__"
    ]
    assert len(main_blocks) == 2, f"{name}: expected a prewarm block and a main() block"
    prewarm_block, main_block = main_blocks
    calls = [ast.unparse(stmt) for stmt in prewarm_block.body]
    assert calls == ["_CLI_ARGS = parse_args()", "prewarm_highgui()"], calls
    assert [ast.unparse(stmt) for stmt in main_block.body] == ["main(_CLI_ARGS)"]


def test_the_entrypoint_set_is_the_five_real_tools():
    assert ENTRYPOINTS == [
        "collect/blind_dagger.py",
        "collect/dagger.py",
        "collect/rollout.py",
        "collect/teleop.py",
        "eval/manifest_eval.py",
    ]


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_entrypoints_own_no_private_key_or_display_helpers(name):
    tree = _tree(name)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in FORBIDDEN_DEFS
    }
    assert not defined, f"{name} re-implements operator_ui helpers: {sorted(defined)}"


def _cv2_aliases(tree: ast.Module) -> set[str]:
    """Names the module binds to the cv2 package (``import cv2``, ``import cv2 as cv``)."""
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "cv2"
    }


@pytest.mark.parametrize("path", HIGHGUI_FENCED, ids=lambda p: str(p.relative_to(REAL)))
def test_only_operator_ui_drives_highgui(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    aliases = _cv2_aliases(tree)
    cv2_calls = sorted(
        f"{node.func.attr}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in aliases
        and node.func.attr in FORBIDDEN_CV2_CALLS
    )
    assert not cv2_calls, f"{path.name} drives HighGUI directly (use OperatorUI): {cv2_calls}"


def _reads_a_key(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Assign):
        return False
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            func = node.func
            attr = getattr(func, "attr", None) or getattr(func, "id", None)
            if attr in KEY_READS:
                return True
    return False


def _drains(stmt: ast.stmt) -> bool:
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call):
            func = node.func
            attr = getattr(func, "attr", None) or getattr(func, "id", None)
            if attr in DRAIN_CALLS:
                return True
    return False


def _key_polling_loops(tree: ast.Module):
    """``(loop, enclosing block)`` for every while-loop that reads an operator key directly."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for stmt in block:
                if isinstance(stmt, ast.While) and any(_reads_a_key(s) for s in stmt.body):
                    yield stmt, block


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_every_key_polling_loop_is_preceded_by_a_drain(name):
    tree = _tree(name)
    undrained = []
    for loop, block in _key_polling_loops(tree):
        preceding = block[: block.index(loop)]
        if not any(_drains(s) for s in preceding):
            undrained.append(loop.lineno)
    assert not undrained, (
        f"{name} line(s) {undrained}: a loop reads operator keys without a preceding "
        "ui.drain_keys() / drain_operator_keys() in the same block; a keystroke buffered "
        "during the previous reset would be consumed as this loop's decision."
    )


def test_the_collector_still_has_its_operator_prompts():
    # Every collector decision goes through the shared UI: placement through ui.gate, the
    # multi-choice prompts (resume-CF, discard/retry, CF/next, next-episode) through
    # ui.choose. No prompt polls keys by hand any more; dropping one is a bug.
    tree = _tree("collect/blind_dagger.py")
    assert list(_key_polling_loops(tree)) == []

    def ui_calls(attr: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ui"
        ]

    assert len(ui_calls("gate")) == 1
    assert len(ui_calls("choose")) == 4
    for call in ui_calls("choose"):
        assert any(kw.arg == "default" for kw in call.keywords), call.lineno


@pytest.mark.parametrize("name", ["eval/manifest_eval.py"])
def test_every_operational_eval_rollout_wires_the_pre_reset_preview(name):
    calls = [
        node
        for node in ast.walk(_tree(name))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "rollout_episode"
    ]
    assert calls
    for call in calls:
        assert any(kw.arg == "pre_reset_callback" for kw in call.keywords), (
            f"{name}:{call.lineno}: next helper would appear only after robot reset"
        )
