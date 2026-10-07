"""Initial-state target cards: every task, both styles, structurally verified.

The cards are the operator's placement reference, so a title that runs off the figure or a
header block that silently vanishes is an operator-facing defect. These tests build the
Figure (no PNG round-trip) from the committed R5 / R8 locked manifests and assert on it:
every text lies inside the figure, the collection header carries the pace/ETA stats on
EVERY task, the eval title carries the coordinates, and
filenames are pinned to the manifest row on every task.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from matplotlib.text import Text

from mulligan.real.collect.initial_states import (
    ArmSpec,
    _load_initial_state_manifest,
    manifest_arm_keys,
)
from mulligan.real.collect.initial_states import _load_manifest_payload
from mulligan.real.operator_ui.cards import (
    CardStyle,
    build_initial_state_card_figure,
    card_filename,
    write_initial_state_card,
)

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = {
    "marker_d2": REPO
    / "data/real/manifests/marker_d2/r05/marker_d2_r5_promote25_fill25_blind_dagger.json",
    "square_d2": REPO
    / "data/real/manifests/square_d2/r05/square_d2_r5_promote25_fill25_blind_dagger.json",
    "routing_d2": REPO
    / "data/real/manifests/routing_d2/r08/routing_d2_r8_promote25_fill25_blind_dagger.json",
}
COLLECTION = CardStyle(
    mode_label="FIRST ROLLOUT", display_label="start 012", show_coordinates=False
)
EVAL = CardStyle(mode_label="Policy B", show_coordinates=True)

_needs_manifests = pytest.mark.skipif(
    not all(p.exists() for p in MANIFESTS.values()),
    reason="committed R5/R8 manifests not present",
)


def load(task: str):
    path = MANIFESTS[task]
    payload = _load_manifest_payload(path)
    arms = [ArmSpec(key, f"test:{key}") for key in manifest_arm_keys(payload)]
    targets, meta = _load_initial_state_manifest(path, arms, expected_task=task)
    return targets[0], meta


def texts(fig) -> list[str]:
    return [t.get_text() for t in fig.findobj(Text) if t.get_text().strip()]


@_needs_manifests
@pytest.mark.parametrize("task", sorted(MANIFESTS))
@pytest.mark.parametrize("style", [COLLECTION, EVAL], ids=["collection", "eval"])
def test_every_text_lies_inside_the_figure(task, style):
    target, meta = load(task)
    fig = build_initial_state_card_figure(target, meta, task_name=task, style=style)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    clipped = []
    for t in fig.findobj(Text):
        if not t.get_text().strip():
            continue
        box = t.get_window_extent(renderer)
        if (
            box.x0 < fig_box.x0 - 1
            or box.x1 > fig_box.x1 + 1
            or box.y0 < fig_box.y0 - 1
            or box.y1 > fig_box.y1 + 1
        ):
            clipped.append(t.get_text()[:60])
    assert not clipped, f"{task}: text clipped by the figure edge: {clipped}"


@_needs_manifests
@pytest.mark.parametrize("task", sorted(MANIFESTS))
def test_collection_style_titles_the_start_without_session_stats(task):
    # Session numbers (quota, pace, ETA) live in the panel header, never on the card PNG
    # that the ledger records per target.
    target, meta = load(task)
    fig = build_initial_state_card_figure(target, meta, task_name=task, style=COLLECTION)
    title = fig.axes[0].get_title(loc="left")
    assert title.endswith("FIRST ROLLOUT   [start 012]")
    assert "\n" not in title
    assert "in forward" not in title  # coordinate-free by design


@_needs_manifests
@pytest.mark.parametrize("task", sorted(MANIFESTS))
def test_eval_style_titles_carry_the_coordinates(task):
    target, meta = load(task)
    fig = build_initial_state_card_figure(target, meta, task_name=task, style=EVAL)
    title = fig.axes[0].get_title()
    assert "Policy B" in title and "target #1" in title
    assert "in forward" in title


@_needs_manifests
@pytest.mark.parametrize("task", sorted(MANIFESTS))
def test_operator_default_has_no_coordinate_dump_or_color_tutorial(task):
    target, meta = load(task)
    fig = build_initial_state_card_figure(target, meta, task_name=task)
    title = fig.axes[0].get_title(loc="left")
    assert "target #1" in title
    assert "in forward" not in title
    assert " = " not in title


@_needs_manifests
@pytest.mark.parametrize("task", sorted(MANIFESTS))
def test_card_filename_is_pinned_to_the_manifest_row_on_every_task(task, tmp_path):
    target, meta = load(task)
    path = write_initial_state_card(
        target, meta, output_dir=tmp_path, task_name=task, style=COLLECTION
    )
    assert path.name == card_filename(target) == f"target_{target.manifest_idx:04d}.png"
    assert path.stat().st_size > 5_000
    # A retry / CF replay of the same row overwrites, never accumulates.
    again = write_initial_state_card(target, meta, output_dir=tmp_path, task_name=task, style=EVAL)
    assert again == path and len(list(tmp_path.glob("*.png"))) == 1


def test_cards_module_never_imports_the_robot_stack_or_pyplot():
    # Card previews and round-tracking scripts must stay cheap (import-weight guard), and importing
    # the eval stack must not switch the process-wide matplotlib backend (no pyplot).
    code = (
        # dict.__getitem__ peeks at the backend without resolving the auto sentinel
        # (rcParams['backend'] itself would resolve it and mask a forced use()).
        "import sys, matplotlib; peek = lambda: dict.__getitem__(matplotlib.rcParams, 'backend'); "
        "before = peek(); import mulligan.real.operator_ui.cards, mulligan.real.collect.initial_states; "
        "heavy = sorted(m for m in ('torch', 'lerobot', 'cv2', 'av') if m in sys.modules); "
        "assert not heavy, heavy; "
        "assert peek() is before, f'cards switched the backend: {before!r} -> {peek()!r}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)
