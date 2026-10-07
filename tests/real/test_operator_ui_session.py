"""``OperatorUI``: the object every real entrypoint holds, exercised headless end to end.

Flag resolution (``from_args``), the loud display check at construction, card writing with
the window disabled, the monitor no-op, gate delegation with an injected keyboard, and an
idempotent ``close`` -- everything an entrypoint touches, without a robot or an X server.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mulligan.real.collect.initial_states import (
    ArmSpec,
    _load_initial_state_manifest,
    manifest_arm_keys,
)
from mulligan.real.collect.initial_states import _load_manifest_payload
from mulligan.real.operator_ui.cards import CardStyle
from mulligan.real.operator_ui.cli import add_operator_ui_args
from mulligan.real.operator_ui.gates import GateOutcome
from mulligan.real.operator_ui.session import OperatorUI, card_window_name
from tests.real.test_operator_ui_keys import FakeListener

REPO = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO / "data/real/manifests/routing_d2/r08/routing_d2_r8_promote25_fill25_blind_dagger.json"
)
_needs_manifest = pytest.mark.skipif(not MANIFEST.exists(), reason="R8 manifest not present")


@pytest.fixture(autouse=True)
def headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


def parse(argv: list[str], *, cards: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_operator_ui_args(parser, cards=cards)
    return parser.parse_args(argv)


def test_headless_with_the_card_window_requested_fails_loud_at_construction(tmp_path):
    args = parse([], cards=True)  # window on by default
    with pytest.raises(RuntimeError, match="initial-state card window needs an X11/Wayland"):
        OperatorUI.from_args(args, cards=True, default_card_dir=tmp_path)


def test_headless_monitor_request_fails_loud_at_construction():
    args = parse(["--monitor-cameras"], cards=False)
    with pytest.raises(RuntimeError, match="--monitor-cameras needs an X11/Wayland display"):
        OperatorUI.from_args(args, cards=False)


def test_from_args_resolves_card_dir_and_monitor_keys(tmp_path):
    args = parse(["--no-show-initial-state-window"], cards=True)
    ui = OperatorUI.from_args(args, cards=True, default_card_dir=tmp_path / "cards")
    assert ui.card_dir == tmp_path / "cards" and not ui.show_card_window
    assert ui.monitor_camera_keys is None

    explicit = parse(
        [
            "--no-show-initial-state-window",
            "--initial-state-visualization-dir",
            str(tmp_path / "x"),
        ],
        cards=True,
    )
    assert OperatorUI.from_args(explicit, cards=True, default_card_dir=None).card_dir == (
        tmp_path / "x"
    )
    no_cards = OperatorUI.from_args(parse([], cards=False), cards=False)
    assert no_cards.card_dir is None and not no_cards.show_card_window


def test_show_card_without_a_card_dir_fails_loud():
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    with pytest.raises(RuntimeError, match="needs a card_dir"):
        ui.show_card(object(), {}, task_name="routing_d2")  # type: ignore[arg-type]


@_needs_manifest
def test_show_card_writes_the_png_and_prints_the_path(tmp_path, capsys):
    payload = _load_manifest_payload(MANIFEST)
    arms = [ArmSpec(k, f"t:{k}") for k in manifest_arm_keys(payload)]
    targets, meta = _load_initial_state_manifest(MANIFEST, arms, expected_task="routing_d2")
    ui = OperatorUI(show_card_window=False, card_dir=tmp_path, monitor_camera_keys=None)
    path = ui.show_card(
        targets[0],
        meta,
        task_name="routing_d2",
        style=CardStyle(mode_label="Policy A"),
        message="Next target shown.",
    )
    assert path == tmp_path / "target_0000.png" and path.stat().st_size > 5_000
    out = capsys.readouterr().out
    assert "Next target shown." in out and str(path) in out
    assert card_window_name("routing_d2") == "routing_d2 initial state"


def test_render_monitor_is_a_no_op_when_the_monitor_is_off():
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui.render_monitor({})  # must not touch HighGUI or raise


def test_gate_uses_the_injected_keyboard_and_close_is_idempotent(capsys):
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui._keyboard = FakeListener(buffered=["\n"], fresh=["k"])
    assert ui.gate(prompt="Place it", can_skip=True) is GateOutcome.SKIP
    assert ui._keyboard.flushes == 1  # the buffered Enter was drained, not read
    assert "Place it, then press Enter to start" in capsys.readouterr().out
    ui.close()
    assert ui._keyboard is None
    ui.close()  # second close: no keyboard, no windows, no error


def test_read_and_drain_delegate_to_the_keyboard():
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui._keyboard = FakeListener(buffered=["Q"], fresh=["R"])
    ui.drain_keys()
    assert ui.read_key() == "r"  # lowercased, buffered 'Q' gone
    assert ui.read_key() is None


@_needs_manifest
def test_repainting_session_state_does_not_rerender_the_same_diagram(tmp_path, monkeypatch):
    from mulligan.real.operator_ui import session
    from mulligan.real.operator_ui.progress import EvalScene
    from mulligan.real.collect.initial_states import load_manifest_targets

    targets, meta = load_manifest_targets(
        MANIFEST, expected_task="routing_d2", model_id=lambda key: key
    )
    ui = OperatorUI(show_card_window=False, card_dir=tmp_path, monitor_camera_keys=None)
    original = session.write_initial_state_card
    rendered = []

    def write(target, *args, **kwargs):
        rendered.append(target.manifest_idx)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(session, "write_initial_state_card", write)
    for label in ("A", "B"):
        ui.show_eval_scene(EvalScene(1, 50, label, 1, 2, targets[0]), meta, task_name="routing_d2")
    ui.show_eval_scene(EvalScene(2, 50, "A", 1, 2, targets[1]), meta, task_name="routing_d2")
    assert rendered == [targets[0].manifest_idx, targets[1].manifest_idx]


def test_choose_sets_the_phase_and_delegates_to_operator_choice(capsys):
    from mulligan.real.operator_ui.progress import CollectionStatus

    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui._keyboard = FakeListener(buffered=["q"], fresh=["x"])
    ui.configure_progress(total=None, completed=0)
    ui.update_status(CollectionStatus(metrics=(("SUCCESSES", "3 / 20"),), fraction=0.15))
    choice = ui.choose(
        prompt="Episode saved",
        choices={"c": "replay", "n": "move on", "q": "quit"},
        default="n",
    )
    assert choice == "n"  # the buffered 'q' was drained; 'x' falls to the default
    assert ui.progress.phase == "Choose what happens next"
    assert ui.progress.detail == "Episode saved"
    assert ui.progress.controls == (
        "'c'/numpad'.' replay",
        "'n'/numpad'2' move on",
        "'q'/numpad'/' quit",
    )
    assert ui.progress.started is not None  # the session clock runs from the first decision
    assert "Episode saved: press" in capsys.readouterr().out


@_needs_manifest
def test_show_collection_scene_returns_the_card_path_and_sets_the_scene(tmp_path):
    from mulligan.real.collect.initial_states import load_manifest_targets
    from mulligan.real.operator_ui.progress import CollectionScene

    targets, meta = load_manifest_targets(
        MANIFEST, expected_task="routing_d2", model_id=lambda key: key
    )
    ui = OperatorUI(show_card_window=False, card_dir=tmp_path, monitor_camera_keys=None)
    ui.configure_progress(total=None, completed=0)
    scene = CollectionScene("start 001", "First rollout", 1, target=targets[0])
    path = ui.show_collection_scene(
        scene,
        meta,
        task_name="routing_d2",
        style=CardStyle(mode_label="FIRST ROLLOUT", display_label="start 001"),
    )
    assert path == tmp_path / f"target_{targets[0].manifest_idx:04d}.png" and path.exists()
    assert ui.progress.scene is scene
    bare = CollectionScene("target 002", "First rollout", 2)
    assert ui.show_collection_scene(bare, meta, task_name="routing_d2", style=CardStyle()) is None
    assert ui.progress.scene is bare


def test_begin_and_end_rollout_name_who_is_in_control():
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui.begin_rollout(0, 1, phase="Human correction", controls=("h back to policy",))
    assert ui.progress.phase == "Human correction"
    assert ui.progress.controls == ("h back to policy",)
    assert ui.progress.rollout_started is not None and ui.progress.max_marks == 1
    ui.begin_rollout(850, 1)
    assert ui.progress.phase == "Policy running"
    assert "'g'/numpad'3' subgoal" in ui.progress.controls[1]
    ui.end_rollout("discard", phase="Episode not saved", detail="Episode 3: DISCARD")
    assert ui.progress.rollout_started is None
    assert (ui.progress.phase, ui.progress.detail) == ("Episode not saved", "Episode 3: DISCARD")
    assert ui.progress.completed == 0
