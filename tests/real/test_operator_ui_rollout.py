"""Execute the real rollout loop against a fake robot; pin terminal/preview/reset order."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mulligan.real.operator_ui.progress import EvalScene
from mulligan.real.operator_ui.session import OperatorUI
from tests.real.test_operator_ui_keys import FakeListener


@pytest.fixture
def rollout(monkeypatch):
    from mulligan.real.collect import rollout as module

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    events = []

    def observation():
        return {
            "robot_state": {
                "cartesian_position": np.zeros(6),
                "gripper_position": 0.0,
                "joint_positions": np.zeros(7),
            },
            "image": {"camera": np.zeros((2, 2, 3), dtype=np.uint8)},
        }

    def step(*_args, **_kwargs):
        events.append("step")
        return {**observation(), "action_info": {}}

    env = SimpleNamespace(get_observation=observation, step=step)
    policy = SimpleNamespace(
        predict=lambda obs: np.zeros(7), reset=lambda: None, action_space="cartesian_velocity"
    )
    monkeypatch.setattr(module, "policy_live_camera_keys", lambda p, keys: keys)
    monkeypatch.setattr(module, "verified_reset", lambda *a, **kw: events.append("reset"))
    monkeypatch.setattr(
        module, "_refresh_obs_until_robot_state_timestamp_advances", lambda env, **kw: kw["obs"]
    )
    monkeypatch.setattr(module, "process_image", lambda img, *_: img)
    monkeypatch.setattr(module, "build_canonical_action", lambda info: np.zeros(7))
    for name in (
        "warn_missing_franka_telemetry_once",
        "init_supplementary_lists",
        "append_action_info",
        "append_joint_velocities",
        "append_franka_telemetry",
        "append_cartesian_velocities",
    ):
        monkeypatch.setattr(module, name, lambda *a, **kw: None)

    def terminal(data, obs, **kw):
        assert len(data["observations"]) == 2  # final camera/state frame already captured
        events.append("terminal")

    monkeypatch.setattr(module, "finalize_episode_data", terminal)
    return module, env, policy, events


@pytest.mark.parametrize(
    "outcome,key",
    [("success", "1"), ("failure", "9"), ("timeout", "0"), ("restart", "r"), ("quit", "q")],
)
def test_actual_rollout_previews_after_terminal_capture_and_before_reset(rollout, outcome, key):
    module, env, policy, events = rollout
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui.configure_progress(total=2, completed=0)
    ui._keyboard = FakeListener(fresh=[None, key])

    def preview(result):
        assert result == outcome
        events.append("preview")

    module.rollout_episode(
        env,
        policy,
        ui,
        max_steps=3,
        freq=100000,
        save_data=True,
        all_camera_keys=["camera"],
        print_timing_summary=False,
        pre_reset_callback=preview,
    )
    expected = ["step"]
    if outcome not in {"restart", "quit"}:
        expected += ["terminal"]
    assert events == expected + ["preview", "reset"]
    assert ui.progress.completed == (0 if outcome in {"restart", "quit"} else 1)


def test_failed_preview_still_resets_and_aborts_instead_of_using_stale_helper(rollout):
    module, env, policy, events = rollout
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui._keyboard = FakeListener()

    def broken_preview(outcome):
        events.append("preview failed")
        raise RuntimeError("Cannot paint next helper")

    with pytest.raises(RuntimeError, match="Cannot paint next helper"):
        module.rollout_episode(
            env,
            policy,
            ui,
            max_steps=1,
            freq=100000,
            save_data=True,
            all_camera_keys=["camera"],
            print_timing_summary=False,
            pre_reset_callback=broken_preview,
        )
    assert events == ["step", "terminal", "preview failed", "reset"]


def test_same_target_next_policy_then_next_round_are_selected_before_motion(rollout, monkeypatch):
    module, env, policy, events = rollout
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    scenes = [EvalScene(1, 2, "A", 1, 2), EvalScene(1, 2, "B", 2, 2), EvalScene(2, 2, "A", 1, 2)]
    monkeypatch.setattr(
        ui,
        "show_eval_scene",
        lambda scene, *a, **kw: events.append(("shown", scene.round_num, scene.label)),
    )
    for current, upcoming in zip(scenes, scenes[1:]):
        ui._keyboard = FakeListener()
        module.rollout_episode(
            env,
            policy,
            ui,
            max_steps=1,
            freq=100000,
            save_data=True,
            all_camera_keys=["camera"],
            print_timing_summary=False,
            pre_reset_callback=lambda outcome: ui.preview_after_rollout(
                outcome,
                current=current,
                upcoming=upcoming,
                manifest_meta={},
                task_name="routing_d2",
            ),
        )
    assert events == [
        "step",
        "terminal",
        ("shown", 1, "B"),
        "reset",
        "step",
        "terminal",
        ("shown", 2, "A"),
        "reset",
    ]
