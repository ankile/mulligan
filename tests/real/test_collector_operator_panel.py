"""The blind collector's operator panel: quota / pace cells per collection mode and the
sub-goal key inside the DAgger segment loops, without a robot or a display."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from mulligan.real.operator_ui.progress import CollectionStatus
from mulligan.real.operator_ui.session import OperatorUI
from tests.sim.test_adaptive_quota import _write_manifest
from tests.real.test_operator_ui_keys import FakeListener


def test_protocol_quota_panel_status_reports_quota_pace_and_eta(tmp_path):
    from mulligan.sim.collect.quota import ProtocolQuotaLedger

    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=tmp_path / "ledger.jsonl",
    )
    status = CollectionStatus(**q.operator_panel_status())
    # ledger protocol order: no_cf first, then with_cf
    assert [label for label, _ in status.metrics] == [
        "NO-CF QUOTA",
        "WITH-CF QUOTA",
        "EST. TIME LEFT",
    ]
    assert status.metrics[0][1] == "0 / 4" and status.metrics[1][1] == "0 / 4"
    assert status.fraction == 0.0 and status.summary.startswith("Credited 0   |   gathering")

    shared_idx, _ = q.match([2.0, 0.0, 0.0])
    q.append_reserved_row(
        q.credit_episode(
            manifest_idx=shared_idx,
            episode_index=0,
            success=True,
            is_counterfactual=False,
            matched_distance=0.0,
            write_ledger=False,
        )
    )
    status = CollectionStatus(**q.operator_panel_status())
    # the shared start credits both arms in both protocols: 2 of 4 units each
    assert status.metrics[0][1] == "2 / 4" and status.metrics[1][1] == "2 / 4"
    assert status.fraction == pytest.approx(4 / 8)
    assert status.metrics[2][0] == "EST. TIME LEFT" and status.metrics[2][1] != "unknown"
    assert status.summary.startswith("Credited 1   |   ") and status.summary.endswith(" left")


def test_collection_status_per_arm_mode():
    from mulligan.real.collect.blind_dagger import _collection_status

    per_arm = _collection_status(
        protocol_quota=None,
        success_counts=Counter({"a": 3, "b": 12}),
        target_success_per_arm=10,
        session_successes=5,
        elapsed_s=600.0,
    )
    # b is capped at its target; 7 remaining at 120 s per success this session
    assert per_arm.metrics == (
        ("SUCCESSES", "13 / 20"),
        ("REMAINING", "7"),
        ("EST. TIME LEFT", "14m 00s"),
    )
    assert per_arm.fraction == 13 / 20 and per_arm.summary == "5 successes this session"
    fresh = _collection_status(
        protocol_quota=None,
        success_counts=Counter({"a": 0}),
        target_success_per_arm=10,
        session_successes=0,
        elapsed_s=0.0,
    )
    assert fresh.metrics[2] == ("EST. TIME LEFT", "Estimating...")
    with pytest.raises(ValueError, match="target_success_per_arm"):
        _collection_status(
            protocol_quota=None,
            success_counts=Counter(),
            target_success_per_arm=None,
            session_successes=0,
            elapsed_s=0.0,
        )


@pytest.fixture
def segment_env(monkeypatch):
    from mulligan.real.collect import dagger as module

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    def observation():
        return {
            "robot_state": {
                "cartesian_position": np.zeros(6),
                "gripper_position": 0.0,
                "joint_positions": np.zeros(7),
            },
            "image": {"camera": np.zeros((2, 2, 3), dtype=np.uint8)},
        }

    env = SimpleNamespace(
        get_observation=observation, step=lambda *a, **kw: {**observation(), "action_info": {}}
    )
    policy = SimpleNamespace(predict=lambda obs: np.zeros(7), action_space="cartesian_velocity")
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
    return module, env, policy


def test_policy_segment_routes_the_subgoal_key_and_counts_steps(segment_env):
    module, env, policy = segment_env
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui._keyboard = FakeListener(buffered=["g"], fresh=[None, "g", None, "1"])
    marks = []

    def on_subgoal() -> None:
        marks.append(ui.progress.step)  # frames recorded when the key was read

    ui._keyboard = FakeListener(buffered=["g"], fresh=[None, "g", None, "1"])
    ui.begin_rollout(0, 1)
    data, outcome, _ = module.policy_rollout_segment(
        env, policy, ui, ["camera"], freq=10_000, initial_gripper_action=None, on_subgoal=on_subgoal
    )
    assert outcome == "success" and len(data["actions"]) == 3
    assert marks == [1]  # the buffered 'g' was drained; the live one landed after frame 0
    assert ui.progress.step == 3 and ui.progress.phase == "Policy running"
    assert "'g'/numpad'3' subgoal" in ui.progress.controls[1]


def test_segments_ignore_the_subgoal_key_without_a_callback(segment_env, monkeypatch):
    module, env, policy = segment_env
    ui = OperatorUI(show_card_window=False, card_dir=None, monitor_camera_keys=None)
    ui._keyboard = FakeListener(fresh=["g", "9"])
    data, outcome, _ = module.policy_rollout_segment(
        env, policy, ui, ["camera"], freq=10_000, initial_gripper_action=None
    )
    assert outcome == "failure" and len(data["actions"]) == 1
    assert "subgoal" not in " ".join(ui.progress.controls)

    class Device:
        control = np.zeros(6)
        control_gripper = -1.0

        def start_control(self):
            pass

        def reset_gripper(self):
            pass

    ui._keyboard = FakeListener(fresh=["g", "d"])
    monkeypatch.setattr(module, "get_action", lambda device: np.zeros(7))
    data, outcome, _ = module.spacemouse_correction_segment(
        env, Device(), ui, ["camera"], freq=10_000, initial_gripper_action=None
    )
    assert outcome == "discard" and ui.progress.phase == "Human correction"
