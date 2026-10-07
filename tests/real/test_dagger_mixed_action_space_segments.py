"""DAgger segments must share one RobotEnv across a position-space policy and a
velocity SpaceMouse correction (routing_d2 UMI-relative collection).

DROID's ``RobotEnv.step`` takes ``action_space`` / ``gripper_action_space`` per call, so
the contract is: every ``env.step`` in both segments passes its spaces explicitly (never
relies on the constructor default), and the gripper hand-off between segments is always
in the SpaceMouse's velocity-sign convention.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mulligan.real.collect.blind_dagger import collection_env_kwargs
from mulligan.real.collect.dagger import (
    gripper_command_to_velocity_sign,
    policy_rollout_segment,
    spacemouse_correction_segment,
)

CAM = "side_1"


class FakeEnv:
    """Records the per-step spaces; produces a DROID-shaped observation."""

    def __init__(self):
        self.step_calls: list[tuple[str | None, str | None]] = []
        self._t = 0
        self.pose = np.array([0.5, 0.0, 0.3, 3.1, 0.0, 0.0])

    def _obs(self) -> dict:
        self._t += 1
        return {
            "image": {CAM: np.zeros((480, 640, 3), dtype=np.uint8)},
            "robot_state": {
                "cartesian_position": self.pose.tolist(),
                "cartesian_velocity": [0.0] * 6,
                "gripper_position": 0.02,
                "joint_positions": [0.0] * 7,
                "joint_velocities": [0.0] * 7,
            },
            "timestamp": {"robot_state": {"read_end": 1000.0 * self._t}},
        }

    def get_observation(self) -> dict:
        return self._obs()

    def step(self, action, action_space=None, gripper_action_space=None):
        self.step_calls.append((action_space, gripper_action_space))
        obs = self._obs()
        obs["action_info"] = {
            "cartesian_velocity": [0.0] * 6,
            "cartesian_position": self.pose.tolist(),
            "joint_velocity": [0.0] * 7,
            "joint_position": [0.0] * 7,
            "gripper_position": float(np.clip(action[-1], 0, 1)),
            "gripper_velocity": 0.0,
        }
        return obs


class FakeUI:
    """The OperatorUI surface the two segment loops touch: keys, monitor, panel phase + step."""

    def __init__(self, keys):
        self._keys = list(keys)
        self.phases: list[str] = []
        self.progress = SimpleNamespace(step=0, marks=0)

    def set_phase(self, phase, detail="", *, controls=()):
        self.phases.append(phase)

    def render_monitor(self, images):
        pass

    def drain_keys(self):
        pass

    def read_key(self):
        return self._keys.pop(0) if self._keys else None


class PositionPolicy:
    """UMI-relative shape: decoded 7D pose, absolute gripper target."""

    action_space = "cartesian_position"
    env_action_space = "cartesian_position"
    gripper_action_space = "position"

    def __init__(self, gripper):
        self._gripper = gripper

    def predict(self, obs):
        return np.array([0.5, 0.0, 0.3, 3.1, 0.0, 0.0, self._gripper], dtype=np.float32)

    def reset(self):
        pass


class VelocityPolicy:
    action_space = "cartesian_velocity"
    gripper_action_space = None

    def predict(self, obs):
        return np.array([0.1, 0, 0, 0, 0, 0, 0.7], dtype=np.float32)

    def reset(self):
        pass


class FakeSpaceMouse:
    pos_sensitivity = 1.0
    rot_sensitivity = 1.0

    def __init__(self):
        self.control = np.array([0.5, 0, 0, 0, 0, 0])
        self.gripper_closed = None

    def start_control(self):
        pass

    def reset_gripper(self):
        self.gripper_closed = False

    @property
    def control_gripper(self):
        return 1 if self.gripper_closed else 0


def _run_policy(policy, keys):
    env = FakeEnv()
    seg, action_str, gripper = policy_rollout_segment(
        env, policy, FakeUI(keys), [CAM], freq=1000.0, initial_gripper_action=None
    )
    return env, seg, action_str, gripper


def test_position_policy_segment_steps_with_its_own_spaces():
    env, seg, action_str, _ = _run_policy(PositionPolicy(0.03), [None, None, "h"])
    assert action_str == "intervention"
    assert len(seg["actions"]) == 2
    assert env.step_calls == [("cartesian_position", "position")] * 2


@pytest.mark.parametrize("gripper, expected", [(0.03, -1.0), (0.97, 1.0)])
def test_position_gripper_handoff_is_velocity_sign(gripper, expected):
    # A slightly-open absolute target (0.03) must NOT read as CLOSED (> 0.0) at the
    # SpaceMouse hand-off -- that would shut the gripper the moment the operator intervenes.
    _, _, _, last_gripper = _run_policy(PositionPolicy(gripper), [None, "h"])
    assert last_gripper == expected


def test_velocity_policy_gripper_handoff_passes_through():
    env, _, _, last_gripper = _run_policy(VelocityPolicy(), [None, "h"])
    assert last_gripper == pytest.approx(0.7)
    assert env.step_calls == [("cartesian_velocity", None)]


def test_spacemouse_segment_steps_velocity_explicitly_and_seeds_gripper():
    env = FakeEnv()
    device = FakeSpaceMouse()
    seg, action_str, _ = spacemouse_correction_segment(
        env,
        device,
        FakeUI([None, None, "h"]),
        [CAM],
        freq=1000.0,
        initial_gripper_action=-1.0,
    )
    assert action_str == "continue"
    assert device.gripper_closed is False
    assert len(seg["actions"]) == 2
    # Never the constructor default: a position-cohort env would otherwise interpret the
    # +-1 SpaceMouse gripper toggle as an absolute position and clip the recorded command.
    assert env.step_calls == [("cartesian_velocity", "velocity")] * 2


def test_gripper_sign_conversion():
    assert gripper_command_to_velocity_sign(0.3, "velocity") == pytest.approx(0.3)
    assert gripper_command_to_velocity_sign(0.5, "position") == -1.0
    assert gripper_command_to_velocity_sign(0.51, "position") == 1.0
    with pytest.raises(ValueError):
        gripper_command_to_velocity_sign(0.5, "bogus")


def test_collection_env_kwargs_accepts_position_cohort():
    assert collection_env_kwargs("cartesian_position", "position") == {
        "action_space": "cartesian_position",
        "gripper_action_space": "position",
    }
    assert collection_env_kwargs("cartesian_velocity", None) == {
        "action_space": "cartesian_velocity"
    }
    with pytest.raises(ValueError):
        collection_env_kwargs("joint_velocity", "position")
