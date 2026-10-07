"""Human-input plumbing shared by the sim collectors.

The DAgger collector (:mod:`mulligan.sim.collect.dagger`) reads everything that
comes from the person at the station through one :class:`Operator` object:

* ``begin_episode(start)`` once per episode, after the env is reset and the
  start state is placed and before the first control step;
* ``poll_key(phase)`` once per control step, before the action is chosen, in
  both the policy phase (``"policy"``) and the human phase (``"human"``);
* ``begin_human_segment(gripper_action)`` when the human takes control;
* ``human_input()`` once per control step of the human phase (after
  ``poll_key("human")`` returned ``None``);
* ``choose(options)`` for the blocking prompts between episodes;
* ``close()`` at shutdown.

Key events (single characters, the same keys the operator presses):

==========  =================================================================
phase       keys
==========  =================================================================
``policy``  ``h`` take over; ``0`` save as recoverable failure (done=False);
            ``9`` save as terminal failure (done=True); ``d`` discard
``human``   ``h`` hand control back to the policy; ``1`` save as success;
            ``0`` / ``9`` / ``d`` as above
choose      ``n`` next start; ``c`` counterfactual replay of the same start
            (human drives first); ``q`` end the session
==========  =================================================================

With ``--auto-save-on-success`` an episode also ends, as a success, when the
env reports success. :class:`SpaceMouseKeyboardOperator` is the interactive
implementation; a scripted operator (for example a replay of recorded human
segments) implements the same methods.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Literal, Optional, Protocol, runtime_checkable

import numpy as np

from mulligan.sim.envs import unwrap_env

Phase = Literal["policy", "human"]

# SpaceMouse input thresholds on the raw [-1, 1] device axes. Frames of a human
# segment are recorded once an input exceeds STRONG (drift guard) and then
# while it exceeds ACTIVE, the gripper moves, or an object moves.
SPACEMOUSE_ACTIVE_THRESHOLD = 0.025
SPACEMOUSE_STRONG_THRESHOLD = 0.1


@dataclass(frozen=True)
class EpisodeStart:
    """What the collector tells the operator at the start of an episode.

    Attributes:
        episode_index: dataset episode index the episode gets if it is saved.
        start_state: the placed start state from the sampler: ``(x, y, yaw)``
            of the nut for Square-Narrow (``NutAssemblySquare``),
            ``((x, y, yaw), (peg_x, peg_y))`` for Square-Broad (``Square_D1``);
            ``None`` without ``--sampler`` (env reset only).
        manifest_idx: row of the quota manifest matched to ``start_state``
            (``None`` without a quota manifest).
        is_counterfactual: a ``c`` replay of the previous start state.
        human_first: the human drives first (counterfactual replays);
            the policy phase starts after ``h``.
    """

    episode_index: int
    start_state: Optional[tuple]
    manifest_idx: Optional[int]
    is_counterfactual: bool
    human_first: bool


@dataclass(frozen=True)
class HumanInput:
    """One control step of human input.

    Attributes:
        action: 7-D env action ``[dx, dy, dz, drot_x, drot_y, drot_z, gripper]``,
            arm entries in [-1, 1], gripper in {-1.0, 1.0}.
        arm_active: arm input above the recording threshold.
        strong: clear intentional arm input; the first such step (or the first
            gripper toggle) starts recording in a human segment.
    """

    action: np.ndarray
    arm_active: bool
    strong: bool


@runtime_checkable
class Operator(Protocol):
    """Source of human decisions and actions for the DAgger collector."""

    def begin_episode(self, start: EpisodeStart) -> None: ...

    def poll_key(self, phase: Phase) -> Optional[str]: ...

    def begin_human_segment(self, gripper_action: float) -> None: ...

    def human_input(self) -> HumanInput: ...

    def choose(self, options: str) -> str: ...

    def close(self) -> None: ...


class KeyboardListener:
    """System-wide keyboard listener that works regardless of window focus.

    Needs a display (pynput); construct it only for interactive sessions.
    """

    def __init__(self):
        from pynput import keyboard

        self.last_key = None
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def _on_press(self, key):
        if hasattr(key, "char") and key.char:
            self.last_key = key.char

    def read_key(self):
        """Return the last pressed key and clear it."""
        key = self.last_key
        self.last_key = None
        return key

    def close(self):
        self.listener.stop()


class SpaceMouseKeyboardOperator:
    """Interactive operator: a 3Dconnexion SpaceMouse plus the keyboard."""

    def __init__(self, pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.5):
        from mulligan.teleop.spacemouse import RobosuiteSpaceMouse

        print("Initializing SpaceMouse...")
        self.device = RobosuiteSpaceMouse(
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
        )
        print("Initializing keyboard listener...")
        self.keyboard = KeyboardListener()

    def begin_episode(self, start: EpisodeStart) -> None:
        # The person at the station sees the placed start in the viewer.
        pass

    def poll_key(self, phase: Phase) -> Optional[str]:
        return self.keyboard.read_key()

    def begin_human_segment(self, gripper_action: float) -> None:
        self.device.start_control()
        # Continue the current gripper command: the device's "closed" state
        # emits -1.0 (see human_input).
        self.device.gripper_closed = gripper_action < 0.0

    def human_input(self) -> HumanInput:
        device = self.device
        control = device.control
        dpos = control[:3] * 0.005 * device.pos_sensitivity
        raw_drot = control[3:6] * 0.005 * device.rot_sensitivity
        # Device [roll, pitch, yaw] -> robot [pitch, roll, -yaw].
        drot = raw_drot[[1, 0, 2]]
        drot[2] = -drot[2]
        dpos = np.clip(dpos * 125, -1, 1)
        drot = np.clip(drot * 50, -1, 1)
        gripper_action = -1.0 if device.control_gripper == 1 else 1.0
        magnitude = np.abs(control)
        return HumanInput(
            action=np.concatenate([dpos, drot, [gripper_action]]),
            arm_active=bool(np.any(magnitude > SPACEMOUSE_ACTIVE_THRESHOLD)),
            strong=bool(np.any(magnitude > SPACEMOUSE_STRONG_THRESHOLD)),
        )

    def choose(self, options: str) -> str:
        while True:
            key = self.keyboard.read_key()
            if key is not None and key in options:
                return key
            time.sleep(0.1)

    def close(self) -> None:
        print("Stopping keyboard listener...")
        self.keyboard.close()
        print("Closing SpaceMouse...")
        self.device.close()


def robot_state(obs: dict) -> np.ndarray:
    """Recorded robot state: eef position (3), eef quaternion (4), gripper qpos (2)."""
    return np.concatenate(
        [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
    )


def wait_for_save(save_future: concurrent.futures.Future, *, poll_s: float = 60.0) -> None:
    """Block until a background episode save finishes; its exception propagates.

    A slow save (for example a long video encode) is not an error: it is waited
    for, with a progress line every ``poll_s`` seconds.
    """
    waited = 0.0
    while True:
        try:
            save_future.result(timeout=poll_s)
            return
        except concurrent.futures.TimeoutError:
            waited += poll_s
            print(f"Still waiting for the background episode save ({waited:.0f}s)...", flush=True)


OBJECT_VEL_THRESHOLD = 0.01  # m/s


def check_objects_moving(env) -> bool:
    """Return True if any task object's linear speed exceeds ``OBJECT_VEL_THRESHOLD``.

    Used to keep recording while objects fall or settle after human input.
    robosuite's NutAssembly and MimicGen's Square envs keep the nut in ``nuts``, not
    ``objects``, so on the Square tasks this always returns False.
    """
    base_env = unwrap_env(env)
    # Empty on the Square envs (no `objects`); see docs/reproduce.md#known-issues
    object_names = [
        obj.root_body for obj in getattr(base_env, "objects", []) if hasattr(obj, "root_body")
    ]
    for name in object_names:
        if np.linalg.norm(base_env.sim.data.get_body_xvelp(name)) > OBJECT_VEL_THRESHOLD:
            return True
    return False
