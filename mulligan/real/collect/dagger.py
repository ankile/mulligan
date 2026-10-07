"""
DAgger data collection on the real Franka robot via DROID.

Combines policy inference (from ``mulligan.real.collect.rollout``) with SpaceMouse
teleoperation (from ``mulligan.real.collect.teleop``) to enable human-in-the-loop
corrections.
Within a single episode, the operator can switch between policy and SpaceMouse
control any number of times. Every timestep is labelled with its source
(policy=0, human=1) and intervention flags mark policy-to-human transitions.

Dataset Structure (extends the ``rollout`` schema):
- observation.state: cartesian_position(6) + gripper_position(1) = 7D
- observation.state.*: decomposed state (cartesian_position, cartesian_velocity, joint_position, joint_velocity, gripper_position)
- observation.images.*: All camera streams, resized and converted to RGB
- action: canonical cartesian_velocity(6) + gripper_velocity(1) = 7D
- action.*: decomposed action in all spaces (cartesian/joint x position/velocity + gripper)
- source: int64 (1,) -- per-timestep 0=policy, 1=human
- intervention: int64 (1,) -- 1 at policy-to-human transitions, 0 elsewhere
- success, is_valid, steps_to_go, reward, done

Controls:
    During POLICY rollout:
    - 'h' key: Trigger human intervention (switch to SpaceMouse)
    - '0' key: Mark episode as TIMEOUT (truncated, done=0)
    - '9' key: Mark episode as FAILURE (terminal, done=1)
    - 'd' key: Discard episode entirely
    - 'q' key: Quit session

    During SPACEMOUSE correction:
    - SpaceMouse: Control robot end-effector (6-DOF)
    - Left button: Toggle gripper
    - 'h' key: Switch back to POLICY control
    - '1' key: Mark episode as SUCCESS and save
    - '0' key: Mark episode as TIMEOUT (truncated, done=0)
    - '9' key: Mark episode as FAILURE (terminal, done=1)
    - 'd' key: Discard episode entirely
    - 'q' key: Quit session

Usage:
    python -m mulligan.real.collect.dagger \
        --model hf://mulligan/real-marker-d2-r05-mulligan-dp \
        --freq 15 \
        --dataset-name <dataset> \
        --dataset-path ./data \
        --task-name marker_d2 \
        --push-to-hub --hf-namespace <your-hf-user-or-org>

The multi-arm, blinded protocol collector used for the paper's rounds is
``mulligan.real.collect.blind_dagger``; this script is its single-policy, unblinded form.
"""

# HighGUI must initialize before lerobot/av load (see mulligan.real.operator_ui.display).
from mulligan.real.operator_ui.display import prewarm_highgui

if __name__ == "__main__":
    prewarm_highgui()

import argparse
import concurrent.futures
import logging
import time
from pathlib import Path

import numpy as np

from mulligan.data.constants import DataSource
from mulligan.real.lifecycle.tasks import TASK_NAME_HELP, get_task_spec, task_name_choices
from mulligan.real.collect.dataset_features import (
    append_action_info,
    append_cartesian_velocities,
    append_franka_telemetry,
    append_joint_velocities,
    build_real_lerobot_features,
    build_canonical_action,
    ensure_dataset_can_store_episode_telemetry,
    finalize_episode_data,
    init_supplementary_lists,
    record_or_verify_camera_role_serials,
    save_episode_to_dataset,
    warn_missing_franka_telemetry_once,
)
from mulligan.real.robot.cameras import (
    DEFAULT_CAMERA_KEYS,
    image_camera_keys_from_obs,
    policy_live_camera_keys,
    select_image_camera_keys,
    serial_key_to_role,
)
from mulligan.real.policy.loader import REAL_PROTOCOL_N_ACTION_STEPS
from mulligan.real.collect.hf_utils import (
    add_hf_namespace_arg,
    add_license_arg,
    ensure_dataset_repo,
    resolve_push_repo_id,
)
from mulligan.real.collect.save_utils import wait_for_background_save
from mulligan.real.collect.rollout import (
    _refresh_obs_until_robot_state_timestamp_advances,
    auto_detect_device,
    parse_camera_keys,
    process_image,
    verified_reset,
)

from mulligan.real.operator_ui.cli import add_operator_ui_args
from mulligan.real.operator_ui.gates import GateOutcome
from mulligan.real.operator_ui.keys import key_label
from mulligan.real.operator_ui.session import OperatorUI

import mulligan.real.policy.lerobot_patches  # noqa: F401  (h264 video codec)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_action(device):
    """Compute 7D action from SpaceMouse state.

    Returns [vel_x, vel_y, vel_z, vel_roll, vel_pitch, vel_yaw, gripper].
    Axes are remapped to match the real Franka robot coordinate frame.
    """
    action = np.zeros(7)
    control = device.control

    # Flip x and y axes
    action[0] = -control[0]
    action[1] = -control[1]
    action[2] = control[2]

    # Shuffle rotation axes: roll on mouse is pitch on robot, so swap them
    action[3] = -control[4]
    action[4] = -control[3]
    action[5] = -control[5]
    action[:3] *= device.pos_sensitivity
    action[3:6] *= device.rot_sensitivity
    action[:6] = np.clip(action[:6], -1.0, 1.0)

    # Gripper: map 0 (open) -> -1.0, 1 (closed) -> 1.0
    gripper = device.control_gripper
    action[6] = 1.0 if gripper == 1 else -1.0

    return action


def gripper_command_to_velocity_sign(gripper_command: float, gripper_action_space: str) -> float:
    """Express a policy's last gripper command in the SpaceMouse's velocity-sign convention.

    The segment hand-off contract (``initial_gripper_action`` of
    :func:`spacemouse_correction_segment`) is the velocity convention: ``> 0`` means the
    gripper is CLOSED / closing, ``<= 0`` OPEN. A velocity-space gripper command already
    is that sign and passes through unchanged. A POSITION-space command (the UMI-relative
    arm, ``gripper_action_space="position"``) is an absolute target in ``[0, 1]`` where
    ``0`` is open — so a slightly-open ``0.03`` would read as CLOSED under the ``> 0``
    test and the SpaceMouse would slam the gripper shut the moment the operator
    intervened. Threshold the position at the half-way point instead.
    """
    if gripper_action_space == "velocity":
        return float(gripper_command)
    if gripper_action_space == "position":
        return 1.0 if float(gripper_command) > 0.5 else -1.0
    raise ValueError(f"Unknown gripper_action_space {gripper_action_space!r}")


# --------------------------------------------------------------------------- #
# Segment data helpers (from robosuite_dagger_state.py)
# --------------------------------------------------------------------------- #


def accumulate_segment_data(accumulated_data, segment_data, source_id):
    """Accumulate segment data into the full episode trajectory.

    Args:
        accumulated_data: Previously accumulated data (None if first segment)
        segment_data: New segment data to add
        source_id: Source ID for this segment (0=policy, 1=human)

    Returns:
        Tuple of (accumulated_data, sources) where sources is the list of
        source IDs for each timestep in this segment.
    """
    num_frames = len(segment_data["actions"])

    if accumulated_data is None:
        accumulated_data = {key: list(values) for key, values in segment_data.items()}
        sources = [source_id] * num_frames
    else:
        for key in segment_data.keys():
            accumulated_data[key].extend(segment_data[key])
        sources = [source_id] * num_frames

    return accumulated_data, sources


def segment_controls(toggle_action: str, *, allow_reset: bool, subgoal: bool) -> tuple[str, str]:
    """The operator panel's key summary for a DAgger segment (mirrors the terminal legend)."""
    first = f"{key_label('h')} {toggle_action}   |   1 success   |   9 failure   |   0 timeout"
    second = [f"{key_label('d')} discard"]
    if subgoal:
        second.insert(0, f"{key_label('g')} subgoal")
    if allow_reset:
        second.append(f"{key_label('r')} reset")
    second.append(f"{key_label('q')} quit")
    return first, "   |   ".join(second)


# --------------------------------------------------------------------------- #
# Policy rollout segment
# --------------------------------------------------------------------------- #


def policy_rollout_segment(
    env,
    policy,
    ui: OperatorUI,
    all_camera_keys,
    *,
    freq,
    save_camera_height=480,
    save_camera_width=640,
    max_steps=0,
    initial_gripper_action,
    allow_reset=False,
    record_step_fn=None,
    on_subgoal=None,
):
    """Run policy inference in a loop, recording every step.

    Runs until the user presses a key to intervene, mark failure, discard,
    or quit. Every policy step is recorded. ``on_subgoal`` (when given) enables the
    sub-goal key ('g' / numpad '3'): the caller records the mark against the frames
    it has already stored; without it the key is inert.

    The policy must conform to the RealWorldPolicy protocol:
    - .predict(raw_obs) -> np.ndarray
    - .reset()
    - .action_space (str)
    - .gripper_action_space (str | None)

    Args:
        env: DROID RobotEnv instance
        policy: RealWorldPolicy (LeRobotRealWorldPolicy or VisionIDQLRealWorldPolicy)
        ui: OperatorUI (keyboard, OpenCV key poll, live camera monitor)
        all_camera_keys: ALL camera keys for dataset saving
        freq: Control loop frequency in Hz
        save_camera_height: Target image height for dataset saving (default 480)
        save_camera_width: Target image width for dataset saving (default 640)
        max_steps: Max steps safety limit (0 = unlimited)
        initial_gripper_action: Last gripper action for continuity

    Returns:
        (segment_data, action_str, last_gripper_action) where action_str is
        one of: "intervention", "success", "timeout", "failure", "discard",
        "quit"; also "reset" when allow_reset=True and the operator presses 'r'.
        ``last_gripper_action`` is ALWAYS in the velocity-sign convention (``> 0``
        closed) so the following :func:`spacemouse_correction_segment` can seed its
        gripper toggle from it regardless of the policy's gripper space
        (:func:`gripper_command_to_velocity_sign`).

    The env is stepped with THIS policy's ``env_action_space`` /
    ``gripper_action_space`` on every call, never the RobotEnv constructor default, so
    a position-space (UMI-relative) policy segment and a velocity SpaceMouse
    correction segment can share one env inside a DAgger episode.
    """
    obs = env.get_observation()

    # env_action_space is cartesian_position for the relative arm; wrappers without
    # the property fall back to action_space. The effective gripper
    # space mirrors DROID's create_action_dict default (velocity iff the arm space is
    # a velocity space) so the hand-off sign conversion matches what the robot executed.
    env_action_space = getattr(policy, "env_action_space", policy.action_space)
    gripper_action_space = getattr(policy, "gripper_action_space", None)
    effective_gripper_space = gripper_action_space or (
        "velocity" if "velocity" in env_action_space else "position"
    )

    segment_data = {
        "observations": [],
        "joint_positions": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }
    for cam_key in all_camera_keys:
        segment_data[f"image_{cam_key}"] = []
    init_supplementary_lists(segment_data)
    warn_missing_franka_telemetry_once(obs)

    ui.set_phase(
        "Policy running",
        controls=segment_controls(
            "intervene", allow_reset=allow_reset, subgoal=on_subgoal is not None
        ),
    )
    # Drop stale keypresses from the previous mode (e.g. repeated 'h' presses).
    ui.drain_keys()

    print("\n" + "=" * 60)
    print("POLICY ROLLOUT MODE" + (f" (max {max_steps} steps)" if max_steps > 0 else ""))
    print("=" * 60)
    print(f"  {key_label('h')} = human INTERVENTION")
    print(f"  {key_label('1')} = mark SUCCESS and save")
    print(f"  {key_label('0')} = mark TIMEOUT (truncated) and save")
    print(f"  {key_label('9')} = mark FAILURE (terminal) and save")
    if on_subgoal is not None:
        print(f"  {key_label('g')} = SUB-GOAL reached (mid-episode subtask mark)")
    print(f"  {key_label('d')} = DISCARD episode")
    if allow_reset:
        print(f"  {key_label('r')} = RESET trajectory and retry from the start without saving")
    print(f"  {key_label('q')} = QUIT session")
    print("=" * 60)

    step = 0
    last_gripper_action = initial_gripper_action

    while max_steps == 0 or step < max_steps:
        loop_start = time.time()

        ui.render_monitor(obs["image"])

        # ---- Keyboard check ------------------------------------------------
        key = ui.read_key()
        if key == "h":
            print(f"\nHUMAN INTERVENTION triggered after {step} policy steps")
            return segment_data, "intervention", last_gripper_action
        elif key == "1":
            print(f"\nEpisode marked as SUCCESS after {step} policy steps")
            return segment_data, "success", last_gripper_action
        elif key == "0":
            print(f"\nEpisode marked as TIMEOUT after {step} policy steps")
            return segment_data, "timeout", last_gripper_action
        elif key == "9":
            print(f"\nEpisode marked as FAILURE after {step} policy steps")
            return segment_data, "failure", last_gripper_action
        elif key == "g" and on_subgoal is not None:
            on_subgoal()
        elif key == "d":
            print(f"\nEpisode DISCARDED after {step} policy steps")
            return segment_data, "discard", last_gripper_action
        elif allow_reset and key == "r":
            print(f"\nRESET requested after {step} policy steps")
            return segment_data, "reset", last_gripper_action
        elif key == "q":
            print(f"\nQUIT requested after {step} policy steps")
            return segment_data, "quit", last_gripper_action

        # ---- Build state for dataset recording ----------------------------
        state = np.concatenate(
            [
                np.array(obs["robot_state"]["cartesian_position"], dtype=np.float32),
                np.array([obs["robot_state"]["gripper_position"]], dtype=np.float32),
            ]
        )

        # ---- Collect observation data BEFORE stepping ----------------------
        segment_data["observations"].append(state.copy())
        segment_data["joint_positions"].append(
            np.array(obs["robot_state"]["joint_positions"], dtype=np.float32)
        )
        for cam_key in all_camera_keys:
            img = process_image(obs["image"][cam_key], save_camera_height, save_camera_width)
            segment_data[f"image_{cam_key}"].append(img)

        # ---- Policy inference (unified RealWorldPolicy interface) ----------
        action = policy.predict(obs)
        last_gripper_action = gripper_command_to_velocity_sign(action[-1], effective_gripper_space)

        # ---- Step the robot ------------------------------------------------
        previous_obs = obs
        obs = env.step(
            action,
            action_space=env_action_space,
            gripper_action_space=gripper_action_space,
        )
        action_info = obs.pop("action_info")
        obs = _refresh_obs_until_robot_state_timestamp_advances(
            env,
            previous_obs=previous_obs,
            obs=obs,
        )
        step += 1
        ui.progress.step += 1

        # ---- Collect action + supplementary data AFTER stepping -----------
        canonical_action = build_canonical_action(action_info)
        segment_data["actions"].append(canonical_action)
        segment_data["rewards"].append(0.0)
        segment_data["dones"].append(0)
        append_action_info(segment_data, action_info)
        append_joint_velocities(segment_data, obs)
        append_franka_telemetry(segment_data, obs)
        append_cartesian_velocities(segment_data, obs, previous_obs=previous_obs)
        if record_step_fn is not None:
            record_step_fn(
                segment_data,
                len(segment_data["actions"]) - 1,
                DataSource.AUTONOMOUS,
            )

        if step % 50 == 0:
            print(f"  Policy step {step}...")

        # ---- Wall-clock-aware sleep ----------------------------------------
        elapsed = time.time() - loop_start
        remaining = 1.0 / freq - elapsed
        if remaining > 0:
            time.sleep(remaining)
        else:
            print(
                f"  WARNING: Loop step took {elapsed * 1000:.0f}ms, exceeding {1000 / freq:.0f}ms budget by {-remaining * 1000:.0f}ms"
            )

    # Max steps reached — truncation, not terminal failure
    print(f"Max steps ({max_steps}) reached during policy rollout — marking as timeout (truncated)")
    return segment_data, "timeout", last_gripper_action


# --------------------------------------------------------------------------- #
# SpaceMouse correction segment
# --------------------------------------------------------------------------- #


def spacemouse_correction_segment(
    env,
    device,
    ui: OperatorUI,
    all_camera_keys,
    *,
    freq,
    save_camera_height=480,
    save_camera_width=640,
    initial_gripper_action,
    allow_reset=False,
    record_step_fn=None,
    on_subgoal=None,
):
    """Collect human correction via SpaceMouse.

    Only records steps when SpaceMouse input is detected (arm movement
    above threshold or gripper moving). Matches the teleop collector's behavior.
    ``on_subgoal`` enables the sub-goal key exactly as in ``policy_rollout_segment``.

    Args:
        env: DROID RobotEnv instance
        device: RobosuiteSpaceMouse instance
        ui: OperatorUI (keyboard, OpenCV key poll, live camera monitor)
        all_camera_keys: ALL camera keys for dataset saving
        freq: Control loop frequency in Hz
        save_camera_height: Target image height for dataset saving (default 480)
        save_camera_width: Target image width for dataset saving (default 640)
        initial_gripper_action: Last gripper action for continuity

    Returns:
        (segment_data, action_str, last_gripper_action) where action_str is
        one of: "continue", "success", "timeout", "failure", "discard", "quit";
        also "reset" when allow_reset=True and the operator presses 'r'.
    """
    obs = env.get_observation()

    device.start_control()

    # Initialize SpaceMouse gripper from last action to prevent jumps
    if initial_gripper_action is not None:
        gripper_is_closed = initial_gripper_action > 0.0
        device.gripper_closed = gripper_is_closed
        print(
            f"Gripper initialized: {'CLOSED' if gripper_is_closed else 'OPEN'} "
            f"(from action={initial_gripper_action:.2f})"
        )
    else:
        device.reset_gripper()

    # Discover camera keys from observation
    camera_keys_available = image_camera_keys_from_obs(obs)

    segment_data = {
        "observations": [],
        "joint_positions": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }
    save_cam_keys = all_camera_keys if all_camera_keys else camera_keys_available
    for cam_key in save_cam_keys:
        segment_data[f"image_{cam_key}"] = []
    init_supplementary_lists(segment_data)
    warn_missing_franka_telemetry_once(obs)

    ui.set_phase(
        "Human correction",
        controls=segment_controls(
            "back to policy", allow_reset=allow_reset, subgoal=on_subgoal is not None
        ),
    )
    # Drop stale keypresses from the previous mode (e.g. repeated 'h' presses).
    ui.drain_keys()

    print("\n" + "=" * 60)
    print("SPACEMOUSE CORRECTION MODE")
    print("=" * 60)
    print(f"  {key_label('h')} = switch back to POLICY control")
    print(f"  {key_label('1')} = mark SUCCESS and save")
    print(f"  {key_label('0')} = mark TIMEOUT (truncated) and save")
    print(f"  {key_label('9')} = mark FAILURE (terminal) and save")
    if on_subgoal is not None:
        print(f"  {key_label('g')} = SUB-GOAL reached (mid-episode subtask mark)")
    print(f"  {key_label('d')} = DISCARD episode")
    if allow_reset:
        print(f"  {key_label('r')} = RESET trajectory and retry from the start without saving")
    print(f"  {key_label('q')} = QUIT session")
    print("  (Recording only when SpaceMouse input detected)")
    print("=" * 60)

    step_count = 0
    prev_gripper = device.control_gripper
    gripper_is_moving = False
    prev_gripper_pos = None
    gripper_settled_count = 0
    gripper_grace_steps = 0

    GRIPPER_POS_THRESHOLD = 0.001
    GRIPPER_SETTLED_FRAMES = 1
    GRIPPER_GRACE_STEPS = 3

    # Track last gripper action
    last_gripper_action = initial_gripper_action if initial_gripper_action is not None else -1.0

    while True:
        loop_start = time.time()

        ui.render_monitor(obs["image"])

        # ---- Keyboard input ------------------------------------------------
        key = ui.read_key()
        if key == "h":
            print(f"\nSwitching back to POLICY ({step_count} correction steps)")
            return segment_data, "continue", last_gripper_action
        elif key == "1":
            print(f"\nSUCCESS! ({step_count} correction steps)")
            return segment_data, "success", last_gripper_action
        elif key == "0":
            print(f"\nTIMEOUT ({step_count} correction steps)")
            return segment_data, "timeout", last_gripper_action
        elif key == "9":
            print(f"\nFAILURE ({step_count} correction steps)")
            return segment_data, "failure", last_gripper_action
        elif key == "g" and on_subgoal is not None:
            on_subgoal()
        elif key == "d":
            print(f"\nDISCARDED ({step_count} correction steps)")
            return segment_data, "discard", last_gripper_action
        elif allow_reset and key == "r":
            print(f"\nRESET requested ({step_count} correction steps)")
            return segment_data, "reset", last_gripper_action
        elif key == "q":
            print(f"\nQUIT ({step_count} correction steps)")
            return segment_data, "quit", last_gripper_action

        # ---- SpaceMouse input ----------------------------------------------
        action = get_action(device)
        gripper = device.control_gripper

        # Detect gripper toggle
        if gripper != prev_gripper:
            gripper_is_moving = True
            gripper_settled_count = 0
            gripper_grace_steps = GRIPPER_GRACE_STEPS
            print("Gripper toggled, tracking motion until settled...")

        # Check for actual arm movement (above noise floor)
        has_arm_input = np.any(np.abs(device.control) > 0.005)
        has_input = has_arm_input or gripper_is_moving

        if has_input:
            # When only gripper is moving, zero out arm to prevent drift
            if not has_arm_input:
                action[:6] = 0.0

            last_gripper_action = action[6]

            # --- Record observation BEFORE stepping -------------------------
            state_obs = np.concatenate(
                [
                    np.array(obs["robot_state"]["cartesian_position"], dtype=np.float32),
                    np.array([obs["robot_state"]["gripper_position"]], dtype=np.float32),
                ]
            )
            segment_data["observations"].append(state_obs)
            segment_data["joint_positions"].append(
                np.array(obs["robot_state"]["joint_positions"], dtype=np.float32)
            )

            for cam_key in save_cam_keys:
                img = process_image(obs["image"][cam_key], save_camera_height, save_camera_width)
                segment_data[f"image_{cam_key}"].append(img)

            # --- Step the robot (SpaceMouse always uses cartesian_velocity) --
            # Both spaces are passed EXPLICITLY per step: the shared env may have been
            # constructed with a position default for a position / UMI-relative policy
            # cohort, and the SpaceMouse's +-1 gripper toggle is a velocity command.
            previous_obs = obs
            obs = env.step(
                action, action_space="cartesian_velocity", gripper_action_space="velocity"
            )
            action_info = obs.pop("action_info")
            obs = _refresh_obs_until_robot_state_timestamp_advances(
                env,
                previous_obs=previous_obs,
                obs=obs,
            )

            # --- Collect action + supplementary data AFTER stepping ---------
            canonical_action = build_canonical_action(action_info)
            segment_data["actions"].append(canonical_action)
            segment_data["rewards"].append(0.0)
            segment_data["dones"].append(0)
            append_action_info(segment_data, action_info)
            append_joint_velocities(segment_data, obs)
            append_franka_telemetry(segment_data, obs)
            append_cartesian_velocities(segment_data, obs, previous_obs=previous_obs)
            if record_step_fn is not None:
                record_step_fn(
                    segment_data,
                    len(segment_data["actions"]) - 1,
                    DataSource.HUMAN,
                )

            # --- Gripper settling detection ---------------------------------
            gripper_pos = obs["robot_state"]["gripper_position"]
            if gripper_is_moving:
                if gripper_grace_steps > 0:
                    gripper_grace_steps -= 1
                elif prev_gripper_pos is not None:
                    gripper_delta = abs(gripper_pos - prev_gripper_pos)
                    if gripper_delta < GRIPPER_POS_THRESHOLD:
                        gripper_settled_count += 1
                        if gripper_settled_count >= GRIPPER_SETTLED_FRAMES:
                            gripper_is_moving = False
                            print(
                                f"Gripper settled ({gripper_settled_count} frames below threshold)"
                            )
                    else:
                        gripper_settled_count = 0
            prev_gripper_pos = gripper_pos

            step_count += 1
            ui.progress.step += 1
            if step_count % 50 == 0:
                print(f"  Correction step {step_count}...")

        prev_gripper = gripper

        # ---- Wall-clock-aware sleep ----------------------------------------
        elapsed = time.time() - loop_start
        remaining = 1.0 / freq - elapsed
        if remaining > 0:
            time.sleep(remaining)
        else:
            print(
                f"  WARNING: Loop step took {elapsed * 1000:.0f}ms, exceeding {1000 / freq:.0f}ms budget by {-remaining * 1000:.0f}ms"
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="DAgger data collection on real Franka robot via DROID"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Policy MODEL_ID: hf://NAMESPACE/REPO[@REV] or a local checkpoint directory.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=REAL_PROTOCOL_N_ACTION_STEPS,
        help="Action chunk steps to execute before re-planning. Defaults to the "
        f"real-robot protocol exec horizon ({REAL_PROTOCOL_N_ACTION_STEPS}); "
        "prediction horizon comes from the checkpoint.",
    )

    # Control
    parser.add_argument(
        "--freq",
        type=int,
        default=15,
        help="Control loop frequency in Hz (default: 15)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Max steps per policy segment safety limit (default: 0 = unlimited)",
    )

    # Camera
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Target image height (default: 480)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Target image width (default: 640)",
    )
    parser.add_argument(
        "--camera-filter",
        type=str,
        default="_left",
        help="Camera suffix filter for policy input (default: '_left')",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=DEFAULT_CAMERA_KEYS,
        help=(
            "Comma-separated camera keys or ZED serials to RECORD; defaults to the "
            f"current robot station's full set ({DEFAULT_CAMERA_KEYS}). The dataset "
            "stores all of these so it is reusable as a datasource for any camera "
            "configuration; the policy is fed only the subset its config names. "
            "Pass an empty string to use --camera-filter discovery instead."
        ),
    )

    # Robot
    parser.add_argument(
        "--randomize-reset",
        action="store_true",
        help="Add noise to reset pose",
    )

    # Dataset (--dataset-name is required for DAgger)
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset name (required, e.g. 'marker-dagger-r1')",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data",
        help="Root path for datasets (default: ./data)",
    )
    parser.add_argument(
        "--task-name",
        choices=task_name_choices(),
        required=True,
        help=TASK_NAME_HELP,
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub after session",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make Hub dataset private (only with --push-to-hub)",
    )
    add_hf_namespace_arg(parser)
    add_license_arg(parser)

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (default: auto-detect cuda/mps/cpu)",
    )

    # Diffusion Policy scheduler overrides
    parser.add_argument(
        "--noise-scheduler",
        type=str,
        default=None,
        choices=["DDPM", "DDIM"],
        help="Override the checkpoint's noise scheduler (e.g., swap DDPM→DDIM for faster inference)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the number of denoising steps at inference (e.g., 8 for DDIM)",
    )

    add_operator_ui_args(parser, cards=False)
    args = parser.parse_args()
    args.task_name = get_task_spec(args.task_name).task_name
    if args.push_to_hub:
        try:
            resolve_push_repo_id(args.dataset_name, args.hf_namespace)
        except ValueError as exc:
            parser.error(str(exc))
    return args


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    args = parse_args()
    device = args.device or auto_detect_device()

    # ---- Dataset config ----------------------------------------------------
    dataset = None
    dataset_name = args.dataset_name
    dataset_path = Path(args.dataset_path) / dataset_name
    saved_episode_count = 0
    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    save_future = None

    # ---- Load policy -------------------------------------------------------
    print("=" * 60)
    print("Real Robot DAgger Data Collection")
    print("=" * 60)
    logger.info(
        "[protocol] Effective n_action_steps (exec horizon) = %s "
        "(real-robot protocol default = %d; prediction horizon is checkpoint-specific).",
        args.n_action_steps,
        REAL_PROTOCOL_N_ACTION_STEPS,
    )

    from mulligan.real.policy.loader import load_policy_by_model_id

    print(f"Loading policy: {args.model}")
    entry = load_policy_by_model_id(
        args.model,
        policy_id=0,
        device=device,
        noise_scheduler=args.noise_scheduler,
        num_inference_steps=args.num_inference_steps,
        default_camera_height=args.camera_height,
        default_camera_width=args.camera_width,
        n_action_steps=args.n_action_steps,
    )
    policy = entry.policy
    if (entry.camera_height, entry.camera_width) != (args.camera_height, args.camera_width):
        print(f"  Image resolution: {entry.camera_height}x{entry.camera_width} (from checkpoint)")

    print(f"Policy: {policy.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Control freq: {args.freq} Hz")
    print(f"Max steps/segment: {args.max_steps if args.max_steps > 0 else 'unlimited'}")
    print(f"Camera filter: *{args.camera_filter}")

    print("\nData saving: ENABLED")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Dataset name: {dataset_name}")
    print()

    # ---- Initialise hardware -----------------------------------------------
    import mulligan.real.robot.camera_config  # noqa: F401  (patches ZED to 15fps)
    from mulligan.real.robot.droid_compat import RobotEnv
    from mulligan.teleop.spacemouse import RobosuiteSpaceMouse

    print("Initializing robot environment...")
    env = RobotEnv(action_space="cartesian_velocity")

    print("Initializing SpaceMouse...")
    spacemouse = RobosuiteSpaceMouse(
        pos_sensitivity=1.3,
        rot_sensitivity=1.6,
    )

    ui = OperatorUI.from_args(args, cards=False)
    _ = ui.keyboard  # enter cbreak mode now, before the first episode
    print("Keyboard listener ready (terminal mode, works over SSH)")
    print()

    # ---- Episode loop ------------------------------------------------------
    # Decoupled camera sets: RECORD the full requested station set (so the dataset is
    # reusable as a datasource for any camera configuration) while feeding the policy
    # only the subset its config names.
    requested_camera_keys = parse_camera_keys(args.camera_keys, args.camera_filter)
    camera_keys: list[str] = []  # subset fed to the policy for inference
    all_camera_keys: list[str] = []  # full set recorded into the dataset
    episode_results = []

    try:
        episode_num = 0
        while True:
            episode_num += 1

            print(f"\n{'=' * 60}")
            print(f"Episode {episode_num}")
            print(f"{'=' * 60}")

            # Reset environment
            obs = verified_reset(env, randomize=args.randomize_reset)
            policy.reset()

            # Discover camera keys on first episode
            if not camera_keys:
                if requested_camera_keys is not None:
                    available = image_camera_keys_from_obs(obs)
                    missing = [k for k in requested_camera_keys if k not in available]
                    if missing:
                        raise RuntimeError(
                            f"Requested cameras are missing from observations: {missing}. "
                            f"Available cameras: {available}"
                        )
                    record_set = list(requested_camera_keys)
                    all_cams = available
                else:
                    record_set, all_cams = select_image_camera_keys(obs, args.camera_filter)

                # The policy consumes only the cameras its config names (a subset of the
                # recorded set); the full record_set is persisted for reuse.
                policy_set = policy_live_camera_keys(policy, record_set)
                camera_keys.extend(policy_set)
                all_camera_keys.extend(record_set)
                print(f"Cameras discovered: {all_cams}")
                print(f"Cameras for dataset: {all_camera_keys}")
                print(f"Cameras fed to policy: {camera_keys}")

                # Update the policy's camera keys (needed for inference)
                if hasattr(policy, "set_camera_keys"):
                    policy.set_camera_keys(camera_keys)

            # ---- Inner loop: policy <-> human switching --------------------
            accumulated_data = None
            accumulated_sources = []
            current_gripper_action = None  # will be inferred from first policy action
            episode_outcome = None
            segment_count = 0
            had_intervention = False

            while True:
                segment_count += 1

                # --- POLICY PHASE ---
                seg_data, action_str, gripper = policy_rollout_segment(
                    env,
                    policy,
                    ui,
                    all_camera_keys,
                    freq=args.freq,
                    save_camera_height=args.camera_height,
                    save_camera_width=args.camera_width,
                    max_steps=args.max_steps,
                    initial_gripper_action=current_gripper_action,
                )

                # Accumulate policy data
                if len(seg_data["actions"]) > 0:
                    accumulated_data, seg_sources = accumulate_segment_data(
                        accumulated_data,
                        seg_data,
                        DataSource.AUTONOMOUS,
                    )
                    accumulated_sources.extend(seg_sources)

                current_gripper_action = gripper

                if action_str == "intervention":
                    had_intervention = True
                    # --- HUMAN PHASE ---
                    seg_data, action_str, gripper = spacemouse_correction_segment(
                        env,
                        spacemouse,
                        ui,
                        all_camera_keys,
                        freq=args.freq,
                        save_camera_height=args.camera_height,
                        save_camera_width=args.camera_width,
                        initial_gripper_action=current_gripper_action,
                    )

                    # Accumulate human data
                    if len(seg_data["actions"]) > 0:
                        accumulated_data, seg_sources = accumulate_segment_data(
                            accumulated_data,
                            seg_data,
                            DataSource.HUMAN,
                        )
                        accumulated_sources.extend(seg_sources)

                    current_gripper_action = gripper

                    if action_str == "continue":
                        # Back to policy
                        policy.reset()  # clear stale action queue
                        print("\n" + ">" * 60)
                        print(f"Resuming POLICY control (segment {segment_count + 1})")
                        print(">" * 60)
                        continue
                    elif action_str == "success":
                        episode_outcome = "success"
                        break
                    elif action_str == "timeout":
                        episode_outcome = "timeout"
                        break
                    elif action_str == "failure":
                        episode_outcome = "failure"
                        break
                    elif action_str == "discard":
                        episode_outcome = "discard"
                        break
                    elif action_str == "quit":
                        episode_outcome = "quit"
                        break
                elif action_str == "success":
                    episode_outcome = "success"
                    break
                elif action_str == "timeout":
                    episode_outcome = "timeout"
                    break
                elif action_str == "failure":
                    episode_outcome = "failure"
                    break
                elif action_str == "discard":
                    episode_outcome = "discard"
                    break
                elif action_str == "quit":
                    episode_outcome = "quit"
                    break

            # ---- Post-episode processing -----------------------------------
            is_success = episode_outcome == "success"
            should_save = (
                episode_outcome in ("success", "timeout", "failure")
                and accumulated_data is not None
                and len(accumulated_sources) > 0
            )

            total_steps = len(accumulated_sources) if accumulated_data else 0
            label = (episode_outcome or "unknown").upper()
            print(
                f"\nEpisode {episode_num}: {label} ({total_steps} steps, {segment_count} segments)"
            )

            episode_results.append(
                {
                    "steps": total_steps,
                    "outcome": episode_outcome,
                    "segments": segment_count,
                    "had_intervention": had_intervention,
                }
            )

            if should_save:
                # Append T+1 terminal frame (final obs + padded action)
                final_obs = env.get_observation()

                final_state = np.concatenate(
                    [
                        np.array(final_obs["robot_state"]["cartesian_position"], dtype=np.float32),
                        np.array([final_obs["robot_state"]["gripper_position"]], dtype=np.float32),
                    ]
                )
                accumulated_data["observations"].append(final_state.copy())
                accumulated_data["joint_positions"].append(
                    np.array(final_obs["robot_state"]["joint_positions"], dtype=np.float32)
                )
                for cam_key in all_camera_keys:
                    img = process_image(
                        final_obs["image"][cam_key], args.camera_height, args.camera_width
                    )
                    accumulated_data[f"image_{cam_key}"].append(img)

                # Finalize: set reward/done on last valid frame, append padded terminal row
                is_terminal = episode_outcome in ("success", "failure")
                finalize_episode_data(
                    accumulated_data,
                    final_obs,
                    is_success=is_success,
                    is_terminal=is_terminal,
                )
                # The terminal padding row is a copy of the last real frame;
                # source labels must have the same length as accumulated_data.
                accumulated_sources.append(accumulated_sources[-1])

                # Discover camera data keys
                cam_data_keys = sorted(k for k in accumulated_data if k.startswith("image_"))

                # Lazy-init dataset on first save
                if dataset is None:
                    if dataset_path.exists():
                        print(f"Loading existing dataset from {dataset_path}")
                        dataset = LeRobotDataset(
                            repo_id=dataset_name,
                            root=str(dataset_path),
                        )
                        dataset = ensure_dataset_can_store_episode_telemetry(
                            dataset,
                            accumulated_data,
                            dataset_path=dataset_path,
                            dataset_name=dataset_name,
                        )
                        print(f"Loaded existing dataset with {dataset.num_episodes} episodes")
                    else:
                        print(f"Creating new dataset at {dataset_path}")

                        features = build_real_lerobot_features(
                            accumulated_data,
                            cam_data_keys,
                            camera_name_fn=serial_key_to_role,
                        )

                        dataset = LeRobotDataset.create(
                            repo_id=dataset_name,
                            fps=args.freq,
                            root=str(dataset_path),
                            robot_type="franka",
                            features=features,
                            image_writer_threads=4,
                            streaming_encoding=False,
                        )
                        print(f"Dataset created at {dataset_path}")
                    # Record the serial<->role provenance map on a fresh create, and on
                    # resume verify the live cabling still matches it (re-cabled /
                    # wrong-station cameras would otherwise mislabel roles).
                    record_or_verify_camera_role_serials(dataset, cam_data_keys)

                # Wait for previous background save
                if save_future is not None:
                    wait_for_background_save(
                        save_future,
                        description="Previous episode background save",
                        timeout=120,
                    )

                # Submit episode saving to background thread
                print("Saving episode in background...")
                save_future = save_executor.submit(
                    save_episode_to_dataset,
                    dataset=dataset,
                    episode_data=accumulated_data,
                    episode_success=is_success,
                    camera_keys=cam_data_keys,
                    task_name=args.task_name,
                    saved_episode_count=saved_episode_count,
                    sources=accumulated_sources,
                    camera_name_fn=serial_key_to_role,
                )
                saved_episode_count += 1

            elif episode_outcome == "discard":
                print("Episode discarded -- not saved")

            # Reset the robot for next episode
            verified_reset(env, randomize=args.randomize_reset)

            if episode_outcome == "quit":
                break

            # Wait for user confirmation before starting next episode
            gate = ui.gate(
                prompt="Ready for the next episode",
                on_reset=lambda: verified_reset(env, randomize=args.randomize_reset),
                any_key_starts=True,
            )
            if gate is GateOutcome.QUIT:
                print("Quitting.")
                episode_outcome = "quit"

            if episode_outcome == "quit":
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")
    finally:
        # Wait for in-flight background save
        if save_future is not None:
            print("Waiting for background episode save to finish...")
            wait_for_background_save(save_future, description="Final background episode save")
        save_executor.shutdown(wait=True)

        if dataset is not None:
            print(f"Finalizing dataset... ({saved_episode_count} episodes saved)")
            dataset.stop_image_writer()
            dataset.finalize()

            # Consolidate episodes parquet files to prevent schema mismatches
            # when the dataset was resumed across multiple sessions
            from mulligan.data.recording import consolidate_episodes_parquet

            consolidate_episodes_parquet(dataset_path)

            # Push to HuggingFace Hub if requested
            if args.push_to_hub:
                print("\nPushing dataset to HuggingFace Hub...")
                hub_api = HfApi()
                repo_id = resolve_push_repo_id(dataset_name, args.hf_namespace)
                print(f"  Repo ID: {repo_id}")

                print(f"  Private: {args.private}")

                if ensure_dataset_repo(hub_api, repo_id, private=args.private):
                    print("  Repository created")
                else:
                    print(f"  Repository already exists: {repo_id}")

                dataset.repo_id = repo_id
                from mulligan.tools.lerobot_hub import push_lerobot_dataset_tagged_main

                push_lerobot_dataset_tagged_main(
                    dataset, private=args.private, license=args.license
                )
                print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")

        ui.close()
        print("Closing SpaceMouse...")
        spacemouse.close()

    # ---- Session summary ---------------------------------------------------
    if episode_results:
        print()
        print("=" * 60)
        print("Session Summary")
        print("=" * 60)

        autonomous_successes = sum(
            1 for r in episode_results if r["outcome"] == "success" and not r["had_intervention"]
        )
        assisted_successes = sum(
            1 for r in episode_results if r["outcome"] == "success" and r["had_intervention"]
        )
        timeouts = sum(1 for r in episode_results if r["outcome"] == "timeout")
        failures = sum(1 for r in episode_results if r["outcome"] == "failure")
        discards = sum(1 for r in episode_results if r["outcome"] == "discard")
        total = autonomous_successes + assisted_successes + timeouts + failures

        for i, r in enumerate(episode_results, 1):
            tag = ""
            if r["outcome"] == "success" and r["had_intervention"]:
                tag = " (assisted)"
            print(
                f"  Episode {i}: {r['outcome'].upper():>8s}{tag}  "
                f"({r['steps']} steps, {r['segments']} segments)"
            )

        print(f"\nTotal episodes: {len(episode_results)}")
        if total > 0:
            print(
                f"Autonomous success rate: {autonomous_successes}/{total} = {autonomous_successes / total:.1%}"
            )
            if assisted_successes > 0:
                print(f"Assisted successes: {assisted_successes}/{total}")
            if timeouts > 0:
                print(f"Timeouts: {timeouts}")
            if failures > 0:
                print(f"Failures: {failures}")
        if discards > 0:
            print(f"Discarded: {discards}")
        print(f"Episodes saved: {saved_episode_count}")
        print("Done.")


if __name__ == "__main__":
    main()
