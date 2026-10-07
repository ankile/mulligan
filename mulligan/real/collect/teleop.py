"""
Real robot teleoperation with DROID using SpaceMouse.

Collects demonstrations from a real Franka robot and saves them to LeRobot
dataset format, compatible with HuggingFace Hub.

Dataset Structure:
- observation.state: cartesian_position(6) + gripper_position(1) = 7D
- observation.state.*: decomposed state (cartesian_position, cartesian_velocity, joint_position, joint_velocity, gripper_position)
- observation.images.<role>: All camera streams, resized and converted to RGB, stored under
  their station role names (side_1, wrist_left, ...); meta/camera_role_serials.json records
  the serial each role was recorded from
- action: canonical cartesian_velocity(6) + gripper_velocity(1) = 7D
- action.*: decomposed action in all spaces (cartesian/joint x position/velocity + gripper)
- Metadata: source, success, is_valid, steps_to_go, reward, done

Recording Behavior:
- Every env.step() is recorded -- if there's no input, we just sleep.
- Gripper toggles are tracked until the gripper physically settles.
- Wall-clock-aware sleep maintains consistent loop rate.

Controls:
    - SpaceMouse: Control robot end-effector (6-DOF)
    - Left button: Toggle gripper
    - '1' key: Mark episode as SUCCESS and save to dataset
    - '0' key: Mark episode as FAILURE (discard)
    - Ctrl+C: Quit and finalize dataset

Usage:
    # Basic teleoperation (no saving)
    python -m mulligan.real.collect.teleop

    # Save demonstrations
    python -m mulligan.real.collect.teleop --save-data --dataset-name my_demos

    # Save and push to HuggingFace Hub
    python -m mulligan.real.collect.teleop --save-data --dataset-name my_demos \
        --push-to-hub --hf-namespace <your-hf-user-or-org>

    # Custom frequency and image size
    python -m mulligan.real.collect.teleop --save-data --freq 10 --camera-height 240 --camera-width 320
"""

# HighGUI must initialize before lerobot/av load (see mulligan.real.operator_ui.display).
from mulligan.real.operator_ui.display import prewarm_highgui

if __name__ == "__main__":
    prewarm_highgui()

import argparse
import concurrent.futures
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from mulligan.teleop.spacemouse import RobosuiteSpaceMouse
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
    select_image_camera_keys,
    serial_key_to_role,
)
from mulligan.real.collect.initial_states import load_manifest_targets
from mulligan.real.operator_ui.cards import cleanup_unreferenced_initial_state_cards
from mulligan.real.operator_ui.cli import add_operator_ui_args
from mulligan.real.operator_ui.gates import GateOutcome
from mulligan.real.operator_ui.keys import key_label
from mulligan.real.operator_ui.session import OperatorUI
from mulligan.real.collect.blind_dagger import (
    _ensure_manifest_snapshot,
    _initial_state_features,
    _is_lerobot_dataset_initialized,
    _prepare_lerobot_create_root,
    _restore_lerobot_precreate_artifacts,
    _sha256_file,
    _target_extra_frame_fields,
    _target_ledger_fields,
)
from mulligan.real.collect.initial_states import (
    InitialStateTarget,
    _format_initial_state_target,
    _initial_state_setup_subject,
)
from mulligan.real.collect.rollout import (
    camera_serials_from_keys,
    parse_camera_keys,
    process_image,
    restrict_zed_cameras_to_serials,
    verified_reset,
)
from mulligan.real.collect.hf_utils import (
    add_hf_namespace_arg,
    add_license_arg,
    ensure_dataset_repo,
    resolve_push_repo_id,
)
from mulligan.real.collect.hub_resume_guard import (
    assert_local_dataset_not_behind_hub,
    assert_local_episode_count_matches_ledger,
    read_local_total_episodes,
)
from mulligan.real.collect.save_utils import wait_for_background_save

# get_action is the canonical SpaceMouse->7D-action helper shared with the DAgger collector.
from mulligan.real.collect.dagger import get_action

import mulligan.real.policy.lerobot_patches  # noqa: F401  (h264 video codec)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi


# --------------------------------------------------------------------------- #
# Episode loop
# --------------------------------------------------------------------------- #


def teleop_episode(
    env,
    device,
    ui: OperatorUI,
    freq=15,
    camera_height=480,
    camera_width=640,
    randomize_reset=False,
    camera_filter="_left",
    requested_camera_keys=None,
    initial_target: InitialStateTarget | None = None,
    manifest_meta: dict | None = None,
    task_name: str | None = None,
):
    """
    Teleoperate a single episode on the real robot.

    Every env.step() call is recorded. When there is no SpaceMouse input and the
    gripper is not moving, the loop simply sleeps without stepping the robot.

    Args:
        env: droid RobotEnv instance
        device: RobosuiteSpaceMouse instance
        ui: OperatorUI (keyboard, target-card window, live camera monitor)
        freq: Control loop frequency in Hz
        camera_height: Target image height after resize
        camera_width: Target image width after resize
        randomize_reset: If True, add cartesian noise to the reset joint pose

    Returns:
        (episode_data, outcome, visualization_path), where outcome is one of
        "success", "failure", or "retry".
    """
    visualization_path = None
    if initial_target is not None:
        if manifest_meta is None:
            raise ValueError("manifest_meta is required when initial_target is set")
        print("Manual initial-state setup:")
        print(f"  {_format_initial_state_target(initial_target)}")
        visualization_path = ui.show_card(initial_target, manifest_meta, task_name=task_name)

    # Reset robot and get initial observation. Route through verified_reset so a
    # silent gRPC reset miss (~10% per DESIGN.md) cannot record a teleop demo
    # from the wrong robot pose — matches every other real tool.
    obs = verified_reset(env, randomize=randomize_reset)

    if initial_target is not None:
        subject = _initial_state_setup_subject(initial_target, task_name)
        outcome = ui.gate(
            prompt=f"Place the {subject} at the shown target",
            on_reset=lambda: verified_reset(env, randomize=randomize_reset),
            any_key_starts=True,
            render=lambda: ui.render_monitor(env.get_observation()["image"]),
        )
        if outcome is GateOutcome.QUIT:
            raise KeyboardInterrupt
        # The reset observation was captured before manual object placement.
        # Refresh here so frame 0 matches the state the operator actually set up.
        obs = env.get_observation()

    device.start_control()
    device.reset_gripper()

    # Discover available camera keys from the observation, filtered
    all_cams = sorted(obs.get("image", {}).keys())
    if requested_camera_keys is None:
        camera_keys, _all_cams = select_image_camera_keys(obs, camera_filter)
    else:
        missing = [key for key in requested_camera_keys if key not in all_cams]
        if missing:
            raise RuntimeError(f"Requested cameras missing: {missing}; available={all_cams}")
        camera_keys = list(requested_camera_keys)
    print(f"Cameras found: {all_cams}")
    print(f"Cameras for dataset: {camera_keys}")

    # Episode storage
    episode_data = {
        "observations": [],  # cartesian_position(6) + gripper_position(1)
        "joint_positions": [],  # joint_positions(7)
        "actions": [],  # cartesian_velocity(6) + gripper(1)
        "rewards": [],
        "dones": [],
    }
    for cam_key in camera_keys:
        episode_data[f"image_{cam_key}"] = []
    init_supplementary_lists(episode_data)
    warn_missing_franka_telemetry_once(obs)

    print("\nEpisode started. Use SpaceMouse to control the robot.")
    print(
        f"Press {key_label('1')} for SUCCESS (save), {key_label('0')} for FAILURE (discard), "
        f"{key_label('r')} to reset/retry same target, Ctrl+C to quit."
    )
    # Only a key pressed AFTER the episode started may end it (never a buffered one).
    ui.drain_keys()

    step_count = 0
    prev_gripper = device.control_gripper
    gripper_is_moving = False
    prev_gripper_pos = None
    gripper_settled_count = 0
    gripper_grace_steps = 0

    GRIPPER_POS_THRESHOLD = 0.001  # position delta below which gripper counts as "settled"
    GRIPPER_SETTLED_FRAMES = 1  # consecutive settled frames required
    GRIPPER_GRACE_STEPS = 3  # ignore settle checks right after toggle (~0.2s at 15 Hz)

    while True:
        loop_start = time.time()

        # ---- Live camera monitor (cropped policy view) ---------------------
        ui.render_monitor(obs["image"])

        # ---- Keyboard input ------------------------------------------------
        key = ui.read_key()
        if key == "r":
            print()
            print("=" * 60)
            print(f"RETRY requested. Discarding {step_count} steps and keeping same target.")
            print("=" * 60)
            return {}, "retry", visualization_path
        if key in ("1", "0"):
            outcome = "success" if key == "1" else "failure"
            label = outcome.upper()
            print()
            print("=" * 60)
            print(f"{label}! Episode length: {step_count} steps")
            print("=" * 60)

            # Store final observation + padded action/reward/done
            if step_count > 0:
                state_obs = np.concatenate(
                    [
                        np.array(obs["robot_state"]["cartesian_position"]),
                        np.array([obs["robot_state"]["gripper_position"]]),
                    ]
                )
                episode_data["observations"].append(state_obs)
                episode_data["joint_positions"].append(
                    np.array(obs["robot_state"]["joint_positions"])
                )
                for cam_key in camera_keys:
                    img = process_image(obs["image"][cam_key], camera_height, camera_width)
                    episode_data[f"image_{cam_key}"].append(img)

                finalize_episode_data(
                    episode_data,
                    obs,
                    is_success=outcome == "success",
                    is_terminal=True,
                )

            return episode_data, outcome, visualization_path

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

            # --- Record current observation BEFORE stepping -----------------
            state_obs = np.concatenate(
                [
                    np.array(obs["robot_state"]["cartesian_position"]),
                    np.array([obs["robot_state"]["gripper_position"]]),
                ]
            )
            episode_data["observations"].append(state_obs)
            episode_data["joint_positions"].append(np.array(obs["robot_state"]["joint_positions"]))

            for cam_key in camera_keys:
                img = process_image(obs["image"][cam_key], camera_height, camera_width)
                episode_data[f"image_{cam_key}"].append(img)

            # --- Step the robot ---------------------------------------------
            previous_obs = obs
            obs = env.step(action)
            action_info = obs.pop("action_info")

            # --- Collect action + supplementary data AFTER stepping ---------
            canonical_action = build_canonical_action(action_info)
            episode_data["actions"].append(canonical_action)
            episode_data["rewards"].append(0.0)
            episode_data["dones"].append(0)
            append_action_info(episode_data, action_info)
            append_joint_velocities(episode_data, obs)
            append_franka_telemetry(episode_data, obs)
            append_cartesian_velocities(episode_data, obs, previous_obs=previous_obs)

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
            if step_count % 50 == 0:
                print(f"  Step {step_count}...")

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


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _read_manifest_teleop_ledger(path: Path) -> tuple[list[dict], set[int], int]:
    if not path.exists():
        return [], set(), 0
    rows: list[dict] = []
    consumed: set[int] = set()
    successes = 0
    seen_episodes: set[int] = set()
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        episode_index = int(row["episode_index"])
        if episode_index in seen_episodes:
            raise ValueError(f"{path}:{line_no}: duplicate episode_index {episode_index}")
        seen_episodes.add(episode_index)
        manifest_idx = row.get("manifest_idx")
        if manifest_idx is not None:
            manifest_idx = int(manifest_idx)
            if manifest_idx in consumed:
                raise ValueError(f"{path}:{line_no}: duplicate manifest_idx {manifest_idx}")
            consumed.add(manifest_idx)
        if bool(row["success"]):
            successes += 1
        rows.append(row)
    if sorted(seen_episodes) != list(range(len(rows))):
        raise ValueError(f"{path}: episode_index values must be contiguous from 0")
    return rows, consumed, successes


def _resume_saved_episode_count(dataset_path: Path, manifest_ledger_rows: list[dict] | None) -> int:
    """Episode index of the next save when (re)opening ``dataset_path``.

    Manifest collection resumes from its ledger (checked against the dataset by
    ``assert_local_episode_count_matches_ledger``). Without a manifest the local
    dataset is the only record of earlier sessions, so the count comes from its
    ``meta/info.json`` (0 for a new dataset).
    """
    if manifest_ledger_rows is not None:
        return len(manifest_ledger_rows)
    return read_local_total_episodes(dataset_path)


def _select_next_manifest_target(
    targets: list[InitialStateTarget],
    consumed_manifest_idxs: set[int],
) -> InitialStateTarget | None:
    for target in targets:
        if target.manifest_idx not in consumed_manifest_idxs:
            return target
    return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _recent_seconds_per_success(
    recent_success_times: deque[float],
    *,
    session_started_monotonic: float,
    session_success_count: int,
) -> float | None:
    if not recent_success_times:
        return None
    if len(recent_success_times) >= 2:
        elapsed = recent_success_times[-1] - recent_success_times[0]
        return elapsed / max(1, len(recent_success_times) - 1)
    elapsed = time.monotonic() - session_started_monotonic
    return elapsed / max(1, session_success_count)


def _collection_progress_lines(
    *,
    saved_episode_count: int,
    saved_success_count: int,
    target_successes: int | None,
    session_saved_count: int,
    session_success_count: int,
    recent_success_times: deque[float],
    session_started_monotonic: float,
) -> list[str]:
    lines = ["Progress"]
    if target_successes is None:
        lines.append(f"  Successful saved episodes: {saved_success_count}")
    else:
        remaining_successes = max(0, target_successes - saved_success_count)
        lines.append(
            "  Successful saved episodes: "
            f"{saved_success_count} / {target_successes} "
            f"({remaining_successes} remaining)"
        )
    lines.append(
        f"  Saved this session: {session_saved_count} episodes, {session_success_count} successes"
    )

    sec_per_success = _recent_seconds_per_success(
        recent_success_times,
        session_started_monotonic=session_started_monotonic,
        session_success_count=session_success_count,
    )
    if sec_per_success is None:
        lines.append("  Pace: unknown")
    else:
        lines.append(
            "  Pace: "
            f"{_format_duration(sec_per_success)}/success "
            f"over last {len(recent_success_times)} success(es)"
        )
    if target_successes is not None:
        remaining_successes = max(0, target_successes - saved_success_count)
        eta_seconds = None if sec_per_success is None else remaining_successes * sec_per_success
        eta = "complete" if remaining_successes == 0 else _format_duration(eta_seconds)
        lines.append(f"  ETA to target: {eta}")
    return lines


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate real robot (DROID) with SpaceMouse and save to LeRobot dataset"
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=15,
        help="Control loop frequency in Hz (default: 15)",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Target camera image height (default: 480)",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Target camera image width (default: 640)",
    )
    parser.add_argument(
        "--camera-filter",
        type=str,
        default="_left",
        help="Camera suffix filter for saving (default: '_left')",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=DEFAULT_CAMERA_KEYS,
        help=(
            "Comma-separated camera keys or bare ZED serials to open and save; "
            f"defaults to the current robot station's set ({DEFAULT_CAMERA_KEYS}). "
            "When set, DROID ZED discovery is restricted before RobotEnv starts. "
            "Pass an empty string to use --camera-filter discovery instead."
        ),
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save teleoperation data to LeRobot dataset",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data",
        help="Root path to save datasets (default: ./data)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name. Auto-generated as droid_{timestamp} if not specified",
    )
    parser.add_argument(
        "--task-name",
        choices=task_name_choices(),
        default=None,
        help=TASK_NAME_HELP + " Required with --save-data or --initial-states-manifest.",
    )
    parser.add_argument(
        "--target-successes",
        type=int,
        default=None,
        help="Stop after this many successful saved episodes.",
    )
    parser.add_argument(
        "--progress-window",
        type=int,
        default=20,
        help="Number of recent successful episodes used for collection pace/ETA.",
    )
    parser.add_argument(
        "--initial-states-manifest",
        type=Path,
        default=None,
        help="Manual setup manifest with pen_x, pen_y, pen_yaw targets.",
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
        help="JSONL ledger for manifest-driven teleop collection.",
    )
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Also save failed episodes (marked with '0')",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub after data collection",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make dataset private on HuggingFace Hub (only with --push-to-hub)",
    )
    add_hf_namespace_arg(parser)
    add_license_arg(parser)
    parser.add_argument(
        "--randomize-reset",
        action="store_true",
        help="Add cartesian noise to the reset pose each episode",
    )
    add_operator_ui_args(parser, cards=True)
    args = parser.parse_args(argv)
    if args.target_successes is not None and args.target_successes <= 0:
        parser.error("--target-successes must be positive")
    if args.progress_window <= 0:
        parser.error("--progress-window must be positive")
    if args.initial_states_manifest is not None and args.ledger_path is None and not args.save_data:
        parser.error("--ledger-path is required with manifest collection unless --save-data is set")
    if args.push_to_hub:
        if not args.save_data:
            parser.error("--push-to-hub requires --save-data")
        try:
            # Without --dataset-name the dataset gets a bare droid_<timestamp> name.
            resolve_push_repo_id(args.dataset_name or "droid_<timestamp>", args.hf_namespace)
        except ValueError as exc:
            parser.error(str(exc))
    if args.task_name is None:
        if args.save_data or args.initial_states_manifest is not None:
            parser.error("--task-name is required with --save-data or --initial-states-manifest")
    else:
        args.task_name = get_task_spec(args.task_name).task_name
    return args


def main():
    from mulligan.real.robot.droid_compat import RobotEnv

    args = parse_args()
    requested_camera_keys = parse_camera_keys(args.camera_keys, args.camera_filter)
    manifest_targets: list[InitialStateTarget] | None = None
    manifest_meta: dict = {}
    ledger_path = args.ledger_path
    ledger_rows: list[dict] = []
    consumed_manifest_idxs: set[int] = set()
    saved_success_count = 0
    manifest_sha256: str | None = None
    manifest_snapshot_path: Path | None = None
    if args.initial_states_manifest is not None:
        # Derive the teleop arm allowlist FROM the manifest (its declared ``arms``, or the
        # distinct row sources as a fallback) so the consumer can never drift from the
        # manifest's source labels. (A hardcoded list silently went stale when a 3rd arm
        # was added -- it crashed the run at load with "source has no matching --arm".)
        # Teleop has no policies, so each arm's model_id is the dummy "teleop".
        manifest_targets, manifest_meta = load_manifest_targets(
            args.initial_states_manifest,
            expected_task=args.task_name,
            model_id=lambda key: "teleop",
        )
        manifest_sha256 = _sha256_file(args.initial_states_manifest)
        if ledger_path is None:
            dataset_name_for_ledger = args.dataset_name or "droid_teleop"
            ledger_path = (
                Path(args.dataset_path)
                / dataset_name_for_ledger
                / "meta"
                / "teleop_manifest_ledger.jsonl"
            )
        ledger_rows, consumed_manifest_idxs, saved_success_count = _read_manifest_teleop_ledger(
            ledger_path
        )

    # ---- Dataset setup -----------------------------------------------------
    dataset = None
    dataset_name = None
    dataset_path = None
    saved_episode_count = len(ledger_rows)
    camera_keys = None  # discovered from first episode observation

    if args.save_data:
        dataset_name = args.dataset_name
        if dataset_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"droid_{timestamp}"

        dataset_path = Path(args.dataset_path) / dataset_name
        push_repo_id = (
            resolve_push_repo_id(dataset_name, args.hf_namespace) if args.push_to_hub else None
        )
        if args.initial_states_manifest is not None:
            if args.ledger_path is None:
                ledger_path = dataset_path / "meta" / "teleop_manifest_ledger.jsonl"
                ledger_rows, consumed_manifest_idxs, saved_success_count = (
                    _read_manifest_teleop_ledger(ledger_path)
                )
            manifest_snapshot_path = dataset_path / "meta" / "initial_states_manifest.json"
        saved_episode_count = _resume_saved_episode_count(
            dataset_path, ledger_rows if args.initial_states_manifest is not None else None
        )
        print("\nData saving: ENABLED")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Dataset name: {dataset_name}")
        if args.initial_states_manifest is not None:
            print(f"  Initial-state manifest: {args.initial_states_manifest}")
            print(f"  Manifest SHA256: {manifest_sha256}")
            print(f"  Manifest ledger: {ledger_path}")
            print(f"  Manifest snapshot: {manifest_snapshot_path}")
            print(f"  Manifest rows consumed: {len(consumed_manifest_idxs)}")
            print(f"  Successful saved episodes in ledger: {saved_success_count}")
            local_dataset_episodes = assert_local_episode_count_matches_ledger(
                dataset_path=dataset_path,
                ledger_count=saved_episode_count,
                ledger_path=ledger_path,
            )
            if local_dataset_episodes:
                print(f"  Local dataset episodes: {local_dataset_episodes}")
        if args.push_to_hub:
            hub_check = assert_local_dataset_not_behind_hub(
                repo_id=push_repo_id,
                dataset_path=dataset_path,
                allow_empty_repo=True,
            )
            if hub_check.remote_total_episodes is not None:
                print(
                    "  Hub resume guard: "
                    f"local={hub_check.local_total_episodes} episode(s), "
                    f"remote={hub_check.remote_total_episodes} episode(s) "
                    f"at {hub_check.repo_id}@main"
                )
    else:
        print("\nData saving: DISABLED")

    # ---- Initialise hardware -----------------------------------------------
    import mulligan.real.robot.camera_config  # noqa: F401  (patches ZED to 15fps)

    if requested_camera_keys is not None:
        requested_serials = camera_serials_from_keys(requested_camera_keys)
        print(f"Restricting ZED cameras to serials: {requested_serials}")
        restrict_zed_cameras_to_serials(requested_serials)

    # Cards (and so the card window) only exist for a manifest-driven collection.
    ui = OperatorUI.from_args(
        args,
        cards=args.initial_states_manifest is not None,
        default_card_dir=dataset_path / "meta" / "initial_state_targets" if dataset_path else None,
    )

    print("Initializing robot environment...")
    env = RobotEnv(action_space="cartesian_velocity")

    print("Initializing SpaceMouse...")
    device = RobosuiteSpaceMouse(
        pos_sensitivity=1.3,
        rot_sensitivity=1.6,
    )

    print("\nSetup complete! Ready to teleoperate.")
    print("=" * 60)

    # ---- Episode loop ------------------------------------------------------
    completed_unsaved_episode_count = 0
    session_started_monotonic = time.monotonic()
    session_saved_count = 0
    session_success_count = 0
    recent_success_times: deque[float] = deque(maxlen=args.progress_window)
    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    save_future = None  # tracks the in-flight background save
    try:
        while True:
            if save_future is not None and save_future.done():
                save_future.result()
                save_future = None
            if args.target_successes is not None and saved_success_count >= args.target_successes:
                print(f"Target successful demos reached: {saved_success_count}")
                break
            current_initial_target = None
            if manifest_targets is not None:
                current_initial_target = _select_next_manifest_target(
                    manifest_targets,
                    consumed_manifest_idxs,
                )
                if current_initial_target is None:
                    raise RuntimeError(
                        "Initial-state manifest exhausted before reaching "
                        f"target_successes={args.target_successes}; "
                        f"current_successes={saved_success_count}"
                    )

            display_episode = (
                saved_episode_count if args.save_data else completed_unsaved_episode_count
            )
            print(f"\n{'=' * 60}")
            print(f"Episode {display_episode}")
            print(f"{'=' * 60}")
            if args.save_data:
                print(f"Dataset episode index: {saved_episode_count}")
            for line in _collection_progress_lines(
                saved_episode_count=saved_episode_count,
                saved_success_count=saved_success_count,
                target_successes=args.target_successes,
                session_saved_count=session_saved_count,
                session_success_count=session_success_count,
                recent_success_times=recent_success_times,
                session_started_monotonic=session_started_monotonic,
            ):
                print(line)

            episode_data, episode_outcome, visualization_path = teleop_episode(
                env,
                device,
                ui,
                freq=args.freq,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                randomize_reset=args.randomize_reset,
                camera_filter=args.camera_filter,
                requested_camera_keys=requested_camera_keys,
                initial_target=current_initial_target,
                manifest_meta=manifest_meta,
                task_name=args.task_name,
            )

            if episode_outcome == "retry":
                print("\nRetrying same manifest target without saving or advancing manifest.")
                continue
            if episode_outcome not in {"success", "failure"}:
                raise RuntimeError(f"Unexpected teleop episode outcome: {episode_outcome!r}")

            episode_success = episode_outcome == "success"
            episode_length = len(episode_data["actions"])
            # T+1 frames: T real steps + 1 padded terminal frame
            num_real_steps = max(0, episode_length - 1)
            print("\nEpisode Summary:")
            print(f"  Steps: {num_real_steps}")
            print(f"  Outcome: {'SUCCESS' if episode_success else 'FAILURE'}")

            if episode_length == 0:
                print("  Empty episode, skipping.")
                continue

            # Discover camera keys on first real episode
            if camera_keys is None:
                camera_keys = sorted(k for k in episode_data if k.startswith("image_"))
                print(f"  Camera image keys: {camera_keys}")

            # ---- Save to LeRobot dataset -----------------------------------
            should_save = episode_success or (args.save_failures and episode_success is False)
            if args.save_data and should_save:
                # Lazy-init dataset on first save
                if dataset is None:
                    if _is_lerobot_dataset_initialized(dataset_path):
                        print(f"Loading existing dataset from {dataset_path}")
                        dataset = LeRobotDataset(
                            repo_id=dataset_name,
                            root=str(dataset_path),
                        )
                        dataset = ensure_dataset_can_store_episode_telemetry(
                            dataset,
                            episode_data,
                            dataset_path=dataset_path,
                            dataset_name=dataset_name,
                        )
                        print(f"Loaded existing dataset with {dataset.num_episodes} episodes")
                    else:
                        print(f"Creating new dataset at {dataset_path}")

                        if args.initial_states_manifest is not None:
                            extra_features = _initial_state_features(manifest_meta)
                        else:
                            extra_features = None
                        features = build_real_lerobot_features(
                            episode_data,
                            camera_keys,
                            extra_features=extra_features,
                            camera_name_fn=serial_key_to_role,
                        )

                        precreate_backup_path = _prepare_lerobot_create_root(dataset_path)
                        dataset = LeRobotDataset.create(
                            repo_id=dataset_name,
                            fps=args.freq,
                            root=str(dataset_path),
                            robot_type="franka",
                            features=features,
                            image_writer_threads=4,
                            streaming_encoding=False,
                        )
                        _restore_lerobot_precreate_artifacts(
                            precreate_backup_path,
                            dataset_path,
                        )
                        print(f"Dataset created at {dataset_path}")
                    # Record the serial<->role provenance map on a fresh create, and on
                    # resume verify the live cabling still matches it (re-cabled /
                    # wrong-station cameras would otherwise mislabel roles).
                    record_or_verify_camera_role_serials(dataset, camera_keys)
                    if args.initial_states_manifest is not None:
                        manifest_snapshot_path, manifest_sha256 = _ensure_manifest_snapshot(
                            args.initial_states_manifest,
                            dataset_path,
                        )
                    if dataset.num_episodes != saved_episode_count:
                        raise RuntimeError(
                            f"Dataset episode count {dataset.num_episodes} does not match "
                            f"ledger/saved count {saved_episode_count}; refusing to append"
                        )

                # Wait for previous background save before starting a new one
                if save_future is not None:
                    save_future.result()
                    save_future = None

                # Submit episode saving to background thread
                print("Saving episode in background...")
                episode_index = saved_episode_count

                def _save_and_log(
                    *,
                    episode_data=episode_data,
                    episode_success=episode_success,
                    episode_camera_keys=camera_keys,
                    episode_index=episode_index,
                    episode_initial_target=current_initial_target,
                    episode_visualization_path=visualization_path,
                    episode_length=episode_length,
                ) -> None:
                    extra_frame_fields = None
                    if episode_initial_target is not None:
                        extra_frame_fields = _target_extra_frame_fields(
                            episode_initial_target,
                            manifest_meta,
                        )
                    save_episode_to_dataset(
                        dataset=dataset,
                        episode_data=episode_data,
                        episode_success=episode_success,
                        camera_keys=episode_camera_keys,
                        task_name=args.task_name,
                        saved_episode_count=episode_index,
                        default_source=DataSource.HUMAN,
                        extra_frame_fields=extra_frame_fields,
                        camera_name_fn=serial_key_to_role,
                    )
                    if episode_initial_target is not None:
                        _append_jsonl(
                            ledger_path,
                            {
                                "episode_index": episode_index,
                                "success": bool(episode_success),
                                "task_name": args.task_name,
                                "dataset_name": dataset_name,
                                "steps": int(max(0, episode_length - 1)),
                                "manifest_idx": episode_initial_target.manifest_idx,
                                "manifest_source": episode_initial_target.source,
                                "manifest_source_index": episode_initial_target.source_index,
                                "manifest_snapshot_path": str(manifest_snapshot_path),
                                "manifest_sha256": manifest_sha256,
                                "initial_state_visualization_path": str(episode_visualization_path),
                                **_target_ledger_fields(episode_initial_target, manifest_meta),
                                "written_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                            },
                        )

                save_future = save_executor.submit(_save_and_log)
                # Optimistically increment -- if save fails, we crash on next
                # iteration's future.result() anyway.
                saved_episode_count += 1
                session_saved_count += 1
                if current_initial_target is not None:
                    consumed_manifest_idxs.add(current_initial_target.manifest_idx)
                if episode_success:
                    saved_success_count += 1
                    session_success_count += 1
                    recent_success_times.append(time.monotonic())

            elif args.save_data and episode_success is False:
                print("Episode marked as failure -- not saved (use --save-failures to save)")
            else:
                completed_unsaved_episode_count += 1

            for line in _collection_progress_lines(
                saved_episode_count=saved_episode_count,
                saved_success_count=saved_success_count,
                target_successes=args.target_successes,
                session_saved_count=session_saved_count,
                session_success_count=session_success_count,
                recent_success_times=recent_success_times,
                session_started_monotonic=session_started_monotonic,
            ):
                print(line)

            print("\nReady for next episode...")

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        print("\nCleaning up...")

        # Wait for in-flight background save
        if save_future is not None:
            print("Waiting for background episode save to finish...")
            wait_for_background_save(save_future, description="Final background episode save")
        save_executor.shutdown(wait=True)

        if args.save_data and dataset is not None:
            print(f"Finalizing dataset... ({saved_episode_count} episodes saved)")
            removed_targets = cleanup_unreferenced_initial_state_cards(ui.card_dir, ledger_path)
            if removed_targets:
                print(f"Removed {len(removed_targets)} unreferenced initial-state target cards")
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
                repo_id = push_repo_id
                print(f"  Repo ID: {repo_id}")

                print(f"  Private: {args.private}")

                if ensure_dataset_repo(hub_api, repo_id, private=args.private):
                    print("  Repository created")
                else:
                    print(f"  Repository already exists: {repo_id}")

                assert_local_dataset_not_behind_hub(
                    repo_id=repo_id,
                    dataset_path=dataset_path,
                    api=hub_api,
                    allow_empty_repo=True,
                )
                dataset.repo_id = repo_id
                from mulligan.tools.lerobot_hub import push_lerobot_dataset_tagged_main

                push_lerobot_dataset_tagged_main(
                    dataset, private=args.private, license=args.license
                )
                print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")

        print("Stopping keyboard listener...")
        ui.close()
        print("Closing SpaceMouse...")
        device.close()
        print("Cleanup complete. Goodbye!")


if __name__ == "__main__":
    main()
