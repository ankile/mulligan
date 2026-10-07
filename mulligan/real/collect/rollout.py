"""
Deploy a trained policy on the real Franka robot via DROID.

Loads a real-robot policy (Diffusion Policy actor or Vision-IDQL critic+actor) through
:func:`mulligan.real.policy.loader.load_policy_by_model_id` and runs closed-loop
inference on the robot at a fixed control frequency. Optionally saves rollout data
(observations, actions, images from selected cameras) to a LeRobotDataset.

The module also provides the shared rollout helpers (camera selection, verified reset,
``rollout_episode``) used by the collectors and ``mulligan.real.eval.manifest_eval``.

Controls:
    - '1' key: Mark episode as SUCCESS (terminal, done=1)
    - '0' key: Mark episode as TIMEOUT / truncation (non-terminal, done=0)
    - '9' key: Mark episode as FAILURE / irrecoverable (terminal, done=1)
    - 'q' key: Quit session

Usage:
    # A released actor from the HuggingFace Hub
    python -m mulligan.real.collect.rollout \
        --model hf://mulligan/real-marker-d2-r05-mulligan-dp

    # A local checkpoint directory, with step and episode limits
    python -m mulligan.real.collect.rollout \
        --model ./outputs/train_real/<run>/final \
        --freq 15 --max-steps 600 --num-episodes 5

    # Save rollout data to a LeRobotDataset (--dataset-name enables saving)
    python -m mulligan.real.collect.rollout \
        --model hf://mulligan/real-marker-d2-r05-mulligan-dp \
        --dataset-name <dataset> --dataset-path ./data --task-name marker_d2

    # By default there is no step limit per episode: the episode runs until you press
    # '1' (success), '0' (timeout), '9' (failure) or 'q' (quit). All outcomes are saved
    # unless --no-save-failures is passed.
"""

# HighGUI must initialize before lerobot/av load (see mulligan.real.operator_ui.display).
from mulligan.real.operator_ui.display import prewarm_highgui

if __name__ == "__main__":
    prewarm_highgui()

import argparse
import concurrent.futures
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import cv2

from mulligan.data.constants import DataSource
from mulligan.real.operator_ui.cli import add_operator_ui_args
from mulligan.real.operator_ui.gates import GateOutcome
from mulligan.real.operator_ui.keys import key_label
from mulligan.real.operator_ui.session import OperatorUI
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
    _robot_state_timestamp_seconds,
)
from mulligan.real.robot.cameras import (
    DEFAULT_CAMERA_KEYS,
    DEFAULT_EXCLUDED_CAMERA_KEYS,
    image_camera_keys_from_obs,
    policy_live_camera_keys,
    select_image_camera_keys,
    serial_key_to_role,
)
from mulligan.real.robot.cli import add_robot_reset_args
from mulligan.real.lifecycle.tasks import TASK_NAME_HELP, get_task_spec, task_name_choices
from mulligan.real.collect.hf_utils import (
    add_hf_namespace_arg,
    add_license_arg,
    ensure_dataset_repo,
    resolve_push_repo_id,
)
from mulligan.real.collect.save_utils import wait_for_background_save
from mulligan.training.timer import TrainingTimer

import mulligan.real.policy.lerobot_patches  # noqa: F401  (h264 video codec)
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RecoverableRolloutError(RuntimeError):
    """Rollout fault whose partial episode should be discarded and retried."""


def _refresh_obs_until_robot_state_timestamp_advances(
    env,
    *,
    previous_obs: dict,
    obs: dict,
    max_wait_s: float = 1.5,
    poll_interval_s: float = 0.01,
) -> dict:
    """Replace stale post-step observations with a fresh robot-state sample.

    DROID can occasionally return an observation from ``env.step()`` whose
    robot-state timestamp is identical to the pre-step observation. Recording
    that frame would make derived Cartesian velocity invalid, so we poll for a
    fresh observation before appending any action-aligned fields.
    """
    previous_ts = _robot_state_timestamp_seconds(previous_obs)
    current_ts = _robot_state_timestamp_seconds(obs)
    if current_ts > previous_ts:
        return obs

    deadline = time.monotonic() + max_wait_s
    attempts = 0
    last_dt = current_ts - previous_ts
    while time.monotonic() < deadline:
        attempts += 1
        time.sleep(poll_interval_s)
        refreshed_obs = env.get_observation()
        current_ts = _robot_state_timestamp_seconds(refreshed_obs)
        last_dt = current_ts - previous_ts
        if current_ts > previous_ts:
            logger.debug(
                "env.step() returned a stale robot_state timestamp; used fresh "
                "observation after %d poll(s), dt=%.6fs.",
                attempts,
                last_dt,
            )
            return refreshed_obs

    raise RecoverableRolloutError(
        "Robot-state timestamp did not advance after env.step(); refusing to save "
        f"an invalid velocity frame (last dt={last_dt:.6f}s after {max_wait_s:.2f}s)."
    )


# --------------------------------------------------------------------------- #
# Camera selection
# --------------------------------------------------------------------------- #


def parse_camera_keys(camera_keys_arg: str | None, camera_filter: str) -> list[str] | None:
    """Parse explicit camera keys.

    Accepts full DROID image keys (``<serial>_left``) or bare ZED serials
    (``<serial>``), in which case ``camera_filter`` is appended.
    """
    if camera_keys_arg is None:
        return None
    if camera_keys_arg.strip() == "":
        return None

    camera_keys = []
    for raw_key in camera_keys_arg.split(","):
        key = raw_key.strip()
        if not key:
            continue
        key = key.removeprefix("observation.images.")

        if key.endswith("_left") or key.endswith("_right"):
            camera_keys.append(key)
        elif camera_filter:
            camera_keys.append(f"{key}{camera_filter}")
        else:
            raise ValueError(
                f"Camera key '{key}' is a bare serial, but --camera-filter is empty. "
                "Pass full keys like '<serial>_left,<serial>_left'."
            )

    if not camera_keys:
        raise ValueError("--camera-keys was provided but no camera keys were parsed")
    if len(set(camera_keys)) != len(camera_keys):
        raise ValueError(f"--camera-keys contains duplicates: {camera_keys}")
    excluded = sorted(set(camera_keys) & set(DEFAULT_EXCLUDED_CAMERA_KEYS))
    if excluded:
        raise ValueError(
            f"--camera-keys includes excluded camera(s) {excluded}. "
            "These cameras must not be stored in real-world datasets."
        )
    return camera_keys


def camera_serials_from_keys(camera_keys: list[str]) -> list[str]:
    serials = []
    for key in camera_keys:
        serial = key.removeprefix("observation.images.").rsplit("_", 1)[0]
        if serial not in serials:
            serials.append(serial)
    return serials


def restrict_zed_cameras_to_serials(serials: list[str]) -> None:
    """Patch DROID camera discovery before RobotEnv opens cameras."""
    allowed = set(serials)
    if not allowed:
        raise ValueError("Camera serial restriction cannot be empty")

    import mulligan.real.robot.droid_compat  # noqa: F401  (OpenCV aruco shim before DROID imports)
    import droid.camera_utils.camera_readers.zed_camera as zed_mod
    import droid.camera_utils.wrappers.multi_camera_wrapper as multi_cam_mod

    def gather_selected_zed_cameras():
        try:
            devices = zed_mod.sl.Camera.get_device_list()
        except NameError:
            return []

        discovered_serials = [str(device.serial_number) for device in devices]
        missing = sorted(allowed - set(discovered_serials))
        if missing:
            raise RuntimeError(
                f"Requested ZED camera serials are not connected: {missing}. "
                f"Discovered serials: {discovered_serials}"
            )

        return [
            zed_mod.ZedCamera(device) for device in devices if str(device.serial_number) in allowed
        ]

    zed_mod.gather_zed_cameras = gather_selected_zed_cameras
    multi_cam_mod.gather_zed_cameras = gather_selected_zed_cameras


def record_subtask_mark(subtask_frames: list[int], *, step: int, subtask_marks: int) -> bool:
    """Handle one sub-goal key press during a rollout; append the marked frame in place.

    ``step`` is the loop counter at key-read time, i.e. the number of frames already
    recorded: the mark lands on the most recently recorded frame (``step - 1``), the
    action that completed the sub-goal (minus operator reaction lag, which the outcome
    review nudges). Returns True when a mark was recorded. Ignored presses print why.
    """
    if subtask_marks <= 0:
        print("Sub-goal key ignored: this task defines no subtask marks.")
        return False
    if len(subtask_frames) >= subtask_marks:
        print(
            f"Sub-goal key ignored: already recorded {subtask_marks} mark(s) at "
            f"frame(s) {subtask_frames}."
        )
        return False
    if step <= 0:
        print("Sub-goal key ignored: no frame recorded yet.")
        return False
    subtask_frames.append(step - 1)
    print(
        f"Sub-goal {len(subtask_frames)}/{subtask_marks} reached at frame {step - 1} "
        "(reward=1.0 spike will be written at save)."
    )
    return True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def process_image(img, height=480, width=640):
    """Convert a BGR(A) uint8 image to RGB and resize."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # handles both BGR and BGRA
    if img.shape[0] != height or img.shape[1] != width:
        # Dataset frames do not need Lanczos-quality resampling on the control
        # path; INTER_AREA is much cheaper for the usual camera downsample.
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def auto_detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def verified_reset(
    env,
    randomize=False,
    max_retries=6,
    joint_threshold=0.15,
    retry_delay_s=1.0,
    retry_backoff=1.5,
    max_retry_delay_s=5.0,
):
    """Reset with verification and retry to handle silent gRPC failures.

    When randomize=True, performs a two-phase reset:
    1. Verified non-randomized reset to home (with retries) — guarantees
       the robot reaches the nominal home position.
    2. Single randomized reset on top — if this silently fails, the robot
       is at least at home, not in an arbitrary position.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    if retry_delay_s < 0:
        raise ValueError(f"retry_delay_s must be >= 0, got {retry_delay_s}")
    if retry_backoff < 1:
        raise ValueError(f"retry_backoff must be >= 1, got {retry_backoff}")
    if max_retry_delay_s < 0:
        raise ValueError(f"max_retry_delay_s must be >= 0, got {max_retry_delay_s}")

    # Phase 1: Verified reset to nominal home position
    next_delay_s = min(max_retry_delay_s, retry_delay_s)
    for attempt in range(1, max_retries + 1):
        obs = env.reset(randomize=False)

        actual_joints = np.array(obs["robot_state"]["joint_positions"])
        target_joints = np.array(env.reset_joints)
        joint_error = np.abs(actual_joints - target_joints).max()

        if joint_error < joint_threshold:
            if attempt > 1:
                logger.info(f"Reset succeeded on attempt {attempt}")
            break

        retry_suffix = ""
        if attempt < max_retries and next_delay_s > 0:
            retry_suffix = f" Waiting {next_delay_s:g}s before retry."
        logger.warning(
            f"Reset attempt {attempt}/{max_retries}: robot didn't reach target "
            f"(max joint error: {joint_error:.3f} rad). Retrying..."
            f"{retry_suffix}"
        )
        if attempt < max_retries and next_delay_s > 0:
            time.sleep(next_delay_s)
            next_delay_s = min(max_retry_delay_s, next_delay_s * retry_backoff)
    else:
        raise RuntimeError(
            f"Reset failed after {max_retries} attempts! Max joint error: {joint_error:.3f} rad"
        )

    # Phase 2: If randomize requested, do one randomized reset on top
    if randomize:
        obs = env.reset(randomize=True)

    return obs


CHUNK_INFO_DIRNAME = "chunk_info"


def open_or_create_rollout_dataset(
    *,
    dataset_path: Path,
    dataset_name: str,
    episode_data: dict,
    cam_data_keys: list[str],
    fps: int,
):
    """Load the rollout dataset at ``dataset_path`` or create it on the first save.

    A dataset counts as existing only when ``meta/info.json`` is present: an IDQL
    policy writes ``chunk_info/`` sidecars under ``dataset_path`` before the first
    episode is saved. ``LeRobotDataset.create`` refuses an existing root, so those
    sidecars are parked next to it and moved back after the create. Cameras are stored
    under their station role names; on both paths the serial<->role camera map is
    recorded (create) or verified (resume).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if (dataset_path / "meta" / "info.json").exists():
        print(f"Loading existing dataset from {dataset_path}")
        dataset = LeRobotDataset(repo_id=dataset_name, root=str(dataset_path))
        dataset = ensure_dataset_can_store_episode_telemetry(
            dataset,
            episode_data,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )
        print(f"Loaded existing dataset with {dataset.num_episodes} episodes")
        # Resume guard: a re-cabled camera or a different station would otherwise
        # append frames under the wrong role names.
        record_or_verify_camera_role_serials(dataset, cam_data_keys)
        return dataset

    parked = None
    if dataset_path.exists():
        unexpected = sorted(p.name for p in dataset_path.iterdir() if p.name != CHUNK_INFO_DIRNAME)
        if unexpected:
            raise RuntimeError(
                f"{dataset_path} exists without meta/info.json and holds {unexpected}; "
                "refusing to create a dataset over it."
            )
        chunk_info = dataset_path / CHUNK_INFO_DIRNAME
        if chunk_info.exists():
            parked = dataset_path.with_name(f"{dataset_path.name}.{CHUNK_INFO_DIRNAME}.precreate")
            if parked.exists():
                raise RuntimeError(f"{parked} already exists; move it aside and retry.")
            chunk_info.rename(parked)
        dataset_path.rmdir()

    print(f"Creating new dataset at {dataset_path}")
    features = build_real_lerobot_features(
        episode_data, cam_data_keys, camera_name_fn=serial_key_to_role
    )
    dataset = LeRobotDataset.create(
        repo_id=dataset_name,
        fps=fps,
        root=str(dataset_path),
        robot_type="franka",
        features=features,
        image_writer_threads=4,
        streaming_encoding=False,
    )
    if parked is not None:
        parked.rename(dataset_path / CHUNK_INFO_DIRNAME)
    # Persist the serial<->role provenance sidecar, as the collectors and eval do.
    record_or_verify_camera_role_serials(dataset, cam_data_keys)
    print(f"Dataset created at {dataset_path}")
    return dataset


# --------------------------------------------------------------------------- #
# Episode loop
# --------------------------------------------------------------------------- #


def rollout_episode(
    env,
    policy,
    ui: OperatorUI,
    *,
    freq=15,
    camera_height=480,
    camera_width=640,
    save_camera_height=None,
    save_camera_width=None,
    camera_filter="_left",
    max_steps=0,
    auto_timeout_at_max_steps=True,
    randomize_reset=False,
    save_data=False,
    all_camera_keys=None,
    requested_camera_keys=None,
    warn_loop_overruns=False,
    print_timing_summary=True,
    reset_max_retries=6,
    reset_retry_delay_s=1.0,
    reset_retry_backoff=1.5,
    reset_max_retry_delay_s=5.0,
    robot_state_refresh_max_wait_s=1.5,
    robot_state_refresh_poll_interval_s=0.01,
    pre_reset_callback=None,
    subtask_marks=0,
):
    """Run one closed-loop policy episode on the real robot.

    Returns ``(num_steps, outcome, episode_data, subtask_frames)``. ``subtask_frames``
    are the frame indices the operator marked with the sub-goal key ('g' / numpad
    '3'), in press order; the same frames carry a ``reward=1.0`` spike in
    ``episode_data`` (see :func:`finalize_episode_data`).

    Args:
        subtask_marks: Number of mid-episode sub-goal marks the task defines
            (``RealTaskSpec.num_subtask_marks``; e.g. 1 for routing_d2's first clip).
            With 0 the sub-goal key is inert. Presses beyond the cap, or before any
            frame is recorded, are ignored loudly; a mark is never undone live — the
            outcome review (``mulligan.tools.outcome_review``) remains the correction path.
        policy: A RealWorldPolicy object (LeRobotRealWorldPolicy or
            VisionIDQLRealWorldPolicy) with predict(raw_obs), reset(), and action_space
            attributes.
        camera_height/camera_width: Resolution for data saving fallback.
        save_camera_height/save_camera_width: Resolution for dataset saving.
            Defaults to camera_height/camera_width if not specified. Use this
            when the policy inference resolution differs from the desired
            dataset storage resolution (e.g. blind eval with mixed policies).
        save_data: If True, collect per-step data for dataset saving.
        all_camera_keys: Mutable list — populated on first call with selected
            cameras. Used to record the same camera streams to the dataset.
        requested_camera_keys: Optional explicit camera keys. When provided,
            these replace suffix-based discovery and must be present.
        warn_loop_overruns: If True, print every control-loop deadline miss.
            Defaults off because per-step terminal writes add noise and jitter.
        print_timing_summary: If True, print aggregate timing stats after
            each episode. Eval callers can disable this to keep operator logs
            focused on outcomes and setup prompts.
        reset_*: Verified reset retry/backoff controls used for the automatic
            post-episode reset.
        robot_state_refresh_*: Controls for polling fresh robot-state
            observations after env.step() returns a stale timestamp.
        pre_reset_callback: Optional ``callback(outcome)`` invoked once on the
            main thread immediately BEFORE the end-of-episode ``verified_reset``,
            after the terminal frame is captured. Lets an eval/collection caller
            render the next operator target card so the operator can stage the
            next scene while the robot is resetting (mirrors the deferred-reset
            card ordering in ``mulligan.real.collect.blind_dagger``). Must NOT touch the
            RobotEnv — reset stays on the main thread below.

    Returns:
        (num_steps, outcome, episode_data) where outcome is 'success',
        'timeout', 'failure', 'restart', or 'quit'. episode_data is a dict
        of lists when save_data=True and the rollout should be saved,
        otherwise None.
    """
    if save_camera_height is None:
        save_camera_height = camera_height
    if save_camera_width is None:
        save_camera_width = camera_width

    # Get a fresh observation without physically resetting — the end-of-episode
    # reset already moved the robot back.  Calling env.reset()
    # here would duplicate that physical motion and waste ~2-5s per episode.
    obs = env.get_observation()
    policy.reset()

    # Discover camera keys on first call (caller passes mutable lists).
    #
    # Decoupled camera sets (mirrors collect.blind_dagger): the task/robot station defines
    # the full RECORDED set (all requested streams), discovered once per session and
    # cached in all_camera_keys so the dataset is reusable as a datasource for any camera
    # configuration. Each POLICY then selects its OWN input subset from that recorded set
    # via its config — recomputed every rollout so a multi-arm session never feeds one
    # arm's camera subset to another.
    if all_camera_keys:
        record_set = list(all_camera_keys)
    else:
        if requested_camera_keys is not None:
            all_cams = image_camera_keys_from_obs(obs)
            missing = [k for k in requested_camera_keys if k not in all_cams]
            if missing:
                raise RuntimeError(
                    f"Requested cameras are missing from observations: {missing}. "
                    f"Available cameras: {all_cams}"
                )
            record_set = list(requested_camera_keys)
            print(f"Cameras discovered: {all_cams}")
            print(f"Cameras recorded (--camera-keys): {record_set}")
        else:
            record_set, all_cams = select_image_camera_keys(obs, camera_filter)
            print(f"Cameras discovered: {all_cams}")
            print(f"Cameras recorded (filter '*{camera_filter}'): {record_set}")
        if all_camera_keys is not None:
            all_camera_keys.extend(record_set)
            print(f"Cameras for dataset: {record_set}")

    # The policy consumes only the cameras its config names (a subset of the recorded
    # set); a policy that declares none falls back to the full set. Recomputed per call.
    policy_set = policy_live_camera_keys(policy, record_set)
    print(f"Cameras fed to policy: {policy_set}")
    if hasattr(policy, "set_camera_keys"):
        policy.set_camera_keys(policy_set)

    save_cam_keys = record_set if save_data else []

    # Init episode data collection
    episode_data = None
    if save_data:
        warn_missing_franka_telemetry_once(obs)
        episode_data = {
            "observations": [],
            "joint_positions": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        for cam_key in save_cam_keys:
            episode_data[f"image_{cam_key}"] = []
        init_supplementary_lists(episode_data)

    if max_steps > 0:
        episode_limit = f"max {max_steps} steps"
    else:
        episode_limit = "no step limit"
    subgoal_hint = (
        f", {key_label('g')}=SUB-GOAL REACHED (x{subtask_marks})" if subtask_marks > 0 else ""
    )
    print(
        f"\nEpisode started ({episode_limit} at {freq} Hz). "
        f"Press {key_label('1')}=SUCCESS, {key_label('0')}=TIMEOUT, "
        f"{key_label('9')}=FAILURE, {key_label('r')}=RESTART, {key_label('q')}=QUIT{subgoal_hint}"
    )
    subtask_frames: list[int] = []
    ui.begin_rollout(max_steps, subtask_marks)
    # Drop any keys buffered during reset/setup or the previous rollout so this
    # episode ends only on a key pressed AFTER it started -- never on a stale one.
    ui.drain_keys()

    step = 0
    outcome = None
    timer = TrainingTimer()
    loop_overrun_count = 0
    max_loop_overrun_ms = 0.0

    while max_steps == 0 or step < max_steps or not auto_timeout_at_max_steps:
        if max_steps > 0 and step == max_steps and not auto_timeout_at_max_steps:
            print(
                f"\n[--no-auto-timeout] Reached max_steps={max_steps}; NOT auto-marking "
                "timeout — the policy keeps running. Press 1=SUCCESS / 0=TIMEOUT / "
                "9=FAILURE to end the episode."
            )
        loop_start = time.perf_counter()
        ui.progress.step = step
        ui.progress.marks = len(subtask_frames)
        ui.render_monitor(obs["image"])

        # ---- Keyboard check ------------------------------------------------
        key = ui.read_key()
        if key == "1":
            outcome = "success"
            break
        elif key == "0":
            outcome = "timeout"
            break
        elif key == "9":
            outcome = "failure"
            break
        elif key == "r":
            outcome = "restart"
            break
        elif key == "q":
            outcome = "quit"
            break
        elif key == "g":
            record_subtask_mark(subtask_frames, step=step, subtask_marks=subtask_marks)

        # ---- Build state for data collection (ALWAYS, regardless of policy) ---
        with timer("build_obs"):
            state = np.concatenate(
                [
                    np.array(obs["robot_state"]["cartesian_position"], dtype=np.float32),
                    np.array([obs["robot_state"]["gripper_position"]], dtype=np.float32),
                ]
            )

        # ---- Policy inference (unified interface) --------------------------
        with timer("inference"):
            action = policy.predict(obs)

        # ---- Collect data BEFORE stepping ----------------------------------
        if save_data:
            with timer("data_collect"):
                episode_data["observations"].append(state.copy())
                episode_data["joint_positions"].append(
                    np.array(obs["robot_state"]["joint_positions"], dtype=np.float32)
                )
                for cam_key in save_cam_keys:
                    img = process_image(
                        obs["image"][cam_key], save_camera_height, save_camera_width
                    )
                    episode_data[f"image_{cam_key}"].append(img)

        # ---- Step the robot ------------------------------------------------
        previous_obs = obs
        with timer("env_step"):
            obs = env.step(
                action,
                # The relative-pose arm commands DROID's 'cartesian_position' space.
                # Wrappers without env_action_space fall back to action_space.
                action_space=getattr(policy, "env_action_space", policy.action_space),
                gripper_action_space=getattr(policy, "gripper_action_space", None),
            )
            action_info = obs.pop("action_info")
            if save_data:
                obs = _refresh_obs_until_robot_state_timestamp_advances(
                    env,
                    previous_obs=previous_obs,
                    obs=obs,
                    max_wait_s=robot_state_refresh_max_wait_s,
                    poll_interval_s=robot_state_refresh_poll_interval_s,
                )
        step += 1

        # ---- Collect action + action_info + velocities AFTER stepping -----
        if save_data:
            with timer("data_collect"):
                # The canonical 'action' column is ALWAYS cartesian_velocity + gripper
                # (its declared feature names are vel_x..vel_yaw, gripper_action), for
                # EVERY arm — velocity, position, or UMI-relative. DROID's IK solver
                # reports the commanded action in every space (extract_action_info
                # requires all of them), so a position/relative arm's velocity is
                # available here too, and its ABSOLUTE pose command is preserved in the
                # typed action.cartesian_position + action.gripper_position columns
                # (append_action_info below). This keeps the canonical column consistent
                # and correctly-named across a MIXED-arm eval dataset — a position pose
                # must never be written under velocity names. Downstream position/relative
                # training reads the typed action.cartesian_position column (or
                # observation.state), never the canonical column.
                canonical_action = build_canonical_action(action_info)
                episode_data["actions"].append(canonical_action)
                episode_data["rewards"].append(0.0)
                episode_data["dones"].append(0)
                append_action_info(episode_data, action_info)
                append_joint_velocities(episode_data, obs)
                append_franka_telemetry(episode_data, obs)
                append_cartesian_velocities(episode_data, obs, previous_obs=previous_obs)

        if step % 50 == 0:
            if max_steps > 0:
                print(f"  Step {step}/{max_steps}...")
            else:
                print(f"  Step {step}...")
            timer.print_stats(prefix="  ")

        # ---- Wall-clock-aware sleep ----------------------------------------
        elapsed = time.perf_counter() - loop_start
        remaining = 1.0 / freq - elapsed
        if remaining > 0:
            time.sleep(remaining)
        else:
            overrun_ms = -remaining * 1000.0
            loop_overrun_count += 1
            max_loop_overrun_ms = max(max_loop_overrun_ms, overrun_ms)
            if warn_loop_overruns:
                print(
                    f"  WARNING: Loop step at {step} took {elapsed * 1000:.0f}ms, "
                    f"exceeding {1000 / freq:.0f}ms budget by {overrun_ms:.0f}ms"
                )

    # Print final timing summary for the episode
    if print_timing_summary and step > 0:
        print(f"\n  Episode timing ({step} steps):")
        timer.print_stats(prefix="    ")
        if loop_overrun_count:
            print(
                f"    Loop overruns: {loop_overrun_count}/{step} steps "
                f"(max {max_loop_overrun_ms:.0f}ms over budget)"
            )

    # If we exhausted max_steps without a keypress, treat as truncation. Only reachable
    # when auto-timeout is enabled — with --no-auto-timeout the loop runs until an
    # operator key sets the outcome, so it never falls through here.
    if outcome is None:
        assert max_steps > 0 and auto_timeout_at_max_steps, (
            "outcome is None only when auto-timeout-at-max-steps is enabled with a step cap"
        )
        outcome = "timeout"
        print(f"Max steps ({max_steps}) reached — marking as timeout (truncated).")

    # ---- Terminal frame (T+1 pattern) for dataset --------------------------
    if save_data and step > 0 and outcome not in {"quit", "restart"}:
        is_success = outcome == "success"
        # Final observation
        state = np.concatenate(
            [
                np.array(obs["robot_state"]["cartesian_position"], dtype=np.float32),
                np.array([obs["robot_state"]["gripper_position"]], dtype=np.float32),
            ]
        )
        episode_data["observations"].append(state.copy())
        episode_data["joint_positions"].append(
            np.array(obs["robot_state"]["joint_positions"], dtype=np.float32)
        )
        for cam_key in save_cam_keys:
            img = process_image(obs["image"][cam_key], save_camera_height, save_camera_width)
            episode_data[f"image_{cam_key}"].append(img)

        # Finalize: set reward/done on last valid frame, append padded terminal row
        is_terminal = outcome in ("success", "failure")
        finalize_episode_data(
            episode_data,
            obs,
            is_success=is_success,
            is_terminal=is_terminal,
            subtask_frames=subtask_frames,
        )

    # Show the next operator target card (if any) BEFORE the blocking reset, so the
    # operator can start staging the next scene while the robot resets. The callback
    # only renders/updates the OpenCV target window on the main thread; it must not
    # touch the RobotEnv. A UI failure must still reset the robot, then propagate:
    # continuing with a stale placement reference could invalidate the next episode.
    try:
        ui.finish_rollout(outcome)
        if pre_reset_callback is not None:
            pre_reset_callback(outcome)
    finally:
        verified_reset(
            env,
            randomize=randomize_reset,
            max_retries=reset_max_retries,
            retry_delay_s=reset_retry_delay_s,
            retry_backoff=reset_retry_backoff,
            max_retry_delay_s=reset_max_retry_delay_s,
        )
        policy.reset()

    if outcome == "quit":
        ui.set_phase("Stopping session", "Robot reset finished. Saving results.")
    elif ui.progress.remaining == 0:
        ui.set_phase("Finishing session", "Saving results. All planned rollouts finished.")
    else:
        ui.set_phase(
            "Preparing next rollout",
            f"Last rollout: {outcome}. Robot reset finished; next setup is shown.",
        )
    if outcome == "restart":
        episode_data = None

    return step, outcome, episode_data, subtask_frames


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy a trained policy on the real Franka robot via DROID"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Policy MODEL_ID: hf://NAMESPACE/REPO[@REV] or a local checkpoint directory.",
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
        help="Max steps per episode safety limit (default: 0 = unlimited)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Number of episodes to run (default: 0 = unlimited)",
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
        help="Camera suffix filter (default: '_left', matching training)",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=DEFAULT_CAMERA_KEYS,
        help=(
            "Comma-separated camera keys or ZED serials to use; defaults to the "
            f"current robot station's set ({DEFAULT_CAMERA_KEYS}). "
            "When set, only those ZED cameras are opened. Pass an empty string "
            "to use --camera-filter discovery instead."
        ),
    )

    # Robot
    parser.add_argument(
        "--randomize-reset",
        action="store_true",
        help="Add noise to reset pose",
    )
    add_robot_reset_args(parser)

    # Dataset saving (--dataset-name implies saving)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name; providing it enables saving (e.g. 'marker-dp-rollouts')",
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
        default=None,
        help=TASK_NAME_HELP + " Required with --dataset-name.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub after session",
    )
    add_hf_namespace_arg(parser)
    add_license_arg(parser)
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make Hub dataset private (only with --push-to-hub)",
    )
    parser.add_argument(
        "--no-save-failures",
        action="store_true",
        help="Skip saving failed episodes (default: save both successes and failures)",
    )

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

    from mulligan.real.policy.loader import REAL_PROTOCOL_N_ACTION_STEPS

    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=REAL_PROTOCOL_N_ACTION_STEPS,
        help=(
            "Action chunk steps to execute before re-planning. Defaults to the "
            f"real-robot protocol exec horizon ({REAL_PROTOCOL_N_ACTION_STEPS}); "
            "prediction horizon comes from the checkpoint. "
            "Lower = more frequent re-planning."
        ),
    )
    # IDQL options
    parser.add_argument(
        "--num-action-samples",
        type=int,
        default=None,
        help="Override number of action candidates for IDQL argmax (default: from artifact metadata)",
    )

    add_operator_ui_args(parser, cards=False)
    args = parser.parse_args()
    if args.task_name is None:
        if args.dataset_name is not None:
            parser.error("--task-name is required with --dataset-name")
    else:
        args.task_name = get_task_spec(args.task_name).task_name
    if args.push_to_hub:
        if args.dataset_name is None:
            parser.error("--push-to-hub requires --dataset-name")
        try:
            resolve_push_repo_id(args.dataset_name, args.hf_namespace)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main():
    args = parse_args()
    if args.reset_max_retries < 1:
        raise ValueError("--reset-max-retries must be >= 1")
    if args.reset_retry_delay_s < 0:
        raise ValueError("--reset-retry-delay-s must be non-negative")
    if args.reset_retry_backoff < 1:
        raise ValueError("--reset-retry-backoff must be >= 1")
    if args.reset_max_retry_delay_s < 0:
        raise ValueError("--reset-max-retry-delay-s must be non-negative")
    if args.robot_state_refresh_max_wait_s < 0:
        raise ValueError("--robot-state-refresh-max-wait-s must be non-negative")
    if args.robot_state_refresh_poll_interval_s <= 0:
        raise ValueError("--robot-state-refresh-poll-interval-s must be > 0")

    device = args.device or auto_detect_device()
    requested_camera_keys = parse_camera_keys(args.camera_keys, args.camera_filter)

    # ---- Dataset config ----------------------------------------------------
    save_data = args.dataset_name is not None
    dataset = None
    dataset_name = args.dataset_name
    dataset_path = Path(args.dataset_path) / dataset_name if save_data else None
    saved_episode_count = 0
    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    save_future = None

    # ---- Load policy -------------------------------------------------------
    print("=" * 60)
    print("Real Robot Policy Rollout")
    print("=" * 60)

    # Capture the operator-requested CLI camera resolution NOW, before the
    # policy-resolution auto-detect below can mutate args.camera_height/width.
    # Saved rollout datasets must use the CLI dims, not the checkpoint's
    # training resolution.
    original_camera_height = args.camera_height
    original_camera_width = args.camera_width

    from mulligan.real.policy.loader import load_policy_by_model_id
    from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

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
        print(
            f"WARNING: Policy was trained at {entry.camera_height}x{entry.camera_width} but "
            f"--camera-height/--camera-width is {args.camera_height}x{args.camera_width}"
        )
        print(
            f"  Overriding to {entry.camera_height}x{entry.camera_width} to match training "
            "resolution"
        )
        args.camera_height = entry.camera_height
        args.camera_width = entry.camera_width

    print(f"Policy: {policy.__class__.__name__}")
    print(f"Device: {device}")
    if isinstance(policy, VisionIDQLRealWorldPolicy):
        if args.num_action_samples is not None:
            policy.num_action_samples = args.num_action_samples
            print(f"  Overriding num_action_samples → {args.num_action_samples}")
        print(f"Action dim: {policy.action_dim}, chunk: {policy.n_action_steps} steps")
        print(f"N action samples: {policy.num_action_samples}")
        print(f"IQL state_dim={policy.iql_state_dim}, action_dim={policy.iql_action_dim}")
    elif args.num_action_samples is not None:
        raise SystemExit(
            "--num-action-samples applies to Vision-IDQL policies only; "
            f"{args.model} loaded as {policy.__class__.__name__}."
        )

    print(f"Control freq: {args.freq} Hz")
    print(f"Max steps/episode: {args.max_steps if args.max_steps > 0 else 'unlimited'}")
    print(f"Camera filter: *{args.camera_filter}")
    if requested_camera_keys is not None:
        print(f"Camera keys: {requested_camera_keys}")

    if save_data:
        print("\nData saving: ENABLED")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Dataset name: {dataset_name}")
        print(f"  Save failures: {not args.no_save_failures}")
    else:
        print("\nData saving: DISABLED")
    print()

    # ---- Initialise hardware -----------------------------------------------
    if requested_camera_keys is not None:
        requested_serials = camera_serials_from_keys(requested_camera_keys)
        print(f"Restricting ZED cameras to serials: {requested_serials}")
        restrict_zed_cameras_to_serials(requested_serials)

    import mulligan.real.robot.camera_config  # noqa: F401  (patches ZED to 15fps)
    from mulligan.real.robot.droid_compat import RobotEnv

    print("Initializing robot environment...")
    if policy.action_space == "cartesian_position":
        # Relative-pose arm: absolute 7D euler poses and the absolute 0/1 gripper.
        env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    else:
        env = RobotEnv(action_space="cartesian_velocity")

    ui = OperatorUI.from_args(args, cards=False)
    _ = ui.keyboard  # enter cbreak mode now, before the first episode
    print("Keyboard listener ready (terminal mode, works over SSH)")
    print()

    # Initial reset to known position
    print("Performing initial robot reset...")
    verified_reset(
        env,
        max_retries=args.reset_max_retries,
        retry_delay_s=args.reset_retry_delay_s,
        retry_backoff=args.reset_retry_backoff,
        max_retry_delay_s=args.reset_max_retry_delay_s,
    )
    print("Robot reset complete.")
    print()

    # ---- Episode loop ------------------------------------------------------
    all_camera_keys: list[str] = []  # ALL cameras for dataset saving
    episode_results = []

    try:
        episode_num = 0
        while True:
            episode_num += 1
            if args.num_episodes > 0 and episode_num > args.num_episodes:
                print(f"\nCompleted {args.num_episodes} episodes. Stopping.")
                break

            print(f"\n{'=' * 60}")
            print(
                f"Episode {episode_num}"
                + (f" / {args.num_episodes}" if args.num_episodes > 0 else "")
            )
            print(f"{'=' * 60}")

            num_steps, outcome, episode_data, _subtask_frames = rollout_episode(
                env,
                policy,
                ui,
                freq=args.freq,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                save_camera_height=original_camera_height,
                save_camera_width=original_camera_width,
                camera_filter=args.camera_filter,
                max_steps=args.max_steps,
                randomize_reset=args.randomize_reset,
                save_data=save_data,
                all_camera_keys=all_camera_keys,
                requested_camera_keys=requested_camera_keys,
                reset_max_retries=args.reset_max_retries,
                reset_retry_delay_s=args.reset_retry_delay_s,
                reset_retry_backoff=args.reset_retry_backoff,
                reset_max_retry_delay_s=args.reset_max_retry_delay_s,
                robot_state_refresh_max_wait_s=args.robot_state_refresh_max_wait_s,
                robot_state_refresh_poll_interval_s=args.robot_state_refresh_poll_interval_s,
            )

            episode_results.append({"steps": num_steps, "outcome": outcome})
            label = outcome.upper()
            print(f"\nEpisode {episode_num}: {label} ({num_steps} steps)")

            # ---- Persist per-chunk IDQL candidate diagnostics (sidecar) -----
            # Written for every episode the policy generated chunks for (incl.
            # failures/timeouts), so blind evals keep candidate-level data. A
            # write failure must not kill a robot episode (the SR experiment
            # stays valid without diagnostics), so this is loud-warn-only.
            chunk_infos = getattr(policy, "last_episode_chunk_infos", None)
            if chunk_infos:
                try:
                    # Inside the dataset dir so the hub push carries them.
                    sidecar_root = (
                        dataset_path / CHUNK_INFO_DIRNAME
                        if dataset_path is not None
                        else Path(args.dataset_path) / CHUNK_INFO_DIRNAME
                    )
                    sidecar_root.mkdir(parents=True, exist_ok=True)
                    sidecar_path = sidecar_root / f"episode_{episode_num:04d}.jsonl"
                    with open(sidecar_path, "w") as f:
                        header = {
                            "episode_num": episode_num,
                            "outcome": outcome,
                            "num_steps": num_steps,
                            "num_chunks": len(chunk_infos),
                        }
                        f.write(json.dumps(header) + "\n")
                        for info in chunk_infos:
                            f.write(json.dumps(info) + "\n")
                except Exception as exc:  # noqa: BLE001
                    print(f"\nWARNING: failed to write chunk-info sidecar: {exc!r}")
                policy.last_episode_chunk_infos = []

            if outcome == "quit":
                break

            # ---- Save episode to dataset -----------------------------------
            is_success = outcome == "success"
            should_save = is_success or (not args.no_save_failures)

            if save_data and episode_data is not None and should_save and num_steps > 0:
                # Discover camera data keys from episode_data
                cam_data_keys = sorted(k for k in episode_data if k.startswith("image_"))

                # Lazy-init dataset on first save
                if dataset is None:
                    dataset = open_or_create_rollout_dataset(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        episode_data=episode_data,
                        cam_data_keys=cam_data_keys,
                        fps=args.freq,
                    )

                # Wait for previous background save before starting a new one
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
                    episode_data=episode_data,
                    episode_success=is_success,
                    camera_keys=cam_data_keys,
                    task_name=args.task_name,
                    saved_episode_count=saved_episode_count,
                    default_source=DataSource.AUTONOMOUS,
                    camera_name_fn=serial_key_to_role,
                )
                saved_episode_count += 1

            elif save_data and not should_save:
                print("Episode marked as failure -- not saved (--no-save-failures active)")

            # Wait for user confirmation before starting next episode
            gate = ui.gate(
                prompt="Ready for the next episode",
                on_reset=lambda: verified_reset(env, randomize=args.randomize_reset),
                any_key_starts=True,
            )
            if gate is GateOutcome.QUIT:
                print("Quitting.")
                outcome = "quit"

            if outcome == "quit":
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")
    finally:
        # Wait for in-flight background save
        if save_future is not None:
            print("Waiting for background episode save to finish...")
            wait_for_background_save(save_future, description="Final background episode save")
        save_executor.shutdown(wait=True)

        if save_data and dataset is not None:
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

    # ---- Session summary ---------------------------------------------------
    if episode_results:
        print()
        print("=" * 60)
        print("Session Summary")
        print("=" * 60)

        successes = sum(1 for r in episode_results if r["outcome"] == "success")
        timeouts = sum(1 for r in episode_results if r["outcome"] == "timeout")
        failures = sum(1 for r in episode_results if r["outcome"] == "failure")
        total = successes + timeouts + failures  # exclude quit

        for i, r in enumerate(episode_results, 1):
            print(f"  Episode {i}: {r['outcome'].upper():>7s}  ({r['steps']} steps)")

        print(f"\nTotal episodes: {len(episode_results)}")
        if total > 0:
            print(f"Success rate: {successes}/{total} = {successes / total:.1%}")
            if timeouts > 0:
                print(f"Timeouts: {timeouts}")
            if failures > 0:
                print(f"Failures: {failures}")
        if save_data:
            print(f"Episodes saved: {saved_episode_count}")
        print("Done.")


if __name__ == "__main__":
    main()
