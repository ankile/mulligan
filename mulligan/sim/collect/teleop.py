# ruff: noqa: E402
"""SpaceMouse teleoperation of the Square tasks into a LeRobot dataset (R0 demos).

Every saved frame is labeled ``source`` = 1 (human) and ``success`` from the
episode outcome, so teleop data combines with DAgger data. Frames are recorded
while the operator gives input, the gripper moves or an object moves; the
physics always steps, so nothing teleports and idle frames are skipped.

Usage (the R0 collections of the paper; ``python`` instead of ``mjpython`` on Linux):

    mjpython -m mulligan.sim.collect.teleop --r0-preset square_narrow
    mjpython -m mulligan.sim.collect.teleop --r0-preset square_broad

``--r0-preset`` sets the defaults of the paper's R0 session of that task (see
``R0_PRESETS``): the blinded uniform/Sobol start list of
``data/sim/start_manifests/<task>/r00/blind_inputs`` in file order, both cameras,
auto-save on success, 200 (Narrow) / 400 (Broad) episodes, SpaceMouse sensitivity
1.2/1.2 on Narrow and the defaults (1.0/1.5) on Broad, written to
``outputs/sim/data/sim-<task>-c00-teleop-mixed`` where ``scripts/sim/split_round.sh
--round 0`` reads it. Flags given explicitly override the preset. Spelled out, the
Square-Narrow session is:

    mjpython -m mulligan.sim.collect.teleop --env NutAssemblySquare --robot Panda \\
        --cameras agentview,robot0_eye_in_hand --save-data \\
        --dataset-path outputs/sim/data --dataset-name sim-square-narrow-c00-teleop-mixed \\
        --pos-sensitivity 1.2 --rot-sensitivity 1.2 \\
        --auto-save-on-success --target-episodes 200 --sampler list --no-sampler-shuffle \\
        --initial-states-file data/sim/start_manifests/square_narrow/r00/blind_inputs/square_narrow_r0_mixed.json

Add ``--push-to-hub --hub-namespace <user-or-org>`` to push the dataset to
``<hub-namespace>/<dataset-name>`` at the end.

Controls:
    SpaceMouse: move the end effector (6-DoF); left button toggles the gripper
    '1': save the episode as a success
    '0': end the episode as a failure (saved only with --save-failures)
    Ctrl+C: quit
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

if platform.system() == "Darwin" and not os.environ.get("MUJOCO_GL"):
    # CGL contexts are not tied to the main thread, so camera observations
    # work next to the onscreen viewer.
    os.environ["MUJOCO_GL"] = "cgl"

from mulligan import apply_runtime_patches

apply_runtime_patches()

import robosuite.macros as macros
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.constants import DataSource, EpisodeOutcome
from mulligan.data.env_state_names import get_environment_state_names
from mulligan.real.collect.hf_utils import add_license_arg
from mulligan.sim.collect.utils import (
    KeyboardListener,
    OBJECT_VEL_THRESHOLD,
    check_objects_moving,
    robot_state,
    wait_for_save,
)
from mulligan.sim.envs import configure_viewer_shadows, create_robosuite_env, unwrap_env

# Camera observations in image convention (origin top-left); must be set before
# the env is created.
macros.IMAGE_CONVENTION = "opencv"

REPO_ROOT = Path(__file__).resolve().parents[3]
_R0_INPUTS = REPO_ROOT / "data" / "sim" / "start_manifests"
_R0_PRESET_COMMON = {
    "robot": "Panda",
    "cameras": "agentview,robot0_eye_in_hand",
    "save_data": True,
    "dataset_path": "outputs/sim/data",
    "auto_save_on_success": True,
    "sampler": "list",
    "sampler_shuffle": False,
}
# Defaults of --r0-preset: the settings of the paper's R0 teleop sessions.
R0_PRESETS = {
    "square_narrow": {
        **_R0_PRESET_COMMON,
        "env": "NutAssemblySquare",
        "dataset_name": "sim-square-narrow-c00-teleop-mixed",
        "pos_sensitivity": 1.2,
        "rot_sensitivity": 1.2,
        "target_episodes": 200,
        "initial_states_file": str(
            _R0_INPUTS / "square_narrow/r00/blind_inputs/square_narrow_r0_mixed.json"
        ),
    },
    "square_broad": {
        **_R0_PRESET_COMMON,
        "env": "Square_D1",
        "dataset_name": "sim-square-broad-c00-teleop-mixed",
        "target_episodes": 400,
        "initial_states_file": str(
            _R0_INPUTS / "square_broad/r00/blind_inputs/square_broad_r0_mixed.json"
        ),
    },
}


def save_example_images(obs, output_dir, camera_keys):
    """Save one PNG per camera observation."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    for cam_key in camera_keys:
        img_array = obs[cam_key]
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_path = output_dir / f"{cam_key}_example.png"
            Image.fromarray(img_array.astype(np.uint8)).save(img_path)
            print(f"  Saved {img_path}")
        else:
            print(f"  Unexpected shape for {cam_key}: {img_array.shape}")


def _format_eta(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _sampler_remaining(sampler) -> int:
    return max(0, len(sampler.planned_points) - int(sampler._next_idx))


def _print_collection_progress(
    *,
    sampler_label,
    sampler,
    base_episode_count,
    saved_episode_count,
    target_episodes,
    session_start_time,
):
    elapsed = time.time() - session_start_time
    rate = saved_episode_count / elapsed if elapsed > 0 and saved_episode_count > 0 else None
    total_episode_count = base_episode_count + saved_episode_count
    lines = ["", "COLLECTION PROGRESS"]

    if target_episodes is not None:
        remaining = max(0, target_episodes - total_episode_count)
        lines.extend(
            [
                f"Total episodes collected: {total_episode_count} / {target_episodes}",
                f"This launch saved: {saved_episode_count}",
                f"Remaining to target: {remaining}",
                f"ETA to target: {_format_eta(remaining / rate if rate else None)}",
            ]
        )
    elif sampler is not None:
        remaining = _sampler_remaining(sampler)
        lines.extend(
            [
                f"Total episodes collected: {sampler._next_idx} / {len(sampler.planned_points)}",
                f"Remaining to target: {remaining}",
                f"ETA to target: {_format_eta(remaining / rate if rate else None)}",
            ]
        )
    else:
        lines.extend(
            [
                f"Total episodes collected: {total_episode_count}",
                f"This launch saved: {saved_episode_count}",
                f"Elapsed: {_format_eta(elapsed)}",
            ]
        )

    if sampler is not None:
        lines.extend(
            [
                f"{sampler_label} manifest: {sampler._next_idx} / {len(sampler.planned_points)}",
                f"{sampler_label} manifest remaining: {_sampler_remaining(sampler)}",
            ]
        )
    print("\n".join(lines))


def _close_episode(episode_data, obs, camera_keys, save_data):
    """Append the final observation and pad action/reward/done (T+1 observations)."""
    episode_data["observations"].append(robot_state(obs))
    episode_data["environment_state"].append(obs["object-state"])
    if save_data:
        for cam_key in camera_keys:
            if cam_key in obs:
                episode_data[cam_key].append(obs[cam_key].copy())
    episode_data["actions"].append(episode_data["actions"][-1])
    episode_data["rewards"].append(episode_data["rewards"][-1])
    episode_data["dones"].append(episode_data["dones"][-1])


def teleop_episode(
    env,
    device,
    kbd_listener,
    max_fr=20,
    save_data=False,
    save_example_images_flag=False,
    output_dir=None,
    has_renderer=True,
    camera_names=None,
    record_gripper_motion=True,
    gripper_vel_threshold=0.01,
    auto_save_on_success=False,
    sampler=None,
    sampler_label=None,
):
    """Teleoperate one episode from a fresh reset.

    Returns ``(episode_data, success)``: ``success`` is True ('1' or auto-save)
    or False ('0').
    """
    obs = env.reset()

    if sampler is not None:
        obs, where = _place_sampled_start(env, sampler)
        print(f"  {sampler_label}: {where}")

    # Small random warmup actions: the viewer opens and starts vary a little.
    num_warmup_steps = np.random.randint(2, 41)
    action_dim = env.action_spec[0].shape[0]
    random_gripper_action = 0.0 if np.random.rand() < 0.5 else -1.0
    for _ in range(num_warmup_steps):
        warmup_action = np.random.randn(action_dim) * 0.01
        warmup_action[-1] = random_gripper_action
        obs, _, _, _ = env.step(warmup_action)
        if has_renderer:
            env.render()
        time.sleep(0.01)

    device.start_control()
    device.reset_gripper()

    # Simulator state after warmup: the episode's true start, stored for exact replays.
    base_env = unwrap_env(env)
    episode_data = {
        "observations": [],
        "environment_state": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "initial_sim_qpos": base_env.sim.data.qpos.copy(),
        "initial_sim_qvel": base_env.sim.data.qvel.copy(),
    }
    camera_keys = [f"{cam}_image" for cam in camera_names or []]
    for cam_key in camera_keys:
        if cam_key in obs:
            episode_data[cam_key] = []

    print("\nEpisode started. Use the SpaceMouse to control the robot.")
    if auto_save_on_success:
        print("Auto-save on success is on. Press '0' for FAILURE, Ctrl+C to quit.")
    else:
        print("Press '1' for SUCCESS and save, '0' for FAILURE, Ctrl+C to quit.")
    print(f"Recording on input, gripper motion or object motion > {OBJECT_VEL_THRESHOLD} m/s")
    print(f"Observation keys: {list(obs.keys())}")
    if save_example_images_flag and output_dir is not None:
        image_keys = [k for k in obs if k.endswith("_image")]
        if image_keys:
            save_example_images(obs, output_dir, image_keys)
        else:
            print("No camera observations found in obs dict")
    print()

    step_count = 0
    prev_gripper = device.control_gripper
    gripper_is_moving = False
    success_announced = False
    done_announced = False
    recording_started = False

    while True:
        start = time.time()

        key = kbd_listener.read_key()
        if key in ("1", "0"):
            success = key == "1"
            print("\n" + "=" * 60)
            print(f"{'SUCCESS' if success else 'FAILURE'}! Episode length: {step_count} steps")
            print("=" * 60 + "\n")
            if step_count > 0:
                _close_episode(episode_data, obs, camera_keys, save_data)
            return episode_data, success

        control = device.control
        gripper = device.control_gripper

        # Base scale 0.0055 (10% above robosuite's 0.005).
        dpos = control[:3] * 0.0055 * device.pos_sensitivity
        raw_drot = control[3:6] * 0.0055 * device.rot_sensitivity
        # Device [roll, pitch, yaw] -> robot [pitch, roll, -yaw].
        drot = raw_drot[[1, 0, 2]]
        drot[2] = -drot[2]
        dpos = np.clip(dpos * 125, -1, 1)
        drot = np.clip(drot * 50, -1, 1)

        if gripper != prev_gripper and record_gripper_motion:
            gripper_is_moving = True
            print("Gripper command changed, tracking motion until complete...")
        gripper_action = 1.0 if gripper == 1 else -1.0
        action = np.concatenate([dpos, drot, [gripper_action]])

        has_input = (
            np.any(np.abs(control) > 0.025) or (gripper != prev_gripper) or gripper_is_moving
        )
        if not recording_started and (np.any(np.abs(control) > 0.1) or gripper != prev_gripper):
            # A clear input starts recording, so SpaceMouse drift does not.
            recording_started = True
            print("Recording started (detected clear user input)")
        should_record = recording_started and (has_input or check_objects_moving(env))

        if should_record:
            episode_data["observations"].append(robot_state(obs))
            episode_data["environment_state"].append(obs["object-state"])
            episode_data["actions"].append(action.copy())
            if save_data:
                for cam_key in camera_keys:
                    if cam_key in obs:
                        episode_data[cam_key].append(obs[cam_key].copy())
            action_to_execute = action
        else:
            # Zero arm motion, keep the gripper command.
            action_to_execute = np.concatenate([np.zeros(6), [gripper_action]])

        obs, reward, done, _ = env.step(action_to_execute)
        if has_renderer:
            env.render()

        if should_record:
            episode_data["rewards"].append(reward)
            # Sparse success reward ends the MDP.
            if reward == 1.0:
                done = True
            episode_data["dones"].append(done)

            if not success_announced and env._check_success():
                success_announced = True
                print("\n" + "*" * 60)
                print("TASK SUCCESSFUL! Environment detected task completion!")
                print("*" * 60 + "\n")
                if auto_save_on_success:
                    print(f"Auto-save: SUCCESS, episode length {step_count} steps")
                    _close_episode(episode_data, obs, camera_keys, save_data)
                    return episode_data, True

            if gripper_is_moving and record_gripper_motion:
                max_qvel = np.max(np.abs(obs["robot0_gripper_qvel"]))
                if max_qvel < gripper_vel_threshold:
                    gripper_is_moving = False
                    print(f"Gripper motion complete (max qvel: {max_qvel:.4f})")
            step_count += 1

        prev_gripper = gripper

        if done and not done_announced:
            done_announced = True
            print(f"\nEpisode completed! Length: {step_count} steps")
            print("Press '1' to mark as SUCCESS and save, '0' to mark as FAILURE")

        if max_fr:
            remaining = 1 / max_fr - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)


def _place_sampled_start(env, sampler):
    """Place the next sampler start; return (observations, description)."""
    from mulligan.sampling.sobol import SquareD1SobolSampler
    from mulligan.sim.placement import place_nut, place_square_broad_state

    if isinstance(sampler, SquareD1SobolSampler):
        nut_state, peg_state = sampler.sample_next()
        obs = place_square_broad_state(env, sampler, nut_state, peg_state)
        (x, y, yaw), (px, py) = nut_state, peg_state
        return obs, (
            f"nut at x={x:.4f}, y={y:.4f}, yaw={np.degrees(yaw):.1f}deg; "
            f"peg at x={px:.4f}, y={py:.4f}"
        )
    x, y, yaw = sampler.sample_next()
    obs = place_nut(env, sampler.to_qpos(x, y, yaw))
    return obs, f"nut placed at x={x:.4f}, y={y:.4f}, yaw={np.degrees(yaw):.1f}deg"


def _record_sampler_start(sampler, env_state_0) -> str:
    """Record the saved episode's start with the sampler; return a description."""
    from mulligan.sampling.sobol import SquareD1SobolSampler
    from mulligan.utils.state_to_grid import (
        extract_nut_pose_from_env_state,
        extract_peg_pos_from_env_state,
    )

    if isinstance(sampler, SquareD1SobolSampler):
        x, y, yaw = extract_nut_pose_from_env_state(env_state_0, task="Square_D1")
        peg_x, peg_y, _ = extract_peg_pos_from_env_state(env_state_0)
        sampler.record_state((x, y, yaw), (peg_x, peg_y))
        return f"recorded state - {len(sampler.collected_points)} total points"
    x, y, yaw = extract_nut_pose_from_env_state(env_state_0)
    sampler.record_state(x, y, yaw)
    return (
        f"recorded state ({x:.4f}, {y:.4f}, {np.degrees(yaw):.1f}deg) - "
        f"{len(sampler.collected_points)} total points"
    )


def _build_sampler(args):
    """Sobol or list start-state sampler for the Square tasks (None without --sampler)."""
    from mulligan.sampling.sobol import (
        SobolSampler,
        SquareD1ListSampler,
        SquareD1SobolSampler,
        SquareListSampler,
    )

    if args.sampler is None:
        return None
    if args.sampler == "sobol":
        return SquareD1SobolSampler() if args.env == "Square_D1" else SobolSampler()
    if not args.initial_states_file:
        raise ValueError("--sampler list requires --initial-states-file")
    list_cls = SquareD1ListSampler if args.env == "Square_D1" else SquareListSampler
    return list_cls(
        states_file=Path(args.initial_states_file),
        seed=args.sampler_shuffle_seed,
        shuffle=args.sampler_shuffle,
    )


def save_episode_to_dataset(
    dataset,
    episode_data,
    episode_success,
    camera_names,
    task_name,
    saved_episode_count,
    base_episode_count=0,
):
    """Write one episode (runs in the background saver thread)."""
    num_frames = len(episode_data["actions"])
    for i in range(num_frames):
        frame = {
            "task": task_name,
            "observation.state": episode_data["observations"][i].astype(np.float32),
            "observation.environment_state": episode_data["environment_state"][i].astype(
                np.float32
            ),
            "action": episode_data["actions"][i].astype(np.float32),
            "steps_to_go": np.array([episode_data["steps_to_go"][i]], dtype=np.int64),
            "source": np.array([DataSource.HUMAN], dtype=np.int64),
            "success": np.array(
                [EpisodeOutcome.SUCCESS if episode_success else EpisodeOutcome.FAILURE],
                dtype=np.int64,
            ),
            "is_valid": np.array([0 if i == num_frames - 1 else 1], dtype=np.int64),
            "reward": np.array([episode_data["rewards"][i]], dtype=np.float32),
            "done": np.array([episode_data["dones"][i]], dtype=np.int64),
            "initial_sim_qpos": episode_data["initial_sim_qpos"].astype(np.float32),
            "initial_sim_qvel": episode_data["initial_sim_qvel"].astype(np.float32),
        }
        for cam_name in camera_names or []:
            cam_key = f"{cam_name}_image"
            if cam_key in episode_data:
                frame[f"observation.images.{cam_name}"] = episode_data[cam_key][i]
        dataset.add_frame(frame)

    # Encode in this thread (no process pool from the saver thread).
    dataset.save_episode(parallel_encoding=False)
    print()
    print("BACKGROUND SAVE COMPLETE")
    print(f"Saved episode outcome: {'SUCCESS' if episode_success else 'FAILURE'}")
    print(f"Dataset episodes total: {base_episode_count + saved_episode_count}")
    print(f"Session episodes saved: {saved_episode_count}")


def _dataset_features(env_name, episode_data, camera_names):
    env_state_dim = episode_data["environment_state"][0].shape[0]
    qpos_dim = episode_data["initial_sim_qpos"].shape[0]
    qvel_dim = episode_data["initial_sim_qvel"].shape[0]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (episode_data["observations"][0].shape[0],),
            "names": [
                "eef_pos_x",
                "eef_pos_y",
                "eef_pos_z",
                "eef_quat_x",
                "eef_quat_y",
                "eef_quat_z",
                "eef_quat_w",
                "gripper_qpos_left",
                "gripper_qpos_right",
            ],
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (env_state_dim,),
            "names": get_environment_state_names(env_name, env_state_dim),
        },
        "action": {
            "dtype": "float32",
            "shape": (episode_data["actions"][0].shape[0],),
            "names": [
                "delta_eef_pos_x",
                "delta_eef_pos_y",
                "delta_eef_pos_z",
                "delta_eef_rot_x",
                "delta_eef_rot_y",
                "delta_eef_rot_z",
                "gripper_action",
            ],
        },
        "steps_to_go": {"dtype": "int64", "shape": (1,), "names": ["steps_to_go"]},
        "source": {"dtype": "int64", "shape": (1,), "names": ["source_id"]},
        "success": {"dtype": "int64", "shape": (1,), "names": ["success_flag"]},
        "is_valid": {"dtype": "int64", "shape": (1,), "names": ["is_valid_flag"]},
        "reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
        "done": {"dtype": "int64", "shape": (1,), "names": ["done_flag"]},
        # Initial MuJoCo state, for exact environment resets.
        "initial_sim_qpos": {
            "dtype": "float32",
            "shape": (qpos_dim,),
            "names": [f"qpos_{i}" for i in range(qpos_dim)],
        },
        "initial_sim_qvel": {
            "dtype": "float32",
            "shape": (qvel_dim,),
            "names": [f"qvel_{i}" for i in range(qvel_dim)],
        },
    }
    for cam_name in camera_names or []:
        cam_key = f"{cam_name}_image"
        if cam_key in episode_data:
            features[f"observation.images.{cam_name}"] = {
                "dtype": "video",
                "shape": episode_data[cam_key][0].shape,
                "names": ["height", "width", "channels"],
            }
    return features


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Teleoperate the Square tasks with a SpaceMouse")
    parser.add_argument(
        "--r0-preset",
        choices=sorted(R0_PRESETS),
        default=None,
        help="Default every flag to the paper's R0 session of this task (explicit flags win)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="NutAssemblySquare",
        choices=["NutAssemblySquare", "Square_D1"],
        help="NutAssemblySquare (Square-Narrow) or Square_D1 (Square-Broad)",
    )
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller (default: the robot's default controller)",
    )
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.5)
    parser.add_argument("--max-fr", type=int, default=20, help="Control-rate limit in Hz")
    parser.add_argument("--camera", type=str, default="agentview", help="Viewer camera")
    parser.add_argument(
        "--save-images", action="store_true", help="Save example camera images each episode"
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated observation cameras (e.g. 'agentview,robot0_eye_in_hand')",
    )
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="./robosuite_images")
    parser.add_argument("--headless", action="store_true", help="No onscreen viewer")
    parser.add_argument("--visual-aids", action="store_true", help="Robosuite visual aids")
    parser.add_argument(
        "--no-record-gripper-motion",
        action="store_false",
        dest="record_gripper_motion",
        help="Do not record frames while the gripper is still moving",
    )
    parser.add_argument("--gripper-vel-threshold", type=float, default=0.01)
    parser.add_argument("--save-data", action="store_true", help="Save to a LeRobot dataset")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (default: {env}_{robot}_{timestamp})",
    )
    parser.add_argument(
        "--auto-save-on-success",
        action="store_true",
        help="Save as a success when the env reports success",
    )
    parser.add_argument(
        "--save-failures", action="store_true", help="Also save episodes ended with '0'"
    )
    parser.add_argument(
        "--target-episodes",
        type=int,
        default=None,
        help="Stop once the dataset has this many episodes",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the dataset to <hub-namespace>/<dataset-name> at the end",
    )
    parser.add_argument(
        "--hub-namespace",
        type=str,
        default=None,
        help="HF user or org to push to (required with --push-to-hub)",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private")
    add_license_arg(parser)
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["sobol", "list"],
        default=None,
        help="Start-state sampler: sobol or list (--initial-states-file). Default: env reset.",
    )
    parser.add_argument(
        "--initial-states-file",
        type=str,
        default=None,
        help="JSON with a top-level 'states' list (nut_x, nut_y, nut_yaw[, peg_x, peg_y])",
    )
    parser.add_argument("--sampler-shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--no-sampler-shuffle",
        action="store_false",
        dest="sampler_shuffle",
        help="Keep the --initial-states-file order",
    )
    preset = parser.parse_known_args(argv)[0].r0_preset
    if preset is not None:
        parser.set_defaults(**R0_PRESETS[preset])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.push_to_hub and not (args.save_data and args.hub_namespace):
        raise ValueError("--push-to-hub requires --save-data and --hub-namespace")

    if platform.system() == "Darwin" and not args.headless:
        import mujoco.viewer

        if not hasattr(mujoco.viewer, "_MJPYTHON"):
            print("ERROR: the MuJoCo viewer on macOS needs mjpython (GUI on the main thread):")
            print(f"  mjpython -m mulligan.sim.collect.teleop {' '.join(sys.argv[1:])}")
            print("or run with --headless.")
            sys.exit(1)

    from mulligan.teleop.spacemouse import RobosuiteSpaceMouse

    print("\nInitializing SpaceMouse...")
    device = RobosuiteSpaceMouse(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )
    print("\nInitializing keyboard listener...")
    kbd_listener = KeyboardListener()

    output_dir = Path(args.output_dir) if args.save_images else None
    camera_names = (
        [cam.strip() for cam in args.cameras.split(",") if cam.strip()] if args.cameras else None
    )
    env_kwargs = dict(
        robot_name=args.robot,
        camera_names=camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        controller=args.controller,
        render_camera=args.camera,
        has_renderer=not args.headless,
        visual_aids=args.visual_aids,
    )

    dataset = None
    dataset_name = None
    dataset_path = None
    base_episode_count = 0
    saved_episode_count = 0
    if args.save_data:
        dataset_name = args.dataset_name or (
            f"{args.env.lower()}_{args.robot.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        dataset_path = Path(args.dataset_path) / dataset_name
        print(f"\nData saving: ENABLED ({dataset_path})")
        if dataset_path.exists():
            dataset = LeRobotDataset.resume(repo_id=dataset_name, root=dataset_path)
            base_episode_count = dataset.num_episodes
            print(f"Resuming existing dataset with {base_episode_count} episodes")
    else:
        print("\nData saving: DISABLED")

    sampler = _build_sampler(args)
    sampler_label = (args.sampler or "").upper()
    if sampler is not None:
        if dataset_path is not None and dataset_path.exists():
            sampler.load_from_dataset(dataset_path)
        print(
            f"{sampler_label} sampling: {len(sampler.collected_points)} collected, "
            f"{sampler._next_idx}/{len(sampler.planned_points)} sampled "
            f"({_sampler_remaining(sampler)} remaining)"
        )
    if args.target_episodes is not None:
        print(f"Dataset target: {args.target_episodes} saved episodes")

    print("\nSetup complete! Ready to teleoperate.")
    print("=" * 60)

    save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    save_future = None
    session_start_time = time.time()
    env = None

    try:
        attempt_count = 0
        while True:
            total_episode_count = base_episode_count + saved_episode_count
            if args.target_episodes is not None and total_episode_count >= args.target_episodes:
                print(f"\nDataset target reached ({total_episode_count} saved episodes).")
                break
            if sampler is not None and _sampler_remaining(sampler) == 0:
                print(f"\n{sampler_label} sampler exhausted.")
                break

            attempt_count += 1
            print(f"\n{'=' * 60}")
            print(f"Episode {total_episode_count + 1} (attempt {attempt_count})")
            print(f"{'=' * 60}")
            _print_collection_progress(
                sampler_label=sampler_label,
                sampler=sampler,
                base_episode_count=base_episode_count,
                saved_episode_count=saved_episode_count,
                target_episodes=args.target_episodes,
                session_start_time=session_start_time,
            )

            # A fresh env per episode (the viewer does not survive a reset).
            if env is not None:
                env.close()
            print(f"Creating environment: {args.env} with robot: {args.robot}")
            env = create_robosuite_env(args.env, **env_kwargs)
            if not args.headless:
                configure_viewer_shadows(env)

            episode_data, episode_success = teleop_episode(
                env,
                device,
                kbd_listener,
                max_fr=args.max_fr,
                save_data=args.save_data,
                save_example_images_flag=args.save_images,
                output_dir=output_dir,
                has_renderer=not args.headless,
                camera_names=camera_names,
                record_gripper_motion=args.record_gripper_motion,
                gripper_vel_threshold=args.gripper_vel_threshold,
                auto_save_on_success=args.auto_save_on_success,
                sampler=sampler,
                sampler_label=sampler_label,
            )

            # episode_data holds T+1 frames (the last one padded).
            episode_length = len(episode_data["actions"])
            print("\nEpisode Summary:")
            print(f"  Steps: {max(0, episode_length - 1)}")
            print(f"  Total Reward: {sum(episode_data['rewards'][:-1]):.3f}")
            episode_data["steps_to_go"] = [episode_length - 1 - t for t in range(episode_length)]

            should_save = episode_success or args.save_failures
            if args.save_data and should_save and episode_length > 0:
                if sampler is not None:
                    where = _record_sampler_start(sampler, episode_data["environment_state"][0])
                    print(f"  {sampler_label}: {where}")
                if dataset is None:
                    print(f"Creating new dataset at {dataset_path}")
                    dataset = LeRobotDataset.create(
                        repo_id=dataset_name,
                        fps=20,
                        root=str(dataset_path),
                        robot_type=args.robot.lower(),
                        features=_dataset_features(args.env, episode_data, camera_names),
                    )
                if save_future is not None:
                    wait_for_save(save_future)
                saved_episode_count += 1
                print("Saving episode in background...")
                save_future = save_executor.submit(
                    save_episode_to_dataset,
                    dataset=dataset,
                    episode_data=episode_data,
                    episode_success=episode_success,
                    camera_names=camera_names,
                    task_name=f"{args.env}_{args.robot}",
                    saved_episode_count=saved_episode_count,
                    base_episode_count=base_episode_count,
                )
                _print_collection_progress(
                    sampler_label=sampler_label,
                    sampler=sampler,
                    base_episode_count=base_episode_count,
                    saved_episode_count=saved_episode_count,
                    target_episodes=args.target_episodes,
                    session_start_time=session_start_time,
                )
            elif args.save_data and not episode_success:
                print("Failure not saved (use --save-failures to save failures)")

            print("\nReady for next episode...")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        print("\nCleaning up...")
        save_error = None
        if save_future is not None:
            print("Waiting for background episode save to finish...")
            try:
                save_future.result()
            except Exception as err:
                save_error = err
                print(f"ERROR: background save failed: {err}")
        save_executor.shutdown(wait=True)

        if dataset is not None:
            total = base_episode_count + saved_episode_count
            print(f"Finalizing dataset ({total} episodes, {saved_episode_count} this session)")
            dataset.finalize()
            if args.push_to_hub and save_error is None:
                repo_id = f"{args.hub_namespace}/{dataset_name}"
                print(f"\nPushing dataset to https://huggingface.co/datasets/{repo_id} ...")
                dataset.repo_id = repo_id
                dataset.push_to_hub(private=args.private, license=args.license)

        print("Stopping keyboard listener...")
        kbd_listener.close()
        print("Closing SpaceMouse...")
        device.close()
        if env is not None:
            env.close()
        if save_error is not None:
            raise RuntimeError("An episode save failed; the dataset was not pushed") from save_error
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
