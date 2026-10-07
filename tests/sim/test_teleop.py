"""R0 teleop episode and dataset writing with a fake SpaceMouse and keyboard (headless, CPU)."""

import json

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.sim.collect import teleop
from mulligan.sim.envs import create_robosuite_env


class FakeDevice:
    pos_sensitivity = 1.0
    rot_sensitivity = 1.0

    def __init__(self):
        self.control = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.control_gripper = 0

    def start_control(self):
        pass

    def reset_gripper(self):
        self.control_gripper = 0


class FakeKeyboard:
    def __init__(self, idle_steps, key):
        self.idle_steps = idle_steps
        self.key = key

    def read_key(self):
        if self.idle_steps == 0:
            return self.key
        self.idle_steps -= 1
        return None


def test_teleop_episode_places_start_and_saves(tmp_path):
    states_file = tmp_path / "states.json"
    state = {"nut_x": -0.1125, "nut_y": 0.15, "nut_yaw": 0.4}
    states_file.write_text(json.dumps({"states": [state]}))
    args = teleop.parse_args(
        [
            "--sampler",
            "list",
            "--initial-states-file",
            str(states_file),
            "--headless",
        ]
    )
    sampler = teleop._build_sampler(args)
    env = create_robosuite_env("NutAssemblySquare", use_render_wrapper=False)
    try:
        data, success = teleop.teleop_episode(
            env,
            FakeDevice(),
            FakeKeyboard(idle_steps=5, key="1"),
            max_fr=0,
            save_data=True,
            has_renderer=False,
            sampler=sampler,
            sampler_label="LIST",
        )
    finally:
        env.close()

    assert success is True
    # 5 recorded steps plus the padded final frame.
    assert len(data["actions"]) == len(data["observations"]) == 6
    np.testing.assert_allclose(data["actions"][0], [0.34375, 0, 0, 0, 0, 0, -1.0])
    assert teleop._record_sampler_start(sampler, data["environment_state"][0])
    np.testing.assert_allclose(
        sampler.collected_points[0][:2], [state["nut_x"], state["nut_y"]], atol=0.02
    )
    assert teleop._sampler_remaining(sampler) == 0

    data["steps_to_go"] = [len(data["actions"]) - 1 - t for t in range(len(data["actions"]))]
    root = tmp_path / "teleop-test"
    dataset = LeRobotDataset.create(
        repo_id="teleop-test",
        fps=20,
        root=str(root),
        robot_type="panda",
        features=teleop._dataset_features("NutAssemblySquare", data, None),
    )
    teleop.save_episode_to_dataset(dataset, data, True, None, "NutAssemblySquare_Panda", 1)
    dataset.finalize()

    frames = pd.concat(pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet")))
    assert len(frames) == 6
    assert [int(np.asarray(v).reshape(-1)[0]) for v in frames["steps_to_go"]] == [5, 4, 3, 2, 1, 0]
    assert {int(np.asarray(v).reshape(-1)[0]) for v in frames["source"]} == {1}
    assert [int(np.asarray(v).reshape(-1)[0]) for v in frames["is_valid"]] == [1] * 5 + [0]


def test_r0_presets():
    # Flags of the paper's R0 launchers (Narrow: 200 episodes, sensitivity 1.2/1.2;
    # Broad: 400 episodes, default sensitivity), both cameras, list order.
    narrow = teleop.parse_args(["--r0-preset", "square_narrow"])
    assert (narrow.env, narrow.target_episodes) == ("NutAssemblySquare", 200)
    assert (narrow.pos_sensitivity, narrow.rot_sensitivity) == (1.2, 1.2)
    broad = teleop.parse_args(["--r0-preset", "square_broad"])
    assert (broad.env, broad.target_episodes) == ("Square_D1", 400)
    assert (broad.pos_sensitivity, broad.rot_sensitivity) == (1.0, 1.5)
    for args, task, n_states in ((narrow, "square_narrow", 200), (broad, "square_broad", 400)):
        assert args.cameras == "agentview,robot0_eye_in_hand"
        assert args.save_data and args.auto_save_on_success
        assert args.sampler == "list" and not args.sampler_shuffle
        assert args.dataset_name == f"sim-{task.replace('_', '-')}-c00-teleop-mixed"
        states = json.loads(open(args.initial_states_file).read())["states"]
        assert len(states) == n_states
    # Explicit flags override the preset; no preset keeps the plain defaults.
    args = teleop.parse_args(["--r0-preset", "square_broad", "--target-episodes", "5"])
    assert args.target_episodes == 5 and args.env == "Square_D1"
    plain = teleop.parse_args([])
    assert plain.target_episodes is None and plain.sampler is None and plain.sampler_shuffle
