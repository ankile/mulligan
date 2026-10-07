from __future__ import annotations

from concurrent.futures import Future
import threading
import time

from huggingface_hub.errors import RepositoryNotFoundError
import numpy as np
import pytest
import torch
from torch import nn


class TinyEncoder(nn.Module):
    def forward(self, x):
        return x.flatten(1)[:, :2]


class TinyQ(nn.Module):
    def forward(self, state, action):
        return state[:, :1]


class TinyV(nn.Module):
    def forward(self, state):
        return state[:, :1]


class FakeEnv:
    def __init__(self, obs):
        self.obs = obs

    def get_observation(self):
        return self.obs


class FakeResetEnv:
    def __init__(self):
        self.reset_joints = np.zeros(2)
        self.reset_calls = 0

    def reset(self, randomize=False):
        self.reset_calls += 1
        return {"robot_state": {"joint_positions": np.ones(2)}}


class FakeHub:
    def __init__(self, exc=None):
        self.exc = exc
        self.created = False

    def repo_info(self, **_kwargs):
        if self.exc is not None:
            raise self.exc
        return object()

    def create_repo(self, **_kwargs):
        self.created = True


class FakeHFDataset(dict):
    @property
    def column_names(self):
        return list(self.keys())


class FakeWritableDataset:
    def __init__(self, features):
        self.features = features
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        missing = (
            set(self.features)
            - {
                "episode_index",
                "frame_index",
                "index",
                "timestamp",
                "task_index",
            }
            - set(frame)
        )
        if missing:
            raise ValueError(f"missing frame features: {missing}")
        self.frames.append(frame)

    def save_episode(self, **_kwargs):
        self.saved += 1


def make_two_frame_real_episode_data() -> dict:
    from mulligan.real.collect.dataset_features import FRANKA_TELEMETRY_SPECS

    return {
        "observations": [np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32)],
        "actions": [np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32)],
        "steps_to_go": [1, 0],
        "rewards": [0.0, 1.0],
        "dones": [0, 1],
        "image_cam_left": [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ],
        "action_info_cartesian_velocity": [np.zeros(6, dtype=np.float32)] * 2,
        "action_info_cartesian_position": [np.zeros(6, dtype=np.float32)] * 2,
        "action_info_joint_velocity": [np.zeros(7, dtype=np.float32)] * 2,
        "action_info_joint_position": [np.zeros(7, dtype=np.float32)] * 2,
        "action_info_gripper_position": [np.zeros(1, dtype=np.float32)] * 2,
        "action_info_gripper_velocity": [np.zeros(1, dtype=np.float32)] * 2,
        "joint_positions": [np.zeros(7, dtype=np.float32)] * 2,
        "joint_velocities": [np.zeros(7, dtype=np.float32)] * 2,
        "cartesian_velocities": [np.zeros(6, dtype=np.float32)] * 2,
        **{
            spec.episode_key: [np.full(spec.shape, np.nan, dtype=np.float32)] * 2
            for spec in FRANKA_TELEMETRY_SPECS
        },
    }


def test_vision_iql_rejects_unprefixed_camera_keys_for_separate_encoders():
    from mulligan.networks.vision_iql import VisionIQL

    with pytest.raises(ValueError, match="observation.images"):
        VisionIQL(
            encoder=nn.ModuleDict({"cam": TinyEncoder()}),
            q1=TinyQ(),
            q2=TinyQ(),
            v_net=TinyV(),
            camera_keys=["cam"],
            separate_encoders=True,
            expectile=0.7,
            gamma=0.99,
            tau=0.005,
        )


def test_vision_iql_rejects_non_moduledict_separate_encoder():
    from mulligan.networks.vision_iql import VisionIQL

    with pytest.raises(ValueError, match="nn.ModuleDict"):
        VisionIQL(
            encoder=TinyEncoder(),
            q1=TinyQ(),
            q2=TinyQ(),
            v_net=TinyV(),
            camera_keys=["observation.images.cam"],
            separate_encoders=True,
            expectile=0.7,
            gamma=0.99,
            tau=0.005,
        )


def test_verified_reset_raises_after_exhausting_retries():
    from mulligan.real.collect.rollout import verified_reset

    env = FakeResetEnv()

    with pytest.raises(RuntimeError, match="Reset failed after 2 attempts"):
        verified_reset(env, max_retries=2, joint_threshold=0.15)

    assert env.reset_calls == 2


def test_run_with_timeout_propagates_non_timeout_errors():
    from mulligan.real.eval.common import run_with_timeout

    def fail():
        raise RuntimeError("hardware disconnected")

    with pytest.raises(RuntimeError, match="hardware disconnected"):
        run_with_timeout(fail, seconds=1, description="hardware op")


def test_run_with_timeout_raises_on_timeout():
    from mulligan.real.eval.common import run_with_timeout

    with pytest.raises(TimeoutError, match="slow upload timed out"):
        run_with_timeout(lambda: time.sleep(2), seconds=1, description="slow upload")


def test_run_with_timeout_rejects_worker_threads():
    from mulligan.real.eval.common import run_with_timeout

    errors: list[BaseException] = []

    def worker():
        try:
            run_with_timeout(lambda: None, seconds=1, description="worker timeout")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "main thread" in str(errors[0])


def test_wait_for_background_save_raises_on_worker_failure():
    from mulligan.real.collect.save_utils import wait_for_background_save

    future = Future()
    future.set_exception(ValueError("encoder failed"))

    with pytest.raises(RuntimeError, match="episode save failed"):
        wait_for_background_save(future, description="episode save")


def test_ensure_dataset_repo_only_creates_on_not_found():
    import httpx

    from mulligan.real.collect.hf_utils import ensure_dataset_repo

    response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))
    missing_hub = FakeHub(RepositoryNotFoundError("missing", response=response))
    assert ensure_dataset_repo(missing_hub, "user/ds", private=True) is True
    assert missing_hub.created is True

    auth_hub = FakeHub(RuntimeError("bad token"))
    with pytest.raises(RuntimeError, match="bad token"):
        ensure_dataset_repo(auth_hub, "user/ds", private=True)
    assert auth_hub.created is False


def test_replay_buffer_rejects_empty_and_underfilled_sampling():
    from mulligan.data.vision_replay_buffer import VisionReplayBuffer

    buffer = VisionReplayBuffer(
        capacity=2,
        camera_keys=["cam"],
        n_image_timestamps=2,
        img_h=4,
        img_w=4,
        action_chunk_size=1,
        action_dim=1,
        state_dim=1,
        n_state_timestamps=2,
    )

    with pytest.raises(ValueError, match="empty"):
        buffer.sample(1)

    buffer.refresh(
        {
            "cam": torch.zeros(1, 2, 3, 4, 4, dtype=torch.uint8),
            "action": torch.zeros(1, 1, 1),
            "reward": torch.zeros(1, 1),
            "done": torch.zeros(1, 1),
            "observation.state": torch.zeros(1, 2, 1),
            "success": torch.zeros(1, dtype=torch.long),
            "source": torch.zeros(1, dtype=torch.long),
            "episode_index": torch.zeros(1, dtype=torch.long),
        }
    )

    with pytest.raises(ValueError, match="batch_size=2"):
        buffer.sample(2)


def test_replay_buffer_samples_normalized_float_images_by_default():
    from mulligan.data.vision_replay_buffer import VisionReplayBuffer

    buffer = VisionReplayBuffer(
        capacity=2,
        camera_keys=["cam"],
        n_image_timestamps=2,
        img_h=1,
        img_w=1,
        action_chunk_size=1,
        action_dim=1,
        state_dim=1,
        n_state_timestamps=2,
    )
    buffer.refresh(
        {
            "cam": torch.full((1, 2, 3, 1, 1), 255, dtype=torch.uint8),
            "action": torch.zeros(1, 1, 1),
            "reward": torch.zeros(1, 1),
            "done": torch.zeros(1, 1),
            "observation.state": torch.zeros(1, 2, 1),
            "success": torch.zeros(1, dtype=torch.long),
            "source": torch.zeros(1, dtype=torch.long),
            "episode_index": torch.zeros(1, dtype=torch.long),
        }
    )

    float_batch = buffer.sample(1)
    assert float_batch["cam"].dtype == torch.float32
    assert float_batch["cam"].max().item() == 1.0

    uint8_batch = buffer.sample(1, image_format="uint8")
    assert uint8_batch["cam"].dtype == torch.uint8


def test_replay_buffer_budget_rejects_zero_capacity():
    from mulligan.data.vision_replay_buffer import VisionReplayBuffer

    with pytest.raises(ValueError, match="too small for one sample"):
        VisionReplayBuffer.from_budget_gb(
            budget_gb=0.0,
            camera_keys=["cam"],
            n_image_timestamps=2,
            img_h=224,
            img_w=224,
            action_chunk_size=1,
            action_dim=1,
            state_dim=1,
            n_state_timestamps=2,
        )


def test_replay_buffer_rejects_unexpected_image_dtype_and_wraps_writes():
    from mulligan.data.vision_replay_buffer import VisionReplayBuffer

    buffer = VisionReplayBuffer(
        capacity=3,
        camera_keys=["cam"],
        n_image_timestamps=2,
        img_h=1,
        img_w=1,
        action_chunk_size=1,
        action_dim=1,
        state_dim=1,
        n_state_timestamps=2,
    )

    def make_batch(values: list[float], *, dtype=torch.uint8):
        n = len(values)
        value_tensor = torch.tensor(values, dtype=torch.float32)
        if dtype == torch.uint8:
            cam = value_tensor.to(torch.uint8).reshape(n, 1, 1, 1, 1).expand(n, 2, 3, 1, 1)
        else:
            cam = torch.zeros(n, 2, 3, 1, 1, dtype=dtype)
        return {
            "cam": cam,
            "action": value_tensor.reshape(n, 1, 1),
            "reward": value_tensor.reshape(n, 1),
            "done": value_tensor.add(10).reshape(n, 1),
            "observation.state": value_tensor.reshape(n, 1, 1).expand(n, 2, 1),
            "success": value_tensor.add(20).long(),
            "source": value_tensor.add(30).long(),
            "episode_index": value_tensor.add(40).long(),
        }

    with pytest.raises(TypeError, match="uint8 or float32"):
        buffer.refresh(make_batch([0.0], dtype=torch.int16))

    with pytest.raises(ValueError, match="exceeds capacity"):
        buffer.refresh(make_batch([0.0, 1.0, 2.0, 3.0]))

    buffer.refresh(make_batch([0.0, 1.0]))
    buffer.refresh(make_batch([2.0, 3.0]))

    assert buffer.size == 3
    assert buffer._write_ptr == 1
    assert buffer.actions[:, 0, 0].tolist() == [3.0, 1.0, 2.0]
    assert buffer.images["cam"][:, 0, 0, 0, 0].tolist() == [3, 1, 2]
    assert buffer.dones[:, 0].tolist() == [13.0, 11.0, 12.0]
    assert buffer.states[:, 0, 0].tolist() == [3.0, 1.0, 2.0]
    assert buffer.success.tolist() == [23, 21, 22]
    assert buffer.source.tolist() == [33, 31, 32]
    assert buffer.episode_index.tolist() == [43, 41, 42]


def test_lerobot_policy_missing_image_key_fails_before_preprocessing():
    from mulligan.real.policy.loader import LeRobotRealWorldPolicy

    policy = object.__new__(LeRobotRealWorldPolicy)
    policy._policy = None
    policy._preprocessor = None
    policy._postprocessor = None
    policy._camera_keys = ["20000002_left"]
    policy._camera_height = 224
    policy._camera_width = 224
    policy._use_joint_positions = False

    raw_obs = {
        "robot_state": {
            "cartesian_position": np.zeros(6, dtype=np.float32),
            "gripper_position": 0.0,
        }
    }

    with pytest.raises(KeyError, match="raw_obs"):
        policy.predict(raw_obs)


def test_parse_camera_keys_handles_prefixes_bare_serials_and_duplicates():
    from mulligan.real.collect.rollout import camera_serials_from_keys
    from mulligan.real.collect.rollout import parse_camera_keys

    assert parse_camera_keys("observation.images.10000001_left, 20000002", "_left") == [
        "10000001_left",
        "20000002_left",
    ]
    assert camera_serials_from_keys(["10000001_left", "10000001_right", "20000002_left"]) == [
        "10000001",
        "20000002",
    ]

    with pytest.raises(ValueError, match="duplicates"):
        parse_camera_keys("10000001_left,observation.images.10000001_left", "_left")

    with pytest.raises(ValueError, match="bare serial"):
        parse_camera_keys("10000001", "")


def test_camera_discovery_fails_on_missing_image_payload():
    from mulligan.real.robot.cameras import image_camera_keys_from_obs
    from mulligan.real.robot.cameras import select_image_camera_keys

    with pytest.raises(RuntimeError, match="missing image data"):
        image_camera_keys_from_obs({"robot_state": {}}, context="test setup")

    with pytest.raises(RuntimeError, match="must be a mapping"):
        image_camera_keys_from_obs({"image": None}, context="test setup")

    selected, available = select_image_camera_keys(
        {"image": {"11111111_left": np.zeros((2, 2, 3)), "22222222_right": np.zeros((2, 2, 3))}},
        "_left",
    )
    assert selected == ["11111111_left"]
    assert available == ["11111111_left", "22222222_right"]

    with pytest.raises(RuntimeError, match="No non-excluded cameras match"):
        select_image_camera_keys({"image": {"20000002_right": np.zeros((2, 2, 3))}}, "_left")


def test_episode_val_split_rejects_empty_or_degenerate_training_splits():
    from mulligan.real.train.policy import make_episode_val_split

    with pytest.raises(ValueError, match="at least 2 episodes"):
        make_episode_val_split(1, 0.2)

    with pytest.raises(ValueError, match="--val-pct"):
        make_episode_val_split(4, 0.0)

    with pytest.raises(ValueError, match="--val-pct"):
        make_episode_val_split(4, 1.0)

    train, val = make_episode_val_split(5, 0.2)
    assert train
    assert val
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(5))


def test_episode_holdout_split_rejects_empty_or_degenerate_training_splits():
    from mulligan.real.train.critic import make_episode_holdout_split

    with pytest.raises(ValueError, match="at least 2 episodes"):
        make_episode_holdout_split(1, 0.2)

    with pytest.raises(ValueError, match="--holdout-pct"):
        make_episode_holdout_split(4, 0.0)

    with pytest.raises(ValueError, match="--holdout-pct"):
        make_episode_holdout_split(4, 1.0)

    train, holdout = make_episode_holdout_split(5, 0.2)
    assert train
    assert holdout
    assert set(train).isdisjoint(holdout)
    assert sorted(train + holdout) == list(range(5))


def test_parse_repo_id_list_rejects_empty_entries():
    from mulligan.real.train.critic import parse_repo_id_list

    assert parse_repo_id_list("a/b, c/d") == ["a/b", "c/d"]
    assert parse_repo_id_list(None) == []

    with pytest.raises(ValueError, match="empty entries"):
        parse_repo_id_list("a/b,,c/d")


def test_real_iql_horizon_decouples_prediction_and_critic_execution():
    from mulligan.real.train.critic import resolve_iql_horizons

    prediction_horizon, critic_horizon = resolve_iql_horizons(
        prediction_horizon=8,
        n_action_steps=6,
    )

    assert prediction_horizon == 8
    assert critic_horizon == 6


def test_real_iql_horizon_rejects_execution_longer_than_prediction():
    from mulligan.real.train.critic import resolve_iql_horizons

    with pytest.raises(ValueError, match="must be <= --chunk-size"):
        resolve_iql_horizons(prediction_horizon=6, n_action_steps=8)


def test_real_dataset_schema_always_includes_intervention():
    from mulligan.real.collect.dataset_features import build_real_lerobot_features

    episode_data = {
        "observations": [np.zeros(7, dtype=np.float32)],
        "actions": [np.zeros(7, dtype=np.float32)],
        "image_cam_left": [np.zeros((4, 4, 3), dtype=np.uint8)],
    }

    features = build_real_lerobot_features(episode_data, ["image_cam_left"])

    assert features["intervention"] == {
        "dtype": "int64",
        "shape": (1,),
        "names": ["intervention_flag"],
    }


def test_real_episode_save_writes_zero_intervention_without_dagger_sources():
    from mulligan.data.constants import DataSource
    from mulligan.real.collect.dataset_features import (
        save_episode_to_dataset as save_real_episode_to_dataset,
    )
    from mulligan.real.collect.dataset_features import build_real_lerobot_features

    episode_data = make_two_frame_real_episode_data()
    features = build_real_lerobot_features(episode_data, ["image_cam_left"])
    dataset = FakeWritableDataset(features)

    save_real_episode_to_dataset(
        dataset=dataset,
        episode_data=episode_data,
        episode_success=True,
        camera_keys=["image_cam_left"],
        task_name="marker_d2",
        saved_episode_count=0,
        default_source=DataSource.AUTONOMOUS,
    )

    assert dataset.saved == 1
    assert [int(frame["intervention"][0]) for frame in dataset.frames] == [0, 0]
    assert [int(frame["source"][0]) for frame in dataset.frames] == [
        int(DataSource.AUTONOMOUS),
        int(DataSource.AUTONOMOUS),
    ]


def test_real_episode_save_rejects_source_length_mismatch():
    from mulligan.data.constants import DataSource
    from mulligan.real.collect.dataset_features import (
        save_episode_to_dataset as save_real_episode_to_dataset,
    )
    from mulligan.real.collect.dataset_features import build_real_lerobot_features

    episode_data = make_two_frame_real_episode_data()
    features = build_real_lerobot_features(episode_data, ["image_cam_left"])
    dataset = FakeWritableDataset(features)

    with pytest.raises(ValueError, match="sources length must match episode length"):
        save_real_episode_to_dataset(
            dataset=dataset,
            episode_data=episode_data,
            episode_success=True,
            camera_keys=["image_cam_left"],
            task_name="marker_d2",
            saved_episode_count=0,
            sources=[DataSource.AUTONOMOUS],
        )

    assert dataset.frames == []
    assert dataset.saved == 0


def test_real_iql_intervention_values_allow_missing_column_for_pure_teleop():
    from mulligan.data.constants import DataSource
    from mulligan.real.train.critic import intervention_values_for_subdataset

    source_values = [int(DataSource.HUMAN), int(DataSource.HUMAN)]

    interventions = intervention_values_for_subdataset(
        repo_id="user/r0-teleop",
        column_names=["source", "success"],
        source_values=source_values,
        hf_dataset=FakeHFDataset({"source": source_values}),
        require_intervention=True,
    )

    assert interventions == [0, 0]


def test_real_iql_intervention_values_reject_missing_column_for_autonomous_data():
    from mulligan.data.constants import DataSource
    from mulligan.real.train.critic import intervention_values_for_subdataset

    source_values = [int(DataSource.HUMAN), int(DataSource.AUTONOMOUS)]

    with pytest.raises(ValueError, match="requires an 'intervention' column"):
        intervention_values_for_subdataset(
            repo_id="user/r1-policy",
            column_names=["source", "success"],
            source_values=source_values,
            hf_dataset=FakeHFDataset({"source": source_values}),
            require_intervention=True,
        )


def test_real_iql_intervention_values_allow_missing_column_for_eval_rollouts():
    from mulligan.data.constants import DataSource
    from mulligan.real.train.critic import intervention_values_for_subdataset

    source_values = [int(DataSource.AUTONOMOUS), int(DataSource.AUTONOMOUS)]

    with pytest.warns(UserWarning, match="zero-intervention"):
        interventions = intervention_values_for_subdataset(
            repo_id="user/insert-marker-d1-mulligan-sobol-r0-25k-eval-sobol20",
            column_names=["source", "success", "policy_id", "round_id"],
            source_values=source_values,
            hf_dataset=FakeHFDataset({"source": source_values}),
            require_intervention=True,
        )

    assert interventions == [0, 0]


def test_real_iql_applies_intervention_penalty_and_reward_shift_by_frame_index():
    from mulligan.real.train.critic import apply_intervention_reward_shaping

    batch = {
        "index": torch.tensor([0, 2]),
        "reward": torch.zeros(2, 3),
    }
    intervention_by_frame = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)

    apply_intervention_reward_shaping(
        batch,
        intervention_by_frame=intervention_by_frame,
        intervention_negative_reward=-1.0,
        reward_shift=0.25,
        horizon=3,
    )

    expected = torch.tensor(
        [
            [0.25, -0.75, 0.25],
            [0.25, -0.75, -0.75],
        ]
    )
    torch.testing.assert_close(batch["reward"], expected)


def test_real_iql_intervention_penalty_uses_dataset_local_indices():
    from mulligan.real.train.critic import apply_intervention_reward_shaping

    batch = {
        "dataset_index": torch.tensor([0, 1]),
        "index": torch.tensor([1, 1]),
        "reward": torch.zeros(2, 2),
    }
    intervention_by_dataset = [
        torch.tensor([0, 1, 0], dtype=torch.long),
        torch.tensor([0, 0, 1], dtype=torch.long),
    ]

    apply_intervention_reward_shaping(
        batch,
        intervention_by_frame=torch.empty(0, dtype=torch.long),
        intervention_by_dataset=intervention_by_dataset,
        intervention_negative_reward=-1.0,
        reward_shift=0.0,
        horizon=2,
    )

    expected = torch.tensor(
        [
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    torch.testing.assert_close(batch["reward"], expected)


def test_append_split_index_requires_frame_to_episode_mapping():
    from mulligan.real.train.policy import append_split_index

    train_indices: list[int] = []
    val_indices: list[int] = []

    append_split_index(
        10,
        frame_to_episode={10: 3},
        val_episode_set={3},
        train_indices=train_indices,
        val_indices=val_indices,
    )

    assert train_indices == []
    assert val_indices == [10]

    with pytest.raises(KeyError):
        append_split_index(
            11,
            frame_to_episode={10: 3},
            val_episode_set={3},
            train_indices=train_indices,
            val_indices=val_indices,
        )


def test_flatten_loss_dict_accepts_scalar_tensors_and_rejects_vectors():
    from mulligan.real.train.policy import flatten_loss_dict

    flat = flatten_loss_dict(
        {
            "loss": torch.tensor(1.25),
            "aux": 2,
            "per_dim": [0.5, 0.75],
        }
    )

    assert flat == {"loss": 1.25, "aux": 2.0, "per_dim_0": 0.5, "per_dim_1": 0.75}

    with pytest.raises(TypeError, match="Unexpected type"):
        flatten_loss_dict({"bad": torch.tensor([1.0, 2.0])})


def test_compute_val_loss_rejects_zero_batches():
    from mulligan.real.train.policy import compute_val_loss

    policy = nn.Linear(1, 1)

    with pytest.raises(ValueError, match="num_batches must be positive"):
        compute_val_loss(
            val_dataloader=[],
            val_dl_iter=iter([]),
            policy=policy,
            preprocessor=lambda batch: batch,
            policy_cfg=object(),
            device=torch.device("cpu"),
            num_batches=0,
        )


class _FakeSubDataset:
    def __init__(self, episode_indices, reported_len=None):
        self.hf_dataset = {"episode_index": episode_indices}
        self.features = {}
        self._reported_len = reported_len

    def __len__(self):
        if self._reported_len is None:
            return len(self.hf_dataset["episode_index"])
        return self._reported_len


def test_compute_multidataset_valid_boundaries_uses_actual_rows():
    from mulligan.data.transforms import compute_multidataset_valid_boundaries

    from_indices, to_indices, _raw, _n = compute_multidataset_valid_boundaries(
        [
            _FakeSubDataset([0, 0, 2]),
            _FakeSubDataset(torch.tensor([5, 5, 6, 6])),
        ]
    )

    assert from_indices == [0, 2, 3, 5]
    assert to_indices == [2, 3, 5, 7]


def test_compute_multidataset_valid_boundaries_caps_stale_extra_rows():
    from mulligan.data.transforms import compute_multidataset_valid_boundaries

    from_indices, to_indices, _raw, _n = compute_multidataset_valid_boundaries(
        [_FakeSubDataset([0, 0, 1, 1, 2, 2], reported_len=4)]
    )

    assert from_indices == [0, 2]
    assert to_indices == [2, 4]


def test_require_multilerobot_subdatasets_fails_loudly_on_api_change():
    from mulligan.data.transforms import require_multilerobot_subdatasets

    class MissingPrivateSubdatasets:
        pass

    with pytest.raises(AttributeError, match="LeRobot API changed"):
        require_multilerobot_subdatasets(MissingPrivateSubdatasets())
