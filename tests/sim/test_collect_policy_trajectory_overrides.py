from types import SimpleNamespace

import json
import random

import numpy as np
import pytest
import torch

from mulligan.sim.collect import rollouts
from mulligan.sim.collect.rollouts import (
    STATE_KEYS_NARROW,
    _apply_idql_collection_overrides,
    _broad_state_to_placements,
    _load_initial_states,
    _sha256_file,
    _write_rollout_audit,
    _write_rollout_lineage,
    collect_trajectories,
)
from mulligan.training.train import _require_policy_stratum, _seed_training_rngs


def test_idql_num_action_samples_override_is_applied() -> None:
    policy = SimpleNamespace(config=SimpleNamespace(num_action_samples=128))

    _apply_idql_collection_overrides(
        policy,
        policy_type="idql",
        num_action_samples=1,
    )

    assert policy.config.num_action_samples == 1


def test_idql_num_action_samples_none_preserves_checkpoint_value() -> None:
    policy = SimpleNamespace(config=SimpleNamespace(num_action_samples=32))

    _apply_idql_collection_overrides(
        policy,
        policy_type="idql",
        num_action_samples=None,
    )

    assert policy.config.num_action_samples == 32


def test_num_action_samples_override_rejects_non_idql_policy() -> None:
    policy = SimpleNamespace(config=SimpleNamespace())

    with pytest.raises(ValueError, match="only valid for IDQL"):
        _apply_idql_collection_overrides(
            policy,
            policy_type="lerobot",
            num_action_samples=1,
        )


def test_num_action_samples_override_rejects_unlocked_value() -> None:
    policy = SimpleNamespace(config=SimpleNamespace(num_action_samples=32))

    with pytest.raises(ValueError, match="must be 1 or 32"):
        _apply_idql_collection_overrides(
            policy,
            policy_type="idql",
            num_action_samples=2,
        )


def test_strict_policy_stratum_fails_loudly() -> None:
    with pytest.raises(ValueError, match="straddled_auto_success.*512.*128"):
        _require_policy_stratum(
            mode="straddled_auto_success",
            available_frames=128,
            required_frames=512,
            strict=True,
        )


def test_non_strict_policy_stratum_allows_human_only_fallback() -> None:
    _require_policy_stratum(
        mode="straddled_all",
        available_frames=0,
        required_frames=512,
        strict=False,
    )


def test_training_rng_seed_controls_python_numpy_and_torch() -> None:
    _seed_training_rngs(4)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    _seed_training_rngs(4)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


HF_REF = "hf://mulligan/sim-square-narrow-r01-mulligan-divl@05e15eb4f0324587337fc7a8e3374398c6044443/seed-1"


def test_rollout_lineage_is_durable_and_hash_locked(tmp_path) -> None:
    packet = tmp_path / "packet.json"
    packet.write_text('{"states": []}\n')
    audit = tmp_path / "audit.json"
    audit.write_text('{"records": [], "_meta": {}}\n')
    dataset_dir = tmp_path / "dataset"
    args = SimpleNamespace(
        initial_states_json_file=str(packet),
        dataset_name="locked-policy-rollouts",
        checkpoint=HF_REF,
        num_action_samples=32,
        seed=2026080311,
        env_seed=None,
        num_episodes=100,
    )

    lineage_path = _write_rollout_lineage(
        dataset_dir=dataset_dir,
        audit_path=audit,
        args=args,
        policy_type="idql",
    )

    lineage = json.loads(lineage_path.read_text())
    assert lineage["schema"] == "mulligan.sim.policy_rollout_lineage.v1"
    assert lineage["initial_states_sha256"] == _sha256_file(packet)
    assert lineage["predecessor_artifact"] == HF_REF
    assert lineage["predecessor_checkpoint"] is None
    assert "env_seed" not in lineage
    assert lineage["num_action_samples"] == 32
    assert lineage["collection_seed"] == 2026080311
    assert (dataset_dir / "meta/rollout_audit.json").read_text() == audit.read_text()


def test_rollout_audit_records_starts_and_local_checkpoint(tmp_path) -> None:
    states = [
        {"nut_x": -0.11, "nut_y": 0.12, "nut_yaw": 0.5, "cell_idx": 3},
        {"nut_x": -0.112, "nut_y": 0.2, "nut_yaw": -1.0, "cell_idx": 7},
    ]
    packet = tmp_path / "starts.json"
    packet.write_text(json.dumps({"states": states}))
    episodes = [
        {
            "requested_episode_index": 1,
            "success": True,
            "length": 5,
            "rewards": [0.0, 1.0],
            "actions": [],
        },
        {
            "requested_episode_index": 0,
            "success": False,
            "length": 9,
            "rewards": [0.0],
            "actions": [],
        },
    ]
    args = SimpleNamespace(
        dataset_name="rollouts",
        checkpoint=str(tmp_path / "ckpt"),
        max_steps=400,
        num_envs=2,
        num_action_samples=32,
        seed=1,
        env_seed=5,
        initial_states_json_file=str(packet),
    )
    audit_path = tmp_path / "audit.json"

    _write_rollout_audit(
        path=audit_path,
        episodes=episodes,
        states=_load_initial_states(packet, STATE_KEYS_NARROW),
        args=args,
        env_name="NutAssemblySquare",
        robot_name="Panda",
        policy_type="idql",
    )

    audit = json.loads(audit_path.read_text())
    assert [(r["cell_idx"], r["success"], r["reward"]) for r in audit["records"]] == [
        (7, 1, 1.0),
        (3, 0, 0.0),
    ]
    assert audit["_meta"]["checkpoint_artifact"] is None
    assert audit["_meta"]["checkpoint_path"] == str(tmp_path / "ckpt")
    assert audit["_meta"]["env_seed"] == 5
    assert audit["_meta"]["initial_states_sha256"] == _sha256_file(packet)


class _ScriptedVecEnv:
    """Env ``i`` succeeds after ``success_after[i]`` steps (``None``: never)."""

    def __init__(self, success_after):
        self.success_after = success_after
        self.placements = []
        self.steps = [0] * len(success_after)

    def __len__(self):
        return len(self.success_after)

    def _obs(self, value):
        return {
            "robot0_eef_pos": np.full(3, value),
            "robot0_eef_quat": np.full(4, value),
            "robot0_gripper_qpos": np.full(2, value),
            "object-state": np.full(14, value),
        }

    def reset_with_placements(self, placements):
        self.placements.append(placements)
        self.steps = [0] * len(self)
        n = len(self)
        return [self._obs(0.0) for _ in range(n)], [np.zeros(16)] * n, [np.zeros(15)] * n

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for idx, limit in enumerate(self.success_after):
            self.steps[idx] += 1
            success = limit is not None and self.steps[idx] >= limit
            obs.append(self._obs(float(self.steps[idx])))
            rewards.append(1.0 if success else 0.0)
            dones.append(False)
            infos.append({"success": success})
        return obs, rewards, dones, infos


class _ConstantPolicy:
    def __init__(self):
        self.batches = []

    def eval(self):
        pass

    def train(self):
        pass

    def reset(self):
        pass

    def select_action(self, batch):
        self.batches.append(batch)
        return torch.zeros(batch["observation.state"].shape[0], 7)


def test_collect_trajectories_pads_final_frame_and_keeps_start_order() -> None:
    vec_env = _ScriptedVecEnv([2, None])
    policy = _ConstantPolicy()
    starts = [
        _broad_state_to_placements(
            {"nut_x": 0.0, "nut_y": 0.0, "nut_yaw": 0.0, "peg_x": 0.1, "peg_y": 0.0}
        )
    ] * 3

    episodes = collect_trajectories(
        policy=policy,
        vec_env=vec_env,
        num_episodes=3,
        max_steps=4,
        device="cpu",
        state_mean=torch.zeros(23),
        state_std=torch.ones(23),
        action_min=torch.full((7,), -1.0),
        action_max=torch.full((7,), 3.0),
        initial_placements=starts,
    )

    assert [ep["requested_episode_index"] for ep in episodes] == [0, 1, 2]
    success, timeout = episodes[0], episodes[1]
    assert success["success"] and success["length"] == 2
    assert len(success["observations"]) == len(success["actions"]) == 3
    assert success["dones"] == [False, True, True]
    assert not timeout["success"] and timeout["length"] == 4
    assert timeout["dones"][-2:] == [False, False]
    # Zero actions in [-1, 1] map to the middle of [action_min, action_max].
    np.testing.assert_allclose(success["actions"][0], np.ones(7))
    assert policy.batches[0]["observation.state"].shape == (2, 9)
    assert policy.batches[0]["observation.environment_state"].shape == (2, 14)
    # The short second round replays its first start on the idle worker.
    assert len(vec_env.placements) == 2 and len(vec_env.placements[1]) == 2


def test_main_refuses_an_existing_dataset_before_collecting(tmp_path) -> None:
    (tmp_path / "data" / "rollouts").mkdir(parents=True)
    argv = ["--checkpoint", str(tmp_path / "missing-checkpoint"), "--dataset-name", "rollouts"]
    with pytest.raises(SystemExit):
        rollouts.main([*argv, "--dataset-path", str(tmp_path / "data")])


def test_main_rejects_non_idql_checkpoints(tmp_path) -> None:
    checkpoint = tmp_path / "seed-1"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps({"policy_type": "diffusion"}))

    with pytest.raises(ValueError, match="supports IDQL checkpoints"):
        rollouts.main(["--checkpoint", str(checkpoint), "--dataset-name", "x"])


def test_rollouts_default_step_budget_is_the_papers() -> None:
    assert rollouts.build_parser().get_default("max_steps") == 400
