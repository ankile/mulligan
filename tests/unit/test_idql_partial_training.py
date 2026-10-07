"""Frozen-actor (critic/value-only) training and parent-checkpoint loading for IDQL/DIVL."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mulligan.agents.factory import NormalizationStats, _load_parent_normalizer
from mulligan.agents.idql import IDQLPolicy
from mulligan.configs.dataset import DatasetConfig
from mulligan.configs.env import EnvConfig
from mulligan.configs.policy import IDQLDIVLConfig, IDQLPolicyConfig
from mulligan.configs.train import TrainConfig
from mulligan.configs.training import TrainingConfig
from mulligan.release.hub import get_checkpoint_files
from mulligan.training.checkpoint_utils import save_normalization_stats
from mulligan.training.normalization import Normalizer

REPO_A = "mulligan/sim-square-narrow-c00-teleop-sobol"
REPO_B = "mulligan/sim-square-narrow-c01-dagger-mulligan"


class _FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.config = SimpleNamespace(
            type="fake_diffusion", horizon=3, n_action_steps=2, n_obs_steps=1
        )

    def forward(self, batch):
        return ((self.weight - batch["action"]) ** 2).mean(), None

    def reset(self) -> None:
        pass


def _normalizer(state_dim: int = 2) -> Normalizer:
    return Normalizer(
        state_mean=torch.zeros(state_dim),
        state_std=torch.ones(state_dim),
        action_min=-torch.ones(1),
        action_max=torch.ones(1),
        device="cpu",
        state_min=-torch.ones(state_dim),
        state_max=torch.ones(state_dim),
    )


def _policy() -> IDQLPolicy:
    config = IDQLPolicyConfig(chunk_size=3, n_action_steps=2, hidden_dims=[8], num_q_networks=2)
    return IDQLPolicy(
        state_dim=2, action_dim=1, actor=_FakeActor(), config=config, normalizer=_normalizer()
    )


def _current_stats(action_min: float = -1.0) -> NormalizationStats:
    return NormalizationStats(
        state_mean=torch.zeros(2),
        state_std=torch.ones(2),
        robot_state_mean=torch.zeros(1),
        robot_state_std=torch.ones(1),
        env_state_mean=torch.zeros(1),
        env_state_std=torch.ones(1),
        state_min=-torch.ones(2),
        state_max=torch.ones(2),
        action_min=torch.tensor([action_min]),
        action_max=torch.tensor([1.0]),
        state_dim=1,
        env_state_dim=1,
        action_dim=1,
    )


def test_parent_normalizer_replaces_recomputed_state_stats(tmp_path: Path) -> None:
    parent = Normalizer(
        state_mean=torch.tensor([1.0, 2.0]),
        state_std=torch.tensor([3.0, 4.0]),
        state_min=torch.tensor([-5.0, -6.0]),
        state_max=torch.tensor([7.0, 8.0]),
        action_min=torch.tensor([-1.0]),
        action_max=torch.tensor([1.0]),
    )
    save_normalization_stats(tmp_path, normalizer=parent)
    loaded = _load_parent_normalizer(tmp_path, _current_stats(), device="cpu")
    assert torch.equal(loaded.state_mean, parent.state_mean)
    assert torch.equal(loaded.state_std, parent.state_std)
    assert torch.equal(loaded.state_min, parent.state_min)
    assert torch.equal(loaded.state_max, parent.state_max)


def test_parent_normalizer_rejects_action_contract_change(tmp_path: Path) -> None:
    parent = Normalizer(
        state_mean=torch.zeros(2),
        state_std=torch.ones(2),
        state_min=-torch.ones(2),
        state_max=torch.ones(2),
        action_min=torch.tensor([-0.5]),
        action_max=torch.tensor([1.0]),
    )
    save_normalization_stats(tmp_path, normalizer=parent)
    with pytest.raises(ValueError, match="action_min"):
        _load_parent_normalizer(tmp_path, _current_stats(), device="cpu")


def test_critic_value_only_update_keeps_actor_frozen() -> None:
    policy = _policy()
    policy.set_critic_value_only_training()
    actor_hash = policy.actor_hash()
    policy.train()
    assert not policy.actor.training
    assert not policy.target_critics.training and not policy.target_value.training

    params = policy.get_optim_params()
    optimizers = {
        "critic": torch.optim.AdamW(params["critic"], lr=1e-2),
        "value": torch.optim.AdamW(params["value"], lr=1e-2),
    }
    critic_before = [p.detach().clone() for p in policy.critics.parameters()]
    batch = {
        "observation.state": torch.randn(4, 1, 1),
        "observation.environment_state": torch.randn(4, 1, 1),
        "action": torch.rand(4, 2, 1) * 2 - 1,
        "reward": torch.randn(4, 1),
        "masks": torch.ones(4, 1),
        "chunk_valid": torch.ones(4, 1),
        "next.observation.state": torch.randn(4, 1, 1),
        "next.observation.environment_state": torch.randn(4, 1, 1),
    }
    info = policy.update(batch, optimizers, update_components="critic_value_only")
    assert "losses/actor" not in info
    assert policy.actor_hash() == actor_hash
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, policy.critics.parameters()))
    with pytest.raises(ValueError, match="critic/value optimizers"):
        policy.update(
            batch,
            {**optimizers, "actor": torch.optim.SGD(policy.actor.parameters(), lr=0.1)},
            update_components="critic_value_only",
        )


def test_complete_checkpoint_roundtrip(tmp_path: Path) -> None:
    parent = _policy()
    parent.save(tmp_path)
    child = _policy()
    child.load_complete_checkpoint(tmp_path, strict_contract=True)
    for a, b in zip(parent.state_dict().values(), child.state_dict().values()):
        assert torch.equal(a, b)


def test_critic_value_only_config_requires_divl_and_parent_artifact() -> None:
    dataset = DatasetConfig(repo_ids=REPO_A)
    env = EnvConfig(name="NutAssemblySquare", robot="Panda")
    training = TrainingConfig(update_components="critic_value_only")
    with pytest.raises(ValueError, match="requires policy.type=idql_divl"):
        TrainConfig(dataset=dataset, env=env, policy=IDQLPolicyConfig(), training=training)
    with pytest.raises(ValueError, match="requires pretrained_artifact"):
        TrainConfig(dataset=dataset, env=env, policy=IDQLDIVLConfig(), training=training)
    config = TrainConfig(
        dataset=dataset,
        env=env,
        policy=IDQLDIVLConfig(),
        training=training,
        pretrained_artifact=f"hf://{REPO_A}@abc/seed-1",
    )
    assert config.training.update_components == "critic_value_only"


def test_removed_update_modes_are_rejected() -> None:
    with pytest.raises(ValueError, match="update_components"):
        TrainingConfig(update_components="actor_only")


def test_dataset_revisions_must_cover_exact_repo_set() -> None:
    with pytest.raises(ValueError, match="keys must exactly match"):
        DatasetConfig(repo_ids=f"{REPO_A},{REPO_B}", revisions={REPO_A: "abc123"})
    config = DatasetConfig(
        repo_ids=f"{REPO_A},{REPO_B}", revisions={REPO_A: "abc123", REPO_B: "def456"}
    )
    assert config.revisions == {REPO_A: "abc123", REPO_B: "def456"}


def test_critic_value_summary_is_uploaded_with_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "policy.pt").write_bytes(b"checkpoint")
    (tmp_path / "critic_value_summary.json").write_text("{}\n")
    assert {path.name for path in get_checkpoint_files(tmp_path)} == {
        "policy.pt",
        "critic_value_summary.json",
    }
