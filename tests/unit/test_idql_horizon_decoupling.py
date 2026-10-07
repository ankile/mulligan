from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mulligan.agents.idql import IDQLPolicy
from mulligan.configs.policy import IDQLPolicyConfig
from mulligan.training.batch_processor import create_batch_processor
from mulligan.training.fast_dataset_loader import prepare_chunked_data
from mulligan.training.train import _get_training_horizons


def _toy_transition_data(num_steps: int = 10) -> dict[str, torch.Tensor]:
    actions = torch.arange(num_steps * 2, dtype=torch.float32).reshape(num_steps, 2)
    states = torch.arange(num_steps * 3, dtype=torch.float32).reshape(num_steps, 3)
    next_states = states + 100.0
    rewards = torch.ones(num_steps, dtype=torch.float32)
    dones = torch.zeros(num_steps, dtype=torch.float32)
    dones[-1] = 1.0
    return {
        "actions": actions,
        "states": states,
        "next_states": next_states,
        "rewards": rewards,
        "dones": dones,
        "episode_indices": torch.zeros(num_steps, dtype=torch.long),
        "dataset_indices": torch.zeros(num_steps, dtype=torch.long),
    }


def test_prepare_chunked_data_uses_requested_horizon_for_targets():
    data = _toy_transition_data()
    gamma = 0.9

    critic = prepare_chunked_data(data, chunk_size=5, gamma=gamma)
    actor = prepare_chunked_data(data, chunk_size=8, gamma=gamma)

    assert critic["actions"].shape == (10, 5, 2)
    assert actor["actions"].shape == (10, 8, 2)

    # The critic target should advance to s_{t+5}, not the actor prediction horizon.
    assert torch.equal(critic["next_states"][0], data["states"][5])
    assert torch.equal(actor["next_states"][0], data["states"][8])

    expected_5_step_reward = sum(gamma**i for i in range(5))
    expected_8_step_reward = sum(gamma**i for i in range(8))
    assert critic["rewards"][0].item() == pytest.approx(expected_5_step_reward)
    assert actor["rewards"][0].item() == pytest.approx(expected_8_step_reward)

    # At t=5 there are exactly five actions left, so the 5-step critic chunk is
    # valid while the 8-step actor chunk is padded.
    assert critic["chunk_valid"][5].item() == 1.0
    assert actor["chunk_valid"][5].item() == 0.0


class _ShapeCheckingActor(nn.Module):
    def __init__(self, *, prediction_horizon: int, execution_horizon: int):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.prediction_horizon = prediction_horizon
        self.config = SimpleNamespace(n_action_steps=execution_horizon, type="diffusion")
        self.diffusion = SimpleNamespace(generate_actions=self._generate_actions)
        self.seen_action_shape: tuple[int, ...] | None = None

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        self.seen_action_shape = tuple(batch["action"].shape)
        assert batch["action"].shape[1] == self.prediction_horizon
        return self.weight * batch["action"].square().mean(), None

    def reset(self) -> None:
        return None

    def _generate_actions(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = obs["observation.state"].shape[0]
        return torch.zeros(batch_size, self.config.n_action_steps, 2)


def test_idql_critic_uses_execution_horizon_actor_uses_prediction_horizon():
    torch.manual_seed(0)
    actor = _ShapeCheckingActor(prediction_horizon=8, execution_horizon=5)
    config = IDQLPolicyConfig(
        chunk_size=8,
        n_action_steps=5,
        hidden_dims=[16],
        num_q_networks=1,
        gamma=0.9,
    )
    policy = IDQLPolicy(state_dim=4, action_dim=2, actor=actor, config=config)

    assert policy.chunk_size == 5
    assert policy.critics[0].action_dim == 10

    batch_size = 3
    critic_loss = policy.compute_loss_critic(
        {
            "observation.state": torch.randn(batch_size, 4),
            "action": torch.randn(batch_size, 5 * 2),
            "reward": torch.randn(batch_size, 1),
            "next.observation.state": torch.randn(batch_size, 4),
            "masks": torch.ones(batch_size, 1),
            "chunk_valid": torch.ones(batch_size, 1),
        }
    )
    assert torch.isfinite(critic_loss["loss_critic"])
    assert critic_loss["chunk_size"] == 5
    assert critic_loss["effective_gamma"] == pytest.approx(0.9**5)

    actor_loss = policy.compute_loss_actor(
        {
            "observation.state": torch.randn(batch_size, 1, 4),
            "action": torch.randn(batch_size, 8, 2),
            "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
        }
    )
    assert torch.isfinite(actor_loss["loss_actor"])
    assert actor.seen_action_shape == (batch_size, 8, 2)


def test_idql_update_routes_separate_critic_and_actor_horizons():
    torch.manual_seed(0)
    actor = _ShapeCheckingActor(prediction_horizon=8, execution_horizon=5)
    config = IDQLPolicyConfig(
        chunk_size=8,
        n_action_steps=5,
        hidden_dims=[16],
        num_q_networks=1,
        gamma=0.9,
    )
    normalizer = SimpleNamespace(
        state_mean=torch.zeros(4),
        state_std=torch.ones(4),
        state_min=-torch.ones(4),
        state_range=2 * torch.ones(4),
    )
    policy = IDQLPolicy(
        state_dim=4,
        action_dim=2,
        actor=actor,
        config=config,
        normalizer=normalizer,
    )
    optimizers = {
        "actor": torch.optim.Adam(policy.actor.parameters(), lr=1e-3),
        "critic": torch.optim.Adam(policy.critics.parameters(), lr=1e-3),
        "value": torch.optim.Adam(policy.value.parameters(), lr=1e-3),
    }

    batch_size = 3
    critic_batch = {
        "observation.state": torch.randn(batch_size, 4),
        "action": torch.randn(batch_size, 5, 2),
        "reward": torch.randn(batch_size, 1),
        "next.observation.state": torch.randn(batch_size, 4),
        "masks": torch.ones(batch_size, 1),
        "chunk_valid": torch.ones(batch_size, 1),
    }
    actor_batch = {
        "observation.state": torch.randn(batch_size, 1, 4),
        "action": torch.randn(batch_size, 8, 2),
        "action_is_pad": torch.zeros(batch_size, 8, dtype=torch.bool),
    }

    metrics = policy.update(critic_batch, optimizers, policy_batch=actor_batch)

    assert torch.isfinite(metrics["losses/critic"])
    assert torch.isfinite(metrics["losses/actor"])
    assert metrics["debug/chunk_size"] == 5
    assert actor.seen_action_shape == (batch_size, 8, 2)


def test_batch_processor_preserves_different_critic_and_policy_action_lengths():
    processor = create_batch_processor(
        preprocessor=lambda batch: batch,
        state_dim=3,
        policy_type="idql",
        device="cpu",
    )
    batch_size = 4
    state = torch.randn(batch_size, 5)
    next_state = torch.randn(batch_size, 5)

    critic_batch = processor.prepare_critic_batch(
        state=state,
        next_state=next_state,
        action=torch.randn(batch_size, 5, 2),
        action_is_pad=torch.zeros(batch_size, 5, dtype=torch.bool),
        reward=torch.randn(batch_size, 1),
        masks=torch.ones(batch_size, 1),
        chunk_valid=torch.ones(batch_size, 1),
    )
    policy_batch = processor.prepare_policy_batch(
        state=state,
        next_state=next_state,
        action=torch.randn(batch_size, 8, 2),
        action_is_pad=torch.zeros(batch_size, 8, dtype=torch.bool),
        reward=torch.randn(batch_size, 1),
        masks=torch.ones(batch_size, 1),
        chunk_valid=torch.ones(batch_size, 1),
    )

    assert critic_batch["action"].shape == (batch_size, 5, 2)
    assert critic_batch["action_is_pad"].shape == (batch_size, 5)
    assert policy_batch["action"].shape == (batch_size, 8, 2)
    assert policy_batch["action_is_pad"].shape == (batch_size, 8)


def test_get_training_horizons_decouples_execution_for_idql():
    cfg = SimpleNamespace(chunk_size=8, n_action_steps=6)
    assert _get_training_horizons("idql", cfg) == (8, 6)
    assert _get_training_horizons(
        "idql_divl", SimpleNamespace(chunk_size=8, n_action_steps=None)
    ) == (8, 8)


def test_get_training_horizons_rejects_unknown_policy_and_missing_chunk_size():
    with pytest.raises(ValueError, match="unsupported policy_type"):
        _get_training_horizons("diffusion", SimpleNamespace(chunk_size=8, n_action_steps=5))
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        _get_training_horizons("idql", SimpleNamespace(chunk_size=None, n_action_steps=5))


def test_idql_config_rejects_execution_horizon_past_current_observation_window():
    with pytest.raises(ValueError, match="available action slots"):
        IDQLPolicyConfig(chunk_size=8, n_action_steps=9)
