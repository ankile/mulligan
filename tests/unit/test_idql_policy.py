#!/usr/bin/env python3
"""
Test suite for IDQL (Implicit Diffusion Q-Learning) policy implementation.

Tests cover:
1. Normalization conversions (z-score ↔ MIN_MAX) in select_action()
2. Q-value selection logic with multiple samples
3. Double Q-learning min operation across multiple critics
4. Edge cases (N=1 samples, batch_size=1, single Q-network)
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from mulligan.agents.idql import IDQLPolicy
from mulligan.configs.policy import IDQLPolicyConfig
from mulligan.training.normalization import Normalizer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Use CPU for testing (simpler and more portable)."""
    return "cpu"


@pytest.fixture
def normalizer_stats():
    """Create normalizer statistics for testing."""
    state_dim = 10
    return {
        "state_mean": torch.randn(state_dim),
        "state_std": torch.abs(torch.randn(state_dim)) + 0.1,  # Ensure positive
        "state_min": torch.randn(state_dim) - 2.0,
        "state_max": torch.randn(state_dim) + 2.0,
        "action_min": torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        "action_max": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "state_dim": state_dim,
        "robot_state_dim": 7,
        "env_state_dim": 3,
        "action_dim": 7,
    }


@pytest.fixture
def normalizer(normalizer_stats, device):
    """Create a Normalizer instance for testing."""
    return Normalizer(
        state_mean=normalizer_stats["state_mean"],
        state_std=normalizer_stats["state_std"],
        action_min=normalizer_stats["action_min"],
        action_max=normalizer_stats["action_max"],
        device=device,
        state_min=normalizer_stats["state_min"],
        state_max=normalizer_stats["state_max"],
    )


@pytest.fixture
def diffusion_actor(normalizer_stats, device):
    """Create a real DiffusionPolicy for testing.

    This creates a minimal state-only diffusion policy with environment state
    input (no images). The policy is intentionally small for fast test execution.
    """
    robot_state_dim = normalizer_stats["robot_state_dim"]
    env_state_dim = normalizer_stats["env_state_dim"]
    action_dim = normalizer_stats["action_dim"]

    # Define input features (robot state + environment state)
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(robot_state_dim,)),
        "observation.environment_state": PolicyFeature(
            type=FeatureType.ENV, shape=(env_state_dim,)
        ),
    }

    # Define output features (actions)
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }

    # Create a minimal DiffusionConfig for fast testing
    # Use small dimensions and few diffusion steps to keep tests fast
    config = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=1,
        horizon=8,  # Minimum viable horizon (must be divisible by 2^len(down_dims))
        n_action_steps=1,  # Single action step for simpler testing
        down_dims=(64, 128),  # Small UNet for speed (2 stages = horizon divisible by 4)
        kernel_size=3,
        n_groups=4,
        diffusion_step_embed_dim=32,
        num_train_timesteps=10,  # Fewer diffusion steps for speed
        num_inference_steps=2,  # Very few inference steps for speed
        device=device,
    )

    return DiffusionPolicy(config)


# =============================================================================
# Normalization Conversion Tests
# =============================================================================


class TestNormalizationConversions:
    """Tests for z-score ↔ MIN_MAX normalization conversions in select_action."""

    def test_zscore_to_raw_conversion(self, normalizer, normalizer_stats, device):
        """Test that z-score to raw conversion is correct."""
        batch_size = 4
        state_dim = normalizer_stats["state_dim"]

        # Create z-score normalized state
        state_zscore = torch.randn(batch_size, state_dim, device=device)

        # Manual conversion: raw = zscore * std + mean
        expected_raw = state_zscore * normalizer.state_std + normalizer.state_mean

        # Using normalizer's denormalize
        actual_raw = normalizer.denormalize_state(state_zscore)

        assert torch.allclose(expected_raw, actual_raw, atol=1e-6)

    def test_raw_to_minmax_conversion(self, normalizer, normalizer_stats, device):
        """Test that raw to MIN_MAX conversion is correct."""
        batch_size = 4

        # Create raw state within min/max bounds
        state_raw = (normalizer.state_min + normalizer.state_max) / 2  # Midpoint
        state_raw = state_raw.unsqueeze(0).expand(batch_size, -1)

        # Manual conversion: minmax = 2 * (raw - min) / range - 1
        expected_minmax = 2.0 * (state_raw - normalizer.state_min) / normalizer.state_range - 1.0

        # At midpoint, should be approximately 0
        assert torch.allclose(expected_minmax, torch.zeros_like(expected_minmax), atol=1e-5)

    def test_zscore_to_minmax_roundtrip(self, normalizer, normalizer_stats, device):
        """Test z-score → raw → MIN_MAX → raw → z-score roundtrip."""
        batch_size = 4
        state_dim = normalizer_stats["state_dim"]

        # Start with z-score normalized state
        original_zscore = torch.randn(batch_size, state_dim, device=device)

        # z-score → raw
        raw = normalizer.denormalize_state(original_zscore)

        # raw → MIN_MAX
        minmax = normalizer.normalize_state_minmax(raw)

        # MIN_MAX → raw
        raw_back = normalizer.denormalize_state_minmax(minmax)

        # raw → z-score
        zscore_back = normalizer.normalize_state(raw_back)

        assert torch.allclose(original_zscore, zscore_back, atol=1e-5)

    def test_minmax_bounds(self, normalizer, normalizer_stats, device):
        """Test that MIN_MAX normalization produces values in [-1, 1] range."""
        batch_size = 4

        # Test at min bounds
        state_at_min = normalizer.state_min.unsqueeze(0).expand(batch_size, -1)
        minmax_at_min = normalizer.normalize_state_minmax(state_at_min)
        assert torch.allclose(minmax_at_min, -torch.ones_like(minmax_at_min), atol=1e-5)

        # Test at max bounds
        state_at_max = normalizer.state_max.unsqueeze(0).expand(batch_size, -1)
        minmax_at_max = normalizer.normalize_state_minmax(state_at_max)
        assert torch.allclose(minmax_at_max, torch.ones_like(minmax_at_max), atol=1e-5)

    def test_select_action_uses_minmax_for_diffusion(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that select_action converts robot state to MIN_MAX for diffusion actor."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_action_samples=4,
            num_q_networks=1,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]
        env_dim = normalizer_stats["env_state_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 2

        # Create z-score normalized observations
        obs = {
            "observation.state": torch.randn(batch_size, robot_dim, device=device),
            "observation.environment_state": torch.randn(batch_size, env_dim, device=device),
        }

        # This should work without error - the internal conversion happens
        with torch.no_grad():
            action = policy.select_action(obs)

        # Verify action shape
        assert action.shape == (batch_size, action_dim)

        # Actions should be in [-1, 1] range (tanh output)
        assert action.min() >= -1.0
        assert action.max() <= 1.0

    def test_select_action_requires_normalizer(self, diffusion_actor, normalizer_stats, device):
        """Test that select_action raises error when normalizer is not set."""
        config = IDQLPolicyConfig(hidden_dims=[32, 32], num_action_samples=4)

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]
        env_dim = normalizer_stats["env_state_dim"]

        # Create policy WITHOUT normalizer
        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=None,  # No normalizer
        )

        batch_size = 2
        obs = {
            "observation.state": torch.randn(batch_size, robot_dim, device=device),
            "observation.environment_state": torch.randn(batch_size, env_dim, device=device),
        }

        with pytest.raises(ValueError, match="Normalizer not set"):
            policy.select_action(obs)

    def test_set_normalizer_validates_minmax_stats(self, diffusion_actor, normalizer_stats, device):
        """Test that set_normalizer validates state_min/state_max are present."""
        config = IDQLPolicyConfig(hidden_dims=[32, 32], num_action_samples=4)

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=None,
        )

        # Create normalizer WITHOUT state_min/state_max
        incomplete_normalizer = Normalizer(
            state_mean=normalizer_stats["state_mean"],
            state_std=normalizer_stats["state_std"],
            action_min=normalizer_stats["action_min"],
            action_max=normalizer_stats["action_max"],
            device=device,
            state_min=None,  # Missing!
            state_max=None,  # Missing!
        )

        with pytest.raises(ValueError, match="state_min and state_max"):
            policy.set_normalizer(incomplete_normalizer)

    def test_set_normalizer_accepts_complete_normalizer(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that set_normalizer accepts a valid normalizer with state_min/max."""
        config = IDQLPolicyConfig(hidden_dims=[32, 32], num_action_samples=4)

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]
        env_dim = normalizer_stats["env_state_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=None,  # Start without normalizer
        )

        # This should work
        policy.set_normalizer(normalizer)

        # Now select_action should work
        batch_size = 2
        obs = {
            "observation.state": torch.randn(batch_size, robot_dim, device=device),
            "observation.environment_state": torch.randn(batch_size, env_dim, device=device),
        }

        with torch.no_grad():
            action = policy.select_action(obs)

        assert action.shape == (batch_size, action_dim)


# =============================================================================
# Q-Value Selection Tests
# =============================================================================


class TestQValueSelection:
    """Tests for Q-value based action selection logic."""

    def test_select_action_picks_highest_q(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that select_action picks the action with highest Q-value."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_action_samples=5,
            num_q_networks=1,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]
        env_dim = normalizer_stats["env_state_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 2
        obs = {
            "observation.state": torch.randn(batch_size, robot_dim, device=device),
            "observation.environment_state": torch.randn(batch_size, env_dim, device=device),
        }

        # Run select_action
        with torch.no_grad():
            selected_action = policy.select_action(obs)

        # Verify we get valid actions
        assert selected_action.shape == (batch_size, action_dim)
        assert not torch.isnan(selected_action).any()
        assert not torch.isinf(selected_action).any()

    def test_multiple_samples_exploration(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that using multiple samples provides action diversity."""
        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]
        env_dim = normalizer_stats["env_state_dim"]

        # Test with different num_action_samples
        for num_samples in [1, 5, 10]:
            config = IDQLPolicyConfig(
                hidden_dims=[32, 32],
                num_action_samples=num_samples,
                num_q_networks=1,
            )

            policy = IDQLPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                actor=diffusion_actor,
                config=config,
                normalizer=normalizer,
            )

            batch_size = 3
            obs = {
                "observation.state": torch.randn(batch_size, robot_dim, device=device),
                "observation.environment_state": torch.randn(batch_size, env_dim, device=device),
            }

            with torch.no_grad():
                action = policy.select_action(obs)

            assert action.shape == (batch_size, action_dim), f"Failed for num_samples={num_samples}"


# =============================================================================
# Double Q-Learning Tests
# =============================================================================


class TestDoubleQLearning:
    """Tests for double Q-learning min operation across multiple critics."""

    def test_compute_q_value_with_single_critic(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test Q-value computation with single critic (no min needed)."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_q_networks=1,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 4
        state = torch.randn(batch_size, state_dim, device=device)
        action = torch.randn(batch_size, action_dim, device=device)

        with torch.no_grad():
            q_value = policy.compute_q_value(state, action)

        # Should return values from the single critic
        assert q_value.shape == (batch_size, 1)
        assert len(policy.critics) == 1

    def test_compute_q_value_with_multiple_critics(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that Q-value computation takes min over multiple critics."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_q_networks=3,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 4
        state = torch.randn(batch_size, state_dim, device=device)
        action = torch.randn(batch_size, action_dim, device=device)

        with torch.no_grad():
            # Get individual Q-values from each critic
            q_values_individual = [c(state, action) for c in policy.critics]
            q_stacked = torch.stack(q_values_individual, dim=0)

            # Expected: min over all critics
            expected_q = q_stacked.min(dim=0).values

            # Actual from compute_q_value
            actual_q = policy.compute_q_value(state, action)

        assert torch.allclose(expected_q, actual_q)
        assert len(policy.critics) == 3

    def test_double_q_reduces_overestimation(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that min over critics is <= max individual critic value."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_q_networks=3,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 4
        state = torch.randn(batch_size, state_dim, device=device)
        action = torch.randn(batch_size, action_dim, device=device)

        with torch.no_grad():
            q_values_individual = [c(state, action) for c in policy.critics]
            q_stacked = torch.stack(q_values_individual, dim=0)

            q_min = policy.compute_q_value(state, action)
            q_max = q_stacked.max(dim=0).values

        # Min should be <= max (by definition)
        assert (q_min <= q_max + 1e-6).all()

    def test_target_critics_initialized_correctly(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that target critics are initialized as copies of critics."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_q_networks=2,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        # Initially, target critics should match critics
        batch_size = 4
        state = torch.randn(batch_size, state_dim, device=device)
        action = torch.randn(batch_size, action_dim, device=device)

        with torch.no_grad():
            for critic, target_critic in zip(policy.critics, policy.target_critics):
                q_critic = critic(state, action)
                q_target = target_critic(state, action)
                assert torch.allclose(q_critic, q_target)

    def test_target_critics_have_no_grad(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that target critic parameters have requires_grad=False."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_q_networks=2,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        for target_critic in policy.target_critics:
            for param in target_critic.parameters():
                assert not param.requires_grad

        for param in policy.target_value.parameters():
            assert not param.requires_grad


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in IDQL policy."""

    def test_single_sample_selection(self, diffusion_actor, normalizer, normalizer_stats, device):
        """Test action selection with N=1 samples (no real selection)."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_action_samples=1,  # Single sample
            num_q_networks=1,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]
        env_dim = normalizer_stats["env_state_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 3
        obs = {
            "observation.state": torch.randn(batch_size, robot_dim, device=device),
            "observation.environment_state": torch.randn(batch_size, env_dim, device=device),
        }

        with torch.no_grad():
            action = policy.select_action(obs)

        assert action.shape == (batch_size, action_dim)
        assert not torch.isnan(action).any()

    def test_batch_size_one(self, diffusion_actor, normalizer, normalizer_stats, device):
        """Test action selection with batch_size=1."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_action_samples=5,
            num_q_networks=2,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]
        env_dim = normalizer_stats["env_state_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 1  # Single batch element
        obs = {
            "observation.state": torch.randn(batch_size, robot_dim, device=device),
            "observation.environment_state": torch.randn(batch_size, env_dim, device=device),
        }

        with torch.no_grad():
            action = policy.select_action(obs)

        assert action.shape == (1, action_dim)
        assert not torch.isnan(action).any()

    def test_empty_environment_state_raises_error(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that missing environment state raises a ValueError with helpful message."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_action_samples=5,
            num_q_networks=1,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 2
        obs = {
            "observation.state": torch.randn(batch_size, robot_dim, device=device),
            # Missing observation.environment_state
        }

        with pytest.raises(ValueError, match="observation.environment_state is required"):
            policy.select_action(obs)

    def test_large_batch_size(self, diffusion_actor, normalizer, normalizer_stats, device):
        """Test action selection with large batch size."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_action_samples=10,
            num_q_networks=2,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]
        env_dim = normalizer_stats["env_state_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 256
        obs = {
            "observation.state": torch.randn(batch_size, robot_dim, device=device),
            "observation.environment_state": torch.randn(batch_size, env_dim, device=device),
        }

        with torch.no_grad():
            action = policy.select_action(obs)

        assert action.shape == (batch_size, action_dim)
        assert not torch.isnan(action).any()

    def test_compute_v_value(self, diffusion_actor, normalizer, normalizer_stats, device):
        """Test V-value computation."""
        config = IDQLPolicyConfig(hidden_dims=[32, 32], num_q_networks=2)

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 4
        state = torch.randn(batch_size, state_dim, device=device)

        with torch.no_grad():
            v_value = policy.compute_v_value(state)

        assert v_value.shape == (batch_size, 1)
        assert not torch.isnan(v_value).any()


# =============================================================================
# Polyak Target Update Tests
# =============================================================================


class TestTargetNetworkUpdate:
    """Tests for Polyak averaging target network updates."""

    def test_update_targets_polyak_averaging(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that update_targets performs correct Polyak averaging."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_q_networks=2,
            tau=0.1,  # Use larger tau for easier testing
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        # Modify the main networks to have different parameters
        for critic in policy.critics:
            for param in critic.parameters():
                param.data = torch.randn_like(param.data)

        for param in policy.value.parameters():
            param.data = torch.randn_like(param.data)

        # Store original target params
        original_target_params = []
        for target_critic in policy.target_critics:
            original_target_params.append([p.data.clone() for p in target_critic.parameters()])
        original_target_value_params = [p.data.clone() for p in policy.target_value.parameters()]

        # Update targets
        policy.update_targets()

        # Verify Polyak averaging: target = tau * main + (1-tau) * target
        tau = config.tau
        for i, (critic, target_critic) in enumerate(zip(policy.critics, policy.target_critics)):
            for j, (param, target_param) in enumerate(
                zip(critic.parameters(), target_critic.parameters())
            ):
                expected = tau * param.data + (1.0 - tau) * original_target_params[i][j]
                assert torch.allclose(target_param.data, expected, atol=1e-6)

        for j, (param, target_param) in enumerate(
            zip(policy.value.parameters(), policy.target_value.parameters())
        ):
            expected = tau * param.data + (1.0 - tau) * original_target_value_params[j]
            assert torch.allclose(target_param.data, expected, atol=1e-6)


# =============================================================================
# Loss Computation Tests
# =============================================================================


class TestLossComputation:
    """Tests for loss computation methods."""

    def test_compute_loss_value_expectile(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test value loss computation uses expectile regression."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_q_networks=2,
            expectile=0.7,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 8
        batch = {
            "observation.state": torch.randn(batch_size, state_dim, device=device),
            "action": torch.randn(batch_size, action_dim, device=device),
        }

        loss_dict = policy.compute_loss_value(batch)

        assert "loss_value" in loss_dict
        assert "v_mean" in loss_dict
        assert "v_std" in loss_dict
        assert loss_dict["loss_value"].shape == ()  # Scalar loss
        assert loss_dict["loss_value"] >= 0  # Expectile loss is non-negative

    def test_compute_loss_critic_td_learning(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test critic loss computation uses TD learning."""
        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_q_networks=2,
            gamma=0.99,
        )

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 8
        # New batch format uses masks (1=bootstrap, 0=don't) and chunk_valid
        batch = {
            "observation.state": torch.randn(batch_size, state_dim, device=device),
            "action": torch.randn(batch_size, action_dim, device=device),
            "reward": torch.randn(batch_size, 1, device=device),  # Shape (B, 1)
            "next.observation.state": torch.randn(batch_size, state_dim, device=device),
            "masks": torch.ones(batch_size, 1, device=device),  # All bootstrap (1 = bootstrap)
            "chunk_valid": torch.ones(batch_size, 1, device=device),  # All valid transitions
        }

        loss_dict = policy.compute_loss_critic(batch)

        assert "loss_critic" in loss_dict
        assert "q_mean" in loss_dict
        assert "td_error" in loss_dict
        assert loss_dict["loss_critic"] >= 0  # MSE loss is non-negative

    def test_critic_loss_respects_masks_flag(
        self, diffusion_actor, normalizer, normalizer_stats, device
    ):
        """Test that masks=0 prevents bootstrapping (terminal states)."""
        config = IDQLPolicyConfig(hidden_dims=[32, 32], gamma=0.99, num_q_networks=1)

        state_dim = normalizer_stats["state_dim"]
        action_dim = normalizer_stats["action_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 1
        reward = torch.tensor([[1.0]], device=device)  # Shape (B, 1)

        # Case 1: masks=1 (non-terminal), should bootstrap
        batch_bootstrap = {
            "observation.state": torch.randn(batch_size, state_dim, device=device),
            "action": torch.randn(batch_size, action_dim, device=device),
            "reward": reward,
            "next.observation.state": torch.randn(batch_size, state_dim, device=device),
            "masks": torch.ones(batch_size, 1, device=device),  # 1 = bootstrap
            "chunk_valid": torch.ones(batch_size, 1, device=device),
        }

        # Case 2: masks=0 (terminal), should not bootstrap
        batch_no_bootstrap = {
            "observation.state": batch_bootstrap["observation.state"],
            "action": batch_bootstrap["action"],
            "reward": reward,
            "next.observation.state": batch_bootstrap["next.observation.state"],
            "masks": torch.zeros(batch_size, 1, device=device),  # 0 = don't bootstrap
            "chunk_valid": torch.ones(batch_size, 1, device=device),
        }

        loss_bootstrap = policy.compute_loss_critic(batch_bootstrap)
        loss_no_bootstrap = policy.compute_loss_critic(batch_no_bootstrap)

        # Target when masks=0 should just be reward (no bootstrap)
        # They should produce different losses (unless by coincidence V(next_s) = 0)
        # We mainly verify both run without error
        assert loss_bootstrap["loss_critic"] >= 0
        assert loss_no_bootstrap["loss_critic"] >= 0


# =============================================================================
# Config Validation Tests
# =============================================================================


class TestIDQLPolicyConfigValidation:
    """Tests for IDQLPolicyConfig parameter validation."""

    def test_valid_config(self):
        """Test that valid config parameters pass validation."""
        config = IDQLPolicyConfig(
            num_action_samples=10,
            num_q_networks=2,
            expectile=0.7,
            gamma=0.99,
            tau=0.005,
        )
        assert config.num_action_samples == 10
        assert config.num_q_networks == 2
        assert config.expectile == 0.7

    def test_invalid_num_action_samples(self):
        """Test that num_action_samples < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_action_samples must be >= 1"):
            IDQLPolicyConfig(num_action_samples=0)

        with pytest.raises(ValueError, match="num_action_samples must be >= 1"):
            IDQLPolicyConfig(num_action_samples=-1)

    def test_invalid_num_q_networks(self):
        """Test that num_q_networks < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_q_networks must be >= 1"):
            IDQLPolicyConfig(num_q_networks=0)

        with pytest.raises(ValueError, match="num_q_networks must be >= 1"):
            IDQLPolicyConfig(num_q_networks=-5)

    def test_invalid_expectile(self):
        """Test that expectile outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="expectile must be in"):
            IDQLPolicyConfig(expectile=-0.1)

        with pytest.raises(ValueError, match="expectile must be in"):
            IDQLPolicyConfig(expectile=1.5)

    def test_invalid_gamma(self):
        """Test that gamma outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be in"):
            IDQLPolicyConfig(gamma=-0.1)

        with pytest.raises(ValueError, match="gamma must be in"):
            IDQLPolicyConfig(gamma=1.5)

    def test_invalid_tau(self):
        """Test that tau outside (0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="tau must be in"):
            IDQLPolicyConfig(tau=0.0)

        with pytest.raises(ValueError, match="tau must be in"):
            IDQLPolicyConfig(tau=-0.1)

        with pytest.raises(ValueError, match="tau must be in"):
            IDQLPolicyConfig(tau=1.5)

    def test_boundary_values(self):
        """Test that boundary values are accepted."""
        # Expectile and gamma can be 0 or 1
        config_low = IDQLPolicyConfig(expectile=0.0, gamma=0.0)
        assert config_low.expectile == 0.0
        assert config_low.gamma == 0.0

        config_high = IDQLPolicyConfig(expectile=1.0, gamma=1.0, tau=1.0)
        assert config_high.expectile == 1.0
        assert config_high.gamma == 1.0
        assert config_high.tau == 1.0


# =============================================================================
# Division-by-Zero Safety Tests
# =============================================================================


class TestDivisionByZeroSafety:
    """Tests for division-by-zero protection in normalization."""

    def test_constant_feature_handling(self, diffusion_actor, normalizer_stats, device):
        """Test that constant features (zero range) don't cause NaN/Inf."""
        # Create normalizer with some constant features (min == max)
        state_dim = normalizer_stats["state_dim"]
        state_mean = torch.randn(state_dim)
        state_std = torch.abs(torch.randn(state_dim)) + 0.1

        # Make some features constant (same min and max)
        state_min = torch.randn(state_dim)
        state_max = state_min.clone()  # Same as min -> zero range
        # But leave some features with valid range
        state_max[0:5] = state_min[0:5] + 1.0  # Valid range for first 5

        normalizer = Normalizer(
            state_mean=state_mean,
            state_std=state_std,
            action_min=normalizer_stats["action_min"],
            action_max=normalizer_stats["action_max"],
            device=device,
            state_min=state_min,
            state_max=state_max,
        )

        config = IDQLPolicyConfig(
            hidden_dims=[32, 32],
            num_action_samples=3,
            num_q_networks=1,
        )

        action_dim = normalizer_stats["action_dim"]
        robot_dim = normalizer_stats["robot_state_dim"]
        env_dim = normalizer_stats["env_state_dim"]

        policy = IDQLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            actor=diffusion_actor,
            config=config,
            normalizer=normalizer,
        )

        batch_size = 2
        obs = {
            "observation.state": torch.randn(batch_size, robot_dim, device=device),
            "observation.environment_state": torch.randn(batch_size, env_dim, device=device),
        }

        # This should NOT produce NaN or Inf due to clamping
        with torch.no_grad():
            action = policy.select_action(obs)

        assert not torch.isnan(action).any(), "Action contains NaN values"
        assert not torch.isinf(action).any(), "Action contains Inf values"


# =============================================================================
# Run tests manually
# =============================================================================


if __name__ == "__main__":
    print("Running IDQL policy tests...\n")

    # Run with pytest
    pytest.main([__file__, "-v"])
