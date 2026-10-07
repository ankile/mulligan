"""Exact target, truncation, and schedule contracts for cached TD(lambda) IQL."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from mulligan.networks.q_network import QNetwork
from mulligan.networks.distributional_v import DistributionalVNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.networks.vision_iql import (
    VisionIQL,
    build_truncated_td_lambda_target,
    truncated_td_lambda_weights,
)
from mulligan.real.train.critic import td_lambda_at_step, validate_td_lambda_config


def test_truncated_weights_match_spec_and_sum_to_one() -> None:
    valid = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    weights = truncated_td_lambda_weights(valid, 0.5)

    torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.25, 0.125, 0.125]))
    torch.testing.assert_close(weights[1], torch.tensor([0.5, 0.5, 0.0, 0.0]))
    torch.testing.assert_close(weights[2], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))


def test_initial_h96_mixture_matches_locked_effective_horizon() -> None:
    valid = torch.ones(1, 16, dtype=torch.bool)
    weights = truncated_td_lambda_weights(valid, 0.95)
    horizons = torch.arange(6, 97, 6)

    assert weights[0, -1].item() == pytest.approx(0.95**15)
    assert (weights * horizons).sum().item() == pytest.approx(67.1848, abs=1e-4)


def test_target_mixes_all_valid_horizons_and_truncates_per_row() -> None:
    gamma = 0.5
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    dones = torch.zeros_like(rewards, dtype=torch.long)
    valid = torch.tensor([[True, True], [True, False]])
    bootstrap = torch.tensor([[8.0, 40.0], [8.0, 999.0]])

    mixed, targets, weights = build_truncated_td_lambda_target(
        rewards,
        dones,
        torch.tensor([1.0, 0.5, 0.25, 0.125]),
        bootstrap,
        torch.tensor([2, 4]),
        valid,
        gamma=gamma,
        td_lambda=0.5,
    )

    # G2 = 1 + .5*2 + .5^2*8 = 4; G4 = 3.25 + .5^4*40 = 5.75.
    torch.testing.assert_close(targets[0], torch.tensor([4.0, 5.75]))
    torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(mixed[0], torch.tensor([4.875]))
    # The invalid long bootstrap is ignored and all mass moves to H2.
    torch.testing.assert_close(weights[1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(mixed[1], torch.tensor([4.0]))


def test_target_stops_rewards_and_bootstrap_at_done() -> None:
    mixed, targets, _ = build_truncated_td_lambda_target(
        torch.tensor([[1.0, 2.0, 100.0, 100.0]]),
        torch.tensor([[0, 1, 1, 1]]),
        torch.tensor([1.0, 0.5, 0.25, 0.125]),
        torch.tensor([[999.0, 999.0]]),
        torch.tensor([2, 4]),
        torch.tensor([[True, True]]),
        gamma=0.5,
        td_lambda=0.95,
    )

    torch.testing.assert_close(targets, torch.tensor([[2.0, 2.0]]))
    torch.testing.assert_close(mixed, torch.tensor([[2.0]]))


def test_lambda_zero_is_exact_shortest_horizon_target() -> None:
    mixed, targets, weights = build_truncated_td_lambda_target(
        torch.tensor([[0.2, 0.3, 0.4, 0.5]]),
        torch.zeros(1, 4, dtype=torch.long),
        torch.tensor([1.0, 0.9, 0.81, 0.729]),
        torch.tensor([[0.7, 0.8]]),
        torch.tensor([2, 4]),
        torch.tensor([[True, True]]),
        gamma=0.9,
        td_lambda=0.0,
    )

    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(mixed[:, 0], targets[:, 0])


def test_td_lambda_schedule_holds_cosine_decays_and_reaches_zero() -> None:
    kwargs = {
        "initial_lambda": 0.95,
        "hold_steps": 25_000,
        "cosine_end_step": 112_500,
    }
    assert td_lambda_at_step(0, **kwargs) == 0.95
    assert td_lambda_at_step(25_000, **kwargs) == 0.95
    assert td_lambda_at_step(68_750, **kwargs) == pytest.approx(0.475)
    assert td_lambda_at_step(112_500, **kwargs) == 0.0
    assert td_lambda_at_step(150_000, **kwargs) == 0.0
    assert td_lambda_at_step(50_000, **kwargs) == pytest.approx(
        0.95 * 0.5 * (1 + math.cos(math.pi * 25_000 / 87_500))
    )


def test_td_lambda_config_requires_flat_cache_and_long_horizon() -> None:
    valid = {
        "enabled": True,
        "use_embedding_cache": True,
        "action_horizon": 6,
        "max_horizon": 96,
        "initial_lambda": 0.95,
        "hold_steps": 25_000,
        "cosine_end_step": 112_500,
        "training_steps": 150_000,
    }
    validate_td_lambda_config(**valid)
    with pytest.raises(ValueError, match="precompute-embeddings"):
        validate_td_lambda_config(**{**valid, "use_embedding_cache": False})
    with pytest.raises(ValueError, match="greater than"):
        validate_td_lambda_config(**{**valid, "max_horizon": 6})


def test_weights_reject_nonprefix_validity() -> None:
    with pytest.raises(ValueError, match="contiguous prefix"):
        truncated_td_lambda_weights(torch.tensor([[True, False, True]]), 0.5)


def test_td_lambda_encoded_loss_trains_both_q_heads() -> None:
    torch.manual_seed(3)
    model = VisionIQL(
        encoder=nn.Identity(),
        q1=QNetwork(3, 2, hidden_dims=[8], use_layer_norm=False),
        q2=QNetwork(3, 2, hidden_dims=[8], use_layer_norm=False),
        v_net=VNetwork(3, hidden_dims=[8]),
        camera_keys=["cam"],
        separate_encoders=False,
        expectile=0.7,
        gamma=0.9,
        tau=0.005,
    )
    curr_state = torch.randn(2, 3)
    next_state = torch.randn(2, 3)
    bootstrap_states = torch.randn(2, 2, 3)
    losses = model.forward_encoded_states(
        curr_state,
        next_state,
        torch.randn(2, 2),
        torch.rand(2, 4),
        torch.zeros(2, 4, dtype=torch.long),
        torch.tensor([1.0, 0.9, 0.81, 0.729]),
        td_lambda_bootstrap_states=bootstrap_states,
        td_lambda_horizons=torch.tensor([2, 4]),
        td_lambda_horizon_valid=torch.tensor([[True, True], [True, False]]),
        td_lambda=0.5,
    )

    assert torch.isfinite(losses["total"])
    assert losses["td_valid_horizon_count_mean"].item() == 1.5
    assert set(
        [
            "td_h2_target_mean",
            "td_h4_target_mean",
            "td_h2_valid_frac",
            "td_h4_valid_frac",
        ]
    ).issubset(losses)
    losses["total"].backward()
    assert model.q1.mlp.network[0].weight.grad is not None
    assert model.q2.mlp.network[0].weight.grad is not None


def test_td_lambda_averages_independent_bootstrap_values_before_mixing() -> None:
    torch.manual_seed(17)
    model = VisionIQL(
        encoder=nn.Identity(),
        q1=QNetwork(3, 2, hidden_dims=[8], use_layer_norm=False),
        q2=QNetwork(3, 2, hidden_dims=[8], use_layer_norm=False),
        v_net=VNetwork(3, hidden_dims=[8]),
        camera_keys=["cam"],
        separate_encoders=False,
        expectile=0.7,
        gamma=0.9,
        tau=0.005,
    )
    batch, horizons, target_views = 2, 2, 4
    curr_state = torch.randn(batch, 3)
    next_state = torch.randn(batch, 3)
    bootstrap_states = torch.randn(batch, horizons, target_views, 3)
    target_curr_states = torch.randn(batch, target_views, 3)
    actions = torch.randn(batch, 2)
    rewards = torch.rand(batch, 4)
    dones = torch.zeros(batch, 4, dtype=torch.long)
    discount_powers = torch.tensor([1.0, 0.9, 0.81, 0.729])
    td_horizons = torch.tensor([2, 4])
    horizon_valid = torch.ones(batch, horizons, dtype=torch.bool)

    def run(target_curr, bootstrap):
        with torch.no_grad():
            return model.forward_encoded_states(
                curr_state,
                next_state,
                actions,
                rewards,
                dones,
                discount_powers,
                target_curr_state=target_curr,
                td_lambda_bootstrap_states=bootstrap,
                td_lambda_horizons=td_horizons,
                td_lambda_horizon_valid=horizon_valid,
                td_lambda=0.5,
            )

    averaged = run(target_curr_states, bootstrap_states)
    per_view = [
        run(target_curr_states[:, view], bootstrap_states[:, :, view])
        for view in range(target_views)
    ]
    torch.testing.assert_close(
        averaged["advantage_mean"],
        torch.stack([item["advantage_mean"] for item in per_view]).mean(),
    )
    torch.testing.assert_close(
        averaged["td_target_mean"],
        torch.stack([item["td_target_mean"] for item in per_view]).mean(),
    )


def test_distributional_adaptive_tau_is_applied_per_view_before_target_average() -> None:
    torch.manual_seed(23)
    model = VisionIQL(
        encoder=nn.Identity(),
        q1=QNetwork(3, 2, hidden_dims=[8], use_layer_norm=False),
        q2=QNetwork(3, 2, hidden_dims=[8], use_layer_norm=False),
        v_net=DistributionalVNetwork(
            state_dim=3,
            hidden_dims=[8],
            num_atoms=11,
            v_min=-0.5,
            v_max=1.5,
        ),
        camera_keys=["cam"],
        separate_encoders=False,
        expectile=0.7,
        gamma=0.9,
        tau=0.005,
        distributional=True,
        tau_base=0.8,
        tau_min=0.5,
        tau_max=0.95,
        tau_entropy_alpha=0.2,
    )
    batch, target_views = 3, 4
    curr_state = torch.randn(batch, 3)
    next_state = torch.randn(batch, 3)
    target_curr_states = torch.randn(batch, target_views, 3)
    target_next_states = torch.randn(batch, target_views, 3)
    actions = torch.randn(batch, 2)
    rewards = torch.rand(batch, 2) * 0.1
    dones = torch.zeros(batch, 2, dtype=torch.long)
    discount_powers = torch.tensor([1.0, 0.9])

    def run(target_curr, target_next):
        with torch.no_grad():
            return model.forward_encoded_states(
                curr_state,
                next_state,
                actions,
                rewards,
                dones,
                discount_powers,
                target_curr_state=target_curr,
                bootstrap_next_state=target_next,
            )

    averaged = run(target_curr_states, target_next_states)
    per_view = [
        run(target_curr_states[:, view], target_next_states[:, view])
        for view in range(target_views)
    ]
    torch.testing.assert_close(
        averaged["advantage_mean"],
        torch.stack([item["advantage_mean"] for item in per_view]).mean(),
    )
    torch.testing.assert_close(
        averaged["td_target_mean"],
        torch.stack([item["td_target_mean"] for item in per_view]).mean(),
    )


def test_critic_lr_schedule_constant_and_warmup_cosine() -> None:
    from mulligan.real.train.critic import critic_lr_at_step

    base = 3e-4
    # constant schedule gives the base value at every step
    for step in (0, 1, 999, 150_000):
        assert critic_lr_at_step(
            step,
            base_lr=base,
            schedule="constant",
            warmup_steps=0,
            total_steps=150_000,
            min_frac=0.1,
        ) == pytest.approx(base)

    kw = dict(
        base_lr=base, schedule="warmup_cosine", warmup_steps=2000, total_steps=150_000, min_frac=0.1
    )
    # linear warmup from 0 to base
    assert critic_lr_at_step(0, **kw) == pytest.approx(0.0)
    assert critic_lr_at_step(1000, **kw) == pytest.approx(base * 0.5)
    assert critic_lr_at_step(2000, **kw) == pytest.approx(base)
    # cosine midpoint = mean of base and floor
    floor = base * 0.1
    mid = (2000 + 150_000) // 2
    assert critic_lr_at_step(mid, **kw) == pytest.approx((base + floor) / 2, rel=1e-3)
    # reaches the floor exactly at total_steps and never goes below it
    assert critic_lr_at_step(150_000, **kw) == pytest.approx(floor)
    lrs = [critic_lr_at_step(s, **kw) for s in range(2000, 150_001, 500)]
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1))
    assert min(lrs) >= floor - 1e-12

    with pytest.raises(ValueError):
        critic_lr_at_step(
            0,
            base_lr=base,
            schedule="warmup_cosine",
            warmup_steps=150_000,
            total_steps=150_000,
            min_frac=0.1,
        )
    with pytest.raises(ValueError):
        critic_lr_at_step(
            0, base_lr=base, schedule="warmup_cosine", warmup_steps=0, total_steps=100, min_frac=1.5
        )
    with pytest.raises(ValueError):
        critic_lr_at_step(
            0, base_lr=base, schedule="nope", warmup_steps=0, total_steps=100, min_frac=0.1
        )
