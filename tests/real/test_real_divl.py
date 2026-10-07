"""Unit/smoke tests for the real-world Vision-IQL DIVL variant.

Mirrors the sim DIVL tests but targets the REAL training path: the
``VisionIQL`` module (encoder + Q1/Q2 + categorical V) and the critic trainer's
value-support derivation.

Covers:
- a distributional VisionIQL runs one forward+backward on CPU random tensors,
  losses finite, V is categorical (logits over num_atoms);
- the non-distributional VisionIQL path is identical to the scalar baseline
  (same value loss given identical weights);
- ``resolve_divl_value_support`` derivation + guards.
"""

import pytest
import torch
import torch.nn as nn

from mulligan.networks.distributional_v import DistributionalVNetwork
from mulligan.networks.q_network import QNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.networks.vision_iql import VisionIQL


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CAM_KEY = "observation.images.cam_left"
IMG_FLAT = 12  # flattened "image" dim for the toy encoder
FEAT_DIM = 8
PROPRIO_DIM = 4
ACTION_DIM = 2
K = 2  # critic horizon (n_action_steps)


class _ToyEncoder(nn.Module):
    """Maps a flat (B, IMG_FLAT) image tensor to (B, FEAT_DIM) features."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(IMG_FLAT, FEAT_DIM)
        self.register_buffer("running", torch.zeros(1))

    def forward(self, x):
        return self.lin(x)


def _state_dim():
    # one camera -> FEAT_DIM visual + PROPRIO_DIM proprio
    return FEAT_DIM + PROPRIO_DIM


def _make_vision_iql(distributional: bool, *, v_min=-1.0, v_max=2.0, num_atoms=51, **kwargs):
    state_dim = _state_dim()
    chunked_action_dim = ACTION_DIM * K
    q1 = QNetwork(state_dim, chunked_action_dim, hidden_dims=[16], use_layer_norm=False)
    q2 = QNetwork(state_dim, chunked_action_dim, hidden_dims=[16], use_layer_norm=False)
    if distributional:
        v_net = DistributionalVNetwork(
            state_dim, hidden_dims=[16], num_atoms=num_atoms, v_min=v_min, v_max=v_max
        )
    else:
        v_net = VNetwork(state_dim, hidden_dims=[16])
    return VisionIQL(
        encoder=_ToyEncoder(),
        q1=q1,
        q2=q2,
        v_net=v_net,
        camera_keys=[CAM_KEY],
        separate_encoders=False,
        expectile=0.7,
        gamma=0.99,
        tau=0.005,
        distributional=distributional,
        **kwargs,
    )


def _random_forward_inputs(batch_size=8):
    return dict(
        curr_images={CAM_KEY: torch.randn(batch_size, IMG_FLAT)},
        next_images={CAM_KEY: torch.randn(batch_size, IMG_FLAT)},
        proprio_curr=torch.randn(batch_size, PROPRIO_DIM),
        proprio_next=torch.randn(batch_size, PROPRIO_DIM),
        actions=torch.randn(batch_size, ACTION_DIM * K),
        rewards=torch.rand(batch_size, K),
        dones=torch.zeros(batch_size, K),
        discount_powers=torch.ones(batch_size, K),
    )


# ---------------------------------------------------------------------------
# Distributional forward/backward smoke
# ---------------------------------------------------------------------------


def test_distributional_vision_iql_forward_backward_finite():
    torch.manual_seed(0)
    model = _make_vision_iql(distributional=True)
    out = model(**_random_forward_inputs())
    assert torch.isfinite(out["total"])
    assert torch.isfinite(out["value_loss"])
    assert torch.isfinite(out["critic_loss"])
    assert torch.isfinite(out["v_mean"])
    out["total"].backward()
    # V-net actually received gradients.
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.v_net.parameters()
    )


def test_td_reward_horizon_can_exceed_ranked_action_horizon():
    torch.manual_seed(0)
    model = _make_vision_iql(distributional=True)
    inputs = _random_forward_inputs()
    batch_size = inputs["actions"].shape[0]
    td_horizon = 3 * K
    inputs["rewards"] = torch.rand(batch_size, td_horizon)
    inputs["dones"] = torch.zeros(batch_size, td_horizon)
    inputs["discount_powers"] = torch.tensor([0.99**i for i in range(td_horizon)])

    out = model(**inputs)

    assert inputs["actions"].shape[1] == ACTION_DIM * K
    assert inputs["rewards"].shape[1] == td_horizon
    assert torch.isfinite(out["total"])
    assert torch.isfinite(out["td_target_mean"])


def test_distributional_v_is_categorical():
    model = _make_vision_iql(distributional=True, num_atoms=51)
    assert isinstance(model.v_net, DistributionalVNetwork)
    assert not hasattr(model, "v_target")
    assert isinstance(model.q1_target, QNetwork)
    assert isinstance(model.q2_target, QNetwork)
    state = torch.randn(5, _state_dim())
    logits = model.v_net(state)
    assert logits.shape == (5, 51)  # logits over the support, not a scalar
    probs = model.v_net.probs(state)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert model.v_net.expected_value(state).shape == (5, 1)


def test_distributional_update_targets_runs():
    """Polyak update moves target Q toward online Q and never targets V."""
    model = _make_vision_iql(distributional=True)
    target_before = [p.detach().clone() for p in model.q1_target.parameters()]
    with torch.no_grad():
        for p in model.q1.parameters():
            p.add_(1.0)
        model.encoder.running.fill_(1.0)
    model.update_targets()
    for before, target, online in zip(
        target_before, model.q1_target.parameters(), model.q1.parameters(), strict=True
    ):
        assert torch.allclose(target, before.lerp(online, model.tau))
    assert all(not p.requires_grad for p in model.q1_target.parameters())
    assert all(not p.requires_grad for p in model.q2_target.parameters())
    assert model.encoder_target.running.item() == pytest.approx(model.tau)


def test_target_critics_stay_eval_and_round_trip_in_training_state():
    model = _make_vision_iql(distributional=True)
    model.train()
    assert not model.encoder_target.training
    assert not model.q1_target.training
    assert not model.q2_target.training
    state = model.state_dict()
    assert any(key.startswith("q1_target.") for key in state)
    assert any(key.startswith("q2_target.") for key in state)
    assert not any(key.startswith("v_target.") for key in state)
    clone = _make_vision_iql(distributional=True)
    clone.load_state_dict(state)


def test_value_labels_come_from_target_q_not_online_q():
    model = _make_vision_iql(distributional=False)
    model.q1_target.forward = lambda state, action: state.new_full((len(state), 1), 2.0)
    model.q2_target.forward = lambda state, action: state.new_full((len(state), 1), 3.0)
    model.q1.forward = lambda state, action: state.new_full((len(state), 1), 100.0)
    model.q2.forward = lambda state, action: state.new_full((len(state), 1), 100.0)
    model.v_net.forward = lambda state: state.new_zeros((len(state), 1))
    out = model(**_random_forward_inputs())
    assert out["value_loss"].item() == pytest.approx(0.7 * 2.0**2)


def test_td_bootstrap_comes_from_online_v():
    model = _make_vision_iql(distributional=False)
    model.v_net.forward = lambda state: state.new_full((len(state), 1), 3.0)
    inputs = _random_forward_inputs()
    inputs["rewards"].zero_()
    inputs["dones"].zero_()
    out = model(**inputs)
    assert out["td_target_mean"].item() == pytest.approx(model.gamma**K * 3.0)


def test_target_clipping_bounds_the_complete_bellman_label():
    model = _make_vision_iql(
        distributional=False,
        clip_targets_min=0.0,
        clip_targets_max=1.0,
    )
    inputs = _random_forward_inputs()
    inputs["rewards"].fill_(5.0)
    inputs["dones"].fill_(True)
    out = model(**inputs)
    assert out["td_target_mean"].item() == pytest.approx(1.0)


def test_target_clipping_does_not_hide_infinite_online_v():
    model = _make_vision_iql(
        distributional=False,
        clip_targets_min=0.0,
        clip_targets_max=1.0,
    )
    model.v_net.forward = lambda state: state.new_full((len(state), 1), float("inf"))
    with pytest.raises(RuntimeError, match="before clipping"):
        model(**_random_forward_inputs())


def test_target_clipping_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="exceeds"):
        _make_vision_iql(
            distributional=False,
            clip_targets_min=1.0,
            clip_targets_max=0.0,
        )


def test_adaptive_tau_path_runs():
    torch.manual_seed(0)
    model = _make_vision_iql(distributional=True, tau_entropy_alpha=0.4)
    out = model(**_random_forward_inputs())
    assert torch.isfinite(out["critic_loss"])


def test_distributional_flag_requires_distributional_v_net():
    state_dim = _state_dim()
    with pytest.raises(ValueError):
        VisionIQL(
            encoder=_ToyEncoder(),
            q1=QNetwork(state_dim, ACTION_DIM * K, hidden_dims=[16]),
            q2=QNetwork(state_dim, ACTION_DIM * K, hidden_dims=[16]),
            v_net=VNetwork(state_dim, hidden_dims=[16]),  # scalar V with the flag on
            camera_keys=[CAM_KEY],
            separate_encoders=False,
            expectile=0.7,
            gamma=0.99,
            tau=0.005,
            distributional=True,
        )


# ---------------------------------------------------------------------------
# Non-distributional path is unchanged
# ---------------------------------------------------------------------------


def test_non_distributional_path_matches_target_q_expectile():
    """Scalar V loss is expectile regression against the target Q minimum."""
    torch.manual_seed(0)
    model = _make_vision_iql(distributional=False)
    inputs = _random_forward_inputs()
    out = model(**inputs)
    assert torch.isfinite(out["total"])

    # Recompute the expectile value loss by hand from the same weights.
    from mulligan.agents.iql_utils import expectile_loss

    with torch.no_grad():
        curr_visual = model.encode(inputs["curr_images"], model.encoder_target)
        curr_state = torch.cat([curr_visual, inputs["proprio_curr"]], dim=-1)
        q_min = torch.min(
            model.q1_target(curr_state, inputs["actions"]),
            model.q2_target(curr_state, inputs["actions"]),
        )
        # q-only grad source detaches the V state.
        v_val = model.v_net(curr_state.detach())
        expected = expectile_loss(q_min - v_val, model.expectile).mean()
    assert torch.allclose(out["value_loss"], expected, atol=1e-6)


def test_non_distributional_v_net_is_scalar():
    model = _make_vision_iql(distributional=False)
    assert not isinstance(model.v_net, DistributionalVNetwork)
    assert model.v_net(torch.randn(3, _state_dim())).shape == (3, 1)
    assert model.distributional is False


# ---------------------------------------------------------------------------
# Value-support derivation (critic trainer helper)
# ---------------------------------------------------------------------------


def test_resolve_divl_value_support_derives_with_margin():
    from mulligan.real.train.critic import resolve_divl_value_support

    v_min, v_max = resolve_divl_value_support(None, None, (0.0, 10.0))
    assert v_min == pytest.approx(-0.5)  # 0 - 0.05*10
    assert v_max == pytest.approx(10.5)  # 10 + 0.05*10


def test_resolve_divl_value_support_keeps_override():
    from mulligan.real.train.critic import resolve_divl_value_support

    assert resolve_divl_value_support(-2.0, 3.0, (0.0, 1.0)) == (-2.0, 3.0)


def test_resolve_divl_value_support_rejects_partial_override():
    from mulligan.real.train.critic import resolve_divl_value_support

    with pytest.raises(ValueError):
        resolve_divl_value_support(-2.0, None, (0.0, 1.0))
    with pytest.raises(ValueError):
        resolve_divl_value_support(None, 3.0, (0.0, 1.0))


def test_resolve_divl_value_support_rejects_degenerate_range():
    from mulligan.real.train.critic import resolve_divl_value_support

    with pytest.raises(ValueError):
        resolve_divl_value_support(None, None, (1.0, 1.0))


class _FakeHF:
    def __init__(self, cols):
        self._cols = cols

    @property
    def column_names(self):
        return list(self._cols)

    def __getitem__(self, key):
        return self._cols[key]


class _FakeSub:
    def __init__(self, cols, n):
        self.hf_dataset = _FakeHF(cols)
        self._n = n

    def __len__(self):
        return self._n


def test_empirical_discounted_rtg_range_done_aware_recursion():
    """Support is bracketed by the done-aware discounted return-to-go, not the
    undiscounted episode reward sum."""
    from mulligan.real.train.critic import empirical_discounted_rtg_range

    # One episode, sparse terminal reward: rewards [0,0,1], done [F,F,T], gamma 0.9.
    # RTG backward: G2=1.0, G1=0.9, G0=0.81 -> range (0.81, 1.0).
    sub = _FakeSub(
        {"reward": [0.0, 0.0, 1.0], "done": [False, False, True], "episode_index": [0, 0, 0]}, 3
    )
    lo, hi = empirical_discounted_rtg_range(
        [sub],
        ["r"],
        gamma=0.9,
        reward_shift=0.0,
        intervention_negative_reward=None,
        intervention_values_by_dataset=[],
    )
    assert lo == pytest.approx(0.81)
    assert hi == pytest.approx(1.0)


def test_empirical_discounted_rtg_range_applies_intervention_shaping():
    """Intervention penalty on a frame pulls its RTG negative (the support's negative
    tail comes from shaped training states, not the eval reward convention)."""
    from mulligan.real.train.critic import empirical_discounted_rtg_range

    sub = _FakeSub(
        {"reward": [0.0, 0.0, 1.0], "done": [False, False, True], "episode_index": [0, 0, 0]}, 3
    )
    interv = torch.tensor([1, 0, 0], dtype=torch.long)
    # shaped [-1,0,1]: G2=1.0, G1=0.9, G0=-1+0.9*0.9=-0.19 -> range (-0.19, 1.0).
    lo, hi = empirical_discounted_rtg_range(
        [sub],
        ["r"],
        gamma=0.9,
        reward_shift=0.0,
        intervention_negative_reward=-1.0,
        intervention_values_by_dataset=[interv],
    )
    assert lo == pytest.approx(-0.19)
    assert hi == pytest.approx(1.0)


def test_empirical_discounted_rtg_range_resets_on_done():
    """A mid-stream done resets accumulation (terminal padding must not accrue)."""
    from mulligan.real.train.critic import empirical_discounted_rtg_range

    sub = _FakeSub(
        {
            "reward": [0.0, 1.0, 0.0, 1.0],
            "done": [False, True, False, True],
            "episode_index": [0, 0, 0, 0],
        },
        4,
    )
    # G3=1.0, G2=0.9, G1=1.0 (reset), G0=0.9 -> range (0.9, 1.0).
    lo, hi = empirical_discounted_rtg_range(
        [sub],
        ["r"],
        gamma=0.9,
        reward_shift=0.0,
        intervention_negative_reward=None,
        intervention_values_by_dataset=[],
    )
    assert lo == pytest.approx(0.9)
    assert hi == pytest.approx(1.0)


def test_discounted_mc_returns_by_dataset_returns_per_frame_targets():
    from mulligan.real.train.critic import discounted_mc_returns_by_dataset

    sub = _FakeSub(
        {
            "reward": [0.0, 0.0, 1.0, 0.0],
            "done": [False, False, True, True],
            "episode_index": [0, 0, 0, 1],
        },
        4,
    )
    returns = discounted_mc_returns_by_dataset(
        [sub],
        ["r"],
        gamma=0.9,
        reward_shift=0.0,
        intervention_negative_reward=None,
        intervention_values_by_dataset=[],
    )
    assert len(returns) == 1
    assert returns[0].tolist() == pytest.approx([0.81, 0.9, 1.0, 0.0])
