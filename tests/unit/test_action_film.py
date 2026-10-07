"""Tests for the observation-conditioned learned action-scaling ("action
FiLM") head on the VisionIQL critic action input.

Invariants under test:
  - flag OFF (default) => forward is byte-identical to the path without the head (no
    film_head module, _film_actions is an identity passthrough, total loss is
    invariant to any FiLM RNG that never runs).
  - identity-at-init: a zero-initialised FiLM head (flag ON) produces the exact
    same Q inputs / losses as the flag-OFF model built from the same critic
    weights.
  - the target FiLM head is Polyak-updated toward the online head like the
    target critics.
  - the pure ``apply_action_film`` helper leaves the trailing (gripper) dims
    untouched and scales only the leading arm dims.
"""

import pytest
import torch
import torch.nn as nn

from mulligan.networks.q_network import QNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.networks.vision_iql import VisionIQL
from mulligan.networks.action_film import apply_action_film

CAM_KEY = "observation.images.cam_left"
IMG_SHAPE = (3, 8, 8)
FEAT_DIM = 6
PROPRIO_DIM = 4
STATE_DIM = FEAT_DIM + PROPRIO_DIM
STEP_DIM = 3  # per-step action dim (2 arm + 1 gripper)
ARM_DIMS = 2
K = 2  # reward/done horizon and action-chunk length
FLAT_ACTION = STEP_DIM * K


class _ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3 * 8 * 8, FEAT_DIM)

    def forward(self, x):
        return self.lin(x.flatten(start_dim=1))


def _make_model(encoder=None, **kwargs):
    encoder = encoder if encoder is not None else _ImageEncoder()
    q1 = QNetwork(STATE_DIM, FLAT_ACTION, hidden_dims=[16], use_layer_norm=False)
    q2 = QNetwork(STATE_DIM, FLAT_ACTION, hidden_dims=[16], use_layer_norm=False)
    v_net = VNetwork(STATE_DIM, hidden_dims=[16])
    return VisionIQL(
        encoder=encoder,
        q1=q1,
        q2=q2,
        v_net=v_net,
        camera_keys=[CAM_KEY],
        separate_encoders=False,
        expectile=0.7,
        gamma=0.99,
        tau=0.005,
        **kwargs,
    )


def _forward_inputs(b, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        curr_images={CAM_KEY: torch.rand(b, *IMG_SHAPE, generator=g)},
        next_images={CAM_KEY: torch.rand(b, *IMG_SHAPE, generator=g)},
        proprio_curr=torch.rand(b, PROPRIO_DIM, generator=g),
        proprio_next=torch.rand(b, PROPRIO_DIM, generator=g),
        actions=torch.randn(b, FLAT_ACTION, generator=g),
        rewards=torch.rand(b, K, generator=g),
        dones=torch.zeros(b, K),
        discount_powers=torch.tensor([0.99**i for i in range(K)]),
    )


def _copy_critic_weights(src: VisionIQL, dst: VisionIQL):
    """Mirror the shared (non-FiLM) critic weights src -> dst."""
    dst.encoder.load_state_dict(src.encoder.state_dict())
    dst.encoder_target.load_state_dict(src.encoder_target.state_dict())
    dst.q1.load_state_dict(src.q1.state_dict())
    dst.q2.load_state_dict(src.q2.state_dict())
    dst.q1_target.load_state_dict(src.q1_target.state_dict())
    dst.q2_target.load_state_dict(src.q2_target.state_dict())
    dst.v_net.load_state_dict(src.v_net.state_dict())


# ---------------------------------------------------------------------------
# pure helper — gripper untouched + shapes
# ---------------------------------------------------------------------------


def test_apply_action_film_zero_params_is_identity():
    actions = torch.randn(5, FLAT_ACTION)
    params = torch.zeros(5, 2 * ARM_DIMS)
    out = apply_action_film(actions, params, arm_dims=ARM_DIMS, step_dim=STEP_DIM)
    assert torch.equal(out, actions)


def test_apply_action_film_gripper_dim_untouched():
    torch.manual_seed(0)
    actions = torch.randn(4, FLAT_ACTION)
    params = torch.randn(4, 2 * ARM_DIMS)  # nonzero mean + log_std
    out = apply_action_film(actions, params, arm_dims=ARM_DIMS, step_dim=STEP_DIM)
    a_steps = actions.reshape(4, K, STEP_DIM)
    o_steps = out.reshape(4, K, STEP_DIM)
    # trailing (gripper) dims are byte-identical
    assert torch.equal(o_steps[:, :, ARM_DIMS:], a_steps[:, :, ARM_DIMS:])
    # leading arm dims are actually changed (nonzero params)
    assert not torch.allclose(o_steps[:, :, :ARM_DIMS], a_steps[:, :, :ARM_DIMS])


def test_apply_action_film_matches_manual_formula():
    torch.manual_seed(1)
    actions = torch.randn(3, FLAT_ACTION)
    mean = torch.randn(3, ARM_DIMS)
    log_std = torch.randn(3, ARM_DIMS).clamp(-2, 2)
    params = torch.cat([mean, log_std], dim=-1)
    out = apply_action_film(actions, params, arm_dims=ARM_DIMS, step_dim=STEP_DIM)
    a_steps = actions.reshape(3, K, STEP_DIM)
    want_arm = (a_steps[:, :, :ARM_DIMS] - mean[:, None, :]) / torch.exp(log_std)[:, None, :]
    assert torch.allclose(out.reshape(3, K, STEP_DIM)[:, :, :ARM_DIMS], want_arm, atol=1e-6)


def test_apply_action_film_clamps_log_std():
    actions = torch.ones(1, FLAT_ACTION)
    # log_std = 100 must clamp to +2 => divide arm dims by exp(2), not exp(100).
    params = torch.cat([torch.zeros(1, ARM_DIMS), torch.full((1, ARM_DIMS), 100.0)], dim=-1)
    out = apply_action_film(actions, params, arm_dims=ARM_DIMS, step_dim=STEP_DIM)
    expected_arm = 1.0 / torch.exp(torch.tensor(2.0))
    o_steps = out.reshape(1, K, STEP_DIM)
    assert torch.allclose(o_steps[:, :, :ARM_DIMS], expected_arm.expand(1, K, ARM_DIMS), atol=1e-6)


# ---------------------------------------------------------------------------
# flag OFF byte-identity
# ---------------------------------------------------------------------------


def test_film_off_no_head_and_identity_passthrough():
    model = _make_model()  # default: action_film_head == False
    assert model.action_film_head is False
    assert not hasattr(model, "film_head")
    a = torch.randn(4, FLAT_ACTION)
    s = torch.randn(4, STATE_DIM)
    assert model._film_actions(a, s, target=False) is a
    assert model._film_actions(a, s, target=True) is a


def test_film_off_total_invariant_to_rng():
    torch.manual_seed(0)
    model = _make_model()
    inp = _forward_inputs(8)
    torch.manual_seed(1)
    a = model(**inp)["total"]
    torch.manual_seed(999)
    b = model(**inp)["total"]
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# identity-at-init (zero-init FiLM head == flag-off model, same weights)
# ---------------------------------------------------------------------------


def test_film_on_identity_at_init_matches_off():
    torch.manual_seed(0)
    off = _make_model()
    on = _make_model(
        action_film_head=True,
        action_film_hidden=16,
        action_film_arm_dims=ARM_DIMS,
        action_film_step_dim=STEP_DIM,
        action_film_state_dim=STATE_DIM,
    )
    _copy_critic_weights(off, on)
    # zero-init final layer => film_head(state) == 0 => identity transform.
    assert on.action_film_head is True
    assert torch.count_nonzero(on.film_head[-1].weight) == 0
    assert torch.count_nonzero(on.film_head[-1].bias) == 0

    inp = _forward_inputs(8)
    out_off = off(**inp)
    out_on = on(**inp)
    for key in ("total", "value_loss", "critic_loss", "q1_mean", "q2_mean"):
        assert torch.allclose(out_off[key], out_on[key], atol=1e-6), key


def test_film_on_nontrivial_after_perturbation_diverges():
    torch.manual_seed(0)
    off = _make_model()
    on = _make_model(
        action_film_head=True,
        action_film_hidden=16,
        action_film_arm_dims=ARM_DIMS,
        action_film_step_dim=STEP_DIM,
        action_film_state_dim=STATE_DIM,
    )
    _copy_critic_weights(off, on)
    with torch.no_grad():
        on.film_head[-1].bias.add_(0.5)  # nonzero mean/log_std => real transform
    inp = _forward_inputs(8)
    assert not torch.allclose(off(**inp)["q1_mean"], on(**inp)["q1_mean"], atol=1e-4)


def test_film_head_trains_through_q_loss():
    """FiLM head params receive gradient from the critic loss."""
    torch.manual_seed(0)
    model = _make_model(
        action_film_head=True,
        action_film_hidden=16,
        action_film_arm_dims=ARM_DIMS,
        action_film_step_dim=STEP_DIM,
        action_film_state_dim=STATE_DIM,
    )
    with torch.no_grad():  # break the identity so gradient is nonzero
        model.film_head[-1].weight.add_(0.1)
    model.zero_grad(set_to_none=True)
    model(**_forward_inputs(8))["total"].backward()
    grad = sum(
        p.grad.abs().sum().item() for p in model.film_head.parameters() if p.grad is not None
    )
    assert grad > 0.0


# ---------------------------------------------------------------------------
# target head Polyak
# ---------------------------------------------------------------------------


def test_film_target_head_polyak_update():
    model = _make_model(
        action_film_head=True,
        action_film_hidden=16,
        action_film_arm_dims=ARM_DIMS,
        action_film_step_dim=STEP_DIM,
        action_film_state_dim=STATE_DIM,
    )
    with torch.no_grad():
        for p in model.film_head.parameters():
            p.fill_(1.0)
        for p in model.film_head_target.parameters():
            p.fill_(0.0)
    model.update_targets(update_encoder=False)
    # lerp_(online=1, tau=0.005) from 0 => 0.005 everywhere.
    for p in model.film_head_target.parameters():
        assert torch.allclose(p, torch.full_like(p, model.tau), atol=1e-8)
        assert not p.requires_grad


def test_film_target_stays_eval_across_train_toggle():
    model = _make_model(
        action_film_head=True,
        action_film_hidden=16,
        action_film_arm_dims=ARM_DIMS,
        action_film_step_dim=STEP_DIM,
        action_film_state_dim=STATE_DIM,
    )
    model.train()
    assert model.film_head.training is True
    assert model.film_head_target.training is False


# ---------------------------------------------------------------------------
# offline-scorer mirror parity + metadata roundtrip
# ---------------------------------------------------------------------------


def _build_film_head():
    torch.manual_seed(3)
    head = nn.Sequential(nn.Linear(STATE_DIM, 16), nn.ReLU(), nn.Linear(16, 2 * ARM_DIMS))
    with torch.no_grad():
        head[-1].bias.add_(0.3)  # nonzero => real transform
    head.eval()
    return head


def test_film_head_requires_dims():
    with pytest.raises(ValueError, match="action_film_state_dim"):
        _make_model(action_film_head=True, action_film_step_dim=None, action_film_state_dim=None)
