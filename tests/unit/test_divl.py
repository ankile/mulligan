"""Unit tests for DIVL (Distributional Implicit Value Learning).

Covers the distributional value network, the DIVL math utilities (HL-Gauss
soft labels, distributional cross-entropy, adaptive tau), and the IDQLPolicy
branching that swaps the scalar V for a categorical V while leaving the
actor + Q-rerank inference path untouched.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mulligan.agents.idql import IDQLPolicy
from mulligan.agents.iql_utils import (
    adaptive_tau,
    distributional_value_loss,
    expectile_loss,
    hl_gauss_target,
)
from mulligan.configs.policy import IDQLDIVLConfig, IDQLPolicyConfig
from mulligan.networks.distributional_v import DistributionalVNetwork


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class _ToyActor(nn.Module):
    """Minimal diffusion-actor stand-in.

    ``diffusion.generate_actions`` returns seed-determined random actions so the
    Q-rerank selection is non-trivial yet reproducible across two policies that
    share critic weights when the RNG is reset before each call.
    """

    def __init__(self, action_dim: int = 2, chunk_size: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.action_dim = action_dim
        self.config = SimpleNamespace(
            horizon=chunk_size,
            n_action_steps=chunk_size,
            n_obs_steps=1,
            type="diffusion",
        )
        self.diffusion = SimpleNamespace(generate_actions=self._generate_actions)

    def forward(self, batch):
        return self.weight * batch["action"].square().mean(), None

    def reset(self):
        return None

    def _generate_actions(self, obs):
        batch_rows = obs["observation.state"].shape[0]
        return torch.randn(batch_rows, self.config.n_action_steps, self.action_dim)


def _toy_normalizer(state_dim: int = 4):
    return SimpleNamespace(
        state_mean=torch.zeros(state_dim),
        state_std=torch.ones(state_dim),
        state_min=-torch.ones(state_dim),
        state_range=2 * torch.ones(state_dim),
    )


def _make_policy(divl: bool, *, state_dim=4, action_dim=2, chunk_size=1, **cfg_kwargs):
    actor = _ToyActor(action_dim=action_dim, chunk_size=chunk_size)
    if divl:
        config = IDQLDIVLConfig(
            hidden_dims=[16],
            num_q_networks=2,
            chunk_size=chunk_size,
            num_atoms=51,
            v_min=-1.0,
            v_max=2.0,
            **cfg_kwargs,
        )
    else:
        config = IDQLPolicyConfig(
            hidden_dims=[16],
            num_q_networks=2,
            chunk_size=chunk_size,
            **cfg_kwargs,
        )
    return IDQLPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        actor=actor,
        config=config,
        normalizer=_toy_normalizer(state_dim),
    )


# ---------------------------------------------------------------------------
# HL-Gauss soft labels
# ---------------------------------------------------------------------------


def test_hl_gauss_rows_sum_to_one():
    atoms = torch.linspace(-1.0, 2.0, 51)
    sigma = 0.75 * (atoms[1] - atoms[0]).item()
    targets = torch.tensor([[-0.9], [0.0], [1.0], [1.9]])
    labels = hl_gauss_target(targets, atoms, sigma)
    assert labels.shape == (4, 51)
    assert torch.allclose(labels.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert (labels >= 0).all()


def test_hl_gauss_concentrates_near_target():
    atoms = torch.linspace(-1.0, 2.0, 51)
    sigma = 0.75 * (atoms[1] - atoms[0]).item()
    target = 0.5
    labels = hl_gauss_target(torch.tensor([[target]]), atoms, sigma)
    # The expected value of the soft-label distribution should track the target.
    recovered = (labels * atoms).sum().item()
    assert recovered == pytest.approx(target, abs=2e-2)
    # The mode (argmax atom) should be the atom closest to the target.
    closest_atom = (atoms - target).abs().argmin().item()
    assert labels.argmax(dim=-1).item() == closest_atom


def test_hl_gauss_rejects_nonpositive_sigma():
    atoms = torch.linspace(-1.0, 1.0, 11)
    with pytest.raises(ValueError):
        hl_gauss_target(torch.zeros(1, 1), atoms, sigma=0.0)


def test_hl_gauss_saturates_far_outside_support_without_zero_mass():
    atoms = torch.linspace(-0.05, 1.05, 101)
    sigma = 0.75 * (atoms[1] - atoms[0]).item()
    targets = torch.tensor([[-1e6], [-0.12], [1.12], [1e6]])
    labels = hl_gauss_target(targets, atoms, sigma)
    assert torch.isfinite(labels).all()
    assert torch.allclose(labels.sum(dim=-1), torch.ones(4), atol=1e-6)
    assert labels[0].argmax().item() == labels[1].argmax().item() == 0
    assert labels[2].argmax().item() == labels[3].argmax().item() == len(atoms) - 1


def test_hl_gauss_fails_loudly_on_nonfinite_target_or_bad_support():
    atoms = torch.linspace(-1.0, 1.0, 11)
    with pytest.raises(RuntimeError, match="targets must be finite"):
        hl_gauss_target(torch.tensor([[float("nan")]]), atoms, sigma=0.1)
    with pytest.raises(RuntimeError, match="evenly spaced"):
        hl_gauss_target(
            torch.zeros(1, 1),
            torch.tensor([-1.0, -0.5, 0.25, 1.0]),
            sigma=0.1,
        )


def test_distributional_value_loss_rejects_unnormalized_labels():
    with pytest.raises(RuntimeError, match="normalized nonnegative labels"):
        distributional_value_loss(torch.zeros(2, 3), torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# DistributionalVNetwork
# ---------------------------------------------------------------------------


def test_expected_value_recovers_softmax_dot_atoms():
    torch.manual_seed(0)
    net = DistributionalVNetwork(state_dim=4, hidden_dims=[16], num_atoms=51, v_min=-1.0, v_max=2.0)
    state = torch.randn(8, 4)
    probs = net.probs(state)
    manual = (probs * net.atoms).sum(dim=-1, keepdim=True)
    assert torch.allclose(net.expected_value(state), manual, atol=1e-6)
    assert net.expected_value(state).shape == (8, 1)


def test_quantile_monotonic_and_in_range():
    torch.manual_seed(1)
    net = DistributionalVNetwork(state_dim=4, hidden_dims=[16], num_atoms=51, v_min=-1.0, v_max=2.0)
    state = torch.randn(16, 4)
    taus = [0.1, 0.3, 0.5, 0.7, 0.9]
    quantiles = [net.quantile(state, t) for t in taus]
    for q in quantiles:
        assert q.shape == (16, 1)
        assert (q >= -1.0 - 1e-6).all() and (q <= 2.0 + 1e-6).all()
    # Monotonic non-decreasing in tau (CDF inversion).
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        assert (hi >= lo - 1e-6).all()


def test_quantile_tau_near_one_returns_top_atom_not_vmin():
    """The float32 CDF can stop just below 1.0, so tau≈1 leaves the
    `cdf >= tau` mask all-False. The guard must map that to the top atom (v_max),
    never argmax's fallback index 0 (=v_min, the opposite extreme)."""
    torch.manual_seed(0)
    net = DistributionalVNetwork(state_dim=4, hidden_dims=[16], num_atoms=51, v_min=-1.0, v_max=2.0)
    state = torch.randn(32, 4)
    # tau exactly 1.0 (and a hair under) must land at/above the 0.99 quantile,
    # i.e. in the top half of the support — and crucially not collapse to v_min.
    for tau in (1.0, 1.0 - 1e-7):
        q = net.quantile(state, tau)
        assert q.shape == (32, 1)
        assert (q <= 2.0 + 1e-6).all()
        # Must be >= the 0.99 quantile everywhere (monotone), and never v_min.
        assert (q >= net.quantile(state, 0.99) - 1e-6).all()
        assert (q > -1.0 + 1e-6).all(), "tau≈1 collapsed to v_min (argmax of an all-False mask)"


def test_quantile_accepts_per_sample_tau():
    torch.manual_seed(2)
    net = DistributionalVNetwork(state_dim=4, hidden_dims=[16], num_atoms=51, v_min=-1.0, v_max=2.0)
    state = torch.randn(5, 4)
    tau = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
    q = net.quantile(state, tau)
    assert q.shape == (5, 1)


def test_normalized_entropy_bounds():
    net = DistributionalVNetwork(state_dim=4, hidden_dims=[16], num_atoms=64, v_min=-1.0, v_max=1.0)
    # Force near-uniform logits -> entropy ~1.0; force peaked logits -> ~0.0.
    with torch.no_grad():
        for layer in net.mlp.network:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
    state = torch.randn(4, 4)
    uniform_entropy = net.normalized_entropy(state)
    assert torch.allclose(uniform_entropy, torch.ones_like(uniform_entropy), atol=1e-4)

    # Peaked distribution via a hand-built one-hot-ish logit set.
    class _PeakNet(DistributionalVNetwork):
        def forward(self, state):  # type: ignore[override]
            logits = torch.full((state.shape[0], self.num_atoms), -30.0)
            logits[:, 0] = 30.0
            return logits

    peak = _PeakNet(state_dim=4, hidden_dims=[16], num_atoms=64, v_min=-1.0, v_max=1.0)
    peak_entropy = peak.normalized_entropy(state)
    assert (peak_entropy >= 0).all()
    assert (peak_entropy < 0.05).all()


def test_distributional_v_network_rejects_bad_support():
    with pytest.raises(ValueError):
        DistributionalVNetwork(state_dim=4, num_atoms=1, v_min=-1.0, v_max=1.0)
    with pytest.raises(ValueError):
        DistributionalVNetwork(state_dim=4, num_atoms=11, v_min=1.0, v_max=1.0)


# ---------------------------------------------------------------------------
# adaptive_tau
# ---------------------------------------------------------------------------


def test_adaptive_tau_fixed_when_alpha_zero():
    ent = torch.rand(8, 1)
    tau = adaptive_tau(ent, tau_base=0.7, tau_min=0.5, tau_max=0.95, alpha=0.0)
    assert tau.shape == (8, 1)
    assert torch.allclose(tau, torch.full_like(tau, 0.7))


def test_adaptive_tau_lowers_on_high_entropy_and_clips():
    # Mild alpha keeps all values inside [tau_min, tau_max] -> strictly decreasing.
    ent = torch.tensor([[0.0], [0.5], [1.0]])
    tau = adaptive_tau(ent, tau_base=0.7, tau_min=0.5, tau_max=0.95, alpha=0.2)
    assert tau[0] > tau[1] > tau[2]
    assert tau[0].item() == pytest.approx(0.7)
    assert tau[2].item() == pytest.approx(0.5)

    # Aggressive alpha drives below tau_min and is clipped up to the floor.
    tau_clipped = adaptive_tau(ent, tau_base=0.7, tau_min=0.5, tau_max=0.95, alpha=0.4)
    assert (tau_clipped >= 0.5 - 1e-6).all() and (tau_clipped <= 0.95 + 1e-6).all()
    assert tau_clipped[2].item() == pytest.approx(0.5)  # 0.7 - 0.4 = 0.30 -> clipped to 0.5


# ---------------------------------------------------------------------------
# Proposition-1 sanity: distributional value fit recovers the scalar target
# ---------------------------------------------------------------------------


def test_distributional_fit_recovers_scalar_target():
    """Fitting the categorical V toward HL-Gauss(target) drives its expected
    value and median toward the same scalar a degenerate expectile fit would."""
    torch.manual_seed(0)
    net = DistributionalVNetwork(
        state_dim=3, hidden_dims=[32], num_atoms=101, v_min=-2.0, v_max=2.0
    )
    sigma = 0.75 * (net.atoms[1] - net.atoms[0]).item()
    state = torch.randn(64, 3)
    target = torch.full((64, 1), 0.7)  # fixed scalar Q target
    soft_labels = hl_gauss_target(target, net.atoms, sigma)

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        loss = distributional_value_loss(net.forward(state), soft_labels).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        ev = net.expected_value(state).mean().item()
        median = net.quantile(state, 0.5).mean().item()
    assert ev == pytest.approx(0.7, abs=5e-2)
    assert median == pytest.approx(0.7, abs=5e-2)


# ---------------------------------------------------------------------------
# IDQLPolicy DIVL branching
# ---------------------------------------------------------------------------


def test_divl_policy_builds_distributional_value_net():
    policy = _make_policy(divl=True)
    assert policy._is_divl is True
    assert isinstance(policy.value, DistributionalVNetwork)
    assert isinstance(policy.target_value, DistributionalVNetwork)

    scalar_policy = _make_policy(divl=False)
    assert scalar_policy._is_divl is False
    assert not isinstance(scalar_policy.value, DistributionalVNetwork)


def test_divl_config_requires_support_at_construction():
    actor = _ToyActor()
    config = IDQLDIVLConfig(hidden_dims=[16], num_q_networks=1)  # v_min/v_max default None
    with pytest.raises(ValueError):
        IDQLPolicy(state_dim=4, action_dim=2, actor=actor, config=config)


def test_divl_loss_value_and_critic_finite_with_expected_keys():
    torch.manual_seed(0)
    policy = _make_policy(divl=True, chunk_size=1)
    batch_size = 8
    batch = {
        "observation.state": torch.randn(batch_size, 4),
        "action": torch.randn(batch_size, 2),
        "reward": torch.randn(batch_size, 1),
        "next.observation.state": torch.randn(batch_size, 4),
        "masks": torch.ones(batch_size, 1),
        "chunk_valid": torch.ones(batch_size, 1),
    }
    value_out = policy.compute_loss_value(batch)
    assert torch.isfinite(value_out["loss_value"])
    for key in ("v_mean", "v_std", "q_data_mean", "advantage_mean"):
        assert key in value_out

    critic_out = policy.compute_loss_critic(batch)
    assert torch.isfinite(critic_out["loss_critic"])
    assert torch.isfinite(critic_out["target_v_mean"])


def test_divl_adaptive_tau_path_runs():
    torch.manual_seed(0)
    policy = _make_policy(divl=True, chunk_size=1, tau_entropy_alpha=0.4)
    batch_size = 8
    batch = {
        "observation.state": torch.randn(batch_size, 4),
        "action": torch.randn(batch_size, 2),
        "reward": torch.randn(batch_size, 1),
        "next.observation.state": torch.randn(batch_size, 4),
        "masks": torch.ones(batch_size, 1),
        "chunk_valid": torch.ones(batch_size, 1),
    }
    out = policy.compute_loss_critic(batch)
    assert torch.isfinite(out["loss_critic"])


def test_select_action_identical_for_idql_and_divl_given_same_critics():
    """V isn't used at inference — given identical critics + actor, idql and
    idql_divl must select the same action chunk."""
    idql = _make_policy(divl=False, chunk_size=1)
    divl = _make_policy(divl=True, chunk_size=1)

    # Share the actor and copy critic weights so the only difference is the
    # value network (which select_action must not touch for selection).
    divl.actor.load_state_dict(idql.actor.state_dict())
    for c_src, c_dst in zip(idql.critics, divl.critics):
        c_dst.load_state_dict(c_src.state_dict())

    obs = {
        "observation.state": torch.randn(3, 2),
        "observation.environment_state": torch.randn(3, 2),
    }

    torch.manual_seed(123)
    idql.reset()
    a_idql = idql.select_action({k: v.clone() for k, v in obs.items()})

    torch.manual_seed(123)
    divl.reset()
    a_divl = divl.select_action({k: v.clone() for k, v in obs.items()})

    assert a_idql.shape == a_divl.shape == (3, 2)
    assert torch.allclose(a_idql, a_divl, atol=1e-6)


def test_divl_full_update_step_runs_and_changes_value_params():
    """End-to-end update(): exercises distributional value loss, critic loss,
    optimizer steps, and target Polyak with the distributional V-network."""
    torch.manual_seed(0)
    policy = _make_policy(divl=True, chunk_size=1)
    optimizers = {
        "actor": torch.optim.Adam(policy.actor.parameters(), lr=1e-3),
        "critic": torch.optim.Adam(policy.critics.parameters(), lr=1e-3),
        "value": torch.optim.Adam(policy.value.parameters(), lr=1e-2),
    }
    before = [p.detach().clone() for p in policy.value.parameters()]

    batch_size = 8
    critic_batch = {
        "observation.state": torch.randn(batch_size, 4),
        "action": torch.randn(batch_size, 1, 2),
        "reward": torch.randn(batch_size, 1),
        "next.observation.state": torch.randn(batch_size, 4),
        "masks": torch.ones(batch_size, 1),
        "chunk_valid": torch.ones(batch_size, 1),
    }
    actor_batch = {
        "observation.state": torch.randn(batch_size, 1, 4),
        "action": torch.randn(batch_size, 1, 2),
        "action_is_pad": torch.zeros(batch_size, 1, dtype=torch.bool),
    }

    metrics = policy.update(critic_batch, optimizers, policy_batch=actor_batch)
    assert torch.isfinite(metrics["losses/value"])
    assert torch.isfinite(metrics["losses/critic"])
    assert torch.isfinite(metrics["losses/actor"])
    # The distributional value net actually trained.
    after = list(policy.value.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_divl_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    policy = _make_policy(divl=True, chunk_size=1)
    policy.eval()

    save_dir = tmp_path / "divl_ckpt"
    policy.save(save_dir)

    fresh_actor = _ToyActor(action_dim=2, chunk_size=1)
    # actor is the second positional argument, as in the original IDQLPolicy.load
    loaded = IDQLPolicy.load(save_dir, fresh_actor, "cpu")

    assert isinstance(loaded.value, DistributionalVNetwork)
    assert loaded.value.num_atoms == policy.value.num_atoms
    assert torch.allclose(loaded.value.atoms, policy.value.atoms)

    state = torch.randn(5, 4)
    with torch.no_grad():
        assert torch.allclose(
            loaded.value.expected_value(state),
            policy.value.expected_value(state),
            atol=1e-6,
        )


def test_divl_can_load_only_scalar_parent_actor(tmp_path):
    torch.manual_seed(0)
    parent = _make_policy(divl=False, chunk_size=1)
    parent.actor.weight.data.fill_(3.25)
    parent.save(tmp_path)

    torch.manual_seed(1)
    child = _make_policy(divl=True, chunk_size=1)
    value_hash_before = child._state_dict_sha256(child.value.state_dict())
    child.load_actor_checkpoint(tmp_path, strict_contract=True)

    assert child.actor_hash() == parent.actor_hash()
    assert child._state_dict_sha256(child.value.state_dict()) == value_hash_before
    assert isinstance(child.value, DistributionalVNetwork)


def test_divl_full_load_requires_the_parent_value_support(tmp_path):
    torch.manual_seed(0)
    parent = _make_policy(divl=True, chunk_size=1)  # support [-1, 2] x 51
    parent.save(tmp_path)

    same = _make_policy(divl=True, chunk_size=1)
    same.load_complete_checkpoint(tmp_path, strict_contract=True)
    assert torch.equal(same.value.atoms, parent.value.atoms)

    config = IDQLDIVLConfig(hidden_dims=[16], num_q_networks=2, num_atoms=51, v_min=-3.0, v_max=2.0)
    child = IDQLPolicy(
        state_dim=4,
        action_dim=2,
        actor=_ToyActor(action_dim=2, chunk_size=1),
        config=config,
        normalizer=_toy_normalizer(4),
    )
    atoms_before = child.value.atoms.clone()
    with pytest.raises(ValueError, match=r"--policy.v_min=-1.0 --policy.v_max=2.0"):
        child.load_complete_checkpoint(tmp_path, strict_contract=True)
    assert torch.equal(child.value.atoms, atoms_before)

    scalar = _make_policy(divl=False, chunk_size=1)
    scalar.save(tmp_path / "scalar")
    with pytest.raises(ValueError, match="only a DIVL parent"):
        _make_policy(divl=True, chunk_size=1).load_complete_checkpoint(tmp_path / "scalar")


def test_critic_value_only_update_freezes_actor_and_updates_qv():
    torch.manual_seed(0)
    policy = _make_policy(divl=True, chunk_size=1)
    policy.set_critic_value_only_training()
    actor_hash_before = policy.actor_hash()
    critic_before = [parameter.detach().clone() for parameter in policy.critics.parameters()]
    value_before = [parameter.detach().clone() for parameter in policy.value.parameters()]
    optimizers = {
        "critic": torch.optim.Adam(policy.critics.parameters(), lr=1e-3),
        "value": torch.optim.Adam(policy.value.parameters(), lr=1e-3),
    }
    batch_size = 8
    critic_batch = {
        "observation.state": torch.randn(batch_size, 4),
        "action": torch.randn(batch_size, 1, 2),
        "reward": torch.randn(batch_size, 1),
        "next.observation.state": torch.randn(batch_size, 4),
        "masks": torch.ones(batch_size, 1),
        "chunk_valid": torch.ones(batch_size, 1),
    }

    policy.train()
    metrics = policy.update(
        critic_batch,
        optimizers,
        update_components="critic_value_only",
    )

    assert not policy.actor.training
    assert policy.actor_hash() == actor_hash_before
    assert "losses/actor" not in metrics
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, policy.critics.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(value_before, policy.value.parameters(), strict=True)
    )


def test_expectile_loss_still_default_for_scalar_idql():
    """Guard: the scalar IDQL value path keeps using expectile_loss."""
    torch.manual_seed(0)
    policy = _make_policy(divl=False, chunk_size=1)
    batch_size = 8
    batch = {
        "observation.state": torch.randn(batch_size, 4),
        "action": torch.randn(batch_size, 2),
        "reward": torch.randn(batch_size, 1),
        "next.observation.state": torch.randn(batch_size, 4),
        "masks": torch.ones(batch_size, 1),
        "chunk_valid": torch.ones(batch_size, 1),
    }
    out = policy.compute_loss_value(batch)
    # Reproduce the expectile loss by hand from the same V/Q to confirm the path.
    state = batch["observation.state"]
    v = policy.value(state)
    q_all = policy._stack_critic_outputs(policy.critics, state, batch["action"])
    q = policy._aggregate_q_values(q_all)
    expected = expectile_loss(q - v, expectile=policy.config.expectile).mean()
    assert torch.allclose(out["loss_value"], expected, atol=1e-6)
