"""Critic-input contract of ``VisionIDQLRealWorldPolicy`` at deploy.

``from_artifact`` refuses critics whose metadata sets a critic-input option this release
does not ship (raw-action transform, visual LayerNorm, vision-only proprio masking), loads
critics that record the defaults or predate the keys, and refuses a metadata
``n_action_steps`` the DP's predicted horizon cannot supply. Q-scoring z-scores the raw
candidate actions.
"""

import json
from types import SimpleNamespace

import pytest
import torch

import mulligan.real.policy.dp as dp_module
from mulligan.networks.q_network import QNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.real.policy.vision_idql import VisionIDQLRealWorldPolicy

ACTION_DIM = 7
STATE_DIM = 12
HIDDEN = [8]


class _CapturingPolicy(VisionIDQLRealWorldPolicy):
    """Records the constructor kwargs from_artifact resolves, without building a policy."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _write_artifact(tmp_path, *, n_action_steps: int, **extra_metadata):
    iql_action_dim = n_action_steps * ACTION_DIM
    metadata = {
        "policy_type": "vision_idql",
        "dp_artifact": "unused",
        "state_dim": STATE_DIM,
        "action_dim": iql_action_dim,
        "hidden_dims": HIDDEN,
        "use_layer_norm": True,
        "n_action_steps": n_action_steps,
        **extra_metadata,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    q = QNetwork(STATE_DIM, iql_action_dim, hidden_dims=HIDDEN, use_layer_norm=True)
    v = VNetwork(STATE_DIM, hidden_dims=HIDDEN, use_layer_norm=True)
    torch.save(
        {
            "q1_state_dict": q.state_dict(),
            "q2_state_dict": q.state_dict(),
            "v_state_dict": v.state_dict(),
        },
        tmp_path / "iql_checkpoint.pt",
    )
    return tmp_path


def _fake_dp(*, horizon: int = 16, n_obs_steps: int = 2, n_action_steps: int = 8):
    config = SimpleNamespace(
        action_feature=SimpleNamespace(shape=(ACTION_DIM,)),
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )
    return SimpleNamespace(config=config, eval=lambda: None)


@pytest.fixture
def fake_load_dp(monkeypatch):
    dps = []

    def load_dp(local_dir, device, strict):
        dp = _fake_dp()
        dps.append(dp)
        return dp, None, None

    monkeypatch.setattr(dp_module, "load_dp", load_dp)
    return dps


def _load(artifact_dir):
    return _CapturingPolicy.from_artifact(artifact_dir, device="cpu", dp_local_dir=artifact_dir)


@pytest.mark.parametrize(
    ("metadata", "option"),
    [
        ({"action_input_transform": "signed_sqrt"}, "action_input_transform"),
        ({"visual_input_layernorm": True}, "visual_input_layernorm"),
        ({"inference_proprio_dropout": 1.0}, "inference_proprio_dropout"),
        (
            {"training_config": {"training": {"proprio_dropout": 1.0}}},
            "inference_proprio_dropout",
        ),
    ],
)
def test_from_artifact_refuses_critic_input_options_outside_the_release(
    tmp_path, fake_load_dp, metadata, option
):
    artifact = _write_artifact(tmp_path, n_action_steps=6, **metadata)
    with pytest.raises(NotImplementedError, match=option):
        _load(artifact)
    assert fake_load_dp == []  # refused before the DP actor is loaded


@pytest.mark.parametrize(
    "metadata",
    [
        {},  # critics that predate the keys
        {
            "action_input_transform": "none",
            "visual_input_layernorm": False,
            "inference_proprio_dropout": 0.0,
            "training_config": {"training": {"proprio_dropout": 0.5}},
        },
    ],
)
def test_from_artifact_loads_default_critic_inputs(tmp_path, fake_load_dp, metadata):
    kwargs = _load(_write_artifact(tmp_path, n_action_steps=6, **metadata)).kwargs
    assert "action_input_transform" not in kwargs
    assert "visual_input_layernorm" not in kwargs
    assert "proprio_dropout" not in kwargs


def test_from_artifact_rejects_n_action_steps_beyond_sliceable_horizon(tmp_path, fake_load_dp):
    # horizon=16, n_obs_steps=2: the candidate slice starts at index 1, so 15 steps remain.
    with pytest.raises(ValueError, match=r"n_action_steps must be in \[1, 15\]"):
        _load(_write_artifact(tmp_path, n_action_steps=16))


def test_from_artifact_accepts_largest_sliceable_n_action_steps(tmp_path, fake_load_dp):
    _load(_write_artifact(tmp_path, n_action_steps=15))
    assert fake_load_dp[0].config.n_action_steps == 15


def _bare_policy(*, normalize=True):
    policy = object.__new__(VisionIDQLRealWorldPolicy)
    policy._normalize_iql_inputs = normalize
    policy._iql_action_mean = torch.full((ACTION_DIM,), 0.1)
    policy._iql_action_std = torch.full((ACTION_DIM,), 0.5)
    return policy


def test_critic_action_input_is_zscored_raw_actions():
    raw = torch.randn(4, 6, ACTION_DIM)
    assert torch.equal(_bare_policy()._critic_action_input(raw), (raw - 0.1) / 0.5)
    assert _bare_policy(normalize=False)._critic_action_input(raw) is raw
