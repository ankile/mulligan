"""Deploy-mirror tests for the action FiLM head and the twin-Q aggregation in
``VisionIDQLRealWorldPolicy``.

Contract under test:
  1. Non-FiLM checkpoints are unchanged: ``load_action_film_head`` returns
     ``(None, 0, 0)`` and ``_film_q_actions`` returns the SAME object, so the
     policy's Q inputs — and therefore ``select_action`` outputs — are unchanged
     for every existing (non-FiLM) checkpoint.
  2. FiLM checkpoints: the deploy path (``load_action_film_head`` +
     ``_film_q_actions``) applies the shared ``apply_action_film`` transform. The
     released FiLM critics are checked end to end in
     ``test_bon_equivalence.py``.
  3. Candidate selection scores with ``torch.min(Q1, Q2)``.
"""

import pytest
import torch
import torch.nn as nn

from mulligan.networks.action_film import apply_action_film
from mulligan.real.policy.vision_idql import (
    VisionIDQLRealWorldPolicy,
    load_action_film_head,
)

STATE_DIM = 10
STEP_DIM = 3
ARM_DIMS = 2
K = 2
FLAT_ACTION = STEP_DIM * K


def _bare_policy(film_head, arm_dims=0, step_dim=0) -> VisionIDQLRealWorldPolicy:
    """Bare instance carrying only the FiLM state — enough for _film_q_actions."""
    policy = object.__new__(VisionIDQLRealWorldPolicy)
    policy._film_head = film_head
    policy._film_arm_dims = arm_dims
    policy._film_step_dim = step_dim
    return policy


# ---------------------------------------------------------------------------
# 1. Non-FiLM byte-identity (regression for every existing checkpoint)
# ---------------------------------------------------------------------------


def test_non_film_metadata_loads_no_head():
    """Flag absent (every pre-FiLM artifact) => (None, 0, 0), no ckpt key read."""
    head, arm, step = load_action_film_head({}, {}, STATE_DIM, "cpu")
    assert head is None and arm == 0 and step == 0
    head, arm, step = load_action_film_head({"action_film_head": False}, {}, STATE_DIM, "cpu")
    assert head is None


def test_non_film_q_actions_is_same_object():
    """No head => _film_q_actions returns the SAME tensor object, so the Q
    inputs — and therefore select_action outputs — are unchanged for
    non-FiLM checkpoints."""
    policy = _bare_policy(None)
    actions = torch.randn(8, FLAT_ACTION)
    states = torch.randn(8, STATE_DIM)
    assert policy._film_q_actions(actions, states) is actions


def test_film_metadata_without_weights_fails_loud():
    metadata = {
        "action_film_head": True,
        "action_film_hidden": 16,
        "action_film_arm_dims": ARM_DIMS,
        "action_film_step_dim": STEP_DIM,
    }
    with pytest.raises(ValueError, match="film_head_state_dict"):
        load_action_film_head(metadata, {}, STATE_DIM, "cpu")


# ---------------------------------------------------------------------------
# 2. FiLM deploy transform
# ---------------------------------------------------------------------------


def test_synthetic_deploy_matches_shared_transform():
    torch.manual_seed(0)
    head = nn.Sequential(nn.Linear(STATE_DIM, 16), nn.ReLU(), nn.Linear(16, 2 * ARM_DIMS))
    with torch.no_grad():
        head[-1].bias.add_(0.4)
    head.eval()
    metadata = {
        "action_film_head": True,
        "action_film_hidden": 16,
        "action_film_arm_dims": ARM_DIMS,
        "action_film_step_dim": STEP_DIM,
    }
    iql_ckpt = {"film_head_state_dict": head.state_dict()}
    loaded, arm, step = load_action_film_head(metadata, iql_ckpt, STATE_DIM, "cpu")
    assert loaded is not None and (arm, step) == (ARM_DIMS, STEP_DIM)

    policy = _bare_policy(loaded, arm, step)
    actions = torch.randn(6, FLAT_ACTION)
    states = torch.randn(6, STATE_DIM)
    with torch.no_grad():
        got = policy._film_q_actions(actions, states)
        want = apply_action_film(actions, head(states), arm_dims=ARM_DIMS, step_dim=STEP_DIM)
    assert torch.allclose(got, want, atol=1e-7)
    assert not torch.allclose(got, actions, atol=1e-4)  # real transform, not a no-op


def test_q_aggregate_is_pair_min_exactly():
    """The selection score reproduces min(Q1,Q2) EXACTLY (torch.equal)."""
    torch.manual_seed(0)
    policy = _bare_policy(None)
    q1_vals = torch.randn(32, 1)
    q2_vals = torch.randn(32, 1)
    got = policy._aggregate_q(q1_vals, q2_vals)
    want = torch.min(q1_vals, q2_vals).squeeze(-1)
    assert torch.equal(got, want)
