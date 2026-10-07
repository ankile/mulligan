"""Characterization + regression tests for the shared ``set_seed`` helper.

The DP and critic trainers share one ``set_seed``. This test pins that both reference
the shared ``mulligan.utils.seeding.set_seed`` and produce identical RNG streams.
"""

import ast
import random
from pathlib import Path

import numpy as np
import torch

from mulligan.real.train import critic as critic_trainer
from mulligan.real.train import policy as policy_trainer

ROOT = Path(__file__).resolve().parents[2]
POLICY_SRC = ROOT / "mulligan" / "real" / "train" / "policy.py"
IQL_SRC = ROOT / "mulligan" / "real" / "train" / "critic.py"


def _draw_triplet(set_seed_fn, seed: int = 123):
    set_seed_fn(seed)
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(4).tolist(),
    )


def test_set_seed_behaviour_identical():
    """Both trainers' set_seed must produce identical RNG streams."""
    triplet_policy = _draw_triplet(policy_trainer.set_seed)
    triplet_iql = _draw_triplet(critic_trainer.set_seed)
    assert triplet_policy == triplet_iql


def _find_set_seed_def(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "set_seed":
            return node
    return None


def test_set_seed_is_shared_object():
    """After extraction, both trainers reference the same function object."""
    from mulligan.utils import seeding

    assert policy_trainer.set_seed is seeding.set_seed
    assert critic_trainer.set_seed is seeding.set_seed


def test_no_local_set_seed_defs():
    """Neither trainer should still carry a local ``def set_seed`` after the move."""
    assert _find_set_seed_def(POLICY_SRC) is None
    assert _find_set_seed_def(IQL_SRC) is None
    policy_text = POLICY_SRC.read_text()
    iql_text = IQL_SRC.read_text()
    assert "from mulligan.utils.seeding import set_seed" in policy_text
    assert "from mulligan.utils.seeding import set_seed" in iql_text
