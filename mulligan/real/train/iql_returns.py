"""Shared return-target helpers for real-world IQL evaluation and analysis."""

from __future__ import annotations

import numpy as np


def discounted_return_target(
    steps_to_go: np.ndarray, success: np.ndarray, *, gamma: float, n_action_steps: int
) -> np.ndarray:
    """Per-frame binary-success return target: ``success * gamma**steps_to_go``.

    ``steps_to_go`` counts recorded environment frames, and IQL's ``gamma`` is
    also per environment frame. The action chunk width therefore does not scale
    the exponent; it is retained only to validate the associated policy config.
    """
    if n_action_steps < 1:
        raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
    steps_to_go = np.asarray(steps_to_go, dtype=float)
    success = np.asarray(success, dtype=float)
    return success * np.power(gamma, steps_to_go)
