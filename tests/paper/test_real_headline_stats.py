"""Statistics helpers used by the real-world paper tables and figures.

``t_interval`` gives the Student-t seed intervals of the simulation bar panels;
``bootstrap_mean_ci`` gives the fixed-seed bootstrap intervals of the real-world
efficiency and stage summaries.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from mulligan.real.lifecycle import stats as S


def test_t_interval_matches_closed_form():
    x = np.array([0.0, 1.0, 3.0 / 7.0, 1.0, 2.0 / 7.0, 5.0 / 7.0])
    mean, se, lo, hi = S.t_interval(x)
    assert mean == pytest.approx(float(x.mean()))
    assert se == pytest.approx(float(x.std(ddof=1) / np.sqrt(len(x))))
    tcrit = float(sp_stats.t.ppf(0.975, df=len(x) - 1))
    assert lo == pytest.approx(mean - tcrit * se)
    assert hi == pytest.approx(mean + tcrit * se)


def test_t_interval_rejects_degenerate_samples():
    with pytest.raises(ValueError):
        S.t_interval(np.array([0.5]))
    with pytest.raises(ValueError):
        S.t_interval(np.array([]))


def test_bootstrap_mean_ci_pinned_seed_is_deterministic():
    x = np.arange(50, dtype=float) / 49.0
    first = S.bootstrap_mean_ci(x)
    second = S.bootstrap_mean_ci(x)
    assert first == second
    mean, lo, hi = first
    assert mean == pytest.approx(float(x.mean()))
    assert lo < mean < hi
    # A different seed must move the (finite-resample) bounds.
    assert S.bootstrap_mean_ci(x, seed=1)[1:] != first[1:]
