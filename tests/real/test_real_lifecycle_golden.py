"""Golden outputs of the real lifecycle geometry and statistics.

The fixture ``data/golden/real_lifecycle.json`` was recorded once on the inputs
below; the release code must reproduce it exactly (JSON floats round-trip, so the
comparison is bitwise).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import mulligan.real.lifecycle.stats as st
from mulligan.real.lifecycle.geometry import TaskGeometry
from mulligan.real.lifecycle.tasks import get_task_spec

GOLDEN = json.loads((Path(__file__).parent / "data/golden/real_lifecycle.json").read_text())
TASKS = ("marker_d2", "square_d2", "routing_d2")


def _arr(x) -> list:
    return np.asarray(x, dtype=np.float64).tolist()


def _lifecycle_record(name: str) -> dict:
    spec = get_task_spec(name)
    g = TaskGeometry(spec)
    sob = g.sobol(seed=7, n=24)
    uni = g.uniform(seed=3, n=12)
    feas, feas_idx = g.sobol_feasible(seed=11, n=16, start=4)
    cand = g.sobol(seed=5, n=64)
    hard = np.linspace(0.0, 1.0, len(cand))
    return {
        "manifest_keys": list(spec.manifest_keys),
        "sampling_keys": list(spec.sampling_keys),
        "sampling_bounds": _arr(spec.sampling_bounds),
        "sampling_grid_dims": list(spec.sampling_grid_dims),
        "sobol": _arr(sob),
        "uniform": _arr(uni),
        "sobol_feasible": _arr(feas),
        "sobol_feasible_idx": [int(i) for i in feas_idx],
        "cell_ids": [int(i) for i in g.cell_ids(sob)],
        "cell_entropy": float(g.cell_entropy(sob)),
        "fill_q95": float(g.fill_quantile(cand, sob)),
        "pairwise_min": _arr(g.pairwise_min_distance(cand[:10], uni)),
        "edge_norm": _arr(g.edge_norm(sob[:6])),
        "wfps": [int(i) for i in g.weighted_fps(cand, hard, uni, beta=0.5, budget=10)],
        "wfps_axis": [
            int(i)
            for i in g.weighted_fps(
                cand, hard, uni, beta=1.0, budget=8, axis_weights=g.placement_axis_weights(0.25)
            )
        ],
    }


def _stats_record() -> dict:
    rng = np.random.default_rng(1234)
    a = (rng.random(40) < 0.6).astype(float)
    b = (rng.random(40) < 0.4).astype(float)
    x = rng.normal(0.5, 0.2, size=30)
    d = rng.integers(-2, 3, size=32).astype(float)
    a2 = (rng.random(25) < 0.7).astype(float)
    b2 = (rng.random(25) < 0.3).astype(float)
    return {
        "wilson": [
            list(st.wilson_ci(s, n)) for s, n in [(0, 10), (7, 16), (16, 16), (21, 32), (850, 1000)]
        ],
        "wilson_1se": [list(st.wilson_1se(s, n)) for s, n in [(3, 16), (20, 32)]],
        "mcnemar": [st.exact_mcnemar_pvalue(i, j) for i, j in [(0, 0), (5, 1), (12, 3), (4, 4)]],
        "t_interval": list(st.t_interval(x)),
        "bootstrap_mean_ci": list(st.bootstrap_mean_ci(x)),
        "bootstrap_paired_delta": list(st.bootstrap_paired_delta(a, b, seed=3)),
        "stratified": list(st.bootstrap_paired_delta_stratified([(a, b), (a2, b2)], seed=3)),
        "signflip": st.signflip_permutation_pvalue(d, seed=5),
    }


@pytest.mark.parametrize("name", TASKS)
def test_task_geometry_matches_golden(name):
    got = json.loads(json.dumps(_lifecycle_record(name)))
    want = GOLDEN["lifecycle"][name]
    assert set(got) == set(want)
    for key in want:
        assert got[key] == want[key], f"{name}.{key} drifted from the golden output"


def test_stats_match_golden():
    got = json.loads(json.dumps(_stats_record()))
    want = GOLDEN["stats"]
    assert set(got) == set(want)
    for key in want:
        assert got[key] == want[key], f"stats.{key} drifted from the golden output"
