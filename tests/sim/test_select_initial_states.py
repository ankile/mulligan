from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from mulligan.sampling import hardness as H
from mulligan.sampling import sim_design
from mulligan.sampling.select_initial_states import (
    FILL,
    PERTURBED,
    PROMOTED,
    HardnessTerm,
    LocalPerturbation,
    StartSpace,
    check_disjoint,
    coverage_guardrail,
    farthest_point_fill,
    min_pairwise_distance,
    nearest_state_distance,
    promote_failures,
    sample_uniform,
    select_initial_states,
)

REPO = Path(__file__).resolve().parents[2]
CONFIGS = sorted((REPO / "configs/sim").glob("*/r*_sampler.yaml"))

# Unit square in (x, yaw) with yaw periodic over [-pi, pi).
SPACE = StartSpace(keys=("x", "yaw"), low=(0.0, -math.pi), high=(1.0, math.pi), periodic=("yaw",))


def _records(states, success, length):
    return [
        {"x": float(s[0]), "yaw": float(s[1]), "success": int(ok), "length": float(n)}
        for s, ok, n in zip(states, success, length, strict=True)
    ]


# --------------------------------------------------------------- geometry


def test_unit_coordinates_and_yaw_periodicity():
    arr = np.array([[0.25, -math.pi], [0.75, 0.0], [0.5, 3 * math.pi]])
    unit = SPACE.to_unit(arr)
    np.testing.assert_allclose(unit[:, 0], [0.25, 0.75, 0.5])
    np.testing.assert_allclose(unit[:, 1], [0.0, 0.5, 0.0], atol=1e-12)
    wrapped = SPACE.wrap(np.array([[0.5, math.pi + 0.1]]))
    np.testing.assert_allclose(wrapped[0, 1], -math.pi + 0.1)


def test_distance_wraps_yaw():
    near_seam = np.array([[0.5, math.pi - 0.01]])
    other_side = np.array([[0.5, -math.pi + 0.01]])
    middle = np.array([[0.5, 0.0]])
    across, _ = nearest_state_distance(SPACE, other_side, near_seam)
    assert across[0] == pytest.approx(0.02)
    far, _ = nearest_state_distance(SPACE, middle, near_seam)
    assert far[0] == pytest.approx(math.pi - 0.01)
    assert min_pairwise_distance(SPACE, np.vstack([near_seam, other_side, middle])) == (
        pytest.approx(0.02)
    )


def test_fill_treats_yaw_as_periodic():
    seeds = np.array([[0.5, math.pi - 0.05]])
    # The second candidate is 0.1 rad from the seed across the seam; the first is at yaw 0.
    candidates = np.array([[0.5, 0.0], [0.5, -math.pi + 0.05]])
    idx, _ = farthest_point_fill(SPACE, candidates, seeds, 1)
    assert idx.tolist() == [0]


def test_sample_uniform_is_a_prefix_stream_and_respects_validity():
    valid_space = StartSpace(
        keys=SPACE.keys,
        low=SPACE.low,
        high=SPACE.high,
        periodic=SPACE.periodic,
        valid=lambda a: a[:, 0] > 0.5,
    )
    many = sample_uniform(valid_space, np.random.default_rng(3), 50)
    few = sample_uniform(valid_space, np.random.default_rng(3), 10)
    assert np.all(many[:, 0] > 0.5)
    np.testing.assert_array_equal(few, many[:10])


# ------------------------------------------------------------------- FPS


def test_plain_farthest_point_fill_without_hardness():
    candidates = np.column_stack([np.linspace(0.0, 1.0, 11), np.zeros(11)])
    seeds = np.array([[0.0, 0.0]])
    idx, objectives = farthest_point_fill(SPACE, candidates, seeds, 3)
    assert candidates[idx[:2], 0].tolist() == [1.0, 0.5]
    assert objectives.tolist() == pytest.approx([1.0, 0.5, 0.2])


def test_hardness_tilts_the_fill_by_one_plus_beta_h():
    seeds = np.array([[0.5, 0.0]])
    candidates = np.array([[0.1, 0.0], [0.9, 0.0], [0.8, 0.0]])
    hard_right = HardnessTerm(beta=1.0, signal=lambda s: (s[:, 0] > 0.7).astype(float))
    factor = np.array([1.0, 2.0, 2.0])
    idx, objectives = farthest_point_fill(SPACE, candidates, seeds, 1, factor=factor)
    assert idx.tolist() == [1]
    assert objectives[0] == pytest.approx(0.4 * 2.0)
    records = _records([[0.3, 1.0]], [1], [10])
    design = select_initial_states(
        space=SPACE,
        budget=1,
        records=records,
        prior=seeds,
        candidates=candidates,
        hardness=[hard_right],
    )
    np.testing.assert_array_equal(design.states, candidates[[1]])


def test_squared_distance_changes_the_tradeoff():
    seeds = np.array([[0.0, 0.0]])
    candidates = np.array([[1.0, 0.0], [0.6, 0.0]])
    factor = np.array([1.0, 1.8])
    plain, _ = farthest_point_fill(SPACE, candidates, seeds, 1, factor=factor)
    squared, _ = farthest_point_fill(SPACE, candidates, seeds, 1, factor=factor, squared=True)
    assert plain.tolist() == [1]  # 0.6 * 1.8 = 1.08 > 1.0
    assert squared.tolist() == [0]  # 0.36 * 1.8 = 0.648 < 1.0


def test_non_periodic_prior_distance_ignores_the_seam():
    seeds = np.array([[0.5, math.pi - 0.05]])
    candidates = np.array([[0.5, 0.0], [0.5, -math.pi + 0.05]])
    idx, _ = farthest_point_fill(SPACE, candidates, seeds, 1, seed_distance_periodic=False)
    assert idx.tolist() == [1]


# ------------------------------------------------------ promotion, perturbation


def test_failures_are_promoted_verbatim_in_record_order():
    states = [[0.11, 0.2], [0.33, -1.0], [0.55, 2.5], [0.77, 0.0]]
    records = _records(states, [1, 0, 0, 1], [50, 400, 400, 80])
    np.testing.assert_array_equal(promote_failures(SPACE, records), np.array(states)[[1, 2]])
    np.testing.assert_array_equal(promote_failures(SPACE, records, cap=1), np.array(states)[[1]])


def test_design_order_counts_and_budget():
    rng = np.random.default_rng(0)
    states = sample_uniform(SPACE, rng, 40)
    records = _records(states, [0] * 5 + [1] * 35, rng.integers(50, 400, size=40))
    perturbation = LocalPerturbation(n=4, sigma={"yaw": 0.05}, seed=1, resample=("x",))
    design = select_initial_states(
        space=SPACE,
        budget=20,
        records=records,
        prior=sample_uniform(SPACE, rng, 10),
        candidates=sample_uniform(SPACE, rng, 500),
        perturbation=perturbation,
    )
    assert design.components == [PROMOTED] * 5 + [PERTURBED] * 4 + [FILL] * 11
    np.testing.assert_array_equal(design.states[:5], states[:5])
    with pytest.raises(ValueError, match="exceed budget"):
        select_initial_states(
            space=SPACE,
            budget=8,
            records=records,
            prior=np.empty((0, 2)),
            candidates=sample_uniform(SPACE, rng, 50),
            perturbation=perturbation,
        )


def test_perturbations_jitter_the_longest_episodes():
    states = np.array([[0.1, 0.0], [0.2, math.pi - 0.01], [0.3, 1.0], [0.4, -1.0]])
    records = _records(states, [1, 1, 1, 1], [100, 390, 300, 390])
    perturbation = LocalPerturbation(n=2, sigma={"x": 1e-3, "yaw": 0.05}, seed=7)
    anchors = perturbation.anchors(records)
    assert [a["x"] for a in anchors] == [0.2, 0.4]  # stable order among the two longest
    out = perturbation.draw(SPACE, records)
    np.testing.assert_array_equal(out, perturbation.draw(SPACE, records))
    assert np.all(np.abs(out[:, 0] - [0.2, 0.4]) < 0.01)
    assert np.all((out[:, 1] >= -math.pi) & (out[:, 1] < math.pi))
    close, _ = nearest_state_distance(SPACE, out[:1], states[1:2])
    assert close[0] < 0.3

    clipped = LocalPerturbation(n=1, sigma={"x": 10.0, "yaw": 0.0}, seed=0)
    edge = clipped.draw(SPACE, records)
    assert 0.0 <= edge[0, 0] <= 1.0
    with pytest.raises(ValueError, match="no sigma or resample"):
        LocalPerturbation(n=1, sigma={"x": 0.1}, seed=0).draw(SPACE, records)


# ----------------------------------------------------------- disjointness


def test_fill_never_selects_an_excluded_start():
    seeds = np.array([[0.0, 0.0]])
    candidates = np.array([[1.0, 0.0], [0.7, 0.0]])
    records = _records([[0.3, 2.0]], [1], [10])
    design = select_initial_states(
        space=SPACE,
        budget=1,
        records=records,
        prior=seeds,
        candidates=candidates,
        exclude=np.array([[1.0, 2 * math.pi]]),  # same start as candidate 0 after the wrap
    )
    np.testing.assert_array_equal(design.states, candidates[[1]])


def test_check_disjoint_reports_and_raises():
    states = np.array([[0.2, 0.0], [0.4, 1.0]])
    report = check_disjoint(SPACE, states, {"grid": np.array([[0.2, 0.5]])})
    assert report == [{"reference": "grid", "n_states": 1, "min_distance": pytest.approx(0.5)}]
    with pytest.raises(ValueError, match="collides with prior"):
        check_disjoint(SPACE, states, {"prior": np.array([[0.4, 1.0 + 1e-8]])})


# -------------------------------------------------------------- guardrail


def test_coverage_guardrail():
    rng = np.random.default_rng(1)
    test_points = sample_uniform(SPACE, rng, 4000)
    spread = sample_uniform(SPACE, rng, 60)
    clustered = np.column_stack([rng.uniform(0.0, 0.1, 60), rng.uniform(-0.3, 0.3, 60)])
    ok = coverage_guardrail(
        SPACE, design=spread, reference=spread, prior=np.empty((0, 2)), test_points=test_points
    )
    assert ok["passed"] and ok["design_p95"] == ok["reference_p95"]
    bad = coverage_guardrail(
        SPACE, design=clustered, reference=spread, prior=np.empty((0, 2)), test_points=test_points
    )
    assert not bad["passed"]
    with pytest.raises(ValueError, match="reference block has 60 states"):
        coverage_guardrail(
            SPACE, design=spread[:5], reference=spread, prior=spread, test_points=test_points
        )


# ------------------------------------------------------ eval-grid term gate


def _grid_table() -> H.GridCellTable:
    axes = [H.GridAxis("x", 0.0, 1.0, 2), H.GridAxis("yaw", -math.pi, math.pi, 3)]
    success = np.array([[90.0], [10.0], [50.0], [100.0], [0.0], [70.0]])
    return H.GridCellTable(space=SPACE, axes=tuple(axes), success=success)


def test_eval_grid_term_is_off_by_default():
    table = _grid_table()
    term = HardnessTerm(beta=1.0, signal=table.failure_rate(), uses_eval_grid=True, name="grid")
    kwargs = dict(
        space=SPACE,
        budget=2,
        records=_records([[0.3, 2.0]], [1], [10]),
        prior=np.array([[0.5, 0.0]]),
        candidates=sample_uniform(SPACE, np.random.default_rng(0), 100),
        hardness=[term],
    )
    with pytest.raises(ValueError, match="allow_eval_grid_feedback"):
        select_initial_states(**kwargs)
    design = select_initial_states(**kwargs, allow_eval_grid_feedback=True)
    assert len(design.states) == 2


def test_config_signals_flag_eval_grid_use():
    inputs = sim_design.RoundInputs(root=REPO, task="square_narrow", space=sim_design.SQUARE_NARROW)
    shape = {"narrow_shape": {"yaw_alpha": 4.0, "weights": {"y_norm": 1.0}}}
    grid = {"grid_sextile_bonus": {"table": "configs/sim/square_narrow/r03_grid_cell_sr.csv"}}
    assert sim_design.build_signal(inputs, shape)[1] is False
    assert sim_design.build_signal(inputs, grid)[1] is True
    blended = {"blend": [{"weight": 0.5, "signal": shape}, {"weight": 0.5, "signal": grid}]}
    assert sim_design.build_signal(inputs, blended)[1] is True


def _uses_grid(spec) -> bool:
    kind, args = next(iter(spec.items()))
    if kind == "blend":
        return any(_uses_grid(part["signal"]) for part in args)
    return kind.startswith("grid_")


def test_round_configs_use_the_eval_grid_term_exactly_where_the_paper_says():
    with_grid = set()
    for path in CONFIGS:
        config = yaml.safe_load(path.read_text())
        for arm in config["arms"].values():
            spec = arm["starts"].get("select_initial_states")
            if spec is None:
                continue
            uses = any(_uses_grid(term["signal"]) for term in spec.get("hardness", []))
            assert uses == bool(spec.get("eval_grid_feedback", False)), path
            if uses:
                with_grid.add((config["task"], config["round"]))
    assert with_grid == {("square_narrow", 3), ("square_broad", 2), ("square_broad", 3)}


# --------------------------------------------------------- hardness signals


def test_grid_cells_and_sextile_bonus():
    table = _grid_table()
    states = np.array([[0.25, -math.pi], [0.25, 0.1], [0.75, math.pi - 1e-9], [0.75, math.pi]])
    assert table.cell_index(states).tolist() == [0, 1, 5, 3]
    sextile = table.sextile_bonus()(states)
    # Six cells, one per sextile: cell 4 (0%) is hardest (bonus 1), cell 3 (100%) easiest.
    np.testing.assert_allclose(sextile, [0.2, 0.8, 0.4, 0.0])
    np.testing.assert_allclose(table.failure_rate()(states), [0.1, 0.9, 0.3, 0.0])
    with pytest.raises(ValueError, match="outside the grid range"):
        table.cell_index(np.array([[1.5, 0.0]]))


def test_grid_table_csv(tmp_path):
    path = tmp_path / "cells.csv"
    path.write_text("cell_idx,sr_seed1,sr_seed2\n0,10,30\n1,50,50\n")
    table = H.GridCellTable.from_csv(SPACE, [H.GridAxis("x", 0.0, 1.0, 2)], path)
    np.testing.assert_allclose(table.mean_success, [20.0, 50.0])
    path.write_text("cell_idx,sr_seed1\n0,10\n")
    with pytest.raises(ValueError, match="1 rows for a 2-cell grid"):
        H.GridCellTable.from_csv(SPACE, [H.GridAxis("x", 0.0, 1.0, 2)], path)


def test_minmax_and_blend():
    x = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    signal = H.minmax(H.blend([(2.0, lambda s: s[:, 0]), (1.0, lambda s: np.ones(len(s)))]))
    np.testing.assert_allclose(signal(x), [0.0, 0.5, 1.0])


def test_narrow_shape_terms():
    space = sim_design.SQUARE_NARROW
    states = np.array([[-0.112, 0.110, math.pi / 2], [-0.112, 0.225, 0.0]])
    weights = {"y_norm": 0.5, "y_edge": 0.25, "y_center2": 1.0, "yaw_sin_alpha": 2.0}
    got = H.narrow_shape(space, weights, yaw_alpha=4.0)(states)
    np.testing.assert_allclose(got, [0.0 + 0.25 + 0.25 + 2.0, 0.5 + 0.25 + 0.25 + 0.0])
    with pytest.raises(ValueError, match="unknown shape terms"):
        H.narrow_shape(space, {"x": 1.0}, yaw_alpha=1.0)


def test_length_model_tracks_length():
    space = sim_design.SQUARE_BROAD
    states = sample_uniform(space, np.random.default_rng(2), 80)
    features = H.broad_features(space, states, ["yaw0"])[:, 0]
    records = [
        {**dict(zip(space.keys, row, strict=True)), "success": 1, "length": 100 + 250 * f}
        for row, f in zip(states, features, strict=True)
    ]
    model = H.fit_length_model(space, records, max_steps=400)
    pred = model(states)
    assert np.corrcoef(pred, features)[0, 1] > 0.9
    assert np.all((pred >= 0) & (pred <= 1))


# ---------------------------------------------------------------- manifest


def test_merge_collapses_shared_starts_and_shuffle_is_seeded():
    a = sim_design.ArmResult("a", "p", np.array([[0.1, 0.0], [0.2, 0.0]]), ["x", "x"])
    b = sim_design.ArmResult("b", "p", np.array([[0.2, 1e-5], [0.3, 0.0]]), ["x", "x"])
    rows = sim_design.merge_arms(SPACE, [a, b], match_tolerance=1e-3)
    assert [r["sources"] for r in rows] == [["a"], ["a", "b"], ["b"]]
    assert rows[1]["source_indices"] == {"a": 1, "b": 0}
    shuffled = sim_design.shuffle_rows(rows, seed=5)
    assert [r["manifest_index"] for r in shuffled] == [0, 1, 2]
    assert shuffled == sim_design.shuffle_rows(rows, seed=5)
    assert {r["source"] for r in shuffled} == {"a", "b", "shared"}
    c = sim_design.ArmResult("c", "q", np.array([[0.1, 0.0]]), ["x"])
    with pytest.raises(ValueError, match="two policies"):
        sim_design.merge_arms(SPACE, [a, c], match_tolerance=1e-3)


# ------------------------------------------------------------------ golden


@pytest.mark.parametrize("config_path", CONFIGS, ids=[f"{p.parent.name}-{p.stem}" for p in CONFIGS])
def test_round_reproduces_the_locked_starts(config_path, tmp_path):
    config = sim_design.load_config(config_path)
    build = sim_design.build_round(config, root=REPO)
    result = sim_design.compare_locked(build, config, REPO)
    assert all(result["sha256_ok"].values()), result["sha256_ok"]
    for name, arm in result["arms"].items():
        assert arm["exact"], (name, arm)
    assert result["manifest"]["rows_identical"], result["manifest"]
    assert result["reproduced"]
    assert build.guardrail is not None and build.guardrail["passed"]
    # The written manifest carries what the collector, the protocol quota (manifest hash)
    # and the splitters read, equal to the locked manifest.
    sim_design.write_outputs(build, tmp_path, report={})
    rebuilt = json.loads((tmp_path / "manifest.json").read_text())
    locked = json.loads((REPO / config["locked"]["manifest"]["path"]).read_text())
    fields = ("task", "keys", "match_tolerance", "states")
    assert {k: rebuilt[k] for k in fields} == {k: locked[k] for k in fields}
