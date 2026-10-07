"""Tests for mulligan.sampling.real_design (real-robot SelectInitialStates).

Unit tests use small synthetic inputs. ``test_marker_r1_reproduces_locked_manifest`` rebuilds
the locked Marker R1 manifest from trimmed recorded inputs in ``fixtures/real_design``:
the lock-time R0 evaluation outcome table, its stage labels, the R0 manifest starts, and the
locked R1 manifest's per-row fields (source files and sha256 recorded inside the JSONs).
"""

from __future__ import annotations

import dataclasses
import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mulligan.real.lifecycle.geometry import TaskGeometry
from mulligan.real.lifecycle.tasks import INCH_TO_M, get_task_spec
from mulligan.sampling.real_design import (
    BASELINE_ARM,
    MULLIGAN_ARM,
    FillConfig,
    OrderConfig,
    PromotionConfig,
    RoundConfig,
    band_phases,
    build_round,
    coverage_fill,
    fresh_candidates,
    holder_x_risk,
    join_stage_labels,
    load_round_config,
    nut_pose_hardness,
    order_holder_pairs,
    order_peg_path,
    promote_failures,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "real_design"
MARKER_KEYS = ("pen_x", "pen_y", "pen_yaw", "holder_x", "holder_y")
SQUARE_KEYS = ("nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y")


def _promo_cfg(**kw) -> PromotionConfig:
    base = dict(
        pool="all_fail",
        cap=3,
        rank_keys=("phase", "num_steps", "stage", "tiebreak"),
        tiebreak="stream_index",
        upstream_max_stage=2,
    )
    base.update(kw)
    return PromotionConfig(**base)


def _candidates(stages, steps, tiers=None) -> pd.DataFrame:
    n = len(stages)
    return pd.DataFrame(
        {
            "stage": stages,
            "num_steps": steps,
            "tier": tiers or [0] * n,
            "stream_index": list(range(n)),
            "manifest_idx": list(range(100, 100 + n)),
        }
    )


# ------------------------------------------------------------------ promotion


def test_promotion_upstream_first_then_shortest_episode():
    cand = _candidates(stages=[4, 1, 3, 0, 2], steps=[50, 300, 40, 200, 100])
    out = promote_failures(cand, _promo_cfg(cap=4))
    # Upstream (stage <= 2) by episode length, then later-stage failures by episode length.
    assert out["stream_index"].tolist() == [4, 3, 1, 2]
    assert out["selection_rank"].tolist() == [1, 2, 3, 4]
    assert out["upstream"].tolist() == [True, True, True, False]


def test_promotion_earliest_stage_first_with_cap():
    cand = _candidates(stages=[4, 1, 3, 1, 2], steps=[50, 300, 40, 200, 100])
    cfg = _promo_cfg(rank_keys=("stage", "num_steps", "tiebreak"), cap=3)
    out = promote_failures(cand, cfg)
    assert out["stream_index"].tolist() == [3, 1, 4]


def test_promotion_tiers_override_stage_order():
    cand = _candidates(stages=[0, 5, 5, 1], steps=[10, 30, 20, 5], tiers=[2, 1, 1, 3])
    cfg = _promo_cfg(rank_keys=("tier", "num_steps", "tiebreak"), cap=10)
    assert promote_failures(cand, cfg)["stream_index"].tolist() == [2, 1, 0, 3]


def test_promotion_rejects_ambiguous_tiebreak():
    cand = _candidates(stages=[1, 1], steps=[5, 5])
    cand["stream_index"] = [7, 7]
    with pytest.raises(ValueError, match="not unique"):
        promote_failures(cand, _promo_cfg())


def _paired(n=4):
    return pd.DataFrame(
        {
            "manifest_idx": range(n),
            "sobol_stream_index": range(10, 10 + n),
            "mulligan_success": [True, False, False, True][:n],
            "mulligan_outcome": ["success", "failure", "timeout", "success"][:n],
            "mulligan_num_steps": [100, 50, 400, 120][:n],
            "mulligan_episode_index": [3, 1, 0, 2][:n],
        }
    )


def _labels(outcomes):
    return pd.DataFrame(
        {
            "episode_index": [0, 1, 2, 3],
            "policy_short": ["mulligan"] * 4,
            "original_outcome": outcomes,
            "max_stage_v2": [3, 1, 7, 7],
            "early_failure_mode_v2": ["timeout_holding", "missed_grasp", None, None],
        }
    )


def test_join_stage_labels_maps_by_episode():
    out = join_stage_labels(
        _paired(),
        _labels(["timeout", "failure", "success", "success"]),
        {"mulligan": "mulligan"},
        "max_stage_v2",
        "early_failure_mode_v2",
    )
    assert out["mulligan_stage"].tolist() == [7, 1, 3, 7]
    assert out["mulligan_failure_mode"].tolist() == ["", "missed_grasp", "timeout_holding", ""]


def test_join_stage_labels_fails_loud():
    with pytest.raises(ValueError, match="disagree"):
        join_stage_labels(
            _paired(),
            _labels(["success", "failure", "success", "success"]),
            {"mulligan": "mulligan"},
            "max_stage_v2",
            "",
        )
    with pytest.raises(ValueError, match="lack episodes"):
        join_stage_labels(
            _paired(),
            _labels(["timeout"] * 4).iloc[:2],
            {"mulligan": "mulligan"},
            "max_stage_v2",
            "",
        )


# ------------------------------------------------------------------ coverage fill


def test_holder_x_risk_is_smoothed_and_normalized():
    hx = np.array([6, 6, 7, 7, 8, 8]) * INCH_TO_M
    outcomes = pd.DataFrame({"holder_x": hx, "a": [1, 1, 1, 0, 0, 0], "b": [1, 1, 0, 1, 0, 1]})
    # Successes per depth over both columns: 6 -> 4/4, 7 -> 2/4, 8 -> 1/4.
    risk = holder_x_risk(outcomes, ["a", "b"])
    smooth = {6: 1 - 5 / 6, 7: 1 - 3 / 6, 8: 1 - 2 / 6}
    lo, hi = min(smooth.values()), max(smooth.values())
    assert risk == pytest.approx({k: (v - lo) / (hi - lo) for k, v in smooth.items()})
    assert risk[6] == 0.0 and risk[8] == 1.0


def test_nut_pose_hardness_terms():
    geom = TaskGeometry(get_task_spec("square_d2"))
    x_max = geom.bounds[0, 1]
    peg = [10.5 * INCH_TO_M, -2.5 * INCH_TO_M]
    arr = np.array([[x_max, 0.0, 0.0, *peg], [0.0, 0.0, np.pi, *peg]])
    edge = nut_pose_hardness(arr, geom, {"x_edge": 0.6, "edge_relu": 0.4}, g_weight=0.5)
    # Forward-edge nut with handle along +x: every term is 1. Centered nut: edge 0, g_radial 0.5.
    assert edge == pytest.approx([1.0, 0.25])
    yaw = nut_pose_hardness(arr, geom, {"yaw": 0.5, "y_edge": 0.3, "x_edge": 0.2}, g_weight=0.4)
    assert yaw == pytest.approx([0.6 * 0.2 + 0.4, 0.6 * 0.5 + 0.4 * 0.5])


def test_band_phases():
    cfg = FillConfig(seed=0, pool_size=8, beta=0.0, placement_weight=1.0)
    assert band_phases(cfg, 25) == [(None, 25)]
    quota = dataclasses.replace(cfg, bands={"mode": "quota", "band": 8, "fraction": 1 / 3})
    assert band_phases(quota, 33) == [(8, 11), (None, 22)]
    balanced = dataclasses.replace(cfg, bands={"mode": "balanced", "bands": [6, 7, 8]})
    assert band_phases(balanced, 20) == [(6, 7), (7, 7), (8, 6)]


def test_fresh_candidates_skip_placed_starts():
    geom = TaskGeometry(get_task_spec("marker_d2"))
    pool = geom.sobol(123, 64)
    kept, stream = fresh_candidates(geom, 123, 64, pool[[3, 10]])
    assert len(kept) == 62 and 3 not in stream and 10 not in stream
    assert np.array_equal(kept, pool[stream])


def test_coverage_fill_quota_and_farthest_point():
    geom = TaskGeometry(get_task_spec("marker_d2"))
    pool = geom.sobol(7, 256)
    support = geom.sobol(8, 16)
    cfg = FillConfig(
        seed=7,
        pool_size=256,
        beta=0.0,
        placement_weight=0.7,
        bands={"mode": "quota", "band": 8, "fraction": 0.5},
    )
    picks = coverage_fill(geom, pool, np.zeros(len(pool)), support, cfg, 6)
    assert len(set(picks)) == 6
    depth = np.rint(pool[picks, 3] / INCH_TO_M).astype(int)
    assert (depth[:3] == 8).all()
    plain = dataclasses.replace(cfg, bands=None)
    first = coverage_fill(geom, pool, np.zeros(len(pool)), support, plain, 1)[0]
    weights = geom.placement_axis_weights(0.7)
    from mulligan.real.lifecycle.geometry import pairwise_min_distance

    d = pairwise_min_distance(pool, support, geom.bounds, geom.periodic_mask, weights)
    assert first == int(np.argmax(d))


def test_coverage_fill_hardness_tilts_choice():
    geom = TaskGeometry(get_task_spec("marker_d2"))
    pool = geom.sobol(9, 128)
    support = geom.sobol(10, 8)
    cfg = FillConfig(seed=9, pool_size=128, beta=0.0, placement_weight=1.0)
    base = coverage_fill(geom, pool, np.zeros(len(pool)), support, cfg, 1)[0]
    hard = np.zeros(len(pool))
    target = (base + 1) % len(pool)
    hard[target] = 1e6
    tilted = dataclasses.replace(cfg, beta=1.0)
    assert coverage_fill(geom, pool, hard, support, tilted, 1) == [target]


# ------------------------------------------------------------------ ordering


def _marker_row(rank, hx, hy, *, verbatim=True, upstream=True):
    return {
        "pen_x": 0.01 * rank,
        "pen_y": 0.0,
        "pen_yaw": 0.0,
        "holder_x": hx * INCH_TO_M,
        "holder_y": hy * INCH_TO_M,
        "selection_rank": rank,
        "is_verbatim": verbatim,
        "upstream": upstream,
    }


def test_holder_pairs_keeps_ours_queue_and_pairs_arms():
    ours = [_marker_row(i, 6 + i % 3, -(i % 2)) for i in range(1, 7)]
    fills = [_marker_row(i, 6 + i % 2, -2, verbatim=False) for i in range(7, 11)]
    base = [_marker_row(i, 6 + i % 3, -(i % 5)) for i in range(1, 11)]
    segments = [("up", ours[:4]), ("later", ours[4:]), ("fps_fill", fills)]
    cfg = OrderConfig(strategy="holder_pairs", seed=3)
    pairs = order_holder_pairs(cfg, segments, [dict(r) for r in base], ("holder_x", "holder_y"))
    assert len(pairs) == 10
    assert all({src for src, _ in pair} == {BASELINE_ARM, MULLIGAN_ARM} for _, pair in pairs)
    assert [seg for seg, _ in pairs] == ["up"] * 4 + ["later"] * 2 + ["fps_fill"] * 4
    ours_ranks = [
        row["selection_rank"] for _, pair in pairs for src, row in pair if src == MULLIGAN_ARM
    ]
    assert sorted(ours_ranks[:4]) == [1, 2, 3, 4] and sorted(ours_ranks[6:]) == [7, 8, 9, 10]
    base_ranks = [
        row["selection_rank"] for _, pair in pairs for src, row in pair if src == BASELINE_ARM
    ]
    assert sorted(base_ranks) == list(range(1, 11))
    again = order_holder_pairs(cfg, segments, [dict(r) for r in base], ("holder_x", "holder_y"))
    assert [[r["selection_rank"] for _, r in p] for _, p in again] == [
        [r["selection_rank"] for _, r in p] for _, p in pairs
    ]
    alt = order_holder_pairs(
        dataclasses.replace(cfg, alternate_arms=True),
        segments,
        [dict(r) for r in base],
        ("holder_x", "holder_y"),
    )
    sources = [src for _, pair in alt for src, _ in pair]
    assert all(a != b for a, b in zip(sources, sources[1:]))


def _peg_row(rank, px, py, *, verbatim=True, upstream=True, failed="iql+dp"):
    return {
        "nut_x": 0.0,
        "nut_y": 0.01 * rank,
        "nut_yaw": 0.0,
        "peg_x": px * INCH_TO_M,
        "peg_y": py * INCH_TO_M,
        "selection_rank": rank,
        "is_verbatim": verbatim,
        "upstream": upstream,
        "failed_arms": failed,
    }


def test_peg_path_assigns_same_peg_baseline_and_leads_true_failures():
    verbatim = [
        _peg_row(1, 9.5, -0.5, failed="iql"),
        _peg_row(2, 10.5, -1.5, failed="dp"),
        _peg_row(3, 9.5, -0.5, failed="iql+dp", upstream=False),
    ]
    fills = [_peg_row(i, 11.5, -2.5, verbatim=False) for i in range(4, 7)]
    # Same peg multiset as the Ours rows, in a different order.
    base_pegs = [(11.5, -2.5), (9.5, -0.5), (11.5, -2.5), (10.5, -1.5), (11.5, -2.5), (9.5, -0.5)]
    base = [_peg_row(i, *peg) for i, peg in enumerate(base_pegs, start=1)]
    plain = OrderConfig(
        strategy="peg_path", seed=5, segment_names={"verbatim": "verbatim_promotions"}
    )
    pairs = order_peg_path(plain, verbatim, fills, base, ("peg_x", "peg_y"))
    for _, pair in pairs:
        (_, a), (_, b) = pair
        assert (a["peg_x"], a["peg_y"]) == (b["peg_x"], b["peg_y"])
    lead = dataclasses.replace(
        plain,
        split_by_phase=True,
        lead_if_failed="iql",
        segment_names={"lead": "lead", "demoted": "demoted"},
    )
    pairs = order_peg_path(lead, verbatim, fills, base, ("peg_x", "peg_y"))
    ours = [
        (seg, row["selection_rank"])
        for seg, pair in pairs
        for src, row in pair
        if src == MULLIGAN_ARM
    ]
    assert ours[:2] == [("lead", 1), ("lead", 3)]
    assert sorted(seg for seg, _ in ours[2:]) == ["demoted"] + ["fps_fill"] * 3


# ------------------------------------------------------------------ end to end


def _write_manifest(path, keys, arr, source):
    states = [{"source": source, **dict(zip(keys, map(float, row)))} for row in arr]
    path.write_text(json.dumps({"states": states}))


def _synthetic_round(tmp_path, task, keys, order):
    geom = TaskGeometry(get_task_spec(task))
    prior = tmp_path / "prior.json"
    _write_manifest(prior, keys, geom.sobol(11, 24), MULLIGAN_ARM)
    evals = geom.sobol(12, 10)
    outcomes = pd.DataFrame(evals, columns=list(keys))
    outcomes["manifest_idx"] = range(10)
    outcomes["sobol_stream_index"] = range(10)
    success = [True, False, False, True, False, True, False, True, True, False]
    outcomes["mulligan_success"] = success
    outcomes["mulligan_outcome"] = ["success" if s else "failure" for s in success]
    outcomes["mulligan_num_steps"] = [100 + 7 * i for i in range(10)]
    outcomes["mulligan_episode_index"] = range(10)
    outcomes.to_csv(tmp_path / "outcomes.csv", index=False)
    pd.DataFrame(
        {
            "episode_index": range(10),
            "policy_short": ["mulligan"] * 10,
            "original_outcome": outcomes["mulligan_outcome"],
            "stage": [7, 3, 0, 7, 5, 7, 1, 7, 7, 2],
        }
    ).to_csv(tmp_path / "labels.csv", index=False)
    evals_manifest = tmp_path / "eval.json"
    _write_manifest(evals_manifest, keys, evals, "eval")
    cfg = RoundConfig(
        task=task,
        round=1,
        budget=8,
        source_eval_seed=12,
        baseline_seed=13,
        promotion=_promo_cfg(arms={"mulligan": "mulligan"}, stage_column="stage", cap=3),
        fill=FillConfig(seed=14, pool_size=512, beta=0.0, placement_weight=0.6),
        order=order,
        inputs={},
    )
    inputs = {
        "eval_outcomes": tmp_path / "outcomes.csv",
        "stage_labels": tmp_path / "labels.csv",
        "support_manifests": [prior],
        "avoid_manifests": [prior, evals_manifest],
    }
    return cfg, inputs, evals


@pytest.mark.parametrize(
    ("task", "keys", "order"),
    [
        (
            "marker_d2",
            MARKER_KEYS,
            OrderConfig(
                strategy="holder_pairs", seed=1, segment_names={"upstream": "up", "later": "later"}
            ),
        ),
        (
            "square_d2",
            SQUARE_KEYS,
            OrderConfig(
                strategy="peg_path", seed=1, segment_names={"verbatim": "verbatim_promotions"}
            ),
        ),
    ],
)
def test_build_round_synthetic(tmp_path, task, keys, order):
    cfg, inputs, evals = _synthetic_round(tmp_path, task, keys, order)
    design, manifest = build_round(cfg, inputs)
    # Failures at stages 0, 1, 2 are upstream; the cap keeps the three of them.
    assert design.promotions["stream_index"].tolist() == [2, 6, 9]
    assert len(design.fills) == 5
    states = manifest["states"]
    assert [row["manifest_idx"] for row in states] == list(range(16))
    assert manifest["arm_counts"] == {BASELINE_ARM: 8, MULLIGAN_ARM: 8}
    ours = [row for row in states if row["source"] == MULLIGAN_ARM]
    assert [row["mulligan_queue_rank"] for row in ours] == list(range(1, 9))
    promoted = [row for row in ours if row["is_verbatim_failure"]]
    for row in promoted:
        assert [row[k] for k in keys] == pytest.approx(
            evals[row["source_eval_stream_index"]], abs=1e-12
        )
    assert all(row["is_verbatim_failure"] for row in ours[:3])
    _, again = build_round(cfg, inputs)
    assert again == manifest


@pytest.mark.parametrize("task", ["marker_d2", "square_d2"])
@pytest.mark.parametrize("rnd", [1, 2, 3, 4, 5])
def test_sampler_configs_load(task, rnd):
    cfg = load_round_config(REPO / f"configs/real/{task}/r{rnd:02d}_sampler.yaml")
    assert (cfg.task, cfg.round, cfg.budget) == (task, rnd, 50)
    assert cfg.order.strategy == {"marker_d2": "holder_pairs", "square_d2": "peg_path"}[task]
    assert set(cfg.inputs) >= {"support_manifests", "avoid_manifests"}


def test_marker_r1_reproduces_locked_manifest():
    cfg = load_round_config(REPO / "configs/real/marker_d2/r01_sampler.yaml")
    r0 = FIXTURES / "marker_d2_r0_manifest_states.json"
    inputs = {
        "eval_outcomes": FIXTURES / "marker_d2_r0_paired_round_outcomes_lock.csv",
        "stage_labels": FIXTURES / "marker_d2_r0_stage_labels.csv",
        "support_manifests": [r0],
        "avoid_manifests": [r0],
    }
    _, manifest = build_round(cfg, inputs)
    locked = json.loads((FIXTURES / "marker_d2_r1_locked_states.json").read_text())["states"]
    rebuilt = manifest["states"]
    assert len(rebuilt) == len(locked) == 100
    for want, got in zip(locked, rebuilt, strict=True):
        for key, value in want.items():
            if key in MARKER_KEYS:
                continue
            assert got[key] == value, (want["manifest_idx"], key)
    # The locked values went through a pandas CSV write/read; compare after the same round trip.
    frame = pd.DataFrame([{k: row[k] for k in MARKER_KEYS} for row in rebuilt])
    back = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    assert back.to_numpy().tolist() == [[row[k] for k in MARKER_KEYS] for row in locked]


def test_marker_r1_rng_isolated():
    # The builder owns its RNGs: global random state must not leak into the design.
    cfg = load_round_config(REPO / "configs/real/marker_d2/r01_sampler.yaml")
    small = dataclasses.replace(cfg, fill=dataclasses.replace(cfg.fill, pool_size=1024))
    r0 = FIXTURES / "marker_d2_r0_manifest_states.json"
    inputs = {
        "eval_outcomes": FIXTURES / "marker_d2_r0_paired_round_outcomes_lock.csv",
        "stage_labels": FIXTURES / "marker_d2_r0_stage_labels.csv",
        "support_manifests": [r0],
        "avoid_manifests": [r0],
    }
    random.seed(0)
    np.random.seed(0)
    _, first = build_round(small, inputs)
    random.seed(1)
    np.random.seed(1)
    _, second = build_round(small, inputs)
    assert first == second


# ------------------------------------------------------------------ the README command


def _readme_step7_argv() -> list[str]:
    """The `real_design` command of README step 7 as an argv list (after the module name)."""
    import re
    import shlex

    text = (REPO / "README.md").read_text()
    block = text[text.index("7. **Start design (Alg. 2)**") :]
    command = re.search(r"```bash\n(.*?)```", block, flags=re.S).group(1)
    words = shlex.split(command.replace("\\\n", " "))
    assert words[:3] == ["python", "-m", "mulligan.sampling.real_design"], words[:3]
    return words[3:]


def test_readme_step7_command_runs(tmp_path, monkeypatch):
    """The documented start-design command runs from the repository root and reproduces R1.

    The shipped manifests resolve through the repo-relative `inputs` of the sampler config; the
    two unreleased inputs are the placeholders the README tells the reader to replace.
    """
    from mulligan.sampling.real_design import main

    out = tmp_path / "manifest.json"
    values = {
        "eval_outcomes": str(FIXTURES / "marker_d2_r0_paired_round_outcomes_lock.csv"),
        "stage_labels": str(FIXTURES / "marker_d2_r0_stage_labels.csv"),
    }
    argv = _readme_step7_argv()
    for i, arg in enumerate(argv):
        if arg == "--input":
            name = argv[i + 1].partition("=")[0]
            argv[i + 1] = f"{name}={values.pop(name)}"
        elif arg == "--out":
            argv[i + 1] = str(out)
    assert not values, f"README step 7 does not pass {sorted(values)}"
    monkeypatch.chdir(REPO)
    assert main(argv) == 0
    rebuilt = json.loads(out.read_text())["states"]
    locked = json.loads((FIXTURES / "marker_d2_r1_locked_states.json").read_text())["states"]
    assert len(rebuilt) == len(locked) == 100
    assert [row["manifest_idx"] for row in rebuilt] == [row["manifest_idx"] for row in locked]
    assert [row["source"] for row in rebuilt] == [row["source"] for row in locked]


def test_input_flag_replaces_a_list_input():
    from mulligan.sampling.real_design import parse_input_overrides, resolve_inputs

    cfg = load_round_config(REPO / "configs/real/marker_d2/r02_sampler.yaml")
    overrides = parse_input_overrides(
        ["support_manifests=a.json", "support_manifests=b.json", "eval_outcomes=o.csv"], cfg
    )
    assert overrides == {"support_manifests": ["a.json", "b.json"], "eval_outcomes": "o.csv"}
    inputs = resolve_inputs(cfg, REPO, overrides)
    assert inputs["support_manifests"] == [Path("a.json"), Path("b.json")]
    assert inputs["eval_outcomes"] == Path("o.csv")
    assert all(p.is_absolute() for p in inputs["avoid_manifests"])
    with pytest.raises(ValueError, match="takes one path"):
        parse_input_overrides(["eval_outcomes=a.csv", "eval_outcomes=b.csv"], cfg)
    with pytest.raises(ValueError, match="unknown input"):
        resolve_inputs(cfg, REPO, {"nope": "x"})
