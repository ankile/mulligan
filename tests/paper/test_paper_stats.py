"""The paper's quoted statistics, recomputed by paper/stats from shipped data or pinned evidence."""

from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from paper.appendix.paper_side import (
    divl_comparison,
    reranking_check,
    training_compute,
)
from paper.stats import (
    count_eval_episodes,
    final_round,
    print_collection_totals,
    r0_demo_durations,
    rlpd_ablations,
    routing_d2_verify,
    sim_contrasts,
    sim_welch_table,
)
from tests.paper.evidence import reads_evidence

HIL_IDQL, HG_DAGGER = "HiL-IDQL+Mulligan", "HG-DAgger"

# sha256 of the frozen manuscript's generated tables (paper/reference/tables/<name>)
MANUSCRIPT_TABLES = {
    sim_welch_table: "70a0cfba3d790b4f542ff0d9d9818849230d72ea46e615e0a2dbb65ed010be40",
    divl_comparison: "2a661ab50249b7fbca96abda81199b639617d461500df1e469e2068a1f0c6fa9",
    training_compute: "0eeb43ad173af50f5b1298fc99bdae4ee7907c8648a649896329295e92039b2d",
}


@pytest.mark.parametrize("module", MANUSCRIPT_TABLES, ids=lambda m: m.NAME)
def test_table_matches_manuscript(module, tmp_path):
    out = module.build(tmp_path)
    assert out == tmp_path / module.NAME
    assert hashlib.sha256(out.read_bytes()).hexdigest() == MANUSCRIPT_TABLES[module]


def test_final_round_margins_and_counts():
    result = final_round.final_round()
    margins = {task: r["margin_pp"] for task, r in result.items()}
    assert margins == pytest.approx({"marker_d2": 34, "square_d2": 10, "routing_d2": 16})
    counts = {task: r["counts"] for task, r in result.items()}
    assert counts["marker_d2"][HIL_IDQL] == (38, 50) and counts["marker_d2"][HG_DAGGER] == (21, 50)
    assert counts["square_d2"][HIL_IDQL] == (36, 50) and counts["square_d2"][HG_DAGGER] == (31, 50)
    assert counts["routing_d2"][HIL_IDQL] == (17, 50) and counts["routing_d2"][HG_DAGGER] == (9, 50)
    # The plotted Cable curve is clip success: 37% vs. 50% at R5.
    clip = final_round.cable_clip_success()
    assert (clip[HG_DAGGER], clip[HIL_IDQL]) == pytest.approx((37, 50))


def test_cable_endpoint_contrast():
    r9i_vs_r9b = routing_d2_verify.verify()[0]
    assert (r9i_vs_r9b["a"], r9i_vs_r9b["b"]) == ("r9i", "r9b")
    assert (r9i_vs_r9b["a_successes"], r9i_vs_r9b["b_successes"]) == (17, 9)
    assert round(r9i_vs_r9b["p"], 3) == 0.077


def test_actor_data_ablation():
    rows = {r["cell_key"]: r for r in sim_contrasts.sampler_ablation()}
    auto = rows["square_narrow_r1_baseline_uniform_no_cf_straddled_auto_success"]
    assert round(auto["delta"], 2) == -3.36
    assert (round(auto["lo"], 1), round(auto["hi"], 1)) == (-6.5, -0.2)


def test_sim_cells_map_to_recipes():
    ids = sim_contrasts.recipe_ids()
    assert set(sim_contrasts.divl_cells()) <= set(ids)
    cells = {cell for pair in sim_welch_table.N32_CELLS.values() for cell in pair}
    assert cells <= set(ids)
    assert sim_contrasts.recipe_id("square_narrow_r1_mulligan_with_cf_human_only") == (
        "square-narrow-r01-mulligan"
    )
    with pytest.raises(KeyError, match="source_cell_key"):
        sim_contrasts.recipe_id("square_narrow_r9_unknown")


def test_cable_arm_labels():
    labels = [routing_d2_verify.arm_label(name) for name in ("r9i", "r9o", "r9b", "r0b")]
    assert labels == [
        "R5 HiL-IDQL+Mulligan",
        "R5 HG-DAgger+Mulligan",
        "R5 HG-DAgger",
        "R0 HG-DAgger",
    ]


def test_sampler_ablation_means():
    rows = {r["cell_key"]: r for r in sim_contrasts.sampler_ablation()}
    means = [
        round(rows[key]["mean"], 2)
        for key in (
            "square_narrow_r1_baseline_uniform_no_cf_human_only",
            "square_narrow_r1_mulligan_no_cf_human_only",
            "square_narrow_r1_mulligan_with_cf_human_only",
        )
    ]
    assert means == [85.73, 92.69, 93.52]


def test_rlpd_ablations():
    # baselines appendix, "RLPD ablations": 5 seeds each, ~80k zero phase, whole-curve
    # mean eval within +/-0.12 of the reference (95% CI).
    q = rlpd_ablations.quoted()
    assert q["n_seeds"] == {"reference_utd20": 5, "early_kill": 5, "utd40": 5}
    assert q["zero_phase_k"] == {"reference_utd20": 90.0, "early_kill": 80.0, "utd40": 90.0}
    assert q["auc_ci95"] == {"early_kill": [-0.0868, 0.0656], "utd40": [-0.1184, 0.0944]}
    assert q["auc_ci_bound"] == 0.12


def test_rlpd_ablations_summary_detects_drift(tmp_path):
    frozen = rlpd_ablations.SUMMARY.read_text().replace('"early_kill": 5', '"early_kill": 4')
    (tmp_path / "summary.json").write_text(frozen)
    summary = rlpd_ablations.summarize(rlpd_ablations.curves())
    with pytest.raises(AssertionError, match="differs"):
        rlpd_ablations.check_summary(summary, tmp_path / "summary.json")


def test_sobol_round0():
    r0 = sim_contrasts.sobol_round0()
    assert round(r0["square_narrow"]["delta"], 2) == 8.23
    assert round(r0["square_broad"]["delta"], 2) == 3.53
    assert (round(r0["square_narrow"]["lo"], 2), round(r0["square_narrow"]["hi"], 2)) == (
        5.24,
        11.22,
    )
    assert (round(r0["square_broad"]["lo"], 2), round(r0["square_broad"]["hi"], 2)) == (1.07, 5.98)


def test_frozen_snapshot_verifiers():
    reranking_check.check_input_hashes()
    assert len(reranking_check.release_settings()) == 10


@reads_evidence
def test_collection_totals():
    t = print_collection_totals.totals()["mulligan_sobol"]
    assert (t["successes"], t["n"]) == (988, 1009)
    worst = t["min_round"]
    assert (worst["task"], worst["round"], worst["successes"], worst["n"]) == (
        "routing_d2",
        "R4",
        100,
        111,
    )


def test_eval_episode_total():
    t = count_eval_episodes.episode_totals()
    assert (t["datasets"], t["counted"], t["counted_plus_no_cf"]) == (13, 2550, 2850)
    assert {task: c["counted"] for task, c in t["by_task"].items()} == {
        "real-marker-d2": 850,
        "real-square-d2": 950,
        "real-routing-d2": 750,
    }
    assert len(t["screen"]) == 2


@reads_evidence
def test_eval_episode_lock_equals_the_evidence_copy():
    """The lock in release/ is the copy pinned in the real_results paper evidence."""
    from mulligan.release.round_counts import LOCK_PATH
    from paper.stats.evidence import pinned_input

    evidence = pinned_input("real_results", "real/results/round_datasets.json")
    assert evidence.read_bytes() == LOCK_PATH.read_bytes()


def test_r0_demo_duration_rounding_is_half_up():
    # Square-Narrow's 165 frames and Square-Broad's 171 frames at 20 Hz.
    assert r0_demo_durations.round_half_up(Decimal(165) / 20) == Decimal("8.3")
    assert r0_demo_durations.round_half_up(Decimal(171) / 20) == Decimal("8.6")
    assert r0_demo_durations.round_half_up(Decimal(152) / 15) == Decimal("10.1")


@pytest.mark.network
def test_r0_demo_durations():
    durations = r0_demo_durations.median_durations()
    assert durations["square_narrow"] == Decimal("8.25")
    assert durations["square_broad"] == Decimal("8.55")
    assert {task: r0_demo_durations.round_half_up(s) for task, s in durations.items()} == {
        "marker": Decimal("12.7"),
        "square": Decimal("10.1"),
        "routing": Decimal("18.6"),
        "square_narrow": Decimal("8.3"),
        "square_broad": Decimal("8.6"),
    }
