"""Guards for the stage-label eval plot battery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mulligan.real.stage_specs import get_label_task_spec
from mulligan.plotting.colors import METHOD_COLORS
from mulligan.real.stage_labeling.eval_battery import (
    ArmPlotSpec,
    _anchor_stage_to_outcome,
    _mcnemar_exact_p_value,
    plot_rung_conversion,
    plot_stage_share,
    run_stage_eval_battery,
)


def test_raw_routing_stage_reach_preserves_raw_success_and_labels(tmp_path):
    """A past two-clip seat is not silently turned into a final task success."""
    spec = get_label_task_spec("routing_d2")
    frame = pd.DataFrame(
        {
            "episode_index": [0, 1, 2],
            "policy_short": ["baseline"] * 3,
            "max_stage": [5, 6, 10],
            "failure_mode": ["none"] * 3,
            "model_task_success": [False] * 3,
        }
    )
    paired = tmp_path / "paired.csv"
    pd.DataFrame({"baseline_episode_index": [0, 1, 2], "baseline_success": [False] * 3}).to_csv(
        paired, index=False
    )
    arms = [ArmPlotSpec("baseline", "Baseline", METHOD_COLORS["uniform"])]
    labels = ("One clip ever seated", "Both clips | one clip", "Both clips ever seated")
    rung_plot = tmp_path / "rungs.svg"
    table, _ = plot_rung_conversion(
        frame,
        spec,
        arms,
        title="Machine stage reach",
        out=rung_plot,
        paired_rounds_csv=paired,
        rung_labels=labels,
    )
    keyed = table.set_index("rung")
    assert keyed.loc["G", "k"] == 2
    assert keyed.loc["I", "n"] == 2
    assert keyed.loc["SR", "k"] == 1
    assert keyed.loc["SR", "rate"] == pytest.approx(1 / 3)
    assert keyed.loc["SR", "metric_label"] == labels[2]
    svg = rung_plot.read_text()
    assert all(label in svg for label in labels)
    assert "seat + release" not in svg and "SR =" not in svg
    share = tmp_path / "share.svg"
    plot_stage_share(
        frame,
        spec,
        arms,
        title="Machine stage reach",
        out=share,
        stage_axis_label="Historical maximum stage; S10 = both clips ever seated",
    )
    assert "both clips ever seated" in share.read_text()
    assert "strict success" not in share.read_text()


def test_empty_conditional_stage_cohort_is_not_an_estimated_zero(tmp_path):
    spec = get_label_task_spec("routing_d2")
    frame = pd.DataFrame(
        {"episode_index": [0, 1], "policy_short": ["baseline", "mulligan"], "max_stage": [2, 10]}
    )
    arms = [
        ArmPlotSpec("baseline", "Baseline", METHOD_COLORS["uniform"]),
        ArmPlotSpec("mulligan", "Ours", METHOD_COLORS["sobol"]),
    ]
    path = tmp_path / "empty_conditional.svg"
    table, comparisons = plot_rung_conversion(frame, spec, arms, title="Stage reach", out=path)
    conditional = table[(table.arm == "baseline") & (table.rung == "I")].iloc[0]
    assert conditional.n == 0
    assert pd.isna(conditional.rate) and pd.isna(conditional.lo) and pd.isna(conditional.hi)
    comparison = comparisons[comparisons.rung == "I"].iloc[0]
    assert comparison.test == "not estimated" and pd.isna(comparison.p_value)
    assert "n/a" in path.read_text()


def test_shared_starts_can_keep_conditional_rates_descriptive(tmp_path):
    spec = get_label_task_spec("routing_d2")
    frame = pd.DataFrame(
        {
            "episode_index": [0, 1, 2, 3, 4, 5],
            "policy_short": ["baseline", "mulligan"] * 3,
            "max_stage": [10, 10, 6, 2, 2, 6],
        }
    )
    paired = tmp_path / "paired.csv"
    pd.DataFrame({"baseline_episode_index": [0, 2, 4], "mulligan_episode_index": [1, 3, 5]}).to_csv(
        paired, index=False
    )
    arms = [
        ArmPlotSpec("baseline", "Baseline", METHOD_COLORS["uniform"]),
        ArmPlotSpec("mulligan", "Ours", METHOD_COLORS["sobol"]),
    ]
    path = tmp_path / "descriptive.svg"
    rates, comparisons = plot_rung_conversion(
        frame,
        spec,
        arms,
        title="Raw stage reach",
        out=path,
        paired_rounds_csv=paired,
        conditional_test="descriptive",
    )
    assert rates[rates.rung == "I"].rate.tolist() == [0.5, 0.5]
    conditional = comparisons[comparisons.rung == "I"].iloc[0]
    assert conditional.test == "descriptive only" and pd.isna(conditional.p_value)
    assert set(comparisons[comparisons.rung != "I"].test) == {"paired exact McNemar"}
    assert "Fisher" not in path.read_text() and "McNemar" in path.read_text()


def test_anchor_stage_to_outcome_pins_top_rung_to_ground_truth(tmp_path):
    # The battery SR must equal the human outcome-edited success, not the VLM's
    # perceptual S7: success -> success_level; recorded failure -> capped below
    # success_level even when the VLM hallucinated S7; lower failure stages stay.
    spec = get_label_task_spec("marker_d2")
    s = spec.ladder.success_level
    df = pd.DataFrame(
        {
            "episode_index": [0, 1, 2, 3],
            "policy_short": ["dp", "dp", "dp", "dp"],
            spec.stage_field: [3, s, 5, s],  # ep0 under-called, ep1 false-S7, ep3 true-S7
            spec.failure_mode_field: ["none", "none", "none", "none"],
        }
    )
    paired = tmp_path / "paired.csv"
    pd.DataFrame({"dp_episode_index": [0, 1, 2, 3], "dp_success": [1, 0, 0, 1]}).to_csv(
        paired, index=False
    )
    out = _anchor_stage_to_outcome(df, spec, [ArmPlotSpec("dp", "DP", "#000")], paired)
    stages = out.sort_values("episode_index")[spec.stage_field].tolist()
    assert stages == [s, s - 1, 5, s]
    # anchored S7 count == recorded successes (the fail-loud invariant)
    assert int((out[spec.stage_field] == s).sum()) == 2


def test_anchor_stage_to_outcome_fails_loud_on_missing_outcome(tmp_path):
    spec = get_label_task_spec("marker_d2")
    df = pd.DataFrame(
        {
            "episode_index": [0, 1],
            "policy_short": ["dp", "dp"],
            spec.stage_field: [3, 3],
            spec.failure_mode_field: ["none", "none"],
        }
    )
    paired = tmp_path / "paired.csv"
    pd.DataFrame({"dp_episode_index": [0], "dp_success": [1]}).to_csv(paired, index=False)
    with pytest.raises(ValueError, match="no outcome"):
        _anchor_stage_to_outcome(df, spec, [ArmPlotSpec("dp", "DP", "#000")], paired)


def test_mcnemar_exact_p_value_triple_is_stable():
    # Pins the (p_value, left_only, right_only) contract so delegating the p-value
    # to the shared lifecycle helper is unchanged, including the
    # all-concordant -> p=1.0 branch.
    left = np.array([1, 1, 0, 1], dtype=bool)
    right = np.array([0, 1, 0, 0], dtype=bool)
    assert _mcnemar_exact_p_value(left, right) == (0.5, 2, 0)
    concordant = np.array([1, 0, 1], dtype=bool)
    assert _mcnemar_exact_p_value(concordant, concordant) == (1.0, 0, 0)


def test_stage_eval_battery_writes_rung_comparison_annotations(tmp_path: Path) -> None:
    spec = get_label_task_spec("marker_d2")
    baseline_stages = [3, 7, 2, 1, 2, 1]
    ours_stages = [3, 7, 3, 3, 7, 3]
    rows = []
    pair_rows = []
    for idx, (baseline_stage, ours_stage) in enumerate(zip(baseline_stages, ours_stages)):
        baseline_episode = 2 * idx
        ours_episode = 2 * idx + 1
        rows.extend(
            [
                {
                    "episode_index": baseline_episode,
                    "policy_short": "baseline_r00_dp",
                    spec.stage_field: baseline_stage,
                    spec.failure_mode_field: "none",
                },
                {
                    "episode_index": ours_episode,
                    "policy_short": "mulligan_r00_dp",
                    spec.stage_field: ours_stage,
                    spec.failure_mode_field: "none",
                },
            ]
        )
        pair_rows.append(
            {
                "round": idx + 1,
                "baseline_episode_index": baseline_episode,
                "mulligan_episode_index": ours_episode,
            }
        )

    labels_csv = tmp_path / "labels_joined.csv"
    paired_csv = tmp_path / "paired_round_outcomes.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)
    pd.DataFrame(pair_rows).to_csv(paired_csv, index=False)

    outputs = run_stage_eval_battery(
        spec=spec,
        labels_csv=labels_csv,
        plot_dir=tmp_path / "plots",
        csv_dir=tmp_path / "csv",
        prefix="marker_d2_test",
        title="marker_d2 test",
        paired_rounds_csv=paired_csv,
    )

    comparisons = pd.read_csv(outputs["rung_comparisons_csv"]).set_index("rung")
    g_row = comparisons.loc["G"]
    assert g_row["test"] == "paired exact McNemar"
    assert g_row["left_only"] == 0
    assert g_row["right_only"] == 4
    assert g_row["p_value"] == pytest.approx(0.125)

    svg = outputs["rung_conversion"].read_text()
    assert "McNemar p=0.125" in svg
    assert "Fisher p=" in svg


def test_stage_eval_battery_annotates_three_arm_main_comparison(tmp_path: Path) -> None:
    spec = get_label_task_spec("square_d2")
    rows = []
    pair_rows = []
    stages = {
        "baseline": [1, 3, 7, 1, 3, 1],
        "mulligan_no_cf": [1, 3, 7, 3, 3, 1],
        "mulligan_with_cf": [3, 3, 7, 7, 7, 3],
    }
    for idx in range(6):
        pair_row = {"round": idx + 1}
        for arm, arm_stages in stages.items():
            episode = idx * 3 + list(stages).index(arm)
            rows.append(
                {
                    "episode_index": episode,
                    "policy_short": arm,
                    spec.stage_field: arm_stages[idx],
                    spec.failure_mode_field: "none",
                }
            )
            pair_row[f"{arm}_episode_index"] = episode
        pair_rows.append(pair_row)

    labels_csv = tmp_path / "labels_joined.csv"
    paired_csv = tmp_path / "paired_round_outcomes.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)
    pd.DataFrame(pair_rows).to_csv(paired_csv, index=False)

    outputs = run_stage_eval_battery(
        spec=spec,
        labels_csv=labels_csv,
        plot_dir=tmp_path / "plots",
        csv_dir=tmp_path / "csv",
        prefix="square_d2_test",
        title="square_d2 test",
        arms=[
            ArmPlotSpec("baseline", "Baseline\nno-CF", METHOD_COLORS["uniform"]),
            ArmPlotSpec("mulligan_no_cf", "Ours\nno-CF", METHOD_COLORS["real_mulligan_no_cf"]),
            ArmPlotSpec(
                "mulligan_with_cf", "Ours\nwith-CF", METHOD_COLORS["real_mulligan_with_cf"]
            ),
        ],
        paired_rounds_csv=paired_csv,
    )

    comparisons = pd.read_csv(outputs["rung_comparisons_csv"])
    main = comparisons[
        (comparisons["left_arm"] == "baseline") & (comparisons["right_arm"] == "mulligan_with_cf")
    ]
    assert set(main["rung"]) == {"G", "I", "SR"}

    svg = outputs["rung_conversion"].read_text()
    assert "baseline vs with-CF" in svg
    assert "McNemar p=" in svg
    assert "Fisher p=" in svg
