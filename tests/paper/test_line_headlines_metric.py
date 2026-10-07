"""Contract tests for the line-headline ``headline_metric`` knob.

The SR surface reads its numerator from policy_summary via
``HEADLINE_METRICS`` so a near-floor-full-success line (Route Cable) can headline
on first-clip rate (score>=1) without forking the plot code. These tests pin
that the knob switches the numerator column, defaults to full success, and fails
loud on an unknown metric or a missing metric column.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from paper.real_headline import (
    HEADLINE_METRICS,
    LineHeadlineConfig,
    PolicyArmSpec,
    RoundHeadlineSpec,
    SuiteLineHeadlineSpec,
    _suite_grouped_frame,
    _with_binomial_sem,
    build_success_rate_table,
)


def test_clip_score_intervals_preserve_within_rollout_dependence(tmp_path) -> None:
    cfg = dataclasses.replace(
        _config(tmp_path, metric="task_progress", score_2=25), episode_score_intervals=True
    )
    path = cfg.rounds[0].eval_dir / "paired_round_outcomes.csv"
    pd.DataFrame({"manifest_idx": range(50), "baseline_score": [0] * 25 + [2] * 25}).to_csv(
        path, index=False
    )
    table = build_success_rate_table(cfg, arms=cfg.headline_arms, headline_labels=True)
    row = table.iloc[0]
    assert row["episode_n"] == 50
    assert row["episode_sem"] == pytest.approx(np.sqrt(0.25 / 49))
    assert row["wilson_lo"] == pytest.approx(0.36)
    assert row["wilson_hi"] == pytest.approx(0.64)
    plotted = _with_binomial_sem(
        _suite_grouped_frame(SuiteLineHeadlineSpec(cfg, "Probe"), table), "success_rate", "n"
    )
    assert plotted.iloc[0]["sem_hi"] == pytest.approx(0.5 + np.sqrt(0.25 / 49))
    # A binary companion remains episode-binomial, not graded-score uncertainty.
    binary = build_success_rate_table(
        cfg, arms=cfg.headline_arms, headline_labels=True, metric="full_success"
    )
    assert "episode_sem" not in binary


def test_clip_score_interval_rejects_summary_mismatch(tmp_path) -> None:
    cfg = dataclasses.replace(
        _config(tmp_path, metric="task_progress", score_2=25), episode_score_intervals=True
    )
    pd.DataFrame({"manifest_idx": range(50), "baseline_score": [0] * 50}).to_csv(
        cfg.rounds[0].eval_dir / "paired_round_outcomes.csv", index=False
    )
    with pytest.raises(ValueError, match="disagree with graded policy summary"):
        build_success_rate_table(cfg, arms=cfg.headline_arms, headline_labels=True)


def test_shared_start_model_rounds_cannot_be_pooled_as_independent(tmp_path) -> None:
    cfg = _config(tmp_path, metric="task_progress")
    cfg = dataclasses.replace(
        cfg,
        episode_score_intervals=True,
        rounds=(cfg.rounds[0], dataclasses.replace(cfg.rounds[0], round_label="R1")),
        round_tick_labels={"R0": "R0", "R1": "R1"},
    )
    line = SuiteLineHeadlineSpec(cfg, "Probe", round_groups=(("R0", "R1"),))
    with pytest.raises(ValueError, match="cannot be pooled as independent"):
        _suite_grouped_frame(line, pd.DataFrame())


def _write_eval_dir(
    eval_dir,
    *,
    successes: int,
    ge1_count: int,
    include_ge1: bool = True,
    score_1: int = 0,
    score_2: int = 0,
    graded_episodes: int | None = None,
    ge2_count: int | None = None,
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "policy_name": "pol",
        "policy_label": "Pol",
        "model_id": "ckpt-a",
        "episodes": 50,
        "successes": successes,
        "score_1": score_1,
        "score_2": score_2,
    }
    if include_ge1:
        summary["ge1_count"] = ge1_count
    if graded_episodes is not None:
        summary["graded_episodes"] = graded_episodes
    if ge2_count is not None:
        summary["ge2_count"] = ge2_count
    pd.DataFrame([summary]).to_csv(eval_dir / "policy_summary.csv", index=False)
    # A minimal paired-outcomes file with a unique per-row state key (manifest_idx
    # alone is enough for a single, non-pooled block).
    pd.DataFrame({"manifest_idx": range(50)}).to_csv(
        eval_dir / "paired_round_outcomes.csv", index=False
    )


def _config(
    tmp_path, *, metric: str, include_ge1: bool = True, score_1: int = 0, score_2: int = 0
) -> LineHeadlineConfig:
    eval_dir = tmp_path / "r0"
    _write_eval_dir(
        eval_dir,
        successes=2,
        ge1_count=17,
        include_ge1=include_ge1,
        score_1=score_1,
        score_2=score_2,
    )
    rounds = (
        RoundHeadlineSpec(
            round_label="R0",
            eval_dir=eval_dir,
            stage_rung_csv=None,
            policy_arms={
                "baseline": PolicyArmSpec(
                    policy_name="pol",
                    paired_prefix="baseline",
                    stage_source_arm=None,
                    sr_provenance="test",
                    stage_provenance="test",
                ),
            },
        ),
    )
    return LineHeadlineConfig(
        task_key="probe",
        out_data_dir=tmp_path / "data",
        out_plot_dir=tmp_path / "plots",
        rounds=rounds,
        r0_teleop_stats_csv=tmp_path / "unused.csv",
        collection_dirs={},
        training_views={},
        round_tick_labels={"R0": "R0"},
        headline_arms=("baseline",),
        arm_order=("baseline",),
        campaign_pair=None,
        headline_metric=metric,
    )


def test_first_clip_metric_reads_ge1_count(tmp_path) -> None:
    cfg = _config(tmp_path, metric="first_clip")
    df = build_success_rate_table(cfg, arms=cfg.headline_arms, headline_labels=True)
    row = df[df["is_pooled"]].iloc[0]
    assert int(row["successes"]) == 17  # ge1_count, not the 2 full successes
    assert int(row["n"]) == 50
    assert row["success_rate"] == pytest.approx(0.34)


def test_full_success_metric_is_the_default(tmp_path) -> None:
    cfg = _config(tmp_path, metric="full_success")
    assert cfg.headline_metric == "full_success"
    df = build_success_rate_table(cfg, arms=cfg.headline_arms, headline_labels=True)
    row = df[df["is_pooled"]].iloc[0]
    assert int(row["successes"]) == 2  # policy_summary "successes"
    assert row["success_rate"] == pytest.approx(0.04)


def test_distinct_auxiliary_eval_dir_is_registered_without_changing_rounds(tmp_path) -> None:
    cfg = _config(tmp_path, metric="full_success")
    auxiliary = tmp_path / "r0_candidate_screen"
    updated = dataclasses.replace(cfg, auxiliary_eval_dirs=(auxiliary,))
    assert updated.auxiliary_eval_dirs == (auxiliary,)
    assert [round_spec.round_label for round_spec in updated.rounds] == ["R0"]


def test_auxiliary_eval_dir_cannot_duplicate_primary_round(tmp_path) -> None:
    cfg = _config(tmp_path, metric="full_success")
    with pytest.raises(ValueError, match="overlap primary round dirs"):
        dataclasses.replace(cfg, auxiliary_eval_dirs=(cfg.rounds[0].eval_dir,))


def test_task_progress_metric_reads_clip_seat_fraction(tmp_path) -> None:
    # Average task progress = clip-seats / clip-seat opportunities == mean_score/2.
    # 11 episodes seat one clip (score 1) + 1 seats both (score 2) -> 11 + 2*1 = 13
    # clip-seats out of 2*50 = 100 opportunities -> 0.13 (== the Route Cable R2 baseline).
    cfg = _config(tmp_path, metric="task_progress", score_1=11, score_2=1)
    df = build_success_rate_table(cfg, arms=cfg.headline_arms, headline_labels=True)
    row = df[df["is_pooled"]].iloc[0]
    assert int(row["successes"]) == 13  # score_1 + 2*score_2 clip-seats
    assert int(row["n"]) == 100  # 2 * episodes clip-seat opportunities
    assert row["success_rate"] == pytest.approx(0.13)


def test_metric_override_reads_second_metric_from_same_summaries(tmp_path) -> None:
    # A graded-headline line (Route Cable) also surfaces raw full success: the
    # override must win over cfg.headline_metric without a config fork.
    cfg = _config(tmp_path, metric="task_progress", score_1=11, score_2=1)
    df = build_success_rate_table(
        cfg, arms=cfg.headline_arms, headline_labels=True, metric="full_success"
    )
    row = df[df["is_pooled"]].iloc[0]
    assert int(row["successes"]) == 2  # policy_summary "successes", not clip-seats
    assert int(row["n"]) == 50
    assert row["success_rate"] == pytest.approx(0.04)


def test_metric_override_unknown_fails_loud(tmp_path) -> None:
    cfg = _config(tmp_path, metric="task_progress", score_1=1, score_2=0)
    with pytest.raises(ValueError, match="unknown metric"):
        build_success_rate_table(
            cfg, arms=cfg.headline_arms, headline_labels=True, metric="ge2_rate"
        )


def test_task_progress_requires_score_columns(tmp_path) -> None:
    cfg = _config(tmp_path, metric="task_progress", score_1=5, score_2=0)
    # Drop the score_2 column the metric needs -> fail loud.
    eval_dir = cfg.rounds[0].eval_dir
    summary = pd.read_csv(eval_dir / "policy_summary.csv").drop(columns=["score_2"])
    summary.to_csv(eval_dir / "policy_summary.csv", index=False)
    with pytest.raises(KeyError, match="score_2"):
        build_success_rate_table(cfg, arms=cfg.headline_arms, headline_labels=True)


def test_unknown_metric_fails_loud() -> None:
    with pytest.raises(ValueError, match="unknown headline_metric"):
        LineHeadlineConfig(
            task_key="probe",
            out_data_dir="d",
            out_plot_dir="p",
            rounds=(),
            r0_teleop_stats_csv="x",
            collection_dirs={},
            training_views={},
            round_tick_labels={},
            headline_metric="ge2_rate",
        )


def test_headline_metrics_round_subset_must_follow_config_order(tmp_path) -> None:
    cfg = _config(tmp_path, metric="full_success")
    r1 = dataclasses.replace(cfg.rounds[0], round_label="R1")
    with pytest.raises(ValueError, match="must follow config round order"):
        dataclasses.replace(
            cfg,
            rounds=(cfg.rounds[0], r1),
            round_tick_labels={"R0": "R0", "R1": "R1"},
            stage_metric_enabled=True,
            headline_metrics_output_name="headline_metrics",
            headline_metrics_round_subsets={"headline_metrics_subset": ("R1", "R0")},
        )


def test_headline_metrics_round_subset_rejects_unknown_round(tmp_path) -> None:
    cfg = _config(tmp_path, metric="full_success")
    with pytest.raises(ValueError, match="unknown rounds.*R2"):
        dataclasses.replace(
            cfg,
            stage_metric_enabled=True,
            headline_metrics_output_name="headline_metrics",
            headline_metrics_round_subsets={"headline_metrics_subset": ("R0", "R2")},
        )


def test_first_clip_requires_numerator_column(tmp_path) -> None:
    cfg = _config(tmp_path, metric="first_clip", include_ge1=False)
    with pytest.raises(KeyError, match="ge1_count"):
        build_success_rate_table(cfg, arms=cfg.headline_arms, headline_labels=True)


def test_metric_registry_covers_configured_lines() -> None:
    assert set(HEADLINE_METRICS) >= {
        "full_success",
        "first_clip",
        "reviewed_full_success",
        "task_progress",
    }
    for spec in HEADLINE_METRICS.values():
        # A metric names either a single count column or a weighted sum of columns.
        assert ("numerator_col" in spec) or ("numerator_cols" in spec)
        assert "noun" in spec


@pytest.mark.parametrize("scores", [[0, 1, 1, 2], [0, 0, 2, 2], [0, 0, 0, 0], [2, 2, 2, 2]])
def test_paper_task_progress_uses_episode_sampling_units(tmp_path, scores) -> None:
    import numpy as np

    from paper.real_headline import (
        SuiteLineHeadlineSpec,
        _read_suite_headline_sr,
        _suite_paper_frame,
    )

    cfg = _config(tmp_path, metric="full_success")
    arms = {arm: cfg.rounds[0].policy_arms["baseline"] for arm in ("baseline", "mulligan_with_cf")}
    cfg = dataclasses.replace(
        cfg,
        headline_arms=tuple(arms),
        rounds=(dataclasses.replace(cfg.rounds[0], policy_arms=arms),),
    )
    cfg.out_data_dir.mkdir()
    values = np.asarray(scores, dtype=float) / 2
    for name, successes, n in (
        ("headline_sr", scores.count(2), len(scores)),
        ("headline_task_progress", sum(scores), 2 * len(scores)),
    ):
        pd.DataFrame(
            [
                {
                    "round": "R0",
                    "arm": arm,
                    "is_pooled": True,
                    "successes": successes,
                    "n": n,
                    "success_rate": successes / n,
                    "wilson_lo": 0,
                    "wilson_hi": 1,
                }
                for arm in arms
            ]
        ).to_csv(cfg.out_data_dir / f"{cfg.task_key}_{name}.csv", index=False)
    line = SuiteLineHeadlineSpec(cfg, "Probe", task_progress=True)
    frame = _suite_paper_frame(line, _read_suite_headline_sr(line), whisker="wilson1se")
    sem = values.std(ddof=1) / np.sqrt(len(values))
    for row in frame.to_dict("records"):
        assert row["n"] == len(scores)
        assert row["value"] == pytest.approx(values.mean())
        assert row["lo"] == pytest.approx(max(0, values.mean() - sem))
        assert row["hi"] == pytest.approx(min(1, values.mean() + sem))
