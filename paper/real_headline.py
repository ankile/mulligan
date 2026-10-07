"""Real-world line-headline tables and paper figures (Marker, Nut, Cable).

Derived surfaces shared by the main-text headline and the appendix:

- held-out success-rate tables and paired campaign tests per round and arm;
- the suite paper headline panels (HG-DAgger vs HG-DAgger+Mulligan, with the
  HiL-IDQL+Mulligan rerank takeover) and their detailed appendix companion;
- episode-duration / throughput efficiency series;
- G/I/SR substage completion;
- DAgger parent-collection credited success rate.

Each task is described by a :class:`LineHeadlineConfig`; the paper's configs are
frozen in ``paper/appendix/real_results/config.json``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

from mulligan.plotting import paper
from mulligan.plotting.colors import METHOD_COLORS

from mulligan.real.lifecycle.stats import (
    bootstrap_mean_ci,
    bootstrap_paired_delta_stratified,
    exact_mcnemar_pvalue,
    wilson_ci,
    wilson_1se,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The deployed (teal->purple) line switches to the reranked policy at R3 on every
# task: the policy plotted in each round is the one that collected the next round's
# data, and the R2 actor collected Marker R3.
PAPER_TAKEOVER_START = {"marker_d2": "R3", "square_d2": "R3", "routing_d2": "R3"}


def _display_path(path: Path) -> str:
    """Repo-relative display form of ``path`` (the raw path outside the repo)."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


DEFAULT_HEADLINE_ARMS = ("baseline", "mulligan_with_cf")
DEFAULT_ARM_ORDER = ("baseline", "mulligan_no_cf", "mulligan_with_cf")
# DP+IQL best-of-N rerank arms: a third arm at the pooled final round. Their
# substage rungs come from their own stage-labeled blocks (per-source
# ArmSourceSpec.stage_rung_csv/stage_arm), so the substage footnote calls the
# pooled-block convention out explicitly.
IQL_RERANK_ARMS = frozenset({"final_iql", "final_marker_iql"})

# Held-out headline metric registry. The SR surface (build_success_rate_table +
# plot_success_rate) reads its numerator/denominator from policy_summary via this
# table so a near-floor-full-success line (e.g. Route Cable, where full success is
# ~0 every round) can headline on a more informative graded quantity without
# forking the plot code. "full_success" is the default
# (numerator = policy_summary "successes" column, denominator = episodes). Only
# the SR surface is metric-aware; the efficiency surface stays full-success (it is
# a completion-time readout) so the two surfaces read as complementary.
#
# Each entry names either a single ``numerator_col`` (a count) or ``numerator_cols``
# (a weighted sum of count columns), plus ``denom_scale`` (denominator =
# episodes * denom_scale, default 1). ``task_progress`` reads the graded 0-2 clip
# score as *average task progress* = the fraction of the two per-episode clip-seat
# opportunities achieved (score_1 seats one clip, score_2 seats both), i.e.
# (score_1 + 2*score_2) / (2*episodes) == mean_score/2 in [0,1]; the Wilson CI is
# on that clip-seat fraction (consistent with the line's own per-threshold ge1/ge2
# Wilson CIs). It is the informative Route Cable headline: partial first-clip credit
# without collapsing to the ~0 full-success rate.
HEADLINE_METRICS: dict[str, dict[str, object]] = {
    "full_success": {
        "numerator_col": "successes",
        "denom_scale": 1,
        "noun": "success rate",
        "panel_title": "Full success\n(binary 0/1)",
    },
    "first_clip": {
        "numerator_col": "ge1_count",
        "denominator_col": "graded_episodes",
        "denom_scale": 1,
        "noun": "first-clip rate (score>=1)",
        "panel_title": "First clip seated",
    },
    "reviewed_full_success": {
        "numerator_col": "ge2_count",
        "denominator_col": "graded_episodes",
        "denom_scale": 1,
        "noun": "full-success rate among graded episodes",
        "panel_title": "Full success\n(graded episodes)",
    },
    "task_progress": {
        "numerator_cols": {"score_1": 1, "score_2": 2},
        "denominator_col": "graded_episodes",
        "denom_scale": 2,
        "noun": "average task progress",
        "panel_title": "Task progress\n(mean score / 2)",
    },
}


def _metric_numerator_denominator(row: Mapping[str, object], metric: str) -> tuple[int, int]:
    """Resolve (numerator, denominator) for ``metric`` from a policy_summary row.

    Numerator is a single count column or a weighted sum of count columns;
    denominator is ``episodes * denom_scale`` (so ``task_progress`` reads
    clip-seats out of 2*episodes clip-seat opportunities). Missing columns fail
    loud — a metric can only headline a line whose ingest emits its inputs.
    """
    spec = HEADLINE_METRICS[metric]
    denominator_col = str(spec.get("denominator_col", "episodes"))
    # Fully reviewed summaries without a graded_episodes column graded every
    # episode, so episodes is exact.
    if denominator_col not in row:
        if denominator_col != "graded_episodes":
            raise KeyError(
                f"policy_summary row {row.get('policy_name')!r} has no "
                f"{denominator_col!r} denominator required by headline_metric={metric!r}"
            )
        denominator_col = "episodes"
    n = int(row[denominator_col]) * int(spec.get("denom_scale", 1))
    if "numerator_cols" in spec:
        numerator = 0
        for col, weight in dict(spec["numerator_cols"]).items():
            if col not in row:
                raise KeyError(
                    f"policy_summary row {row.get('policy_name')!r} has no {col!r} column "
                    f"required by headline_metric={metric!r}; available={sorted(row)}"
                )
            numerator += int(row[col]) * int(weight)
        return numerator, n
    col = str(spec["numerator_col"])
    if col not in row:
        raise KeyError(
            f"policy_summary row {row.get('policy_name')!r} has no {col!r} column "
            f"required by headline_metric={metric!r}; available={sorted(row)}"
        )
    return int(row[col]), n


EVAL_FPS = 15.0
THROUGHPUT_BOOTSTRAPS = 20_000
NEUTRAL = METHOD_COLORS["gray_neutral"]

ARM_STYLES: dict[str, dict[str, object]] = {
    "baseline": {
        "label": "Baseline no-CF",
        "color": METHOD_COLORS["uniform"],
        "marker": "o",
        "headline_dx": -0.09,
        "all_dx": -0.12,
    },
    "mulligan_no_cf": {
        "label": "Ours no-CF",
        "color": METHOD_COLORS["real_mulligan_no_cf"],
        "marker": "s",
        "headline_dx": 0.0,
        "all_dx": 0.0,
    },
    "mulligan_with_cf": {
        "label": "Ours with-CF",
        "headline_label": "Ours with-CF/headline",
        "color": METHOD_COLORS["real_mulligan_with_cf"],
        "marker": "D",
        "headline_dx": 0.09,
        "all_dx": 0.12,
    },
    "final_iql": {
        # The square DP+IQL lineage arm (per-round deployed critic generation:
        # R3 = S1, R4+ = S2…), connected across rounds like baseline/with-CF —
        # same convention as final_marker_iql.
        # Shares each round's x-position with baseline/with-CF, so it takes a
        # wider right offset than mulligan_with_cf (+0.09) to avoid overplot.
        "label": "Ours DP+IQL rerank",
        "color": METHOD_COLORS["real_iql_rerank"],
        "marker": "P",
        "headline_dx": 0.24,
        "all_dx": 0.24,
    },
    "final_marker_iql": {
        # The DP+IQL lineage arm (per-round deployed critic generation: R2 =
        # M1@125k, R3 = critic@100k), connected across rounds exactly like the
        # baseline/with-CF lineages — the critic
        # differing per round is no different from the DP checkpoint differing
        # per round; per-point critic identity lives in the provenance strings.
        # Shares each round's x-position with baseline/with-CF, so it takes a
        # wider right offset than mulligan_with_cf (+0.09) to avoid overplot.
        "label": "Ours DP+IQL rerank",
        "color": METHOD_COLORS["real_iql_rerank"],
        "marker": "P",
        "headline_dx": 0.24,
        "all_dx": 0.24,
    },
}


@dataclass(frozen=True)
class ArmSourceSpec:
    """One (eval block, policy, paired-prefix) source contributing to a pooled arm.

    A headline arm may pool episodes across several same-checkpoint eval blocks
    (e.g. the square_d2 R3 DP arm draws from the R3 three-arm eval plus the
    two disjoint 50-start held-out blocks A and B). Each source names its own eval dir,
    the policy row to read there, the paired-outcome column prefix, and a short
    provenance label used in the pooled CSV.

    ``stage_rung_csv`` + ``stage_arm`` point at this block's published
    stage-label battery row set; a block that was never stage-labeled (e.g. the
    square R3 held-out block A) leaves both None and is excluded from the
    pooled substage rungs while still contributing to SR/efficiency.
    """

    eval_dir: Path
    policy_name: str
    paired_prefix: str
    block_label: str
    stage_rung_csv: Path | None = None
    stage_arm: str | None = None

    def __post_init__(self) -> None:
        if (self.stage_rung_csv is None) != (self.stage_arm is None):
            raise ValueError(
                f"{self.block_label}: stage_rung_csv and stage_arm must be set together"
            )


@dataclass(frozen=True)
class StageLabelSourceSpec:
    """Per-episode stage-label source powering one arm's GRANULAR stage metric.

    The granular headline scores each episode ``final stage / success rung``
    from ONE stage-labeled 50-episode block (a pooled binary arm still reads
    its granular score from its single labeled block — the ``n_note`` column
    carries the support caveat). ``labels_csv`` is the per-episode label file
    (``labels_joined.csv`` or a cascade-refined ``stage_labels.csv``);
    ``policy_short`` selects the arm's rows there. ``binary_col=None`` derives
    the per-episode binary as ``stage == success rung``.

    ``eval_dir`` names the eval block the labels index into (its
    ``paired_round_outcomes.csv`` per-arm ``*_episode_index`` columns are the
    pairing backbone and its ``policy_summary.csv`` the committed binary
    cross-check target); it defaults to ``labels_csv.parent.parent`` — the
    common ``<eval_dir>/stage_labels/<file>`` layout — and must resolve to one
    of the arm's SR source blocks. ``allow_binary_mismatch`` bounds the
    stage-derived-vs-committed success-count disagreement (0 by default; the
    only allowed exception is a reviewed outcome edit not reflected in the
    stage labels, e.g. marker_d2 R4 dp 29 vs 30).
    """

    labels_csv: Path
    stage_col: str
    policy_short: str
    binary_col: str | None = None
    eval_dir: Path | None = None
    allow_binary_mismatch: int = 0

    def block_eval_dir(self) -> Path:
        if self.eval_dir is not None:
            return self.eval_dir
        return self.labels_csv.parent.parent


@dataclass(frozen=True)
class PolicyArmSpec:
    """One plotted policy arm in one round.

    ``stage_source_arm=None`` excludes a single-source arm from the substage
    surfaces (used for a headline-only arm whose block has no published stage
    battery). ``pooled_sources`` overrides the single-source default: when set,
    the arm's held-out SR, efficiency, and substage rungs all pool across the
    listed blocks (SR/efficiency over every block, substage over the blocks
    that carry ``stage_rung_csv``/``stage_arm``); the blocks are verified
    same-checkpoint and initial-state disjoint before pooling, and
    ``stage_source_arm`` must stay None (per-source ``stage_arm`` replaces it).

    ``stage_label_source`` feeds the granular normalized-stage headline; on a
    ``stage_metric_enabled`` line every arm must carry one (fail loud at config
    construction, not silently drop an arm from the paper figure).

    ``same_start_repeat_pooling`` declares the pooled blocks as REPEATED eval
    sets of one shared initial-state manifest (e.g. square_d2 R5 pools two
    baseline / with-CF 50-rollout sets on the same starts): the
    disjointness check inverts — every pooled block must draw EXACTLY the same
    start set, so a partial or wrong-manifest block still fails loud. Pooled
    counts then hold repeated measures per start (the Wilson CI treats episodes
    as independent; note the clustering in ``sr_provenance``).
    """

    policy_name: str
    paired_prefix: str
    stage_source_arm: str | None
    sr_provenance: str
    stage_provenance: str
    pooled_sources: tuple[ArmSourceSpec, ...] | None = None
    stage_label_source: StageLabelSourceSpec | None = None
    same_start_repeat_pooling: bool = False

    def __post_init__(self) -> None:
        if self.pooled_sources is not None and self.stage_source_arm is not None:
            raise ValueError(
                f"{self.policy_name}: pooled arms carry stage info per source "
                "(ArmSourceSpec.stage_arm); stage_source_arm must be None"
            )
        if self.same_start_repeat_pooling and self.pooled_sources is None:
            raise ValueError(
                f"{self.policy_name}: same_start_repeat_pooling requires pooled_sources"
            )


@dataclass(frozen=True)
class RoundHeadlineSpec:
    """All source artifacts for one held-out eval round.

    ``stage_rung_csv=None`` declares that this round's stage-label battery has
    not been published yet: the round joins the SR/efficiency surfaces but is
    excluded from the substage surfaces (rather than failing the whole build).
    """

    round_label: str
    eval_dir: Path
    policy_arms: Mapping[str, PolicyArmSpec]
    stage_rung_csv: Path | None


@dataclass(frozen=True)
class HumanFrameSourceSpec:
    """Source row for one round's contribution to a training view."""

    source_kind: str
    source_name: str


@dataclass(frozen=True)
class TrainingViewSpec:
    """Selected cumulative training view and its source-round lineage."""

    label: str
    round_sources: Mapping[str, HumanFrameSourceSpec]


@dataclass(frozen=True)
class NestedProbabilitySpec:
    """Two nested binary events shown as P(A) and P(B | A)."""

    output_name: str
    condition_metric: str
    outcome_metric: str
    condition_label: str
    outcome_label: str

    def __post_init__(self) -> None:
        if not self.output_name or Path(self.output_name).name != self.output_name:
            raise ValueError(f"invalid nested-probability output_name={self.output_name!r}")
        for role, metric in (
            ("condition", self.condition_metric),
            ("outcome", self.outcome_metric),
        ):
            if metric not in HEADLINE_METRICS:
                raise ValueError(
                    f"unknown nested-probability {role} metric {metric!r}; "
                    f"expected one of {sorted(HEADLINE_METRICS)}"
                )
            metric_spec = HEADLINE_METRICS[metric]
            if int(metric_spec.get("denom_scale", 1)) != 1 or "numerator_col" not in metric_spec:
                raise ValueError(
                    f"nested-probability {role} metric {metric!r} must be a single "
                    "binary count over episodes"
                )
        if self.condition_metric == self.outcome_metric:
            raise ValueError("nested-probability condition and outcome metrics must differ")


@dataclass(frozen=True)
class LineHeadlineConfig:
    """Task-specific pins for the shared line-headline generator."""

    task_key: str
    out_data_dir: Path
    out_plot_dir: Path
    rounds: tuple[RoundHeadlineSpec, ...]
    r0_teleop_stats_csv: Path
    collection_dirs: Mapping[str, Path]
    training_views: Mapping[str, TrainingViewSpec]
    round_tick_labels: Mapping[str, str]
    sr_tick_labels: Mapping[str, str] | None = None
    headline_arms: tuple[str, ...] = DEFAULT_HEADLINE_ARMS
    arm_order: tuple[str, ...] = DEFAULT_ARM_ORDER
    # (baseline arm, treatment arm) for the campaign-pooled paired readout;
    # None skips the surface (e.g. a line with no shared control arm).
    campaign_pair: tuple[str, str] | None = ("baseline", "mulligan_with_cf")
    # Granular normalized-stage headline (per-episode final stage / success
    # rung). When enabled, every configured arm must carry a
    # stage_label_source and the run emits the *_headline_stage* surfaces.
    stage_metric_enabled: bool = False
    # Rounds with binary/graded outcomes but no granular stage labels. They
    # remain on non-stage surfaces and are omitted only from stage outputs.
    stage_excluded_rounds: tuple[str, ...] = ()
    # (baseline arm, treatment arm) for the per-round + pooled paired
    # binary-vs-granular significance table (*_paired_campaign_stage.csv).
    stage_campaign_pair: tuple[str, str] | None = None
    # Which policy_summary column headlines the SR surface; see HEADLINE_METRICS.
    headline_metric: str = "full_success"
    # Bootstrap whole rollout scores for graded progress, preserving within-episode
    # dependence between subtask events. Off by default.
    episode_score_intervals: bool = False
    # Per-line display-label overrides keyed by arm name. The ARM_STYLES labels
    # encode the pen-line 3-arm protocol ("Baseline no-CF"); a withCF-only
    # 2-arm line like Route Cable has no no-CF arm, so its baseline must read
    # "Baseline uniform".
    arm_label_overrides: Mapping[str, str] = field(default_factory=dict)
    # Optional P(A) / P(B | A) companion for nested binary outcome counts.
    nested_probability: NestedProbabilitySpec | None = None
    # Optional horizontal summary of decision-facing progression metrics. Every
    # line gets raw full success + granular normalized stage; a line whose
    # headline_metric is graded gets that metric as the middle panel. The
    # underlying single-panel surfaces are generated as well.
    headline_metrics_output_name: str | None = None
    # Optional additional renders of the same headline-metrics figure with a
    # selected round sequence. Keys are output names and values are ordered
    # round labels. The full figure above remains the canonical progression;
    # these variants support compact comparison views without a second plotter.
    headline_metrics_round_subsets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Same-round diagnostic eval blocks that are deliberately analyzed on a
    # dedicated companion surface instead of being folded into the DAgger
    # progression. This is for genuinely different policy mechanisms (for
    # example an R5 critic-candidate screen), not same-policy repeats: repeats
    # belong in ``PolicyArmSpec.pooled_sources`` so they contribute to the
    # headline statistics. Registering an auxiliary block keeps the lifecycle
    # gate aware of its paired outcomes and compare video without pretending it
    # is a new training round or silently pooling incomparable policies.
    auxiliary_eval_dirs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.headline_metric not in HEADLINE_METRICS:
            raise ValueError(
                f"{self.task_key}: unknown headline_metric {self.headline_metric!r}; "
                f"expected one of {sorted(HEADLINE_METRICS)}"
            )
        if self.stage_campaign_pair is not None and not self.stage_metric_enabled:
            raise ValueError(f"{self.task_key}: stage_campaign_pair requires stage_metric_enabled")
        round_labels = {round_spec.round_label for round_spec in self.rounds}
        unknown_stage_exclusions = set(self.stage_excluded_rounds) - round_labels
        if unknown_stage_exclusions:
            raise ValueError(
                f"{self.task_key}: stage_excluded_rounds contains unknown rounds "
                f"{sorted(unknown_stage_exclusions)}"
            )
        if self.headline_metrics_output_name is not None:
            output_name = self.headline_metrics_output_name
            if not output_name or Path(output_name).name != output_name:
                raise ValueError(f"invalid headline-metrics output_name={output_name!r}")
            if not self.stage_metric_enabled:
                raise ValueError(
                    f"{self.task_key}: headline_metrics_output_name requires stage_metric_enabled"
                )
        if self.headline_metrics_round_subsets and self.headline_metrics_output_name is None:
            raise ValueError(
                f"{self.task_key}: headline_metrics_round_subsets requires "
                "headline_metrics_output_name"
            )
        configured_rounds = tuple(round_spec.round_label for round_spec in self.rounds)
        for output_name, subset_rounds in self.headline_metrics_round_subsets.items():
            if not output_name or Path(output_name).name != output_name:
                raise ValueError(f"invalid headline-metrics subset output_name={output_name!r}")
            if output_name == self.headline_metrics_output_name:
                raise ValueError(
                    f"{self.task_key}: headline-metrics subset output {output_name!r} "
                    "duplicates the full output"
                )
            if not subset_rounds:
                raise ValueError(
                    f"{self.task_key}: headline-metrics subset {output_name!r} has no rounds"
                )
            if len(set(subset_rounds)) != len(subset_rounds):
                raise ValueError(
                    f"{self.task_key}: headline-metrics subset {output_name!r} has "
                    f"duplicate rounds {subset_rounds}"
                )
            unknown_rounds = set(subset_rounds) - set(configured_rounds)
            if unknown_rounds:
                raise ValueError(
                    f"{self.task_key}: headline-metrics subset {output_name!r} has "
                    f"unknown rounds {sorted(unknown_rounds)}"
                )
            expected_order = tuple(label for label in configured_rounds if label in subset_rounds)
            if subset_rounds != expected_order:
                raise ValueError(
                    f"{self.task_key}: headline-metrics subset {output_name!r} must "
                    f"follow config round order; expected {expected_order}, got {subset_rounds}"
                )
            stage_rounds = set(subset_rounds) - set(self.stage_excluded_rounds)
            if not stage_rounds:
                raise ValueError(
                    f"{self.task_key}: headline-metrics subset {output_name!r} has no "
                    "rounds with stage labels"
                )
        if self.stage_metric_enabled:
            missing = [
                f"{round_spec.round_label}/{arm}"
                for round_spec in self.rounds
                if round_spec.round_label not in self.stage_excluded_rounds
                for arm, arm_spec in round_spec.policy_arms.items()
                if arm_spec.stage_label_source is None
            ]
            if missing:
                raise ValueError(
                    f"{self.task_key}: stage_metric_enabled but these arms have no "
                    f"stage_label_source: {missing}"
                )
        registered_round_dirs = {Path(round_spec.eval_dir).resolve() for round_spec in self.rounds}
        for round_spec in self.rounds:
            for arm in round_spec.policy_arms.values():
                for source in arm.pooled_sources or ():
                    registered_round_dirs.add(Path(source.eval_dir).resolve())
        auxiliary_dirs = [Path(path).resolve() for path in self.auxiliary_eval_dirs]
        duplicates = sorted({path for path in auxiliary_dirs if auxiliary_dirs.count(path) > 1})
        if duplicates:
            raise ValueError(
                f"{self.task_key}: duplicate auxiliary_eval_dirs: "
                f"{[str(path) for path in duplicates]}"
            )
        overlap = sorted(set(auxiliary_dirs) & registered_round_dirs)
        if overlap:
            raise ValueError(
                f"{self.task_key}: auxiliary eval dirs overlap primary round dirs: "
                f"{[str(path) for path in overlap]}"
            )


@dataclass(frozen=True)
class SuiteLineHeadlineSpec:
    """One line panel in a suite-level headline plot.

    ``round_groups`` compresses the panel's x-axis: each inner tuple of round
    labels is pooled into a single point (successes and n summed across the
    blocks, Wilson CI recomputed on the pooled counts; an arm present in only
    some of a group's rounds pools over the rounds it has). A multi-round group
    is ticked "<first>-<last>". None keeps one x-position per config round.

    ``exclude_rounds`` names config rounds deliberately left off the panel.
    Every config round must be either grouped or excluded — a new round added
    to the line config fails loud here instead of silently dropping off the
    suite figure.
    """

    cfg: LineHeadlineConfig
    panel_title: str
    round_groups: tuple[tuple[str, ...], ...] | None = None
    exclude_rounds: tuple[str, ...] = ()
    # Two-clip progress = mean episode score / 2, with episode-level SEM.
    task_progress: bool = False

    def __post_init__(self) -> None:
        all_rounds = _eval_round_order(self.cfg)
        groups = self.round_groups
        if groups is None:
            groups = tuple((label,) for label in all_rounds if label not in self.exclude_rounds)
        flat = [label for group in groups for label in group]
        if len(set(flat)) != len(flat):
            raise ValueError(f"{self.cfg.task_key}: duplicate rounds across round_groups")
        overlap = set(flat) & set(self.exclude_rounds)
        if overlap:
            raise ValueError(
                f"{self.cfg.task_key}: rounds {sorted(overlap)} are both grouped and excluded"
            )
        covered = set(flat) | set(self.exclude_rounds)
        if covered != set(all_rounds):
            raise ValueError(
                f"{self.cfg.task_key}: round_groups + exclude_rounds must cover the config "
                f"rounds exactly; missing={sorted(set(all_rounds) - covered)}, "
                f"unknown={sorted(covered - set(all_rounds))}"
            )
        if tuple(flat) != tuple(label for label in all_rounds if label in set(flat)):
            raise ValueError(
                f"{self.cfg.task_key}: round_groups must follow the config round order; got {flat}"
            )

    def panel_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Ordered (tick label, member rounds) for this panel's x-axis."""
        if self.round_groups is None:
            return tuple(
                (_compact_label(_sr_tick_text(self.cfg, label)), (label,))
                for label in _eval_round_order(self.cfg)
                if label not in self.exclude_rounds
            )
        out = []
        for group in self.round_groups:
            if len(group) == 1:
                out.append((_compact_label(_sr_tick_text(self.cfg, group[0])), group))
            else:
                out.append((f"{group[0]}-{group[-1]}", group))
        return tuple(out)


def _round_sort_key(round_label: str) -> tuple[int, str]:
    if round_label.startswith("R") and round_label[1:].isdigit():
        return int(round_label[1:]), round_label
    return 10_000, round_label


def _eval_round_order(cfg: LineHeadlineConfig) -> tuple[str, ...]:
    return tuple(round_spec.round_label for round_spec in cfg.rounds)


def _round_x(round_labels: tuple[str, ...]) -> dict[str, float]:
    return {label: float(idx) for idx, label in enumerate(round_labels)}


def _round_axis_break_fracs(
    tick_labels: tuple[str, ...],
    x_by_round: Mapping[str, float],
    xlim: tuple[float, float],
) -> list[float]:
    """Axes-fraction midpoints of numeric gaps between adjacent round ticks.

    Adjacent plain ``R<k>`` ticks whose round numbers jump by more than one
    mark deliberately omitted rounds (a paper spec's ``exclude_rounds``, e.g.
    Route Cable's R0 -> R4); the caller draws
    :func:`mulligan.plotting.paper.x_axis_break` at each returned fraction once
    layout is final.
    """
    span = xlim[1] - xlim[0]
    fracs: list[float] = []
    for left, right in zip(tick_labels, tick_labels[1:]):
        nums = [
            int(label[1:]) if label.startswith("R") and label[1:].isdigit() else None
            for label in (left, right)
        ]
        if None in nums or nums[1] - nums[0] <= 1:
            continue
        mid = 0.5 * (x_by_round[left] + x_by_round[right])
        fracs.append((mid - xlim[0]) / span)
    return fracs


def _sort_by_round_axis(
    df: pd.DataFrame, round_x: Mapping[str, float], *, source: str
) -> pd.DataFrame:
    out = df.copy()
    out["_round_axis"] = out["round"].astype(str).map(round_x)
    if out["_round_axis"].isna().any():
        missing = sorted(out.loc[out["_round_axis"].isna(), "round"].astype(str).unique())
        raise RuntimeError(f"{source}: round labels missing from configured axis: {missing}")
    return out.sort_values("_round_axis").drop(columns="_round_axis")


def _compact_label(text: str) -> str:
    return (
        text.replace("source eval", "source")
        .replace("complete eval", "complete")
        .replace("final eval", "final")
    )


def _sr_tick_text(cfg: LineHeadlineConfig, label: str) -> str:
    """SR-surface tick text for one round, with an optional headline override.

    The SR headline (and the suite panel it feeds) can relabel a round without
    disturbing the substage/efficiency ticks that keep pointing at their own
    single blocks (e.g. a pooled final round reads differently on the SR axis
    than the complete-eval-only efficiency/substage axes).
    """
    if cfg.sr_tick_labels is not None and label in cfg.sr_tick_labels:
        return cfg.sr_tick_labels[label]
    return cfg.round_tick_labels[label]


def _data_path(cfg: LineHeadlineConfig, name: str) -> Path:
    return cfg.out_data_dir / f"{cfg.task_key}_{name}.csv"


def _read_policy_summary(path: Path) -> dict[str, dict[str, object]]:
    rows = pd.read_csv(path).to_dict("records")
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        policy_name = str(row["policy_name"])
        if policy_name in out:
            raise RuntimeError(f"{path}: duplicate policy_name={policy_name!r}")
        out[policy_name] = row
    if not out:
        raise RuntimeError(f"{path}: no policy_summary rows")
    return out


def _policy_row(
    summary: Mapping[str, dict[str, object]],
    policy_name: str,
    *,
    source: Path,
) -> dict[str, object]:
    try:
        return summary[policy_name]
    except KeyError as exc:
        raise KeyError(
            f"{source}: missing policy_name={policy_name!r}; available={sorted(summary)}"
        ) from exc


def _arm_label(arm: str, *, headline: bool, overrides: Mapping[str, str] | None = None) -> str:
    if overrides and arm in overrides:
        return overrides[arm]
    style = ARM_STYLES[arm]
    if headline and "headline_label" in style:
        return str(style["headline_label"])
    return str(style["label"])


def _arm_sr_sources(
    round_spec: RoundHeadlineSpec, arm_spec: PolicyArmSpec
) -> tuple[ArmSourceSpec, ...]:
    """The eval blocks contributing to one arm's held-out success rate.

    Single-source arms (the common case) resolve to the round's own eval dir;
    pooled arms return their explicit block list.
    """
    if arm_spec.pooled_sources is not None:
        if not arm_spec.pooled_sources:
            raise RuntimeError(
                f"{round_spec.round_label} {arm_spec.policy_name}: empty pooled_sources"
            )
        return arm_spec.pooled_sources
    return (
        ArmSourceSpec(
            eval_dir=round_spec.eval_dir,
            policy_name=arm_spec.policy_name,
            paired_prefix=arm_spec.paired_prefix,
            block_label=round_spec.round_label,
        ),
    )


# Every active D2 task's physical initial-state coordinates plus manifest_idx.
# The key MUST include each task's real object-pose columns: manifest_idx alone
# is only unique WITHIN a manifest, so two independently-seeded blocks reuse
# idx 0..N and would false-overlap on manifest_idx. Square blocks carry
# nut/peg columns, marker blocks carry pen/holder columns, Route Cable blocks
# carry rope/clip-left/clip-right columns; the absent family drops out of the
# key per block, so the fingerprint stays task-correct.
_STATE_COORD_COLS = (
    "nut_x",
    "nut_y",
    "nut_yaw",
    "peg_x",
    "peg_y",
    "pen_x",
    "pen_y",
    "pen_yaw",
    "holder_x",
    "holder_y",
    "rope_x",
    "clip_left_x",
    "clip_left_y",
    "clip_left_yaw",
    "clip_right_x",
    "clip_right_y",
    "clip_right_yaw",
    "manifest_idx",
)


def _verify_pooled_coord_sets(
    context: str,
    coord_sets: list[tuple[str, set[tuple[float, ...]]]],
    *,
    same_start_repeat: bool,
) -> None:
    """Cross-block initial-state contract for one pooled arm.

    Default: pooled blocks must draw DISJOINT starts (pooling overlapping
    blocks double-counts). With ``same_start_repeat`` the contract inverts:
    every block must draw EXACTLY the same start set (complete repeated eval
    sets of one shared manifest) — a partial or wrong-manifest repeat block
    still fails loud.
    """
    for i in range(len(coord_sets)):
        for j in range(i + 1, len(coord_sets)):
            label_i, keys_i = coord_sets[i]
            label_j, keys_j = coord_sets[j]
            if same_start_repeat:
                if keys_i != keys_j:
                    raise RuntimeError(
                        f"{context}: same-start repeat blocks {label_i!r} and {label_j!r} do "
                        f"NOT share an identical start set ({len(keys_i)} vs {len(keys_j)} "
                        f"keys, {len(keys_i & keys_j)} shared); repeat pooling requires "
                        "complete same-manifest sets"
                    )
            else:
                overlap = keys_i & keys_j
                if overlap:
                    raise RuntimeError(
                        f"{context}: pooled blocks {label_i!r} and {label_j!r} share "
                        f"{len(overlap)} initial states; pooling would double-count"
                    )


def _state_coord_set(eval_dir: Path) -> set[tuple[float, ...]]:
    """Initial-state fingerprints for one eval block, from its paired outcomes.

    The paired-outcome coordinates are validated against each block's pinned
    initial-state manifest at ingest time, so they are the uniform, manifest-
    backed key for a cross-block disjointness check. Each task contributes only
    its own object-pose columns (square: nut/peg; marker: pen/holder); the
    missing family simply drops out of the key.
    """
    paired = pd.read_csv(eval_dir / "paired_round_outcomes.csv")
    cols = [col for col in _STATE_COORD_COLS if col in paired.columns]
    if not cols:
        raise RuntimeError(
            f"{eval_dir}/paired_round_outcomes.csv: no initial-state coordinate columns"
        )
    keys: set[tuple[float, ...]] = set()
    for rec in paired[cols].to_dict("records"):
        keys.add(tuple(round(float(rec[col]), 6) for col in cols))
    if len(keys) != len(paired):
        raise RuntimeError(
            f"{eval_dir}/paired_round_outcomes.csv: non-unique initial-state keys "
            f"({len(keys)} unique of {len(paired)} rows)"
        )
    return keys


def build_success_rate_table(
    cfg: LineHeadlineConfig,
    *,
    arms: tuple[str, ...],
    headline_labels: bool,
    metric: str | None = None,
) -> pd.DataFrame:
    """Per-(round, arm) rate table for ``metric`` (default: the line's headline).

    ``metric`` overrides ``cfg.headline_metric`` so a companion surface can read
    a second metric from the same policy summaries (e.g. Route Cable's raw
    full-success rate next to its task-progress headline).
    """
    if metric is None:
        metric = cfg.headline_metric
    if metric not in HEADLINE_METRICS:
        raise ValueError(
            f"{cfg.task_key}: unknown metric {metric!r}; expected one of {sorted(HEADLINE_METRICS)}"
        )
    rows: list[dict[str, object]] = []
    for round_spec in cfg.rounds:
        for arm in cfg.arm_order:
            if arm not in arms or arm not in round_spec.policy_arms:
                continue
            arm_spec = round_spec.policy_arms[arm]
            sources = _arm_sr_sources(round_spec, arm_spec)

            per_block: list[dict[str, object]] = []
            model_ids: set[str] = set()
            coord_sets: list[tuple[str, set[tuple[float, ...]]]] = []
            for src in sources:
                summary_path = src.eval_dir / "policy_summary.csv"
                summary = _read_policy_summary(summary_path)
                row = _policy_row(summary, src.policy_name, source=summary_path)
                successes, n = _metric_numerator_denominator(row, metric)
                if n <= 0:
                    raise RuntimeError(f"{summary_path}: {src.policy_name} has non-positive n={n}")
                model_ids.add(str(row["model_id"]))
                coord_sets.append((src.block_label, _state_coord_set(src.eval_dir)))
                per_block.append(
                    {
                        "block_label": src.block_label,
                        "policy_name": src.policy_name,
                        "policy_label": row["policy_label"],
                        "model_id": str(row["model_id"]),
                        "successes": successes,
                        "n": n,
                    }
                )

            # (a) pooled sources must be the same frozen checkpoint.
            if len(model_ids) != 1:
                raise RuntimeError(
                    f"{cfg.task_key} {round_spec.round_label} {arm}: pooled blocks are NOT the "
                    f"same checkpoint; model_ids={sorted(model_ids)}"
                )
            # (b) pooled blocks: disjoint starts, or identical starts when the
            # arm is declared a same-start repeat pool.
            _verify_pooled_coord_sets(
                f"{cfg.task_key} {round_spec.round_label} {arm}",
                coord_sets,
                same_start_repeat=arm_spec.same_start_repeat_pooling,
            )

            model_id = next(iter(model_ids))
            block_labels = [str(b["block_label"]) for b in per_block]
            blocks_provenance = "; ".join(block_labels)

            # (c) per-block rows accompany the pooled row (only when >1 block).
            if len(per_block) > 1:
                for blk in per_block:
                    lo, hi = wilson_ci(int(blk["successes"]), int(blk["n"]))
                    rows.append(
                        {
                            "round": round_spec.round_label,
                            "arm": arm,
                            "arm_label": _arm_label(
                                arm, headline=headline_labels, overrides=cfg.arm_label_overrides
                            ),
                            "block": blk["block_label"],
                            "is_pooled": False,
                            "n_blocks": 1,
                            "blocks": blk["block_label"],
                            "source_policy_name": blk["policy_name"],
                            "source_policy_label": blk["policy_label"],
                            "model_id": blk["model_id"],
                            "successes": int(blk["successes"]),
                            "n": int(blk["n"]),
                            "success_rate": int(blk["successes"]) / int(blk["n"]),
                            "wilson_lo": lo,
                            "wilson_hi": hi,
                            "provenance": arm_spec.sr_provenance,
                        }
                    )

            # (d) pooled row with Wilson CI on the pooled n.
            pooled_k = sum(int(b["successes"]) for b in per_block)
            pooled_n = sum(int(b["n"]) for b in per_block)
            lo, hi = wilson_ci(pooled_k, pooled_n)
            episode_interval = {}
            if cfg.episode_score_intervals and metric == "task_progress":
                if len(sources) != 1:
                    raise ValueError(
                        "Episode-score intervals require one paired eval block per point"
                    )
                src = sources[0]
                paired = pd.read_csv(src.eval_dir / "paired_round_outcomes.csv")
                scores = paired[f"{src.paired_prefix}_score"].to_numpy(dtype=float) / 2.0
                if (
                    len(scores) < 2
                    or not np.isfinite(scores).all()
                    or not np.isin(scores, (0.0, 0.5, 1.0)).all()
                    or len(scores) * 2 != pooled_n
                    or not np.isclose(scores.sum() * 2, pooled_k)
                ):
                    raise ValueError("Paired rollout scores disagree with graded policy summary")
                _, lo, hi = bootstrap_mean_ci(scores)
                episode_interval = {
                    "interval_method": "episode_percentile_bootstrap_95",
                    "episode_n": len(scores),
                    "episode_sem": float(scores.std(ddof=1) / np.sqrt(len(scores))),
                }
            rows.append(
                {
                    "round": round_spec.round_label,
                    "arm": arm,
                    "arm_label": _arm_label(
                        arm, headline=headline_labels, overrides=cfg.arm_label_overrides
                    ),
                    "block": "pooled",
                    "is_pooled": True,
                    "n_blocks": len(per_block),
                    "blocks": blocks_provenance,
                    "source_policy_name": arm_spec.policy_name,
                    "source_policy_label": per_block[0]["policy_label"],
                    "model_id": model_id,
                    "successes": pooled_k,
                    "n": pooled_n,
                    "success_rate": pooled_k / pooled_n,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                    "provenance": arm_spec.sr_provenance,
                    **episode_interval,
                }
            )
    if not rows:
        raise RuntimeError(f"{cfg.task_key}: no success-rate rows built")
    return pd.DataFrame(rows)


def build_paired_campaign_table(
    cfg: LineHeadlineConfig,
    *,
    baseline_arm: str,
    treatment_arm: str,
    n_boot: int = 20_000,
    bootstrap_seed: int = 20260704,
) -> pd.DataFrame:
    """Campaign-pooled paired comparison of two arms across every headline round.

    For each round, the two arms' eval blocks are matched by eval dir; a block
    joins only when BOTH arms rolled out from its paired starts (e.g. the
    square R3 held-out block A carries no baseline arm and drops out).
    Emits one row per paired block plus a pooled row: the pooled exact McNemar
    runs on the summed discordant pairs and the bootstrap CI resamples pairs
    within each block (stratified). Each round compares that round's own
    actors — the pooled row reads as a campaign-level treatment effect over
    the line's lifetime, not a single-checkpoint comparison; per-block rows
    remain the confirmatory quantities.
    """
    block_rows: list[dict[str, object]] = []
    strata: list[tuple[np.ndarray, np.ndarray]] = []
    for round_spec in cfg.rounds:
        if baseline_arm not in round_spec.policy_arms:
            continue
        if treatment_arm not in round_spec.policy_arms:
            continue
        base_by_dir = {
            src.eval_dir: src
            for src in _arm_sr_sources(round_spec, round_spec.policy_arms[baseline_arm])
        }
        for t_src in _arm_sr_sources(round_spec, round_spec.policy_arms[treatment_arm]):
            b_src = base_by_dir.get(t_src.eval_dir)
            if b_src is None:
                continue
            summary_source = t_src.eval_dir / "policy_summary.csv"
            paired_source = t_src.eval_dir / "paired_round_outcomes.csv"
            summary = _read_policy_summary(summary_source)
            paired = pd.read_csv(paired_source)
            arm_success: dict[str, np.ndarray] = {}
            model_ids: dict[str, str] = {}
            for label, src in (("baseline", b_src), ("treatment", t_src)):
                col = f"{src.paired_prefix}_success"
                if col not in paired.columns:
                    raise RuntimeError(f"{paired_source}: missing column {col!r}")
                success = paired[col].astype(bool).to_numpy()
                row = _policy_row(summary, src.policy_name, source=summary_source)
                if int(success.sum()) != int(row["successes"]):
                    raise RuntimeError(
                        f"{round_spec.round_label} [{t_src.block_label}] {label}: "
                        f"policy_summary successes={int(row['successes'])}, "
                        f"paired successes={int(success.sum())}"
                    )
                arm_success[label] = success
                model_ids[label] = str(row["model_id"])
            b_success = arm_success["baseline"]
            t_success = arm_success["treatment"]
            n = len(paired)
            t_only = int((t_success & ~b_success).sum())
            b_only = int((b_success & ~t_success).sum())
            strata.append((t_success, b_success))
            block_rows.append(
                {
                    "round": round_spec.round_label,
                    "block": t_src.block_label,
                    "is_pooled": False,
                    "n_blocks": 1,
                    "n": n,
                    "baseline_arm": baseline_arm,
                    "treatment_arm": treatment_arm,
                    "baseline_policy_name": b_src.policy_name,
                    "treatment_policy_name": t_src.policy_name,
                    "baseline_model_id": model_ids["baseline"],
                    "treatment_model_id": model_ids["treatment"],
                    "baseline_successes": int(b_success.sum()),
                    "treatment_successes": int(t_success.sum()),
                    "treatment_only": t_only,
                    "baseline_only": b_only,
                    "paired_delta": float((t_success.sum() - b_success.sum()) / n),
                    "mcnemar_exact_pvalue": exact_mcnemar_pvalue(t_only, b_only),
                    "paired_delta_bootstrap_ci95_lo": float("nan"),
                    "paired_delta_bootstrap_ci95_hi": float("nan"),
                }
            )
    if not block_rows:
        raise RuntimeError(
            f"{cfg.task_key}: no paired blocks carry both {baseline_arm!r} and {treatment_arm!r}"
        )
    pooled_t = sum(int(row["treatment_successes"]) for row in block_rows)
    pooled_b = sum(int(row["baseline_successes"]) for row in block_rows)
    pooled_t_only = sum(int(row["treatment_only"]) for row in block_rows)
    pooled_b_only = sum(int(row["baseline_only"]) for row in block_rows)
    pooled_n = sum(int(row["n"]) for row in block_rows)
    delta, ci_lo, ci_hi = bootstrap_paired_delta_stratified(
        strata, n_boot=n_boot, seed=bootstrap_seed
    )
    block_rows.append(
        {
            "round": "campaign",
            "block": "; ".join(f"{row['round']}: {row['block']}" for row in block_rows),
            "is_pooled": True,
            "n_blocks": len(block_rows),
            "n": pooled_n,
            "baseline_arm": baseline_arm,
            "treatment_arm": treatment_arm,
            "baseline_policy_name": "per-round actors (see block rows)",
            "treatment_policy_name": "per-round actors (see block rows)",
            "baseline_model_id": "per-round",
            "treatment_model_id": "per-round",
            "baseline_successes": pooled_b,
            "treatment_successes": pooled_t,
            "treatment_only": pooled_t_only,
            "baseline_only": pooled_b_only,
            "paired_delta": delta,
            "mcnemar_exact_pvalue": exact_mcnemar_pvalue(pooled_t_only, pooled_b_only),
            "paired_delta_bootstrap_ci95_lo": ci_lo,
            "paired_delta_bootstrap_ci95_hi": ci_hi,
        }
    )
    return pd.DataFrame(block_rows)


def _with_binomial_sem(df: pd.DataFrame, value_col: str, n_col: str) -> pd.DataFrame:
    out = df.copy()
    p = out[value_col].astype(float)
    n = out[n_col].astype(float)
    sem = (p * (1.0 - p) / n).pow(0.5)
    if "episode_sem" in out.columns:
        sem = out["episode_sem"].astype(float)
    out["sem_lo"] = (p - sem).clip(lower=0.0, upper=1.0)
    out["sem_hi"] = (p + sem).clip(lower=0.0, upper=1.0)
    return out


def _with_wilson_1se(df: pd.DataFrame, k_col: str = "successes", n_col: str = "n") -> pd.DataFrame:
    """Attach boundary-safe Wilson ``z=1`` bounds to binomial rows."""

    out = df.copy()
    bounds = [wilson_1se(int(k), int(n)) for k, n in zip(out[k_col], out[n_col], strict=True)]
    out["wilson_1se_lo"] = [lo for lo, _ in bounds]
    out["wilson_1se_hi"] = [hi for _, hi in bounds]
    if "episode_sem" in out.columns:
        out["wilson_1se_lo"] = (out["success_rate"] - out["episode_sem"]).clip(0.0, 1.0)
        out["wilson_1se_hi"] = (out["success_rate"] + out["episode_sem"]).clip(0.0, 1.0)
    return out


def _plot_arm_series(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    round_x: Mapping[str, float],
    y_col: str,
    lo_col: str,
    hi_col: str,
    arms: tuple[str, ...],
    all_arm_offsets: bool,
    label_overrides: Mapping[str, str] | None = None,
    inner_lo_col: str | None = None,
    inner_hi_col: str | None = None,
    markersize: float = 6.0,
    linewidth: float = 1.8,
) -> None:
    """Per-arm round series with CI whiskers.

    ``inner_lo_col``/``inner_hi_col`` switch on the combined two-whisker
    grammar shared by the binary and granular headline families: the ``lo_col``
    / ``hi_col`` interval draws as a thin capped outer 95% whisker, and the
    inner interval (±1 SE) as a thick capless bar on top of it.
    ``markersize``/``linewidth`` keep the full-size defaults; print-size paper
    variants pass the paper-style sizes.
    """
    if (inner_lo_col is None) != (inner_hi_col is None):
        raise ValueError("inner_lo_col and inner_hi_col must be set together")
    for arm in arms:
        if arm not in ARM_STYLES:
            raise KeyError(f"no ARM_STYLES entry for arm={arm!r}")
        style = ARM_STYLES[arm]
        sub = _sort_by_round_axis(
            df[df["arm"] == arm],
            round_x,
            source=f"{arm} arm series",
        )
        if sub.empty:
            continue
        dx_key = "all_dx" if all_arm_offsets else "headline_dx"
        x = [round_x[round_label] + float(style[dx_key]) for round_label in sub["round"]]
        y = [100.0 * value for value in sub[y_col]]
        # Clamp at 0: wilson_ci(0, n) puts the lower bound a float-epsilon
        # ABOVE p_hat=0, and matplotlib rejects the resulting -7e-18 yerr.
        yerr = [
            [
                max(0.0, 100.0 * (value - lo))
                for value, lo in zip(sub[y_col], sub[lo_col], strict=True)
            ],
            [
                max(0.0, 100.0 * (hi - value))
                for value, hi in zip(sub[y_col], sub[hi_col], strict=True)
            ],
        ]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=style["color"],
            marker=style["marker"],
            markersize=markersize,
            linewidth=linewidth if len(sub) > 1 else 0.0,
            linestyle="-" if len(sub) > 1 else "none",
            # elinewidth explicitly: it inherits linewidth, which is 0 for a
            # single-point arm and would leave the CI whisker invisible.
            elinewidth=1.2 if inner_lo_col is not None else linewidth,
            capsize=3,
            label=_arm_label(arm, headline=not all_arm_offsets, overrides=label_overrides),
            zorder=3,
        )
        if inner_lo_col is not None:
            inner_yerr = [
                [
                    max(0.0, 100.0 * (value - lo))
                    for value, lo in zip(sub[y_col], sub[inner_lo_col], strict=True)
                ],
                [
                    max(0.0, 100.0 * (hi - value))
                    for value, hi in zip(sub[y_col], sub[inner_hi_col], strict=True)
                ],
            ]
            ax.errorbar(
                x,
                y,
                yerr=inner_yerr,
                fmt="none",
                ecolor=style["color"],
                elinewidth=4.2,
                capsize=0,
                alpha=0.85,
                zorder=4,
            )


def _arm_round_episode_data(
    cfg: LineHeadlineConfig, round_spec: RoundHeadlineSpec, arm: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode (success, num_steps) arrays for one arm at one round.

    Concatenates the arm's SR source blocks with the same validation contract as
    the headline SR pooling: per-block episode/success counts must reconcile
    with policy_summary, pooled blocks must be the same frozen checkpoint, and
    pooled blocks must draw disjoint initial states.
    """
    arm_spec = round_spec.policy_arms[arm]
    sources = _arm_sr_sources(round_spec, arm_spec)

    model_ids: set[str] = set()
    coord_sets: list[tuple[str, set[tuple[float, ...]]]] = []
    success_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    for src in sources:
        summary_source = src.eval_dir / "policy_summary.csv"
        paired_source = src.eval_dir / "paired_round_outcomes.csv"
        summary = _read_policy_summary(summary_source)
        paired = pd.read_csv(paired_source)
        row = _policy_row(summary, src.policy_name, source=summary_source)
        success_col = f"{src.paired_prefix}_success"
        step_col = f"{src.paired_prefix}_num_steps"
        missing = [col for col in (success_col, step_col) if col not in paired.columns]
        if missing:
            raise RuntimeError(f"{paired_source}: missing columns for {arm}: {missing}")
        block_successes = paired[success_col].astype(bool).to_numpy()
        block_steps = paired[step_col].astype(float).to_numpy()
        block_n = int(row["episodes"])
        block_k = int(row["successes"])
        if len(block_steps) != block_n:
            raise RuntimeError(
                f"{summary_source}: {src.policy_name} episodes={block_n}, "
                f"but {paired_source} has {len(block_steps)} rows"
            )
        if int(block_successes.sum()) != block_k:
            raise RuntimeError(
                f"{round_spec.round_label} {arm} [{src.block_label}]: policy_summary "
                f"successes={block_k}, paired successes={int(block_successes.sum())}"
            )
        block_success_steps = block_steps[block_successes]
        if len(block_success_steps):
            block_mean = float(block_success_steps.mean())
            summary_mean = float(row["mean_success_steps"])
            if not math.isclose(block_mean, summary_mean, rel_tol=0.0, abs_tol=5e-4):
                raise RuntimeError(
                    f"{round_spec.round_label} {arm} [{src.block_label}]: paired mean "
                    f"success steps {block_mean:.6f} != policy_summary {summary_mean:.6f}"
                )
        model_ids.add(str(row["model_id"]))
        coord_sets.append((src.block_label, _state_coord_set(src.eval_dir)))
        success_parts.append(block_successes)
        step_parts.append(block_steps)

    # Same pooling contract as build_success_rate_table:
    # (a) pooled sources must be the same frozen checkpoint;
    if len(model_ids) != 1:
        raise RuntimeError(
            f"{cfg.task_key} {round_spec.round_label} {arm}: pooled blocks are NOT the "
            f"same checkpoint; model_ids={sorted(model_ids)}"
        )
    # (b) pooled blocks: disjoint starts, or identical starts when the arm is
    # declared a same-start repeat pool.
    _verify_pooled_coord_sets(
        f"{cfg.task_key} {round_spec.round_label} {arm}",
        coord_sets,
        same_start_repeat=arm_spec.same_start_repeat_pooling,
    )

    return np.concatenate(success_parts), np.concatenate(step_parts)


def _arm_round_unit_data(
    cfg: LineHeadlineConfig, round_spec: RoundHeadlineSpec, arm: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-episode task units + per-unit completion-time increments (graded line).

    A graded line's task unit is one completed sub-goal (Route Cable: one seated
    clip; full success = 2 units). Returns ``(units, steps, unit_steps)``:
    per-episode unit counts, per-episode edited lengths, and one entry per
    completed unit = steps between consecutive unit completions within its
    episode (sub-goal k at its reviewed subtask-mark frame, the terminal unit at
    the edited episode end). Time after the last completed unit — e.g. failing
    at the second clip until timeout — contributes episode time but no unit, so
    unit times measure speed while progress is being made.

    Requires the round's ``subtask_frames.csv`` sidecar (exported from the
    reviewed outcome-edit record) and cross-checks per-episode mark counts and
    the summed units against policy_summary. Single-source arms only.
    """
    arm_spec = round_spec.policy_arms[arm]
    sources = _arm_sr_sources(round_spec, arm_spec)
    if len(sources) != 1:
        raise RuntimeError(
            f"{cfg.task_key} {round_spec.round_label} {arm}: graded unit data supports "
            f"single-source arms only, got {len(sources)} pooled blocks"
        )
    src = sources[0]
    paired_path = src.eval_dir / "paired_round_outcomes.csv"
    paired = pd.read_csv(paired_path)
    frames_path = src.eval_dir / "subtask_frames.csv"
    if not frames_path.is_file():
        raise FileNotFoundError(
            f"{frames_path}: subtask-frame sidecar missing; run the line's subtask-frame "
            "export script before building suite unit metrics"
        )
    frames_by_ep: dict[int, list[int]] = {}
    for frame_row in pd.read_csv(frames_path).to_dict("records"):
        frames_by_ep.setdefault(int(frame_row["episode_index"]), []).append(int(frame_row["frame"]))

    prefix = src.paired_prefix
    cols = {
        "score": f"{prefix}_score",
        "steps": f"{prefix}_num_steps",
        "episode": f"{prefix}_episode_index",
        "success": f"{prefix}_success",
    }
    missing = [col for col in cols.values() if col not in paired.columns]
    if missing:
        raise RuntimeError(f"{paired_path}: missing columns for {arm}: {missing}")

    # Partial outcome-editor passes leave later paired rounds intentionally
    # ungraded. Progress/unit statistics use only reviewed-score rows; binary
    # outcome and efficiency surfaces continue to use the full paired file.
    paired = paired[pd.to_numeric(paired[cols["score"]], errors="raise").notna()].copy()
    if paired.empty:
        raise RuntimeError(f"{paired_path}: no reviewed graded rows for {arm}")

    units: list[int] = []
    steps: list[float] = []
    unit_steps: list[float] = []
    for row in paired.to_dict("records"):
        ep = int(row[cols["episode"]])
        score = int(row[cols["score"]])
        n_steps = float(row[cols["steps"]])
        success = bool(row[cols["success"]])
        marks = sorted(frames_by_ep.get(ep, []))
        expected_marks = score - (1 if success else 0)
        if len(marks) != expected_marks:
            raise RuntimeError(
                f"{cfg.task_key} {round_spec.round_label} {arm} ep{ep}: score={score} "
                f"success={success} implies {expected_marks} subtask marks, sidecar has "
                f"{len(marks)}"
            )
        completion_times = [float(frame + 1) for frame in marks]
        if success:
            completion_times.append(n_steps)
        increments = np.diff([0.0, *completion_times])
        if np.any(increments <= 0):
            raise RuntimeError(
                f"{cfg.task_key} {round_spec.round_label} {arm} ep{ep}: non-increasing unit "
                f"completion times {completion_times} (episode length {n_steps})"
            )
        units.append(score)
        steps.append(n_steps)
        unit_steps.extend(increments.tolist())

    summary_source = src.eval_dir / "policy_summary.csv"
    summary_row = _policy_row(
        _read_policy_summary(summary_source), src.policy_name, source=summary_source
    )
    expected_units = int(summary_row["score_1"]) + 2 * int(summary_row["score_2"])
    if int(sum(units)) != expected_units:
        raise RuntimeError(
            f"{cfg.task_key} {round_spec.round_label} {arm}: paired unit total {sum(units)} "
            f"!= policy_summary clip-seats {expected_units}"
        )
    return np.asarray(units, dtype=int), np.asarray(steps, dtype=float), np.asarray(unit_steps)


def build_efficiency_table(cfg: LineHeadlineConfig) -> pd.DataFrame:
    """Per-round successful-episode length and success throughput.

    Pooled arms concatenate per-episode success/step data across the same
    block list their headline SR pools (validated same-checkpoint and
    initial-state disjoint), so the efficiency surface shares the SR round
    axis: a rerank arm's blocks fold into the single pooled final-round
    x-position instead of a dedicated block tick.
    """
    rows: list[dict[str, object]] = []
    seed = 20260702
    arms = cfg.headline_arms
    for round_idx, round_spec in enumerate(cfg.rounds):
        for arm_idx, arm in enumerate(arms):
            if arm not in round_spec.policy_arms:
                continue
            arm_spec = round_spec.policy_arms[arm]
            sources = _arm_sr_sources(round_spec, arm_spec)
            successes, steps = _arm_round_episode_data(cfg, round_spec, arm)
            n = int(len(steps))
            k = int(successes.sum())
            success_steps = steps[successes]
            if len(success_steps) == 0:
                mean_success_steps = float("nan")
                mean_success_steps_se = float("nan")
            else:
                mean_success_steps = float(success_steps.mean())
                mean_success_steps_se = (
                    float(success_steps.std(ddof=1) / math.sqrt(len(success_steps)))
                    if len(success_steps) > 1
                    else float("nan")
                )
            total_steps = float(steps.sum())
            if total_steps <= 0:
                raise RuntimeError(
                    f"{round_spec.round_label} {arm}: total step count must be positive"
                )
            total_success_steps = float(success_steps.sum())
            total_non_success_steps = total_steps - total_success_steps
            throughput = k / (total_steps / EVAL_FPS / 60.0)
            lo, hi = _bootstrap_throughput_ci(
                successes,
                steps,
                seed=seed + round_idx * 17 + arm_idx,
            )
            rows.append(
                {
                    "round": round_spec.round_label,
                    "arm": arm,
                    "arm_label": _arm_label(arm, headline=True, overrides=cfg.arm_label_overrides),
                    "source_policy_name": arm_spec.policy_name,
                    "paired_prefix": "; ".join(dict.fromkeys(src.paired_prefix for src in sources)),
                    "n_blocks": len(sources),
                    "blocks": "; ".join(src.block_label for src in sources),
                    "successes": k,
                    "n": n,
                    "success_rate": k / n,
                    "total_steps": int(total_steps),
                    "total_success_steps": int(total_success_steps),
                    "total_non_success_steps": int(total_non_success_steps),
                    "total_duration_min": total_steps / EVAL_FPS / 60.0,
                    "mean_all_episode_steps": total_steps / n,
                    "mean_success_steps": mean_success_steps,
                    "mean_success_steps_se": mean_success_steps_se,
                    "mean_success_duration_s": mean_success_steps / EVAL_FPS,
                    "throughput_denominator": "all_edited_episode_steps",
                    "success_throughput_per_min": throughput,
                    "success_throughput_ci95_lo": lo,
                    "success_throughput_ci95_hi": hi,
                    "provenance": arm_spec.sr_provenance,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_throughput_ci(
    successes: np.ndarray,
    steps: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float]:
    if len(successes) != len(steps):
        raise RuntimeError(f"success/step length mismatch: {len(successes)} vs {len(steps)}")
    if len(steps) == 0:
        raise RuntimeError("cannot compute throughput from zero eval episodes")
    if np.any(steps <= 0):
        raise RuntimeError("all eval episode step counts must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(steps), size=(THROUGHPUT_BOOTSTRAPS, len(steps)))
    boot_steps = steps[idx].sum(axis=1)
    boot_successes = successes[idx].sum(axis=1)
    boot = boot_successes / (boot_steps / EVAL_FPS / 60.0)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def _bootstrap_throughput_se(
    successes: np.ndarray,
    steps: np.ndarray,
    *,
    seed: int,
) -> float:
    """Bootstrap standard error of the completed-units/time ratio."""

    if len(successes) != len(steps):
        raise RuntimeError(f"success/step length mismatch: {len(successes)} vs {len(steps)}")
    if len(steps) == 0:
        raise RuntimeError("cannot compute throughput from zero eval episodes")
    if np.any(steps <= 0):
        raise RuntimeError("all eval episode step counts must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(steps), size=(THROUGHPUT_BOOTSTRAPS, len(steps)))
    boot_steps = steps[idx].sum(axis=1)
    boot_successes = successes[idx].sum(axis=1)
    boot = boot_successes / (boot_steps / EVAL_FPS / 60.0)
    return float(boot.std(ddof=1))


def _load_rung_index(path: Path, cache: dict[Path, pd.DataFrame], *, task_key: str) -> pd.DataFrame:
    if path in cache:
        return cache[path]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {task_key} stage-label battery file: {path}. "
            "Run the per-round publish_plot stage-label scripts first."
        )
    table = pd.read_csv(path)
    required = {"arm", "rung", "k", "n", "rate", "lo", "hi"}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    indexed = table.set_index(["arm", "rung"])
    cache[path] = indexed
    return indexed


def build_substage_table(
    cfg: LineHeadlineConfig,
    *,
    arms: tuple[str, ...],
    headline_labels: bool,
) -> pd.DataFrame:
    """Per-round G/I/SR substage completion rungs.

    Pooled arms sum rung numerators/denominators across their stage-labeled
    blocks (the same block list the headline SR pools, minus any block without
    a published stage battery), with Wilson CIs on the pooled counts — the same
    convention as the SR and efficiency surfaces.
    """
    rows: list[dict[str, object]] = []
    cache: dict[Path, pd.DataFrame] = {}
    for round_spec in cfg.rounds:
        for arm in cfg.arm_order:
            if arm not in arms or arm not in round_spec.policy_arms:
                continue
            arm_spec = round_spec.policy_arms[arm]
            if arm_spec.pooled_sources is not None:
                stage_sources = [
                    (src.stage_rung_csv, src.stage_arm, src.block_label)
                    for src in arm_spec.pooled_sources
                    if src.stage_rung_csv is not None
                ]
                # A pooled arm with no stage-labeled block opts out entirely.
                if not stage_sources:
                    continue
            else:
                # A headline-only arm with no stage battery opts out, as does a
                # round without one (stage_rung_csv=None), without failing the build.
                if arm_spec.stage_source_arm is None or round_spec.stage_rung_csv is None:
                    continue
                stage_sources = [
                    (
                        round_spec.stage_rung_csv,
                        arm_spec.stage_source_arm,
                        round_spec.round_label,
                    )
                ]

            rung_counts: dict[str, tuple[int, int]] = {}
            for rung in ("G", "I", "SR"):
                k_total = 0
                n_total = 0
                for stage_csv, stage_arm, block_label in stage_sources:
                    indexed = _load_rung_index(stage_csv, cache, task_key=cfg.task_key)
                    try:
                        row = indexed.loc[(stage_arm, rung)]
                    except KeyError as exc:
                        raise KeyError(
                            f"{stage_csv}: missing arm={stage_arm!r}, rung={rung!r} "
                            f"(block {block_label!r})"
                        ) from exc
                    k_total += int(row["k"])
                    n_total += int(row["n"])
                rung_counts[rung] = (k_total, n_total)

            # Conditional-rung consistency: I is P(insert | G), so its pooled
            # denominator must equal the pooled G numerator.
            if rung_counts["I"][1] != rung_counts["G"][0]:
                raise RuntimeError(
                    f"{cfg.task_key} {round_spec.round_label} {arm}: pooled I denominator "
                    f"{rung_counts['I'][1]} != pooled G numerator {rung_counts['G'][0]}"
                )

            blocks = "; ".join(block_label for _, _, block_label in stage_sources)
            source_arms = "; ".join(dict.fromkeys(stage_arm for _, stage_arm, _ in stage_sources))
            source_csvs = "; ".join(
                dict.fromkeys(_display_path(stage_csv) for stage_csv, _, _ in stage_sources)
            )
            for rung in ("G", "I", "SR"):
                k, n = rung_counts[rung]
                lo, hi = wilson_ci(k, n)
                rows.append(
                    {
                        "round": round_spec.round_label,
                        "arm": arm,
                        "arm_label": _arm_label(
                            arm, headline=headline_labels, overrides=cfg.arm_label_overrides
                        ),
                        "source_arm": source_arms,
                        "source_csv": source_csvs,
                        "n_blocks": len(stage_sources),
                        "blocks": blocks,
                        "rung": rung,
                        "k": k,
                        "n": n,
                        "rate": k / n,
                        "wilson_lo": lo,
                        "wilson_hi": hi,
                        "provenance": arm_spec.stage_provenance,
                    }
                )
    return pd.DataFrame(rows)


SUBSTAGE_PANELS: tuple[tuple[str, str], ...] = (
    ("G", "G: grasp + transport\nP(S>=3)"),
    ("I", "I: insert/seat after G\nP(S7 | S>=3)"),
    ("SR", "SR: success\nP(S7)"),
)


COLLECTION_ARM_STYLES: dict[str, dict[str, object]] = {
    "baseline_uniform": {
        "label": "Baseline collection",
        "color": METHOD_COLORS["uniform"],
        "marker": "o",
        "dx": -0.06,
        "annotate_xy": (-10, -14),
        "ha": "right",
    },
    "mulligan_sobol": {
        # Same blue as the Ours source/no-CF arm identity on the eval headline
        # plots ("sobol" == "real_mulligan_no_cf" hex), so the collection arm reads
        # as the same lineage across the surfaces.
        "color": METHOD_COLORS["sobol"],
        "label": "Ours collection",
        "marker": "s",
        "dx": 0.06,
        "annotate_xy": (10, 10),
        "ha": "left",
    },
}


def build_dagger_collection_success_table(cfg: LineHeadlineConfig) -> pd.DataFrame:
    """Per-round, per-collection-arm DAgger parent success rates on fresh episodes.

    One row per (round, collection arm). Counterfactual replay rows are excluded
    entirely: successes are the credited FRESH source-ledger rows of the arm, and
    the denominator adds that arm's retained uncredited fresh failures (attributed
    via ``manifest_states.csv``). ``autonomous_successes`` counts the fresh
    credited episodes completed with zero operator interventions.
    """

    rows: list[dict[str, object]] = []
    for round_label in sorted(cfg.collection_dirs, key=_round_sort_key):
        collection_dir = cfg.collection_dirs[round_label]
        summary_path = collection_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        required = {
            "source_successes",
            "source_fresh_episodes",
            "source_counterfactual_episodes",
            "parent_ledger_total_rows",
            "n_uncredited_failures",
            "uncredited_failures",
            "source_repo",
        }
        missing = required - set(summary)
        if missing:
            raise RuntimeError(f"{summary_path}: missing required keys {sorted(missing)}")
        ledger_path = collection_dir / "source_ledger.csv"
        ledger = pd.read_csv(ledger_path)
        required_cols = {"arm_key", "is_counterfactual", "intervention_count"}
        missing_cols = required_cols - set(ledger.columns)
        if missing_cols:
            raise RuntimeError(f"{ledger_path}: missing required columns {sorted(missing_cols)}")
        if len(ledger) != int(summary["source_successes"]):
            raise RuntimeError(
                f"{ledger_path}: {len(ledger)} rows != source_successes "
                f"({summary['source_successes']}) in {summary_path}"
            )
        fresh = ledger[~ledger["is_counterfactual"].astype(bool)]
        if len(fresh) != int(summary["source_fresh_episodes"]):
            raise RuntimeError(
                f"{ledger_path}: {len(fresh)} fresh rows != source_fresh_episodes "
                f"({summary['source_fresh_episodes']}) in {summary_path}"
            )
        arm_keys = set(fresh["arm_key"].astype(str))
        if arm_keys != set(COLLECTION_ARM_STYLES):
            raise RuntimeError(
                f"{ledger_path}: fresh arm_key set {sorted(arm_keys)} != expected "
                f"{sorted(COLLECTION_ARM_STYLES)}"
            )
        uncredited = summary["uncredited_failures"]
        if len(uncredited) != int(summary["n_uncredited_failures"]):
            raise RuntimeError(
                f"{summary_path}: uncredited_failures has {len(uncredited)} entries "
                f"!= n_uncredited_failures ({summary['n_uncredited_failures']})"
            )
        # Retained uncredited failures carry no arm_key in the summary; attribute
        # each through its manifest state (manifest `source` uses the same arm
        # vocabulary as the ledger `arm_key`). CF replay failures are excluded
        # exactly like CF ledger rows.
        manifest_path = collection_dir / "manifest_states.csv"
        manifest = pd.read_csv(manifest_path).set_index("manifest_idx")
        fresh_failures: Counter[str] = Counter()
        for failure in uncredited:
            if bool(failure["is_counterfactual"]):
                continue
            source_arm = str(manifest.loc[int(failure["manifest_idx"]), "source"])
            if source_arm not in COLLECTION_ARM_STYLES:
                raise RuntimeError(
                    f"{manifest_path}: uncredited failure episode "
                    f"{failure['episode_index']} maps to unknown arm {source_arm!r}"
                )
            fresh_failures[source_arm] += 1
        for arm_key, style in COLLECTION_ARM_STYLES.items():
            arm_fresh = fresh[fresh["arm_key"] == arm_key]
            successes = len(arm_fresh)
            if successes <= 0:
                raise RuntimeError(f"{ledger_path}: no fresh {arm_key} rows")
            n = successes + fresh_failures[arm_key]
            lo, hi = wilson_ci(successes, n)
            autonomous = int((arm_fresh["intervention_count"].astype(int) == 0).sum())
            rows.append(
                {
                    "round": round_label,
                    "arm_key": arm_key,
                    "arm_label": style["label"],
                    "successes": successes,
                    "n": n,
                    "success_rate": successes / n,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                    "uncredited_failures": fresh_failures[arm_key],
                    "autonomous_successes": autonomous,
                    "autonomous_rate": autonomous / n,
                    "source_repo": summary["source_repo"],
                    "provenance": (
                        f"{_display_path(ledger_path)} fresh (non-CF) rows for "
                        f"{arm_key}: successes=credited fresh source rows, "
                        "n=+retained uncredited fresh failures attributed via "
                        f"{_display_path(manifest_path)}; autonomous=zero-intervention "
                        "credited episodes"
                    ),
                }
            )
    if not rows:
        raise RuntimeError(f"{cfg.task_key}: no DAgger collection directories configured")
    return pd.DataFrame(rows)


def _read_suite_headline_sr(line: SuiteLineHeadlineSpec) -> pd.DataFrame:
    path = _data_path(line.cfg, "headline_task_progress" if line.task_progress else "headline_sr")
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: missing per-line headline SR CSV; run python -m paper.appendix.real_results.prepare first"
        )
    df = pd.read_csv(path)
    required = {
        "round",
        "arm",
        "is_pooled",
        "successes",
        "n",
        "success_rate",
        "wilson_lo",
        "wilson_hi",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing required columns {sorted(missing)}")
    # The suite panel plots the pooled row per (round, arm); per-block rows stay
    # in the CSV for provenance only.
    df = df[df["is_pooled"].astype(bool)].copy()
    expected_rounds = _eval_round_order(line.cfg)
    actual_rounds = tuple(sorted(df["round"].astype(str).unique(), key=_round_sort_key))
    if actual_rounds != expected_rounds:
        raise RuntimeError(
            f"{path}: round mismatch; expected {expected_rounds}, got {actual_rounds}"
        )
    # Mirror build_success_rate_table: an arm contributes a row only in rounds
    # whose policy_arms include it (e.g. a final BoN arm exists at one round).
    expected_index = {
        (round_spec.round_label, arm)
        for round_spec in line.cfg.rounds
        for arm in line.cfg.headline_arms
        if arm in round_spec.policy_arms
    }
    actual_index = {(str(row["round"]), str(row["arm"])) for row in df.to_dict("records")}
    missing_index = sorted(expected_index - actual_index)
    extra_index = sorted(actual_index - expected_index)
    if missing_index or extra_index:
        raise RuntimeError(
            f"{path}: arm/round grid mismatch; missing={missing_index}, extra={extra_index}"
        )
    return df


def _suite_grouped_frame(line: SuiteLineHeadlineSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Pool the per-round headline rows onto the panel's (possibly grouped) axis.

    Successes and n sum across a group's member rounds; the Wilson CI is
    recomputed on the pooled counts. An arm contributes a group point only where
    it has rows (e.g. a rerank arm seated mid-campaign pools over its own rounds
    within the group — the k/n annotation carries the smaller support).
    """
    if line.cfg.episode_score_intervals and line.cfg.headline_metric == "task_progress":
        rows = []
        for tick_label, member_rounds in line.panel_groups():
            if len(member_rounds) != 1:
                raise ValueError(
                    "Shared-start graded rounds cannot be pooled as independent episodes"
                )
            sub = df[df["round"].astype(str) == member_rounds[0]].copy()
            if "episode_sem" not in sub or sub["episode_sem"].isna().any():
                raise ValueError(
                    "Missing episode-score uncertainty; regenerate the line headline table"
                )
            sub["round"] = tick_label
            rows.extend(sub.to_dict("records"))
        return pd.DataFrame(rows)
    rows: list[dict[str, object]] = []
    for tick_label, member_rounds in line.panel_groups():
        for arm in line.cfg.headline_arms:
            sub = df[(df["arm"] == arm) & (df["round"].astype(str).isin(member_rounds))]
            if sub.empty:
                continue
            successes = int(sub["successes"].sum())
            n = int(sub["n"].sum())
            lo, hi = wilson_ci(successes, n)
            rows.append(
                {
                    "round": tick_label,
                    "arm": arm,
                    "successes": successes,
                    "n": n,
                    "success_rate": successes / n,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                }
            )
    return pd.DataFrame(rows)


# Shared legend labels for the suite panels: each line's own labels encode its
# per-line protocol ("Baseline no-CF" vs "Baseline uniform"), but the suite
# figure draws one legend over all panels, so same-colored series need one
# shared name. Sample count varies by line/round (Marker R5 uses N=32 while the
# earlier marker and current square/routing points use N=16), so it is not part
# of the shared label; exact protocol detail stays in per-point provenance.
SUITE_ARM_LABELS: dict[str, str] = {
    "baseline": "Baseline (uniform DAgger)",
    "mulligan_with_cf": "Ours (evidence sampler)",
    "final_iql": "Ours DP+IQL rerank",
    "final_marker_iql": "Ours DP+IQL rerank",
}


def _plot_suite_line(
    ax: plt.Axes,
    line: SuiteLineHeadlineSpec,
    df: pd.DataFrame,
    *,
    label_overrides: Mapping[str, str] | None = None,
    title_fontsize: float | None = 11.0,
    markersize: float = 6.0,
    linewidth: float = 1.8,
    annotate_fontsize: float = 8.0,
    delta_suffix: str = " pp",
    interval: str = "combined",
    takeover_start: str | None = None,
) -> None:
    tick_labels = tuple(label for label, _ in line.panel_groups())
    if takeover_start is not None and takeover_start not in tick_labels:
        raise ValueError(
            f"{line.cfg.task_key}: takeover_start {takeover_start!r} is not a panel tick"
        )
    x_by_round = _round_x(tick_labels)
    if interval not in ("combined", "wilson1se"):
        raise ValueError(f"unknown suite-line interval {interval!r}")
    grouped = _with_wilson_1se(
        _with_binomial_sem(_suite_grouped_frame(line, df), "success_rate", "n")
    )
    _plot_arm_series(
        ax,
        grouped,
        round_x=x_by_round,
        y_col="success_rate",
        lo_col="wilson_1se_lo" if interval == "wilson1se" else "wilson_lo",
        hi_col="wilson_1se_hi" if interval == "wilson1se" else "wilson_hi",
        arms=line.cfg.headline_arms,
        all_arm_offsets=False,
        label_overrides=SUITE_ARM_LABELS if label_overrides is None else label_overrides,
        inner_lo_col=None if interval == "wilson1se" else "sem_lo",
        inner_hi_col=None if interval == "wilson1se" else "sem_hi",
        markersize=markersize,
        linewidth=linewidth,
    )
    piv = grouped.pivot_table(index="round", columns="arm", values="success_rate", aggfunc="first")
    if {"baseline", "mulligan_with_cf"} <= set(piv.columns):
        # Top delta vs baseline: the deployed policy, i.e. the DP+IQL rerank arm
        # where it has a point at this x and the takeover has started, else the
        # Ours with-CF DP.
        rerank_arms = [arm for arm in line.cfg.headline_arms if arm in IQL_RERANK_ARMS]
        for tick_idx, tick_label in enumerate(tick_labels):
            treatment = float(piv.loc[tick_label, "mulligan_with_cf"])
            if takeover_start is not None and tick_idx < tick_labels.index(takeover_start):
                rerank_arms_here = []
            else:
                rerank_arms_here = rerank_arms
            for arm in rerank_arms_here:
                if arm in piv.columns and pd.notna(piv.loc[tick_label, arm]):
                    treatment = float(piv.loc[tick_label, arm])
            delta = 100.0 * (treatment - float(piv.loc[tick_label, "baseline"]))
            ax.annotate(
                f"{delta:+.0f}{delta_suffix}",
                (x_by_round[tick_label], 96.0),
                ha="center",
                va="top",
                fontsize=annotate_fontsize,
                color=NEUTRAL,
            )
    if title_fontsize is None:
        ax.set_title(line.panel_title)
    else:
        ax.set_title(line.panel_title, fontsize=title_fontsize)
    ax.set_xticks(list(x_by_round.values()))
    ax.set_xticklabels(list(tick_labels))
    ax.set_xlim(-0.35, len(tick_labels) - 1 + 0.35)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=NEUTRAL, alpha=0.18, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


# Paper-facing restyle of the suite headline (the source of the paper's
# figs/real_world_headline_mulligan_vs_baseline.pdf). Two series only:
# "mulligan" is the COMBINED method lineage — at any x-position where a DP+BoN
# rerank arm has a point, that point replaces the with-CF one (the deployed
# full method takes over), so each panel reads as one Baseline-vs-Ours
# comparison. The takeover stays visible: rerank-sourced points (and the line
# from the last with-CF point onward) draw in the rerank purple. Deliberately
# smaller x-offsets than ARM_STYLES: the offsets are pure overplot relief and
# must not read as data.
PAPER_SUITE_SERIES: dict[str, dict[str, object]] = {
    "baseline": {
        "label": paper.POLICY_ARM_LABELS["baseline"],
        "color": paper.BASELINE,
        "marker": "o",
        "dx": -0.04,
    },
    "mulligan": {
        "label": paper.POLICY_ARM_LABELS["mulligan_with_cf"],
        "color": paper.OURS,
        "marker": "D",
        "dx": 0.04,
    },
}

PAPER_RERANK_LABEL = paper.POLICY_ARM_LABELS["final_iql"]

PAPER_WHISKERS = ("wilson1se", "ci95", "sem")


def _suite_paper_frame(
    line: SuiteLineHeadlineSpec,
    df: pd.DataFrame,
    *,
    whisker: str,
    takeover_start: str | None = None,
) -> pd.DataFrame:
    """Two-series (baseline / combined ours) frame on the panel's grouped axis.

    Same pooled numbers as :func:`_suite_grouped_frame`; the "mulligan" series takes
    the rerank arm's point wherever one exists at that x-position, else the
    with-CF point. ``takeover_start`` (a tick label) delays the takeover: rerank
    points at earlier x-positions are ignored in favor of the with-CF point
    (e.g. Route Cable shows the rerank lineage from R5, not from its R4 seat).
    ``whisker`` picks the single interval: ``"wilson1se"`` = boundary-safe
    Wilson ``z=1`` interval on the pooled counts, ``"ci95"`` = Wilson 95% CI,
    and ``"sem"`` = plug-in Wald ±1 binomial SE.
    For ``line.task_progress``, all whisker selections use episode-score SEM
    for the normalized two-clip score instead of a binomial interval.
    """
    if whisker not in PAPER_WHISKERS:
        raise ValueError(f"unknown whisker {whisker!r}; expected one of {PAPER_WHISKERS}")
    if whisker == "wilson1se":
        lo_col, hi_col = "wilson_1se_lo", "wilson_1se_hi"
    elif whisker == "ci95":
        lo_col, hi_col = "wilson_lo", "wilson_hi"
    else:
        lo_col, hi_col = "sem_lo", "sem_hi"
    grouped = _with_wilson_1se(
        _with_binomial_sem(_suite_grouped_frame(line, df), "success_rate", "n")
    )
    if line.task_progress:
        # Recover the episode-score second moment from clip totals and full
        # successes: sum(score**2) = total_clips + 2 * full_successes.
        # This keeps both clips from the same episode in one sampling unit.
        binary_line = replace(line, task_progress=False)
        binary = _suite_grouped_frame(binary_line, _read_suite_headline_sr(binary_line))
        paired = grouped.merge(
            binary, on=["round", "arm"], suffixes=("", "_binary"), validate="one_to_one"
        )
        n = paired["n_binary"].to_numpy(dtype=float)
        clips = paired["successes"].to_numpy(dtype=float)
        full = paired["successes_binary"].to_numpy(dtype=float)
        if (
            len(paired) != len(grouped)
            or np.any(n <= 1)
            or np.any(paired["n"].to_numpy() != 2 * n)
            or np.any(clips < 2 * full)
            or np.any(clips > n + full)
        ):
            raise ValueError(f"{line.cfg.task_key}: inconsistent two-clip progress counts")
        mean = clips / (2 * n)
        sem = np.sqrt(np.maximum(0, (clips + 2 * full) / 4 - n * mean**2) / (n * (n - 1)))
        grouped = paired.assign(
            episode_lo=np.maximum(0, mean - sem),
            episode_hi=np.minimum(1, mean + sem),
            n=n.astype(int),
        )
        lo_col, hi_col = "episode_lo", "episode_hi"
    rerank_arms = tuple(arm for arm in line.cfg.headline_arms if arm in IQL_RERANK_ARMS)
    tick_labels = tuple(label for label, _ in line.panel_groups())
    if takeover_start is not None and takeover_start not in tick_labels:
        raise ValueError(
            f"{line.cfg.task_key}: takeover_start {takeover_start!r} is not a panel tick; "
            f"ticks are {tick_labels}"
        )
    rows: list[dict[str, object]] = []
    for tick_idx, tick_label in enumerate(tick_labels):
        at_tick = grouped[grouped["round"].astype(str) == tick_label]
        by_arm = {str(row["arm"]): row for _, row in at_tick.iterrows()}
        if "baseline" not in by_arm or "mulligan_with_cf" not in by_arm:
            raise RuntimeError(
                f"{line.cfg.task_key} {tick_label}: paper headline needs baseline and "
                f"mulligan_with_cf points; have {sorted(by_arm)}"
            )
        ours_row = by_arm["mulligan_with_cf"]
        rerank_allowed = takeover_start is None or tick_idx >= tick_labels.index(takeover_start)
        if rerank_allowed:
            for arm in rerank_arms:
                if arm in by_arm:
                    ours_row = by_arm[arm]
        for series, row in (("baseline", by_arm["baseline"]), ("mulligan", ours_row)):
            rows.append(
                {
                    "round": tick_label,
                    "series": series,
                    "source_arm": str(row["arm"]),
                    "n": int(row["n"]),
                    "value": float(row["success_rate"]),
                    "lo": float(row[lo_col]),
                    "hi": float(row[hi_col]),
                }
            )
    return pd.DataFrame(rows)


def draw_suite_paper_headline_panels(
    axes: "np.ndarray | list[plt.Axes]",
    lines: tuple[SuiteLineHeadlineSpec, ...],
    *,
    whisker: str = "wilson1se",
    takeover_start: Mapping[str, str] | None = None,
    r0_reference: bool = True,
) -> list[tuple[plt.Axes, float]]:
    """Panel bodies of the paper suite headline, one axis per suite line: the
    combined Baseline-vs-Ours two-series grammar (:data:`PAPER_SUITE_SERIES`),
    one whisker family, and per-lineage R0 rules. ``takeover_start`` maps
    task_key -> first tick label where the rerank arm may take the lineage
    over (earlier rerank points are ignored in favor of with-CF). The caller
    owns figure assembly, legend (:func:`suite_paper_headline_legend`),
    layout, and save; returns the (axis, x_frac) x-axis-break marks to draw
    after the final layout."""

    if not lines:
        raise ValueError("at least one suite line is required")
    if len(axes) != len(lines):
        raise ValueError(f"got {len(axes)} axes for {len(lines)} suite lines")
    takeover_start = dict(takeover_start or {})
    unknown = sorted(set(takeover_start) - {line.cfg.task_key for line in lines})
    if unknown:
        raise ValueError(f"takeover_start names unknown task keys: {unknown}")
    axis_breaks: list[tuple[plt.Axes, float]] = []
    for ax, line in zip(axes, lines, strict=True):
        frame = _suite_paper_frame(
            line,
            _read_suite_headline_sr(line),
            whisker=whisker,
            takeover_start=takeover_start.get(line.cfg.task_key),
        )
        tick_labels = tuple(label for label, _ in line.panel_groups())
        x_by_round = _round_x(tick_labels)
        for series, style in PAPER_SUITE_SERIES.items():
            sub = _sort_by_round_axis(
                frame[frame["series"] == series],
                x_by_round,
                source=f"{line.cfg.task_key} paper {series} series",
            )
            x = [x_by_round[label] + float(style["dx"]) for label in sub["round"]]
            y = [100.0 * value for value in sub["value"]]
            err_lo = [
                max(0.0, 100.0 * (value - lo))
                for value, lo in zip(sub["value"], sub["lo"], strict=True)
            ]
            err_hi = [
                max(0.0, 100.0 * (hi - value))
                for value, hi in zip(sub["value"], sub["hi"], strict=True)
            ]
            is_rerank = [arm in IQL_RERANK_ARMS for arm in sub["source_arm"].astype(str)]
            if any(is_rerank) and not all(is_rerank[is_rerank.index(True) :]):
                raise RuntimeError(
                    f"{line.cfg.task_key}: rerank takeover points must form a suffix of "
                    f"the combined series; got source arms {list(sub['source_arm'])}"
                )
            takeover = is_rerank.index(True) if any(is_rerank) else len(x)
            # Line segments: the with-CF color up to the last with-CF point,
            # the rerank purple from the first rerank point onward, with the
            # color switching at the MIDPOINT of the transition segment so
            # neither point visually claims the other's color.
            if 0 < takeover < len(x):
                x_mid = 0.5 * (x[takeover - 1] + x[takeover])
                y_mid = 0.5 * (y[takeover - 1] + y[takeover])
                ax.plot(
                    x[:takeover] + [x_mid],
                    y[:takeover] + [y_mid],
                    color=style["color"],
                    zorder=3,
                )
                ax.plot(
                    [x_mid] + x[takeover:],
                    [y_mid] + y[takeover:],
                    color=paper.OURS_RERANK,
                    zorder=3,
                )
            elif takeover == 0:
                ax.plot(x, y, color=paper.OURS_RERANK, zorder=3)
            else:
                ax.plot(x, y, color=style["color"], zorder=3)
            for lo_idx, hi_idx, color, marker, label in (
                (0, takeover, style["color"], style["marker"], str(style["label"])),
                (takeover, len(x), paper.OURS_RERANK, "P", PAPER_RERANK_LABEL),
            ):
                if lo_idx == hi_idx:
                    continue
                ax.errorbar(
                    x[lo_idx:hi_idx],
                    y[lo_idx:hi_idx],
                    yerr=[err_lo[lo_idx:hi_idx], err_hi[lo_idx:hi_idx]],
                    color=color,
                    marker=marker,
                    markersize=5.5 if marker == "P" else 4.5,
                    linestyle="none",
                    elinewidth=1.0,
                    label=label,
                    zorder=4,
                )
        ax.set_title(line.panel_title)
        ax.set_xticks(list(x_by_round.values()))
        ax.set_xticklabels(list(tick_labels))
        xlim = (-0.35, len(tick_labels) - 1 + 0.35)
        ax.set_xlim(*xlim)
        axis_breaks.extend(
            (ax, frac) for frac in _round_axis_break_fracs(tick_labels, x_by_round, xlim)
        )
        ax.set_ylim(0, 100)
        paper.style_axes(ax)
        if r0_reference:
            # Each lineage's R0 (BC-only) level, carried forward from R1 on
            # (the series itself passes through its R0 marker). The Ours rule
            # stays teal even under a rerank takeover: R0 is a DP point.
            for series, style in PAPER_SUITE_SERIES.items():
                sub = _sort_by_round_axis(
                    frame[frame["series"] == series],
                    x_by_round,
                    source=f"{line.cfg.task_key} paper {series} r0 reference",
                )
                color = paper.OURS if series == "mulligan" else str(style["color"])
                paper.r0_reference(ax, 100.0 * float(sub["value"].iloc[0]), color, xmin=1.0 - 0.35)
    # Callers label any task-progress panels separately.
    axes[0].set_ylabel("Success rate (%)")
    return axis_breaks


def suite_paper_headline_legend(
    axes: "np.ndarray | list[plt.Axes]",
) -> tuple[list[object], list[str]]:
    """Deduped figure-legend entries for suite-headline panels: HG-DAgger plus
    one HiL-IDQL+Mulligan campaign entry (actor and reranked markers side by side) when a rerank
    takeover is present, so both stages read as the same lineage. Render with
    ``handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}``."""
    handles_by_label: dict[str, object] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles_by_label.setdefault(label, handle)
    baseline_label = str(PAPER_SUITE_SERIES["baseline"]["label"])
    ours_label = str(PAPER_SUITE_SERIES["mulligan"]["label"])
    legend_handles: list[object] = [handles_by_label[baseline_label]]
    legend_labels = [baseline_label]
    if PAPER_RERANK_LABEL in handles_by_label:
        legend_handles.append((handles_by_label[ours_label], handles_by_label[PAPER_RERANK_LABEL]))
        legend_labels.append(PAPER_RERANK_LABEL)
    else:
        legend_handles.append(handles_by_label[ours_label])
        legend_labels.append(ours_label)
    legend_handles.append(paper.static_bc_legend_handle())
    legend_labels.append(paper.STATIC_BC_LABEL)
    return legend_handles, legend_labels


def plot_suite_paper_headline_detailed(
    lines: tuple[SuiteLineHeadlineSpec, ...],
    *,
    xlabel: str = paper.ROUND_XLABEL,
    sources: tuple[str, ...] | None = None,
    takeover_start: Mapping[str, str] | None = None,
) -> paper.FigureRecord:
    """Appendix companion of the paper headline: every headline arm drawn
    separately with boundary-safe Wilson ``z=1`` whiskers and the
    per-round delta row) authored at print size. Per-point k/n counts stay on
    the full-size SVG (:func:`plot_suite_headline_success_rate`) — they do not
    survive print scale; the caption carries the ns."""

    if not lines:
        raise ValueError("at least one suite line is required")
    with paper.paper_rc():
        fig, axes_obj = plt.subplots(
            1, len(lines), figsize=paper.fig_size(1.0, height_in=2.55), sharey=True
        )
        axes = np.atleast_1d(axes_obj)
        axis_breaks: list[tuple[plt.Axes, float]] = []
        for ax, line in zip(axes, lines, strict=True):
            _plot_suite_line(
                ax,
                line,
                _read_suite_headline_sr(line),
                label_overrides=paper.POLICY_ARM_LABELS,
                title_fontsize=None,
                markersize=4.5,
                linewidth=1.4,
                annotate_fontsize=6.0,
                interval="wilson1se",
                # Bare "+20" labels: the panels are too narrow for a " pp"
                # suffix on adjacent round ticks; the caption names the unit.
                delta_suffix="",
                takeover_start=(takeover_start or {}).get(line.cfg.task_key),
            )
            tick_labels = tuple(label for label, _ in line.panel_groups())
            axis_breaks.extend(
                (ax, frac)
                for frac in _round_axis_break_fracs(
                    tick_labels, _round_x(tick_labels), ax.get_xlim()
                )
            )
        # Same clip-as-task convention as the paper headline: one plain ylabel.
        axes[0].set_ylabel("Success rate (%)")
        # Full method names need a shared strip outside the data panels.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
        fig.tight_layout(rect=(0, 0.21, 1, 1))
        paper.round_xlabel(fig, y=0.105, label=xlabel)
        for ax, frac in axis_breaks:
            paper.x_axis_break(ax, x_frac=frac)
        return paper.save_paper_figure(
            fig,
            "real_world_headline_mulligan_vs_baseline_detailed",
            width_frac=1.0,
            sources=sources
            if sources is not None
            else tuple(_display_path(_data_path(line.cfg, "headline_sr")) for line in lines),
        )


def plot_paper_substage(
    cfg: LineHeadlineConfig, *, sources: tuple[str, ...] | None = None
) -> paper.FigureRecord:
    """Print-size paper render of one line's headline G/I/SR substage panels.

    Reads the PERSISTED substage CSV written by ``paper.appendix.real_results.prepare``
    (never re-runs the ingest) and keeps the full-size grammar: same three
    conditional-completion panels, Wilson ``z=1`` whiskers. Per-point k/n counts stay
    on the full-size SVG — they do not survive print scale; the caption carries
    the ns. Plain ``R0..RN`` ticks — the per-round descriptors move to the
    caption too."""

    csv_path = _data_path(cfg, "substage_completion")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path}: missing persisted substage CSV; run python -m paper.appendix.real_results.prepare first"
        )
    df = pd.read_csv(csv_path)
    substage_rounds = set(df["round"].astype(str))
    round_labels = tuple(label for label in _eval_round_order(cfg) if label in substage_rounds)
    x_by_round = _round_x(round_labels)
    with paper.paper_rc():
        fig, axes = plt.subplots(1, 3, figsize=paper.fig_size(1.0, height_in=2.25), sharey=True)
        for ax, (rung, title) in zip(axes, SUBSTAGE_PANELS, strict=True):
            sub = _with_wilson_1se(df[df["rung"] == rung], "k", "n")
            _plot_arm_series(
                ax,
                sub,
                round_x=x_by_round,
                y_col="rate",
                lo_col="wilson_1se_lo",
                hi_col="wilson_1se_hi",
                arms=cfg.headline_arms,
                all_arm_offsets=False,
                label_overrides=paper.POLICY_ARM_LABELS,
                markersize=4.5,
                linewidth=1.4,
            )
            ax.set_title(title, fontsize=8.0)
            ax.set_xticks(list(x_by_round.values()))
            ax.set_xticklabels(list(round_labels))
            ax.set_xlim(-0.35, len(round_labels) - 1 + 0.35)
            ax.set_ylim(0, 100)
            paper.style_axes(ax)
        axes[0].set_ylabel("Completion rate (%)")
        # Keep the longer method names clear of the late-round success points.
        handles, labels = axes[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
        fig.tight_layout(rect=(0, 0.21, 1, 1))
        paper.round_xlabel(fig, y=0.105)
        return paper.save_paper_figure(
            fig,
            f"real_world_{cfg.task_key}_substage",
            width_frac=1.0,
            sources=sources if sources is not None else (_display_path(csv_path),),
        )


def _suite_efficiency_arm_series(
    line: SuiteLineHeadlineSpec,
    metric: str,
    panel_idx: int,
    *,
    measurement_interval: str | None = None,
) -> dict[str, dict[str, list]]:
    """Per-arm efficiency series for one suite panel, on the grouped axis.

    Shared by the full-size and paper grammars so their numbers cannot drift:
    returns ``{arm: {"ticks": [...], "ys": [...], "err_lo": [...],
    "err_hi": [...], "ks": [...]}}`` with x-offsets left to the caller. The
    bootstrap seed depends on the (panel, group, arm) indices. ``measurement_interval`` selects either ``"ci95"`` (t-based
    duration CI / percentile-bootstrap throughput CI) or ``"sem"`` (ordinary
    duration SEM / bootstrap throughput standard error). ``None`` preserves
    the full-size grammar: SEM for duration and 95% bootstrap CI for throughput.
    """
    if measurement_interval is None:
        measurement_interval = "sem" if metric == "success_duration" else "ci95"
    if measurement_interval not in ("sem", "ci95"):
        raise ValueError(f"unknown measurement_interval {measurement_interval!r}")
    cfg = line.cfg
    rounds_by_label = {spec.round_label: spec for spec in cfg.rounds}
    groups = line.panel_groups()
    out: dict[str, dict[str, list]] = {}
    for arm_idx, arm in enumerate(cfg.headline_arms):
        ticks: list[str] = []
        ys: list[float] = []
        err_lo: list[float] = []
        err_hi: list[float] = []
        ks: list[int] = []
        for group_idx, (tick_label, member_rounds) in enumerate(groups):
            graded = cfg.headline_metric == "task_progress"
            unit_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for label in member_rounds:
                if arm not in rounds_by_label[label].policy_arms:
                    continue
                if graded:
                    unit_parts.append(_arm_round_unit_data(cfg, rounds_by_label[label], arm))
                else:
                    successes, ep_steps = _arm_round_episode_data(cfg, rounds_by_label[label], arm)
                    unit_parts.append((successes.astype(int), ep_steps, ep_steps[successes]))
            if not unit_parts:
                continue
            units = np.concatenate([part[0] for part in unit_parts])
            steps = np.concatenate([part[1] for part in unit_parts])
            unit_steps = np.concatenate([part[2] for part in unit_parts])
            k = int(units.sum())
            if len(unit_steps) != k:
                raise RuntimeError(
                    f"{cfg.task_key} {tick_label} {arm}: {len(unit_steps)} unit times "
                    f"for {k} completed units"
                )
            if metric == "success_duration":
                if k == 0:
                    continue
                durations = unit_steps / EVAL_FPS
                y = float(durations.mean())
                se = float(durations.std(ddof=1) / math.sqrt(k)) if k > 1 else 0.0
                if measurement_interval == "ci95" and k > 1:
                    from scipy.stats import t as t_dist

                    half = se * float(t_dist.ppf(0.975, k - 1))
                else:
                    half = se
                lo_err, hi_err = half, half
            else:
                y = k / (float(steps.sum()) / EVAL_FPS / 60.0)
                bootstrap_seed = 20260717 + panel_idx * 1009 + group_idx * 17 + arm_idx
                if measurement_interval == "ci95":
                    lo, hi = _bootstrap_throughput_ci(units, steps, seed=bootstrap_seed)
                    lo_err, hi_err = y - lo, hi - y
                else:
                    se = _bootstrap_throughput_se(units, steps, seed=bootstrap_seed)
                    lo_err = hi_err = se
            ticks.append(tick_label)
            ys.append(y)
            err_lo.append(lo_err)
            err_hi.append(hi_err)
            ks.append(k)
        out[arm] = {"ticks": ticks, "ys": ys, "err_lo": err_lo, "err_hi": err_hi, "ks": ks}
    return out


# Paper-grammar overplot-relief x-offsets (smaller than the full-size
# ARM_STYLES offsets; they must not read as data).
_PAPER_EFFICIENCY_DX: dict[str, float] = {
    "baseline": -0.06,
    "mulligan_with_cf": 0.06,
    "final_iql": 0.18,
    "final_marker_iql": 0.18,
}

_PAPER_ARM_COLORS: dict[str, str] = {
    "baseline": paper.BASELINE,
    "mulligan_with_cf": paper.OURS,
    "final_iql": paper.OURS_RERANK,
    "final_marker_iql": paper.OURS_RERANK,
}


def _suite_efficiency_lineage_series(
    line: SuiteLineHeadlineSpec,
    metric: str,
    panel_idx: int,
    *,
    takeover_start: str | None,
) -> dict[str, dict[str, list]]:
    """Baseline and deployed-Ours efficiency series using headline takeover.

    Ours uses the DP point until a rerank point exists (and is allowed by
    ``takeover_start``), then uses the DP+BoN point. This is the efficiency
    counterpart of :func:`_suite_paper_frame`'s success-rate grammar.
    """
    all_arms = _suite_efficiency_arm_series(line, metric, panel_idx, measurement_interval="sem")
    tick_labels = tuple(label for label, _ in line.panel_groups())
    if takeover_start is not None and takeover_start not in tick_labels:
        raise ValueError(
            f"{line.cfg.task_key}: takeover_start {takeover_start!r} is not a panel tick; "
            f"ticks are {tick_labels}"
        )
    rerank_arms = tuple(arm for arm in line.cfg.headline_arms if arm in IQL_RERANK_ARMS)

    def point(arm: str, tick: str) -> dict[str, float | int] | None:
        data = all_arms[arm]
        if tick not in data["ticks"]:
            return None
        idx = data["ticks"].index(tick)
        return {
            "y": float(data["ys"][idx]),
            "err_lo": float(data["err_lo"][idx]),
            "err_hi": float(data["err_hi"][idx]),
            "k": int(data["ks"][idx]),
        }

    out = {
        "baseline": {
            "ticks": [],
            "ys": [],
            "err_lo": [],
            "err_hi": [],
            "ks": [],
            "source_arms": [],
        },
        "mulligan": {
            "ticks": [],
            "ys": [],
            "err_lo": [],
            "err_hi": [],
            "ks": [],
            "source_arms": [],
        },
    }
    for tick_idx, tick in enumerate(tick_labels):
        baseline = point("baseline", tick)
        ours = point("mulligan_with_cf", tick)
        if baseline is None or ours is None:
            raise RuntimeError(
                f"{line.cfg.task_key} {tick}: efficiency headline needs baseline and "
                "mulligan_with_cf points"
            )
        ours_arm = "mulligan_with_cf"
        rerank_allowed = takeover_start is None or tick_idx >= tick_labels.index(takeover_start)
        if rerank_allowed:
            for arm in rerank_arms:
                candidate = point(arm, tick)
                if candidate is not None:
                    ours, ours_arm = candidate, arm
        for series_key, arm, values in (
            ("baseline", "baseline", baseline),
            ("mulligan", ours_arm, ours),
        ):
            out[series_key]["ticks"].append(tick)
            out[series_key]["ys"].append(values["y"])
            out[series_key]["err_lo"].append(values["err_lo"])
            out[series_key]["err_hi"].append(values["err_hi"])
            out[series_key]["ks"].append(values["k"])
            out[series_key]["source_arms"].append(arm)
    return out


def _draw_efficiency_lineage(
    ax: plt.Axes,
    x_by_round: Mapping[str, float],
    series: dict[str, dict[str, list]],
) -> float:
    """Draw HG-DAgger plus one DP-to-DP+BoN deployed lineage."""
    baseline = series["baseline"]
    baseline_x = [x_by_round[tick] - 0.04 for tick in baseline["ticks"]]
    ax.errorbar(
        baseline_x,
        baseline["ys"],
        yerr=[baseline["err_lo"], baseline["err_hi"]],
        color=paper.BASELINE,
        marker="o",
        markersize=4.5,
        linewidth=1.4,
        elinewidth=1.0,
        label=paper.POLICY_ARM_LABELS["baseline"],
        zorder=3,
    )

    ours = series["mulligan"]
    ours_x = [x_by_round[tick] + 0.04 for tick in ours["ticks"]]
    ours_y = [float(value) for value in ours["ys"]]
    is_rerank = [arm in IQL_RERANK_ARMS for arm in ours["source_arms"]]
    if any(is_rerank) and not all(is_rerank[is_rerank.index(True) :]):
        raise RuntimeError(
            f"efficiency rerank takeover points must form a suffix; got {ours['source_arms']}"
        )
    takeover = is_rerank.index(True) if any(is_rerank) else len(ours_x)
    if 0 < takeover < len(ours_x):
        x_mid = 0.5 * (ours_x[takeover - 1] + ours_x[takeover])
        y_mid = 0.5 * (ours_y[takeover - 1] + ours_y[takeover])
        ax.plot(ours_x[:takeover] + [x_mid], ours_y[:takeover] + [y_mid], color=paper.OURS)
        ax.plot([x_mid] + ours_x[takeover:], [y_mid] + ours_y[takeover:], color=paper.OURS_RERANK)
    elif takeover == 0:
        ax.plot(ours_x, ours_y, color=paper.OURS_RERANK)
    else:
        ax.plot(ours_x, ours_y, color=paper.OURS)
    for lo_idx, hi_idx, color, marker, label in (
        (0, takeover, paper.OURS, "D", paper.POLICY_ARM_LABELS["mulligan_with_cf"]),
        (takeover, len(ours_x), paper.OURS_RERANK, "P", PAPER_RERANK_LABEL),
    ):
        if lo_idx == hi_idx:
            continue
        ax.errorbar(
            ours_x[lo_idx:hi_idx],
            ours_y[lo_idx:hi_idx],
            yerr=[ours["err_lo"][lo_idx:hi_idx], ours["err_hi"][lo_idx:hi_idx]],
            color=color,
            marker=marker,
            markersize=5.5 if marker == "P" else 4.5,
            linestyle="none",
            elinewidth=1.0,
            label=label,
            zorder=4,
        )
    return max(
        y + hi for data in series.values() for y, hi in zip(data["ys"], data["err_hi"], strict=True)
    )


def _plot_suite_efficiency_paper(
    lines: tuple[SuiteLineHeadlineSpec, ...],
    metric: str,
    *,
    xlabel: str = paper.ROUND_XLABEL,
    combined_lineage: bool = False,
    sources: tuple[str, ...] = (),
    takeover_start: Mapping[str, str] | None = None,
) -> paper.FigureRecord:
    """Paper efficiency grammar, detailed or headline-lineage form."""
    base_name = (
        "real_world_success_speed"
        if metric == "success_duration"
        else "real_world_success_throughput"
    )
    name = f"{base_name}_headline" if combined_lineage else base_name
    takeover_start = dict(takeover_start or {})
    with paper.paper_rc():
        fig, axes_obj = plt.subplots(1, len(lines), figsize=paper.fig_size(1.0, height_in=2.3))
        axes = np.atleast_1d(axes_obj)
        axis_breaks: list[tuple[plt.Axes, float]] = []
        for panel_idx, (ax, line) in enumerate(zip(axes, lines, strict=True)):
            groups = line.panel_groups()
            tick_labels = tuple(label for label, _ in groups)
            x_by_round = _round_x(tick_labels)
            if combined_lineage:
                series = _suite_efficiency_lineage_series(
                    line,
                    metric,
                    panel_idx,
                    takeover_start=takeover_start.get(line.cfg.task_key),
                )
                panel_upper = _draw_efficiency_lineage(ax, x_by_round, series)
            else:
                series = _suite_efficiency_arm_series(
                    line, metric, panel_idx, measurement_interval="sem"
                )
                panel_upper = 0.0
                for arm in line.cfg.headline_arms:
                    data = series[arm]
                    if not data["ticks"]:
                        continue
                    xs = [x_by_round[t] + _PAPER_EFFICIENCY_DX[arm] for t in data["ticks"]]
                    marker = str(ARM_STYLES[arm]["marker"])
                    ax.errorbar(
                        xs,
                        data["ys"],
                        yerr=[data["err_lo"], data["err_hi"]],
                        color=_PAPER_ARM_COLORS[arm],
                        marker=marker,
                        markersize=5.5 if marker == "P" else 4.5,
                        linewidth=1.4 if len(xs) > 1 else 0.0,
                        linestyle="-" if len(xs) > 1 else "none",
                        elinewidth=1.0,
                        label=paper.POLICY_ARM_LABELS[arm],
                        zorder=3,
                    )
                    panel_upper = max(
                        panel_upper,
                        max(y + hi for y, hi in zip(data["ys"], data["err_hi"], strict=True)),
                    )
            ax.set_title(line.panel_title)
            ax.set_xticks(list(x_by_round.values()))
            ax.set_xticklabels(list(tick_labels))
            xlim = (-0.35, len(tick_labels) - 1 + 0.35)
            ax.set_xlim(*xlim)
            axis_breaks.extend(
                (ax, frac) for frac in _round_axis_break_fracs(tick_labels, x_by_round, xlim)
            )
            floor_upper = 10.0 if metric == "success_duration" else 0.1
            ax.set_ylim(0, max(floor_upper, panel_upper * 1.18))
            paper.style_axes(ax)
        axes[0].set_ylabel(
            "Time per task unit (s)" if metric == "success_duration" else "Task units per minute"
        )
        # Headline layout grammar: one horizontal figure-level legend strip
        # under the panel row (panels keep their own y scales, so no wspace
        # tightening here). Handles gathered across panels: a panel may lack
        # an arm (e.g. a line without a rerank round).
        handles_by_label: dict[str, object] = {}
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                handles_by_label.setdefault(label, handle)
        legend_handles = list(handles_by_label.values())
        legend_labels = list(handles_by_label.keys())
        handler_map = None
        if combined_lineage:
            baseline_label = paper.POLICY_ARM_LABELS["baseline"]
            ours_label = paper.POLICY_ARM_LABELS["mulligan_with_cf"]
            legend_handles = [handles_by_label[baseline_label]]
            legend_labels = [baseline_label]
            if PAPER_RERANK_LABEL in handles_by_label:
                legend_handles.append(
                    (handles_by_label[ours_label], handles_by_label[PAPER_RERANK_LABEL])
                )
                legend_labels.append(PAPER_RERANK_LABEL)
            else:
                legend_handles.append(handles_by_label[ours_label])
                legend_labels.append(ours_label)
            handler_map = {tuple: HandlerTuple(ndivide=None, pad=0.15)}
        fig.legend(
            legend_handles,
            legend_labels,
            handler_map=handler_map,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(legend_labels),
            columnspacing=1.6,
            fontsize=6.5,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.21, 1, 1))
        paper.round_xlabel(fig, y=0.105, label=xlabel)
        for ax, frac in axis_breaks:
            paper.x_axis_break(ax, x_frac=frac)
        return paper.save_paper_figure(fig, name, width_frac=1.0, sources=sources)
