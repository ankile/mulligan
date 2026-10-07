"""Standard stage-label eval plots for real-world held-out sessions."""

from __future__ import annotations

import itertools
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mulligan-matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mulligan.plotting.colors import (  # noqa: E402
    METHOD_COLORS,
    NEUTRAL_INK,
    NEUTRAL_INK_MID,
    label_text_color,
    stage_palette,
)
from mulligan.real.lifecycle.stats import exact_mcnemar_pvalue  # noqa: E402
from mulligan.real.stage_specs.tasks import StageLabelTaskSpec  # noqa: E402


@dataclass(frozen=True)
class ArmPlotSpec:
    key: str
    label: str
    color: str
    # Stem of the paired-rounds episode column when it differs from ``key`` (e.g.
    # the stage-label ``policy_short`` is ``dp``/``iql_cfinal`` but the paired-rounds
    # CSV names the arm ``mulligan_with_cf``/``iql``). ``None`` => derive from ``key``.
    paired_key: str | None = None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _mcnemar_exact_p_value(
    left_success: np.ndarray, right_success: np.ndarray
) -> tuple[float, int, int]:
    if left_success.shape != right_success.shape:
        raise ValueError(
            f"Paired event arrays have different shapes: {left_success.shape} vs {right_success.shape}"
        )
    left_only = int(np.logical_and(left_success, ~right_success).sum())
    right_only = int(np.logical_and(~left_success, right_success).sum())
    # Delegate the exact two-tailed p-value to the shared lifecycle helper (same
    # formula, incl. the zero-discordant -> 1.0 branch); keep the triple shape.
    return exact_mcnemar_pvalue(left_only, right_only), left_only, right_only


def _hypergeom_prob(a: int, row1: int, col1: int, total: int) -> float:
    row2 = total - row1
    col2 = total - col1
    if not (0 <= a <= row1 and 0 <= col1 - a <= row2 and 0 <= row1 - a <= col2):
        return 0.0
    logp = (
        math.lgamma(col1 + 1)
        - math.lgamma(a + 1)
        - math.lgamma(col1 - a + 1)
        + math.lgamma(col2 + 1)
        - math.lgamma(row1 - a + 1)
        - math.lgamma(col2 - row1 + a + 1)
        - math.lgamma(total + 1)
        + math.lgamma(row1 + 1)
        + math.lgamma(total - row1 + 1)
    )
    return math.exp(logp)


def _fisher_exact_two_sided(k_left: int, n_left: int, k_right: int, n_right: int) -> float:
    if not (0 <= k_left <= n_left and 0 <= k_right <= n_right):
        raise ValueError(
            f"Invalid Fisher table counts: left={k_left}/{n_left}, right={k_right}/{n_right}"
        )
    total = n_left + n_right
    if total == 0:
        return 1.0
    col1 = k_left + k_right
    col2 = total - col1
    lo = max(0, n_left - col2)
    hi = min(n_left, col1)
    observed = _hypergeom_prob(k_left, n_left, col1, total)
    p_value = 0.0
    for a in range(lo, hi + 1):
        prob = _hypergeom_prob(a, n_left, col1, total)
        if prob <= observed + 1e-12:
            p_value += prob
    return min(1.0, p_value)


def _format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def _rung_annotation_pair(arms: list[ArmPlotSpec]) -> tuple[str, str, str] | None:
    keys = [arm.key for arm in arms]
    if len(keys) == 2:
        return keys[0], keys[1], ""
    if "baseline" in keys and "mulligan_with_cf" in keys:
        return "baseline", "mulligan_with_cf", "baseline vs with-CF"
    # General headline pair: baseline vs the rightmost (final treatment) arm,
    # e.g. the quad block's baseline vs DP+IQL rerank. This reduces to the
    # baseline-vs-with-CF branch above for the 3-arm no-CF/with-CF blocks.
    if "baseline" in keys and keys[-1] != "baseline":
        right = keys[-1]
        right_label = {arm.key: arm.label for arm in arms}[right].replace("\n", " ")
        return "baseline", right, f"baseline vs {right_label}"
    return None


def default_arm_specs(policy_keys: list[str]) -> list[ArmPlotSpec]:
    """Stable display labels/colors for common real-world eval arm keys."""
    preferred = [
        "baseline_r00_dp",
        "baseline",
        "mulligan_r00_dp",
        "mulligan_no_cf",
        "mulligan_with_cf",
    ]
    ordered = [key for key in preferred if key in policy_keys] + [
        key for key in policy_keys if key not in preferred
    ]
    specs = []
    for key in ordered:
        lower = key.lower()
        if "baseline" in lower or "uniform" in lower:
            label = "Baseline" if key == "baseline" else "Baseline uniform R0"
            color = METHOD_COLORS["uniform"]
        elif "no_cf" in lower:
            label = "Ours (no-CF)"
            color = METHOD_COLORS["sobol_light"]
        elif "with_cf" in lower:
            label = "Ours (with-CF)"
            color = METHOD_COLORS["sobol"]
        elif "sobol" in lower or "mulligan" in lower:
            label = "Ours Sobol R0"
            color = METHOD_COLORS["sobol"]
        else:
            label = key.replace("_", " ")
            color = METHOD_COLORS["gray_neutral"]
        specs.append(ArmPlotSpec(key=key, label=label, color=color))
    return specs


def _load_labels(labels_csv: Path, spec: StageLabelTaskSpec) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    required = {"episode_index", "policy_short", spec.stage_field, spec.failure_mode_field}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{labels_csv} missing columns {sorted(missing)}")
    df = df.copy()
    df[spec.stage_field] = df[spec.stage_field].astype(int)
    return df


def _episode_column_for_arm(pair_df: pd.DataFrame, arm: ArmPlotSpec) -> str:
    lower = arm.key.lower()
    candidates = [f"{arm.key}_episode_index"]
    if arm.paired_key:
        # Explicit alias wins: the paired-rounds CSV may name the arm differently
        # from the stage-label policy_short (e.g. dp -> mulligan_with_cf).
        candidates.insert(0, f"{arm.paired_key}_episode_index")
    if "baseline" in lower:
        candidates.append("baseline_episode_index")
    if "mulligan" in lower or "sobol" in lower:
        candidates.append("mulligan_episode_index")
    candidates.extend(
        [
            f"{lower}_episode_index",
            f"{arm.label.lower().replace(' ', '_')}_episode_index",
        ]
    )
    for candidate in candidates:
        if candidate in pair_df.columns:
            return candidate
    raise ValueError(
        f"Could not infer paired episode column for arm {arm.key!r}; "
        f"available columns: {sorted(pair_df.columns)}"
    )


def _paired_stages(
    df: pd.DataFrame,
    spec: StageLabelTaskSpec,
    arms: list[ArmPlotSpec],
    paired_rounds_csv: Path,
) -> pd.DataFrame:
    pair_df = pd.read_csv(paired_rounds_csv)
    stage_cols = {}
    out = pd.DataFrame(index=pair_df.index)
    for arm in arms:
        episode_col = _episode_column_for_arm(pair_df, arm)
        sub = df[df.policy_short == arm.key]
        duplicated = sub["episode_index"].duplicated()
        if duplicated.any():
            dupes = sorted(sub.loc[duplicated, "episode_index"].astype(int).unique().tolist())
            raise ValueError(f"Duplicate labeled episode_index values for arm {arm.key}: {dupes}")
        stage_by_episode = sub.set_index("episode_index")[spec.stage_field]
        stages = pair_df[episode_col].map(stage_by_episode)
        if stages.isna().any():
            missing = pair_df.loc[stages.isna(), episode_col].astype(int).tolist()
            raise ValueError(
                f"{paired_rounds_csv} references unlabeled episodes for arm {arm.key}: {missing}"
            )
        stage_cols[arm.key] = stages.astype(int).to_numpy()
        out[f"{arm.key}_episode_index"] = pair_df[episode_col].astype(int)
        out[f"{arm.key}_stage"] = stage_cols[arm.key]
    return out


def _anchor_stage_to_outcome(
    df: pd.DataFrame,
    spec: StageLabelTaskSpec,
    arms: list[ArmPlotSpec],
    paired_rounds_csv: Path,
) -> pd.DataFrame:
    """Anchor the top of the stage ladder to the human outcome-edited success.

    The battery reports POLICY success rate, for which the frame-validated /
    outcome-edited success flag in ``paired_rounds_csv`` is GROUND TRUTH — not the
    VLM's perceptual S{success_level} (seated + released) call, which has a known
    soft spot on the release event. Without this, the rung SR (Gemini P(S7)) and
    the headline eval SR (outcome-edited) disagree by a few episodes/arm, which is
    exactly the kind of inconsistency a policy report must not show.

    Per arm: success ⇒ stage := success_level; recorded failure ⇒
    stage := min(stage, success_level - 1) — a recorded failure can never be
    counted as strict success even if the VLM hallucinated the release. Lower
    (intermediate) rungs stay VLM-derived. Guarantees battery SR ≡ outcome SR by
    construction. The real fail-loud guards are the raises below: a paired CSV that
    references an episode with no outcome, or a partial set of arm success columns.
    """
    pair_df = pd.read_csv(paired_rounds_csv)
    success_level = spec.ladder.success_level
    # Some paired CSVs carry only episode-index columns (used purely for McNemar
    # pairing, no outcome column). Anchor only when EVERY arm has its ground-truth
    # success column; skip cleanly when NONE do (pairing-only CSV); raise on
    # a partial set, which is malformed.
    success_cols = {
        arm.key: _episode_column_for_arm(pair_df, arm)[: -len("_episode_index")] + "_success"
        for arm in arms
    }
    present = [c for c in success_cols.values() if c in pair_df.columns]
    if not present:
        return df
    if len(present) != len(arms):
        missing = sorted(c for c in success_cols.values() if c not in pair_df.columns)
        raise ValueError(
            f"{paired_rounds_csv}: partial outcome columns — has {sorted(present)} but "
            f"missing {missing}; provide a success column for every arm or none"
        )
    df = df.copy()
    for arm in arms:
        episode_col = _episode_column_for_arm(pair_df, arm)
        success_col = success_cols[arm.key]
        success_by_episode = dict(
            zip(pair_df[episode_col].astype(int), pair_df[success_col].astype(int))
        )
        arm_mask = df.policy_short == arm.key
        eps = df.loc[arm_mask, "episode_index"].astype(int)
        missing = sorted(set(eps) - set(success_by_episode))
        if missing:
            raise ValueError(
                f"{paired_rounds_csv} has no outcome for arm {arm.key!r} episodes {missing}"
            )
        succ = eps.map(success_by_episode).to_numpy()
        stages = df.loc[arm_mask, spec.stage_field].to_numpy()
        anchored = np.where(succ == 1, success_level, np.minimum(stages, success_level - 1))
        df.loc[arm_mask, spec.stage_field] = anchored
        # (No count assertion here: anchored == success_level iff succ == 1 by
        # construction, so anchored-S7 count always equals the join's success count
        # — it cannot catch a mis-join. The join is guarded above by the
        # missing-episode and partial-column raises.)
    return df


def build_rung_comparisons(
    df: pd.DataFrame,
    spec: StageLabelTaskSpec,
    arms: list[ArmPlotSpec],
    rung_tab: pd.DataFrame,
    *,
    paired_rounds_csv: Path | None = None,
    conditional_test: Literal["fisher", "descriptive"] = "fisher",
) -> pd.DataFrame:
    """Exact pairwise significance tests for the rung-conversion funnel."""
    if conditional_test not in {"fisher", "descriptive"}:
        raise ValueError(f"Unknown conditional test: {conditional_test}")
    comparison_columns = [
        "rung",
        "left_arm",
        "right_arm",
        "left_k",
        "left_n",
        "right_k",
        "right_n",
        "test",
        "p_value",
        "paired_n",
        "left_only",
        "right_only",
        "note",
    ]
    if len(arms) < 2:
        return pd.DataFrame(
            [
                {
                    "rung": "",
                    "left_arm": "",
                    "right_arm": "",
                    "left_k": "",
                    "left_n": "",
                    "right_k": "",
                    "right_n": "",
                    "test": "not computed",
                    "p_value": "",
                    "paired_n": "",
                    "left_only": "",
                    "right_only": "",
                    "note": (
                        f"rung-comparison annotations require at least two arms; got {len(arms)}"
                    ),
                }
            ],
            columns=comparison_columns,
        )

    transport_level = max(0, spec.ladder.success_level - 4)
    success_level = spec.ladder.success_level
    tab = rung_tab.set_index(["arm", "rung"])
    rows = []
    paired = _paired_stages(df, spec, arms, paired_rounds_csv) if paired_rounds_csv else None

    for left_arm, right_arm in itertools.combinations(arms, 2):
        if paired is not None:
            for rung, threshold in (("G", transport_level), ("SR", success_level)):
                left_events = paired[f"{left_arm.key}_stage"].to_numpy() >= threshold
                right_events = paired[f"{right_arm.key}_stage"].to_numpy() >= threshold
                p_value, left_only, right_only = _mcnemar_exact_p_value(left_events, right_events)
                left_rec = tab.loc[(left_arm.key, rung)]
                right_rec = tab.loc[(right_arm.key, rung)]
                rows.append(
                    {
                        "rung": rung,
                        "left_arm": left_arm.key,
                        "right_arm": right_arm.key,
                        "left_k": int(left_rec.k),
                        "left_n": int(left_rec.n),
                        "right_k": int(right_rec.k),
                        "right_n": int(right_rec.n),
                        "test": "paired exact McNemar",
                        "p_value": p_value,
                        "paired_n": len(paired),
                        "left_only": left_only,
                        "right_only": right_only,
                        "note": "paired over matched held-out starts",
                    }
                )
        else:
            for rung in ("G", "SR"):
                left_rec = tab.loc[(left_arm.key, rung)]
                right_rec = tab.loc[(right_arm.key, rung)]
                p_value = _fisher_exact_two_sided(
                    int(left_rec.k), int(left_rec.n), int(right_rec.k), int(right_rec.n)
                )
                rows.append(
                    {
                        "rung": rung,
                        "left_arm": left_arm.key,
                        "right_arm": right_arm.key,
                        "left_k": int(left_rec.k),
                        "left_n": int(left_rec.n),
                        "right_k": int(right_rec.k),
                        "right_n": int(right_rec.n),
                        "test": "two-sided Fisher exact",
                        "p_value": p_value,
                        "paired_n": "",
                        "left_only": "",
                        "right_only": "",
                        "note": "unpaired arm-level test; pass paired_rounds_csv for McNemar",
                    }
                )

        left_i = tab.loc[(left_arm.key, "I")]
        right_i = tab.loc[(right_arm.key, "I")]
        conditional_defined = int(left_i.n) > 0 and int(right_i.n) > 0
        p_value = (
            _fisher_exact_two_sided(int(left_i.k), int(left_i.n), int(right_i.k), int(right_i.n))
            if conditional_defined and conditional_test == "fisher"
            else float("nan")
        )
        rows.append(
            {
                "rung": "I",
                "left_arm": left_arm.key,
                "right_arm": right_arm.key,
                "left_k": int(left_i.k),
                "left_n": int(left_i.n),
                "right_k": int(right_i.k),
                "right_n": int(right_i.n),
                "test": "descriptive only"
                if conditional_test == "descriptive"
                else "two-sided Fisher exact"
                if conditional_defined
                else "not estimated",
                "p_value": p_value,
                "paired_n": "",
                "left_only": "",
                "right_only": "",
                "note": "arm-specific conditional cohorts; no independent-sample test"
                if conditional_test == "descriptive"
                else f"conditional on each arm reaching S≥{transport_level}"
                if conditional_defined
                else "at least one arm has no episodes reaching the condition",
            }
        )
    return (
        pd.DataFrame(rows, columns=comparison_columns)
        .sort_values(["left_arm", "right_arm", "rung"])
        .reset_index(drop=True)
    )


def plot_stage_share(
    df: pd.DataFrame,
    spec: StageLabelTaskSpec,
    arms: list[ArmPlotSpec],
    *,
    title: str,
    out: Path,
    stage_axis_label: str | None = None,
) -> None:
    stages = list(range(spec.ladder.max_stage + 1))
    stage_colors = stage_palette(len(stages))
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.75 * len(arms)))
    for row, arm in enumerate(arms):
        g = df[df.policy_short == arm.key]
        n = len(g)
        counts = Counter(g[spec.stage_field])
        left = 0.0
        for stage in stages:
            share = counts.get(stage, 0) / n if n else 0.0
            if share == 0:
                continue
            ax.barh(
                row,
                share,
                left=left,
                color=stage_colors[stage],
                edgecolor="white",
                height=0.6,
            )
            if share >= 0.045:
                ax.text(
                    left + share / 2,
                    row,
                    f"S{stage}\n{100 * share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=label_text_color(stage_colors[stage]),
                )
            left += share
        ax.text(1.01, row, f"n={n}", va="center", fontsize=9, transform=ax.get_yaxis_transform())

    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([arm.label for arm in arms])
    ax.set_xlim(0, 1)
    ticks = np.arange(0, 1.01, 0.2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(100 * tick)}%" for tick in ticks])
    ax.set_xlabel(
        stage_axis_label
        if stage_axis_label is not None
        else f"Share of episodes by max stage reached (S{spec.ladder.success_level} = strict success)"
    )
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_rung_conversion(
    df: pd.DataFrame,
    spec: StageLabelTaskSpec,
    arms: list[ArmPlotSpec],
    *,
    title: str,
    out: Path,
    paired_rounds_csv: Path | None = None,
    rung_labels: tuple[str, str, str] | None = None,
    conditional_test: Literal["fisher", "descriptive"] = "fisher",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot maximum-stage reach at the lower and upper ladder thresholds.

    ``rung_labels`` names G, I, and SR respectively. Those stable table keys
    represent lower-threshold reach, upper reach conditional on lower reach,
    and upper-threshold reach. For raw historical maximum stages, the SR key
    is not a claim about the final task outcome. Supply task-specific labels
    for that case; the defaults are the anchored battery wording.
    """
    transport_level = max(0, spec.ladder.success_level - 4)
    success_level = spec.ladder.success_level
    labels = (
        rung_labels
        if rung_labels is not None
        else (
            f"G: grasp + transport\nP(S≥{transport_level})",
            f"I: insert + seat + release\nP(S{success_level} | S≥{transport_level})",
            f"SR = G × I\nP(S{success_level})",
        )
    )
    rungs = list(zip(("G", "I", "SR"), labels, strict=True))
    rows = []
    for arm in arms:
        g = df[df.policy_short == arm.key]
        n = len(g)
        k_g = int((g[spec.stage_field] >= transport_level).sum())
        k_sr = int((g[spec.stage_field] >= success_level).sum())
        for rung, k, denom in (("G", k_g, n), ("I", k_sr, k_g), ("SR", k_sr, n)):
            lo, hi = wilson(k, denom) if denom else (float("nan"), float("nan"))
            rows.append(
                {
                    "arm": arm.key,
                    "rung": rung,
                    "k": k,
                    "n": denom,
                    "rate": k / denom if denom else float("nan"),
                    "lo": lo,
                    "hi": hi,
                }
            )
    tab = pd.DataFrame(rows)
    if rung_labels is not None:
        tab["metric_label"] = tab["rung"].map(dict(rungs))
    comparisons = build_rung_comparisons(
        df,
        spec,
        arms,
        tab,
        paired_rounds_csv=paired_rounds_csv,
        conditional_test=conditional_test,
    )
    _render_rung_conversion(tab, rungs, arms, comparisons, title=title, out=out)
    return tab, comparisons


# Sentinel: default ``annotation_pair`` => derive from arm keys via
# ``_rung_annotation_pair``. ``None`` explicitly means "no bracket".
_AUTO_ANNOTATION_PAIR = object()


def _render_rung_conversion(
    tab: pd.DataFrame,
    rungs: list[tuple[str, str]],
    arms: list[ArmPlotSpec],
    comparisons: pd.DataFrame,
    *,
    title: str,
    out: Path,
    annotation_pair: tuple[str, str, str] | None | object = _AUTO_ANNOTATION_PAIR,
) -> None:
    """Grouped rung-conversion bar chart shared by the stage-label battery and
    the routing graded-score funnel.

    ``tab`` has one row per (arm, rung) with columns arm/rung/k/n/rate/lo/hi;
    ``rungs`` is the ordered ``[(key, x-axis label)]`` list; ``comparisons``
    supplies the optional baseline-vs-final significance bracket (columns
    left_arm/right_arm/rung/test/p_value). The stat computation that fills
    ``tab``/``comparisons`` differs by caller (stage ladder vs graded clip
    score); only the drawing is shared here.

    ``annotation_pair`` selects the ``(left_key, right_key, short_label)`` that
    the significance bracket compares. It defaults to deriving the pair from the
    arm keys via ``_rung_annotation_pair`` (unchanged stage-label behavior);
    pass an explicit triple to override the derived label (e.g. a short
    ``"baseline vs DP+IQL"`` when the arm's legend label is long), or ``None``
    to suppress the bracket.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    n_arms = len(arms)
    group_w = 0.8
    bar_w = group_w / n_arms
    x_base = np.arange(len(rungs))
    bar_x: dict[tuple[str, str], float] = {}
    for j, arm in enumerate(arms):
        xs = x_base + (j - (n_arms - 1) / 2) * bar_w
        sub = tab[tab.arm == arm.key].set_index("rung")
        ys = [100 * sub.loc[r].rate for r, _ in rungs]
        yerr = [
            [max(0.0, 100 * (sub.loc[r].rate - sub.loc[r].lo)) for r, _ in rungs],
            [max(0.0, 100 * (sub.loc[r].hi - sub.loc[r].rate)) for r, _ in rungs],
        ]
        ax.bar(
            xs, ys, bar_w * 0.92, color=arm.color, label=arm.label, edgecolor="white", linewidth=0.5
        )
        ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor=NEUTRAL_INK_MID, capsize=2.5, lw=1.0)
        for x, rung in zip(xs, [r for r, _ in rungs]):
            rec = sub.loc[rung]
            bar_x[(arm.key, rung)] = float(x)
            ax.text(
                x,
                100 * rec.rate + 1.5 if rec.n else 1.5,
                f"{100 * rec.rate:.0f}%\n{int(rec.k)}/{int(rec.n)}" if rec.n else "n/a\n0/0",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=NEUTRAL_INK,
            )
    ylim_top = 105.0
    if annotation_pair is _AUTO_ANNOTATION_PAIR:
        annotation_pair = _rung_annotation_pair(arms)
    if annotation_pair is not None and not comparisons.empty:
        left_key, right_key, pair_label = annotation_pair
        pair_rows = comparisons[
            ((comparisons["left_arm"] == left_key) & (comparisons["right_arm"] == right_key))
            | ((comparisons["left_arm"] == right_key) & (comparisons["right_arm"] == left_key))
        ]
        comparisons_by_rung = pair_rows.set_index("rung")
        for i, (rung, _) in enumerate(rungs):
            if rung not in comparisons_by_rung.index:
                continue
            rec = comparisons_by_rung.loc[rung]
            if not np.isfinite(float(rec.p_value)):
                continue
            sub = tab[tab.rung == rung]
            # Clear both the CI and the two-line percentage/count label,
            # including a 100% bar whose upper CI cannot provide headroom.
            y = max(100 * float(sub["hi"].max()) + 9.0, 100 * float(sub["rate"].max()) + 15.0)
            x0 = min(bar_x[(left_key, rung)], bar_x[(right_key, rung)]) - bar_w * 0.4
            x1 = max(bar_x[(left_key, rung)], bar_x[(right_key, rung)]) + bar_w * 0.4
            tick = 1.4
            test_name = "McNemar" if "McNemar" in rec.test else "Fisher"
            test_label = f"{test_name} {_format_p_value(float(rec.p_value))}"
            if pair_label:
                test_label = f"{pair_label}\n{test_label}"
            ax.plot(
                [x0, x0, x1, x1],
                [y - tick, y, y, y - tick],
                color=NEUTRAL_INK_MID,
                lw=0.8,
                clip_on=False,
            )
            ax.text(
                (x0 + x1) / 2,
                y + 1.5,
                test_label,
                ha="center",
                va="bottom",
                fontsize=7.3,
                color=NEUTRAL_INK,
                clip_on=False,
            )
            ylim_top = max(ylim_top, y + 15.0)
    ax.set_xticks(x_base)
    ax.set_xticklabels([label for _, label in rungs], fontsize=9)
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel("Conversion rate (%, Wilson 95% CI)")
    ax.set_title(title, fontsize=10)
    # Legend outside the axes (upper-left, hanging off the right edge) so it can
    # never overlap the rightmost arm's bar labels or a headline annotation
    # bracket; bbox_inches="tight" on savefig keeps it in frame.
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    df: pd.DataFrame,
    spec: StageLabelTaskSpec,
    arms: list[ArmPlotSpec],
    rung_tab: pd.DataFrame,
    out: Path,
) -> pd.DataFrame:
    """Summarize stage reach; G/I/SR inherit the input rung table's meaning.

    With raw max-stage labels these are historical reach rates, including
    SR_rate. They are final success rates only for outcome-anchored inputs.
    """
    rows = []
    stages = list(range(spec.ladder.max_stage + 1))
    for arm in arms:
        g = df[df.policy_short == arm.key]
        sub = rung_tab[rung_tab.arm == arm.key].set_index("rung")
        counts = Counter(g[spec.stage_field])
        failure_modes = Counter(g[spec.failure_mode_field].dropna())
        failure_modes.pop("none", None)
        rows.append(
            {
                "arm": arm.key,
                "n": len(g),
                "mean_stage": round(float(g[spec.stage_field].mean()), 3),
                "G_rate": round(float(sub.loc["G"].rate), 4),
                "G_lo": round(float(sub.loc["G"].lo), 4),
                "G_hi": round(float(sub.loc["G"].hi), 4),
                "I_rate": round(float(sub.loc["I"].rate), 4),
                "I_lo": round(float(sub.loc["I"].lo), 4),
                "I_hi": round(float(sub.loc["I"].hi), 4),
                "SR_rate": round(float(sub.loc["SR"].rate), 4),
                "SR_lo": round(float(sub.loc["SR"].lo), 4),
                "SR_hi": round(float(sub.loc["SR"].hi), 4),
                **{f"S{stage}": counts.get(stage, 0) for stage in stages},
                "top_failure_modes": "; ".join(
                    f"{mode}:{count}" for mode, count in failure_modes.most_common(3)
                ),
            }
        )
    summary = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    return summary


def run_stage_eval_battery(
    *,
    spec: StageLabelTaskSpec,
    labels_csv: Path,
    plot_dir: Path,
    csv_dir: Path,
    prefix: str,
    title: str,
    arms: list[ArmPlotSpec] | None = None,
    paired_rounds_csv: Path | None = None,
) -> dict[str, Path]:
    """Generate the standard stage-share + rung-conversion eval battery."""
    df = _load_labels(labels_csv, spec)
    if arms is None:
        arms = default_arm_specs(sorted(df["policy_short"].astype(str).unique()))
    present = set(df["policy_short"].astype(str).unique())
    missing = [arm.key for arm in arms if arm.key not in present]
    if missing:
        raise ValueError(f"{labels_csv} missing arms {missing}; have {sorted(present)}")

    # Ground-truth anchor: the human outcome-edited success (in the paired CSV) is
    # authoritative for the top rung, not the VLM's perceptual S7. Keeps the rung
    # SR identical to the headline eval SR. No-op when there is no paired CSV.
    if paired_rounds_csv is not None:
        df = _anchor_stage_to_outcome(df, spec, arms, paired_rounds_csv)

    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    stage_share = plot_dir / f"{prefix}_stage_share.svg"
    rung_conversion = plot_dir / f"{prefix}_rung_conversion.svg"
    rung_csv = csv_dir / f"{prefix}_rung_conversions.csv"
    comparison_csv = csv_dir / f"{prefix}_rung_comparisons.csv"
    summary_csv = csv_dir / f"{prefix}_subtask_summary.csv"

    plot_stage_share(df, spec, arms, title=f"{title} — stage-share by arm", out=stage_share)
    rung_tab, comparisons = plot_rung_conversion(
        df,
        spec,
        arms,
        title=f"{title} — two-rung conversion funnel by arm",
        out=rung_conversion,
        paired_rounds_csv=paired_rounds_csv,
    )
    rung_tab.to_csv(rung_csv, index=False)
    comparisons.to_csv(comparison_csv, index=False)
    write_summary(df, spec, arms, rung_tab, summary_csv)
    return {
        "stage_share": stage_share,
        "rung_conversion": rung_conversion,
        "rung_conversions_csv": rung_csv,
        "rung_comparisons_csv": comparison_csv,
        "summary_csv": summary_csv,
    }
