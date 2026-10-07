"""Shared per-round episode-length distribution figure for real task lines.

Renders the per-round episode-length figure for N policy arms on any registered
:class:`~mulligan.real.lifecycle.tasks.RealTaskSpec`.

For a held-out paired eval it renders one row per outcome category (success /
timeout / terminated) x one lane per policy arm. Episode length is RECOMPUTED
from the eval dataset frames -- through the first ``done=True`` frame (inclusive)
OR through the last ``is_valid=True`` frame -- NOT trusted from the reconciled
``*_num_steps`` column, then cross-checked
against the ingest's ``paired_round_outcomes.csv`` ``*_num_steps`` with any
discrepancy printed loudly and written into the sidecar CSV. There is a known
definitional +1 offset on many timeouts: the outcome-editor reconciliation sets
``num_steps = outcome_frame + 1`` for terminal outcomes but timeouts keep the
raw action-step count, one short of the recomputed frame count. That offset is
preserved and reported, not silently corrected.

On the SUCCESS row two (or more) dashed vertical lines mark each policy family's
R0 TELEOP human-demo median episode length (same done/is_valid rule). The demo
repos come from ``RealTaskSpec.r0_teleop_demo_repos`` -- the R0 teleop splits
ONLY. The R1+ DAgger correction episodes are NOT teleop demos (they contain
policy-driven frames, so their episode length is not a demo length) and are
EXCLUDED; including them biases the demo-length median upward. A line whose spec has ``r0_teleop_demo_repos=None`` skips the median lines
with a LOUD log message rather than guessing.

Convention -- library-first, no CLI: a per-round driver constructs an
:class:`EpisodeLengthConfig` in Python (round-specific repo ids, arms, labels,
color keys, output paths) and calls :func:`run`. Colors MUST come from
``mulligan.plotting.colors`` -- the driver passes ``METHOD_COLORS`` KEYS, never
hardcoded hex.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mulligan.plotting.colors import METHOD_COLORS
from mulligan.real.eval.outcome_results import task_effective_prefix_length
from mulligan.real.lifecycle.plotting import REPO_ROOT
from mulligan.real.lifecycle.tasks import RealTaskSpec, get_task_spec

# Frame-metadata columns required to recompute episode length from the eval frames.
_TERMINAL_COLUMNS = ("episode_index", "frame_index", "done", "is_valid")


@dataclass(frozen=True)
class ArmSpec:
    """One policy arm in the episode-length figure."""

    prefix: str
    """CSV column prefix in ``paired_round_outcomes.csv`` (e.g. ``"baseline"``)."""

    label: str
    """Display label for the arm's lane / count annotation."""

    color_key: str
    """``mulligan.plotting.colors.METHOD_COLORS`` key for this arm's points/median."""


@dataclass(frozen=True)
class DemoMedianSpec:
    """One R0-teleop demo-length median reference line on the SUCCESS row.

    The repos are resolved from ``RealTaskSpec.r0_teleop_demo_repos[demo_set_key]``
    at plot time, so which physical repos back a family stay single-sourced in the
    task registry; this spec only chooses which families to draw, their color, and
    their labels.
    """

    demo_set_key: str
    """Key into ``RealTaskSpec.r0_teleop_demo_repos`` (e.g. ``"uniform"`` / ``"sobol"``)."""

    color_key: str
    """``METHOD_COLORS`` key for the dashed reference line."""

    legend_label: str
    """Legend prefix; ``" = {median:.0f} (n={n})"`` is appended."""

    csv_arm: str
    """``arm`` value written in the sidecar CSV demo-median footer row."""

    csv_arm_label: str
    """``arm_label`` value written in the sidecar CSV demo-median footer row."""


@dataclass(frozen=True)
class EpisodeLengthConfig:
    """Round-specific inputs for one episode-length distribution figure."""

    task: str
    """RealTaskSpec lifecycle key, e.g. ``"routing_d2"``."""

    eval_repo: str
    """HF dataset repo id whose frames the episode lengths are recomputed from."""

    outcomes_csv: Path
    """``paired_round_outcomes.csv`` produced by the held-out eval ingest."""

    out_csv: Path
    """Output per-episode-length sidecar CSV path."""

    out_svg: Path
    """Output SVG path."""

    arms: tuple[ArmSpec, ...]
    """Ordered policy arms (baseline first, per docs/plotting_principles.md)."""

    plot_title: str
    """Figure suptitle."""

    eval_repo_by_arm: dict[str, str] | None = None
    """Optional explicit arm-prefix -> HF repo mapping for cross-repo evals.

    When set, every configured arm must appear exactly once. Episode indices are
    resolved only inside that arm's repo, which prevents colliding indices from
    being silently mapped to the wrong frames in same-start composite analyses.
    ``eval_repo`` remains the value/explorer repo identity at the umbrella level.
    """

    outcome_rows: tuple[tuple[str, str], ...] = (
        ("success", "Successful"),
        ("timeout", "Timeouts"),
        ("failure", "Terminated (early-declared failure)"),
    )
    """``(outcome_value, row_title)`` per row, in top-to-bottom order. The
    ``outcome_value`` matches the ``*_outcome`` labels in ``paired_round_outcomes.csv``."""

    demo_median_sets: tuple[DemoMedianSpec, ...] = ()
    """Demo-length median reference lines to draw on the success row. Empty ⇒ none.
    Skipped LOUDLY (with a log line) if the task spec has no ``r0_teleop_demo_repos``."""

    demo_median_on_outcome: str = "success"
    """The outcome-row whose panel carries the dashed demo-median lines."""

    demo_legend_title: str = "R0 teleop demo medians (dashed, SUCCESS row only)"
    """Figure-legend title for the demo-median lines."""

    expected_num_starts: int | None = None
    """Optional exact paired-start count check on ``outcomes_csv`` (fail loud)."""

    expected_eval_episodes: int | None = None
    """Optional exact recomputed-episode count check on the eval dataset (fail loud)."""

    jitter_seed: int = 20260711
    """Seed for the per-lane vertical scatter jitter (pin per round for reproducibility)."""

    def __post_init__(self) -> None:
        if len(self.arms) < 1:
            raise ValueError("need >= 1 arm")
        prefixes = [arm.prefix for arm in self.arms]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError(f"arm prefixes must be unique, got {prefixes}")
        if self.eval_repo_by_arm is not None:
            got = set(self.eval_repo_by_arm)
            expected = set(prefixes)
            if got != expected:
                raise ValueError(
                    "eval_repo_by_arm keys must exactly match arm prefixes; "
                    f"expected {sorted(expected)}, got {sorted(got)}"
                )
        if not self.outcome_rows:
            raise ValueError("need >= 1 outcome row")
        outcomes = [outcome for outcome, _ in self.outcome_rows]
        if len(set(outcomes)) != len(outcomes):
            raise ValueError(f"outcome_rows values must be unique, got {outcomes}")
        if self.demo_median_sets and self.demo_median_on_outcome not in outcomes:
            raise ValueError(
                f"demo_median_on_outcome {self.demo_median_on_outcome!r} is not an outcome row "
                f"({outcomes})"
            )


# ---------------------------------------------------------------------------
# Core computations (no I/O; unit-tested directly)
# ---------------------------------------------------------------------------


def recompute_lengths(hf_dataset: Any) -> dict[int, int]:
    """``episode_index -> task-relevant effective length``.

    The first ``done=True`` row is included. The first ``is_valid=False`` row is
    excluded, so a timeout ends at the last valid row. Fails loud if an episode
    has no terminal/invalid boundary, non-contiguous frame indices, a non-prefix
    validity mask, or a terminal flag that returns to zero.
    """
    missing = set(_TERMINAL_COLUMNS) - set(hf_dataset.column_names)
    if missing:
        raise ValueError(f"eval dataset missing terminal metadata columns: {sorted(missing)}")
    df = pd.DataFrame(
        {
            "episode_index": np.asarray(hf_dataset["episode_index"], dtype=np.int64),
            "frame_index": np.asarray(hf_dataset["frame_index"], dtype=np.int64),
            "done": np.asarray(hf_dataset["done"], dtype=bool),
            "is_valid": np.asarray(hf_dataset["is_valid"], dtype=bool),
        }
    )
    lengths: dict[int, int] = {}
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        fi = g["frame_index"].to_numpy()
        if fi[0] != 0 or not np.array_equal(fi, np.arange(len(fi))):
            raise ValueError(f"episode {ep}: non-contiguous frame_index {fi[:5]}...")
        done = g["done"].to_numpy()
        is_valid = g["is_valid"].to_numpy()
        if not bool(done.any()) and bool(is_valid.all()):
            raise ValueError(f"episode {ep}: no done=True or is_valid=False frame found")
        lengths[int(ep)] = task_effective_prefix_length(
            done,
            is_valid,
            episode_index=int(ep),
        )
    return lengths


def build_episode_table(
    cfg: EpisodeLengthConfig,
    lengths: dict[int, int],
    df_out: pd.DataFrame,
    *,
    lengths_by_arm: dict[str, dict[int, int]] | None = None,
):
    """Per-episode ``(arm, outcome, recomputed_length, csv_num_steps)`` long table.

    Every arm iterates ALL paired starts, so a missing arm column fails loud
    (KeyError) and an arm that ends up with zero episodes fails loud too (that means
    the ingest is wrong). An empty OUTCOME CATEGORY within an arm is legitimate (e.g.
    no terminated episodes) and simply yields no rows for that category.
    """
    records: list[dict[str, Any]] = []
    for arm in cfg.arms:
        arm_lengths = lengths if lengths_by_arm is None else lengths_by_arm[arm.prefix]
        source_repo = (
            cfg.eval_repo if cfg.eval_repo_by_arm is None else cfg.eval_repo_by_arm[arm.prefix]
        )
        ep_col = f"{arm.prefix}_episode_index"
        out_col = f"{arm.prefix}_outcome"
        ns_col = f"{arm.prefix}_num_steps"
        for _, row in df_out.iterrows():
            ep = int(row[ep_col])
            if ep not in arm_lengths:
                raise ValueError(
                    f"eval episode_index {ep} ({arm.prefix}) not in recomputed lengths "
                    f"for {source_repo}"
                )
            records.append(
                {
                    "arm": arm.prefix,
                    "arm_label": arm.label,
                    "color_key": arm.color_key,
                    "source_repo": source_repo,
                    "episode_index": ep,
                    "manifest_idx": int(row["manifest_idx"]),
                    "outcome": str(row[out_col]),
                    "recomputed_length": arm_lengths[ep],
                    "csv_num_steps": int(row[ns_col]),
                }
            )
    # Count per arm from the records (not the frame) so the check still fires loudly
    # when df_out is empty and the frame carries no columns at all.
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["arm"]] = counts.get(rec["arm"], 0) + 1
    for arm in cfg.arms:
        if counts.get(arm.prefix, 0) == 0:
            raise ValueError(
                f"arm {arm.prefix!r} has zero episodes in {cfg.outcomes_csv}; the ingest is wrong "
                "(every configured arm must cover every paired start)"
            )
    tbl = pd.DataFrame.from_records(records)
    tbl["length_minus_numsteps"] = tbl["recomputed_length"] - tbl["csv_num_steps"]
    return tbl


def demo_median(
    repos: list[str] | tuple[str, ...],
    *,
    load_lengths: Callable[[str], dict[int, int]] | None = None,
) -> tuple[float, int, list[tuple[str, int, int]]]:
    """Median human-demo episode length across ``repos`` (same done|~is_valid rule).

    Returns ``(median_length, n_episodes, [(repo, n_ep, repo_median), ...])``.
    ``load_lengths`` maps a repo id to its ``episode_index -> length`` dict; it
    defaults to a LeRobot loader and is injectable for tests.
    """
    if load_lengths is None:
        load_lengths = _lerobot_lengths
    all_lengths: list[int] = []
    per_repo: list[tuple[str, int, int]] = []
    for repo in repos:
        lengths = load_lengths(repo)
        vals = sorted(lengths.values())
        if not vals:
            raise ValueError(f"demo repo {repo} yielded no episodes")
        all_lengths.extend(vals)
        per_repo.append((repo, len(vals), int(np.median(vals))))
    if not all_lengths:
        raise ValueError(f"no demo episodes found across {list(repos)}")
    return float(np.median(all_lengths)), len(all_lengths), per_repo


def resolve_demo_medians(
    cfg: EpisodeLengthConfig,
    spec: RealTaskSpec,
    *,
    load_lengths: Callable[[str], dict[int, int]] | None = None,
) -> list[tuple[DemoMedianSpec, float, int, list[tuple[str, int, int]]]]:
    """Resolve each configured demo-median line's repos + median from the task spec.

    Returns ``[(DemoMedianSpec, median, n, per_repo), ...]``. If the task spec has no
    ``r0_teleop_demo_repos`` the whole set is SKIPPED with a loud log message and an
    empty list is returned (the plot then omits the demo lines) -- never guessed.
    """
    if not cfg.demo_median_sets:
        return []
    if spec.r0_teleop_demo_repos is None:
        print(
            f"[episode_lengths] SKIPPING demo-median reference lines: task {spec.name!r} has no "
            "verified r0_teleop_demo_repos in its RealTaskSpec (add the R0 teleop splits there to "
            "enable the demo-length medians)."
        )
        return []
    resolved: list[tuple[DemoMedianSpec, float, int, list[tuple[str, int, int]]]] = []
    for demo_spec in cfg.demo_median_sets:
        if demo_spec.demo_set_key not in spec.r0_teleop_demo_repos:
            raise KeyError(
                f"demo set {demo_spec.demo_set_key!r} not in {spec.name} r0_teleop_demo_repos "
                f"(have {sorted(spec.r0_teleop_demo_repos)})"
            )
        repos = spec.r0_teleop_demo_repos[demo_spec.demo_set_key]
        median, n, per_repo = demo_median(repos, load_lengths=load_lengths)
        resolved.append((demo_spec, median, n, per_repo))
    return resolved


def dataset_revision(repo: str) -> str:
    """Released repos are read at their pin (``release/revisions.json``); a repo of a new
    campaign is not released and is read at Hub ``main`` (logged)."""
    from mulligan.release.download import load_revisions

    revisions = load_revisions()
    if repo in revisions:
        return revisions[repo]["revision"]
    print(f"[episode_lengths] {repo} is not a released repo; reading Hub main")
    return "main"


def _lerobot_lengths(repo: str) -> dict[int, int]:
    """R0-demo lengths for one repo via LeRobot (no videos)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=repo,
        revision=dataset_revision(repo),
        force_cache_sync=True,
        download_videos=False,
    )
    return recompute_lengths(ds.hf_dataset)


# ---------------------------------------------------------------------------
# Plot + sidecar CSV
# ---------------------------------------------------------------------------


def plot(
    cfg: EpisodeLengthConfig,
    tbl: pd.DataFrame,
    fps: float,
    demo_medians: list[tuple[DemoMedianSpec, float, int, list[tuple[str, int, int]]]],
) -> Path:
    rng = np.random.default_rng(cfg.jitter_seed)
    x_max = float(tbl["recomputed_length"].max()) * 1.03

    n_arms = len(cfg.arms)
    fig, axes = plt.subplots(len(cfg.outcome_rows), 1, figsize=(10.5, 7.4), sharex=True)
    if len(cfg.outcome_rows) == 1:
        axes = np.array([axes])
    lane_gap = 1.0
    jitter = 0.14

    for ax, (outcome, row_title) in zip(axes, cfg.outcome_rows, strict=True):
        sub = tbl[tbl["outcome"] == outcome]
        for lane, arm in enumerate(cfg.arms):
            arm_sub = sub[sub["arm"] == arm.prefix]
            color = METHOD_COLORS[arm.color_key]
            y0 = (n_arms - 1 - lane) * lane_gap  # baseline on top
            xs = arm_sub["recomputed_length"].to_numpy(dtype=float)
            n = len(xs)
            if n:
                ys = y0 + rng.uniform(-jitter, jitter, size=n)
                ax.scatter(xs, ys, s=42, color=color, edgecolor="white", linewidth=0.6, zorder=3)
                med = float(np.median(xs))
                ax.plot(
                    [med, med],
                    [y0 - 0.30, y0 + 0.30],
                    color=color,
                    lw=2.4,
                    zorder=4,
                    solid_capstyle="butt",
                )
                ax.text(
                    med,
                    y0 + 0.36,
                    f"med {med:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                )
            ax.text(
                -0.012 * x_max,
                y0,
                f"{arm.label}\nn={n}",
                ha="right",
                va="center",
                fontsize=7.5,
                color=color,
            )
        # dashed demo-median lines on the configured outcome row only.
        if outcome == cfg.demo_median_on_outcome:
            for demo_spec, median, _n, _per_repo in demo_medians:
                ax.axvline(
                    median,
                    ls="--",
                    lw=1.5,
                    color=METHOD_COLORS[demo_spec.color_key],
                    alpha=0.9,
                    zorder=1,
                )
        ax.set_ylim(-0.7, (n_arms - 1) * lane_gap + 0.7)
        ax.set_yticks([])
        ax.set_xlim(0, x_max)
        ax.set_title(f"{row_title}   (n={len(sub)})", fontsize=10, loc="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", ls=":", lw=0.5, color="#ddd", zorder=0)

    axes[-1].set_xlabel("episode length (steps, through first done / last valid frame)", fontsize=9)

    # seconds axis on top of the first panel.
    sec_ax = axes[0].secondary_xaxis("top", functions=(lambda s: s / fps, lambda t: t * fps))
    sec_ax.set_xlabel(f"seconds (control freq {fps:.0f} Hz)", fontsize=8.5)

    if demo_medians:
        demo_handles = [
            mlines.Line2D(
                [],
                [],
                ls="--",
                lw=1.5,
                color=METHOD_COLORS[demo_spec.color_key],
                label=f"{demo_spec.legend_label} = {median:.0f} (n={n})",
            )
            for demo_spec, median, n, _per_repo in demo_medians
        ]
        fig.legend(
            handles=demo_handles,
            loc="lower center",
            ncol=len(demo_handles),
            frameon=False,
            fontsize=8,
            title=cfg.demo_legend_title,
            title_fontsize=8.5,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.suptitle(cfg.plot_title, fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.86, bottom=0.14, hspace=0.42)
    cfg.out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return cfg.out_svg


def write_sidecar_csv(
    cfg: EpisodeLengthConfig,
    tbl: pd.DataFrame,
    demo_medians: list[tuple[DemoMedianSpec, float, int, list[tuple[str, int, int]]]],
) -> Path:
    """Write the per-episode sidecar CSV with the demo medians as footer rows."""
    out_cols = [
        "arm",
        "arm_label",
        "source_repo",
        "episode_index",
        "manifest_idx",
        "outcome",
        "recomputed_length",
        "csv_num_steps",
        "length_minus_numsteps",
    ]
    per_ep = tbl[out_cols].copy()
    per_ep.insert(0, "kind", "eval_episode")
    demo_rows = pd.DataFrame(
        [
            {
                "kind": "demo_median",
                "arm": demo_spec.csv_arm,
                "arm_label": demo_spec.csv_arm_label,
                "outcome": "demo",
                "recomputed_length": int(round(median)),
                "csv_num_steps": n,
            }
            for demo_spec, median, n, _per_repo in demo_medians
        ]
    )
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([per_ep, demo_rows], ignore_index=True).to_csv(cfg.out_csv, index=False)
    return cfg.out_csv


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run(
    cfg: EpisodeLengthConfig,
    *,
    load_lengths: Callable[[str], dict[int, int]] | None = None,
) -> dict[str, Path]:
    """Full recompute + cross-check + sidecar CSV + plot.

    Returns ``{"csv": out_csv, "svg": out_svg}``. ``load_lengths`` is injectable for
    tests; production uses the default LeRobot loader for both the eval dataset and
    the demo repos.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    spec = get_task_spec(cfg.task)
    if not cfg.outcomes_csv.exists():
        raise FileNotFoundError(f"missing paired outcomes CSV: {cfg.outcomes_csv}")
    df_out = pd.read_csv(cfg.outcomes_csv)
    if cfg.expected_num_starts is not None and len(df_out) != cfg.expected_num_starts:
        raise ValueError(f"expected {cfg.expected_num_starts} paired starts, got {len(df_out)}")

    repo_by_arm = cfg.eval_repo_by_arm or {arm.prefix: cfg.eval_repo for arm in cfg.arms}
    lengths_by_repo: dict[str, dict[int, int]] = {}
    fps_by_repo: dict[str, float] = {}
    for repo_id in dict.fromkeys(repo_by_arm.values()):
        eval_ds = LeRobotDataset(
            repo_id=repo_id,
            revision=dataset_revision(repo_id),
            force_cache_sync=True,
            download_videos=False,
        )
        fps_by_repo[repo_id] = float(eval_ds.meta.fps)
        lengths_by_repo[repo_id] = recompute_lengths(eval_ds.hf_dataset)
    if len(set(fps_by_repo.values())) != 1:
        raise ValueError(f"cross-repo eval FPS mismatch: {fps_by_repo}")
    fps = next(iter(fps_by_repo.values()))
    total_eval_episodes = sum(len(v) for v in lengths_by_repo.values())
    if cfg.expected_eval_episodes is not None and total_eval_episodes != cfg.expected_eval_episodes:
        raise ValueError(
            f"expected {cfg.expected_eval_episodes} eval episodes across "
            f"{sorted(lengths_by_repo)}, got {total_eval_episodes}"
        )

    lengths_by_arm = {arm.prefix: lengths_by_repo[repo_by_arm[arm.prefix]] for arm in cfg.arms}
    # The flat mapping preserves the single-repo API and is ignored for a
    # cross-repo config, where lengths_by_arm is authoritative.
    flat_lengths = next(iter(lengths_by_repo.values()))
    tbl = build_episode_table(cfg, flat_lengths, df_out, lengths_by_arm=lengths_by_arm)

    # ---- cross-check recomputed length vs reconciled num_steps ----
    disc = tbl[tbl["length_minus_numsteps"] != 0]
    print(f"episodes: {len(tbl)}  ({len(tbl)} = {len(cfg.arms)} arms x {len(df_out)} starts)")
    print("recomputed-length vs csv_num_steps discrepancies:", len(disc))
    if len(disc):
        by = disc.groupby("outcome")["length_minus_numsteps"]
        print(by.agg(["count", "min", "max", "mean"]).to_string())
        print(
            disc[
                [
                    "arm",
                    "episode_index",
                    "outcome",
                    "recomputed_length",
                    "csv_num_steps",
                    "length_minus_numsteps",
                ]
            ].to_string(index=False)
        )

    # ---- per-cell counts + medians ----
    print("\nper-cell (outcome x arm) counts + median recomputed length:")
    for outcome, _ in cfg.outcome_rows:
        for arm in cfg.arms:
            s = tbl[(tbl["outcome"] == outcome) & (tbl["arm"] == arm.prefix)]["recomputed_length"]
            med = f"{float(np.median(s)):.0f}" if len(s) else "-"
            print(f"  {outcome:8s} {arm.label:28s} n={len(s):2d}  median={med}")

    # ---- demo medians ----
    demo_medians = resolve_demo_medians(cfg, spec, load_lengths=load_lengths)
    if demo_medians:
        print("\ndemo-set medians (R0 teleop splits only; R1+ DAgger corrections excluded):")
        for demo_spec, median, n, per_repo in demo_medians:
            print(f"  {demo_spec.legend_label}: median={median:.0f} over n={n} episodes")
            for repo, nrep, medrep in per_repo:
                print(f"    {repo}: n={nrep} median={medrep}")

    write_sidecar_csv(cfg, tbl, demo_medians)
    print(f"\nwrote {_display(cfg.out_csv)}")
    plot(cfg, tbl, fps, demo_medians)
    print(f"wrote {_display(cfg.out_svg)}")
    return {"csv": cfg.out_csv, "svg": cfg.out_svg}
