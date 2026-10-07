"""Real-robot start selection (paper Alg. 2, ``SelectInitialStates``) for Marker and Nut.

A DAgger round collects ``budget`` starts for each of two arms. The Ours arm gets

1. **failure promotion**: the exact starts of the previous evaluation's failures, ranked
   earliest breakdown first (stage labels) and capped at ``cap``;
2. **coverage fill**: fresh scrambled-Sobol candidates chosen by farthest-point selection
   against every collected Ours start plus the promotions, scored ``d_min * (1 + beta * h)``
   with an optional hardness term ``h`` and down-weighted placement axes.

The baseline arm gets fresh flat-uniform starts. The collection manifest then pairs one
baseline and one Ours start at a time in an operator-friendly order: holder-batched for
Marker, peg-batched with an optimal baseline assignment for Nut.

Everything is deterministic given a round config (``configs/real/<task>/rNN_sampler.yaml``)
and the recorded inputs it names: the evaluation outcome table, its stage labels, and the
previously placed manifests. Only the rules the locked Marker and Nut R1-R5 manifests used
are implemented; each round's config selects among them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from mulligan.real.lifecycle.geometry import TaskGeometry, wrap_angle
from mulligan.real.lifecycle.tasks import RealTaskSpec, get_task_spec

INCH_TO_M = 0.0254
BASELINE_ARM = "baseline_uniform"
MULLIGAN_ARM = "mulligan_sobol"
FILL_SEGMENT = "fps_fill"
EXACT_DUP_TOL = 1e-9
MANIFEST_SCHEMA = "real_manual_initial_states_v1"
RANK_KEYS = ("tier", "phase", "stage", "num_steps", "tiebreak")


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class PromotionConfig:
    """Which evaluation failures are promoted and in what order.

    ``pool`` is one of
      * ``all_fail``: starts where every arm in ``arms`` failed (one arm = its failures);
        stage, steps and outcome come from the first listed arm;
      * ``any_fail``: starts where any arm in ``arms`` failed; stage, steps and outcome
        come from the first failing arm in list order;
      * ``tiered``: the Marker R5 tiers over the actor/reranked arm pair (``tiers``);
      * ``table``: a precomputed per-start failure table (``table`` options).
    Candidates are sorted by ``rank_keys`` (a subset of ``RANK_KEYS``) and the first
    ``cap`` are promoted. ``phase`` is 0 for starts that broke down at or before
    ``upstream_max_stage`` and 1 otherwise.
    """

    pool: str
    cap: int
    rank_keys: tuple[str, ...]
    tiebreak: str
    upstream_max_stage: int
    arms: dict[str, str] = field(default_factory=dict)
    stage_column: str = ""
    failure_mode_column: str = ""
    tiers: dict[str, Any] = field(default_factory=dict)
    table: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FillConfig:
    """Coverage fill: Sobol candidate pool, hardness tilt and optional holder-band quotas.

    ``hardness`` is ``None`` (pure coverage), ``holder_x_risk`` (smoothed per-holder-depth
    failure rate of the listed success columns of the evaluation table) or ``nut_pose``
    (``(1 - g_weight) * sum_i w_i * term_i + g_weight * g_radial``). ``bands`` is ``None``,
    ``{"mode": "quota", "band": b, "fraction": f}`` (at least ``ceil(f * fills)`` fills at
    holder depth ``b`` inches, selected first) or ``{"mode": "balanced", "bands": [...]}``
    (fills split evenly across holder depths, remainder to the first bands).
    """

    seed: int
    pool_size: int
    beta: float
    placement_weight: float
    hardness: str | None = None
    hardness_params: dict[str, Any] = field(default_factory=dict)
    bands: dict[str, Any] | None = None


@dataclass(frozen=True)
class OrderConfig:
    """Collection order of the paired manifest.

    ``holder_pairs`` (Marker): Ours segments in order, each paired with holder-matched
    baseline rows along a nearest-neighbour holder walk. ``segments`` is ``phase``
    (upstream promotions, later-stage promotions, fills) or ``tier`` (one segment per
    promotion tier). ``alternate_arms`` flips pairs so the two arms strictly alternate.

    ``peg_path`` (Nut): the Ours queue is peg-batched per segment, then baseline rows are
    assigned to the Ours path by a linear assignment. ``split_by_phase`` orders upstream
    promotions before later-stage ones. ``lead_if_failed`` moves promotions on which that
    arm did not fail into a peg-and-class even-spread tail with the fills.
    """

    strategy: str
    seed: int
    segments: str = "phase"
    segment_names: dict[str, str] = field(default_factory=dict)
    alternate_arms: bool = False
    split_by_phase: bool = False
    lead_if_failed: str | None = None


@dataclass(frozen=True)
class RoundConfig:
    task: str
    round: int
    budget: int
    source_eval_seed: int
    baseline_seed: int
    promotion: PromotionConfig
    fill: FillConfig
    order: OrderConfig
    inputs: dict[str, Any]


def load_round_config(path: str | Path) -> RoundConfig:
    raw = yaml.safe_load(Path(path).read_text())
    promo = dict(raw["promotion"])
    promo["rank_keys"] = tuple(promo["rank_keys"])
    unknown = set(promo["rank_keys"]) - set(RANK_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown promotion.rank_keys {sorted(unknown)}")
    return RoundConfig(
        task=raw["task"],
        round=int(raw["round"]),
        budget=int(raw["budget"]),
        source_eval_seed=int(raw["source_eval_seed"]),
        baseline_seed=int(raw["baseline_seed"]),
        promotion=PromotionConfig(**promo),
        fill=FillConfig(**raw["fill"]),
        order=OrderConfig(**raw["order"]),
        inputs=dict(raw["inputs"]),
    )


# ---------------------------------------------------------------- recorded inputs


def manifest_states(
    path: str | Path, keys: tuple[str, ...], source: str | None = None
) -> np.ndarray:
    """``(n, K)`` start array of a placed manifest, optionally restricted to one arm."""
    rows = json.loads(Path(path).read_text())["states"]
    if source is not None:
        rows = [row for row in rows if row["source"] == source]
    missing = [key for key in keys if rows and key not in rows[0]]
    if missing:
        raise ValueError(f"{path}: states are missing keys {missing}")
    return np.asarray(
        [[float(row[key]) for key in keys] for row in rows], dtype=np.float64
    ).reshape(-1, len(keys))


def join_stage_labels(
    outcomes: pd.DataFrame,
    labels: pd.DataFrame,
    arms: dict[str, str],
    stage_column: str,
    failure_mode_column: str,
) -> pd.DataFrame:
    """Attach each arm's stage label and failure mode to the paired outcome table.

    ``arms`` maps the outcome-table column prefix (``<prefix>_success``, ``_outcome``,
    ``_num_steps``, ``_episode_index``) to the label table's ``policy_short``. Adds
    ``<prefix>_stage`` and ``<prefix>_failure_mode``.
    """
    out = outcomes.copy()
    for prefix, policy_short in arms.items():
        lab = labels[labels["policy_short"] == policy_short].set_index("episode_index")
        if lab.index.duplicated().any():
            raise ValueError(f"stage labels of {policy_short!r} repeat an episode_index")
        episodes = out[f"{prefix}_episode_index"].astype(int)
        missing = sorted(set(episodes) - set(lab.index))
        if missing:
            raise ValueError(f"stage labels of {policy_short!r} lack episodes {missing}")
        if "original_outcome" in lab.columns:
            label_outcome = episodes.map(lab["original_outcome"]).astype(str)
            if not label_outcome.eq(out[f"{prefix}_outcome"].astype(str)).all():
                raise ValueError(f"stage labels of {policy_short!r} disagree with the outcomes")
        out[f"{prefix}_stage"] = episodes.map(lab[stage_column]).astype(int)
        if failure_mode_column:
            out[f"{prefix}_failure_mode"] = episodes.map(lab[failure_mode_column]).fillna("")
        else:
            out[f"{prefix}_failure_mode"] = ""
    return out


def _as_candidates(rows: pd.DataFrame, rank_arm: pd.Series, cfg: PromotionConfig) -> pd.DataFrame:
    """Canonical promotion-candidate columns from a stage-labelled outcome table."""
    out = rows.copy()
    out["rank_arm"] = rank_arm.to_numpy()
    pick = {col: [] for col in ("stage", "num_steps", "outcome", "failure_mode")}
    for arm, (_, row) in zip(out["rank_arm"], out.iterrows(), strict=True):
        pick["stage"].append(int(row[f"{arm}_stage"]))
        pick["num_steps"].append(int(row[f"{arm}_num_steps"]))
        pick["outcome"].append(str(row[f"{arm}_outcome"]))
        pick["failure_mode"].append(str(row[f"{arm}_failure_mode"]))
    for col, values in pick.items():
        out[col] = values
    out["failed_arms"] = [
        "+".join(arm for arm in cfg.arms if not bool(row[f"{arm}_success"]))
        for _, row in out.iterrows()
    ]
    out["tier"] = 0
    out["group"] = ""
    out["stream_index"] = out["sobol_stream_index"].astype(int)
    out["manifest_idx"] = out["manifest_idx"].astype(int)
    return out


def _marker_tiers(ev: pd.DataFrame, cfg: PromotionConfig) -> pd.DataFrame:
    """Marker R5 promotion tiers over an actor arm and its reranked arm.

    1. actor insert-phase failures that froze at the hole (``freeze_failure_mode``);
    2. starts the actor solved but the reranked arm failed (critic supervision);
    3. the remaining actor insert-phase failures;
    4. actor grasp-phase failures in the extreme pen-pose corner.
    Stage, steps and outcome come from the actor, except tier 2 (reranked arm).
    """
    t = cfg.tiers
    actor, reranked = t["actor_arm"], t["reranked_arm"]
    actor_fail = ~ev[f"{actor}_success"].astype(bool)
    reranked_fail = ~ev[f"{reranked}_success"].astype(bool)
    insert = ev[f"{actor}_stage"].astype(int) >= int(t["insert_stage"])
    freeze = actor_fail & insert & ev[f"{actor}_failure_mode"].eq(t["freeze_failure_mode"])
    pen_r = np.hypot(ev["pen_x"].to_numpy(np.float64), ev["pen_y"].to_numpy(np.float64))
    corner = (pen_r >= float(t["pen_r_min_m"])) | (
        ev["pen_y"].abs() > float(t["pen_y_abs_min_in"]) * INCH_TO_M
    )
    masks = [
        (t["names"][0], freeze, actor),
        (t["names"][1], ~actor_fail & reranked_fail, reranked),
        (t["names"][2], actor_fail & insert & ~freeze, actor),
        (t["names"][3], actor_fail & ~insert & corner, actor),
    ]
    parts = []
    for tier, (name, mask, arm) in enumerate(masks, start=1):
        part = _as_candidates(ev[mask], pd.Series([arm] * int(mask.sum())), cfg)
        part["tier"] = tier
        part["group"] = name
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    if out["manifest_idx"].duplicated().any():
        raise ValueError("promotion tiers overlap")
    return out


def _failure_table(table: pd.DataFrame, cfg: PromotionConfig) -> pd.DataFrame:
    """A precomputed failure table: one row per failed start with its breakdown rung."""
    t = cfg.table
    out = table[table[t["filter_column"]].eq(t["filter_value"])].copy()
    out["stage"] = out[t["stage_column"]].astype(int)
    out["num_steps"] = out["num_steps"].astype(int)
    out["outcome"] = out["outcome"].astype(str)
    out["failure_mode"] = out[t["failure_mode_column"]].astype(str)
    out["stream_index"] = out["sobol_stream_index"].astype(int)
    out["manifest_idx"] = out[t["manifest_idx_column"]].astype(int)
    out["failed_arms"] = t["filter_value"]
    out["tier"] = 0
    out["group"] = ""
    return out


def promotion_candidates(cfg: PromotionConfig, inputs: dict[str, Path]) -> pd.DataFrame:
    """Every promotable start of the round's evaluation with its ranking columns."""
    if cfg.pool == "table":
        return _failure_table(pd.read_csv(inputs["failure_table"]), cfg)
    outcomes = pd.read_csv(inputs["eval_outcomes"]).sort_values("manifest_idx")
    labels = pd.read_csv(inputs["stage_labels"])
    ev = join_stage_labels(
        outcomes.reset_index(drop=True), labels, cfg.arms, cfg.stage_column, cfg.failure_mode_column
    )
    if cfg.pool == "tiered":
        return _marker_tiers(ev, cfg)
    failed = pd.DataFrame({arm: ~ev[f"{arm}_success"].astype(bool) for arm in cfg.arms})
    if cfg.pool == "all_fail":
        mask = failed.all(axis=1)
        rank_arm = pd.Series([next(iter(cfg.arms))] * int(mask.sum()))
    elif cfg.pool == "any_fail":
        mask = failed.any(axis=1)
        rank_arm = failed[mask].idxmax(axis=1).reset_index(drop=True)
    else:
        raise ValueError(f"unknown promotion pool {cfg.pool!r}")
    return _as_candidates(ev[mask].reset_index(drop=True), rank_arm, cfg)


def promote_failures(candidates: pd.DataFrame, cfg: PromotionConfig) -> pd.DataFrame:
    """Rank candidates earliest breakdown first (per ``rank_keys``) and keep ``cap``."""
    ranked = candidates.copy()
    ranked["phase"] = (ranked["stage"] > cfg.upstream_max_stage).astype(int)
    ranked["upstream"] = ranked["phase"].eq(0)
    ranked["tiebreak"] = ranked[cfg.tiebreak].astype(int)
    if ranked["tiebreak"].duplicated().any():
        raise ValueError(f"promotion tiebreak column {cfg.tiebreak!r} is not unique")
    ranked = ranked.sort_values(list(cfg.rank_keys)).reset_index(drop=True)
    promoted = ranked.head(cfg.cap).copy()
    promoted["selection_rank"] = np.arange(1, len(promoted) + 1)
    return promoted


# ------------------------------------------------------------------ coverage fill


def fresh_candidates(
    geom: TaskGeometry, seed: int, n: int, avoid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Snapped Sobol candidates of stream ``seed`` that repeat no previously placed start.

    Returns the kept candidates and their positions in the Sobol stream."""
    pool = geom.sobol(seed, n)
    keep = geom.pairwise_min_distance(pool, avoid) > EXACT_DUP_TOL
    return pool[keep], np.flatnonzero(keep)


def holder_depth(arr: np.ndarray, keys: tuple[str, ...]) -> np.ndarray:
    """Holder depth in whole inches (the Marker holder_x grid row)."""
    return np.rint(arr[:, keys.index("holder_x")] / INCH_TO_M).astype(int)


def holder_x_risk(outcomes: pd.DataFrame, success_columns: list[str]) -> dict[int, float]:
    """Laplace-smoothed failure rate per holder depth, min-max normalized to [0, 1]."""
    depth = np.rint(outcomes["holder_x"].to_numpy(np.float64) / INCH_TO_M).astype(int)
    smooth = {}
    for hx in sorted(set(depth.tolist())):
        sub = outcomes[depth == hx]
        k = int(sum(int(sub[col].astype(bool).sum()) for col in success_columns))
        n = len(sub) * len(success_columns)
        smooth[hx] = 1.0 - (k + 1.0) / (n + 2.0)
    lo, hi = min(smooth.values()), max(smooth.values())
    if not hi > lo:
        raise ValueError("holder-depth risk has no dynamic range")
    return {hx: (value - lo) / (hi - lo) for hx, value in smooth.items()}


def nut_pose_hardness(
    arr: np.ndarray, geom: TaskGeometry, weights: dict[str, float], g_weight: float
) -> np.ndarray:
    """Nut-pose hardness ``(1 - g_weight) * sum_i w_i * term_i + g_weight * g_radial``.

    Terms (all in [0, 1]): ``x_edge`` = |x| / x_max (forward reach), ``y_edge`` = |y| / y_max,
    ``edge_relu`` = relu((edge_norm - 0.6) / 0.4), ``yaw`` = |wrap(yaw)| / pi. ``g_radial`` =
    0.5 + 0.5 * cos(yaw) * x / x_max is high when the nut handle points along the reach axis.
    """
    keys = geom.keys
    x = arr[:, keys.index("nut_x")]
    y = arr[:, keys.index("nut_y")]
    yaw = arr[:, keys.index("nut_yaw")]
    x_max, y_max = geom.bounds[0, 1], geom.bounds[1, 1]
    terms = {
        "x_edge": lambda: np.abs(x) / x_max,
        "y_edge": lambda: np.abs(y) / y_max,
        "edge_relu": lambda: np.clip((geom.edge_norm(arr) - 0.6) / 0.4, 0.0, None),
        "yaw": lambda: np.abs(wrap_angle(yaw)) / np.pi,
    }
    feature = None
    for name, weight in weights.items():
        term = weight * terms[name]()
        feature = term if feature is None else feature + term
    g_radial = np.clip(0.5 + 0.5 * (np.cos(yaw) * (x / x_max)), 0.0, 1.0)
    return (1.0 - g_weight) * feature + g_weight * g_radial


def fill_hardness(
    cfg: FillConfig, geom: TaskGeometry, pool: np.ndarray, inputs: dict[str, Path]
) -> np.ndarray:
    if cfg.hardness is None:
        return np.zeros(len(pool), dtype=np.float64)
    if cfg.hardness == "holder_x_risk":
        risk = holder_x_risk(
            pd.read_csv(inputs["eval_outcomes"]), cfg.hardness_params["success_columns"]
        )
        depth = holder_depth(pool, geom.keys)
        missing = sorted(set(depth.tolist()) - set(risk))
        if missing:
            raise ValueError(f"no holder-depth risk for candidate depths {missing}")
        return np.array([risk[int(hx)] for hx in depth], dtype=np.float64)
    if cfg.hardness == "nut_pose":
        p = cfg.hardness_params
        return nut_pose_hardness(pool, geom, dict(p["weights"]), float(p["g_weight"]))
    raise ValueError(f"unknown fill hardness {cfg.hardness!r}")


def band_phases(cfg: FillConfig, n_fill: int) -> list[tuple[int | None, int]]:
    """``(holder depth or None for the whole pool, count)`` selection phases."""
    if cfg.bands is None:
        return [(None, n_fill)]
    if cfg.bands["mode"] == "quota":
        quota = math.ceil(n_fill * float(cfg.bands["fraction"]))
        return [(int(cfg.bands["band"]), quota), (None, n_fill - quota)]
    if cfg.bands["mode"] == "balanced":
        bands = [int(b) for b in cfg.bands["bands"]]
        base, rem = divmod(n_fill, len(bands))
        return [(b, base + (1 if i < rem else 0)) for i, b in enumerate(bands)]
    raise ValueError(f"unknown fill band mode {cfg.bands['mode']!r}")


def coverage_fill(
    geom: TaskGeometry,
    pool: np.ndarray,
    hardness: np.ndarray,
    support: np.ndarray,
    cfg: FillConfig,
    n_fill: int,
) -> list[int]:
    """Weighted farthest-point fill (``d_min * (1 + beta * h)``), phase by phase.

    Each phase selects from one holder depth (or the whole pool) against the support
    plus every earlier pick, so a quota is a floor, not a cap."""
    axis_weights = geom.placement_axis_weights(cfg.placement_weight)
    picks: list[int] = []
    prior = support
    for band, count in band_phases(cfg, n_fill):
        if count == 0:
            continue
        idx = (
            np.arange(len(pool))
            if band is None
            else np.flatnonzero(holder_depth(pool, geom.keys) == band)
        )
        if len(idx) < count:
            raise ValueError(f"holder depth {band}: {len(idx)} candidates for {count} fills")
        local = geom.weighted_fps(pool[idx], hardness[idx], prior, cfg.beta, count, axis_weights)
        chosen = [int(idx[i]) for i in local]
        picks.extend(chosen)
        prior = np.vstack([prior, pool[chosen]])
    if len(set(picks)) != n_fill:
        raise ValueError(f"coverage fill returned {len(set(picks))} unique picks for {n_fill}")
    return picks


# ------------------------------------------------------------------- start design


@dataclass
class StartDesign:
    """Ours promotions (ranks 1..P), Ours fills (ranks P+1..budget) and baseline starts."""

    promotions: pd.DataFrame
    fills: pd.DataFrame
    baseline: pd.DataFrame


def resolve_inputs(
    cfg: RoundConfig,
    root: str | Path,
    overrides: dict[str, str | Path | Sequence[str | Path]] | None = None,
) -> dict[str, Any]:
    """Absolute input paths: config paths are relative to ``root``; overrides replace them.

    An override of a list input (``support_manifests``, ``avoid_manifests``) is a sequence of
    paths and replaces the whole list; an override of a single-path input is one path.
    """
    root = Path(root)
    out: dict[str, Any] = {}
    for name, value in cfg.inputs.items():
        if isinstance(value, list):
            out[name] = [root / v for v in value]
        else:
            out[name] = root / value
    for name, value in (overrides or {}).items():
        if name not in out:
            raise ValueError(f"unknown input {name!r}; known: {sorted(out)}")
        many = not isinstance(value, (str, Path))
        if isinstance(out[name], list):
            out[name] = [Path(v) for v in value] if many else [Path(value)]
        elif many:
            raise ValueError(f"input {name!r} takes one path, got {list(value)}")
        else:
            out[name] = Path(value)
    return out


def parse_input_overrides(items: Sequence[str], cfg: RoundConfig) -> dict[str, str | list[str]]:
    """``NAME=PATH`` flags -> overrides; repeat a flag to give a list input several paths."""
    grouped: dict[str, list[str]] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not path:
            raise ValueError(f"--input expects NAME=PATH, got {item!r}")
        grouped.setdefault(name, []).append(path)
    out: dict[str, str | list[str]] = {}
    for name, paths in grouped.items():
        if isinstance(cfg.inputs.get(name), list):
            out[name] = paths
        elif len(paths) > 1:
            raise ValueError(f"input {name!r} takes one path, got {paths}")
        else:
            out[name] = paths[0]
    return out


def build_start_design(cfg: RoundConfig, inputs: dict[str, Any]) -> StartDesign:
    spec = get_task_spec(cfg.task)
    geom = TaskGeometry(spec)
    keys = geom.keys

    promotions = promote_failures(promotion_candidates(cfg.promotion, inputs), cfg.promotion)
    n_fill = cfg.budget - len(promotions)

    support = np.vstack(
        [manifest_states(path, keys, MULLIGAN_ARM) for path in inputs["support_manifests"]]
        + [promotions[list(keys)].to_numpy(np.float64)]
    )
    avoid = np.vstack([manifest_states(path, keys) for path in inputs["avoid_manifests"]])
    pool, stream = fresh_candidates(geom, cfg.fill.seed, cfg.fill.pool_size, avoid)
    hardness = fill_hardness(cfg.fill, geom, pool, inputs)
    picks = coverage_fill(geom, pool, hardness, support, cfg.fill, n_fill)
    fills = pd.DataFrame(pool[picks], columns=list(keys))
    fills["selection_rank"] = np.arange(len(promotions) + 1, cfg.budget + 1)
    fills["fill_stream_index"] = stream[picks].astype(int)
    fills["fps_hardness"] = hardness[picks]

    baseline = pd.DataFrame(geom.uniform(cfg.baseline_seed, cfg.budget), columns=list(keys))
    baseline["selection_rank"] = np.arange(1, cfg.budget + 1)
    if np.any(geom.pairwise_min_distance(baseline[list(keys)].to_numpy(), avoid) <= EXACT_DUP_TOL):
        raise ValueError("a fresh baseline start repeats a previously placed start")
    for name, frame in (("promotions", promotions), ("fills", fills), ("baseline", baseline)):
        _check_on_grid(spec, frame, name)
    return StartDesign(promotions=promotions, fills=fills, baseline=baseline)


def _check_on_grid(spec: RealTaskSpec, frame: pd.DataFrame, name: str) -> None:
    for placement in spec.sampled_placements.values():
        xk, yk = placement.keys
        for x, y in frame[[xk, yk]].itertuples(index=False):
            if not placement.is_on_grid(float(x), float(y)):
                raise ValueError(f"{name}: placement {xk}/{yk} = ({x}, {y}) is off grid")


# ----------------------------------------------------------- Marker: holder pairs

Key = tuple[float, float]


def _placement_key(row: dict[str, Any], keys: tuple[str, str], decimals: int | None) -> Key:
    x, y = float(row[keys[0]]) / INCH_TO_M, float(row[keys[1]]) / INCH_TO_M
    if decimals is None:
        return (int(round(x)), int(round(y)))
    return (round(x, decimals), round(y, decimals))


def _distance(a: Key, b: Key) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _itinerary(keys: set[Key], rng: random.Random) -> list[Key]:
    """Nearest-neighbour walk over placement cells with seeded tie breaks."""
    if not keys:
        return []
    tie = {key: rng.random() for key in sorted(keys)}
    current = min(keys, key=lambda key: (tie[key], key))
    remaining = set(keys)
    order = []
    while remaining:
        order.append(current)
        remaining.remove(current)
        if remaining:
            current = min(remaining, key=lambda key: (_distance(current, key), tie[key], key))
    return order


def _group(rows: list[dict[str, Any]], key_fn) -> dict[Key, list[dict[str, Any]]]:
    grouped: dict[Key, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped


def _pop_baseline_segment(pool, ours_rows, future_rows, rng, key_fn) -> list[dict[str, Any]]:
    """One baseline row per Ours row: exact holder matches first, then the nearest
    holders whose rows later segments do not need."""
    demand = Counter(key_fn(row) for row in ours_rows)
    future = Counter(key_fn(row) for row in future_rows)
    chosen = []
    for key in _itinerary(set(demand), rng):
        for _ in range(min(demand[key], len(pool.get(key, [])))):
            chosen.append(pool[key].pop(0))
    needed = len(ours_rows) - len(chosen)
    if needed <= 0:
        return chosen
    candidates = []
    for key, rows in pool.items():
        if not rows:
            continue
        surplus = len(rows) - future[key]
        nearest = min((_distance(key, k) for k in demand), default=0)
        for row in rows:
            candidates.append((1 if surplus <= 0 else 0, nearest, -surplus, rng.random(), key, row))
    if len(candidates) < needed:
        raise ValueError(f"needed {needed} baseline rows but only {len(candidates)} remain")
    candidates.sort()
    fill_ids = {id(item[-1]) for item in candidates[:needed]}
    for key in list(pool):
        kept = []
        for row in pool[key]:
            (chosen if id(row) in fill_ids else kept).append(row)
        pool[key] = kept
    return chosen


def _schedule_segment_pairs(segment, ours_rows, baseline_rows, rng, key_fn):
    ours_by = _group(ours_rows, key_fn)
    base_by = _group(baseline_rows, key_fn)
    for rows in ours_by.values():
        rng.shuffle(rows)
    for rows in base_by.values():
        rng.shuffle(rows)
    pairs = []
    for holder in _itinerary(set(ours_by) | set(base_by), rng):
        while ours_by.get(holder):
            ours = ours_by[holder].pop(0)
            if base_by.get(holder):
                base = base_by[holder].pop(0)
            else:
                available = sorted(k for k, rows in base_by.items() if rows)
                if not available:
                    raise ValueError(f"{segment}: no baseline row left to pair")
                tie = {k: rng.random() for k in available}
                near = min(available, key=lambda k: (_distance(holder, k), tie[k], k))
                base = base_by[near].pop(0)
            pair = [(BASELINE_ARM, base), (MULLIGAN_ARM, ours)]
            if rng.randrange(2):
                pair.reverse()
            pairs.append((segment, pair))
    if any(base_by.values()):
        raise ValueError(f"{segment}: baseline rows left unpaired")
    return pairs


def order_holder_pairs(
    cfg: OrderConfig,
    segments: list[tuple[str, list[dict]]],
    baseline_rows: list[dict],
    keys: tuple[str, str],
) -> list[tuple[str, list[tuple[str, dict]]]]:
    rng = random.Random(cfg.seed)

    def key_fn(row):
        return _placement_key(row, keys, None)

    pool = _group(baseline_rows, key_fn)
    pairs = []
    for i, (segment, rows) in enumerate(segments):
        future = [row for _, later in segments[i + 1 :] for row in later]
        base = _pop_baseline_segment(pool, rows, future, rng, key_fn)
        pairs.extend(_schedule_segment_pairs(segment, rows, base, rng, key_fn))
    if any(pool.values()):
        raise ValueError("baseline rows left unscheduled")
    if cfg.alternate_arms:
        previous = None
        for _, pair in pairs:
            if previous is not None and pair[0][0] == previous:
                pair.reverse()
            previous = pair[-1][0]
    return pairs


# ---------------------------------------------------------------- Nut: peg path


def _peg_batched(rows: list[dict], rng: random.Random, key_fn) -> list[dict]:
    by_peg = _group(rows, key_fn)
    for peg_rows in by_peg.values():
        rng.shuffle(peg_rows)
    return [row for peg in _itinerary(set(by_peg), rng) for row in by_peg[peg]]


def _even_spread_tail(
    tail: list[tuple[str, dict]], rng: random.Random, key_fn
) -> list[tuple[str, dict]]:
    """Order a tail so any dropped suffix is unbiased across pegs and row classes: each row
    sits at the even-spread position of its peg group, then of its class group."""

    def positions(group_of) -> dict[int, float]:
        groups: dict[Any, list[int]] = defaultdict(list)
        for idx in range(len(tail)):
            groups[group_of(idx)].append(idx)
        pos = {}
        for key in sorted(groups, key=str):
            members = list(groups[key])
            rng.shuffle(members)
            for i, idx in enumerate(members):
                pos[idx] = (i + 0.5) / len(members)
        return pos

    peg_pos = positions(lambda idx: key_fn(tail[idx][1]))
    cls_pos = positions(lambda idx: tail[idx][0])
    keyed = sorted((peg_pos[i], cls_pos[i], rng.random(), i) for i in range(len(tail)))
    return [tail[i] for *_, i in keyed]


def _assign_baseline(baseline_rows: list[dict], ours_path: list[dict], key_fn) -> list[dict]:
    """Assign baseline rows to Ours slots: same peg as the slot (and the previous slot)
    first, then Manhattan peg distance, then baseline rank."""
    n = len(ours_path)
    ours_pegs = [key_fn(row) for row in ours_path]
    base_pegs = [key_fn(row) for row in baseline_rows]
    cost = np.zeros((n, n), dtype=np.float64)
    for slot, peg in enumerate(ours_pegs):
        prev = ours_pegs[slot - 1] if slot else None
        for j, base_peg in enumerate(base_pegs):
            primary = int(base_peg != peg)
            distance = _distance(base_peg, peg)
            if prev is not None:
                primary += int(prev != base_peg)
                distance += _distance(prev, base_peg)
            cost[slot, j] = (
                primary * 1_000_000.0 + distance * 1_000.0 + int(baseline_rows[j]["selection_rank"])
            )
    slots, cols = linear_sum_assignment(cost)
    if slots.tolist() != list(range(n)):
        raise ValueError("unexpected baseline assignment slot order")
    return [baseline_rows[j] for j in cols]


def order_peg_path(
    cfg: OrderConfig,
    verbatim: list[dict],
    fills: list[dict],
    baseline_rows: list[dict],
    keys: tuple[str, str],
) -> list[tuple[str, list[tuple[str, dict]]]]:
    rng = random.Random(cfg.seed)
    names = cfg.segment_names

    def key_fn(row):
        return _placement_key(row, keys, 1)

    def verbatim_order(rows):
        if not cfg.split_by_phase:
            return _peg_batched(rows, rng, key_fn)
        upstream = [row for row in rows if row["upstream"]]
        later = [row for row in rows if not row["upstream"]]
        return _peg_batched(upstream, rng, key_fn) + _peg_batched(later, rng, key_fn)

    if cfg.lead_if_failed is None:
        ours = [(names["verbatim"], row) for row in verbatim_order(verbatim)]
        ours += [(FILL_SEGMENT, row) for row in _peg_batched(fills, rng, key_fn)]
    else:
        is_lead = [cfg.lead_if_failed in row["failed_arms"].split("+") for row in verbatim]
        lead = [row for row, flag in zip(verbatim, is_lead, strict=True) if flag]
        demoted = [row for row, flag in zip(verbatim, is_lead, strict=True) if not flag]
        ours = [(names["lead"], row) for row in verbatim_order(lead)]
        tail = [(names["demoted"], row) for row in demoted] + [(FILL_SEGMENT, row) for row in fills]
        ours += _even_spread_tail(tail, rng, key_fn)
    base = _assign_baseline(baseline_rows, [row for _, row in ours], key_fn)
    pairs = []
    for (segment, ours_row), base_row in zip(ours, base, strict=True):
        pair = [(BASELINE_ARM, base_row), (MULLIGAN_ARM, ours_row)]
        if rng.randrange(2):
            pair.reverse()
        pairs.append((segment, pair))
    return pairs


# ------------------------------------------------------------------------ manifest


def _ours_rows(design: StartDesign, keys: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    verbatim = []
    for rec in design.promotions.to_dict("records"):
        row = {key: float(rec[key]) for key in keys}
        row.update(
            is_verbatim=True,
            selection_rank=int(rec["selection_rank"]),
            upstream=bool(rec["upstream"]),
            group=str(rec["group"]),
            failed_arms=str(rec["failed_arms"]),
            stream_index=int(rec["stream_index"]),
            manifest_idx=int(rec["manifest_idx"]),
            stage=int(rec["stage"]),
            num_steps=int(rec["num_steps"]),
            outcome=str(rec["outcome"]),
            failure_mode=str(rec["failure_mode"]),
        )
        verbatim.append(row)
    fills = []
    for rec in design.fills.to_dict("records"):
        row = {key: float(rec[key]) for key in keys}
        row.update(
            is_verbatim=False,
            selection_rank=int(rec["selection_rank"]),
            fill_stream_index=int(rec["fill_stream_index"]),
        )
        fills.append(row)
    return verbatim, fills


def _schedule(cfg: RoundConfig, spec: RealTaskSpec, design: StartDesign) -> list[tuple[str, list]]:
    keys = TaskGeometry(spec).keys
    verbatim, fills = _ours_rows(design, keys)
    baseline = [
        {**{key: float(rec[key]) for key in keys}, "selection_rank": int(rec["selection_rank"])}
        for rec in design.baseline.to_dict("records")
    ]
    placement = next(iter(spec.sampled_placements.values()))
    order = cfg.order
    names = order.segment_names
    if order.strategy == "holder_pairs":
        if order.segments == "phase":
            segments = [
                (names["upstream"], [row for row in verbatim if row["upstream"]]),
                (names["later"], [row for row in verbatim if not row["upstream"]]),
            ]
        elif order.segments == "tier":
            # Promotions are ranked tier first, so first appearance is tier order.
            tiers = list(dict.fromkeys(row["group"] for row in verbatim))
            segments = [(tier, [row for row in verbatim if row["group"] == tier]) for tier in tiers]
        else:
            raise ValueError(f"unknown holder_pairs segments {order.segments!r}")
        segments.append((FILL_SEGMENT, fills))
        return order_holder_pairs(order, segments, baseline, placement.keys)
    if order.strategy == "peg_path":
        return order_peg_path(order, verbatim, fills, baseline, placement.keys)
    raise ValueError(f"unknown order strategy {order.strategy!r}")


def _state_row(
    cfg: RoundConfig, keys: tuple[str, ...], segment: str, source: str, row: dict, queue_rank: int
) -> dict:
    out: dict[str, Any] = {
        "source": source,
        "sources": [source],
        "source_index": row["selection_rank"] - 1,
        "collection_segment": segment,
        **{key: row[key] for key in keys},
    }
    if source == BASELINE_ARM:
        out.update(
            baseline_selection_rank=row["selection_rank"],
            candidate_source="fresh_uniform_baseline",
            sample_source="fresh_uniform_baseline",
            uniform_seed=cfg.baseline_seed,
        )
        return out
    out.update(
        mulligan_queue_rank=queue_rank,
        mulligan_selection_rank=row["selection_rank"],
        is_verbatim_failure=row["is_verbatim"],
    )
    if row["is_verbatim"]:
        out.update(
            candidate_source="promoted_eval_failure",
            sample_source="verbatim_eval_failure_graduation",
            source_eval_seed=cfg.source_eval_seed,
            source_eval_stream_index=row["stream_index"],
            source_eval_manifest_idx=row["manifest_idx"],
            target_outcome=row["outcome"],
            target_stage=row["stage"],
            target_num_steps=row["num_steps"],
            failure_mode=row["failure_mode"],
            failed_arms=row["failed_arms"],
        )
    else:
        out.update(
            candidate_source="fresh_sobol_fps_fill",
            sample_source="fresh_sobol_fps_fill",
            fill_sobol_seed=cfg.fill.seed,
            fill_stream_index=row["fill_stream_index"],
            fps_beta=cfg.fill.beta,
        )
    return out


def order_metrics(states: list[dict], keys: tuple[str, str]) -> dict[str, Any]:
    cells = [_placement_key(row, keys, 1) for row in states]
    runs, run = [], 1
    for a, b in zip(cells, cells[1:], strict=False):
        if a == b:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    return {"placement_changes": len(runs) - 1, "max_same_placement_run": max(runs)}


def build_manifest(cfg: RoundConfig, design: StartDesign) -> dict[str, Any]:
    """The two-arm blind collection manifest (``real_manual_initial_states_v1``)."""
    spec = get_task_spec(cfg.task)
    geom = TaskGeometry(spec)
    keys = geom.keys
    pairs = _schedule(cfg, spec, design)
    states = []
    for queue_rank, (segment, pair) in enumerate(pairs, start=1):
        for source, row in pair:
            states.append(_state_row(cfg, keys, segment, source, row, queue_rank))
    states = [{**row, "manifest_idx": i} for i, row in enumerate(states)]
    counts = Counter(row["source"] for row in states)
    if counts != {BASELINE_ARM: cfg.budget, MULLIGAN_ARM: cfg.budget}:
        raise ValueError(f"expected {cfg.budget} rows per arm, got {dict(counts)}")
    placement = next(iter(spec.sampled_placements.values()))
    unit = {True: "rad", False: "m"}
    display = {True: "deg", False: "in"}
    return {
        "task": cfg.task,
        "round": cfg.round,
        "schema": MANIFEST_SCHEMA,
        "keys": list(spec.manifest_keys),
        "units": {key: unit[key.endswith("_yaw")] for key in spec.manifest_keys},
        "operator_frame": {
            "description": (
                f"Manual {cfg.task} placement targets in the operator frame. +x is forward, "
                "+y is left, yaw is the object heading."
            ),
            "display_units": {key: display[key.endswith("_yaw")] for key in spec.manifest_keys},
        },
        "bounds": {key: list(b) for key, b in zip(spec.state_keys, spec.bounds, strict=True)},
        "sampled_placements": {
            name: {
                "keys": list(p.keys),
                "bounds": [list(axis) for axis in p.bounds],
                "snap_m": p.snap_m,
                "grid_dims": list(p.grid_dims),
            }
            for name, p in spec.sampled_placements.items()
        },
        "match_tolerance": EXACT_DUP_TOL,
        "arms": [BASELINE_ARM, MULLIGAN_ARM],
        "arm_counts": dict(sorted(counts.items())),
        "allow_duplicate_states": True,
        "duplicate_state_resolution": "manifest_idx",
        "protocol_arms": {"no_cf": [BASELINE_ARM, MULLIGAN_ARM], "with_cf": [MULLIGAN_ARM]},
        "protocol_targets": {"no_cf": cfg.budget, "with_cf": cfg.budget},
        "start_design": {
            "promoted": int(len(design.promotions)),
            "fills": int(len(design.fills)),
            "promotion_cap": cfg.promotion.cap,
            "source_eval_seed": cfg.source_eval_seed,
            "promoted_eval_stream_indices": sorted(
                design.promotions["stream_index"].astype(int).tolist()
            ),
            "fill_sobol_seed": cfg.fill.seed,
            "fill_stream_indices": sorted(design.fills["fill_stream_index"].astype(int).tolist()),
            "fps_beta": cfg.fill.beta,
            "fps_placement_axis_weight": cfg.fill.placement_weight,
            "fps_hardness": cfg.fill.hardness,
            "baseline_uniform_seed": cfg.baseline_seed,
        },
        "manifest_order": {
            "strategy": cfg.order.strategy,
            "seed": cfg.order.seed,
            "segments": dict(
                Counter(
                    row["collection_segment"] for row in states if row["source"] == MULLIGAN_ARM
                )
            ),
            **order_metrics(states, placement.keys),
        },
        "states": states,
    }


def build_round(cfg: RoundConfig, inputs: dict[str, Any]) -> tuple[StartDesign, dict[str, Any]]:
    design = build_start_design(cfg, inputs)
    return design, build_manifest(cfg, design)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/real/<task>/rNN_sampler.yaml")
    parser.add_argument(
        "--inputs-root", required=True, help="root the config's input paths are relative to"
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override one input; repeat it to give a list input (support_manifests, "
        "avoid_manifests) several paths",
    )
    parser.add_argument("--out", required=True, help="manifest JSON to write")
    args = parser.parse_args(argv)
    cfg = load_round_config(args.config)
    overrides = parse_input_overrides(args.input, cfg)
    _, manifest = build_round(cfg, resolve_inputs(cfg, args.inputs_root, overrides))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out}: {manifest['start_design']['promoted']} promoted + {manifest['start_design']['fills']} fills"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
