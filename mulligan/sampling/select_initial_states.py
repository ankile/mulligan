"""SelectInitialStates: performance-guided choice of a round's collection starts.

One round of the method (paper Alg. 2, App. F) picks ``budget`` starts for the
supervised episodes:

1. **Failure promotion.** Every failed start of the round's diagnostic rollouts
   is replayed verbatim (in record order, up to an optional cap).
2. **Local perturbations** (optional). Jittered copies of the starts of the
   longest diagnostic episodes (Square-Narrow R1-R3: one around each of the 20
   longest).
3. **Coverage fill.** Greedy farthest-point selection from a candidate pool,
   maximizing ``d(c, P u G u X) * (1 + sum_k beta_k h_k(c))`` where ``d`` is the
   nearest-neighbor distance in unit-normalized coordinates with periodic yaw,
   ``P`` the previously collected starts, ``G`` the promoted and perturbed starts,
   ``X`` the fills chosen so far and ``h_k`` hardness signals
   (:mod:`mulligan.sampling.hardness`). With no hardness terms the fill is plain
   farthest-point coverage.

Candidates within ``exclude_tolerance`` of an excluded start (earlier rounds,
evaluation starts) are never selected. Hardness terms computed from the
evaluation grid are refused unless ``allow_eval_grid_feedback`` is set; the paper
used them only on Square-Narrow R3 and Square-Broad R2-R3 (App. F.5).

Round-specific inputs and the task bounds live in :mod:`mulligan.sampling.sim_design`
and ``configs/sim/<task>/rNN_sampler.yaml``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

PROMOTED = "promoted_failure"
PERTURBED = "local_perturbation"
FILL = "coverage_fill"


@dataclass(frozen=True)
class StartSpace:
    """Box of initial states; ``periodic`` keys wrap around their range."""

    keys: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    periodic: tuple[str, ...] = ()
    valid: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if not (len(self.keys) == len(self.low) == len(self.high)):
            raise ValueError("keys, low and high must have the same length")
        unknown = set(self.periodic) - set(self.keys)
        if unknown:
            raise ValueError(f"periodic keys not in space: {sorted(unknown)}")

    @property
    def dim(self) -> int:
        return len(self.keys)

    @property
    def periodic_mask(self) -> np.ndarray:
        return np.asarray([key in self.periodic for key in self.keys], dtype=bool)

    def array(self, states: Sequence[Mapping[str, float]]) -> np.ndarray:
        arr = np.asarray([[float(s[k]) for k in self.keys] for s in states], dtype=np.float64)
        return arr.reshape(len(states), self.dim)

    def states(self, arr: np.ndarray) -> list[dict[str, float]]:
        return [{k: float(v) for k, v in zip(self.keys, row, strict=True)} for row in arr]

    def wrap(self, arr: np.ndarray) -> np.ndarray:
        """Wrap periodic coordinates into ``[low, high)``."""
        out = np.array(arr, dtype=np.float64, copy=True)
        for i in np.flatnonzero(self.periodic_mask):
            lo, hi = self.low[i], self.high[i]
            out[:, i] = ((out[:, i] - lo) % (hi - lo)) + lo
        return out

    def to_unit(self, arr: np.ndarray) -> np.ndarray:
        """Scale every coordinate to [0, 1); periodic coordinates are taken modulo 1."""
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        unit = (np.asarray(arr, dtype=np.float64) - low) / (high - low)
        mask = self.periodic_mask
        unit[:, mask] = unit[:, mask] % 1.0
        return unit

    def is_valid(self, arr: np.ndarray) -> np.ndarray:
        if self.valid is None:
            return np.ones(len(arr), dtype=bool)
        return np.asarray(self.valid(arr), dtype=bool)


def sample_uniform(space: StartSpace, rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` i.i.d. uniform starts that satisfy ``space.valid``.

    Draws row-major blocks of at least 1024 rows and keeps the valid rows in order,
    so the result is a prefix of one fixed stream for a given ``rng``.
    """
    rows: list[np.ndarray] = []
    have = 0
    while have < n:
        block = rng.uniform(space.low, space.high, size=(max(1024, n - have), space.dim))
        block = space.wrap(block)
        block = block[space.is_valid(block)]
        rows.append(block)
        have += len(block)
    return np.concatenate(rows)[:n] if rows else np.empty((0, space.dim))


def squared_unit_distance(space: StartSpace, points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Squared distance from every unit-coordinate row of ``points`` to one row ``ref``."""
    delta = np.abs(points - ref)
    mask = space.periodic_mask
    delta[:, mask] = np.minimum(delta[:, mask], 1.0 - delta[:, mask])
    return np.sum(delta * delta, axis=1)


def nearest_unit_distance(
    space: StartSpace, points: np.ndarray, reference: np.ndarray, *, periodic: bool = True
) -> np.ndarray:
    """Distance from each state in ``points`` to its nearest state in ``reference``.

    Both inputs are physical coordinates; distances are Euclidean in unit coordinates.
    ``periodic=False`` ignores the yaw wrap-around (a plain KD-tree query), which is
    how the Square-Broad builders measured the distance to the prior starts.
    """
    pts = space.to_unit(points)
    ref = space.to_unit(reference)
    if len(ref) == 0:
        return np.full(len(pts), np.inf)
    if not periodic:
        dist, _ = cKDTree(ref).query(pts, k=1)
        return np.asarray(dist, dtype=np.float64)
    best = np.full(len(pts), np.inf)
    for row in ref:
        best = np.minimum(best, squared_unit_distance(space, pts, row[None, :]))
    return np.sqrt(best)


def _with_periodic_images(arr: np.ndarray, mask: np.ndarray, period: np.ndarray) -> np.ndarray:
    """``arr`` stacked with its copies shifted by -period and +period along each masked axis."""
    images = [arr]
    for i in np.flatnonzero(mask):
        shifted = []
        for image in images:
            for sign in (-1.0, 1.0):
                moved = image.copy()
                moved[:, i] += sign * period[i]
                shifted.append(moved)
        images.extend(shifted)
    return np.concatenate(images)


def nearest_state_distance(
    space: StartSpace, points: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor distance in physical units (periodic keys wrap), and its index.

    This is the distance of the disjointness checks: ``||a - b||_2`` over the raw
    coordinates with the periodic difference ``min(|d|, period - |d|)``.
    """
    if len(reference) == 0:
        return np.full(len(points), np.inf), np.full(len(points), -1, dtype=np.int64)
    period = np.asarray(space.high, dtype=np.float64) - np.asarray(space.low, dtype=np.float64)
    tiled = _with_periodic_images(space.wrap(reference), space.periodic_mask, period)
    dist, idx = cKDTree(tiled).query(space.wrap(points), k=1)
    return np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=np.int64) % len(reference)


@dataclass(frozen=True)
class HardnessTerm:
    """One ``beta * h(c)`` term of the fill objective.

    ``signal`` maps an ``(n, dim)`` array of physical states to ``n`` scores.
    ``uses_eval_grid`` marks signals computed from evaluation-grid results.
    """

    beta: float
    signal: Callable[[np.ndarray], np.ndarray]
    uses_eval_grid: bool = False
    name: str = ""


@dataclass(frozen=True)
class LocalPerturbation:
    """Jitter the starts of the longest diagnostic episodes.

    The ``n`` perturbations are one each around the ``n`` longest episodes
    (stable order on ties). Per anchor and per key
    in space order, a key in ``resample`` is redrawn uniformly over its range and
    any other key gets Gaussian noise with ``sigma[key]``, clipped to the range
    (wrapped for periodic keys).
    """

    n: int
    sigma: Mapping[str, float]
    seed: int
    resample: tuple[str, ...] = ()

    def anchors(self, records: Sequence[Mapping]) -> list[Mapping]:
        ranked = sorted(records, key=lambda r: float(r["length"]), reverse=True)
        return ranked[: self.n]

    def draw(self, space: StartSpace, records: Sequence[Mapping]) -> np.ndarray:
        missing = set(space.keys) - set(self.sigma) - set(self.resample)
        if missing:
            raise ValueError(f"perturbation has no sigma or resample rule for {sorted(missing)}")
        anchors = self.anchors(records)
        if len(anchors) < self.n:
            raise ValueError(f"need {self.n} perturbation anchors, have {len(anchors)} records")
        rng = np.random.default_rng(self.seed)
        out = np.empty((len(anchors), space.dim), dtype=np.float64)
        for row, anchor in enumerate(anchors):
            for i, key in enumerate(space.keys):
                lo, hi = space.low[i], space.high[i]
                if key in self.resample:
                    out[row, i] = float(rng.uniform(lo, hi))
                    continue
                value = float(anchor[key]) + rng.normal(0.0, self.sigma[key])
                if key in space.periodic:
                    out[row, i] = ((value - lo) % (hi - lo)) + lo
                else:
                    out[row, i] = float(np.clip(value, lo, hi))
        return out


@dataclass
class Design:
    """Selected starts in order (promoted, perturbed, fill) with a component label each."""

    states: np.ndarray
    components: list[str]


def promote_failures(
    space: StartSpace, records: Sequence[Mapping], *, cap: int | None = None
) -> np.ndarray:
    """Starts of the failed diagnostic episodes, verbatim, in record order."""
    failures = [r for r in records if int(r["success"]) == 0]
    if cap is not None:
        failures = failures[:cap]
    return space.array(failures)


def hardness_factor(terms: Sequence[HardnessTerm], candidates: np.ndarray) -> np.ndarray:
    """``1 + sum_k beta_k h_k(c)`` for every candidate, accumulated term by term."""
    factor = np.ones(len(candidates), dtype=np.float64)
    for term in terms:
        scores = np.asarray(term.signal(candidates), dtype=np.float64)
        if scores.shape != (len(candidates),):
            raise ValueError(f"hardness term {term.name!r} returned shape {scores.shape}")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"hardness term {term.name!r} returned non-finite scores")
        factor = factor + term.beta * scores
    return factor


def farthest_point_fill(
    space: StartSpace,
    candidates: np.ndarray,
    seeds: np.ndarray,
    n: int,
    *,
    factor: np.ndarray | None = None,
    squared: bool = False,
    seed_distance_periodic: bool = True,
    blocked: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy weighted farthest-point selection; returns chosen indices and objectives.

    Each step picks ``argmax_c d(c, seeds u chosen) * factor(c)`` over the unblocked
    candidates (first index on ties). ``squared`` uses ``d**2`` in place of ``d``.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    if factor is None:
        factor = np.ones(len(candidates), dtype=np.float64)
    available = np.ones(len(candidates), dtype=bool)
    if blocked is not None:
        available &= ~np.asarray(blocked, dtype=bool)
    if int(available.sum()) < n:
        raise ValueError(f"need {n} fill states, only {int(available.sum())} candidates available")

    cand_unit = space.to_unit(candidates)
    if seed_distance_periodic:
        seed_unit = space.to_unit(seeds)
        min_d = np.full(len(candidates), np.inf)
        for row in seed_unit:
            min_d = np.minimum(min_d, squared_unit_distance(space, cand_unit, row[None, :]))
        if not squared:
            min_d = np.sqrt(min_d)
    else:
        min_d = nearest_unit_distance(space, candidates, seeds, periodic=False)
        if squared:
            min_d = min_d * min_d

    chosen: list[int] = []
    objectives: list[float] = []
    for _ in range(n):
        objective = np.where(available, min_d * factor, -np.inf)
        idx = int(np.argmax(objective))
        chosen.append(idx)
        objectives.append(float(objective[idx]))
        available[idx] = False
        step = squared_unit_distance(space, cand_unit, cand_unit[idx : idx + 1])
        min_d = np.minimum(min_d, step if squared else np.sqrt(step))
    return np.asarray(chosen, dtype=np.int64), np.asarray(objectives)


def select_initial_states(
    *,
    space: StartSpace,
    budget: int,
    records: Sequence[Mapping],
    prior: np.ndarray,
    candidates: np.ndarray,
    hardness: Sequence[HardnessTerm] = (),
    perturbation: LocalPerturbation | None = None,
    promotion_cap: int | None = None,
    squared_distance: bool = False,
    prior_distance_periodic: bool = True,
    exclude: np.ndarray | None = None,
    exclude_tolerance: float = 1e-6,
    allow_eval_grid_feedback: bool = False,
) -> Design:
    """Choose ``budget`` starts: promoted failures, perturbations, then coverage fill.

    ``records`` are the round's diagnostic rollouts (state keys, ``success``,
    ``length``); ``prior`` the starts already collected for this arm's lineage;
    ``candidates`` the fill pool in physical coordinates; ``exclude`` the states
    no fill may come within ``exclude_tolerance`` of (earlier rounds, evaluation
    starts). The defaults give the paper's objective ``d(c, .) (1 + beta h(c))``;
    ``squared_distance`` and ``prior_distance_periodic=False`` reproduce the
    Square-Narrow and Square-Broad rounds as collected (see the round configs).
    """
    eval_terms = [t.name or "unnamed" for t in hardness if t.uses_eval_grid]
    if eval_terms and not allow_eval_grid_feedback:
        raise ValueError(
            f"hardness terms {eval_terms} use evaluation-grid results; pass "
            "allow_eval_grid_feedback=True to use them (paper App. F.5)"
        )

    promoted = promote_failures(space, records, cap=promotion_cap)
    perturbed = (
        perturbation.draw(space, records)
        if perturbation is not None and perturbation.n > 0
        else np.empty((0, space.dim))
    )
    n_fill = budget - len(promoted) - len(perturbed)
    if n_fill < 0:
        raise ValueError(
            f"{len(promoted)} promoted + {len(perturbed)} perturbed starts exceed budget {budget}"
        )

    blocked = None
    if exclude is not None and len(exclude):
        dist, _ = nearest_state_distance(space, candidates, exclude)
        blocked = dist <= exclude_tolerance

    seeds = np.concatenate(
        [np.asarray(prior, dtype=np.float64).reshape(-1, space.dim), promoted, perturbed]
    )
    idx, _ = farthest_point_fill(
        space,
        candidates,
        seeds,
        n_fill,
        factor=hardness_factor(hardness, candidates) if hardness else None,
        squared=squared_distance,
        seed_distance_periodic=prior_distance_periodic,
        blocked=blocked,
    )
    states = np.concatenate([promoted, perturbed, candidates[idx]])
    components = [PROMOTED] * len(promoted) + [PERTURBED] * len(perturbed) + [FILL] * len(idx)
    return Design(states=states, components=components)


def check_disjoint(
    space: StartSpace,
    states: np.ndarray,
    others: Mapping[str, np.ndarray],
    *,
    tolerance: float = 1e-6,
) -> list[dict]:
    """Raise if any state lies within ``tolerance`` of a state in ``others``.

    Returns one report row per reference set with its minimum distance.
    """
    reports = []
    for label, other in others.items():
        dist, nearest = nearest_state_distance(space, states, other)
        close = np.flatnonzero(dist <= tolerance)
        if len(close):
            first = int(close[0])
            raise ValueError(
                f"state {first} collides with {label} state {int(nearest[first])} "
                f"(distance {float(dist[first]):.3g} <= {tolerance:g})"
            )
        reports.append(
            {"reference": label, "n_states": len(other), "min_distance": float(dist.min())}
        )
    return reports


def min_pairwise_distance(space: StartSpace, states: np.ndarray) -> float:
    """Smallest physical distance (periodic keys wrap) between two distinct states."""
    if len(states) < 2:
        return float("inf")
    period = np.asarray(space.high, dtype=np.float64) - np.asarray(space.low, dtype=np.float64)
    wrapped = space.wrap(states)
    tiled = _with_periodic_images(wrapped, space.periodic_mask, period)
    dist, _ = cKDTree(tiled).query(wrapped, k=2)
    return float(dist[:, 1].min())


def coverage_p95(space: StartSpace, states: np.ndarray, test_points: np.ndarray) -> float:
    """95th percentile over ``test_points`` of the unit-coordinate distance to ``states``."""
    tiled = _with_periodic_images(
        space.to_unit(states), space.periodic_mask, np.ones(space.dim, dtype=np.float64)
    )
    dist, _ = cKDTree(tiled).query(space.to_unit(test_points), k=1)
    return float(np.quantile(dist, 0.95))


def coverage_guardrail(
    space: StartSpace,
    *,
    design: np.ndarray,
    reference: np.ndarray,
    prior: np.ndarray,
    test_points: np.ndarray,
    slack: float = 1.0,
) -> dict:
    """Compare the cumulative p95 coverage radius of ``prior + design`` with ``prior + reference``.

    ``reference`` is a fresh Sobol block of the same size. The guardrail passes when
    the design's radius is at most ``slack`` times the reference radius.
    """
    if len(reference) != len(design):
        raise ValueError(f"reference block has {len(reference)} states, design has {len(design)}")
    design_p95 = coverage_p95(space, np.concatenate([prior, design]), test_points)
    reference_p95 = coverage_p95(space, np.concatenate([prior, reference]), test_points)
    ceiling = slack * reference_p95
    return {
        "design_p95": design_p95,
        "reference_p95": reference_p95,
        "slack": slack,
        "ceiling": ceiling,
        "passed": bool(design_p95 <= ceiling),
    }
