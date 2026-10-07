"""Hardness signals ``h(c)`` for the coverage fill of SelectInitialStates.

Each signal maps an ``(n, dim)`` array of physical starts to ``n`` scores; larger
means harder. The simulation rounds used three kinds (paper App. F.5):

- a geometric shape term on Square-Narrow, fitted to episode lengths over nut
  ``y`` and yaw (:func:`narrow_shape`);
- a ridge model of diagnostic episode length on Square-Broad
  (:func:`fit_length_model`);
- a per-cell term from the previous round's evaluation-grid success rates
  (:class:`GridCellTable`). This one uses evaluation results, so
  :class:`~mulligan.sampling.select_initial_states.HardnessTerm` marks it and the
  sampler refuses it unless the caller opts in.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from mulligan.sampling.select_initial_states import StartSpace

Signal = Callable[[np.ndarray], np.ndarray]

NARROW_SHAPE_TERMS = ("y_norm", "y_edge", "y_center2", "yaw_sin_alpha")
BROAD_LENGTH_FEATURES = (
    "nut_edge",
    "peg_edge",
    "yaw_pi2",
    "yaw0",
    "relative_yaw_perp",
    "close_015",
    "close_toward",
)
RIDGE_ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
N_SEXTILES = 6


def narrow_shape(space: StartSpace, weights: Mapping[str, float], yaw_alpha: float) -> Signal:
    """Square-Narrow shape term: a weighted sum over nut ``y`` and yaw features.

    Terms: ``y_norm`` (nut ``y`` scaled to [0, 1]), ``y_edge = |2 y_norm - 1|``,
    ``y_center2 = (y_norm - 0.5)^2`` and ``yaw_sin_alpha = |sin(yaw)|^alpha``.
    """
    unknown = set(weights) - set(NARROW_SHAPE_TERMS)
    if unknown:
        raise ValueError(f"unknown shape terms {sorted(unknown)}; expected {NARROW_SHAPE_TERMS}")
    names = [name for name in weights]
    w = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    iy = space.keys.index("nut_y")
    iyaw = space.keys.index("nut_yaw")
    y_lo, y_hi = space.low[iy], space.high[iy]

    def signal(states: np.ndarray) -> np.ndarray:
        y_norm = (states[:, iy] - y_lo) / (y_hi - y_lo)
        columns = {
            "y_norm": lambda: y_norm,
            "y_edge": lambda: np.abs(2.0 * y_norm - 1.0),
            "y_center2": lambda: (y_norm - 0.5) ** 2,
            "yaw_sin_alpha": lambda: (
                np.clip(np.abs(np.sin(states[:, iyaw])), 0.0, 1.0) ** float(yaw_alpha)
            ),
        }
        return np.column_stack([columns[name]() for name in names]) @ w

    return signal


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float64) + np.pi) % (2.0 * np.pi)) - np.pi


def _edge_score(xy: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    unit = (xy - low[None, :]) / (high - low)[None, :]
    dist_to_edge = np.minimum(unit, 1.0 - unit).min(axis=1)
    return 1.0 - np.clip(2.0 * dist_to_edge, 0.0, 1.0)


def broad_features(space: StartSpace, states: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Square-Broad geometric features of (nut x, y, yaw, peg x, y), one column per name.

    ``nut_edge``/``peg_edge``: closeness of the object to the edge of its range;
    ``yaw_pi2``/``yaw0``: nut yaw near +-90 deg / near 0; ``relative_yaw_perp``:
    ``|sin|`` of nut yaw relative to the nut-to-peg bearing; ``close_015``: Gaussian
    bump at 15 cm nut-peg distance; ``close_toward``: ``close_015`` times the
    alignment of the nut yaw with the peg bearing.
    """
    idx = {key: space.keys.index(key) for key in ("nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y")}
    low = np.asarray(space.low, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)
    nut = [idx["nut_x"], idx["nut_y"]]
    peg = [idx["peg_x"], idx["peg_y"]]
    dx = states[:, idx["peg_x"]] - states[:, idx["nut_x"]]
    dy = states[:, idx["peg_y"]] - states[:, idx["nut_y"]]
    dist = np.hypot(dx, dy)
    yaw = _wrap_pi(states[:, idx["nut_yaw"]])
    yaw_rel = _wrap_pi(yaw - np.arctan2(dy, dx))
    close = np.exp(-0.5 * ((dist - 0.15) / 0.035) ** 2)
    features = {
        "nut_edge": lambda: _edge_score(states[:, nut], low[nut], high[nut]),
        "peg_edge": lambda: _edge_score(states[:, peg], low[peg], high[peg]),
        "yaw_pi2": lambda: (
            0.5
            * (
                1.0
                + np.maximum(
                    np.cos(_wrap_pi(yaw - np.pi / 2.0)), np.cos(_wrap_pi(yaw + np.pi / 2.0))
                )
            )
        ),
        "yaw0": lambda: 0.5 * (1.0 + np.cos(yaw)),
        "relative_yaw_perp": lambda: np.abs(np.sin(yaw_rel)),
        "close_015": lambda: close,
        "close_toward": lambda: close * (0.5 * (1.0 + np.cos(yaw_rel))),
    }
    unknown = [name for name in names if name not in features]
    if unknown:
        raise ValueError(f"unknown Square-Broad features {unknown}")
    return np.column_stack([features[name]() for name in names])


def _fit_ridge(xz: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(xz)), xz])
    penalty = np.eye(x1.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(x1.T @ x1 + alpha * penalty, x1.T @ y)


@dataclass(frozen=True)
class LengthModel:
    """Ridge regression of episode length / max steps on geometric features."""

    space: StartSpace
    features: tuple[str, ...]
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def __call__(self, states: np.ndarray) -> np.ndarray:
        xz = (broad_features(self.space, states, self.features) - self.mean) / self.scale
        return np.clip(np.column_stack([np.ones(len(xz)), xz]) @ self.coef, 0.0, 1.0)


def fit_length_model(
    space: StartSpace,
    records: Sequence[Mapping],
    *,
    max_steps: float,
    features: Sequence[str] = BROAD_LENGTH_FEATURES,
    alphas: Sequence[float] = RIDGE_ALPHAS,
) -> LengthModel:
    """Fit :class:`LengthModel` with the ridge penalty chosen by leave-one-out CV.

    Features are standardized (ddof=1); the intercept is not penalized. The alpha
    with the lowest LOO mean squared error wins, higher LOO Spearman breaking ties.
    """
    x = broad_features(space, space.array(records), features)
    y = np.asarray([float(r["length"]) / max_steps for r in records], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=1)
    if np.any(scale <= 0):
        raise ValueError(f"constant feature columns: {np.flatnonzero(scale <= 0).tolist()}")
    xz = (x - mean) / scale

    rows = []
    for alpha in alphas:
        pred = np.empty(len(y), dtype=np.float64)
        for holdout in range(len(y)):
            keep = np.ones(len(y), dtype=bool)
            keep[holdout] = False
            coef = _fit_ridge(xz[keep], y[keep], alpha)
            pred[holdout] = np.clip(np.r_[1.0, xz[holdout]] @ coef, 0.0, 1.0)
        rows.append(
            (float(np.mean((pred - y) ** 2)), -float(spearmanr(pred, y).statistic), float(alpha))
        )
    alpha = min(rows)[2]
    return LengthModel(
        space=space,
        features=tuple(features),
        alpha=alpha,
        mean=mean,
        scale=scale,
        coef=_fit_ridge(xz, y, alpha),
    )


@dataclass(frozen=True)
class GridAxis:
    key: str
    low: float
    high: float
    n_bins: int


@dataclass(frozen=True)
class GridCellTable:
    """Per-cell success rates (percent) of one policy on an equal-width evaluation grid.

    ``success`` has one row per cell (row-major over ``axes``, last axis fastest)
    and one column per evaluation seed.
    """

    space: StartSpace
    axes: tuple[GridAxis, ...]
    success: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(np.prod([axis.n_bins for axis in self.axes]))

    @property
    def mean_success(self) -> np.ndarray:
        return self.success.mean(axis=1)

    def cell_index(self, states: np.ndarray) -> np.ndarray:
        flat = np.zeros(len(states), dtype=np.int64)
        for axis in self.axes:
            unit = (states[:, self.space.keys.index(axis.key)] - axis.low) / (axis.high - axis.low)
            if axis.key in self.space.periodic:
                unit = unit % 1.0
            elif np.any((unit < 0.0) | (unit > 1.0)):
                raise ValueError(f"{axis.key} outside the grid range [{axis.low}, {axis.high}]")
            idx = np.clip(np.floor(unit * axis.n_bins).astype(np.int64), 0, axis.n_bins - 1)
            flat = flat * axis.n_bins + idx
        return flat

    def sextile_bonus(self) -> Signal:
        """Rank cells by mean success into six equal groups; bonus 1 (hardest) to 0."""
        order = np.argsort(self.mean_success, kind="stable")
        bucket = np.empty(self.n_cells, dtype=np.int64)
        for b, positions in enumerate(np.array_split(np.arange(self.n_cells), N_SEXTILES)):
            bucket[order[positions]] = b
        bonus = (N_SEXTILES - 1 - bucket).astype(np.float64) / float(N_SEXTILES - 1)
        return lambda states: bonus[self.cell_index(states)]

    def failure_rate(self) -> Signal:
        """``1 - mean success / 100`` of the cell a start falls in."""
        failure = 1.0 - self.mean_success / 100.0
        return lambda states: failure[self.cell_index(states)]

    @classmethod
    def from_csv(cls, space: StartSpace, axes: Sequence[GridAxis], path: Path) -> GridCellTable:
        """Read a ``cell_idx,sr_seed<k>...`` table with one row per cell in order."""
        with Path(path).open(newline="") as f:
            reader = csv.DictReader(f)
            columns = [c for c in reader.fieldnames or [] if c.startswith("sr_seed")]
            rows = list(reader)
        if not columns:
            raise ValueError(f"{path}: no sr_seed<k> columns")
        cells = [int(r["cell_idx"]) for r in rows]
        if cells != list(range(len(rows))):
            raise ValueError(f"{path}: cell_idx must be 0..n-1 in order")
        table = cls(
            space=space,
            axes=tuple(axes),
            success=np.asarray([[float(r[c]) for c in columns] for r in rows], dtype=np.float64),
        )
        if len(rows) != table.n_cells:
            raise ValueError(f"{path}: {len(rows)} rows for a {table.n_cells}-cell grid")
        return table


def blend(parts: Sequence[tuple[float, Signal]]) -> Signal:
    """Weighted sum of signals."""

    def signal(states: np.ndarray) -> np.ndarray:
        total = np.zeros(len(states), dtype=np.float64)
        for weight, part in parts:
            total = total + weight * np.asarray(part(states), dtype=np.float64)
        return total

    return signal


def minmax(signal: Signal) -> Signal:
    """Rescale a signal to [0, 1] over the states it is evaluated on."""

    def scaled(states: np.ndarray) -> np.ndarray:
        score = np.asarray(signal(states), dtype=np.float64)
        return (score - score.min()) / max(float(score.max() - score.min()), 1e-9)

    return scaled
