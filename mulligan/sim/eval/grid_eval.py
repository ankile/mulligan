#!/usr/bin/env python3
r"""Evaluate policies on fixed manifests of Square initial states (the paper's grid eval).

Each manifest point is one start (nut pose, plus the peg position on Square-Broad);
the policy runs one episode from each. The locked grids are

- Square-Narrow (``NutAssemblySquare``): 8,000 valid Sobol points,
  ``make-valid-sobol-manifest --task square_narrow --num-points 8000 --seed 2026052402``
  (``manifest_hash`` ``45a9963f...``). Round 0 used the equal-tile grid from
  ``make-equal-tile-manifest`` (80 cells x 100 points, seed 20260524).
- Square-Broad (``Square_D1``): 30,000 valid Sobol points,
  ``make-valid-sobol-manifest --task square_broad --num-points 30000 --seed 2026052499
  --candidate-power 16`` (``manifest_hash`` ``e0b50566...``).

Usage (``scripts/sim/eval_cell.sh`` generates the locked grid at the same path on first use):
    python -m mulligan.sim.eval.grid_eval make-valid-sobol-manifest \
        --task square_narrow --num-points 8000 --seed 2026052402 \
        --output outputs/sim/grids/square_narrow_sobol8k.json

    python -m mulligan.sim.eval.grid_eval eval \
        --artifact-path hf://mulligan/sim-square-narrow-r01-mulligan-divl@<revision>/seed-1 \
        --point-manifest outputs/sim/grids/square_narrow_sobol8k.json \
        --output-dir results/narrow-r01-divl/seed1 \
        --num-action-samples 32

    # Shard k of n evaluates the contiguous point slice
    # [num_points * (k - 1) // n, num_points * k // n); merge the shards with
    python -m mulligan.sim.eval.grid_eval merge results/narrow-r01-divl/seed1
    # or every unmerged directory below a root, then check the merged files:
    python -m mulligan.sim.eval.grid_eval merge --auto-find results/
    python -m mulligan.sim.eval.grid_eval merge --verify results/

``--artifact-path`` is a local checkpoint directory or
``hf://<org>/<repo>@<revision>/<subdir>`` (see :func:`mulligan.release.hub.resolve_checkpoint`).
Environment resets are unseeded unless ``--env-seed`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from mulligan.release.hub import resolve_checkpoint
from mulligan.sim.placement import nut_qpos
from mulligan.utils.progress import emit_completion, emit_progress

SCHEMA_V1 = "mulligan.eval.initial_state_points.v1"
SCHEMA_V2 = "mulligan.eval.initial_state_points.v2"
RESULTS_SCHEMA = "mulligan.eval.initial_state_point_results.v2"
EVAL_RESUME_SCHEMA = "mulligan.eval.initial_state_point_resume.v1"
EVAL_RESUME_BATCH_INTERVAL = 10
EVAL_SEED_STRATEGY = "sha256_eval_seed_and_ordered_batch_point_indices_v1"
ENV_SEED_STRATEGY = "sha256_env_seed_and_point_index_v1"

NARROW_X_RANGE = (-0.115, -0.110)
NARROW_Y_RANGE = (0.110, 0.225)
NARROW_YAW_RANGE = (-math.pi, math.pi)
NARROW_N_X_BINS = 2
NARROW_N_Y_BINS = 5
NARROW_N_YAW_BINS = 8

BROAD_NUT_X_RANGE = (-0.115, 0.115)
BROAD_NUT_Y_RANGE = (-0.255, 0.255)
BROAD_NUT_YAW_RANGE = (-math.pi, math.pi)
BROAD_PEG_X_RANGE = (-0.1, 0.3)
BROAD_PEG_Y_RANGE = (-0.2, 0.2)
BROAD_MIN_CLEARANCE = 0.13263


def _configure_eval_action_samples(policy: Any, num_action_samples: int | None) -> None:
    """Set the IDQL best-of-N candidate count used during evaluation."""
    from mulligan.agents.idql import IDQLPolicy

    if isinstance(policy, IDQLPolicy) and num_action_samples is not None:
        original_samples = policy.config.num_action_samples
        policy.config.num_action_samples = num_action_samples
        print(f"Overriding num_action_samples: {original_samples} -> {num_action_samples}")


@dataclass(frozen=True)
class AxisSpec:
    name: str
    value_key: str
    range: tuple[float, float]
    n_bins: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value_key": self.value_key,
            "range": list(self.range),
            "n_bins": self.n_bins,
        }


@dataclass(frozen=True)
class TaskSpec:
    key: str
    task: str
    task_variant: str
    env_name: str
    protocol: str
    sampler: str
    axes: tuple[AxisSpec, ...]
    default_num_points: int
    default_seed: int
    min_clearance: float | None = None
    candidate_power: int | None = None

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def n_cells(self) -> int:
        total = 1
        for axis in self.axes:
            total *= axis.n_bins
        return total

    def to_grid_dict(self) -> dict[str, Any]:
        return {
            "name": f"{self.key}_posthoc_equal_width_tiles",
            "axes": [axis.to_dict() for axis in self.axes],
            "n_cells": self.n_cells,
        }


TASK_SPECS: dict[str, TaskSpec] = {
    "square_narrow": TaskSpec(
        key="square_narrow",
        task="NutAssemblySquare",
        task_variant="Square_D0",
        env_name="NutAssemblySquare",
        protocol="square_narrow_global_valid_sobol8k_points",
        sampler="sobol_scrambled_global_valid_prefix",
        axes=(
            AxisSpec("nut_x", "nut_x", NARROW_X_RANGE, NARROW_N_X_BINS),
            AxisSpec("nut_y", "nut_y", NARROW_Y_RANGE, NARROW_N_Y_BINS),
            AxisSpec("nut_yaw", "nut_yaw", NARROW_YAW_RANGE, NARROW_N_YAW_BINS),
        ),
        default_num_points=8000,
        default_seed=2026052402,
    ),
    "square_broad": TaskSpec(
        key="square_broad",
        task="Square_D1",
        task_variant="Square_D1",
        env_name="Square_D1",
        protocol="square_broad_global_valid_sobol30k_points",
        sampler="sobol_scrambled_global_valid_prefix",
        axes=(
            AxisSpec("nut_x", "nut_x", BROAD_NUT_X_RANGE, 3),
            AxisSpec("nut_y", "nut_y", BROAD_NUT_Y_RANGE, 5),
            AxisSpec("nut_yaw", "nut_yaw", BROAD_NUT_YAW_RANGE, 8),
            AxisSpec("peg_x", "peg_x", BROAD_PEG_X_RANGE, 3),
            AxisSpec("peg_y", "peg_y", BROAD_PEG_Y_RANGE, 2),
        ),
        default_num_points=30000,
        default_seed=2026052499,
        min_clearance=BROAD_MIN_CLEARANCE,
        candidate_power=16,
    ),
}
TASK_ALIASES = {
    "square_narrow": "square_narrow",
    "NutAssemblySquare": "square_narrow",
    "square_broad": "square_broad",
    "Square_D1": "square_broad",
}


@dataclass(frozen=True)
class EqualTileGrid:
    x_range: tuple[float, float] = NARROW_X_RANGE
    y_range: tuple[float, float] = NARROW_Y_RANGE
    yaw_range: tuple[float, float] = NARROW_YAW_RANGE
    n_x_bins: int = NARROW_N_X_BINS
    n_y_bins: int = NARROW_N_Y_BINS
    n_yaw_bins: int = NARROW_N_YAW_BINS

    @property
    def n_cells(self) -> int:
        return self.n_x_bins * self.n_y_bins * self.n_yaw_bins

    def cell_indices(self, cell_idx: int) -> tuple[int, int, int]:
        if cell_idx < 0 or cell_idx >= self.n_cells:
            raise ValueError(f"cell_idx out of range [0,{self.n_cells}): {cell_idx}")
        yaw_idx = cell_idx % self.n_yaw_bins
        y_idx = (cell_idx // self.n_yaw_bins) % self.n_y_bins
        x_idx = cell_idx // (self.n_yaw_bins * self.n_y_bins)
        return x_idx, y_idx, yaw_idx

    def cell_edges(self, cell_idx: int) -> dict[str, float]:
        x_idx, y_idx, yaw_idx = self.cell_indices(cell_idx)
        x_edges = np.linspace(self.x_range[0], self.x_range[1], self.n_x_bins + 1)
        y_edges = np.linspace(self.y_range[0], self.y_range[1], self.n_y_bins + 1)
        yaw_edges = np.linspace(self.yaw_range[0], self.yaw_range[1], self.n_yaw_bins + 1)
        return {
            "x_lo": float(x_edges[x_idx]),
            "x_hi": float(x_edges[x_idx + 1]),
            "y_lo": float(y_edges[y_idx]),
            "y_hi": float(y_edges[y_idx + 1]),
            "yaw_lo": float(yaw_edges[yaw_idx]),
            "yaw_hi": float(yaw_edges[yaw_idx + 1]),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "square_narrow_equal_width_tiles",
            "x_range": list(self.x_range),
            "y_range": list(self.y_range),
            "yaw_range": list(self.yaw_range),
            "n_x_bins": self.n_x_bins,
            "n_y_bins": self.n_y_bins,
            "n_yaw_bins": self.n_yaw_bins,
            "n_cells": self.n_cells,
        }


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sobol_points(n: int, *, seed: int, dimension: int) -> np.ndarray:
    from scipy.stats.qmc import Sobol

    sampler = Sobol(d=dimension, scramble=True, seed=seed)
    m = int(math.ceil(math.log2(n)))
    return sampler.random_base2(m)[:n]


def _task_spec(task: str) -> TaskSpec:
    try:
        return TASK_SPECS[TASK_ALIASES[task]]
    except KeyError as exc:
        raise ValueError(
            f"unsupported task {task!r}; expected one of {sorted(TASK_ALIASES)}"
        ) from exc


def _edges(axis: AxisSpec) -> np.ndarray:
    return np.linspace(axis.range[0], axis.range[1], axis.n_bins + 1)


def _cell_indices_from_point(spec: TaskSpec, point: dict[str, Any]) -> tuple[int, ...]:
    indices = []
    for axis in spec.axes:
        value = float(point[axis.value_key])
        lo, hi = axis.range
        if not (lo <= value < hi):
            raise ValueError(f"{axis.value_key} out of range [{lo},{hi}): {value}")
        raw = (value - lo) / (hi - lo) * axis.n_bins
        idx = min(axis.n_bins - 1, max(0, int(math.floor(raw))))
        indices.append(idx)
    return tuple(indices)


def _flat_cell_idx(spec: TaskSpec, indices: tuple[int, ...]) -> int:
    if len(indices) != len(spec.axes):
        raise ValueError(f"expected {len(spec.axes)} cell indices, got {len(indices)}")
    cell_idx = 0
    multiplier = 1
    for idx, axis in reversed(list(zip(indices, spec.axes, strict=True))):
        if idx < 0 or idx >= axis.n_bins:
            raise ValueError(f"axis {axis.name} index out of range: {idx}")
        cell_idx += idx * multiplier
        multiplier *= axis.n_bins
    return cell_idx


def _cell_indices_from_flat(spec: TaskSpec, cell_idx: int) -> tuple[int, ...]:
    if cell_idx < 0 or cell_idx >= spec.n_cells:
        raise ValueError(f"cell_idx out of range [0,{spec.n_cells}): {cell_idx}")
    remaining = cell_idx
    indices_reversed = []
    for axis in reversed(spec.axes):
        indices_reversed.append(remaining % axis.n_bins)
        remaining //= axis.n_bins
    return tuple(reversed(indices_reversed))


def _cell_edges_from_indices(spec: TaskSpec, indices: tuple[int, ...]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for idx, axis in zip(indices, spec.axes, strict=True):
        axis_edges = _edges(axis)
        payload[f"{axis.name}_lo"] = float(axis_edges[idx])
        payload[f"{axis.name}_hi"] = float(axis_edges[idx + 1])
    return payload


def _is_valid_point(spec: TaskSpec, row: dict[str, float]) -> bool:
    if spec.min_clearance is None:
        return True
    dist = math.hypot(row["nut_x"] - row["peg_x"], row["nut_y"] - row["peg_y"])
    return dist >= spec.min_clearance


def make_equal_tile_manifest(
    *,
    points_per_cell: int = 100,
    seed: int = 20260524,
) -> dict[str, Any]:
    if points_per_cell <= 0:
        raise ValueError(f"points_per_cell must be positive; got {points_per_cell}")

    grid = EqualTileGrid()
    points: list[dict[str, Any]] = []
    point_idx = 0
    for cell_idx in range(grid.n_cells):
        x_idx, y_idx, yaw_idx = grid.cell_indices(cell_idx)
        edges = grid.cell_edges(cell_idx)
        raw = _sobol_points(points_per_cell, seed=seed + 1_000_003 * cell_idx, dimension=3)
        for within_cell_idx, unit in enumerate(raw):
            nut_x = edges["x_lo"] + float(unit[0]) * (edges["x_hi"] - edges["x_lo"])
            nut_y = edges["y_lo"] + float(unit[1]) * (edges["y_hi"] - edges["y_lo"])
            nut_yaw = edges["yaw_lo"] + float(unit[2]) * (edges["yaw_hi"] - edges["yaw_lo"])
            if nut_yaw >= math.pi:
                nut_yaw = math.nextafter(math.pi, -math.inf)
            points.append(
                {
                    "point_idx": point_idx,
                    "cell_idx": cell_idx,
                    "within_cell_idx": within_cell_idx,
                    "x_idx": x_idx,
                    "y_idx": y_idx,
                    "yaw_idx": yaw_idx,
                    "nut_x": nut_x,
                    "nut_y": nut_y,
                    "nut_yaw": nut_yaw,
                    "cell_edges": edges,
                    "sampler": "sobol_scrambled_per_cell",
                    "seed": seed + 1_000_003 * cell_idx,
                }
            )
            point_idx += 1

    # Same keys as the locked round-0 grid (manifest_hash 8f57d528...): v1 manifests
    # carry no task_variant/env_name; the evaluator falls back to "task".
    manifest = {
        "schema": SCHEMA_V1,
        "task": "NutAssemblySquare",
        "protocol": "square_narrow_equal_tile_sobol_points",
        "sampler": "sobol_scrambled_per_cell",
        "base_seed": seed,
        "points_per_cell": points_per_cell,
        "grid": grid.to_dict(),
        "num_points": len(points),
        "points": points,
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def make_valid_sobol_manifest(
    *,
    task: str,
    num_points: int | None = None,
    seed: int | None = None,
    candidate_power: int | None = None,
    require_all_cells: bool = True,
) -> dict[str, Any]:
    spec = _task_spec(task)
    num_points = spec.default_num_points if num_points is None else num_points
    seed = spec.default_seed if seed is None else seed
    if num_points <= 0:
        raise ValueError(f"num_points must be positive; got {num_points}")

    power = candidate_power or spec.candidate_power or int(math.ceil(math.log2(num_points)))
    accepted: list[dict[str, Any]] = []
    rejected = 0
    while len(accepted) < num_points:
        if power > 26:
            raise RuntimeError(
                f"failed to get {num_points} valid {spec.key} Sobol points by 2^{power - 1} "
                "candidates; check bounds or validity predicate"
            )
        raw = _sobol_points(2**power, seed=seed, dimension=spec.dimension)
        accepted.clear()
        rejected = 0
        for candidate_idx, unit in enumerate(raw):
            row: dict[str, float] = {}
            for dim, axis in enumerate(spec.axes):
                lo, hi = axis.range
                value = lo + float(unit[dim]) * (hi - lo)
                if value >= hi:
                    value = math.nextafter(hi, -math.inf)
                row[axis.value_key] = value
            if not _is_valid_point(spec, row):
                rejected += 1
                continue
            indices = _cell_indices_from_point(spec, row)
            cell_idx = _flat_cell_idx(spec, indices)
            point = {
                "point_idx": len(accepted),
                "candidate_idx": int(candidate_idx),
                "cell_idx": cell_idx,
                "cell_edges": _cell_edges_from_indices(spec, indices),
                "sampler": spec.sampler,
                "seed": seed,
                **row,
            }
            for axis, idx in zip(spec.axes, indices, strict=True):
                point[f"{axis.name}_idx"] = idx
            if spec.min_clearance is not None:
                point["nut_peg_distance"] = float(
                    math.hypot(row["nut_x"] - row["peg_x"], row["nut_y"] - row["peg_y"])
                )
            accepted.append(point)
            if len(accepted) == num_points:
                break
        if len(accepted) < num_points:
            power += 1

    counts = np.bincount([int(p["cell_idx"]) for p in accepted], minlength=spec.n_cells)
    if require_all_cells and np.any(counts == 0):
        empty = np.flatnonzero(counts == 0)
        raise RuntimeError(
            f"{spec.key} Sobol manifest left {len(empty)} / {spec.n_cells} cells empty; "
            f"first empty cells: {empty[:20].tolist()}"
        )

    manifest = {
        "schema": SCHEMA_V2,
        "task": spec.task,
        "task_variant": spec.task_variant,
        "env_name": spec.env_name,
        "task_key": spec.key,
        "protocol": spec.protocol,
        "sampler": spec.sampler,
        "base_seed": seed,
        "candidate_power": power,
        "num_candidates": 2**power,
        "num_rejected_before_prefix_complete": rejected,
        "min_clearance": spec.min_clearance,
        "grid": spec.to_grid_dict(),
        "num_points": len(accepted),
        "cell_count_min": int(counts.min()),
        "cell_count_max": int(counts.max()),
        "occupied_cells": int(np.count_nonzero(counts)),
        "require_all_cells": require_all_cells,
        "points": accepted,
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        manifest = json.load(f)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    schema = manifest.get("schema")
    if schema == SCHEMA_V1:
        validate_equal_tile_manifest(manifest)
        return
    if schema == SCHEMA_V2:
        validate_valid_sobol_manifest(manifest)
        return
    raise ValueError(f"unsupported manifest schema: {schema!r}")


def validate_equal_tile_manifest(manifest: dict[str, Any]) -> None:
    expected_hash = manifest.get("manifest_hash")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("manifest missing non-empty manifest_hash")
    actual_hash = manifest_hash(manifest)
    if actual_hash != expected_hash:
        raise ValueError(f"manifest hash mismatch: expected {expected_hash}, got {actual_hash}")
    if manifest.get("schema") != SCHEMA_V1:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')!r}")
    if manifest.get("task") != "NutAssemblySquare":
        raise ValueError(f"unsupported manifest task: {manifest.get('task')!r}")

    grid_payload = manifest.get("grid")
    if not isinstance(grid_payload, dict):
        raise ValueError("manifest grid must be an object")
    grid = EqualTileGrid(
        x_range=tuple(grid_payload["x_range"]),
        y_range=tuple(grid_payload["y_range"]),
        yaw_range=tuple(grid_payload["yaw_range"]),
        n_x_bins=int(grid_payload["n_x_bins"]),
        n_y_bins=int(grid_payload["n_y_bins"]),
        n_yaw_bins=int(grid_payload["n_yaw_bins"]),
    )
    if grid != EqualTileGrid():
        raise ValueError(f"manifest grid is not the locked Square-Narrow equal-tile grid: {grid}")

    points = manifest.get("points")
    if not isinstance(points, list):
        raise ValueError("manifest points must be a list")
    if int(manifest.get("num_points", -1)) != len(points):
        raise ValueError(f"num_points mismatch: {manifest.get('num_points')} != {len(points)}")
    points_per_cell = int(manifest.get("points_per_cell", -1))
    if points_per_cell <= 0:
        raise ValueError(f"invalid points_per_cell: {points_per_cell}")
    expected_total = grid.n_cells * points_per_cell
    if len(points) != expected_total:
        raise ValueError(f"expected {expected_total} points, found {len(points)}")

    counts = np.zeros(grid.n_cells, dtype=np.int64)
    seen_point_idx: set[int] = set()
    for expected_point_idx, point in enumerate(points):
        point_idx = int(point["point_idx"])
        if point_idx != expected_point_idx:
            raise ValueError(f"point_idx at row {expected_point_idx} is {point_idx}")
        if point_idx in seen_point_idx:
            raise ValueError(f"duplicate point_idx: {point_idx}")
        seen_point_idx.add(point_idx)

        cell_idx = int(point["cell_idx"])
        x_idx, y_idx, yaw_idx = grid.cell_indices(cell_idx)
        if (int(point["x_idx"]), int(point["y_idx"]), int(point["yaw_idx"])) != (
            x_idx,
            y_idx,
            yaw_idx,
        ):
            raise ValueError(f"point {point_idx} has inconsistent cell indices")
        edges = grid.cell_edges(cell_idx)
        x = float(point["nut_x"])
        y = float(point["nut_y"])
        yaw = float(point["nut_yaw"])
        if not (edges["x_lo"] <= x < edges["x_hi"]):
            raise ValueError(f"point {point_idx} nut_x out of tile bounds: {x}")
        if not (edges["y_lo"] <= y < edges["y_hi"]):
            raise ValueError(f"point {point_idx} nut_y out of tile bounds: {y}")
        if not (edges["yaw_lo"] <= yaw < edges["yaw_hi"]):
            raise ValueError(f"point {point_idx} nut_yaw out of tile bounds: {yaw}")
        if not (-math.pi <= yaw < math.pi):
            raise ValueError(f"point {point_idx} nut_yaw out of [-pi, pi): {yaw}")
        counts[cell_idx] += 1

    bad_counts = np.flatnonzero(counts != points_per_cell)
    if len(bad_counts):
        raise ValueError(
            f"cells with counts != {points_per_cell}: "
            f"{[(int(i), int(counts[i])) for i in bad_counts[:10]]}"
        )


def validate_valid_sobol_manifest(manifest: dict[str, Any]) -> None:
    expected_hash = manifest.get("manifest_hash")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("manifest missing non-empty manifest_hash")
    actual_hash = manifest_hash(manifest)
    if actual_hash != expected_hash:
        raise ValueError(f"manifest hash mismatch: expected {expected_hash}, got {actual_hash}")
    spec = _task_spec(str(manifest["task_key"]))
    if manifest["task"] != spec.task:
        raise ValueError(f"manifest task mismatch: {manifest['task']!r} != {spec.task!r}")
    if manifest["task_variant"] != spec.task_variant:
        raise ValueError(
            f"manifest task_variant mismatch: {manifest['task_variant']!r} != {spec.task_variant!r}"
        )
    if manifest["env_name"] != spec.env_name:
        raise ValueError(
            f"manifest env_name mismatch: {manifest['env_name']!r} != {spec.env_name!r}"
        )
    if manifest["protocol"] != spec.protocol:
        raise ValueError(f"manifest protocol mismatch: {manifest['protocol']!r}")
    if manifest["sampler"] != spec.sampler:
        raise ValueError(f"manifest sampler mismatch: {manifest['sampler']!r} != {spec.sampler!r}")
    if manifest["grid"] != spec.to_grid_dict():
        raise ValueError("manifest grid payload does not match locked task spec")
    if spec.min_clearance is None:
        if manifest["min_clearance"] is not None:
            raise ValueError(f"manifest min_clearance should be null: {manifest['min_clearance']}")
    elif not math.isclose(float(manifest["min_clearance"]), spec.min_clearance, abs_tol=1e-12):
        raise ValueError(
            f"manifest min_clearance mismatch: {manifest['min_clearance']} != {spec.min_clearance}"
        )
    if int(manifest["num_candidates"]) != 2 ** int(manifest["candidate_power"]):
        raise ValueError(
            f"num_candidates mismatch: {manifest['num_candidates']} != 2^{manifest['candidate_power']}"
        )

    points = manifest.get("points")
    if not isinstance(points, list):
        raise ValueError("manifest points must be a list")
    if int(manifest.get("num_points", -1)) != len(points):
        raise ValueError(f"num_points mismatch: {manifest.get('num_points')} != {len(points)}")
    counts = np.zeros(spec.n_cells, dtype=np.int64)
    prev_candidate_idx = -1
    for expected_point_idx, point in enumerate(points):
        point_idx = int(point["point_idx"])
        if point_idx != expected_point_idx:
            raise ValueError(f"point_idx at row {expected_point_idx} is {point_idx}")
        candidate_idx = int(point["candidate_idx"])
        if candidate_idx <= prev_candidate_idx:
            raise ValueError(f"candidate_idx sequence is not strictly increasing at {point_idx}")
        prev_candidate_idx = candidate_idx
        for axis in spec.axes:
            value = float(point[axis.value_key])
            lo, hi = axis.range
            if not (lo <= value < hi):
                raise ValueError(f"point {point_idx} {axis.value_key} out of range: {value}")
        if spec.min_clearance is not None:
            dist = math.hypot(
                float(point["nut_x"]) - float(point["peg_x"]),
                float(point["nut_y"]) - float(point["peg_y"]),
            )
            if dist < spec.min_clearance:
                raise ValueError(f"point {point_idx} violates min_clearance: {dist}")
        indices = _cell_indices_from_point(spec, point)
        expected_cell_idx = _flat_cell_idx(spec, indices)
        if int(point["cell_idx"]) != expected_cell_idx:
            raise ValueError(f"point {point_idx} cell_idx mismatch")
        for axis, expected_idx in zip(spec.axes, indices, strict=True):
            if int(point[f"{axis.name}_idx"]) != expected_idx:
                raise ValueError(f"point {point_idx} {axis.name}_idx mismatch")
        counts[expected_cell_idx] += 1

    if int(manifest["occupied_cells"]) != int(np.count_nonzero(counts)):
        raise ValueError("occupied_cells mismatch")
    if int(manifest["cell_count_min"]) != int(counts.min()):
        raise ValueError("cell_count_min mismatch")
    if int(manifest["cell_count_max"]) != int(counts.max()):
        raise ValueError("cell_count_max mismatch")
    if bool(manifest.get("require_all_cells", True)) and np.any(counts == 0):
        empty = np.flatnonzero(counts == 0)
        raise ValueError(
            f"valid Sobol manifest has empty cells: {len(empty)} empty, first={empty[:20].tolist()}"
        )


def _manifest_task_spec(manifest: dict[str, Any]) -> TaskSpec | None:
    if manifest.get("schema") == SCHEMA_V2:
        return _task_spec(str(manifest["task_key"]))
    return None


def point_to_placement(
    point: dict[str, Any],
    nut_qpos_start: int,
    *,
    task_key: str = "square_narrow",
) -> list[tuple[str, int | str, np.ndarray | list[float]]]:
    nut_placement = (
        "qpos",
        nut_qpos_start,
        nut_qpos(float(point["nut_x"]), float(point["nut_y"]), float(point["nut_yaw"])),
    )
    placements: list[tuple[str, int | str, np.ndarray | list[float]]] = []
    if task_key == "square_broad":
        placements.append(("body_pos", "peg1", [float(point["peg_x"]), float(point["peg_y"])]))
    placements.append(nut_placement)
    return placements


def _get_nut_qpos_start(env) -> int:
    nut = env.nuts[0]
    joint_id = env.sim.model.joint_name2id(nut.joints[0])
    return int(env.sim.model.jnt_qposadr[joint_id])


def _shard_range(total: int, shard_idx: int | None, n_shards: int | None) -> tuple[int, int]:
    if shard_idx is None and n_shards is None:
        return 0, total
    if shard_idx is None or n_shards is None:
        raise ValueError("shard_idx and n_shards must be provided together")
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive; got {n_shards}")
    if shard_idx < 1 or shard_idx > n_shards:
        raise ValueError(f"shard_idx must be in [1,{n_shards}]; got {shard_idx}")
    start = (total * (shard_idx - 1)) // n_shards
    end = (total * shard_idx) // n_shards
    return start, end


def _result_paths(output_dir: Path, shard_idx: int | None, n_shards: int | None) -> dict[str, Path]:
    if shard_idx is None:
        return {
            "results": output_dir / "results.json",
            "point_results": output_dir / "point_results.json",
            "cell_results": output_dir / "cell_results.json",
        }
    suffix = f"_shard_{shard_idx}_of_{n_shards}"
    return {
        "results": output_dir / f"results{suffix}.json",
        "point_results": output_dir / f"point_results{suffix}.json",
        "cell_results": output_dir / f"cell_results{suffix}.json",
    }


def _resume_path(output_dir: Path, shard_idx: int | None, n_shards: int | None) -> Path:
    suffix = "" if shard_idx is None else f"_shard_{shard_idx}_of_{n_shards}"
    return output_dir / f"resume{suffix}.json"


def _resume_config(
    *,
    artifact_path: str,
    manifest_hash_value: str,
    max_steps: int,
    num_envs: int,
    device: str,
    use_sync_envs: bool,
    num_action_samples: int | None,
    shard_idx: int | None,
    n_shards: int | None,
    point_start_idx: int,
    point_end_idx: int,
    eval_seed: int | None = None,
    env_seed: int | None = None,
) -> dict[str, Any]:
    config = {
        "artifact_path": artifact_path,
        "manifest_hash": manifest_hash_value,
        "max_steps": max_steps,
        "num_envs": num_envs,
        "device": device,
        "use_sync_envs": use_sync_envs,
        "num_action_samples": num_action_samples,
        "shard_idx": shard_idx,
        "n_shards": n_shards,
        "point_start_idx": point_start_idx,
        "point_end_idx": point_end_idx,
    }
    # Seeds are recorded only when set.
    if eval_seed is not None:
        config["eval_seed"] = eval_seed
        config["eval_seed_strategy"] = EVAL_SEED_STRATEGY
    if env_seed is not None:
        config["env_seed"] = env_seed
        config["env_seed_strategy"] = ENV_SEED_STRATEGY
    return config


def _seed_eval_batch(eval_seed: int, batch: list[dict[str, Any]]) -> int:
    """Seed stochastic policy sampling reproducibly for one ordered point batch."""
    if eval_seed < 0:
        raise ValueError(f"eval_seed must be non-negative; got {eval_seed}")
    if not batch:
        raise ValueError("cannot seed an empty evaluation batch")
    point_indices = [int(point["point_idx"]) for point in batch]
    payload = f"{EVAL_SEED_STRATEGY}|{eval_seed}|{','.join(map(str, point_indices))}"
    batch_seed = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    batch_seed %= 2**63 - 1

    import torch

    random.seed(batch_seed)
    np.random.seed(batch_seed % 2**32)
    torch.manual_seed(batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(batch_seed)
    return batch_seed


def _point_env_seeds(env_seed: int, batch: list[dict[str, Any]]) -> list[int]:
    """One reset seed per start, from ``env_seed`` and the start's point index only.

    Each start's reset noise is then independent of --num-envs, of the batch it
    lands in and of a resume.
    """
    seeds = []
    for point in batch:
        payload = f"{ENV_SEED_STRATEGY}|{env_seed}|{int(point['point_idx'])}"
        seeds.append(int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big"))
    return seeds


def _load_resume_results(
    *,
    path: Path,
    expected_config: dict[str, Any],
    points: list[dict[str, Any]],
    num_envs: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        payload = json.load(f)
    if payload.get("schema") != EVAL_RESUME_SCHEMA:
        raise ValueError(f"unsupported eval resume schema in {path}: {payload.get('schema')!r}")
    if payload.get("config") != expected_config:
        raise ValueError(
            f"eval resume config mismatch in {path}; refusing to mix results from a "
            "different artifact, manifest, shard, or evaluation configuration"
        )
    point_results = payload.get("point_results")
    if not isinstance(point_results, list):
        raise ValueError(f"eval resume point_results must be a list in {path}")
    if len(point_results) > len(points):
        raise ValueError(
            f"eval resume has {len(point_results)} results for only {len(points)} assigned points"
        )
    if len(point_results) % num_envs != 0:
        raise ValueError(
            f"eval resume result count {len(point_results)} is not a complete num_envs={num_envs} batch"
        )
    expected_indices = [int(point["point_idx"]) for point in points[: len(point_results)]]
    actual_indices = [int(row["point_idx"]) for row in point_results]
    if actual_indices != expected_indices:
        raise ValueError(
            f"eval resume point coverage is not the required contiguous prefix in {path}: "
            f"expected {expected_indices[:1]}..{expected_indices[-1:]}, "
            f"got {actual_indices[:1]}..{actual_indices[-1:]}"
        )
    for point, row in zip(points, point_results, strict=False):
        if int(row["cell_idx"]) != int(point["cell_idx"]):
            raise ValueError(
                f"eval resume cell_idx mismatch for point {point['point_idx']} in {path}"
            )
    return point_results


def _save_resume_results(
    *, path: Path, config: dict[str, Any], point_results: list[dict[str, Any]]
) -> None:
    write_json_atomic(
        path,
        {
            "schema": EVAL_RESUME_SCHEMA,
            "config": config,
            "point_results": point_results,
        },
    )


def _cell_metadata(manifest: dict[str, Any], cell_idx: int) -> dict[str, Any]:
    if manifest.get("schema") == SCHEMA_V1:
        x_idx, y_idx, yaw_idx = EqualTileGrid().cell_indices(cell_idx)
        return {"x_idx": x_idx, "y_idx": y_idx, "yaw_idx": yaw_idx}
    spec = _task_spec(str(manifest["task_key"]))
    indices = _cell_indices_from_flat(spec, cell_idx)
    return {f"{axis.name}_idx": int(idx) for axis, idx in zip(spec.axes, indices, strict=True)}


def summarize_results(
    *,
    manifest: dict[str, Any],
    point_results: list[dict[str, Any]],
    require_complete: bool = True,
) -> dict[str, Any]:
    manifest_num_points = int(manifest["num_points"])
    if require_complete and len(point_results) != manifest_num_points:
        raise ValueError(f"expected {manifest_num_points} point results, got {len(point_results)}")

    grid = manifest["grid"]
    n_cells = int(grid["n_cells"])
    by_cell: list[list[dict[str, Any]]] = [[] for _ in range(n_cells)]
    seen: set[int] = set()
    for row in point_results:
        point_idx = int(row["point_idx"])
        if point_idx in seen:
            raise ValueError(f"duplicate point_idx in results: {point_idx}")
        seen.add(point_idx)
        by_cell[int(row["cell_idx"])].append(row)

    cell_results = []
    for cell_idx, rows in enumerate(by_cell):
        if (
            require_complete
            and manifest.get("schema") == SCHEMA_V1
            and len(rows) != int(manifest["points_per_cell"])
        ):
            raise ValueError(
                f"cell {cell_idx} expected {manifest['points_per_cell']} results, got {len(rows)}"
            )
        if rows:
            successes = [int(r["success"]) for r in rows]
            lengths = [int(r["length"]) for r in rows]
            rewards = [float(r["reward"]) for r in rows]
            success_rate = float(np.mean(successes))
            avg_reward = float(np.mean(rewards))
            avg_length = float(np.mean(lengths))
        else:
            success_rate = None
            avg_reward = None
            avg_length = None
        cell_results.append(
            {
                "cell_idx": cell_idx,
                **_cell_metadata(manifest, cell_idx),
                "n_points": len(rows),
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "avg_length": avg_length,
            }
        )

    all_successes = [int(r["success"]) for r in point_results]
    non_empty_rates = [
        float(c["success_rate"]) for c in cell_results if c["success_rate"] is not None
    ]
    return {
        "overall_success_rate": float(np.mean(all_successes)) if all_successes else None,
        "overall_std_across_cells": float(np.std(non_empty_rates)) if non_empty_rates else None,
        "num_points": len(point_results),
        "manifest_num_points": manifest_num_points,
        "num_cells": n_cells,
        "occupied_cells": int(sum(1 for rows in by_cell if rows)),
        "partial": len(point_results) != manifest_num_points,
        "points_per_cell": manifest.get("points_per_cell"),
        "cells": cell_results,
    }


def _point_result_from_rollout(
    point: dict[str, Any],
    *,
    success: bool,
    reward: float,
    length: int,
    task_key: str,
) -> dict[str, Any]:
    row = {
        "point_idx": int(point["point_idx"]),
        "candidate_idx": int(point.get("candidate_idx", point["point_idx"])),
        "cell_idx": int(point["cell_idx"]),
        "nut_x": float(point["nut_x"]),
        "nut_y": float(point["nut_y"]),
        "nut_yaw": float(point["nut_yaw"]),
        "success": int(success),
        "reward": float(reward),
        "length": int(length),
    }
    if "within_cell_idx" in point:
        row["within_cell_idx"] = int(point["within_cell_idx"])
    for key, value in point.items():
        if key.endswith("_idx") and key not in row:
            row[key] = int(value)
    if task_key == "square_broad":
        row["peg_x"] = float(point["peg_x"])
        row["peg_y"] = float(point["peg_y"])
        row["nut_peg_distance"] = float(point["nut_peg_distance"])
    return row


def _make_eval_env(env_name: str, robot_name: str):
    """Module-level environment factory (picklable for AsyncVectorEnv workers)."""
    from mulligan.sim.envs import create_robosuite_env

    return create_robosuite_env(
        env_name=env_name,
        robot_name=robot_name,
        camera_names=None,
        visual_aids=False,
        # No videos: skip the offscreen GL context (no effect on the physics).
        use_render_wrapper=False,
    )


def step_rollouts_from_obs(
    vec_env,
    policy,
    preprocessor,
    postprocessor,
    obs_list: list[dict],
    max_steps: int,
    device: str,
) -> tuple[list[bool], list[float], list[int]]:
    """Run one episode per worker of an already-reset vector env.

    The caller places each env in its initial state and passes the resulting
    observations. An episode ends on success, ``done`` or ``max_steps``.

    Returns:
        (successes, rewards, lengths), one entry per env.
    """
    import torch

    from mulligan.agents.idql import IDQLPolicy
    from mulligan.training.normalization import Normalizer

    num_envs = len(vec_env)
    policy.reset()

    robot_state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

    env_done = [False] * num_envs
    env_success = [False] * num_envs
    env_rewards = [0.0] * num_envs
    env_lengths = [0] * num_envs

    for _ in range(max_steps):
        if all(env_done):
            break

        robot_states = []
        env_states = []
        for obs in obs_list:
            state_parts = [obs[k].flatten() for k in robot_state_keys if k in obs]
            robot_states.append(np.concatenate(state_parts))
            env_state = obs.get("object-state", None)
            env_states.append(env_state.flatten() if env_state is not None else None)

        robot_state_dim = len(robot_states[0])

        # IDQL checkpoints: z-score robot and object state separately.
        if not isinstance(preprocessor, Normalizer) or not isinstance(policy, IDQLPolicy):
            raise TypeError(
                "grid evaluation is only defined for IDQL policies with a Normalizer, "
                f"got {type(policy).__name__} with {type(preprocessor).__name__}"
            )
        robot_mean = preprocessor.state_mean[:robot_state_dim]
        robot_std = preprocessor.state_std[:robot_state_dim]
        env_mean = preprocessor.state_mean[robot_state_dim:]
        env_std = preprocessor.state_std[robot_state_dim:]

        robot_batch = torch.from_numpy(np.stack(robot_states)).float().to(device)
        obs_dict = {"observation.state": (robot_batch - robot_mean) / robot_std}
        if env_states[0] is not None:
            env_batch = torch.from_numpy(np.stack(env_states)).float().to(device)
            obs_dict["observation.environment_state"] = (env_batch - env_mean) / env_std

        with torch.no_grad():
            actions_normalized = policy.select_action(obs_dict)
            actions = postprocessor.denormalize_action(actions_normalized).cpu().numpy()

        obs_list, rewards, dones, infos = vec_env.step(list(actions))

        for env_idx in range(num_envs):
            if env_done[env_idx]:
                continue
            env_rewards[env_idx] += rewards[env_idx]
            env_lengths[env_idx] += 1
            is_success = infos[env_idx].get("success", False)
            if is_success:
                env_success[env_idx] = True
            if dones[env_idx] or is_success or env_lengths[env_idx] >= max_steps:
                env_done[env_idx] = True

    return env_success, env_rewards, env_lengths


def _eta_s(
    *, start_time: float, start_done: int, done: int, total: int, now: float | None = None
) -> float | None:
    """Seconds left at this process's rate; a resumed prefix does not count as throughput."""
    if done >= total:
        return 0.0
    completed = done - start_done
    elapsed = (time.monotonic() if now is None else now) - start_time
    if completed <= 0 or elapsed <= 0:
        return None
    return (total - done) * elapsed / completed


def evaluate_manifest(
    *,
    artifact_path: str,
    manifest_path: Path,
    output_dir: Path,
    max_steps: int = 400,
    num_envs: int = 10,
    device: str = "cuda",
    use_sync_envs: bool = False,
    num_action_samples: int | None = 32,
    shard_idx: int | None = None,
    n_shards: int | None = None,
    eval_seed: int | None = None,
    env_seed: int | None = None,
) -> dict[str, Any]:
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive; got {num_envs}")
    if eval_seed is not None and eval_seed < 0:
        raise ValueError(f"eval_seed must be non-negative; got {eval_seed}")
    if env_seed is not None and env_seed < 0:
        raise ValueError(f"env_seed must be non-negative; got {env_seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _result_paths(output_dir, shard_idx, n_shards)
    outputs = [str(path) for path in paths.values()]
    try:
        manifest = load_manifest(manifest_path)
        spec = _manifest_task_spec(manifest)
        task_key = spec.key if spec is not None else "square_narrow"
        env_name = str(manifest.get("env_name") or manifest["task"])
        points_all = manifest["points"]
        start_idx, end_idx = _shard_range(len(points_all), shard_idx, n_shards)
        points = points_all[start_idx:end_idx]
        if len(points) % num_envs != 0:
            raise ValueError(
                f"num_points must be divisible by num_envs because reset_with_placements "
                f"requires one placement per vector worker; got shard size {len(points)} "
                f"and num_envs={num_envs}"
            )
        resume_path = _resume_path(output_dir, shard_idx, n_shards)
        resume_config = _resume_config(
            artifact_path=artifact_path,
            manifest_hash_value=str(manifest["manifest_hash"]),
            max_steps=max_steps,
            num_envs=num_envs,
            device=device,
            use_sync_envs=use_sync_envs,
            num_action_samples=num_action_samples,
            shard_idx=shard_idx,
            n_shards=n_shards,
            point_start_idx=start_idx,
            point_end_idx=end_idx,
            eval_seed=eval_seed,
            env_seed=env_seed,
        )
        point_results = _load_resume_results(
            path=resume_path,
            expected_config=resume_config,
            points=points,
            num_envs=num_envs,
        )
        resumed_points = len(point_results)
        if resumed_points:
            print(
                f"✓ Resuming manifest evaluation from {resume_path}: "
                f"{resumed_points}/{len(points)} points complete"
            )
        emit_progress(
            job_type="grid-eval",
            phase="start",
            status="running",
            progress=resumed_points / max(1, len(points)),
            current=resumed_points,
            total=len(points),
            message=(
                "manifest point evaluation resumed"
                if resumed_points
                else "manifest point evaluation started"
            ),
            metrics={
                "artifact_path": artifact_path,
                "manifest_path": str(manifest_path),
                "manifest_hash": manifest["manifest_hash"],
                "task_key": task_key,
                "env_name": env_name,
                "num_points": len(points),
                "manifest_num_points": len(points_all),
                "point_start_idx": start_idx,
                "point_end_idx": end_idx,
                "num_cells": manifest["grid"]["n_cells"],
                "shard_idx": shard_idx,
                "n_shards": n_shards,
                "num_envs": num_envs,
                "max_steps": max_steps,
                "num_action_samples": num_action_samples,
                "eval_seed": eval_seed,
                "env_seed": env_seed,
            },
            outputs=outputs,
        )

        def emit_success(results: dict[str, Any]) -> None:
            emit_completion(
                job_type="grid-eval",
                success=True,
                current=len(points),
                total=len(points),
                exit_code=0,
                message=(
                    "manifest point evaluation complete "
                    f"success_rate={results['overall_success_rate'] * 100:.1f}%"
                ),
                metrics={
                    "overall_success_rate": results["overall_success_rate"],
                    "manifest_hash": manifest["manifest_hash"],
                    "num_points": len(points),
                    "num_cells": manifest["grid"]["n_cells"],
                    "shard_idx": shard_idx,
                },
                outputs=outputs,
            )

        def finalize_results() -> dict[str, Any]:
            summary = summarize_results(
                manifest=manifest,
                point_results=point_results,
                require_complete=shard_idx is None,
            )
            results = {
                "schema": RESULTS_SCHEMA,
                "task": manifest["task"],
                "task_variant": manifest.get("task_variant"),
                "task_key": task_key,
                "artifact_path": artifact_path,
                "manifest_path": str(manifest_path),
                "manifest_hash": manifest["manifest_hash"],
                "config": {
                    "max_steps": max_steps,
                    "num_envs": num_envs,
                    "device": device,
                    "use_sync_envs": use_sync_envs,
                    "num_action_samples": num_action_samples,
                    "shard_idx": shard_idx,
                    "n_shards": n_shards,
                    "point_start_idx": start_idx,
                    "point_end_idx": end_idx,
                    "eval_seed": eval_seed,
                    "eval_seed_strategy": EVAL_SEED_STRATEGY if eval_seed is not None else None,
                    "env_seed": env_seed,
                    "env_seed_strategy": ENV_SEED_STRATEGY if env_seed is not None else None,
                },
                "manifest": {key: value for key, value in manifest.items() if key != "points"},
                **summary,
                "point_results_path": str(paths["point_results"]),
                "cell_results_path": str(paths["cell_results"]),
            }
            write_json_atomic(paths["point_results"], point_results)
            write_json_atomic(paths["cell_results"], summary["cells"])
            write_json_atomic(paths["results"], results)
            _save_resume_results(
                path=resume_path,
                config=resume_config,
                point_results=point_results,
            )
            return results

        if resumed_points == len(points):
            print(
                f"✓ Resume already covers all {len(points)} assigned points; "
                "finalizing result sidecars without policy or GPU initialization"
            )
            results = finalize_results()
            emit_success(results)
            return results

        from mulligan.sim.vec_env import AsyncVectorEnv, SyncVectorEnv
        from mulligan.utils.load_pretrained import load_policy_from_checkpoint

        print(f"Resolving checkpoint: {artifact_path}")
        artifact_dir = resolve_checkpoint(artifact_path)
        print(f"Using checkpoint directory: {artifact_dir}")

        policy, preprocessor, postprocessor = load_policy_from_checkpoint(
            checkpoint_path=artifact_dir,
            device=device,
        )
        policy.eval()

        _configure_eval_action_samples(policy, num_action_samples)

        ref_env = _make_eval_env(env_name=env_name, robot_name="Panda")
        ref_env.reset()
        nut_qpos_start = _get_nut_qpos_start(ref_env)
        ref_env.close()

        make_env = partial(_make_eval_env, env_name=env_name, robot_name="Panda")
        env_fns = [make_env for _ in range(num_envs)]
        vec_env = SyncVectorEnv(env_fns) if use_sync_envs else AsyncVectorEnv(env_fns)

        start_time = time.monotonic()
        try:
            for start in range(resumed_points, len(points), num_envs):
                batch = points[start : start + num_envs]
                if eval_seed is not None:
                    _seed_eval_batch(eval_seed, batch)
                if env_seed is not None:
                    vec_env.seed_envs(_point_env_seeds(env_seed, batch))
                placements = [
                    point_to_placement(point, nut_qpos_start, task_key=task_key) for point in batch
                ]
                obs_list, _qpos_list, _qvel_list = vec_env.reset_with_placements(placements)
                successes, rewards, lengths = step_rollouts_from_obs(
                    vec_env=vec_env,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    obs_list=obs_list,
                    max_steps=max_steps,
                    device=device,
                )
                for point, success, reward, length in zip(
                    batch, successes, rewards, lengths, strict=True
                ):
                    point_results.append(
                        _point_result_from_rollout(
                            point,
                            success=success,
                            reward=reward,
                            length=length,
                            task_key=task_key,
                        )
                    )
                done = len(point_results)
                batch_number = done // num_envs
                if batch_number % EVAL_RESUME_BATCH_INTERVAL == 0 or done == len(points):
                    _save_resume_results(
                        path=resume_path,
                        config=resume_config,
                        point_results=point_results,
                    )
                    print(f"✓ Saved eval resume checkpoint at {done}/{len(points)} points")
                running_sr = float(np.mean([r["success"] for r in point_results]))
                emit_progress(
                    job_type="grid-eval",
                    phase="evaluate",
                    status="running",
                    progress=done / max(1, len(points)),
                    current=done,
                    total=len(points),
                    eta_s=_eta_s(
                        start_time=start_time,
                        start_done=resumed_points,
                        done=done,
                        total=len(points),
                    ),
                    message=f"points {done}/{len(points)} running_sr={running_sr * 100:.1f}%",
                    metrics={
                        "points_done": done,
                        "num_points": len(points),
                        "last_point_idx": int(point_results[-1]["point_idx"]),
                        "last_cell_idx": int(point_results[-1]["cell_idx"]),
                        "running_success_rate": running_sr,
                        "shard_idx": shard_idx,
                    },
                )
            # Write the results before closing the envs: MuJoCo/EGL teardown can
            # fail after every rollout finished. That failure still propagates, and
            # the success event waits for close().
            results = finalize_results()
        finally:
            vec_env.close()

        emit_success(results)
        return results
    except Exception as exc:
        emit_completion(
            job_type="grid-eval",
            success=False,
            exit_code=1,
            message=f"manifest point evaluation failed: {type(exc).__name__}: {exc}",
            metrics={"manifest_path": str(manifest_path), "artifact_path": artifact_path},
            outputs=outputs,
        )
        raise


# Shard settings that change outcomes; a merge refuses shards that differ in any.
MERGE_AGREEMENT_KEYS = ("num_action_samples", "max_steps", "eval_seed", "env_seed")


def merge_shards(
    *,
    manifest_path: Path,
    output_dir: Path,
    n_shards: int,
) -> dict[str, Any]:
    """Merge the ``n_shards`` shard results in ``output_dir`` into one results.json.

    Every shard must be present and cover its point range, and the shards must agree on
    the checkpoint and on the settings that change outcomes (:data:`MERGE_AGREEMENT_KEYS`).
    """
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive; got {n_shards}")
    manifest = load_manifest(manifest_path)
    point_results: list[dict[str, Any]] = []
    artifact_paths: set[str] = set()
    configs: list[dict[str, Any]] = []
    for shard_idx in range(1, n_shards + 1):
        shard_paths = _result_paths(output_dir, shard_idx, n_shards)
        if not shard_paths["point_results"].exists():
            raise FileNotFoundError(f"missing shard point results: {shard_paths['point_results']}")
        if not shard_paths["results"].exists():
            raise FileNotFoundError(f"missing shard results: {shard_paths['results']}")
        with shard_paths["point_results"].open() as f:
            shard_point_results = json.load(f)
        with shard_paths["results"].open() as f:
            shard_results = json.load(f)
        if shard_results["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError(f"manifest hash mismatch in shard {shard_idx}")
        artifact_paths.add(shard_results["artifact_path"])
        configs.append(shard_results["config"])
        expected_start, expected_end = _shard_range(
            int(manifest["num_points"]), shard_idx, n_shards
        )
        point_indices = [int(row["point_idx"]) for row in shard_point_results]
        if point_indices != list(range(expected_start, expected_end)):
            raise ValueError(
                f"shard {shard_idx} point_idx coverage mismatch: "
                f"expected [{expected_start},{expected_end}), "
                f"got first/last={point_indices[:1]}/{point_indices[-1:]}"
            )
        point_results.extend(shard_point_results)

    if len(artifact_paths) != 1:
        raise ValueError(f"shards disagree on artifact_path: {sorted(artifact_paths)}")
    for key in MERGE_AGREEMENT_KEYS:
        values = [config.get(key) for config in configs]
        if len(set(values)) != 1:
            raise ValueError(f"shards disagree on {key}: {values} (shards 1..{n_shards})")
    point_results.sort(key=lambda row: int(row["point_idx"]))
    if len(point_results) != int(manifest["num_points"]):
        raise ValueError(
            f"shards cover {len(point_results)} of {manifest['num_points']} manifest points"
        )
    summary = summarize_results(
        manifest=manifest,
        point_results=point_results,
        require_complete=True,
    )
    paths = _result_paths(output_dir, None, None)
    results = {
        "schema": RESULTS_SCHEMA,
        "task": manifest["task"],
        "task_variant": manifest.get("task_variant"),
        "task_key": manifest.get("task_key"),
        "artifact_path": next(iter(artifact_paths)),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "config": {
            "merged_from_shards": n_shards,
            "shard_configs": configs,
        },
        "manifest": {key: value for key, value in manifest.items() if key != "points"},
        **summary,
        "point_results_path": str(paths["point_results"]),
        "cell_results_path": str(paths["cell_results"]),
    }
    write_json_atomic(paths["point_results"], point_results)
    write_json_atomic(paths["cell_results"], summary["cells"])
    write_json_atomic(paths["results"], results)
    print(
        f"Merged {len(point_results)} points from {n_shards} shards into {paths['results']} "
        f"overall_sr={summary['overall_success_rate']}"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_old = subparsers.add_parser("make-equal-tile-manifest")
    make_old.add_argument("--output", type=Path, required=True)
    make_old.add_argument("--points-per-cell", type=int, default=100)
    make_old.add_argument("--seed", type=int, default=20260524)

    make = subparsers.add_parser("make-valid-sobol-manifest")
    make.add_argument("--task", required=True, choices=sorted(TASK_ALIASES))
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--num-points", type=int, default=None)
    make.add_argument("--seed", type=int, default=None)
    make.add_argument("--candidate-power", type=int, default=None)
    make.add_argument("--allow-empty-cells", action="store_true")

    eval_p = subparsers.add_parser("eval")
    eval_p.add_argument(
        "--artifact-path",
        required=True,
        help="Local checkpoint directory or hf://<org>/<repo>@<revision>/<subdir>.",
    )
    eval_p.add_argument("--point-manifest", type=Path, required=True)
    eval_p.add_argument("--output-dir", type=Path, required=True)
    eval_p.add_argument("--max-steps", type=int, default=400)
    eval_p.add_argument("--num-envs", type=int, default=10)
    eval_p.add_argument("--device", type=str, default="cuda")
    eval_p.add_argument("--use-sync-envs", action="store_true")
    eval_p.add_argument(
        "--num-action-samples",
        type=int,
        default=32,
        help="IDQL best-of-N candidates per step (the paper uses 32).",
    )
    eval_p.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "Seed stochastic policy sampling per ordered point batch. Required for "
            "reproducible matched checkpoint comparisons."
        ),
    )
    eval_p.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help=(
            "Seed the environment resets: each start's reset uses a seed derived from "
            "env_seed and its point index. Default: unseeded."
        ),
    )
    eval_p.add_argument(
        "--shard-idx",
        type=int,
        default=None,
        help="1-based shard index; shard k of n evaluates a contiguous slice of the points.",
    )
    eval_p.add_argument("--n-shards", type=int, default=None)

    merge = subparsers.add_parser(
        "merge",
        description=(
            "Merge the shards of one or more output directories into results.json. "
            "The point manifest and shard count default to what the shards record. "
            "--status only reports whether each directory is ready; --verify checks that "
            "every results.json below the given roots covers its whole manifest."
        ),
    )
    merge.add_argument(
        "paths", nargs="*", type=Path, help="Shard directories (roots with --auto-find/--verify)."
    )
    merge.add_argument("--output-dir", type=Path, default=None, help="One shard directory.")
    merge.add_argument("--point-manifest", type=Path, default=None)
    merge.add_argument("--n-shards", type=int, default=None)
    merge.add_argument(
        "--auto-find",
        action="store_true",
        help="Merge every directory below the given roots that has shards and no results.json.",
    )
    mode = merge.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="Report shard status; do not merge.")
    mode.add_argument("--verify", action="store_true", help="Check merged results.json files.")

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--point-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "make-equal-tile-manifest":
        manifest = make_equal_tile_manifest(
            points_per_cell=args.points_per_cell,
            seed=args.seed,
        )
        write_json_atomic(args.output, manifest)
        print(
            f"Wrote {args.output} with {manifest['num_points']} points, "
            f"hash={manifest['manifest_hash']}"
        )
        return 0
    if args.command == "make-valid-sobol-manifest":
        manifest = make_valid_sobol_manifest(
            task=args.task,
            num_points=args.num_points,
            seed=args.seed,
            candidate_power=args.candidate_power,
            require_all_cells=not args.allow_empty_cells,
        )
        write_json_atomic(args.output, manifest)
        print(
            f"Wrote {args.output} with {manifest['num_points']} points, "
            f"occupied_cells={manifest['occupied_cells']}/{manifest['grid']['n_cells']}, "
            f"cell_count_range=[{manifest['cell_count_min']},{manifest['cell_count_max']}], "
            f"hash={manifest['manifest_hash']}"
        )
        return 0
    if args.command == "validate-manifest":
        manifest = load_manifest(args.point_manifest)
        print(
            f"OK {args.point_manifest}: {manifest['num_points']} points, "
            f"occupied_cells={manifest.get('occupied_cells', manifest['grid']['n_cells'])}/"
            f"{manifest['grid']['n_cells']}, hash={manifest['manifest_hash']}"
        )
        return 0
    if args.command == "eval":
        evaluate_manifest(
            artifact_path=args.artifact_path,
            manifest_path=args.point_manifest,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            num_envs=args.num_envs,
            device=args.device,
            use_sync_envs=args.use_sync_envs,
            num_action_samples=args.num_action_samples,
            shard_idx=args.shard_idx,
            n_shards=args.n_shards,
            eval_seed=args.eval_seed,
            env_seed=args.env_seed,
        )
        return 0
    if args.command == "merge":
        from mulligan.sim.eval import shards

        paths = [*args.paths, *([args.output_dir] if args.output_dir else [])]
        if not paths:
            raise SystemExit("grid_eval merge: give at least one directory")
        if args.status:
            return shards.run_status(paths)
        if args.verify:
            return shards.run_verify(paths)
        return shards.run_merge(
            paths,
            auto_find=args.auto_find,
            point_manifest=args.point_manifest,
            n_shards=args.n_shards,
        )
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
