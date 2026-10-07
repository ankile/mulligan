"""Shared builders for fresh real-world held-out eval manifests.

A fresh eval manifest draws a Sobol block that no earlier collection or eval
manifest used: :func:`build_fresh_eval_manifest` checks the block against the
stream registry (``FreshEvalManifestConfig.registry_path``, one row per claimed Sobol
range; the paper campaigns' registry is ``data/real/sobol_stream_ranges.csv``) and appends
its own claim when ``apply_registry=True``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mulligan.real.lifecycle.geometry import (
    TaskGeometry,
    unit_dist_to_point,
    wrap_angle,
)
from mulligan.real.lifecycle.tasks import (
    GridSampledPlacement,
    RealTaskSpec,
    get_task_spec,
)


@dataclass(frozen=True)
class EvalManifestStateOverride:
    manifest_idx: int
    values: dict[str, float]
    reason: str


@dataclass(frozen=True)
class FreshEvalManifestConfig:
    task: str
    output_path: Path
    registry_path: Path
    usage_id: str
    registry_source: str
    source: str
    sobol_seed: int
    count: int
    start_index: int
    phase: str
    purpose: str
    disjoint_from: str
    audit_manifests: tuple[Path, ...]
    order_policy: str
    order_policy_description: str
    audit_path_root: Path | None = None
    placement_name: str | None = None
    serpentine_outer_axis: str = "x"
    manual_state_corrections: tuple[EvalManifestStateOverride, ...] = ()
    match_tolerance: float = 1e-9
    disjoint_tolerance: float = 1e-6


def _parse_ranges(field: str) -> set[int]:
    out: set[int] = set()
    for token in str(field).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if ".." in token:
            lo, hi = token.split("..", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(token))
    return out


def _registry_collisions(cfg: FreshEvalManifestConfig, indices: set[int]) -> dict[str, list[int]]:
    if not cfg.registry_path.exists():
        raise RuntimeError(f"registry not found: {cfg.registry_path}")
    collisions: dict[str, list[int]] = {}
    with cfg.registry_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["usage_id"] == cfg.usage_id:
                continue
            if int(row["sobol_seed"]) != cfg.sobol_seed:
                continue
            overlap = sorted(indices & _parse_ranges(row["ranges"]))
            if overlap:
                collisions[row["usage_id"]] = overlap
    return collisions


def _compress_ranges(indices: set[int]) -> str:
    """Render a set of ints as a sorted, comma-joined ``a..b`` range string. Contiguous blocks
    collapse to one ``lo..hi`` token (a full contiguous claim renders as ``start..end``);
    rejection-sampled gaps produce multiple tokens."""
    if not indices:
        raise RuntimeError("cannot compress an empty stream-index set")
    ordered = sorted(indices)
    tokens: list[str] = []
    lo = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        tokens.append(f"{lo}..{prev}")
        lo = prev = value
    tokens.append(f"{lo}..{prev}")
    return ",".join(tokens)


def _registry_row(cfg: FreshEvalManifestConfig, stream: np.ndarray) -> dict[str, str]:
    # The claimed Sobol stream indices are the ACTUAL consumed ones (non-contiguous when
    # placement rejection sampling drops points), so the registry range never under-claims a
    # feasibility-filtered eval block. A contiguous block collapses to one ``start..end`` token,
    # identical to the pre-rejection behavior.
    idx = {int(v) for v in stream}
    return {
        "usage_id": cfg.usage_id,
        "status": "proposed",
        "source": cfg.registry_source,
        "sobol_seed": str(cfg.sobol_seed),
        "n_unique_stream_indices": str(len(idx)),
        "min_stream_index": str(min(idx)),
        "max_stream_index": str(max(idx)),
        "ranges": _compress_ranges(idx),
    }


def _append_registry_row(cfg: FreshEvalManifestConfig, stream: np.ndarray) -> None:
    desired = _registry_row(cfg, stream)
    with cfg.registry_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"{cfg.registry_path} is empty")
    fieldnames = list(rows[0].keys())
    existing = {row["usage_id"]: row for row in rows}.get(cfg.usage_id)
    if existing is not None:
        comparable_existing = {key: existing[key] for key in fieldnames}
        comparable_desired = {key: desired[key] for key in fieldnames}
        if comparable_existing != comparable_desired:
            raise RuntimeError(
                f"{cfg.registry_path}: existing {cfg.usage_id!r} differs: "
                f"{comparable_existing} != {comparable_desired}"
            )
        print(f"[registry] {cfg.usage_id} already present; no update needed")
        return
    with cfg.registry_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writerow(desired)
    print(f"[registry] appended {cfg.usage_id}")


def _min_pairwise(unit: np.ndarray, periodic_mask: np.ndarray) -> float:
    best = float("inf")
    for i, row in enumerate(unit):
        dists = unit_dist_to_point(unit, row, periodic_mask)
        dists[i] = float("inf")
        best = min(best, float(dists.min()))
    return best


def _min_cross(a_unit: np.ndarray, b_unit: np.ndarray, periodic_mask: np.ndarray) -> float:
    best = float("inf")
    for row in a_unit:
        best = min(best, float(unit_dist_to_point(b_unit, row, periodic_mask).min()))
    return best


def _states_from_manifest(path: Path, *, task: str, geom: TaskGeometry) -> np.ndarray:
    payload = json.loads(path.read_text())
    if payload.get("task") != task:
        raise RuntimeError(f"{path}: expected task {task}, got {payload.get('task')!r}")
    return geom.states_from_rows(payload["states"])


def _audit_manifest_label(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise RuntimeError(f"audit manifest {path} is not under audit_path_root {root}") from exc


def _unit_for_key(key: str) -> str:
    """Manifest unit label for a persisted column: radians for a resolved orientation angle,
    a bare ``index`` for a discrete orientation choice index, meters otherwise."""
    if key.endswith("_yaw"):
        return "rad"
    if key.endswith("_oidx"):
        return "index"
    return "m"


def _display_unit_for_key(key: str) -> str:
    if key.endswith("_yaw"):
        return "deg"
    if key.endswith("_oidx"):
        return "index"
    return "in"


def _display_bounds(geom: TaskGeometry) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key, (lo, hi) in zip(geom.keys, geom.bounds, strict=True):
        if key.endswith("_yaw"):
            out[f"{key}_deg"] = [float(np.degrees(lo)), float(np.degrees(hi))]
        elif key.endswith("_oidx"):
            # A discrete orientation index has no metric display; report the raw index range.
            out[key] = [float(lo), float(hi)]
        else:
            out[f"{key}_in"] = [float(lo * 39.37007874015748), float(hi * 39.37007874015748)]
    return out


def _manifest_header(cfg: FreshEvalManifestConfig, geom: TaskGeometry) -> dict:
    spec = geom.spec
    sampled_placements = {
        name: {
            "keys": list(placement.keys),
            "bounds": [list(axis) for axis in placement.bounds],
            "snap_m": placement.snap_m,
            "grid_dims": list(placement.grid_dims),
        }
        for name, placement in spec.sampled_placements.items()
    }
    return {
        "task": cfg.task,
        "schema": "real_manual_initial_states_v1",
        "keys": list(spec.manifest_keys),
        # Units are keyed over the full persisted manifest columns (which include any resolved
        # orientation ANGLE and its choice INDEX), not just the joint-sampling columns, so an
        # oidx never reads as meters and a resolved yaw is labeled radians.
        "units": {key: _unit_for_key(key) for key in spec.manifest_keys},
        "operator_frame": {
            "description": (
                f"Manual {cfg.task} placement targets in the operator frame. "
                "+x is forward, +y is left, yaw is the object heading. "
                "Sampled scene placements are recorded in the same frame."
            ),
            "display_units": {key: _display_unit_for_key(key) for key in spec.manifest_keys},
        },
        "bounds": {
            key: [float(lo), float(hi)]
            for key, (lo, hi) in zip(geom.keys, geom.bounds, strict=True)
        },
        "bounds_display": _display_bounds(geom),
        "sampled_placements": sampled_placements,
        "match_tolerance": cfg.match_tolerance,
    }


def _placement_key(row: dict, placement: GridSampledPlacement) -> tuple[float, float]:
    x_key, y_key = placement.keys
    return (round(float(row[x_key]), 6), round(float(row[y_key]), 6))


def _placement_changes(rows: list[dict], placement: GridSampledPlacement | None) -> int:
    if placement is None:
        return 0
    return sum(
        _placement_key(a, placement) != _placement_key(b, placement)
        for a, b in zip(rows, rows[1:], strict=False)
    )


def _apply_manual_state_corrections(
    rows: list[dict],
    corrections: tuple[EvalManifestStateOverride, ...],
) -> list[dict]:
    if not corrections:
        return []

    by_manifest_idx = {int(row["manifest_idx"]): row for row in rows}
    applied: list[dict] = []
    for correction in corrections:
        try:
            row = by_manifest_idx[correction.manifest_idx]
        except KeyError as exc:
            raise RuntimeError(
                f"manual correction references missing manifest_idx={correction.manifest_idx}"
            ) from exc
        updates: dict[str, dict[str, float]] = {}
        for key, value in correction.values.items():
            if key not in row:
                raise RuntimeError(
                    f"manual correction for manifest_idx={correction.manifest_idx} "
                    f"references missing key {key!r}"
                )
            old = float(row[key])
            new = float(value)
            row[key] = new
            updates[key] = {"old": old, "new": new}
        applied.append(
            {
                "manifest_idx": correction.manifest_idx,
                "updates": updates,
                "reason": correction.reason,
            }
        )
    return applied


def _placement_batched_order(
    rows: list[dict],
    *,
    placement: GridSampledPlacement,
    order_policy: str,
    outer_axis: str,
) -> list[dict]:
    xs = sorted({_placement_key(row, placement)[0] for row in rows})
    ys = sorted({_placement_key(row, placement)[1] for row in rows})
    cell_rank: dict[tuple[float, float], int] = {}
    rank = 0
    if outer_axis == "x":
        for outer_idx, x in enumerate(xs):
            inner_ys = ys if outer_idx % 2 == 0 else list(reversed(ys))
            for y in inner_ys:
                cell_rank[(x, y)] = rank
                rank += 1
    elif outer_axis == "y":
        for outer_idx, y in enumerate(ys):
            inner_xs = xs if outer_idx % 2 == 0 else list(reversed(xs))
            for x in inner_xs:
                cell_rank[(x, y)] = rank
                rank += 1
    else:
        raise ValueError(f"outer_axis must be 'x' or 'y', got {outer_axis!r}")

    ordered = sorted(
        rows, key=lambda row: (cell_rank[_placement_key(row, placement)], int(row["source_index"]))
    )
    for manifest_idx, row in enumerate(ordered):
        row["manifest_idx"] = manifest_idx
        row["order_policy"] = order_policy
    return ordered


def _validate_manifest_rows(
    rows: list[dict],
    *,
    cfg: FreshEvalManifestConfig,
    geom: TaskGeometry,
    spec: RealTaskSpec,
) -> None:
    if len(rows) != cfg.count:
        raise RuntimeError(f"manifest has {len(rows)} states, expected {cfg.count}")
    if [int(row["manifest_idx"]) for row in rows] != list(range(cfg.count)):
        raise RuntimeError("manifest_idx is not contiguous")

    states = geom.states_from_rows(rows)
    if not np.isfinite(states).all():
        raise RuntimeError("manifest contains non-finite state values")
    bounds = np.asarray(geom.bounds, dtype=float)
    if np.any(states < bounds[:, 0] - cfg.match_tolerance) or np.any(
        states > bounds[:, 1] + cfg.match_tolerance
    ):
        raise RuntimeError("manifest contains state outside task bounds")

    # Validate EVERY sampled placement (not only an ordering placement): its (x, y) lands on the
    # registry grid, and — when it carries a discrete orientation — the persisted index is in
    # range and its resolved angle agrees. This covers multi-placement tasks (routing's two
    # clips) that have no single serpentine-ordering placement; a no-op for placement-free tasks.
    for name, placement in spec.sampled_placements.items():
        for row in rows:
            x, y = float(row[placement.keys[0]]), float(row[placement.keys[1]])
            if not placement.is_on_grid(x, y):
                raise RuntimeError(
                    f"{name} ({x},{y}) is not an in-bounds grid point "
                    f"(bounds {placement.bounds}, snap {placement.snap_m})"
                )
            if placement.has_orient:
                oidx = int(round(float(row[placement.orient_key])))
                if not 0 <= oidx < placement.n_orient:
                    raise RuntimeError(f"{name} orientation index {oidx} out of range")
                angle = float(row[placement.orient_angle_key])
                if placement.index_for_angle(angle) != oidx:
                    raise RuntimeError(
                        f"{name} orientation index {oidx} disagrees with angle {angle}"
                    )
    if spec.placement_min_separation_m is not None:
        for row in rows:
            sep = spec.min_placement_separation(row)
            if sep < spec.placement_min_separation_m - cfg.match_tolerance:
                raise RuntimeError(
                    f"sampled placements only {sep} m apart, below the required "
                    f"{spec.placement_min_separation_m} m minimum"
                )


def build_fresh_eval_manifest(
    cfg: FreshEvalManifestConfig, *, apply_registry: bool = False
) -> dict:
    spec = get_task_spec(cfg.task)
    geom = TaskGeometry(spec)
    placement = spec.sampled_placements.get(cfg.placement_name) if cfg.placement_name else None
    if cfg.placement_name is not None and placement is None:
        raise RuntimeError(f"{cfg.task}: no sampled placement named {cfg.placement_name!r}")

    # Feasible draw: for a task with an inter-placement min-separation constraint (routing's two
    # clips) rejection sampling drops the physically-unplaceable draws and the kept Sobol stream
    # indices are non-contiguous; without the constraint this is the contiguous
    # start..start+count-1 block (as for pen / single-placement tasks). The robot loader enforces
    # the same constraint, so skipping this would let
    # an unlucky seed emit an eval manifest the robot then refuses to load.
    arr, stream_indices = geom.sobol_feasible(cfg.sobol_seed, cfg.count, start=cfg.start_index)
    stream = np.asarray(stream_indices, dtype=np.int64)
    collisions = _registry_collisions(cfg, set(map(int, stream)))
    if collisions:
        raise RuntimeError(f"Sobol block collides with registry claims: {collisions}")

    unit = geom.normalise(arr)
    within = _min_pairwise(unit, geom.periodic_mask)
    if within <= cfg.disjoint_tolerance:
        raise RuntimeError(f"within-set collision: min pairwise {within:.3g}")

    prior_distances: dict[str, float] = {}
    for manifest_path in cfg.audit_manifests:
        if not manifest_path.exists():
            raise RuntimeError(f"audit manifest missing: {manifest_path}")
        dist = _min_cross(
            unit,
            geom.normalise(_states_from_manifest(manifest_path, task=cfg.task, geom=geom)),
            geom.periodic_mask,
        )
        if dist <= cfg.disjoint_tolerance:
            raise RuntimeError(f"collision with {manifest_path}: min dist {dist:.3g}")
        prior_distances[_audit_manifest_label(manifest_path, cfg.audit_path_root)] = dist

    states: list[dict] = []
    for idx, row in enumerate(arr):
        state = {key: float(value) for key, value in zip(geom.keys, row, strict=True)}
        for key in spec.state_keys:
            if key.endswith("_yaw"):
                state[key] = float(wrap_angle(state[key]))
        # Resolve each sampled placement's discrete orientation INDEX (a joint-sampling column)
        # to its persisted ANGLE column (e.g. routing clip_*_oidx -> clip_*_yaw). No-op for pen /
        # single-placement tasks that carry no orientation.
        spec.resolve_placement_orientations(state)
        state.update(
            {
                "source": cfg.source,
                "sources": [cfg.source],
                "source_index": int(stream[idx]),
                "sobol_seed": cfg.sobol_seed,
                "sobol_stream_index": int(stream[idx]),
                "manifest_idx": idx,
                "order_policy": "source_index",
            }
        )
        states.append(state)

    source_order_changes = _placement_changes(states, placement)
    if placement is not None:
        states = _placement_batched_order(
            states,
            placement=placement,
            order_policy=cfg.order_policy,
            outer_axis=cfg.serpentine_outer_axis,
        )
    applied_corrections = _apply_manual_state_corrections(states, cfg.manual_state_corrections)
    placement_changes = _placement_changes(states, placement)
    manifest = {
        **_manifest_header(cfg, geom),
        "phase": cfg.phase,
        "source": cfg.source,
        "arms": [cfg.source],
        "arm_counts": {cfg.source: cfg.count},
        "seeds": {"sobol": cfg.sobol_seed},
        "order_policy": cfg.order_policy,
        "order_policy_description": cfg.order_policy_description,
        "source_index_order_placement_changes": source_order_changes,
        "placement_changes": placement_changes,
        "placement_moves_reduced_by": source_order_changes - placement_changes,
        **({"manual_state_corrections": applied_corrections} if applied_corrections else {}),
        "sobol_stream": {
            "start_index": cfg.start_index,
            "end_index_exclusive": cfg.start_index + cfg.count,
        },
        "registry_usage_id": cfg.usage_id,
        "disjoint_from": cfg.disjoint_from,
        "purpose": cfg.purpose,
        "within_set_min_periodic_dist": within,
        "cross_set_min_periodic_dist_to_prior_manifests": prior_distances,
        "states": states,
    }

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {cfg.output_path} ({len(states)} states)")
    print(
        f"  sobol seed {cfg.sobol_seed}, indices {cfg.start_index}..{cfg.start_index + cfg.count - 1}"
    )
    print(
        f"  placement changes {source_order_changes} -> {placement_changes} with {cfg.order_policy}"
    )
    print(f"  within-set min periodic dist = {within:.4f}")
    for path, dist in prior_distances.items():
        print(f"  min periodic dist to {path} = {dist:.4f}")

    _validate_manifest_rows(states, cfg=cfg, geom=geom, spec=spec)
    print("VALIDATED: contiguous manifest_idx, sampled placements grid-snapped, states in bounds")

    if apply_registry:
        _append_registry_row(cfg, stream)
    return manifest
