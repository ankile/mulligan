"""Manual initial-state manifests: the target model, per-task keys, loader, and formatting.

A real-world round's start states live in a locked JSON manifest (``task``, ``keys``,
``states`` rows, optional ``bounds`` / ``arms``). This module is the single reader for
that format across collection (``mulligan.real.collect.blind_dagger``), teleop, eval, the operator card
renderers (``mulligan.real.operator_ui.cards``), and the round-tracking scripts. It validates every
row against the task registry (``mulligan.real.lifecycle.tasks``) so a stale or hand-edited
manifest fails loud on load rather than misguiding the operator at the robot.

Deliberately importable without torch / lerobot so card previews and round-tracking scripts stay
cheap.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mulligan.real.lifecycle.tasks import INCH_TO_M, find_task_spec_by_task_name

M_TO_INCH = 1.0 / INCH_TO_M  # the registry owns INCH_TO_M; re-exported for scripts
MARKER_D2_INITIAL_STATE_KEYS = ["pen_x", "pen_y", "pen_yaw", "holder_x", "holder_y"]
NUT_PEG_INITIAL_STATE_KEYS = ["nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"]
ROUTING_D2_INITIAL_STATE_KEYS = [
    "rope_x",
    "clip_left_x",
    "clip_left_y",
    "clip_left_yaw",
    "clip_left_oidx",
    "clip_right_x",
    "clip_right_y",
    "clip_right_yaw",
    "clip_right_oidx",
]


@dataclass(frozen=True)
class ArmSpec:
    key: str
    model_id: str


@dataclass(frozen=True)
class InitialStateTarget:
    manifest_idx: int
    source: str
    source_index: int
    raw: dict
    # Pen pose is OPTIONAL: pen tasks (marker/square) set it; the non-pen routing_d2 line
    # has no free pose and leaves these None. Pen-frame card rendering reads these and so is
    # pen-task-only by construction; routing collection uses the rope/clip accessors below.
    pen_x: float | None = None
    pen_y: float | None = None
    pen_yaw: float | None = None
    peg_x: float | None = None
    peg_y: float | None = None

    @property
    def nut_x(self) -> float:
        return float(self.raw.get("nut_x", self.pen_x))

    @property
    def nut_y(self) -> float:
        return float(self.raw.get("nut_y", self.pen_y))

    @property
    def nut_yaw(self) -> float:
        return float(self.raw.get("nut_yaw", self.pen_yaw))

    @property
    def holder_x(self) -> float | None:
        """Chosen RED-holder x (marker_d2, snapped, operator frame m); None if absent."""
        value = self.raw.get("holder_x")
        return None if value is None else float(value)

    @property
    def holder_y(self) -> float | None:
        value = self.raw.get("holder_y")
        return None if value is None else float(value)

    # --- routing_d2 accessors (rope + two clips); None when absent ----------------
    @property
    def rope_x(self) -> float | None:
        """Rope x position (routing_d2, continuous, operator frame m); None if absent."""
        value = self.raw.get("rope_x")
        return None if value is None else float(value)

    def _clip_value(self, key: str) -> float | None:
        value = self.raw.get(key)
        return None if value is None else float(value)

    @property
    def clip_left_x(self) -> float | None:
        return self._clip_value("clip_left_x")

    @property
    def clip_left_y(self) -> float | None:
        return self._clip_value("clip_left_y")

    @property
    def clip_left_yaw(self) -> float | None:
        """Resolved left-clip angle (routing_d2, radians); None if absent."""
        return self._clip_value("clip_left_yaw")

    @property
    def clip_right_x(self) -> float | None:
        return self._clip_value("clip_right_x")

    @property
    def clip_right_y(self) -> float | None:
        return self._clip_value("clip_right_y")

    @property
    def clip_right_yaw(self) -> float | None:
        """Resolved right-clip angle (routing_d2, radians); None if absent."""
        return self._clip_value("clip_right_yaw")


# The three key sets map to different target constructions (collection, eval, operator cards),
# so this stays an explicit list rather than registry-driven.
def _manifest_keys(manifest_meta: dict) -> list[str]:
    keys = list(manifest_meta["keys"])
    if keys not in (
        MARKER_D2_INITIAL_STATE_KEYS,
        NUT_PEG_INITIAL_STATE_KEYS,
        ROUTING_D2_INITIAL_STATE_KEYS,
    ):
        raise ValueError(
            f"unsupported initial-state keys {keys!r}; expected "
            f"{MARKER_D2_INITIAL_STATE_KEYS!r}, {NUT_PEG_INITIAL_STATE_KEYS!r}, or "
            f"{ROUTING_D2_INITIAL_STATE_KEYS!r}"
        )
    return keys


def _validate_sampled_placements_against_registry(
    path: Path, row: dict, row_idx: int, task: str
) -> None:
    """Validate every sampled placement the registry defines for ``task`` against the row.

    Single source of truth: the allowed grid lives in the registry, so a stale or
    hand-edited manifest with an off-grid placement (e.g. a peg/holder the operator card
    would then misdraw) fails loudly here. No-op for tasks with no sampled placements
    so it is safe to call unconditionally.
    """
    spec = find_task_spec_by_task_name(task)
    if spec is None or not spec.sampled_placements:
        return
    for placement_name, placement in spec.sampled_placements.items():
        kx, ky = placement.keys
        hx, hy = float(row[kx]), float(row[ky])
        if not (math.isfinite(hx) and math.isfinite(hy)):
            raise ValueError(f"{path}: states[{row_idx}] non-finite {kx}/{ky}")
        if not placement.is_on_grid(hx, hy):
            raise ValueError(
                f"{path}: states[{row_idx}] {placement_name} ({hx:.5f},{hy:.5f}) is "
                f"not an in-bounds grid point for {task} "
                f"(bounds {placement.bounds}, snap {placement.snap_m})"
            )
        if placement.has_orient:
            # Both the resolved angle and the choice index are persisted; validate the angle
            # is one of the registry's allowed orientations (fail loud on a stale/off-set
            # angle the operator card would misdraw) AND that the index agrees with it.
            angle = float(row[placement.orient_angle_key])
            if not math.isfinite(angle):
                raise ValueError(
                    f"{path}: states[{row_idx}] non-finite {placement.orient_angle_key}"
                )
            idx = placement.index_for_angle(angle)
            stored_idx = int(round(float(row[placement.orient_key])))
            if stored_idx != idx:
                raise ValueError(
                    f"{path}: states[{row_idx}] {placement_name} index {stored_idx} "
                    f"({placement.orient_key}) disagrees with angle {angle:.5f} rad "
                    f"({placement.orient_angle_key}, expected index {idx}) for {task}"
                )
    # Inter-placement separation (e.g. routing clips' taps must not share holes): every pair
    # of placements must be >= placement_min_separation_m apart. Snapping clips to the dot
    # grid happens above; this is the pairwise constraint the builder rejection-samples for,
    # so a stale/hand-edited manifest with clips too close fails loudly here.
    if spec.placement_min_separation_m is not None:
        sep = spec.min_placement_separation(row)
        tol = 1e-9
        if sep < spec.placement_min_separation_m - tol:
            raise ValueError(
                f"{path}: states[{row_idx}] sampled placements are {sep * M_TO_INCH:.3f} in "
                f"apart, below the required {spec.placement_min_separation_m * M_TO_INCH:.3f} in "
                f"minimum for {task} (taps would share holes)"
            )


def _check_planar_pose_bounds(
    path: Path, row_idx: int, manifest_idx: int, task: str, spec, values: tuple[float, ...]
) -> None:
    """Range-check the free pose's (x, y) (``values[:2]``, named by ``spec.state_keys``)
    against the registry bounds. The yaw (row 2) is periodic, so it is not checked."""
    bounds = spec.bounds_arr  # (3, 2): rows x, y, yaw of the free pose
    tol = 1e-9
    for axis in (0, 1):
        key = spec.state_keys[axis]
        lo, hi = float(bounds[axis, 0]), float(bounds[axis, 1])
        val = values[axis]
        if not (lo - tol <= val <= hi + tol):
            raise ValueError(
                f"{path}: states[{row_idx}] (manifest_idx={manifest_idx}) "
                f"{key}={val:.5f} m is out of bounds for {task} "
                f"(allowed [{lo:.5f}, {hi:.5f}] m)"
            )


def _target_value(target: InitialStateTarget, key: str) -> float:
    if key in target.raw:
        return float(target.raw[key])
    if key == "pen_x":
        return target.pen_x
    if key == "pen_y":
        return target.pen_y
    if key == "pen_yaw":
        return target.pen_yaw
    if key == "nut_x":
        return target.nut_x
    if key == "nut_y":
        return target.nut_y
    if key == "nut_yaw":
        return target.nut_yaw
    if key == "peg_x" and target.peg_x is not None:
        return target.peg_x
    if key == "peg_y" and target.peg_y is not None:
        return target.peg_y
    raise KeyError(f"target #{target.manifest_idx} has no value for {key!r}")


def _load_initial_state_manifest(
    path: Path,
    arms: list[ArmSpec],
    *,
    expected_task: str | set[str] | None = None,
) -> tuple[list[InitialStateTarget], dict]:
    payload = _load_manifest_payload(path)
    task = payload["task"]
    if expected_task is not None:
        expected_tasks = {expected_task} if isinstance(expected_task, str) else set(expected_task)
        if task not in expected_tasks:
            raise ValueError(f"{path}: expected task in {sorted(expected_tasks)!r}, got {task!r}")
    keys = _manifest_keys(payload)

    arm_keys = {arm.key for arm in arms}
    states = payload["states"]
    if not isinstance(states, list) or not states:
        raise ValueError(f"{path}: manifest must contain a non-empty states list")

    targets: list[InitialStateTarget] = []
    seen_idxs: set[int] = set()
    for row_idx, row in enumerate(states):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: states[{row_idx}] must be an object")
        if "manifest_idx" not in row:
            raise ValueError(
                f"{path}: states[{row_idx}] is missing the required provenance key "
                "'manifest_idx' (do not fabricate it from row order)"
            )
        manifest_idx = int(row["manifest_idx"])
        if manifest_idx in seen_idxs:
            raise ValueError(f"{path}: duplicate manifest_idx {manifest_idx}")
        seen_idxs.add(manifest_idx)
        if manifest_idx != row_idx:
            raise ValueError(
                f"{path}: states must be stored in manifest_idx order; "
                f"row {row_idx} has manifest_idx={manifest_idx}"
            )
        if "source" in row:
            source = str(row["source"])
        elif "sources" in row:
            sources = list(row["sources"])
            if len(sources) != 1:
                raise ValueError(
                    f"{path}: states[{row_idx}] must have exactly one source for "
                    f"real blinded collection, got {sources!r}"
                )
            source = str(sources[0])
        else:
            raise ValueError(f"{path}: states[{row_idx}] must contain source or sources")
        if "sources" in row and "source" in row:
            sources = list(row["sources"])
            if sources != [source]:
                raise ValueError(
                    f"{path}: states[{row_idx}] has source={source!r} but sources={sources!r}"
                )
        if source not in arm_keys:
            raise ValueError(
                f"{path}: manifest source {source!r} has no matching --arm. "
                f"Known arms: {sorted(arm_keys)}"
            )
        if keys == MARKER_D2_INITIAL_STATE_KEYS:
            values = (float(row["pen_x"]), float(row["pen_y"]), float(row["pen_yaw"]))
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: states[{row_idx}] contains non-finite target values")
            # Validate every sampled placement's coords against the REGISTRY grid
            # (single source of truth): finite + in-bounds + on the snap grid. The
            # operator card draws the holder from this same registry, so a stale or
            # hand-edited manifest with an off-grid holder fails loudly here rather
            # than silently misguiding placement.
            #
            # The manifest carries holder keys, so the resolved spec MUST define the
            # matching sampled placements or there is NO grid to validate against --
            # an unregistered task (e.g. a future marker_d2_r1) would otherwise load
            # the holder with zero validation and silently misguide placement. Fail
            # loudly instead.
            spec = find_task_spec_by_task_name(task)
            if spec is None or not spec.sampled_placements:
                raise ValueError(
                    f"{path}: manifest carries holder keys {MARKER_D2_INITIAL_STATE_KEYS!r} "
                    f"but task {task!r} resolves to "
                    f"{'no registered task spec' if spec is None else 'a spec with no sampled placements'}"
                    f" -- cannot validate the holder against a registry grid. Register the "
                    f"task (with its sampled placements) before collecting."
                )
            # Validate the pen (x, y) against the REGISTRY bounds (rows 0/1 of
            # bounds_arr, in meters). Finiteness is checked above; this catches a
            # stale/hand-edited manifest with an out-of-bounds pen (e.g. pen_x=99 m)
            # that the holder grid validation would not. pen_yaw (row 2) is periodic
            # by design, so it is intentionally NOT range-checked here.
            _check_planar_pose_bounds(path, row_idx, manifest_idx, task, spec, values)
            _validate_sampled_placements_against_registry(path, row, row_idx, task)
            target = InitialStateTarget(
                manifest_idx=manifest_idx,
                source=source,
                source_index=int(row["source_index"]),
                pen_x=values[0],
                pen_y=values[1],
                pen_yaw=values[2],
                raw=dict(row),
            )
        elif keys == ROUTING_D2_INITIAL_STATE_KEYS:
            # routing_d2: no free pose. Validate the rope's continuous x against the registry
            # bounds and both clips (x/y on grid + orientation angle/index) against the
            # registry. The manifest MUST resolve to the registered routing_d2 spec or there
            # is no grid/orientation set to validate against -- fail loudly rather than load
            # a clip with zero validation and misguide the operator.
            spec = find_task_spec_by_task_name(task)
            if spec is None or not spec.sampled_placements:
                raise ValueError(
                    f"{path}: manifest carries routing keys {ROUTING_D2_INITIAL_STATE_KEYS!r} "
                    f"but task {task!r} resolves to "
                    f"{'no registered task spec' if spec is None else 'a spec with no sampled placements'}"
                    f" -- cannot validate the clips against a registry grid. Register the task "
                    f"(with its sampled placements) before collecting."
                )
            rope_x = float(row["rope_x"])
            if not math.isfinite(rope_x):
                raise ValueError(f"{path}: states[{row_idx}] non-finite rope_x")
            rope_bounds = spec.bounds_arr  # (1, 2): row rope_x
            tol = 1e-9
            lo, hi = float(rope_bounds[0, 0]), float(rope_bounds[0, 1])
            if not (lo - tol <= rope_x <= hi + tol):
                raise ValueError(
                    f"{path}: states[{row_idx}] (manifest_idx={manifest_idx}) rope_x={rope_x:.5f} m "
                    f"is out of bounds for {task} (allowed [{lo:.5f}, {hi:.5f}] m)"
                )
            _validate_sampled_placements_against_registry(path, row, row_idx, task)
            target = InitialStateTarget(
                manifest_idx=manifest_idx,
                source=source,
                source_index=int(row["source_index"]),
                raw=dict(row),
            )
        else:
            values = (
                float(row["nut_x"]),
                float(row["nut_y"]),
                float(row["nut_yaw"]),
                float(row["peg_x"]),
                float(row["peg_y"]),
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: states[{row_idx}] contains non-finite target values")
            # square_d2: the nut (x, y) gets the same registry range check as the pen.
            spec = find_task_spec_by_task_name(task)
            if spec is not None:
                _check_planar_pose_bounds(path, row_idx, manifest_idx, task, spec, values)
            # square_d2 samples the peg over a registry grid (the earlier fixed-peg Nut
            # setup had no sampled peg, so this no-ops); validate the chosen peg lands on
            # that grid before the operator card draws it.
            _validate_sampled_placements_against_registry(path, row, row_idx, task)
            target = InitialStateTarget(
                manifest_idx=manifest_idx,
                source=source,
                source_index=int(row["source_index"]),
                pen_x=values[0],
                pen_y=values[1],
                pen_yaw=values[2],
                raw=dict(row),
                peg_x=values[3],
                peg_y=values[4],
            )
        targets.append(target)
    return targets, payload


def _load_manifest_payload(path: Path) -> dict:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return payload


def _format_initial_state_target(
    target: InitialStateTarget,
    *,
    display_label: str | None = None,
) -> str:
    label = display_label or f"target #{target.manifest_idx + 1}"
    if "rope_x" in target.raw:
        return (
            f"{label}: "
            f"rope x={target.rope_x * M_TO_INCH:+.1f} in forward (spans full width) | "
            f"clip_L (x={target.clip_left_x * M_TO_INCH:+.1f}, "
            f"y={target.clip_left_y * M_TO_INCH:+.1f} in, "
            f"yaw={math.degrees(target.clip_left_yaw):+.0f} deg) | "
            f"clip_R (x={target.clip_right_x * M_TO_INCH:+.1f}, "
            f"y={target.clip_right_y * M_TO_INCH:+.1f} in, "
            f"yaw={math.degrees(target.clip_right_yaw):+.0f} deg)"
        )
    if "nut_x" in target.raw:
        yaw_deg = math.degrees(target.nut_yaw)
        if target.peg_x is None or target.peg_y is None:
            raise ValueError(f"Nut target #{target.manifest_idx} is missing peg_x/peg_y")
        return (
            f"{label}: "
            f"nut x={target.nut_x * M_TO_INCH:+.1f} in forward, "
            f"nut y={target.nut_y * M_TO_INCH:+.1f} in left, "
            f"nut yaw={yaw_deg:+.0f} deg, "
            f"peg x={target.peg_x * M_TO_INCH:+.1f} in forward, "
            f"peg y={target.peg_y * M_TO_INCH:+.1f} in left"
        )
    yaw_deg = math.degrees(target.pen_yaw)
    line = (
        f"{label}: "
        f"x={target.pen_x * M_TO_INCH:+.1f} in forward, "
        f"y={target.pen_y * M_TO_INCH:+.1f} in left, "
        f"yaw={yaw_deg:+.0f} deg"
    )
    if target.holder_x is not None and target.holder_y is not None:
        line += (
            f" | HOLDER (x={target.holder_x * M_TO_INCH:+.1f} in forward, "
            f"y={target.holder_y * M_TO_INCH:+.1f} in left)"
        )
    return line


def _initial_state_setup_subject(target: InitialStateTarget, task_name: str) -> str:
    """Human-readable physical objects the operator must place for a manifest target."""
    if "rope_x" in target.raw or task_name.lower().startswith("routing"):
        return "rope and two clips"
    if "nut_x" in target.raw or task_name.lower().startswith("square"):
        return "nut and square peg"
    if target.holder_x is not None and target.holder_y is not None:
        return "marker and holder"
    return "marker"


def manifest_arm_keys(payload: dict) -> list[str]:
    """The arm whitelist a manifest implies: its declared ``arms``, else its row sources.

    Consumers without an ``--arm`` flag (teleop, eval, card previews) still need the
    loader's ``source in arm_keys`` consistency check, so they derive the whitelist from
    the manifest itself: the top-level ``arms`` list when present, else the union of every
    row's ``source`` / ``sources`` in first-seen order.
    """
    declared = payload.get("arms")
    if declared:
        return [str(arm) for arm in declared]
    states = payload.get("states")
    if not isinstance(states, list):
        raise ValueError("manifest must contain a states list to derive arms")
    seen: dict[str, None] = {}
    for row in states:
        if not isinstance(row, dict):
            continue
        if "source" in row:
            seen.setdefault(str(row["source"]), None)
        elif "sources" in row:
            for source in row["sources"]:
                seen.setdefault(str(source), None)
    return list(seen)


def load_manifest_targets(
    path: Path,
    *,
    expected_task: str | set[str] | None,
    model_id: Callable[[str], str],
) -> tuple[list[InitialStateTarget], dict]:
    """Load a manifest whose arm whitelist comes from the manifest itself.

    For consumers without an ``--arm`` flag (teleop, eval, card previews): the arms are
    :func:`manifest_arm_keys` and ``model_id(key)`` supplies each arm's placeholder model
    id (the loader only uses arms for its ``source in arm_keys`` consistency check).
    """
    payload = _load_manifest_payload(path)
    arms = [ArmSpec(key=key, model_id=model_id(key)) for key in manifest_arm_keys(payload)]
    return _load_initial_state_manifest(path, arms, expected_task=expected_task)
