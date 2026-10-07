"""Task registry for real-world lifecycle tooling.

A :class:`RealTaskSpec` captures everything task-specific that the shared
geometry/ingest/plot code needs: the 3-DOF pen pose keys (x, y, yaw — yaw is
always the last, periodic column), physical bounds in the operator frame
(meters/radians; +x forward, +y left), fixed scene objects, optional
grid-snapped sampled scene placements (e.g. a holder), and the discretization
used for coverage metrics. Round-evolving choices (feature libraries, gates,
promotion ordering) live in the per-round start-design configs, not here.

The **joint sampling space** is the 3-DOF pen plus the 2-DOF (x, y) of each
:class:`GridSampledPlacement`. Sobol/FPS sample this joint space in one shot
(the placement dims live in the SAME operator frame + metric as the pen), then
the placement dims are snapped to their grid on output for the operator. A task
with no placements has a sampling space identical to its 3-DOF pen, so the
shared geometry reduces to the plain pen geometry for such tasks.

Adding a new task = adding one ``RealTaskSpec`` entry here. The values must
match the task's collection manifests exactly; the registry is the single
source of truth for analysis-side geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

INCH_TO_M = 0.0254


@dataclass(frozen=True)
class GridSampledPlacement:
    """A scene object whose (x, y) is sampled JOINTLY with the pen (same operator
    frame + metric), then SNAPPED to a regular grid for the operator to place by hand.

    It is part of the task's joint Sobol/FPS sampling space (so the samplers cover and
    diversify it like any spatial dim), but it is NOT part of the robot pose
    (:attr:`RealTaskSpec.state_keys`). The grid is purely an operator-convenience
    quantization applied after sampling; the sampler sees the continuous box.
    """

    keys: tuple[str, str]
    """Manifest columns for the (x, y) of this placement, e.g. ``("holder_x", "holder_y")``."""

    bounds: tuple[tuple[float, float], tuple[float, float]]
    """((x_lo, x_hi), (y_lo, y_hi)) randomization box in the operator frame (meters)."""

    snap_m: float
    """Grid snap resolution in meters (e.g. ``INCH_TO_M`` for a 1-inch grid)."""

    orient_key: str | None = None
    """Optional SAMPLING/manifest column for a DISCRETE orientation, stored as an integer
    CHOICE INDEX in ``{0, .., n_orient-1}`` (e.g. ``"clip_left_oidx"``). When set, the
    placement contributes a 3rd joint-sampling dim: bounds ``[0, n_orient-1]``, snap 1,
    non-periodic, ``n_orient`` coverage cells. The index — not the angle — lives in the
    sampling space so the regular-grid snap + coverage machinery is reused unchanged even
    though the angle set is an irregular lattice. ``None`` ⇒ no orientation (x,y only,
    as for the holder/peg placements)."""

    orient_angle_key: str | None = None
    """Optional OPERATOR-FACING manifest column holding the RESOLVED angle in radians
    (e.g. ``"clip_left_yaw"``), written by the manifest builder via
    :meth:`angle_for_index`. It is NOT a sampling dim (the index ``orient_key`` is);
    persisting both keeps the angle available for operator cards/overlays while the index
    drives snap/coverage. Required iff ``orient_key`` is set."""

    orient_angles: tuple[float, ...] = ()
    """index → angle (radians) map for the discrete orientation, e.g.
    ``(-math.pi/4, math.pi/4, math.pi/2)``. Length ``n_orient``; required iff
    ``orient_key`` is set."""

    @property
    def has_orient(self) -> bool:
        """True iff this placement carries a discrete orientation index dim."""
        return self.orient_key is not None

    @property
    def n_orient(self) -> int:
        """Number of discrete orientation choices (0 if no orientation)."""
        return len(self.orient_angles)

    def angle_for_index(self, idx: int) -> float:
        """Resolve an orientation choice index to its angle (radians)."""
        return float(self.orient_angles[int(idx)])

    def index_for_angle(self, angle: float, *, tol: float = 1e-6) -> int:
        """Inverse of :meth:`angle_for_index`: nearest allowed orientation index, or raise
        if ``angle`` matches no allowed angle within ``tol`` (fail loud — a stale/hand-edited
        manifest angle off the registry set must not silently snap to the closest class)."""
        diffs = [abs(float(angle) - a) for a in self.orient_angles]
        best = int(np.argmin(diffs))
        if diffs[best] > tol:
            raise ValueError(
                f"angle {angle!r} rad is not within {tol} of any allowed orientation "
                f"{self.orient_angles!r}"
            )
        return best

    @property
    def sampling_keys(self) -> tuple[str, ...]:
        """Joint-sampling-space columns this placement contributes: (x, y[, orient index])."""
        return (*self.keys, *((self.orient_key,) if self.has_orient else ()))

    @property
    def manifest_keys(self) -> tuple[str, ...]:
        """All persisted manifest columns: (x, y[, orient ANGLE, orient INDEX])."""
        if self.has_orient:
            return (*self.keys, self.orient_angle_key, self.orient_key)
        return self.keys

    def sampling_bounds_rows(self) -> list[list[float]]:
        """Per-dim physical bounds rows for the joint sampling space (x, y[, index])."""
        rows = [list(self.bounds[0]), list(self.bounds[1])]
        if self.has_orient:
            rows.append([0.0, float(self.n_orient - 1)])
        return rows

    def sampling_snaps(self) -> list[float]:
        """Snap per joint-sampling dim: (snap_m, snap_m[, 1.0] for the index)."""
        snaps = [self.snap_m, self.snap_m]
        if self.has_orient:
            snaps.append(1.0)
        return snaps

    def sampling_periodic(self) -> list[bool]:
        """Periodic flags per joint-sampling dim (x, y, and the index are all non-periodic)."""
        flags = [False, False]
        if self.has_orient:
            flags.append(False)
        return flags

    @property
    def sampling_grid_dims(self) -> tuple[int, ...]:
        """Coverage cells per joint-sampling dim: ((nx, ny)[, n_orient])."""
        if self.has_orient:
            return (*self.grid_dims, self.n_orient)
        return self.grid_dims

    def axis_points(self, axis: int) -> np.ndarray:
        """The allowed (snapped) coordinates along ``axis`` (0=x, 1=y), lo..hi inclusive."""
        lo, hi = self.bounds[axis]
        n = int(round((hi - lo) / self.snap_m))
        return lo + np.arange(n + 1) * self.snap_m

    @property
    def grid_dims(self) -> tuple[int, int]:
        """Number of snap points per axis (the coverage cells along x, y)."""
        return (len(self.axis_points(0)), len(self.axis_points(1)))

    def grid_points(self) -> np.ndarray:
        """(M, 2) array of every allowed snapped (x, y) point (operator frame, meters)."""
        gx, gy = np.meshgrid(self.axis_points(0), self.axis_points(1), indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel()])

    def snap(self, x: float, y: float) -> tuple[float, float]:
        """Snap a continuous (x, y) to the nearest grid point, clipped to bounds."""
        out = []
        for axis, value in enumerate((x, y)):
            lo, hi = self.bounds[axis]
            snapped = lo + round((value - lo) / self.snap_m) * self.snap_m
            out.append(float(min(max(snapped, lo), hi)))
        return out[0], out[1]

    def is_on_grid(self, x: float, y: float, *, tol: float = 1e-6) -> bool:
        """True iff (x, y) is in-bounds AND lands on a snap grid point (within ``tol``).

        Snapping clips to bounds, so an out-of-bounds value snaps to a different point
        and fails this check too -- one test covers both in-bounds and on-grid.
        """
        sx, sy = self.snap(x, y)
        return abs(sx - x) <= tol and abs(sy - y) <= tol


@dataclass(frozen=True)
class DPTrainingRecipe:
    """Default DiffusionPolicy training recipe for a real task line.

    The DP trainer (``mulligan.real.train.policy``) resolves every training arg as:
    explicit CLI flag > this recipe (when ``--task`` is given) > built-in fallback, so a
    launch wrapper can pass just ``--task`` plus the data/deviation it is actually testing
    and inherit the rest. Only the fields whose generic argparse default differs from the
    real-DP recipe live here; fields already correct at the argparse level (``--vision-backbone
    resnet18``, ``--visual-normalization auto``; the launcher passes
    ``--policy diffusion``) are not restated. Per-robot-station capture constants (image
    resize target, video backend, camera roles + default crops) live in
    :mod:`mulligan.real.robot.cameras`, not here. Active final-line tasks override these
    defaults where their locked recipe differs.
    """

    chunk_size: int = 12
    """DP prediction horizon, i.e. frames the model predicts."""

    n_action_steps: int = 6
    """DP execution horizon, i.e. frames executed per inference (predict 12 / execute 6)."""

    down_dims: tuple[int, ...] = (512, 1024)
    """UNet channel widths (~60M params; the generic lerobot default (512,1024,2048) is ~266M)."""

    batch_size: int = 64
    training_steps: int = 50_000
    save_freq: int = 25_000

    eval_freq: int = 10_000
    """Steps between held-out val-loss + action-reconstruction evaluations."""

    drop_n_last_frames: int = 2
    """Anchor frames dropped per episode END (2 rather than the generic config
    default 7). Late-episode anchors whose action chunk overflows the boundary are PADDED and
    the pad is masked out of the loss, so 2 keeps the most late-phase data without bias."""


@dataclass(frozen=True)
class RealTaskSpec:
    """Physical/task constants shared by all rounds of one real task line."""

    name: str
    """Lifecycle key, e.g. ``"square_d2"`` (lowercase)."""

    task_name: str
    """Collection/LeRobot task name, e.g. ``"square_d2"``."""

    state_keys: tuple[str, ...]
    """Free continuous manifest keys. The default convention is the 3-DOF pen
    ``(x, y, yaw)`` with yaw the trailing periodic dim; a non-pen task (e.g. routing's
    1-DOF ``("rope_x",)``) passes ``state_periodic_mask`` explicitly."""

    bounds: tuple[tuple[float, float], ...]
    """Physical randomization bounds per free dim (m, m, rad for the 3-DOF pen)."""

    state_periodic_mask: tuple[bool, ...] | None = None
    """Optional per-free-dim periodic mask. ``None`` ⇒ the 3-DOF pen convention
    ``(False, False, True)`` (and ``_register`` then requires a trailing ``*_yaw`` dim).
    A non-pen task supplies its own mask, e.g.
    ``(False,)`` for routing's single continuous ``rope_x``."""

    fixed_objects: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Fixed scene objects (e.g. ``{"peg": (x, y)}``), operator frame meters."""

    fixed_object_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    """Manifest keys per fixed object (e.g. ``{"peg": ("peg_x", "peg_y")}``)."""

    sampled_placements: dict[str, GridSampledPlacement] = field(default_factory=dict)
    """Grid-snapped scene placements sampled JOINTLY with the pen, e.g. a holder
    randomized over a box and snapped to a 1-inch grid. Each contributes its 2 (x, y)
    dims to the joint sampling space; its manifest columns carry the chosen snapped
    coordinates. Empty for tasks whose only randomized object is the pen."""

    placement_min_separation_m: float | None = None
    """Optional minimum center-to-center distance (meters) required between EVERY PAIR of
    distinct sampled placements — e.g. routing clips whose mounting taps occupy the holes
    around each clip, so two clips closer than this would fight for the same holes. The
    manifest builder enforces it by REJECTION SAMPLING and the collection loader validates
    it (fail loud). ``None`` ⇒ no inter-placement constraint (a single placement, like the
    marker holder / square peg, has no pairs to separate)."""

    grid_dims: tuple[int, ...] = (4, 5, 4)
    """Coverage-grid cells over the FREE dims (pen x, y, yaw); 4x5x4 = 80 cells by
    convention. Length must equal ``len(state_keys)`` (a 1-tuple for a 1-DOF free task)."""

    max_steps: int = 400
    """Episode step cap at 15 Hz."""

    num_subtask_marks: int = 0
    """Number of mid-episode subtask reward marks the outcome editor requires per
    changed episode for this task (single source of truth so the operator never has
    to pass ``--subtask-marks`` by hand). ``0`` (default) means the task has no
    intermediate sub-goal — the editor's single terminal-outcome behavior.
    ``> 0`` means every changed episode must carry exactly this many single-frame
    ``reward=1.0`` sub-goal spikes strictly before the terminal outcome frame. Routing seats the
    rope in two clips: the FIRST seat is one
    mid-episode mark (=1); the second seat IS the terminal success, so ``num_subtask_marks=1``.
    Downstream ``mulligan.tools.outcome_review`` resolves this from the dataset's task name."""

    consumed_camera_roles: tuple[str, ...] = ()
    """Default camera ROLE names a policy CONSUMES for this task, in order (e.g.
    ``("side_1", "wrist_left")``). When set, the DP trainer uses these as the default
    ``--camera-keys`` if none is passed explicitly (explicit --camera-keys overrides).
    Empty for tasks with no task-default camera set (the prior suffix-filter behavior)."""

    camera_crop_overrides: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    """Per-task REPLACEMENT crop boxes overriding STATION_CAMERA_DEFAULT_CROPS, keyed by
    camera ROLE, in STORED-frame (640x480) px ``(x0, y0, x1, y1)`` half-open — same
    convention as the station defaults. Use when a task's content envelope justifies a
    tighter ROI than the station-wide default AND the role is shared with other task
    lines (a per-task override never perturbs the other lines' crops mid-lifecycle).
    The DP trainer merges these over the station defaults when ``--task`` is given;
    explicit ``--side-crop`` still wins. The box rides in the checkpoint's
    ``camera_crop_boxes`` as usual, so eval needs no task lookup."""

    training: DPTrainingRecipe = field(default_factory=DPTrainingRecipe)
    """Default DP training recipe for this task line (chunk/exec horizon, batch, steps,
    eval cadence, drop_n_last). The DP trainer fills any training arg a ``--task`` run
    leaves unset from here (explicit CLI flags still win)."""

    r0_teleop_demo_repos: dict[str, tuple[str, ...]] | None = None
    """R0 TELEOP demo split repos per demo-set, keyed by demo-set name (a
    ``mulligan.plotting.colors.METHOD_COLORS`` key, e.g. ``"uniform"`` for the baseline
    family and ``"sobol"`` for the Ours family). Consumed ONLY by
    :mod:`mulligan.real.lifecycle.episode_lengths` to draw each family's teleop
    demo-length MEDIAN reference line on the per-round episode-length plot.

    These are the R0 TELEOP demo splits ONLY (``*-c00-teleop-*``). The R1+ DAgger correction
    episodes are NOT teleop demos -- they contain policy-driven frames, so their
    episode length is not a human-demo length. Including them here would bias the
    "human demo length" reference upward; excluding them was a user-caught bug fix,
    so the exclusion is deliberate and load-bearing -- keep this field pointed at
    ``*-c00-teleop-*`` teleop splits only.

    ``None`` for a line without verified R0 teleop splits: the episode-length plot then SKIPS the median reference lines with a
    loud log message rather than guessing a repo id."""

    @property
    def bounds_arr(self) -> np.ndarray:
        return np.asarray(self.bounds, dtype=np.float64)

    @property
    def free_periodic_mask(self) -> np.ndarray:
        """(len(state_keys),) bool periodic mask over the free dims. Resolves the optional
        ``state_periodic_mask``, defaulting to the 3-DOF ``(F, F, T)`` pen."""
        if self.state_periodic_mask is not None:
            return np.asarray(self.state_periodic_mask, dtype=bool)
        return np.asarray([False, False, True], dtype=bool)

    @property
    def manifest_keys(self) -> tuple[str, ...]:
        """All persisted manifest columns: free dims, then fixed-object position keys, then
        each sampled placement's columns (x, y[, orient angle, orient index]).
        split_protocol pins the manifest to this exact tuple."""
        fixed = tuple(k for pair in self.fixed_object_keys.values() for k in pair)
        placement = tuple(k for p in self.sampled_placements.values() for k in p.manifest_keys)
        return (*self.state_keys, *fixed, *placement)

    # --- Joint sampling space (pen + sampled placements) ---------------------
    # For a task with no placements these all reduce to the 3-DOF pen, so the shared
    # geometry is the plain pen geometry.

    @property
    def sampling_keys(self) -> tuple[str, ...]:
        """Ordered keys of the joint sampling space (free dims + each placement's sampling
        dims). Placement orientation contributes its INDEX column, not the angle column."""
        placement = tuple(k for p in self.sampled_placements.values() for k in p.sampling_keys)
        return (*self.state_keys, *placement)

    @property
    def sampling_bounds(self) -> np.ndarray:
        """(K, 2) physical bounds for the joint sampling space."""
        rows = [list(b) for b in self.bounds]
        for p in self.sampled_placements.values():
            rows.extend(p.sampling_bounds_rows())
        return np.asarray(rows, dtype=np.float64)

    @property
    def sampling_periodic_mask(self) -> np.ndarray:
        """(K,) bool mask of periodic sampling dims (free-dim mask + placement dims, all of
        which — x, y, orient index — are non-periodic)."""
        mask = list(self.free_periodic_mask)
        for p in self.sampled_placements.values():
            mask.extend(p.sampling_periodic())
        return np.asarray(mask, dtype=bool)

    @property
    def sampling_snap(self) -> np.ndarray:
        """(K,) snap resolution per sampling dim; ``np.nan`` for continuous (free) dims."""
        snap = [np.nan] * len(self.state_keys)
        for p in self.sampled_placements.values():
            snap.extend(p.sampling_snaps())
        return np.asarray(snap, dtype=np.float64)

    @property
    def sampling_grid_dims(self) -> tuple[int, ...]:
        """(K,) coverage cells per sampling dim (free grid_dims + each placement's grid)."""
        dims = list(self.grid_dims)
        for p in self.sampled_placements.values():
            dims.extend(p.sampling_grid_dims)
        return tuple(dims)

    def resolve_placement_orientations(self, row: dict) -> None:
        """In-place: for every sampled placement carrying a discrete orientation, round its
        SAMPLED index column (``orient_key``) to an exact int and write the resolved ANGLE column
        (``orient_angle_key``) via :meth:`GridSampledPlacement.angle_for_index`. Shared by the
        collection and held-out-eval manifest builders so the persisted index and angle can never
        disagree (loaders cross-check them). A no-op for tasks whose placements have no orientation
        (or no placements at all), so pen / single-placement tasks are unaffected."""
        for placement in self.sampled_placements.values():
            if placement.has_orient:
                idx = int(round(float(row[placement.orient_key])))
                row[placement.orient_key] = float(idx)
                row[placement.orient_angle_key] = placement.angle_for_index(idx)

    def min_placement_separation(self, row: dict) -> float:
        """Minimum center-to-center distance (meters) over EVERY PAIR of sampled placements,
        reading each placement's (x, y) from ``row`` by its manifest keys. ``math.inf`` when
        there are fewer than two placements (no pair to separate). Single source of truth for
        the ``placement_min_separation_m`` constraint, shared by the manifest builder
        (rejection sampling) and the collection loader (validation)."""
        pts = [
            (float(row[p.keys[0]]), float(row[p.keys[1]])) for p in self.sampled_placements.values()
        ]
        best = math.inf
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                best = min(best, math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
        return best


_TASK_SPECS: dict[str, RealTaskSpec] = {}

# Released R0 teleop splits are named ``mulligan/real-<task>-c00-teleop-<set>``.
_R0_TELEOP_REPO_MARKER = "-c00-teleop-"


def _register(spec: RealTaskSpec) -> RealTaskSpec:
    if spec.name in _TASK_SPECS:
        raise ValueError(f"duplicate RealTaskSpec name {spec.name!r}")
    n_free = len(spec.state_keys)
    if n_free < 1 or len(spec.bounds) != n_free:
        raise ValueError(
            f"{spec.name}: state_keys/bounds must have matching length >= 1, got "
            f"{n_free}/{len(spec.bounds)}"
        )
    # Free-dim periodic mask: the 3-DOF pen convention is assumed only when no
    # explicit state_periodic_mask is given, and then the trailing dim must be the *_yaw
    # periodic dim. A non-pen task (e.g. routing's 1-DOF rope_x) supplies its own mask.
    if spec.state_periodic_mask is None:
        if n_free != 3 or not spec.state_keys[2].endswith("_yaw"):
            raise ValueError(
                f"{spec.name}: without an explicit state_periodic_mask the free dims must be "
                f"the 3-DOF pen (x, y, *_yaw); got {spec.state_keys}"
            )
    elif len(spec.state_periodic_mask) != n_free:
        raise ValueError(
            f"{spec.name}: state_periodic_mask length {len(spec.state_periodic_mask)} "
            f"!= len(state_keys) {n_free}"
        )
    # grid_dims must be a valid per-free-dim coverage grid: one positive cell count per free
    # dim whose product is >= 2. A product of 1 (e.g. (1,)) makes cell_entropy divide by
    # math.log(1) == 0 -> nan; reject it here rather than emit a silent nan downstream.
    if len(spec.grid_dims) != n_free:
        raise ValueError(
            f"{spec.name}: grid_dims must have one cell count per free dim ({n_free}), "
            f"got {spec.grid_dims}"
        )
    if not all(int(d) >= 1 for d in spec.grid_dims):
        raise ValueError(f"{spec.name}: grid_dims cells must be >= 1, got {spec.grid_dims}")
    if int(np.prod(spec.grid_dims)) < 2:
        raise ValueError(
            f"{spec.name}: grid_dims must have >= 2 total cells (product), got {spec.grid_dims}"
        )
    if spec.num_subtask_marks < 0:
        raise ValueError(
            f"{spec.name}: num_subtask_marks must be >= 0, got {spec.num_subtask_marks}"
        )
    if set(spec.fixed_objects) != set(spec.fixed_object_keys):
        raise ValueError(
            f"{spec.name}: fixed_objects and fixed_object_keys must describe the same objects"
        )
    for obj, xy in spec.fixed_objects.items():
        if len(xy) != 2 or not all(math.isfinite(float(v)) for v in xy):
            raise ValueError(f"{spec.name}: fixed object {obj!r} position must be a finite (x, y)")
    for obj, placement in spec.sampled_placements.items():
        if len(placement.keys) != 2 or placement.keys[0] == placement.keys[1]:
            raise ValueError(f"{spec.name}: placement {obj!r} needs two distinct manifest keys")
        if not (math.isfinite(placement.snap_m) and placement.snap_m > 0):
            raise ValueError(f"{spec.name}: placement {obj!r} snap_m must be finite and > 0")
        for axis, (lo, hi) in enumerate(placement.bounds):
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError(
                    f"{spec.name}: placement {obj!r} axis {axis} bounds must be lo < hi"
                )
            steps = (hi - lo) / placement.snap_m
            if abs(steps - round(steps)) > 1e-6 or round(steps) < 1:
                raise ValueError(
                    f"{spec.name}: placement {obj!r} axis {axis} range must be a positive "
                    f"integer multiple of snap_m (got {steps} steps)"
                )
        # Discrete orientation (optional): orient_key, orient_angle_key, and orient_angles
        # must all be set together, the angle set must have >= 2 distinct finite values, and
        # the three orientation columns must be named distinctly.
        has_any_orient = (
            placement.orient_key is not None
            or placement.orient_angle_key is not None
            or placement.orient_angles
        )
        if has_any_orient:
            if placement.orient_key is None or placement.orient_angle_key is None:
                raise ValueError(
                    f"{spec.name}: placement {obj!r} orientation needs both orient_key and "
                    f"orient_angle_key set"
                )
            if placement.orient_key == placement.orient_angle_key:
                raise ValueError(
                    f"{spec.name}: placement {obj!r} orient_key and orient_angle_key must differ"
                )
            angles = [float(a) for a in placement.orient_angles]
            if len(angles) < 2:
                raise ValueError(
                    f"{spec.name}: placement {obj!r} orient_angles needs >= 2 angles, got {angles}"
                )
            if not all(math.isfinite(a) for a in angles):
                raise ValueError(f"{spec.name}: placement {obj!r} orient_angles must be finite")
            if len({round(a, 9) for a in angles}) != len(angles):
                raise ValueError(
                    f"{spec.name}: placement {obj!r} orient_angles must be distinct, got {angles}"
                )
    if spec.placement_min_separation_m is not None:
        if not (
            math.isfinite(spec.placement_min_separation_m) and spec.placement_min_separation_m > 0
        ):
            raise ValueError(
                f"{spec.name}: placement_min_separation_m must be finite and > 0, got "
                f"{spec.placement_min_separation_m}"
            )
        if len(spec.sampled_placements) < 2:
            raise ValueError(
                f"{spec.name}: placement_min_separation_m needs >= 2 sampled placements to "
                f"separate, got {len(spec.sampled_placements)}"
            )
    all_keys = list(spec.manifest_keys)
    if len(all_keys) != len(set(all_keys)):
        raise ValueError(f"{spec.name}: duplicate manifest keys {all_keys}")
    sampling_keys = list(spec.sampling_keys)
    if len(sampling_keys) != len(set(sampling_keys)):
        raise ValueError(f"{spec.name}: duplicate sampling keys {sampling_keys}")
    if spec.r0_teleop_demo_repos is not None:
        if not spec.r0_teleop_demo_repos:
            raise ValueError(
                f"{spec.name}: r0_teleop_demo_repos is an empty dict; use None to mean "
                "'no verified teleop demo repos' (which skips the median reference lines)"
            )
        seen_repos: set[str] = set()
        for demo_set, repos in spec.r0_teleop_demo_repos.items():
            if not repos:
                raise ValueError(f"{spec.name}: r0_teleop_demo_repos[{demo_set!r}] has no repos")
            for repo in repos:
                if not repo or _R0_TELEOP_REPO_MARKER not in repo:
                    raise ValueError(
                        f"{spec.name}: r0_teleop_demo_repos[{demo_set!r}] entry {repo!r} must be a "
                        f"non-empty R0 TELEOP split id containing {_R0_TELEOP_REPO_MARKER!r} (R1+ "
                        "DAgger corrections are not teleop demos and must not seed the "
                        "demo-length median)"
                    )
                if repo in seen_repos:
                    raise ValueError(
                        f"{spec.name}: r0_teleop_demo_repos repeats {repo!r} across demo sets"
                    )
                seen_repos.add(repo)
    _TASK_SPECS[spec.name] = spec
    return spec


def get_task_spec(name: str) -> RealTaskSpec:
    """Look up a task spec by lifecycle key (e.g. ``"square_d2"``)."""
    try:
        return _TASK_SPECS[name]
    except KeyError:
        raise KeyError(f"unknown real task {name!r}; registered: {sorted(_TASK_SPECS)}") from None


def registered_task_specs() -> tuple[RealTaskSpec, ...]:
    """All registered specs, in registration order."""
    return tuple(_TASK_SPECS.values())


def task_name_choices() -> tuple[str, ...]:
    """``--task-name`` values the collectors accept: every registered task's collection
    task name, sorted."""
    return tuple(sorted(spec.task_name for spec in _TASK_SPECS.values()))


TASK_NAME_HELP = (
    "Released task of the collection, stored in the dataset: marker_d2 (Insert Marker), "
    "square_d2 (Thread Nut), routing_d2 (Route Cable)."
)


def find_task_spec_by_task_name(task_name: str) -> RealTaskSpec | None:
    """Look up a spec by collection/LeRobot task name (e.g. ``"square_d2"``).

    Returns ``None`` for unregistered names so callers can route non-real
    (sim) manifests to their own task handling.
    """
    for spec in _TASK_SPECS.values():
        if spec.task_name == task_name:
            return spec
    return None


# marker_d2 (Insert Marker): free 3-DOF pen (+/- 3 in x, +/- 6 in y, full yaw), plus
# a RED holder sampled JOINTLY with the pen over a box and snapped to a 1-inch grid (the
# operator places the holder at the snapped point). holder_x/holder_y are recorded in the
# manifest. Range x in [6, 8] in, y in [-4, 0] in -> a 3x5 = 15-point grid.
MARKER_D2 = _register(
    RealTaskSpec(
        name="marker_d2",
        task_name="marker_d2",
        state_keys=("pen_x", "pen_y", "pen_yaw"),
        bounds=(
            (-3.0 * INCH_TO_M, 3.0 * INCH_TO_M),  # pen_x: +/- 3 in
            (-6.0 * INCH_TO_M, 6.0 * INCH_TO_M),  # pen_y: +/- 6 in
            (-math.pi, math.pi),
        ),
        # Finer pen coverage grid than the inherited (4,5,4): 1 in x (6) x 1 in y (12)
        # x 45 deg yaw (8). The holder's 3x5 grid JOINS this in sampling_grid_dims (the
        # joint pen+holder coverage grid). NB: this is the coverage/diagnostic grid; a
        # future robot grid-eval should pick its own coarser cell set (~50 rollouts/cell).
        grid_dims=(6, 12, 8),
        sampled_placements={
            "holder": GridSampledPlacement(
                keys=("holder_x", "holder_y"),
                bounds=((6.0 * INCH_TO_M, 8.0 * INCH_TO_M), (-4.0 * INCH_TO_M, 0.0)),
                snap_m=INCH_TO_M,  # 1-inch grid -> x in {6,7,8}, y in {-4,-3,-2,-1,0}
            )
        },
        # marker_d2 stores 4 cameras under ROLE names (side_1, wrist_left, side_2,
        # wrist_right); the DP policy consumes side_1 + wrist_left. This is the train
        # default --camera-keys when none is passed explicitly.
        consumed_camera_roles=("side_1", "wrist_left"),
        # Active final-line marker_d2 cadence: train to 100k, evaluate more densely than
        # checkpoints so checkpoint storage stays bounded while preserving selection signal.
        training=DPTrainingRecipe(training_steps=100_000, save_freq=25_000, eval_freq=5_000),
        # R0 teleop demo splits, 100 teleop demos each.
        r0_teleop_demo_repos={
            "uniform": ("mulligan/real-marker-d2-c00-teleop-baseline",),
            "sobol": ("mulligan/real-marker-d2-c00-teleop-sobol",),
        },
    )
)

# square_d2 (Thread Nut): free 3-DOF nut (+/- 5 in x, +/- 9 in y, full yaw) plus a peg
# sampled JOINTLY with the nut over a box and snapped to a 1-inch grid (the operator places
# the peg at the snapped point); peg_x/peg_y are recorded in the manifest. The peg grid is
# offset by 0.5 in so each peg center lands ON a physical 1-inch marker dot (the dots sit at
# half-integer inches in the operator frame): x in {9.5, 10.5, 11.5} in, y in {-4.5, -3.5,
# -2.5, -1.5, -0.5} in -> a 3x5 = 15-point grid. peg_x starts at 9.5 in for physical
# feasibility: the nut reaches nut_x = +5 in forward and its rotated footprint extends ~3.2 in
# past its center (the handle tip), so a nearer peg could overlap the nut at setup.
SQUARE_D2 = _register(
    RealTaskSpec(
        name="square_d2",
        task_name="square_d2",
        state_keys=("nut_x", "nut_y", "nut_yaw"),
        bounds=(
            (-5.0 * INCH_TO_M, 5.0 * INCH_TO_M),  # nut_x: +/- 5 in
            (-9.0 * INCH_TO_M, 9.0 * INCH_TO_M),  # nut_y: +/- 9 in
            (-math.pi, math.pi),
        ),
        # Finer nut coverage grid than the inherited (4,5,4): 1 in x (10) x 1 in y (18)
        # x 45 deg yaw (8). The peg's 3x5 grid JOINS this in sampling_grid_dims (the
        # joint nut+peg coverage grid). NB: this is the coverage/diagnostic grid; a
        # future robot grid-eval should pick its own coarser cell set (~50 rollouts/cell).
        grid_dims=(10, 18, 8),
        sampled_placements={
            "peg": GridSampledPlacement(
                keys=("peg_x", "peg_y"),
                bounds=((9.5 * INCH_TO_M, 11.5 * INCH_TO_M), (-4.5 * INCH_TO_M, -0.5 * INCH_TO_M)),
                # 1-inch grid, half-inch offset so pegs land ON the marker dots ->
                # x in {9.5,10.5,11.5}, y in {-4.5,-3.5,-2.5,-1.5,-0.5}
                snap_m=INCH_TO_M,
            )
        },
        consumed_camera_roles=("side_1", "wrist_left"),
        # Active final-line square_d2 cadence matches marker_d2: train to 100k, evaluate
        # more densely than checkpoints so checkpoint storage stays bounded.
        training=DPTrainingRecipe(training_steps=100_000, save_freq=25_000, eval_freq=5_000),
        # R0 teleop demo splits, 100 teleop demos each.
        r0_teleop_demo_repos={
            "uniform": ("mulligan/real-square-d2-c00-teleop-baseline",),
            "sobol": ("mulligan/real-square-d2-c00-teleop-sobol",),
        },
    )
)

# routing_d2: cable/rope routing. The FIRST non-pen real line: it has NO free 3-DOF pose.
# Its only free continuous DOF is the rope's x position (the rope spans the full y-axis and
# only translates along x), so state_keys = ("rope_x",) with an explicit 1-DOF non-periodic
# state_periodic_mask. Two clips (clip_left / clip_right) are GridSampledPlacements on the
# SAME 0.5-offset 1-inch dot grid as square_d2's peg (so clip centers land ON physical
# marker dots): full x in [-6.5, 9.5] in (17 dots), y split by side (left +0.5..+9.5,
# right -9.5..-0.5; 10 dots each). Each clip also carries a DISCRETE orientation in
# {-45, -90, -135} deg, represented in the joint sampling space as an integer CHOICE INDEX
# (orient_key, snap 1, 3 cells) so the regular-grid snap + coverage machinery is reused
# unchanged on the irregular angle lattice; the manifest persists BOTH the resolved angle
# (clip_*_yaw, operator/overlay facing) and the index (clip_*_oidx, sampler/coverage facing).
# 0 deg = clip long axis along +x (forward); the three NEGATIVE yaws all rotate the clip's
# mouth toward -y (operator right), so every clip "points" roughly left->right, indicating the
# direction the rope threads through the clips. The 9 manifest keys = rope_x + per-clip
# (x, y, yaw, oidx). Clips must be >= 2 in apart (their taps occupy the holes around each
# clip, so closer clips would fight for the same holes) -- enforced by rejection sampling +
# loader validation via placement_min_separation_m. Collected in the new role-named 4-cam
# room under the ImageNet-norm default from day one. Unlike the pen lines (side_1 +
# wrist_left), the policy consumes side_1 + side_2 + wrist_left: the rope spans the full y-axis, so
# the two
# opposing side views cover it end-to-end, and the wrist view resolves the fine clip-seat
# manipulation.
_ROUTING_CLIP_ANGLES = (-math.pi / 4.0, -math.pi / 2.0, -3.0 * math.pi / 4.0)

ROUTING_D2 = _register(
    RealTaskSpec(
        name="routing_d2",
        task_name="routing_d2",
        state_keys=("rope_x",),
        state_periodic_mask=(False,),  # rope_x is a single continuous, non-periodic DOF
        bounds=((-6.5 * INCH_TO_M, 9.5 * INCH_TO_M),),  # rope_x: [-6.5, 9.5] in (continuous)
        # 16 coverage cells over the 16-inch continuous rope_x span (the per-clip x/y/orient
        # cells JOIN this in sampling_grid_dims). NB: coverage/diagnostic grid only; a future
        # robot grid-eval should pick its own coarser cell set.
        grid_dims=(16,),
        sampled_placements={
            "clip_left": GridSampledPlacement(
                keys=("clip_left_x", "clip_left_y"),
                bounds=((-6.5 * INCH_TO_M, 9.5 * INCH_TO_M), (0.5 * INCH_TO_M, 9.5 * INCH_TO_M)),
                snap_m=INCH_TO_M,  # 0.5-offset 1-inch dots: x 17 pts, y 10 pts (left half)
                orient_key="clip_left_oidx",
                orient_angle_key="clip_left_yaw",
                orient_angles=_ROUTING_CLIP_ANGLES,
            ),
            "clip_right": GridSampledPlacement(
                keys=("clip_right_x", "clip_right_y"),
                bounds=((-6.5 * INCH_TO_M, 9.5 * INCH_TO_M), (-9.5 * INCH_TO_M, -0.5 * INCH_TO_M)),
                snap_m=INCH_TO_M,  # 0.5-offset 1-inch dots: x 17 pts, y 10 pts (right half)
                orient_key="clip_right_oidx",
                orient_angle_key="clip_right_yaw",
                orient_angles=_ROUTING_CLIP_ANGLES,
            ),
        },
        # Clips' taps occupy the holes around each clip; require >= 2 in between the two clip
        # centers so they never compete for the same holes (rejection-sampled + loader-checked).
        placement_min_separation_m=2.0 * INCH_TO_M,
        # The task seats the rope in two clips: the first seat is a mid-episode sub-goal
        # (one outcome-editor 'g' mark -> reward=1.0 spike), the second seat IS the
        # terminal success. So exactly one intermediate subtask mark per episode.
        num_subtask_marks=1,
        consumed_camera_roles=("side_1", "side_2", "wrist_left"),
        # R0 teleop demo splits, 100 teleop demos each.
        r0_teleop_demo_repos={
            "uniform": ("mulligan/real-routing-d2-c00-teleop-baseline",),
            "sobol": ("mulligan/real-routing-d2-c00-teleop-sobol",),
        },
        # Routing-fit side ROIs (fit from the R0 reset-frame homographies;
        # station defaults kept for the other lines/consumers).
        # side_2: content envelope (all clip placements + rope at both x extremes) spans
        #   image x [146, 423], y [135, 423] -> margins L16/T15/R17/B22. Excludes the wall,
        #   desk, operator sliver, and most of the collection-GUI monitor (bottom sliver
        #   y 120-135 is unremovable without cutting deep clip_right).
        # side_1: content envelope x [162, 543], y [138, 457] -> margins L22/T18/R17/B13.
        #   The station box spent its top ~130 px on whiteboard AND its bottom edge (447)
        #   cut the rope's near-corner extreme by ~10 px, so the bottom moves DOWN to 470.
        camera_crop_overrides={
            "side_1": (140, 120, 560, 470),
            "side_2": (130, 120, 440, 445),
        },
        training=DPTrainingRecipe(training_steps=100_000, save_freq=25_000, eval_freq=5_000),
    )
)
