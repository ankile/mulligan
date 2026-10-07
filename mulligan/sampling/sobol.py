"""
Initial-state samplers for Square-Narrow (``NutAssemblySquare``) and
Square-Broad (``Square_D1``) teleop/DAgger collection.

Uses a scrambled Sobol sequence to produce low-discrepancy points in the
(x, y, yaw) state space.  Achieves ~300x better space-filling than FPS
seeded from training data (discrepancy ~0.0004 vs ~0.117).

The sequence is deterministic given a seed, so resuming is trivial: just
count how many episodes were already collected and skip that many points.

Round-0 starts (``data/sim/start_manifests/*/r00/init_states``):

- Square-Narrow baseline arm: ``UniformSquareSampler(seed=202605231)``;
  Mulligan arm: :func:`sample_square_sobol2d_uniform_x` (seed 42, x seed
  202605234, pool 1024).
- Square-Broad baseline arm: ``UniformSquareD1Sampler(seed=202605232)``;
  Mulligan arm: ``SquareD1SobolSampler(seed=42, batch_size=2048)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats.qmc import Sobol

from mulligan.utils.manifest_matching import min_pairwise_cyclic_distance
from mulligan.utils.quaternion import z_rot_to_quat_wxyz
from mulligan.utils.state_to_grid import (
    extract_nut_pose_from_env_state,
    extract_peg_pos_from_env_state,
)

SQUARE_NARROW_X_RANGE = (-0.115, -0.11)
SQUARE_NARROW_Y_RANGE = (0.11, 0.225)
SQUARE_NARROW_YAW_RANGE = (-np.pi, np.pi)


def _build_circular_yaw_tree(points: np.ndarray, *, yaw_index: int) -> tuple[cKDTree, np.ndarray]:
    """Build a KDTree that treats one column as a wrapped yaw coordinate.

    Tiles the point set across yaw offsets ``{-2pi, 0, +2pi}`` so a standard
    Euclidean ``cKDTree.query`` finds the nearest neighbour across the wrap
    boundary. Returns the tiled tree plus a flat index back to the original
    row in ``points``.

    For one-shot nearest-state matching (i.e. classifying a saved env_state
    against a manifest), prefer ``mulligan.utils.manifest_matching.ManifestMatcher``
    — its cos/sin embedding scales to multiple angular columns without the
    ``3 ** n_angular`` tile blowup.
    """
    shifted_points = []
    source_indices = []
    for yaw_offset in (-2.0 * np.pi, 0.0, 2.0 * np.pi):
        shifted = points.copy()
        shifted[:, yaw_index] += yaw_offset
        shifted_points.append(shifted)
        source_indices.extend(range(len(points)))
    return cKDTree(np.vstack(shifted_points)), np.asarray(source_indices, dtype=np.int64)


def _min_circular_yaw_distance(points: np.ndarray, *, yaw_index: int) -> float:
    """Minimum pairwise distance with the yaw column compared circularly.

    Delta-wrap metric (delegated to
    :func:`mulligan.utils.manifest_matching.min_pairwise_cyclic_distance`), kept on
    that semantics so logged FPS coverage radii stay numerically comparable
    across rounds.
    """
    return min_pairwise_cyclic_distance(points, [yaw_index])


def _raw_frame_lookup_by_index(dataset) -> dict[int, dict]:
    try:
        hf_dataset = dataset.hf_dataset
    except AttributeError:
        return {
            int(ep["dataset_from_index"]): dataset[int(ep["dataset_from_index"])]
            for ep in dataset.meta.episodes
        }
    return {int(np.asarray(item["index"]).item()): item for item in hf_dataset}


def _scan_dataset_episodes(dataset, extract):
    """Iterate every episode's first frame and apply `extract`.

    `extract(env_state) -> (rounded_dedup_key: tuple, point: Any)` lets each
    sampler return both a stable hashable key (used to count unique initial
    states for resume) and the in-memory representation it stores in
    `collected_points`.

    Returns: (n_unique, points_to_append).
    """
    seen: set[tuple] = set()
    points: list = []
    frame_lookup = _raw_frame_lookup_by_index(dataset)
    for ep_idx in range(dataset.num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        frame_idx = int(ep["dataset_from_index"])
        frame = frame_lookup[frame_idx]
        env_state = np.asarray(frame["observation.environment_state"])
        rounded_key, point = extract(env_state)
        seen.add(rounded_key)
        points.append(point)
    return len(seen), points


class SobolSampler:
    """Sobol quasi-random sampler over (x, y, yaw) for initial states.

    Note: the Sobol sequence is pre-generated once at __init__ time with
    `batch_size` points. There is no online extension; if a long DAgger run
    exhausts the budget, `sample_next` raises and the caller must
    reconstruct the sampler with a larger batch_size.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = SQUARE_NARROW_X_RANGE,
        y_range: tuple[float, float] = SQUARE_NARROW_Y_RANGE,
        yaw_range: tuple[float, float] = SQUARE_NARROW_YAW_RANGE,
        table_offset_z: float = 0.82,
        nut_z_offset: float = 0.07,
        seed: int = 42,
        batch_size: int = 1024,
        include_boundary: bool = True,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.yaw_range = yaw_range
        self.table_offset_z = table_offset_z
        self.nut_z_offset = nut_z_offset

        # Generate Sobol points in [0, 1]^3
        sobol = Sobol(d=3, scramble=True, seed=seed)
        raw = sobol.random(batch_size)  # (batch_size, 3)

        # Map [0,1]^3 -> physical ranges
        points = np.empty_like(raw)
        points[:, 0] = x_range[0] + raw[:, 0] * (x_range[1] - x_range[0])
        points[:, 1] = y_range[0] + raw[:, 1] * (y_range[1] - y_range[0])
        points[:, 2] = yaw_range[0] + raw[:, 2] * (yaw_range[1] - yaw_range[0])

        if include_boundary:
            if batch_size < 4:
                raise ValueError(
                    f"batch_size must be >= 4 when include_boundary=True "
                    f"(needed for the 4 corner injections), got {batch_size}"
                )
            # Inject 4 boundary corner points at the front (x_lo/hi × y_lo/hi,
            # yaw from Sobol so it's still quasi-random in that dimension).
            boundary = np.array(
                [
                    [x_range[0], y_range[0], points[0, 2]],
                    [x_range[1], y_range[0], points[1, 2]],
                    [x_range[0], y_range[1], points[2, 2]],
                    [x_range[1], y_range[1], points[3, 2]],
                ]
            )
            # Replace the first 4 Sobol points with explicit boundary points
            points[:4] = boundary

        self.planned_points: list[tuple[float, float, float]] = [
            (float(row[0]), float(row[1]), float(row[2])) for row in points
        ]
        self.collected_points: list[tuple[float, float, float]] = []
        self._next_idx = 0

    def sample_next(self) -> tuple[float, float, float]:
        """Return the next Sobol point; raises once the batch is exhausted."""
        if self._next_idx >= len(self.planned_points):
            raise RuntimeError(
                f"Sobol: budget exhausted "
                f"({len(self.planned_points)} planned, requested idx {self._next_idx}); "
                f"silent wrap-around would re-use points and destroy quasi-random coverage. "
                f"Increase batch_size or restructure the experiment."
            )

        return self.planned_points[self._next_idx]

    def record_state(self, x: float, y: float, yaw: float) -> None:
        """Record an observed initial state and advance to the next point."""
        self.collected_points.append((x, y, yaw))
        self._next_idx += 1

    def to_qpos(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Convert (x, y, yaw) to 7D qpos [x, y, z, qw, qx, qy, qz].

        Maps yaw to [0, 2*pi) before computing the quaternion to match
        robosuite's native NutAssemblySquare sampler convention.
        """
        z = self.table_offset_z + self.nut_z_offset
        position = np.array([x, y, z])
        quat_wxyz = z_rot_to_quat_wxyz(yaw % (2.0 * np.pi))

        return np.concatenate([position, quat_wxyz])

    @staticmethod
    def _extract_for_resume(env_state):
        x, y, yaw = extract_nut_pose_from_env_state(env_state)
        return (round(x, 4), round(y, 4), round(yaw, 4)), (x, y, yaw)

    def load_from_dataset(self, dataset_path: Path) -> None:
        """Resume from an existing local dataset by advancing past collected episodes."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id=dataset_path.name,
            root=str(dataset_path),
            download_videos=False,
        )
        n_unique, points = _scan_dataset_episodes(dataset, self._extract_for_resume)
        self.collected_points.extend(points)
        self._next_idx += n_unique
        remaining = max(0, len(self.planned_points) - self._next_idx)
        print(
            f"Sobol sampler: loaded {dataset.num_episodes} episodes "
            f"({n_unique} unique initial states) from {dataset_path.name}, "
            f"resuming at index {self._next_idx}, {remaining} remaining"
        )


class SquareD1SobolSampler:
    """5D Sobol sampler for Square_D1: joint (nut_x, nut_y, nut_yaw, peg_x, peg_y).

    Square_D1 has wider placement ranges and a randomized peg position.
    This sampler produces low-discrepancy points in the joint 5D space
    with collision rejection to ensure nut and peg don't overlap.
    """

    def __init__(
        self,
        nut_x_range: tuple[float, float] = (-0.115, 0.115),
        nut_y_range: tuple[float, float] = (-0.255, 0.255),
        nut_yaw_range: tuple[float, float] = (-np.pi, np.pi),
        peg_x_range: tuple[float, float] = (-0.1, 0.3),
        peg_y_range: tuple[float, float] = (-0.2, 0.2),
        min_clearance: float = 0.13263,  # nut + peg1 horizontal_radius (0.110 + 0.02263)
        table_offset_z: float = 0.82,
        nut_z_offset: float = 0.07,
        seed: int = 42,
        batch_size: int = 2048,
    ) -> None:
        self.table_offset_z = table_offset_z
        self.nut_z_offset = nut_z_offset

        # Generate Sobol points in [0, 1]^5
        sobol = Sobol(d=5, scramble=True, seed=seed)
        raw = sobol.random(batch_size)

        # Map [0,1]^5 -> physical ranges
        points = np.empty_like(raw)
        points[:, 0] = nut_x_range[0] + raw[:, 0] * (nut_x_range[1] - nut_x_range[0])
        points[:, 1] = nut_y_range[0] + raw[:, 1] * (nut_y_range[1] - nut_y_range[0])
        points[:, 2] = nut_yaw_range[0] + raw[:, 2] * (nut_yaw_range[1] - nut_yaw_range[0])
        points[:, 3] = peg_x_range[0] + raw[:, 3] * (peg_x_range[1] - peg_x_range[0])
        points[:, 4] = peg_y_range[0] + raw[:, 4] * (peg_y_range[1] - peg_y_range[0])

        # Reject collisions: nut_xy vs peg_xy must be far enough apart
        dist = np.linalg.norm(points[:, :2] - points[:, 3:5], axis=1)
        valid = dist > min_clearance
        points = points[valid]

        # Store as list of ((nut_x, nut_y, nut_yaw), (peg_x, peg_y)) tuples
        self.planned_points: list[tuple[tuple[float, float, float], tuple[float, float]]] = [
            (
                (float(r[0]), float(r[1]), float(r[2])),
                (float(r[3]), float(r[4])),
            )
            for r in points
        ]
        self.collected_points: list[tuple[tuple[float, float, float], tuple[float, float]]] = []
        self._next_idx = 0

        print(
            f"SquareD1SobolSampler: {len(self.planned_points)} valid points "
            f"from {batch_size} candidates "
            f"({batch_size - len(self.planned_points)} rejected for collision)"
        )

    def sample_next(self) -> tuple[tuple[float, float, float], tuple[float, float]]:
        """Return the next Sobol point as ((nut_x, nut_y, nut_yaw), (peg_x, peg_y))."""
        if self._next_idx >= len(self.planned_points):
            raise RuntimeError(
                f"SquareD1Sobol: budget exhausted "
                f"({len(self.planned_points)} planned, requested idx {self._next_idx}); "
                f"silent wrap-around would re-use points and destroy quasi-random coverage. "
                f"Increase batch_size or restructure the experiment."
            )

        return self.planned_points[self._next_idx]

    def record_state(
        self,
        nut_state: tuple[float, float, float],
        peg_state: tuple[float, float],
    ) -> None:
        """Record an observed initial state and advance to the next point."""
        self.collected_points.append((nut_state, peg_state))
        self._next_idx += 1

    def nut_to_qpos(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Convert (x, y, yaw) to 7D qpos [x, y, z, qw, qx, qy, qz].

        Maps yaw to [0, 2*pi) to match robosuite's native Square_D1 sampler convention.
        """
        z = self.table_offset_z + self.nut_z_offset
        position = np.array([x, y, z])
        quat_wxyz = z_rot_to_quat_wxyz(yaw % (2.0 * np.pi))

        return np.concatenate([position, quat_wxyz])

    @staticmethod
    def _extract_for_resume(env_state):
        x, y, yaw = extract_nut_pose_from_env_state(env_state, task="Square_D1")
        px, py, _ = extract_peg_pos_from_env_state(env_state)
        rounded = (round(x, 4), round(y, 4), round(yaw, 4), round(px, 4), round(py, 4))
        return rounded, ((x, y, yaw), (px, py))

    def load_from_dataset(self, dataset_path: Path) -> None:
        """Resume from an existing local dataset by advancing past collected episodes."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id=dataset_path.name,
            root=str(dataset_path),
            download_videos=False,
        )
        n_unique, points = _scan_dataset_episodes(dataset, self._extract_for_resume)
        self.collected_points.extend(points)
        self._next_idx += n_unique
        remaining = max(0, len(self.planned_points) - self._next_idx)
        print(
            f"SquareD1Sobol: loaded {dataset.num_episodes} episodes "
            f"({n_unique} unique states) from {dataset_path.name}, "
            f"resuming at index {self._next_idx}, {remaining} remaining"
        )


class UniformSquareD1Sampler:
    """5D uniform-random sampler for Square_D1: (nut_x, nut_y, nut_yaw, peg_x, peg_y).

    Exists because MimicGen's Square_D1 env only randomizes the peg once, in
    `_load_model` (env creation time), not in `_reset_internal` — so
    `env.reset()` keeps the peg fixed across episodes within a worker. This
    sampler emits a fresh uniform sample every call so each episode gets an
    independent peg placement, matching what `SquareD1SobolSampler` provides
    but without Sobol's 2048-point budget (Sobol exhausts across rounds).

    Bounds default to the MimicGen Square_D1 `_get_initial_placement_bounds`
    values so the distribution matches the teleop/DAgger Sobol runs:
      - nut_x ∈ [-0.115, 0.115]
      - nut_y ∈ [-0.255, 0.255]
      - nut_yaw ∈ [0, 2π)
      - peg_x ∈ [-0.1, 0.3]
      - peg_y ∈ [-0.2, 0.2]

    Collision rejection keeps the same `min_clearance=0.13263` as the Sobol
    sampler (nut horizontal_radius 0.110 + peg horizontal_radius 0.02263).
    """

    def __init__(
        self,
        nut_x_range: tuple[float, float] = (-0.115, 0.115),
        nut_y_range: tuple[float, float] = (-0.255, 0.255),
        nut_yaw_range: tuple[float, float] = (0.0, 2.0 * np.pi),
        peg_x_range: tuple[float, float] = (-0.1, 0.3),
        peg_y_range: tuple[float, float] = (-0.2, 0.2),
        min_clearance: float = 0.13263,
        table_offset_z: float = 0.82,
        nut_z_offset: float = 0.07,
        seed: int = 42,
        max_rejection_retries: int = 5000,
        # Unused args kept for interface compatibility with SquareD1SobolSampler
        # so the same construction call site (batch_size=...) works for both.
        batch_size: int | None = None,
    ) -> None:
        self.nut_x_range = nut_x_range
        self.nut_y_range = nut_y_range
        self.nut_yaw_range = nut_yaw_range
        self.peg_x_range = peg_x_range
        self.peg_y_range = peg_y_range
        self.min_clearance = min_clearance
        self.table_offset_z = table_offset_z
        self.nut_z_offset = nut_z_offset
        self.max_rejection_retries = max_rejection_retries
        self._rng = np.random.default_rng(seed)

        # Interface parity with SquareD1SobolSampler (callers read _next_idx).
        self.collected_points: list[tuple[tuple[float, float, float], tuple[float, float]]] = []
        self._next_idx = 0

        print(
            f"UniformSquareD1Sampler: seed={seed}, "
            f"peg_x={peg_x_range}, peg_y={peg_y_range}, "
            f"nut_x={nut_x_range}, nut_y={nut_y_range}"
        )

    def sample_next(self) -> tuple[tuple[float, float, float], tuple[float, float]]:
        """Return a fresh uniform-random (nut_state, peg_state) tuple.

        Rejects samples where nut and peg horizontal distance falls below
        `min_clearance` (matches the Sobol sampler's collision check).
        """
        for _ in range(self.max_rejection_retries):
            nut_x = self._rng.uniform(*self.nut_x_range)
            nut_y = self._rng.uniform(*self.nut_y_range)
            nut_yaw = self._rng.uniform(*self.nut_yaw_range)
            peg_x = self._rng.uniform(*self.peg_x_range)
            peg_y = self._rng.uniform(*self.peg_y_range)
            if np.hypot(nut_x - peg_x, nut_y - peg_y) > self.min_clearance:
                return ((float(nut_x), float(nut_y), float(nut_yaw)), (float(peg_x), float(peg_y)))
        raise RuntimeError(
            f"UniformSquareD1Sampler: failed to find a collision-free sample "
            f"after {self.max_rejection_retries} retries — check bounds."
        )

    def record_state(
        self,
        nut_state: tuple[float, float, float],
        peg_state: tuple[float, float],
    ) -> None:
        """Record an observed initial state (tracks count, no-op otherwise)."""
        self.collected_points.append((nut_state, peg_state))
        self._next_idx += 1

    def nut_to_qpos(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Convert (x, y, yaw) to 7D qpos [x, y, z, qw, qx, qy, qz]."""
        z = self.table_offset_z + self.nut_z_offset
        position = np.array([x, y, z])
        quat_wxyz = z_rot_to_quat_wxyz(yaw % (2.0 * np.pi))
        return np.concatenate([position, quat_wxyz])


class SquareD1ListSampler(SquareD1SobolSampler):
    """List-based 5D sampler for Square_D1 — replays a precomputed list of states.

    Loads ((nut_x, nut_y, nut_yaw), (peg_x, peg_y)) tuples from a JSON file
    with a top-level `states` list of dicts containing the 5 keys `nut_x`,
    `nut_y`, `nut_yaw`, `peg_x`, `peg_y` (e.g. the round start lists under
    `data/sim/start_manifests/square_broad/`).

    Inherits `sample_next`, `record_state`, `nut_to_qpos` from
    `SquareD1SobolSampler`. Overrides `load_from_dataset`
    so that resume only counts states that actually appear in the planned
    list; counting every unique state of the current DAgger dataset would
    advance `_next_idx` past the end of the list.

    The list is shuffled once at init with the given `seed`, so re-running
    with the same seed reproduces the order — combined with overlap-only
    resume, that gives correct continuation from a partial collection.
    """

    def __init__(
        self,
        states_file: Path,
        table_offset_z: float = 0.82,
        nut_z_offset: float = 0.07,
        seed: int = 42,
        shuffle: bool = True,
        match_tolerance: float = 1e-3,
    ) -> None:
        # NOTE: intentionally NOT calling super().__init__() — SquareD1SobolSampler
        # would generate a 5D Sobol sequence we don't want; we get our planned_points
        # from the hard-state file instead. We inherit only `sample_next`,
        # `record_state`, and `nut_to_qpos` for shared behavior. If new state is
        # ever added to SquareD1SobolSampler.__init__ and a subclass uses it, this
        # constructor must be updated explicitly.
        self.table_offset_z = table_offset_z
        self.nut_z_offset = nut_z_offset
        self.states_file = Path(states_file)

        with open(self.states_file) as f:
            payload = json.load(f)

        if "states" not in payload:
            raise ValueError(
                f"{self.states_file}: expected top-level 'states' list, got keys={list(payload.keys())}"
            )

        points = []
        for s in payload["states"]:
            nut = (float(s["nut_x"]), float(s["nut_y"]), float(s["nut_yaw"]))
            peg = (float(s["peg_x"]), float(s["peg_y"]))
            points.append((nut, peg))

        if shuffle:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(points))
            points = [points[i] for i in order]

        self.planned_points: list[tuple[tuple[float, float, float], tuple[float, float]]] = points
        self.collected_points: list[tuple[tuple[float, float, float], tuple[float, float]]] = []
        self._next_idx = 0
        self._match_tolerance = match_tolerance

        # Build a 5D KDTree for tolerance-based overlap checking. Hash-based
        # key matching with fixed-decimal rounding (the previous approach)
        # silently miscounts when the recorded env_state diverges from the
        # planned point by an amount that straddles the rounding boundary
        # (physics drift of ~7e-5 put counterfactual replays just outside
        # their planned 4-decimal key). Tolerance-based
        # nearest-neighbor matching is robust to this.
        self._planned_array = np.array(
            [[nut[0], nut[1], nut[2], peg[0], peg[1]] for (nut, peg) in points]
        )
        self._planned_tree, self._planned_tree_source_idx = _build_circular_yaw_tree(
            self._planned_array, yaw_index=2
        )

        # Sanity-check that the rounding-to-4-decimals dedup key (used by
        # SquareD1SobolSampler.load_from_*) and the tolerance-based KDTree
        # match are compatible: if any two distinct planned points are closer
        # than 2 * match_tolerance, a tolerance match could pick the wrong
        # neighbor.
        if len(self._planned_array) > 1:
            min_dist = _min_circular_yaw_distance(self._planned_array, yaw_index=2)
            if min_dist <= 2.0 * match_tolerance:
                raise ValueError(
                    f"Hard-state list has two planned points within "
                    f"{min_dist:.4g} of each other, which is <= 2 * "
                    f"match_tolerance ({match_tolerance}). Tolerance-based "
                    f"resume matching would be ambiguous; tighten match_tolerance "
                    f"or deduplicate the input list."
                )

        print(
            f"SquareD1ListSampler: loaded {len(self.planned_points)} states "
            f"from {self.states_file.name} (shuffle={shuffle}, seed={seed}, "
            f"match_tolerance={match_tolerance})"
        )

    def _planned_idx_for(
        self,
        x: float,
        y: float,
        yaw: float,
        px: float,
        py: float,
    ) -> int | None:
        """Return the planned-list index nearest the given 5D state, or None
        if no planned point is within `match_tolerance` (state is off-list)."""
        dist, idx = self._planned_tree.query([x, y, yaw, px, py], k=1)
        if dist > self._match_tolerance:
            return None
        return int(self._planned_tree_source_idx[int(idx)])

    def _count_overlap_with_dataset(self, dataset) -> tuple[int, list]:
        """Return (#unique-list-overlap-states, recorded points) for a
        LeRobotDataset. Uses tolerance-based nearest-planned-point matching
        and dedups by planned-point INDEX (so counterfactual replays whose
        recorded env_state differs slightly from the human run still map to
        the same planned point — fixing the over/undercount caused by
        rounding-boundary effects)."""
        seen_planned_idx: set[int] = set()
        recorded: list[tuple[tuple[float, float, float], tuple[float, float]]] = []
        frame_lookup = _raw_frame_lookup_by_index(dataset)
        for ep_idx in range(dataset.num_episodes):
            ep = dataset.meta.episodes[ep_idx]
            frame_idx = int(ep["dataset_from_index"])
            frame = frame_lookup[frame_idx]
            env_state = np.asarray(frame["observation.environment_state"])
            x, y, yaw = extract_nut_pose_from_env_state(env_state, task="Square_D1")
            px, py, _ = extract_peg_pos_from_env_state(env_state)
            idx = self._planned_idx_for(x, y, yaw, px, py)
            if idx is None or idx in seen_planned_idx:
                continue
            seen_planned_idx.add(idx)
            recorded.append(((x, y, yaw), (px, py)))
        return len(recorded), recorded

    def load_from_dataset(self, dataset_path: Path) -> None:
        """Resume an ordered list collection from a local dataset."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=dataset_path.name, root=str(dataset_path))
        if dataset.num_episodes > len(self.planned_points):
            raise ValueError(
                f"{dataset_path.name} has {dataset.num_episodes} episodes, "
                f"but the ordered state list has only {len(self.planned_points)} points."
            )
        n_overlap, _ = self._count_overlap_with_dataset(dataset)
        self._next_idx = dataset.num_episodes
        self.collected_points = list(self.planned_points[: self._next_idx])

        remaining = max(0, len(self.planned_points) - self._next_idx)
        print(
            f"SquareD1List: loaded {dataset.num_episodes} episodes from "
            f"{dataset_path.name}; {n_overlap} unique states overlap with "
            f"hard-state list, resuming ordered list at _next_idx={self._next_idx} "
            f"({remaining} remaining)"
        )


class SquareListSampler(SobolSampler):
    """List-based 3D sampler for Square-Narrow (NutAssemblySquare) — replays a
    precomputed list of (nut_x, nut_y, nut_yaw) states.

    Square-Narrow analog of `SquareD1ListSampler`. Loads tuples from a JSON file
    with a top-level `states` list of dicts containing the three keys `nut_x`,
    `nut_y`, `nut_yaw`. The peg is fixed on Square-Narrow, so peg keys (if
    present) are ignored.

    Inherits `sample_next`, `record_state`, and `to_qpos` from `SobolSampler`.
    Overrides `load_from_dataset` so resume only counts states that actually
    appear in the planned list (same rationale as `SquareD1ListSampler`).
    """

    def __init__(
        self,
        states_file: Path,
        table_offset_z: float = 0.82,
        nut_z_offset: float = 0.07,
        seed: int = 42,
        shuffle: bool = True,
        match_tolerance: float = 1e-3,
    ) -> None:
        # Intentionally NOT calling super().__init__() — SobolSampler would
        # generate a 3D Sobol sequence we don't want. Same pattern as
        # SquareD1ListSampler.
        self.table_offset_z = table_offset_z
        self.nut_z_offset = nut_z_offset
        self.states_file = Path(states_file)

        with open(self.states_file) as f:
            payload = json.load(f)
        if "states" not in payload:
            raise ValueError(
                f"{self.states_file}: expected top-level 'states' list, "
                f"got keys={list(payload.keys())}"
            )

        points: list[tuple[float, float, float]] = []
        for s in payload["states"]:
            points.append((float(s["nut_x"]), float(s["nut_y"]), float(s["nut_yaw"])))
        if shuffle:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(points))
            points = [points[i] for i in order]

        self.planned_points: list[tuple[float, float, float]] = points
        self.collected_points: list[tuple[float, float, float]] = []
        self._next_idx = 0
        self._match_tolerance = match_tolerance

        self._planned_array = np.array(points, dtype=np.float64)
        self._planned_tree, self._planned_tree_source_idx = _build_circular_yaw_tree(
            self._planned_array, yaw_index=2
        )

        # Same neighbor-distance sanity check as the 5D variant.
        if len(self._planned_array) > 1:
            min_dist = _min_circular_yaw_distance(self._planned_array, yaw_index=2)
            if min_dist <= 2.0 * match_tolerance:
                raise ValueError(
                    f"Square-Narrow start list has two planned points within "
                    f"{min_dist:.4g} of each other, which is <= 2 * "
                    f"match_tolerance ({match_tolerance}). Tolerance-based "
                    f"resume matching would be ambiguous; tighten match_tolerance "
                    f"or deduplicate the input list."
                )

        print(
            f"SquareListSampler: loaded {len(self.planned_points)} states "
            f"from {self.states_file.name} (shuffle={shuffle}, seed={seed}, "
            f"match_tolerance={match_tolerance})"
        )

    def _planned_idx_for(self, x: float, y: float, yaw: float) -> int | None:
        """Return planned-list index nearest the 3D state, or None if off-list."""
        dist, idx = self._planned_tree.query([x, y, yaw], k=1)
        if dist > self._match_tolerance:
            return None
        return int(self._planned_tree_source_idx[int(idx)])

    def _count_overlap_with_dataset(self, dataset) -> tuple[int, list[tuple[float, float, float]]]:
        seen_planned_idx: set[int] = set()
        recorded: list[tuple[float, float, float]] = []
        frame_lookup = _raw_frame_lookup_by_index(dataset)
        for ep_idx in range(dataset.num_episodes):
            ep = dataset.meta.episodes[ep_idx]
            frame_idx = int(ep["dataset_from_index"])
            frame = frame_lookup[frame_idx]
            env_state = np.asarray(frame["observation.environment_state"])
            x, y, yaw = extract_nut_pose_from_env_state(env_state)
            idx = self._planned_idx_for(x, y, yaw)
            if idx is None or idx in seen_planned_idx:
                continue
            seen_planned_idx.add(idx)
            recorded.append((x, y, yaw))
        return len(recorded), recorded

    def load_from_dataset(self, dataset_path: Path) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=dataset_path.name, root=str(dataset_path))
        if dataset.num_episodes > len(self.planned_points):
            raise ValueError(
                f"{dataset_path.name} has {dataset.num_episodes} episodes, "
                f"but the ordered state list has only {len(self.planned_points)} points."
            )
        n_overlap, _ = self._count_overlap_with_dataset(dataset)
        self._next_idx = dataset.num_episodes
        self.collected_points = list(self.planned_points[: self._next_idx])
        remaining = max(0, len(self.planned_points) - self._next_idx)
        print(
            f"SquareList: loaded {dataset.num_episodes} episodes from "
            f"{dataset_path.name}; {n_overlap} unique states overlap with "
            f"hard-state list, resuming ordered list at _next_idx={self._next_idx} "
            f"({remaining} remaining)"
        )


class UniformSquareSampler:
    """3D uniform-random sampler for Square-Narrow (NutAssemblySquare): (nut_x, nut_y, nut_yaw).

    Mirrors `SobolSampler`'s interface (`sample_next` → 3-tuple, `to_qpos`,
    `record_state`, `_next_idx`, `collected_points`) so callers that already
    accept `SobolSampler` accept this drop-in. Unlike `SobolSampler` (whose
    points are quasi-random and deterministic given a seed), this sampler
    emits a fresh `np.random.default_rng(seed)`-uniform sample on every
    call — no shared budget, no overlap with the Sobol training prefix.

    Bounds default to the same values as `SobolSampler` to match the
    placement distribution training was collected under.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = SQUARE_NARROW_X_RANGE,
        y_range: tuple[float, float] = SQUARE_NARROW_Y_RANGE,
        yaw_range: tuple[float, float] = SQUARE_NARROW_YAW_RANGE,
        table_offset_z: float = 0.82,
        nut_z_offset: float = 0.07,
        seed: int = 42,
        # Unused, kept for interface parity with SobolSampler.
        batch_size: int | None = None,
        include_boundary: bool | None = None,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.yaw_range = yaw_range
        self.table_offset_z = table_offset_z
        self.nut_z_offset = nut_z_offset
        self._rng = np.random.default_rng(seed)

        # Interface parity with SobolSampler.
        self.collected_points: list[tuple[float, float, float]] = []
        self._next_idx = 0

        print(f"UniformSquareSampler: seed={seed}, x={x_range}, y={y_range}, yaw={yaw_range}")

    def sample_next(self) -> tuple[float, float, float]:
        """Return a fresh uniform-random (x, y, yaw)."""
        x = float(self._rng.uniform(*self.x_range))
        y = float(self._rng.uniform(*self.y_range))
        yaw = float(self._rng.uniform(*self.yaw_range))
        return (x, y, yaw)

    def record_state(self, x: float, y: float, yaw: float) -> None:
        """Record an observed initial state (tracks count, no-op otherwise)."""
        self.collected_points.append((x, y, yaw))
        self._next_idx += 1

    def to_qpos(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Convert (x, y, yaw) to 7D qpos [x, y, z, qw, qx, qy, qz]."""
        z = self.table_offset_z + self.nut_z_offset
        position = np.array([x, y, z])
        quat_wxyz = z_rot_to_quat_wxyz(yaw % (2.0 * np.pi))
        return np.concatenate([position, quat_wxyz])


def sample_square_sobol2d_uniform_x(
    n_states: int,
    *,
    sobol_seed: int = 42,
    x_seed: int = 202605234,
    start_index: int = 0,
    pool_size: int = 1024,
) -> list[dict[str, float | int]]:
    """Square-Narrow starts from a nested 2-D Sobol stream plus uniform x.

    The nut x range is 5 mm wide, so x is a nuisance dimension drawn i.i.d.
    uniform (``np.random.default_rng(x_seed)``), while (y, yaw) come from one
    scrambled 2-D Sobol stream of ``pool_size`` points. Later rounds take later
    ``[start_index, start_index + n_states)`` slices of the same stream; each
    state records its ``sobol_stream_index``.
    """
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    if start_index + n_states > pool_size:
        raise ValueError(
            f"requested Sobol indices [{start_index}, {start_index + n_states}) "
            f"but pool_size={pool_size}"
        )
    if pool_size & (pool_size - 1) != 0:
        raise ValueError(f"pool_size must be a power of two for Sobol, got {pool_size}")

    raw = Sobol(d=2, scramble=True, seed=sobol_seed).random_base2(m=int(np.log2(pool_size)))
    xs = np.random.default_rng(x_seed).uniform(*SQUARE_NARROW_X_RANGE, size=pool_size)
    block = raw[start_index : start_index + n_states]
    x_block = xs[start_index : start_index + n_states]
    y_lo, y_hi = SQUARE_NARROW_Y_RANGE
    yaw_lo, yaw_hi = SQUARE_NARROW_YAW_RANGE
    ys = y_lo + block[:, 0] * (y_hi - y_lo)
    yaws = yaw_lo + block[:, 1] * (yaw_hi - yaw_lo)
    return [
        {
            "nut_x": float(x),
            "nut_y": float(y),
            "nut_yaw": float(yaw),
            "sobol_stream_index": int(start_index + idx),
        }
        for idx, (x, y, yaw) in enumerate(zip(x_block, ys, yaws, strict=True))
    ]
