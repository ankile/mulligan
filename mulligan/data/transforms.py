"""
Training-time dataset transforms for LeRobot datasets.

Episode filtering, action remapping, episode/valid-prefix boundary computation and
relative-pose statistics. Recording and repair utilities are in
:mod:`mulligan.data.recording`.

Episode filtering (``filter_episodes``) is controlled by two flags:
- include_failures: If False (default), only include successful episodes
- include_policy_data: If False (default), only include human episodes (for DAgger datasets)

| include_failures | include_policy_data | Non-DAgger Dataset | DAgger Dataset               |
|------------------|---------------------|--------------------|-----------------------------|
| False (default)  | False (default)     | Successful only    | Successful human only       |
| True             | False               | All episodes       | All human episodes          |
| False            | True                | Successful only    | Successful (any source)     |
| True             | True                | All episodes       | All episodes (no filtering) |
"""

import logging
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.constants import DataSource, EpisodeOutcome

logger = logging.getLogger(__name__)


def require_multilerobot_subdatasets(dataset):
    """Return MultiLeRobotDataset sub-datasets, failing loudly if LeRobot changed."""
    if not hasattr(dataset, "_datasets"):
        raise AttributeError(
            "MultiLeRobotDataset is missing private attribute ._datasets. "
            "LeRobot API changed; check the installed lerobot against the pinned commit 0530dd9b "
            "before training with filtered datasets."
        )
    return dataset._datasets


def remove_features_from_lerobot_subdatasets(sub_datasets, excluded_features: set[str]) -> None:
    """Remove feature metadata from LeRobot sub-datasets to avoid decoding excluded videos.

    LeRobot v3.0 uses each sub-dataset's `meta.info["features"]` to decide which
    video columns to decode. Keep this private coupling in one helper so API
    changes fail in one place instead of diverging across training scripts.
    """
    if not excluded_features:
        return
    for sub_dataset in sub_datasets:
        # `meta.info` is a typed `DatasetInfo` in production; dict-style access
        # (`info["features"]`) is deprecated in lerobot 0.5.2 and warns, so read
        # and write the `features` attribute directly. `info.features` is the same
        # mutable dict, so reassigning it is the documented update path.
        features = sub_dataset.meta.info.features
        sub_dataset.meta.info.features = {
            key: value for key, value in features.items() if key not in excluded_features
        }


def _column_is_policy_critical(name: str, protected_prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p) for p in protected_prefixes)


def fill_null_float_columns_with_nan(
    hf_dataset,
    *,
    protected_prefixes: tuple[str, ...] = ("observation.", "action"),
):
    """Replace null float values with NaN in a LeRobot ``hf_dataset``.

    LeRobot's ``hf_transform_to_torch`` runs ``torch.tensor(x)`` over EVERY column of
    an accessed row, so a Python ``None`` raises
    ``RuntimeError: Could not infer dtype of NoneType``. This bites on auxiliary
    columns that are recorded but sometimes empty — e.g. marker_d2 R0's
    ``telemetry.franka.motor_torques_external``, a ``fixed_size_list<float32>[7]``
    whose child elements are all null on un-instrumented frames. The crash fires in
    ``__getitem__`` (and therefore in :func:`filter_episodes`, before training even
    starts), not because of the lerobot fast-forward — the same transform/crash exists
    pre-FF — but because the data carries Python ``None`` where a float is expected.

    This substitutes those nulls with float ``NaN``, which torch and numpy's nan-aware
    aggregations (``np.nanmean`` etc.) handle natively, WITHOUT dropping the column.
    Only FLOATING columns are touched: floating scalars, and list / large_list /
    fixed_size_list of floating (filling null child elements).

    Fail-loud contract: a null in a POLICY-CRITICAL column (name equal to / prefixed by
    any ``protected_prefixes`` entry — ``action`` or ``observation.*`` by default) is
    NOT an unrecorded-aux-signal, it is corrupt supervision, so it RAISES instead of
    being silently NaN-filled. Likewise a whole-cell-null list (an entirely missing row,
    not just empty child elements) raises rather than being guessed at.

    Returns ``(hf_dataset, filled)`` where ``filled`` maps column name -> number of null
    floats replaced. No-op: returns the SAME object and ``{}`` when nothing is null, so
    clean datasets pay only a metadata null-count scan and are never rebuilt.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    from datasets import Dataset
    from lerobot.datasets.io_utils import hf_transform_to_torch

    table = hf_dataset.data
    filled: dict[str, int] = {}

    def _nan_scalar(dtype):
        return pa.scalar(float("nan"), type=dtype)

    def _guard(name: str, dtype) -> None:
        if _column_is_policy_critical(name, protected_prefixes):
            raise ValueError(
                f"policy-critical column {name!r} (type {dtype}) contains null float "
                "values; NaN-filling it would silently corrupt supervision. Fix the "
                "dataset upstream instead."
            )

    for name in table.column_names:
        col = table.column(name)
        dtype = col.type
        is_list = (
            pa.types.is_fixed_size_list(dtype)
            or pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
        )
        if pa.types.is_floating(dtype):
            n_null = col.null_count
            if n_null == 0:
                continue
            _guard(name, dtype)
            new_col = pc.fill_null(col, _nan_scalar(dtype))
            filled[name] = int(n_null)
        elif is_list and pa.types.is_floating(dtype.value_type):
            arr = col.combine_chunks()
            if arr.null_count:
                raise ValueError(
                    f"column {name!r} has {arr.null_count} null LIST cells (whole rows "
                    "missing, not just empty child elements); null-child NaN-fill does not "
                    "cover parent-null lists — investigate the dataset."
                )
            child = arr.values
            child_nulls = child.null_count
            if child_nulls == 0:
                continue
            _guard(name, dtype)
            child_filled = pc.fill_null(child, _nan_scalar(child.type))
            if pa.types.is_fixed_size_list(dtype):
                new_col = pa.FixedSizeListArray.from_arrays(child_filled, dtype.list_size)
            elif pa.types.is_large_list(dtype):
                # large_list uses int64 offsets + a distinct builder (LargeListArray).
                new_col = pa.LargeListArray.from_arrays(arr.offsets, child_filled)
            else:
                new_col = pa.ListArray.from_arrays(arr.offsets, child_filled)
            filled[name] = int(child_nulls)
        else:
            continue
        table = table.set_column(table.schema.get_field_index(name), name, new_col)

    if not filled:
        return hf_dataset, {}

    new_ds = Dataset(table)
    new_ds.set_transform(hf_transform_to_torch)
    return new_ds, filled


def fill_null_floats_in_lerobot_subdatasets(
    sub_datasets,
    *,
    protected_prefixes: tuple[str, ...] = ("observation.", "action"),
) -> dict[str, int]:
    """Apply :func:`fill_null_float_columns_with_nan` to each LeRobot sub-dataset.

    Mutates each sub-dataset's ``reader.hf_dataset`` in place (only when a fill is
    needed) and returns the aggregated ``column -> nulls-filled`` count across all
    sub-datasets. A no-op (returns ``{}``) for clean datamixes.
    """
    total: dict[str, int] = {}
    for sub_dataset in sub_datasets:
        # Read via the public `hf_dataset` property (lazily activates the reader, never None);
        # write the rebuilt table back through `reader.hf_dataset` (the property is read-only).
        new_hf, filled = fill_null_float_columns_with_nan(
            sub_dataset.hf_dataset, protected_prefixes=protected_prefixes
        )
        if filled:
            sub_dataset.reader.hf_dataset = new_hf
            for key, value in filled.items():
                total[key] = total.get(key, 0) + value
    return total


def _validate_2d_shape(arr, n_cols: int, name: str, repo_id: str):
    """Raise ``ValueError`` unless ``arr`` is ``(N, n_cols)``.

    Mirrors the per-column shape guards the action remaps share. For a single-
    width column (``n_cols == 1``) a 1-D ``(N,)`` array is promoted to ``(N, 1)``
    rather than rejected, matching the gripper handling in every remap.
    """
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    if n_cols == 1 and arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] != n_cols:
        raise ValueError(f"{name} must be (N, {n_cols}); got {arr.shape} on {repo_id!r}")
    return arr


def _apply_action_remap(
    dataset,
    *,
    required_columns: tuple[str, ...],
    build_action,
    new_names: list[str],
    new_shape: list[int] | None,
    log_message: str,
) -> None:
    """Engine for a canonical ``action`` column remap.

    The caller supplies (a) the supplementary columns it requires, (b) how to build
    the replacement ``action`` array from the raw arrow columns, (c) the new feature
    ``names``, (d) the new feature ``shape`` (None keeps 7D), and (e) the completion
    log line. The engine owns the loud missing-column and 7D canonical-shape guards,
    the read-via-``with_format(None)`` raw access, the column swap that preserves
    the torch transform + every other column, the per-sub-dataset stats recompute,
    and the multi-dataset stats re-aggregation.

    ``build_action(raw, sub_dataset)`` receives the format-less ``raw`` HF dataset
    plus the sub-dataset (for ``repo_id`` in error messages) and MUST return the
    replacement ``action`` array, validating its own source columns loudly. Missing
    required columns and a non-7D canonical ``action`` feature both raise — there is
    no silent fallback, because a wrong action target is a corrupt experiment.
    """
    import numpy as np
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.io_utils import hf_transform_to_torch

    sub_datasets = require_multilerobot_subdatasets(dataset)

    for sub_dataset in sub_datasets:
        hf = sub_dataset.hf_dataset
        for required in required_columns:
            if required not in hf.column_names:
                raise KeyError(
                    f"action remap requires column {required!r} on sub-dataset "
                    f"{sub_dataset.repo_id!r}; available columns: {sorted(hf.column_names)}"
                )

        # Read the raw (pre-transform) arrow columns as plain python/numpy, build
        # the replacement action, then swap the column value while preserving every
        # other column + the torch-conversion transform.
        raw = hf.with_format(None)
        new_action = np.asarray(build_action(raw, sub_dataset), dtype=np.float32)

        new_hf = raw.remove_columns("action").add_column(
            "action", [row.tolist() for row in new_action]
        )
        new_hf.set_transform(hf_transform_to_torch)
        sub_dataset.reader.hf_dataset = new_hf

        # The canonical action must start at 7D; update names (and shape when the
        # representation grows) so every shape-deriving reader (incl. the policy
        # output head) sees the new layout.
        col_feature = sub_dataset.meta.info.features["action"]
        if tuple(col_feature["shape"]) != (7,):
            raise ValueError(
                f"Expected (7,) canonical 'action' on {sub_dataset.repo_id!r}; "
                f"got shape {col_feature['shape']}"
            )
        if new_shape is not None:
            col_feature["shape"] = list(new_shape)
        col_feature["names"] = list(new_names)

        # Recompute per-sub-dataset stats for the rewritten column from the new value.
        col_stats = get_feature_stats(new_action, axis=0, keepdims=True)
        col_stats = {k: np.squeeze(v, axis=0) if k != "count" else v for k, v in col_stats.items()}
        sub_dataset.meta.stats["action"] = col_stats

    # Re-aggregate the multi-dataset stats so pre/post-processors normalize on the
    # new action statistics.
    dataset.stats = aggregate_stats([sub.meta.stats for sub in sub_datasets])
    logger.info(log_message, len(sub_datasets))


def remap_action_to_position_r6_in_subdatasets(dataset) -> None:
    """Overwrite each sub-dataset's canonical ``action`` column with a 6D-rotation pose.

    Production real-robot data carries a 7D canonical ``action`` =
    ``cartesian_velocity(6) + gripper_velocity(1)`` plus supplementary decomposed
    columns. This replaces the value of the canonical ``action`` column (and only
    that column) with the commanded absolute pose from ``action.cartesian_position``
    (euler) + ``action.gripper_position``, the orientation encoded as the continuous
    6D rotation of Zhou et al. 2019 (the first two columns of the rotation matrix,
    seam-free, unlike euler's ``+/-pi`` wrap). The resulting canonical ``action`` is
    10D::

        [x, y, z (m), r6 (6), gripper_position (1, absolute 0/1)]

    Every downstream reader keys off the literal feature name ``action``, so the
    policy output dim follows the new 10D shape. Used by ``--action-mode relative``
    (the absolute trajectory it then relativizes).

    Fails loudly if the required supplementary columns are missing — there is no
    fallback to velocity or euler, because a silent target swap would be a corrupt
    experiment.
    """
    import numpy as np

    from mulligan.real.policy.rotation6d import R6_DIM, euler_to_r6

    new_dim = 3 + R6_DIM + 1  # 10

    def build_action(raw, sub_dataset):
        cart_pos = _validate_2d_shape(
            raw["action.cartesian_position"], 6, "action.cartesian_position", sub_dataset.repo_id
        )
        grip_pos = _validate_2d_shape(
            raw["action.gripper_position"], 1, "action.gripper_position", sub_dataset.repo_id
        )
        # Split euler pose into translation (x/y/z) and orientation (roll/pitch/yaw),
        # then encode orientation as continuous 6D rotation (seam-free).
        xyz = cart_pos[:, :3]
        euler = cart_pos[:, 3:6]
        r6 = euler_to_r6(euler).astype(np.float32)  # (N, 6)
        if r6.shape != (cart_pos.shape[0], R6_DIM):
            raise RuntimeError(
                f"euler_to_r6 returned {r6.shape}, expected {(cart_pos.shape[0], R6_DIM)} "
                f"on {sub_dataset.repo_id!r}"
            )
        new_action = np.concatenate([xyz, r6, grip_pos], axis=1).astype(np.float32)
        if new_action.shape[1] != new_dim:
            raise RuntimeError(
                f"r6 position action must be {new_dim}D; got {new_action.shape} "
                f"on {sub_dataset.repo_id!r}"
            )
        return new_action

    _apply_action_remap(
        dataset,
        required_columns=("action", "action.cartesian_position", "action.gripper_position"),
        build_action=build_action,
        new_names=["x", "y", "z", "r6_0", "r6_1", "r6_2", "r6_3", "r6_4", "r6_5", "gripper"],
        new_shape=[new_dim],  # shape GROWS 7 -> 10
        log_message=(
            "Remapped canonical 'action' -> cartesian_position_r6 + gripper_position "
            "(10D absolute target, 6D-rotation orientation) across %d sub-dataset(s)."
        ),
    )


def raw_metadata_column(hf_dataset, column: str):
    """Read numeric metadata without invoking LeRobot's full-row transform.

    HF projection preserves selected/reordered row indices. Prebuilt-cache
    metadata adapters already expose plain tensor columns through __getitem__.
    """
    from datasets import Dataset

    if isinstance(hf_dataset, Dataset):
        return hf_dataset.select_columns([column]).with_format(None)[column][:]
    return hf_dataset[column]


def _iter_multidataset_episode_ranges(sub_datasets):
    """Yield ``(global_offset, local_from, local_to, sub_dataset, episode_label)`` per
    episode across a MultiLeRobotDataset's sub-datasets, in global frame order.

    Shared spine for global episode-boundary computation. Reading the materialized
    ``episode_index`` column keeps boundaries aligned with the rows ``__getitem__``
    can actually serve; some real-world split datasets carry stale extra rows in the
    loaded HF table, so the column is capped to the LeRobot dataset length.
    """
    global_offset = 0
    for sub_dataset in sub_datasets:
        if not hasattr(sub_dataset, "__len__"):
            raise TypeError(
                "sub_dataset must define __len__ so global frame boundaries match "
                f"served rows; got {type(sub_dataset).__name__}"
            )
        episode_indices = raw_metadata_column(sub_dataset.hf_dataset, "episode_index")
        if hasattr(episode_indices, "tolist"):
            episode_indices = episode_indices.tolist()

        # Every LeRobotDataset defines __len__; calling it directly lets a broken sub-dataset
        # raise instead of silently substituting the (possibly stale) episode_index row count.
        dataset_len = len(sub_dataset)
        if len(episode_indices) < dataset_len:
            raise RuntimeError(
                "LeRobot sub-dataset has fewer episode_index rows than its reported length "
                f"({len(episode_indices)=}, {dataset_len=})"
            )
        num_frames = dataset_len
        if num_frames == 0:
            continue
        episode_indices = episode_indices[:num_frames]

        current_episode = episode_indices[0]
        start_idx = 0
        for local_idx in range(1, num_frames):
            if episode_indices[local_idx] != current_episode:
                yield global_offset, start_idx, local_idx, sub_dataset, current_episode
                current_episode = episode_indices[local_idx]
                start_idx = local_idx

        yield global_offset, start_idx, num_frames, sub_dataset, current_episode
        global_offset += num_frames


def compute_relative_pose_pertimestep_stats(
    dataset,
    *,
    horizon: int,
    n_obs_steps: int,
    drop_n_last_frames: int,
) -> dict:
    """Compute UMI per-timestep ``(T, 10)`` relative-pose action stats and install them.

    The relative action representation (``action_mode="relative"``) couples the
    anchor pose with each future pose, so its normalization statistics are a
    WINDOWED quantity, not a per-frame column reduction: timestep ``t+k`` carries a
    growing displacement and must be normalized with ITS OWN statistics
    (UMI ``temporally_independent_normalization=True``, so that ``t+1 ≠ t+2``).

    Must run AFTER the absolute-pose remap (for the UMI relative arm that is
    :func:`remap_action_to_position_r6_in_subdatasets` — the COMMANDED
    action.cartesian_position EE-pose 10D; the robot's soft controller means the
    proprio trajectory is the lagged ACHIEVED motion, not the target, so the relative
    targets are sourced from the command) and after ``drop_n_last_frames`` is known. For every valid anchor (honoring
    ``drop_n_last_frames`` AND the ``is_valid`` prefix exactly as the training sampler
    does — see :func:`episode_anchor_exclusive_end`), it builds the
    ``(T, 10)`` absolute window — clamping/padding at the episode end and MASKING
    the padded rows so they cannot pollute the large-``k`` min/max where the relative
    magnitude is biggest — relativizes it against ``window[n_obs_steps-1]``, and
    accumulates per-timestep, per-dim min/max/mean/std over the pooled valid rows of
    ALL sub-datasets. Because per-``k`` valid counts differ (``drop_n_last`` thins the
    high-``k`` anchors and padding masks the tail), the pooled reduction is done
    manually here rather than via ``aggregate_stats`` (whose single per-sub ``count``
    cannot express a per-``k`` count).

    Installs the result as ``dataset.stats["action"]`` (the dict
    ``make_pre_post_processors`` consumes), OVERWRITING the per-frame abs-pose stats
    the remap left there. Returns the stats dict (for tests / logging).

    Fails loudly on a degenerate window set (a timestep with zero valid rows, or a
    non-10D action column) — a silently empty stat would corrupt normalization.
    """
    import numpy as np

    # Single source of truth for the valid-prefix contract (same local-import pattern
    # as compute_multidataset_valid_boundaries).
    from mulligan.real.eval.outcome_results import valid_prefix_length
    from mulligan.real.policy.relative_pose import POSE_DIM, relativize_pose
    from mulligan.real.policy.rotation6d import euler_to_r6

    if horizon < 1 or n_obs_steps < 1 or not (1 <= n_obs_steps <= horizon):
        raise ValueError(
            f"compute_relative_pose_pertimestep_stats needs 1<=n_obs_steps<=horizon; "
            f"got horizon={horizon}, n_obs_steps={n_obs_steps}"
        )
    if drop_n_last_frames < 0:
        raise ValueError(f"drop_n_last_frames must be >=0; got {drop_n_last_frames}")

    sub_datasets = require_multilerobot_subdatasets(dataset)
    anchor_idx = n_obs_steps - 1
    offsets = np.arange(horizon)

    windows: list[np.ndarray] = []  # each (T, 10)
    pads: list[np.ndarray] = []  # each (T,) bool, True = padded (episode-end clamp)
    # Proprio-anchored: per-window PROPRIO anchor pose (6D euler [xyz, rpy]) at the anchor step,
    # from observation.state — NOT the command window[anchor_idx]. Must match
    # RelativePoseActionProcessorStep so the fitted stats align with the batch-time
    # relativization. (rel[0] then encodes the command-vs-proprio lead, not identity.)
    anchor_proprio6: list[np.ndarray] = []  # each (6,)

    for sub in sub_datasets:
        hf = sub.hf_dataset
        action = np.asarray(hf["action"], dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != POSE_DIM:
            raise ValueError(
                f"relative per-timestep stats require a (N, {POSE_DIM}) absolute-pose 'action' "
                f"column on {sub.repo_id!r}; got {action.shape}. Run the proprio-pose-r6 remap first."
            )
        if "observation.state.cartesian_position" not in hf.column_names:
            raise ValueError(
                f"relative per-timestep stats (proprio-anchored) need "
                f"'observation.state.cartesian_position' on {sub.repo_id!r} to source the "
                f"proprio anchor; column is missing."
            )
        obs_cart = np.asarray(hf["observation.state.cartesian_position"], dtype=np.float64)
        if obs_cart.ndim != 2 or obs_cart.shape[1] != 6:
            raise ValueError(
                f"observation.state.cartesian_position on {sub.repo_id!r} must be (N, 6) "
                f"[xyz, rpy]; got {obs_cart.shape}"
            )
        episode_indices = hf["episode_index"]
        if hasattr(episode_indices, "tolist"):
            episode_indices = episode_indices.tolist()
        dataset_len = len(sub)
        episode_indices = episode_indices[:dataset_len]
        if len(episode_indices) < dataset_len:
            raise RuntimeError(
                f"sub-dataset {sub.repo_id!r} has fewer episode_index rows ({len(episode_indices)}) "
                f"than its length ({dataset_len})"
            )

        # Outcome-edited real datasets carry an is_valid column (valid prefix, then a
        # post-outcome retract/reset junk suffix + terminal pad). The training sampler
        # clamps anchors so no supervised action timestep lands on an is_valid==0 frame
        # (compute_multidataset_valid_boundaries + clamp_soft_truncated_anchor_ends in
        # the DP trainer); the stats must honor the SAME boundaries, or they are fit
        # over junk windows the model never trains on.
        is_valid_col: list | None = None
        if "is_valid" in hf.column_names:
            col = hf["is_valid"]
            is_valid_col = col.tolist() if hasattr(col, "tolist") else list(col)
        # Repeated done=1 terminal tails clamp the prefix exactly as the training
        # sampler does (clamp_prefix_to_first_done inside
        # compute_multidataset_valid_boundaries) — the stats windows must stop
        # where training anchors stop.
        done_col = None
        if is_valid_col is not None and "done" in hf.column_names:
            done = np.asarray(hf["done"])
            done_col = done.reshape(len(done), -1)[:, 0].astype(bool)

        # Local per-episode [from, to) boundaries within this sub-dataset.
        ep_bounds: list[tuple[int, int]] = []
        if dataset_len > 0:
            cur = episode_indices[0]
            start = 0
            for i in range(1, dataset_len):
                if episode_indices[i] != cur:
                    ep_bounds.append((start, i))
                    cur = episode_indices[i]
                    start = i
            ep_bounds.append((start, dataset_len))

        # Max forward action offset from the anchor frame (anchor = window index
        # n_obs_steps-1, window spans horizon rows) — same quantity the DP trainer
        # derives from max(action_delta_indices).
        max_forward_action_offset = horizon - n_obs_steps
        for ep_from, ep_to in ep_bounds:
            # Anchor range matches the training sampler exactly: episode end clamped to
            # the is_valid prefix, then episode_anchor_exclusive_end applies drop_n_last
            # and (for soft-truncated episodes) keeps every supervised action timestep
            # off is_valid==0 frames.
            if is_valid_col is None:
                ep_valid_to = ep_to
            else:
                prefix = valid_prefix_length(
                    is_valid_col[ep_from:ep_to], episode_index=int(episode_indices[ep_from])
                )
                if done_col is not None:
                    prefix = clamp_prefix_to_first_done(
                        done_col[ep_from : ep_from + prefix],
                        episode_index=int(episode_indices[ep_from]),
                    )
                ep_valid_to = ep_from + prefix
            anchor_end = episode_anchor_exclusive_end(
                ep_from,
                ep_valid_to,
                ep_to,
                drop_n_last=drop_n_last_frames,
                max_forward_action_offset=max_forward_action_offset,
            )
            # `a` is the window START; the anchor frame is a + anchor_idx, so the last
            # allowed start is anchor_end - 1 - anchor_idx (empty range if the episode
            # is shorter than the required drop).
            for a in range(ep_from, anchor_end - anchor_idx):
                ks = a + offsets
                pad = ks >= ep_to  # rows past the RAW episode end are padded (clamped)
                clamped = np.minimum(ks, ep_to - 1)
                windows.append(action[clamped])  # (T, 10)
                pads.append(pad)
                # Anchor step = the (clamped) window index anchor_idx, i.e. the SAME
                # step the command anchor came from — but the proprio pose there.
                anchor_proprio6.append(obs_cart[clamped[anchor_idx]])  # (6,) euler

    if not windows:
        raise RuntimeError(
            "relative per-timestep stats found no valid anchors across all sub-datasets "
            f"(horizon={horizon}, drop_n_last_frames={drop_n_last_frames}); cannot fit "
            "the relative action normalizer."
        )

    abs_windows = np.stack(windows, axis=0)  # (A, T, 10)
    pad_mask = np.stack(pads, axis=0)  # (A, T)
    # Proprio anchor: build the 10D base pose from the proprio euler pose at each anchor
    # step ([xyz, euler->r6, grip=0]), NOT abs_windows[:, anchor_idx] (the command).
    proprio6 = np.stack(anchor_proprio6, axis=0)  # (A, 6) euler [xyz, rpy]
    base = np.concatenate(
        [
            proprio6[:, :3],
            np.asarray(euler_to_r6(proprio6[:, 3:6]), dtype=np.float64),
            np.zeros((proprio6.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    )[:, None, :]  # (A, 1, 10)
    rel_windows = relativize_pose(abs_windows, base).astype(np.float64)  # (A, T, 10)

    mn = np.empty((horizon, POSE_DIM), dtype=np.float64)
    mx = np.empty((horizon, POSE_DIM), dtype=np.float64)
    mean = np.empty((horizon, POSE_DIM), dtype=np.float64)
    std = np.empty((horizon, POSE_DIM), dtype=np.float64)
    per_t_count = np.empty((horizon,), dtype=np.int64)
    for k in range(horizon):
        valid = ~pad_mask[:, k]
        col = rel_windows[valid, k, :]  # (Nk, 10)
        if col.shape[0] == 0:
            raise RuntimeError(
                f"relative per-timestep stats: timestep k={k} has zero valid (non-padded) "
                f"rows. Every episode is shorter than the anchor+{k} reach after "
                f"drop_n_last_frames={drop_n_last_frames}; the normalizer for that step "
                "would be undefined."
            )
        mn[k] = col.min(axis=0)
        mx[k] = col.max(axis=0)
        mean[k] = col.mean(axis=0)
        std[k] = col.std(axis=0)
        per_t_count[k] = col.shape[0]

    # Degenerate-dim guard (defensive; rarely fires with the proprio anchor). A
    # command-frame anchor would relativize the k=anchor_idx row to EXACTLY identity (zero
    # range). With the proprio anchor the anchor row is rel[0] = command⊖proprio = the
    # (varying) controller lead, so it is normally non-degenerate. This guard is a safety net: any
    # dim that IS still zero-range (e.g. a truly static dim in a tiny dataset) is widened
    # to a UNIT span centered on its constant so MIN_MAX maps it to the well-conditioned
    # mid-range (0) with denom=1 instead of dividing by eps=1e-8. Information-free for a
    # constant dim and the unnormalize round-trip stays exact; it just removes the 1/eps
    # blow-up. Non-degenerate dims are untouched.
    DEGENERATE_FLOOR = 1e-6
    degenerate = (mx - mn) < DEGENERATE_FLOOR
    if degenerate.any():
        center = (mn + mx) / 2.0
        mn = np.where(degenerate, center - 0.5, mn)
        mx = np.where(degenerate, center + 0.5, mx)
        logger.info(
            "  Widened %d degenerate (zero-range) per-timestep action dim(s) to a unit "
            "span (avoids the MIN_MAX 1/eps blow-up on the constant anchor row).",
            int(degenerate.sum()),
        )

    stats = {
        "min": mn.astype(np.float32),
        "max": mx.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        # count must be shape (1,) per lerobot's stat validator; use the smallest
        # per-timestep valid count (the binding sample size for the late steps).
        "count": np.array([int(per_t_count.min())], dtype=np.int64),
    }
    dataset.stats["action"] = stats
    logger.info(
        "Installed UMI per-timestep relative-pose action stats: shape (%d, %d) from %d anchors "
        "(per-step valid counts %d..%d).",
        horizon,
        POSE_DIM,
        abs_windows.shape[0],
        int(per_t_count.min()),
        int(per_t_count.max()),
    )
    return stats


def compute_multidataset_valid_boundaries(
    sub_datasets,
    *,
    stop_at_first_done: bool = True,
) -> tuple[list[int], list[int], list[int], int]:
    """Global episode boundaries with each episode END clamped to its valid prefix.

    Outcome-edited real LeRobot episodes carry an ``is_valid`` column: a leading
    valid run followed by an invalid suffix. Some outcome-edited eval datasets also
    retain a repeated ``done=1`` terminal tail inside the valid run. By default,
    ``stop_at_first_done=True`` preserves the diffusion-policy contract: sampling
    stops after the first terminal frame. IQL passes ``False`` because every valid
    terminal-tail state is an intentional reward/value anchor; only ``is_valid=0``
    excludes a frame from appearing as the current state.
    Sub-datasets WITHOUT an ``is_valid`` column are returned unchanged. The shared
    :func:`mulligan.real.eval.outcome_results.valid_prefix_length` validates the
    valid-prefix-then-invalid-suffix pattern and fails loudly on corruption.

    Returns ``(from_indices, valid_to_indices, raw_to_indices, n_suffix_excluded)``
    where ``n_suffix_excluded`` is the total number of invalid-suffix frames removed
    from anchor eligibility (0 when no dataset carries ``is_valid``).
    """
    # Local import mirrors this module's other `mulligan.real.*` imports and keeps the
    # single source of truth for the valid-prefix contract in outcome_results.
    from mulligan.real.eval.outcome_results import valid_prefix_length

    from_indices: list[int] = []
    valid_to_indices: list[int] = []
    raw_to_indices: list[int] = []
    n_suffix_excluded = 0
    # Cache the (possibly large) is_valid column per sub-dataset so it is read once,
    # not once per episode.
    is_valid_cache: dict[int, list | None] = {}
    done_cache: dict[int, np.ndarray | None] = {}
    for global_offset, local_from, local_to, sub, ep in _iter_multidataset_episode_ranges(
        sub_datasets
    ):
        ep_from = global_offset + local_from
        ep_to_raw = global_offset + local_to
        from_indices.append(ep_from)
        raw_to_indices.append(ep_to_raw)

        key = id(sub)
        if key not in is_valid_cache:
            if "is_valid" in sub.features:
                col = raw_metadata_column(sub.hf_dataset, "is_valid")
                is_valid_cache[key] = col.tolist() if hasattr(col, "tolist") else list(col)
            else:
                is_valid_cache[key] = None
        is_valid_col = is_valid_cache[key]

        if is_valid_col is None:
            prefix = local_to - local_from
        else:
            prefix = valid_prefix_length(is_valid_col[local_from:local_to], episode_index=int(ep))

        if key not in done_cache:
            if "done" in sub.features:
                done = np.asarray(raw_metadata_column(sub.hf_dataset, "done"))
                done_cache[key] = done.reshape(len(done), -1)[:, 0].astype(bool)
            else:
                done_cache[key] = None
        done_col = done_cache[key]
        if stop_at_first_done and is_valid_col is not None and done_col is not None:
            prefix = clamp_prefix_to_first_done(
                done_col[local_from : local_from + prefix], episode_index=int(ep)
            )

        ep_to_valid = ep_from + prefix
        n_suffix_excluded += ep_to_raw - ep_to_valid
        valid_to_indices.append(ep_to_valid)

    return from_indices, valid_to_indices, raw_to_indices, n_suffix_excluded


def clamp_prefix_to_first_done(done_prefix, *, episode_index: int) -> int:
    """Valid-prefix length clamped to end at the FIRST ``done==1`` frame (inclusive).

    Some outcome-edited eval datasets retain a repeated ``done=1`` terminal tail
    inside the valid run; anchors and stats windows must stop at the first
    terminal frame. ``done_prefix`` is the episode's boolean done column already
    sliced to the valid prefix. Single source of truth for this clamp — used by
    :func:`compute_multidataset_valid_boundaries` AND every helper that promises
    "same boundaries as the training sampler" (a locally re-derived prefix
    silently diverges when a clamp is added here). Raises when done returns to
    zero after the first terminal frame: that is corrupted data, not a tail.
    """
    from mulligan.real.eval.outcome_results import first_done_inclusive_length

    return first_done_inclusive_length(done_prefix, episode_index=episode_index)


def episode_anchor_exclusive_end(
    ep_from: int,
    ep_valid_to: int,
    ep_raw_to: int,
    *,
    drop_n_last: int,
    max_forward_action_offset: int,
) -> int:
    """One past the last allowed *anchor* index for a DP-style chunked-action episode.

    An anchor at index ``i`` supervises action timesteps ``i + d`` for every forward
    delta ``d`` in the policy's ``action_delta_indices`` (LeRobot masks only
    ``action_is_pad`` — frames beyond the RAW episode end — never in-episode
    ``is_valid==0`` frames). Collection always stores exactly ONE terminal padding
    frame (``is_valid=0``, action = copy of the last real action; see
    ``mulligan.real.collect.dataset_features.finalize_episode_data``), so:

    * ``ep_raw_to - ep_valid_to <= 1`` (no ``is_valid`` column, or terminal pad
      only): the rule is ``ep_valid_to - drop_n_last``.
    * ``ep_raw_to - ep_valid_to > 1`` (outcome-edited soft truncation left real
      junk frames — retract/reset/hover tails — before the pad): additionally
      require ``anchor + max_forward_action_offset <= ep_valid_to - 1`` so NO
      supervised action timestep lands on an ``is_valid==0`` frame, i.e. drop
      ``max(drop_n_last, max_forward_action_offset)`` from the valid end.

    Never returns below ``ep_from`` (an episode whose valid prefix is shorter than
    the required drop simply contributes no anchors).
    """
    if ep_valid_to > ep_raw_to:
        raise ValueError(
            f"ep_valid_to must not exceed ep_raw_to, got {ep_valid_to=} > {ep_raw_to=}"
        )
    suffix_len = ep_raw_to - ep_valid_to
    drop = drop_n_last if suffix_len <= 1 else max(drop_n_last, max_forward_action_offset)
    return max(ep_from, ep_valid_to - drop)


def clamp_soft_truncated_anchor_ends(
    from_indices: list[int],
    valid_to_indices: list[int],
    raw_to_indices: list[int],
    *,
    drop_n_last: int,
    max_forward_action_offset: int,
) -> tuple[list[int], int]:
    """Sampler ``to`` indices that keep supervised action chunks out of junk suffixes.

    Companion to :func:`compute_multidataset_valid_boundaries` for
    ``EpisodeAwareSampler`` callers: the sampler draws anchors from
    ``[from, to - drop_n_last)`` per episode, so this returns per-episode ``to``
    values such that the anchor range equals
    ``[ep_from, episode_anchor_exclusive_end(...))``.

    Episodes whose anchor end is unchanged (no ``is_valid`` column, terminal-pad-only
    suffix, or ``max_forward_action_offset <= drop_n_last``) return their
    ``valid_to`` UNCHANGED. Only soft-truncated episodes (junk suffix longer than the
    single terminal pad frame) lose the last
    ``max_forward_action_offset - drop_n_last`` tail anchors.

    Returns ``(sampler_to_indices, n_anchors_excluded)`` where ``n_anchors_excluded``
    counts anchors removed relative to the plain ``valid_to - drop_n_last`` rule.
    """
    sampler_to_indices: list[int] = []
    n_anchors_excluded = 0
    for ep_from, ep_valid_to, ep_raw_to in zip(
        from_indices, valid_to_indices, raw_to_indices, strict=True
    ):
        anchor_end = episode_anchor_exclusive_end(
            ep_from,
            ep_valid_to,
            ep_raw_to,
            drop_n_last=drop_n_last,
            max_forward_action_offset=max_forward_action_offset,
        )
        prior_anchor_end = max(ep_from, ep_valid_to - drop_n_last)
        if anchor_end == prior_anchor_end:
            sampler_to_indices.append(ep_valid_to)
        else:
            n_anchors_excluded += prior_anchor_end - anchor_end
            sampler_to_indices.append(anchor_end + drop_n_last)
    return sampler_to_indices, n_anchors_excluded


def _extract_int(val) -> int:
    """Convert tensor/array/scalar to int."""
    if hasattr(val, "item"):
        return int(val.item())
    elif hasattr(val, "__getitem__"):
        return int(val[0])
    else:
        return int(val)


def _check_episode_has_matching_frames(
    temp_dataset: LeRobotDataset,
    from_idx: int,
    to_idx: int,
    need_success_filter: bool,
    need_source_filter: bool,
) -> bool:
    """
    Check if episode has ANY valid frame matching the filter criteria.

    This mirrors how train.py does frame-level filtering:
    - Skip invalid frames (padded final observations with garbage actions)
    - For each frame, check if source==HUMAN (if filtering by source)
    - For each frame, check if success==SUCCESS (if filtering by success)
    - Include episode if ANY valid frame passes BOTH filters

    Args:
        temp_dataset: The LeRobot dataset
        from_idx: Start frame index for this episode
        to_idx: End frame index for this episode (exclusive)
        need_success_filter: Whether to filter by success==SUCCESS
        need_source_filter: Whether to filter by source==HUMAN

    Returns:
        True if episode has at least one valid frame matching all filter criteria
    """
    has_source = "source" in temp_dataset.features
    has_success = "success" in temp_dataset.features
    has_is_valid = "is_valid" in temp_dataset.features

    for frame_idx in range(int(from_idx), int(to_idx)):
        frame = temp_dataset.hf_dataset[frame_idx]

        # Skip invalid frames (padded final observations with garbage actions)
        # This matches train.py behavior
        if has_is_valid:
            is_valid = _extract_int(frame["is_valid"])
            if not is_valid:
                continue  # Skip invalid frame

        # Check source filter (if needed)
        if need_source_filter:
            if not has_source:
                # No source column = all data is human (demo datasets)
                pass
            else:
                source = _extract_int(frame["source"])
                if source != DataSource.HUMAN:
                    continue  # This frame doesn't match, try next

        # Check success filter (if needed)
        if need_success_filter:
            if not has_success:
                # No success column = treat all as successful
                pass
            else:
                success = _extract_int(frame["success"])
                if success != EpisodeOutcome.SUCCESS:
                    continue  # This frame doesn't match, try next

        # Frame passed all filters!
        return True

    # No frame matched all filters
    return False


def filter_episodes(
    repo_id: str,
    root: Path | None,
    include_failures: bool = False,
    include_policy_data: bool = False,
    revision: str | None = None,
) -> tuple[list[int] | None, int]:
    """
    Filter a dataset based on success and source criteria.

    This is the main filtering function that handles all combinations of filtering options.

    Args:
        repo_id: Dataset repository ID
        root: Root directory containing datasets
        include_failures: If True, include failed episodes. Default False (successful only).
        include_policy_data: If True, include policy-generated episodes in DAgger datasets.
                            Default False (human only for DAgger datasets).
        revision: Hub revision LeRobot falls back to for files missing under ``root``.

    Returns:
        Tuple of (filtered_episodes, matching_frame_count):
            - filtered_episodes: List of episode indices to include, or None if no filtering needed
            - matching_frame_count: Number of frames matching the filter criteria

    Filtering matrix:
        | include_failures | include_policy_data | Non-DAgger        | DAgger                    |
        |------------------|---------------------|-------------------|---------------------------|
        | False            | False               | Successful only   | Successful human only     |
        | True             | False               | All episodes      | All human episodes        |
        | False            | True                | Successful only   | Successful (any source)   |
        | True             | True                | All episodes      | All episodes (no filter)  |
    """
    temp_dataset = LeRobotDataset(repo_id, root=root, revision=revision)

    # Substitute null floats (e.g. unrecorded telemetry vectors stored as null-child
    # fixed_size_list<float>) with NaN BEFORE iterating frames: the per-frame reads below
    # go through hf_transform_to_torch, which raises on a Python None. NaN keeps the column
    # while staying torch/numpy-compatible; policy-critical columns still fail loudly.
    new_hf, _null_filled = fill_null_float_columns_with_nan(temp_dataset.reader.hf_dataset)
    if _null_filled:
        temp_dataset.reader.hf_dataset = new_hf
        logger.warning(
            "filter_episodes(%s): NaN-filled null float columns %s", repo_id, _null_filled
        )

    has_source = "source" in temp_dataset.features
    has_success = "success" in temp_dataset.features
    is_dagger = has_source and has_success

    # Determine what filtering we need
    need_success_filter = not include_failures and has_success
    need_source_filter = not include_policy_data and has_source

    # If no filtering needed, return None (use all episodes)
    if not need_success_filter and not need_source_filter:
        filter_desc = "no filtering"
        if include_failures and include_policy_data:
            filter_desc = "all data (no filtering)"
        elif not has_success and not has_source:
            filter_desc = "no success/source columns, using all episodes"
        print(f"  Dataset '{repo_id}': {filter_desc}")
        print(f"  Using all {temp_dataset.meta.total_episodes} episodes")
        # Count all frames (no filtering)
        total_frames = len(temp_dataset.hf_dataset)
        return None, total_frames

    # Build filter description
    filter_parts = []
    if need_success_filter:
        filter_parts.append("successful")
    if need_source_filter:
        filter_parts.append("human")
    filter_desc = " ".join(filter_parts) + " episodes"

    if is_dagger:
        print(f"  Dataset '{repo_id}' is a DAgger dataset, filtering for {filter_desc}...")
    else:
        print(f"  Dataset '{repo_id}': filtering for {filter_desc}...")

    # Iterate through episodes and filter
    # This mirrors train.py: check each frame for BOTH source and success
    filtered_episodes = []

    for ep_idx in range(temp_dataset.meta.total_episodes):
        ep_data = temp_dataset.meta.episodes[ep_idx]
        from_idx = ep_data["dataset_from_index"]
        to_idx = ep_data["dataset_to_index"]

        # Check if ANY frame in this episode matches all filter criteria
        has_matching_frames = _check_episode_has_matching_frames(
            temp_dataset=temp_dataset,
            from_idx=from_idx,
            to_idx=to_idx,
            need_success_filter=need_success_filter,
            need_source_filter=need_source_filter,
        )

        if has_matching_frames:
            filtered_episodes.append(ep_idx)

    # Count matching frames across ALL frames in dataset (not just within episode boundaries)
    # This matches how count_frames.py counts and how data is actually loaded
    total_matching_frames = 0
    for frame_idx in range(len(temp_dataset.hf_dataset)):
        frame = temp_dataset.hf_dataset[frame_idx]

        # Check if frame matches all criteria
        passes_filters = True

        # Check is_valid
        if "is_valid" in temp_dataset.features:
            is_valid = _extract_int(frame["is_valid"])
            if not is_valid:
                passes_filters = False

        # Check source filter
        if passes_filters and need_source_filter:
            if "source" in temp_dataset.features:
                source = _extract_int(frame["source"])
                if source != DataSource.HUMAN:
                    passes_filters = False

        # Check success filter
        if passes_filters and need_success_filter:
            if "success" in temp_dataset.features:
                success = _extract_int(frame["success"])
                if success != EpisodeOutcome.SUCCESS:
                    passes_filters = False

        if passes_filters:
            total_matching_frames += 1

    print(
        f"  Filtered {temp_dataset.meta.total_episodes} episodes → {len(filtered_episodes)} {filter_desc} "
        f"({total_matching_frames} matching frames)"
    )

    if len(filtered_episodes) == 0:
        # Consistently raise errors for all empty filter results
        if need_source_filter and need_success_filter:
            raise ValueError(
                f"Dataset '{repo_id}' has no successful human episodes! "
                "Consider using --include-failures or --include-policy-data flags."
            )
        elif need_success_filter:
            raise ValueError(
                f"Dataset '{repo_id}' has no successful episodes! "
                "Consider using --include-failures flag."
            )
        else:
            raise ValueError(
                f"Dataset '{repo_id}' has no human episodes! "
                "Consider using --include-policy-data flag."
            )

    return filtered_episodes, total_matching_frames
