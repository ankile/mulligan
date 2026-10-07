"""Recording and repair utilities for LeRobot datasets written by the collectors.

Resume-time reconciliation, episode renumbering and parquet consolidation. Training-time
transforms live in :mod:`mulligan.data.transforms`.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def consolidate_episodes_parquet(dataset_root: str | Path) -> None:
    """Consolidate all episodes parquet files into a single file with a unified schema.

    When LeRobot resumes a dataset across multiple sessions, each session writes to
    a new parquet file. If the LeRobot code changed between sessions, these files
    can have different schemas (e.g., one with stats columns, another without),
    causing Dataset.from_parquet to fail when loading.

    This function reads all episode parquet files, concatenates them with a unified
    schema (missing columns filled with NaN), and writes a single consolidated file.

    Call this after dataset.finalize() and before push_to_hub().

    Args:
        dataset_root: Path to the dataset root directory.
    """
    episodes_dir = Path(dataset_root) / "meta" / "episodes"
    if not episodes_dir.exists():
        return

    parquet_files = sorted(episodes_dir.glob("*/*.parquet"))
    if len(parquet_files) <= 1:
        return  # Nothing to consolidate

    logger.info(f"Consolidating {len(parquet_files)} episodes parquet files into one...")

    # Read all files
    dfs = [pd.read_parquet(f) for f in parquet_files]

    # Concat with unified schema (pandas fills missing columns with NaN)
    merged = pd.concat(dfs, ignore_index=True)

    # Verify episode indices are sequential
    expected = list(range(len(merged)))
    actual = sorted(merged["episode_index"].tolist())
    if actual != expected:
        raise ValueError(
            f"Episode indices are not sequential after merge: "
            f"expected {expected[:5]}...{expected[-3:]}, "
            f"got {actual[:5]}...{actual[-3:]}"
        )

    # Rows self-point to the episodes parquet file that holds them; after
    # consolidation every row lives in chunk-000/file-000. Stale pointers break
    # per-episode readers (e.g. lerobot dataset_tools._load_episode_with_stats).
    for col in ("meta/episodes/chunk_index", "meta/episodes/file_index"):
        if col in merged.columns:
            merged[col] = 0

    # Write to the first chunk's file-000
    output_path = episodes_dir / "chunk-000" / "file-000.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    # Delete all other parquet files
    for f in parquet_files:
        if f != output_path:
            f.unlink()
            logger.info(f"  Deleted {f}")

    # Clean up empty directories
    for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
        if chunk_dir != output_path.parent and not any(chunk_dir.iterdir()):
            chunk_dir.rmdir()

    logger.info(
        f"  Consolidated {len(parquet_files)} files -> {output_path} "
        f"({len(merged)} episodes, {len(merged.columns)} columns)"
    )


# ---------------------------------------------------------------------------
# Crash-resume integrity
#
# LeRobot's ``save_episode()`` bumps ``info.json``'s ``total_episodes`` (and
# rewrites ``stats.json``) *eagerly* per episode, but the episode's frame and
# metadata parquet only get a footer at ``finalize()``. A crash in that window
# (e.g. a failed verified robot reset before the next eval rollout) leaves
# ``info.json`` counting an episode whose parquet was never footered. On the
# next resume LeRobot trusts ``total_episodes`` and appends *past* the orphaned
# episode, leaving a permanent gap that later makes
# ``consolidate_episodes_parquet`` fail ("episode indices are not sequential").
#
# ``reconcile_resumed_dataset`` runs before ``LeRobotDataset.resume`` and refuses
# to proceed silently: it rolls the counter (and ``total_frames`` / ``splits`` /
# ``stats.json``) back to the largest *durable contiguous* episode prefix on
# disk, quarantining the un-footered trailing files. It only auto-heals the safe
# trailing-partial shape; a mid-dataset gap fails loud and points at the repair
# tool, because closing a hole means renumbering and that is too destructive to
# do silently on every startup.
# ---------------------------------------------------------------------------


@dataclass
class ResumeReconcileResult:
    """Outcome of :func:`reconcile_resumed_dataset`."""

    healthy: bool
    durable_episodes: int
    info_total_episodes: int
    repaired: bool = False
    quarantined: list[Path] = field(default_factory=list)
    message: str = ""
    # True when NOTHING durable survived and the whole on-disk dataset (incl.
    # info.json) was quarantined. The caller must NOT try to resume -- there is no
    # dataset left to open; it must re-create from scratch.
    reset_empty: bool = False


def _scan_parquet_episode_indices(
    paths: list[Path],
) -> tuple[dict[Path, set[int]], list[Path]]:
    """Read the ``episode_index`` column of each parquet file defensively.

    Returns ``(readable, corrupt)`` where ``readable`` maps each fully-readable
    file to the set of episode indices it holds, and ``corrupt`` lists files that
    raised on read (an un-footered / truncated parquet left by a crash).
    """
    readable: dict[Path, set[int]] = {}
    corrupt: list[Path] = []
    for path in paths:
        try:
            col = pd.read_parquet(path, columns=["episode_index"])["episode_index"]
        except Exception:  # noqa: BLE001 -- any read failure means "not durable"
            corrupt.append(path)
            continue
        readable[path] = {int(v) for v in col.tolist()}
    return readable, corrupt


def _largest_contiguous_prefix(indices: set[int]) -> int:
    """Largest ``K`` such that ``{0, 1, ..., K-1}`` are all present in ``indices``."""
    k = 0
    while k in indices:
        k += 1
    return k


def recompute_aggregate_stats_from_episode_meta(episode_meta: pd.DataFrame) -> dict:
    """Re-derive dataset-level ``stats.json`` from per-episode meta-parquet stats.

    Reconstructs each episode's stats dict from the flattened
    ``stats/<feature>/<stat>`` columns and re-aggregates with LeRobot's own
    :func:`aggregate_stats`, so the result is byte-for-byte what LeRobot would
    have written had exactly these episodes been recorded. Mirrors LeRobot's
    shape contract: ``count`` is ``(1,)`` and image-feature stats are ``(3,1,1)``.
    """
    from lerobot.datasets.compute_stats import aggregate_stats

    stat_cols = [c for c in episode_meta.columns if c.startswith("stats/")]
    if not stat_cols:
        raise ValueError("episode metadata has no stats/* columns to recompute aggregate stats")
    per_episode: list[dict] = []
    for _, row in episode_meta.iterrows():
        stats: dict[str, dict[str, np.ndarray]] = {}
        for col in stat_cols:
            parts = col.split("/")
            if len(parts) != 3:
                raise ValueError(
                    f"unexpected stats column {col!r}; expected 'stats/<feature>/<stat>'"
                )
            _, feature, stat = parts
            value = np.asarray(row[col], dtype=np.float64).ravel()
            if stat == "count":
                value = value.reshape(1)
            elif "image" in feature:
                # Meta parquet stores image stats flat as (3,); LeRobot's stats
                # contract (and _validate_stat_value) requires (3, 1, 1).
                value = value.reshape(3, 1, 1)
            stats.setdefault(feature, {})[stat] = value
        per_episode.append(stats)
    return aggregate_stats(per_episode)


def _parse_chunk_file_indices(mp4_path: Path) -> tuple[int, int]:
    """Parse ``(chunk_index, file_index)`` from a ``.../chunk-XXX/file-YYY.mp4`` path."""
    chunk = int(mp4_path.parent.name.split("-")[1])
    file = int(mp4_path.stem.split("-")[1])
    return chunk, file


def _quarantine_orphaned_videos(root: Path, kept_meta, quarantine) -> None:
    """Quarantine video files past the highest (chunk, file) any KEPT episode uses.

    After a trailing-partial heal, the rolled-back tail's video may live in a
    separate, higher-indexed video file (the per-episode-footer path opens a fresh
    video file per episode). Leaving it risks a write collision when the tail is
    re-collected. Video files at or below the max index a kept episode references
    are NEVER touched, so no kept episode's video is removed. ``kept_meta=None``
    (prefix 0, nothing kept) quarantines every video file.
    """
    videos_dir = root / "videos"
    if not videos_dir.exists():
        return
    for key_dir in sorted(p for p in videos_dir.glob("*") if p.is_dir()):
        key = key_dir.name
        ck_col, fi_col = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
        if kept_meta is not None and ck_col in kept_meta.columns and len(kept_meta):
            max_idx = max((int(c), int(f)) for c, f in zip(kept_meta[ck_col], kept_meta[fi_col]))
        else:
            max_idx = (-1, -1)  # no kept episode references this key -> all orphaned
        for mp4 in sorted(key_dir.glob("*/*.mp4")):
            if _parse_chunk_file_indices(mp4) > max_idx:
                quarantine(mp4)


def _quarantine_streaming_temp_dirs(root: Path, quarantine) -> None:
    """Quarantine stray streaming-encoder temp dirs left by a crashed session."""
    for entry in sorted(root.glob("tmp*")):
        if entry.is_dir() and any(entry.glob("*_streaming.mp4")):
            quarantine(entry)


def reconcile_resumed_dataset(dataset_path: str | Path) -> ResumeReconcileResult:
    """Reconcile ``info.json`` with the durably-footered parquet before resuming.

    Call this *before* ``LeRobotDataset.resume`` on an existing on-disk eval
    dataset. It guarantees ``total_episodes`` never runs ahead of the data that
    can actually be read back, so a crash can never silently orphan episodes.

    Behaviour:
      * Healthy (counter matches a clean contiguous durable set): no-op.
      * Trailing-partial (durable is exactly ``{0..K-1}`` with ``K`` < counter,
        the tail un-footered/absent): auto-heal -- quarantine the trailing
        files, roll ``total_episodes`` / ``total_frames`` / ``splits`` /
        ``stats.json`` back to ``K``. The orphaned tail gets re-collected.
      * Mid-dataset gap (durable has a hole below its max, e.g. ``{0..14,19..29}``):
        raise loudly with the missing indices and the repair-tool command.
    """
    root = Path(dataset_path)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return ResumeReconcileResult(
            healthy=True,
            durable_episodes=0,
            info_total_episodes=0,
            message="no dataset on disk; nothing to reconcile",
        )

    from lerobot.datasets.io_utils import load_info, write_info, write_stats

    info = load_info(root)
    total = int(info.total_episodes)
    if total == 0:
        return ResumeReconcileResult(
            healthy=True, durable_episodes=0, info_total_episodes=0, message="empty dataset"
        )

    meta_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    data_files = sorted((root / "data").glob("*/*.parquet"))
    meta_readable, meta_corrupt = _scan_parquet_episode_indices(meta_files)
    data_readable, data_corrupt = _scan_parquet_episode_indices(data_files)

    durable_meta = set().union(*meta_readable.values()) if meta_readable else set()
    durable_data = set().union(*data_readable.values()) if data_readable else set()
    # An episode is durable only if BOTH its metadata and its frame data are
    # footered and readable.
    durable = durable_meta & durable_data
    prefix = _largest_contiguous_prefix(durable)

    if prefix == total and durable == set(range(total)) and not meta_corrupt and not data_corrupt:
        return ResumeReconcileResult(
            healthy=True,
            durable_episodes=total,
            info_total_episodes=total,
            message=f"healthy: {total} durable contiguous episodes match info.json",
        )

    # Non-contiguous durable set => a mid-dataset gap. Renumbering to close it is
    # destructive; refuse to guess and hand off to the deliberate repair tool.
    if durable != set(range(prefix)):
        max_durable = max(durable)
        missing = sorted(set(range(max_durable + 1)) - durable)
        raise ValueError(
            f"{root}: refusing to resume a dataset with a mid-dataset episode gap. "
            f"info.json reports total_episodes={total} but the durable on-disk episodes "
            f"are non-contiguous: missing {missing[:20]}"
            f"{' (+more)' if len(missing) > 20 else ''} below the highest durable index "
            f"{max_durable}. Closing the hole requires renumbering. Run:\n"
            f"  uv run python -m mulligan.tools.repair_eval_dataset --dataset-path {root}\n"
            "then resume."
        )

    # Clean trailing-partial: durable == {0..prefix-1}, prefix <= total. Every
    # on-disk file must be entirely below `prefix` (keep) or entirely at/above it
    # (quarantine). A readable file that straddles `prefix` would need per-row
    # surgery -- defer that to the repair tool rather than risk it on startup.
    keep_or_quarantine: list[tuple[Path, set[int]]] = [
        *meta_readable.items(),
        *data_readable.items(),
    ]
    straddlers = [path for path, eps in keep_or_quarantine if eps and min(eps) < prefix <= max(eps)]
    if straddlers:
        raise ValueError(
            f"{root}: durable episodes form a clean prefix [0,{prefix}) but parquet file(s) "
            f"{[str(p) for p in straddlers]} hold episodes on both sides of the boundary. "
            "Refusing to do partial-file surgery on startup. Run:\n"
            f"  uv run python -m mulligan.tools.repair_eval_dataset --dataset-path {root}\n"
            "then resume."
        )

    # pid suffix avoids same-second collisions between two quarantine runs.
    # Sibling of the dataset, NOT inside it: LeRobot's push_to_hub uploads the whole
    # dataset folder (ignoring only images/ + videos/), so quarantined corrupt files
    # left inside would be pushed to the Hub. pid suffix avoids same-second collisions.
    quarantine_root = (
        root.parent
        / f"{root.name}.reconcile_quarantine"
        / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    )
    quarantined: list[Path] = []

    def _quarantine(path: Path) -> None:
        dest = quarantine_root / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        path.rename(dest)
        quarantined.append(dest)

    if prefix == 0:
        # Nothing durable survived (e.g. the first session crashed before any
        # finalize bumped a footer onto disk). We must NOT leave a half-dataset that
        # over-counts: writing info.json=0 while meta/episodes is empty BRICKS the
        # next LeRobotDataset.resume (load_episodes finds no parquet -> Hub pull ->
        # 404). Instead quarantine the WHOLE on-disk dataset (incl. info.json/stats)
        # so info.json no longer exists; the caller re-creates from scratch (the
        # whole lost session is re-collected). reset_empty signals that to the caller.
        # Whole LeRobot subtrees move, not just the parquet/video files: the fresh
        # create refuses a root that still holds meta/ (tasks.parquet, empty chunk
        # dirs), data/ or videos/. Eval sidecars at the root stay in place.
        for name in ("meta", "data", "videos", "images"):
            if (root / name).exists():
                _quarantine(root / name)
        _quarantine_streaming_temp_dirs(root, _quarantine)
        message = (
            f"no durable episodes survived (info.json claimed {total}); quarantined the whole "
            f"on-disk dataset ({len(quarantined)} item(s)) under {quarantine_root}. Re-run the "
            "eval to start fresh -- the lost session re-collects from scratch."
        )
        logger.warning("reconcile_resumed_dataset: %s", message)
        print(f"[reconcile] {message}", flush=True)
        return ResumeReconcileResult(
            healthy=False,
            durable_episodes=0,
            info_total_episodes=total,
            repaired=True,
            quarantined=quarantined,
            message=message,
            reset_empty=True,
        )

    # Build the kept-prefix meta BEFORE moving anything (these files all have
    # max < prefix, so they are never in the quarantine set -- no read-after-move).
    prefix_meta = pd.concat(
        [pd.read_parquet(p) for p, eps in meta_readable.items() if eps and max(eps) < prefix],
        ignore_index=True,
    )
    prefix_meta = prefix_meta[prefix_meta["episode_index"] < prefix]
    total_frames = int(prefix_meta["length"].sum())
    new_stats = recompute_aggregate_stats_from_episode_meta(prefix_meta)

    # Readable but empty (0-row) files hold no episode and go with the tail.
    to_quarantine = list(meta_corrupt) + list(data_corrupt)
    to_quarantine += [path for path, eps in keep_or_quarantine if not eps or min(eps) >= prefix]
    for path in to_quarantine:
        _quarantine(path)
    _quarantine_orphaned_videos(root, prefix_meta, _quarantine)
    _quarantine_streaming_temp_dirs(root, _quarantine)

    # Roll back stats.json first, then info.json LAST as the commit point: if a
    # write fails partway, info.json still over-counts and the next resume re-runs
    # this heal cleanly (idempotent) rather than under-counting durable data.
    write_stats(new_stats, root)
    info.total_episodes = prefix
    info.total_frames = total_frames
    info.splits = {"train": f"0:{prefix}"}
    write_info(info, root)

    message = (
        f"healed trailing-partial: rolled total_episodes {total} -> {prefix} "
        f"(total_frames -> {total_frames}); quarantined {len(quarantined)} un-footered/"
        f"orphaned file(s) under {quarantine_root}. The orphaned tail will be re-collected."
    )
    logger.warning("reconcile_resumed_dataset: %s", message)
    print(f"[reconcile] {message}", flush=True)
    return ResumeReconcileResult(
        healthy=False,
        durable_episodes=prefix,
        info_total_episodes=total,
        repaired=True,
        quarantined=quarantined,
        message=message,
    )


@dataclass
class RenumberResult:
    """Outcome of :func:`renumber_dataset_to_durable_contiguous`."""

    index_map: dict[int, int]
    dropped_episodes: list[int]
    total_episodes: int
    total_frames: int


def renumber_dataset_to_durable_contiguous(dataset_path: str | Path) -> RenumberResult:
    """Renumber a gapped dataset's durable episodes to a dense ``0..K-1`` range.

    This is the deliberate repair for a *mid-dataset gap* (a dataset where
    ``total_episodes`` ran ahead of the durably-footered data and a later
    resume appended past the hole). It keeps every episode whose data AND metadata
    are readable, drops the phantom indices, and rebuilds:

      * each data parquet: ``episode_index`` remapped, global ``index`` rebuilt
        contiguously in file order (the on-disk ``index`` can itself be stale --
        the same counter-overrun that caused the gap inflates it -- so it is
        recomputed, never trusted);
      * each episode-meta parquet: ``episode_index`` remapped and
        ``dataset_from_index`` / ``dataset_to_index`` rebuilt contiguously;
      * ``info.json`` (``total_episodes`` / ``total_frames`` / ``splits``);
      * ``stats.json`` (re-aggregated over the kept episodes only);
      * the episode-meta files are then consolidated, which also asserts the
        result is contiguous ``0..K-1``.

    Video bytes and per-episode video pointers are untouched: episodes keep their
    original ``data/chunk_index`` / ``data/file_index`` and video timestamps; only
    their ``episode_index`` label changes.

    Returns the old->new ``index_map`` (callers, e.g. the eval results.json
    remapper, use it to follow the renumbering) plus the dropped phantom indices.
    """
    from lerobot.datasets.io_utils import load_info, write_info, write_stats

    root = Path(dataset_path)
    info = load_info(root)

    data_files = sorted((root / "data").glob("*/*.parquet"))
    meta_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    data_readable, data_corrupt = _scan_parquet_episode_indices(data_files)
    meta_readable, meta_corrupt = _scan_parquet_episode_indices(meta_files)
    if data_corrupt or meta_corrupt:
        raise ValueError(
            f"{root}: refusing to renumber while un-footered parquet files exist "
            f"(data={data_corrupt}, meta={meta_corrupt}). Resume once to let "
            "reconcile_resumed_dataset quarantine them, or quarantine them by hand first."
        )

    data_eps = set().union(*data_readable.values()) if data_readable else set()
    meta_eps = set().union(*meta_readable.values()) if meta_readable else set()
    # In a valid dataset every episode has exactly one meta row and its data frames,
    # so the two episode sets MUST match. A mismatch means an inconsistent / partially
    # renumbered dataset (e.g. an interrupted earlier repair) -- refuse rather than
    # silently renumber the (wrong) intersection.
    if data_eps != meta_eps:
        raise ValueError(
            f"{root}: data and meta episode sets disagree (data-only={sorted(data_eps - meta_eps)[:20]}, "
            f"meta-only={sorted(meta_eps - data_eps)[:20]}). This looks like an interrupted/partial "
            "repair; restore from a backup before renumbering."
        )
    durable = sorted(data_eps & meta_eps)
    if not durable:
        raise ValueError(f"{root}: no durable episodes found to renumber")
    index_map = {old: new for new, old in enumerate(durable)}
    dropped = sorted(set(range(durable[-1] + 1)) - set(durable))
    if not dropped and durable == list(range(len(durable))):
        # Already contiguous: no episode_index renumber needed. But a PRIOR partial
        # repair could have crashed between write_info and write_stats, leaving
        # info.json/stats.json stale while data+meta are already contiguous (so the
        # mixed-set guard won't fire). Re-derive total_frames + stats from the
        # authoritative meta and rewrite info+stats idempotently rather than trusting
        # the on-disk counters (which is what makes a re-run self-healing).
        merged_meta = pd.concat(
            [pd.read_parquet(p) for p in meta_readable], ignore_index=True
        ).sort_values("episode_index")
        total_frames = int(merged_meta["length"].sum())
        info.total_episodes = len(durable)
        info.total_frames = total_frames
        info.splits = {"train": f"0:{len(durable)}"}
        write_info(info, root)
        write_stats(recompute_aggregate_stats_from_episode_meta(merged_meta), root)
        logger.info(
            "%s: already contiguous (%d episodes); reconciled info/stats", root, len(durable)
        )
        return RenumberResult(
            index_map=index_map,
            dropped_episodes=[],
            total_episodes=len(durable),
            total_frames=total_frames,
        )

    # --- Rewrite data parquet: remap episode_index, rebuild global index. ---
    # While rewriting, record each NEW episode's actual frame range/count straight
    # from the rebuilt data. These are the single source of truth for the meta
    # offsets below -- we never re-derive them by independently summing `length`,
    # so a frame permutation or a length/data mismatch cannot pass a totals-only check.
    # Every rewritten frame is built and validated in memory first; nothing on disk
    # changes until all checks pass, so a refused layout leaves the dataset intact.
    running_frame = 0
    data_from: dict[int, int] = {}
    data_to: dict[int, int] = {}
    data_count: dict[int, int] = {}
    new_data: list[tuple[Path, pd.DataFrame | None]] = []
    for path in sorted(data_readable, key=lambda p: min(data_readable[p], default=-1)):
        df = pd.read_parquet(path)
        df = df[df["episode_index"].isin(index_map)].copy()
        if df.empty:
            new_data.append((path, None))
            continue
        df = df.sort_values("index").reset_index(drop=True)
        df["episode_index"] = df["episode_index"].map(index_map).astype(df["episode_index"].dtype)
        df["index"] = np.arange(running_frame, running_frame + len(df), dtype=df["index"].dtype)
        for new_ep, grp in df.groupby("episode_index"):
            new_ep = int(new_ep)
            if new_ep in data_count:
                raise ValueError(
                    f"{root}: new episode {new_ep} spans more than one data file; "
                    "frame layout is not single-file-per-episode contiguous."
                )
            data_from[new_ep] = int(grp["index"].min())
            data_to[new_ep] = int(grp["index"].max()) + 1
            data_count[new_ep] = len(grp)
        running_frame += len(df)
        new_data.append((path, df))
    total_frames = running_frame

    # Validate the data-derived ranges form a contiguous 0..total_frames cover in
    # NEW episode order. This catches any cross-file interleave / permutation that
    # a totals-only check would miss.
    running = 0
    for new_idx in range(len(durable)):
        if new_idx not in data_from:
            raise ValueError(f"{root}: new episode {new_idx} has no frames after renumber")
        if data_from[new_idx] != running:
            raise ValueError(
                f"{root}: non-contiguous frame layout at new episode {new_idx} "
                f"(expected from_index {running}, got {data_from[new_idx]}); refusing to corrupt."
            )
        running = data_to[new_idx]
    if running != total_frames:
        raise ValueError(f"{root}: frame total mismatch ({running} != {total_frames})")

    # --- Rewrite episode-meta parquet: remap episode_index, set offsets from data. ---
    new_meta: list[tuple[Path, pd.DataFrame | None]] = []
    for path in meta_files:
        if path not in meta_readable:
            continue
        mdf = pd.read_parquet(path)
        mdf = mdf[mdf["episode_index"].isin(index_map)].copy()
        if mdf.empty:
            new_meta.append((path, None))
            continue
        mdf["episode_index"] = mdf["episode_index"].map(index_map)
        for _, row in mdf.iterrows():
            new_ep = int(row["episode_index"])
            if int(row["length"]) != data_count[new_ep]:
                raise ValueError(
                    f"{root}: meta length {int(row['length'])} != data frame count "
                    f"{data_count[new_ep]} for new episode {new_ep}; refusing to corrupt."
                )
        mdf["dataset_from_index"] = mdf["episode_index"].map(data_from)
        mdf["dataset_to_index"] = mdf["episode_index"].map(data_to)
        new_meta.append((path, mdf))

    for path, frame in [*new_data, *new_meta]:
        if frame is None:
            path.unlink()
        else:
            frame.to_parquet(path, index=False)

    info.total_episodes = len(durable)
    info.total_frames = total_frames
    info.splits = {"train": f"0:{len(durable)}"}
    write_info(info, root)

    consolidate_episodes_parquet(root)
    # Glob the remaining meta (consolidate normally collapses to one file, but if
    # the original file-000 was emptied it may live elsewhere) rather than assuming
    # chunk-000/file-000.
    merged_meta = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").glob("*/*.parquet"))],
        ignore_index=True,
    )
    write_stats(recompute_aggregate_stats_from_episode_meta(merged_meta), root)

    logger.warning(
        "%s: renumbered %d durable episodes to [0,%d); dropped phantom indices %s; "
        "total_frames -> %d",
        root,
        len(durable),
        len(durable),
        dropped,
        total_frames,
    )
    return RenumberResult(
        index_map=index_map,
        dropped_episodes=dropped,
        total_episodes=len(durable),
        total_frames=total_frames,
    )
