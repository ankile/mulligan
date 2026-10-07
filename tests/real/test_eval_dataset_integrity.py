"""Tests for crash-resume integrity helpers in mulligan.data.transforms.

These cover the two halves of the eval crash-safety fix:
  * reconcile_resumed_dataset -- the startup guard that keeps info.json's episode
    counter from running ahead of the durably-footered parquet (heals a trailing
    partial, fails loud on a mid-dataset gap).
  * renumber_dataset_to_durable_contiguous -- the deliberate repair that closes a
    mid-dataset gap by renumbering the durable episodes to a dense range.

Fixtures build a minimal on-disk LeRobot v3.0 layout (info.json + per-episode
data and episode-meta parquet, including flattened stats columns) directly, so
no video encoding or robot is needed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lerobot.datasets.io_utils import load_info, load_stats, write_info
from lerobot.datasets.utils import DatasetInfo

from mulligan.data.recording import (
    ResumeReconcileResult,
    _largest_contiguous_prefix,
    _scan_parquet_episode_indices,
    recompute_aggregate_stats_from_episode_meta,
    reconcile_resumed_dataset,
    renumber_dataset_to_durable_contiguous,
)

# Per-episode stats columns mirror LeRobot's flattened layout: one numeric
# feature "action" (2-dim) and one image feature (flat (3,) on disk, (3,1,1) live).
_STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]


def _episode_stats_columns(ep: int, length: int) -> dict:
    cols: dict[str, object] = {}
    rng = np.arange(2, dtype=np.float64) + ep  # deterministic, distinct per episode
    img = np.arange(3, dtype=np.float64) + ep
    for stat in _STAT_KEYS:
        cols[f"stats/action/{stat}"] = [rng] if stat != "count" else [[float(length)]]
        cols[f"stats/observation.images.cam/{stat}"] = (
            [img] if stat != "count" else [[float(length)]]
        )
    return cols


def _make_dataset(
    root: Path,
    episodes: list[tuple[int, int]],  # (episode_index, length)
    *,
    info_total_episodes: int,
    info_total_frames: int,
    data_index_starts: dict[int, int] | None = None,
) -> None:
    """Write a minimal on-disk dataset: one data + one meta parquet per episode."""
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    # meta dataset_from/to_index is the contiguous truth; data `index` may be
    # injected stale via data_index_starts to mimic the counter-overrun bug.
    offset = 0
    for ep, length in episodes:
        meta_from, meta_to = offset, offset + length
        offset += length
        data_start = (data_index_starts or {}).get(ep, meta_from)
        pd.DataFrame(
            {
                "episode_index": [ep] * length,
                "index": list(range(data_start, data_start + length)),
                "action": [[0.0, 0.0]] * length,
            }
        ).to_parquet(root / "data" / "chunk-000" / f"file-{ep:03d}.parquet", index=False)

        meta_cols = {
            "episode_index": [ep],
            "length": [length],
            "tasks": [["t"]],
            "dataset_from_index": [meta_from],
            "dataset_to_index": [meta_to],
            "data/chunk_index": [0],
            "data/file_index": [ep],
            "meta/episodes/chunk_index": [0],
            "meta/episodes/file_index": [ep],
            **_episode_stats_columns(ep, length),
        }
        pd.DataFrame(meta_cols).to_parquet(
            root / "meta" / "episodes" / "chunk-000" / f"file-{ep:03d}.parquet", index=False
        )

    info = DatasetInfo(
        codebase_version="v3.0",
        fps=15,
        features={
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.images.cam": {"dtype": "video", "shape": (3, 1, 1), "names": None},
        },
        total_episodes=info_total_episodes,
        total_frames=info_total_frames,
        splits={"train": f"0:{info_total_episodes}"},
    )
    write_info(info, root)


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


def test_largest_contiguous_prefix() -> None:
    assert _largest_contiguous_prefix(set()) == 0
    assert _largest_contiguous_prefix({0, 1, 2}) == 3
    assert _largest_contiguous_prefix({0, 1, 3}) == 2  # gap at 2
    assert _largest_contiguous_prefix({1, 2, 3}) == 0  # no 0


def test_scan_parquet_episode_indices_flags_corrupt(tmp_path: Path) -> None:
    good = tmp_path / "good.parquet"
    pd.DataFrame({"episode_index": [0, 0, 1]}).to_parquet(good, index=False)
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file")  # un-footered / truncated stand-in
    readable, corrupt = _scan_parquet_episode_indices([good, bad])
    assert readable == {good: {0, 1}}
    assert corrupt == [bad]


def test_recompute_aggregate_stats_matches_lerobot(tmp_path: Path) -> None:
    from lerobot.datasets.compute_stats import aggregate_stats

    _make_dataset(tmp_path, [(0, 10), (1, 20)], info_total_episodes=2, info_total_frames=30)
    meta = pd.concat(
        [
            pd.read_parquet(p)
            for p in (tmp_path / "meta" / "episodes" / "chunk-000").glob("*.parquet")
        ],
        ignore_index=True,
    ).sort_values("episode_index")
    recomputed = recompute_aggregate_stats_from_episode_meta(meta)
    # Build the expected per-episode stats by hand and aggregate the same way.
    per_ep = []
    for ep, length in [(0, 10), (1, 20)]:
        per_ep.append(
            {
                "action": {
                    s: (np.array([float(length)]) if s == "count" else (np.arange(2.0) + ep))
                    for s in _STAT_KEYS
                },
                "observation.images.cam": {
                    s: (
                        np.array([float(length)])
                        if s == "count"
                        else (np.arange(3.0) + ep).reshape(3, 1, 1)
                    )
                    for s in _STAT_KEYS
                },
            }
        )
    expected = aggregate_stats(per_ep)
    assert np.allclose(recomputed["action"]["mean"], expected["action"]["mean"])
    assert recomputed["observation.images.cam"]["mean"].shape == (3, 1, 1)
    assert int(recomputed["action"]["count"][0]) == 30


# --------------------------------------------------------------------------- #
# reconcile_resumed_dataset
# --------------------------------------------------------------------------- #


def test_reconcile_healthy_is_noop(tmp_path: Path) -> None:
    _make_dataset(
        tmp_path, [(0, 10), (1, 10), (2, 10)], info_total_episodes=3, info_total_frames=30
    )
    result = reconcile_resumed_dataset(tmp_path)
    assert isinstance(result, ResumeReconcileResult)
    assert result.healthy and not result.repaired
    assert load_info(tmp_path).total_episodes == 3


def test_reconcile_heals_trailing_partial(tmp_path: Path) -> None:
    # info claims 4 episodes but only 0,1,2 are durable on disk (ep 3 never footered).
    _make_dataset(
        tmp_path, [(0, 10), (1, 10), (2, 10)], info_total_episodes=4, info_total_frames=40
    )
    result = reconcile_resumed_dataset(tmp_path)
    assert result.repaired and not result.healthy
    assert result.durable_episodes == 3
    info = load_info(tmp_path)
    assert info.total_episodes == 3
    assert info.total_frames == 30
    assert info.splits == {"train": "0:3"}
    # stats.json recomputed over the durable 3 episodes.
    stats = load_stats(tmp_path)
    assert int(stats["action"]["count"][0]) == 30


def test_reconcile_quarantines_corrupt_trailing_file(tmp_path: Path) -> None:
    _make_dataset(
        tmp_path, [(0, 10), (1, 10), (2, 10)], info_total_episodes=4, info_total_frames=40
    )
    # A footerless trailing meta+data file for ep 3 (counter bumped, never footered).
    (tmp_path / "data" / "chunk-000" / "file-003.parquet").write_bytes(b"garbage")
    (tmp_path / "meta" / "episodes" / "chunk-000" / "file-003.parquet").write_bytes(b"garbage")
    result = reconcile_resumed_dataset(tmp_path)
    assert result.repaired
    assert result.durable_episodes == 3
    assert len(result.quarantined) == 2
    # The corrupt files were moved out, not left to break the next load.
    assert not (tmp_path / "data" / "chunk-000" / "file-003.parquet").exists()
    assert all(q.exists() for q in result.quarantined)


def test_reconcile_fails_loud_on_mid_gap(tmp_path: Path) -> None:
    # Durable {0,1,3}: a hole at 2 below the max -> must not auto-heal.
    _make_dataset(
        tmp_path, [(0, 10), (1, 10), (3, 10)], info_total_episodes=4, info_total_frames=40
    )
    with pytest.raises(ValueError, match="mid-dataset episode gap"):
        reconcile_resumed_dataset(tmp_path)
    # info.json untouched on the fail-loud path.
    assert load_info(tmp_path).total_episodes == 4


# --------------------------------------------------------------------------- #
# renumber_dataset_to_durable_contiguous
# --------------------------------------------------------------------------- #


def test_renumber_closes_mid_gap_and_rebuilds_indices(tmp_path: Path) -> None:
    # Episodes 0,1 then a hole (2,3 lost) then 4,5 whose data `index` is stale-inflated
    # by the 2 lost episodes' frames (mimics the real counter-overrun bug).
    lost_frames = 10 + 10
    _make_dataset(
        tmp_path,
        [(0, 10), (1, 10), (4, 10), (5, 10)],
        info_total_episodes=6,
        info_total_frames=60,
        data_index_starts={4: 20 + lost_frames, 5: 30 + lost_frames},
    )
    result = renumber_dataset_to_durable_contiguous(tmp_path)
    assert result.index_map == {0: 0, 1: 1, 4: 2, 5: 3}
    assert result.dropped_episodes == [2, 3]
    assert result.total_episodes == 4
    assert result.total_frames == 40

    info = load_info(tmp_path)
    assert info.total_episodes == 4 and info.total_frames == 40

    data = pd.concat(
        [pd.read_parquet(p) for p in (tmp_path / "data" / "chunk-000").glob("*.parquet")],
        ignore_index=True,
    )
    assert sorted(data["episode_index"].unique().tolist()) == [0, 1, 2, 3]
    assert sorted(data["index"].tolist()) == list(range(40))  # rebuilt contiguous

    meta = pd.read_parquet(tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    meta = meta.sort_values("episode_index")
    assert meta["episode_index"].tolist() == [0, 1, 2, 3]
    assert meta["dataset_from_index"].tolist() == [0, 10, 20, 30]
    assert meta["dataset_to_index"].tolist() == [10, 20, 30, 40]


def test_renumber_noop_when_already_contiguous(tmp_path: Path) -> None:
    _make_dataset(tmp_path, [(0, 10), (1, 10)], info_total_episodes=2, info_total_frames=20)
    result = renumber_dataset_to_durable_contiguous(tmp_path)
    assert result.dropped_episodes == []
    assert result.total_episodes == 2


def test_reconcile_resets_when_nothing_durable(tmp_path: Path) -> None:
    # First session crashes before any finalize: info counts 1 episode but the only
    # data+meta parquet are un-footered. durable is empty (prefix 0). This must
    # quarantine the WHOLE dataset (incl. info.json) and signal reset_empty, NOT write
    # info.json=0 (which would brick the next LeRobotDataset.resume: load_episodes
    # finds no parquet -> Hub pull -> 404).
    _make_dataset(tmp_path, [(0, 10)], info_total_episodes=1, info_total_frames=10)
    (tmp_path / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"garbage")
    (tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"garbage")
    result = reconcile_resumed_dataset(tmp_path)
    assert result.repaired and result.durable_episodes == 0
    assert result.reset_empty  # tells the caller to re-create from scratch
    # info.json is quarantined (gone) so the next startup takes the create path,
    # and the broken parquet are moved out, not left to break a load.
    assert not (tmp_path / "meta" / "info.json").exists()
    assert not (tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
    # Re-running on the now-empty dir is a clean no-op (no info.json -> healthy),
    # not a permanent crash loop.
    assert reconcile_resumed_dataset(tmp_path).healthy


def test_renumber_reconciles_stale_info_on_contiguous(tmp_path: Path) -> None:
    # A prior partial repair can leave data+meta already contiguous but info.json /
    # stats.json stale (crash between write_info and write_stats). The "already
    # contiguous" early-return must still re-derive + rewrite info/stats, not trust
    # the stale on-disk counter.
    _make_dataset(tmp_path, [(0, 10), (1, 10)], info_total_episodes=2, info_total_frames=999)
    result = renumber_dataset_to_durable_contiguous(tmp_path)
    assert result.dropped_episodes == []
    assert result.total_frames == 20
    assert load_info(tmp_path).total_frames == 20  # corrected from stale 999
    assert int(load_stats(tmp_path)["action"]["count"][0]) == 20


def test_renumber_refuses_mismatched_data_meta_sets(tmp_path: Path) -> None:
    # An interrupted earlier repair can leave data and meta keyed on different
    # episode numbering. Renumbering the intersection would corrupt -> must refuse.
    _make_dataset(tmp_path, [(0, 10), (1, 10)], info_total_episodes=2, info_total_frames=20)
    (tmp_path / "meta" / "episodes" / "chunk-000" / "file-001.parquet").unlink()
    with pytest.raises(ValueError, match="data and meta episode sets disagree"):
        renumber_dataset_to_durable_contiguous(tmp_path)


def test_renumber_multi_episode_files(tmp_path: Path) -> None:
    # Mimic the real size-batched layout: one data file per *group* of episodes with
    # the gap on a file boundary (fileA={0,1}, fileB={3,4}, ep2 dropped), and the
    # post-gap data `index` stale-inflated by ep2's frames.
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    (tmp_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    lengths = {0: 5, 1: 7, 3: 4, 4: 6}  # ep2 (len 9) lost
    lost = 9
    file_groups = {"file-000": [0, 1], "file-001": [3, 4]}
    # contiguous *meta* offsets (the truth), but data `index` carries the +lost gap.
    meta_off = {0: (0, 5), 1: (5, 12), 3: (12, 16), 4: (16, 22)}
    data_off = {0: 0, 1: 5, 3: 12 + lost, 4: 16 + lost}
    for fname, eps in file_groups.items():
        rows = []
        for ep in eps:
            for k in range(lengths[ep]):
                rows.append({"episode_index": ep, "index": data_off[ep] + k, "action": [0.0, 0.0]})
        pd.DataFrame(rows).to_parquet(
            tmp_path / "data" / "chunk-000" / f"{fname}.parquet", index=False
        )
        mrows = []
        for ep in eps:
            mrows.append(
                {
                    "episode_index": ep,
                    "length": lengths[ep],
                    "tasks": ["t"],
                    "dataset_from_index": meta_off[ep][0],
                    "dataset_to_index": meta_off[ep][1],
                    "data/chunk_index": 0,
                    "data/file_index": int(fname.split("-")[1]),
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": int(fname.split("-")[1]),
                    **{k: v[0] for k, v in _episode_stats_columns(ep, lengths[ep]).items()},
                }
            )
        pd.DataFrame(mrows).to_parquet(
            tmp_path / "meta" / "episodes" / "chunk-000" / f"{fname}.parquet", index=False
        )
    write_info(
        DatasetInfo(
            codebase_version="v3.0",
            fps=15,
            features={
                "action": {"dtype": "float32", "shape": (2,), "names": None},
                "observation.images.cam": {"dtype": "video", "shape": (3, 1, 1), "names": None},
            },
            total_episodes=5,
            total_frames=22 + lost,
            splits={"train": "0:5"},
        ),
        tmp_path,
    )

    result = renumber_dataset_to_durable_contiguous(tmp_path)
    assert result.index_map == {0: 0, 1: 1, 3: 2, 4: 3}
    assert result.dropped_episodes == [2]
    assert result.total_frames == 22

    data = pd.concat(
        [pd.read_parquet(p) for p in sorted((tmp_path / "data" / "chunk-000").glob("*.parquet"))],
        ignore_index=True,
    )
    assert sorted(data["index"].tolist()) == list(range(22))  # rebuilt contiguous
    meta = pd.read_parquet(
        tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).sort_values("episode_index")
    assert meta["episode_index"].tolist() == [0, 1, 2, 3]
    # offsets match the (re-derived-from-data) contiguous lengths 5,7,4,6.
    assert meta["dataset_from_index"].tolist() == [0, 5, 12, 16]
    assert meta["dataset_to_index"].tolist() == [5, 12, 16, 22]


# --------------------------------------------------------------------------- #
# tail truncation (repair_eval_dataset --keep-episodes)
# --------------------------------------------------------------------------- #


def _make_dataset_with_videos(
    root: Path, episodes: list[tuple[int, int]], *, cams: tuple[str, ...] = ("cam_a", "cam_b")
) -> None:
    """Dataset with one data parquet + one mp4 per episode per cam + meta video cols.

    Meta is a single consolidated file (file-000), mirroring a post-consolidate
    real-eval dataset -- the layout the truncation must rewrite in place.
    """
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    for cam in cams:
        (root / "videos" / f"observation.images.{cam}" / "chunk-000").mkdir(parents=True)

    offset, rows = 0, []
    for ep, length in episodes:
        meta_from, meta_to = offset, offset + length
        offset += length
        pd.DataFrame(
            {
                "episode_index": [ep] * length,
                "index": list(range(meta_from, meta_to)),
                "action": [[0.0, 0.0]] * length,
            }
        ).to_parquet(root / "data" / "chunk-000" / f"file-{ep:03d}.parquet", index=False)
        row = {
            "episode_index": [ep],
            "length": [length],
            "tasks": [["t"]],
            "dataset_from_index": [meta_from],
            "dataset_to_index": [meta_to],
            "data/chunk_index": [0],
            "data/file_index": [ep],
            "meta/episodes/chunk_index": [0],
            "meta/episodes/file_index": [0],
            **_episode_stats_columns(ep, length),
        }
        for cam in cams:
            key = f"observation.images.{cam}"
            row[f"videos/{key}/chunk_index"] = [0]
            row[f"videos/{key}/file_index"] = [ep]
            (root / "videos" / key / "chunk-000" / f"file-{ep:03d}.mp4").write_bytes(
                f"mp4-{key}-{ep}".encode()
            )
        rows.append(pd.DataFrame(row))
    pd.concat(rows, ignore_index=True).to_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    write_info(
        DatasetInfo(
            codebase_version="v3.0",
            fps=15,
            features={
                "action": {"dtype": "float32", "shape": (2,), "names": None},
                **{
                    f"observation.images.{cam}": {
                        "dtype": "video",
                        "shape": (3, 1, 1),
                        "names": None,
                    }
                    for cam in cams
                },
            },
            total_episodes=len(episodes),
            total_frames=offset,
            splits={"train": f"0:{len(episodes)}"},
        ),
        root,
    )


def test_truncate_drops_tail_data_meta_and_moves_videos(tmp_path: Path) -> None:
    from mulligan.tools.repair_eval_dataset import _truncate_to_episodes

    root = tmp_path / "ds"
    _make_dataset_with_videos(root, [(0, 5), (1, 7), (2, 4), (3, 2), (4, 3)])
    backup = tmp_path / "ds.pre_repair"
    backup.mkdir()

    dropped = _truncate_to_episodes(root, keep=3, backup_root=backup)
    assert dropped == [3, 4]

    # data: 0,1,2 kept; 3,4 unlinked
    data_files = {p.name for p in (root / "data" / "chunk-000").glob("*.parquet")}
    assert data_files == {"file-000.parquet", "file-001.parquet", "file-002.parquet"}

    # meta: 3 rows, pointers normalized to the single consolidated file-000
    meta = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert sorted(meta["episode_index"].tolist()) == [0, 1, 2]
    assert set(meta["meta/episodes/file_index"].tolist()) == {0}

    # videos: kept files stay live; dropped files MOVED into the backup (recoverable)
    for cam in ("cam_a", "cam_b"):
        live = {
            p.name
            for p in (root / "videos" / f"observation.images.{cam}" / "chunk-000").glob("*.mp4")
        }
        assert live == {"file-000.mp4", "file-001.mp4", "file-002.mp4"}
        moved = backup / "videos" / f"observation.images.{cam}" / "chunk-000"
        assert {p.name for p in moved.glob("*.mp4")} == {"file-003.mp4", "file-004.mp4"}

    # The truncated dataset is now contiguous 0..2 and renumber rebuilds info to match.
    result = renumber_dataset_to_durable_contiguous(root)
    assert result.total_episodes == 3
    assert result.total_frames == 16  # 5 + 7 + 4
    assert load_info(root).total_episodes == 3


def test_drop_episodes_mid_dataset_removes_and_leaves_gap(tmp_path: Path) -> None:
    # Surgical removal of a MIDDLE set (episodes 1 and 3 of 0..4): the survivors are
    # rewritten gapped {0,2,4}, their videos kept, the dropped videos moved aside,
    # and renumber then closes the gap to a dense 0..2.
    from mulligan.tools.repair_eval_dataset import _drop_episodes

    root = tmp_path / "ds"
    _make_dataset_with_videos(root, [(0, 5), (1, 7), (2, 4), (3, 2), (4, 3)])
    backup = tmp_path / "ds.pre_repair"
    backup.mkdir()

    dropped = _drop_episodes(root, {1, 3}, backup_root=backup)
    assert dropped == [1, 3]

    # data: 1,3 unlinked; 0,2,4 survive
    data_files = {p.name for p in (root / "data" / "chunk-000").glob("*.parquet")}
    assert data_files == {"file-000.parquet", "file-002.parquet", "file-004.parquet"}

    # meta: survivors keep their ORIGINAL (gapped) episode_index until renumber
    meta = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert sorted(meta["episode_index"].tolist()) == [0, 2, 4]

    # dropped videos moved to backup; survivors stay live
    for cam in ("cam_a", "cam_b"):
        live = {
            p.name
            for p in (root / "videos" / f"observation.images.{cam}" / "chunk-000").glob("*.mp4")
        }
        assert live == {"file-000.mp4", "file-002.mp4", "file-004.mp4"}
        moved = backup / "videos" / f"observation.images.{cam}" / "chunk-000"
        assert {p.name for p in moved.glob("*.mp4")} == {"file-001.mp4", "file-003.mp4"}

    # renumber closes the gap {0,2,4} -> dense 0..2 and the index_map follows.
    result = renumber_dataset_to_durable_contiguous(root)
    assert result.index_map == {0: 0, 2: 1, 4: 2}
    assert result.total_episodes == 3
    assert result.total_frames == 12  # 5 + 4 + 3
    assert load_info(root).total_episodes == 3


def test_drop_episodes_refuses_to_drop_all(tmp_path: Path) -> None:
    from mulligan.tools.repair_eval_dataset import _drop_episodes

    root = tmp_path / "ds"
    _make_dataset_with_videos(root, [(0, 5), (1, 7)])
    backup = tmp_path / "ds.pre_repair"
    backup.mkdir()
    with pytest.raises(ValueError, match="refusing to drop all"):
        _drop_episodes(root, {0, 1}, backup_root=backup)


def test_drop_episodes_rejects_out_of_range(tmp_path: Path) -> None:
    from mulligan.tools.repair_eval_dataset import _drop_episodes

    root = tmp_path / "ds"
    _make_dataset_with_videos(root, [(0, 5), (1, 7)])
    backup = tmp_path / "ds.pre_repair"
    backup.mkdir()
    with pytest.raises(ValueError, match="out of range"):
        _drop_episodes(root, {1, 9}, backup_root=backup)


def test_episodes_for_rounds_maps_via_results(tmp_path: Path) -> None:
    from mulligan.tools.repair_eval_dataset import _episodes_for_rounds

    results = tmp_path / "results.json"
    rollouts = [
        {"round": r // 2 + 1, "policy_id": r % 2, "episode_index": r}
        for r in range(8)  # rounds 1,1,2,2,3,3,4,4 -> eps 0..7
    ]
    results.write_text(json.dumps({"rollouts": rollouts}))
    # rounds 2 and 4 -> their recorded episodes {2,3} and {6,7}
    assert _episodes_for_rounds(results, {2, 4}) == {2, 3, 6, 7}
    # a round with no records fails loud rather than dropping nothing
    with pytest.raises(ValueError, match="have no rollout records"):
        _episodes_for_rounds(results, {99})


def test_truncate_refuses_to_delete_shared_data_file(tmp_path: Path) -> None:
    # If a dropped episode's data file is shared by a kept episode (NOT the
    # one-file-per-episode layout), refuse rather than delete a kept episode's data.
    from mulligan.tools.repair_eval_dataset import _truncate_to_episodes

    root = tmp_path / "ds"
    _make_dataset_with_videos(root, [(0, 5), (1, 7), (2, 4)])
    # Point episode 2's data at episode 1's file (simulate a shared multi-ep file).
    meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta = pd.read_parquet(meta_path)
    meta.loc[meta["episode_index"] == 2, "data/file_index"] = 1
    meta.to_parquet(meta_path, index=False)
    backup = tmp_path / "ds.pre_repair"
    backup.mkdir()
    with pytest.raises(ValueError, match="shares data file"):
        _truncate_to_episodes(root, keep=2, backup_root=backup)


def test_remap_results_for_truncation(tmp_path: Path) -> None:
    from mulligan.tools.repair_eval_dataset import _remap_results_json

    results = tmp_path / "results.json"
    rollouts = [
        {"round": r // 2 + 1, "policy_id": r % 2, "outcome": "success", "episode_index": r}
        for r in range(6)  # episodes 0..5 -> rounds 1,1,2,2,3,3
    ]
    results.write_text(
        json.dumps(
            {
                "rollouts": rollouts,
                "round_plans": [{"round": n} for n in range(1, 51)],
                "summary": [
                    {"policy_id": 0, "num_rounds": 3, "successes": 3},
                    {"policy_id": 1, "num_rounds": 3, "successes": 3},
                ],
                "session_note": "kept as-is",
            }
        )
    )
    # Keep episodes 0..3 (rounds 1,2); drop 4,5 (round 3). identity map for survivors.
    report = _remap_results_json(results, {0: 0, 1: 1, 2: 2, 3: 3})
    assert report["kept"] == 4 and report["dropped"] == 2
    assert report["reopened_rounds"] == [3]
    data = json.loads(results.read_text())
    assert [r["episode_index"] for r in data["rollouts"]] == [0, 1, 2, 3]
    assert len(data["round_plans"]) == 50  # plans untouched (deterministic from seed)
    assert data["session_note"] == "kept as-is"  # other top-level fields untouched


# --------------------------------------------------------------------------- #
# repair paths
# --------------------------------------------------------------------------- #


def _write_empty_like(src: Path, dest: Path) -> None:
    pd.read_parquet(src).iloc[:0].to_parquet(dest, index=False)


def test_reconcile_reset_leaves_root_creatable(tmp_path: Path) -> None:
    # The reset_empty heal tells the operator to re-run the eval, which creates the
    # dataset fresh. Leftover meta/ (tasks.parquet, empty chunk dirs), data/ and
    # videos/ dirs made that create refuse the root.
    from mulligan.real.eval.common import _prepare_new_dataset_root

    root = tmp_path / "ds"
    _make_dataset(root, [(0, 10)], info_total_episodes=1, info_total_frames=10)
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"garbage")
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"garbage")
    pd.DataFrame({"task_index": [0]}, index=["t"]).to_parquet(root / "meta" / "tasks.parquet")
    (root / "videos" / "observation.images.cam" / "chunk-000").mkdir(parents=True)
    (root / "results.json").write_text("{}")
    (root / "chunk_info").mkdir()

    result = reconcile_resumed_dataset(root)
    assert result.reset_empty
    assert sorted(p.name for p in root.iterdir()) == ["chunk_info", "results.json"]
    backup = _prepare_new_dataset_root(root)
    assert backup is not None and (backup / "results.json").exists()


def test_reconcile_heals_with_empty_readable_parquet(tmp_path: Path) -> None:
    # A footered 0-row parquet is readable with no episodes; the heal must not call
    # max()/min() on its empty episode set.
    _make_dataset(
        tmp_path, [(0, 10), (1, 10), (2, 10)], info_total_episodes=4, info_total_frames=40
    )
    meta_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    data_dir = tmp_path / "data" / "chunk-000"
    _write_empty_like(meta_dir / "file-000.parquet", meta_dir / "file-003.parquet")
    _write_empty_like(data_dir / "file-000.parquet", data_dir / "file-003.parquet")
    result = reconcile_resumed_dataset(tmp_path)
    assert result.repaired and result.durable_episodes == 3
    assert load_info(tmp_path).total_episodes == 3
    assert not (meta_dir / "file-003.parquet").exists()
    assert not (data_dir / "file-003.parquet").exists()


def test_renumber_tolerates_empty_readable_data_parquet(tmp_path: Path) -> None:
    _make_dataset(tmp_path, [(0, 10), (2, 10)], info_total_episodes=3, info_total_frames=30)
    data_dir = tmp_path / "data" / "chunk-000"
    _write_empty_like(data_dir / "file-000.parquet", data_dir / "file-009.parquet")
    result = renumber_dataset_to_durable_contiguous(tmp_path)
    assert result.index_map == {0: 0, 2: 1}
    assert not (data_dir / "file-009.parquet").exists()


def test_renumber_refusal_leaves_files_untouched(tmp_path: Path) -> None:
    # Gap at 1; one data file holds episodes {0, 3} and another holds {2}. The
    # renumbered layout is non-contiguous, so renumber refuses -- and must do so
    # before rewriting any parquet, so data and meta keep the same numbering.
    _make_dataset(tmp_path, [(0, 5), (2, 5), (3, 5)], info_total_episodes=4, info_total_frames=15)
    data_dir = tmp_path / "data" / "chunk-000"
    merged = pd.concat(
        [
            pd.read_parquet(data_dir / "file-000.parquet"),
            pd.read_parquet(data_dir / "file-003.parquet"),
        ],
        ignore_index=True,
    )
    merged.to_parquet(data_dir / "file-000.parquet", index=False)
    (data_dir / "file-003.parquet").unlink()
    before = {p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}

    with pytest.raises(ValueError, match="non-contiguous frame layout"):
        renumber_dataset_to_durable_contiguous(tmp_path)
    after = {p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    assert after == before
