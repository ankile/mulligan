"""Repair a real-eval LeRobot dataset: mid-dataset episode gap, junk tail, or
surgical removal of specific rounds/episodes.

Three failure modes, three modes of this tool.

**Gap repair (default).** ``save_episode`` bumps ``info.json``'s episode counter
eagerly, so a crash before the next finalize leaves the counter ahead of the
durably-footered parquet and loses the un-footered episodes. The result is a gap
(e.g. episodes ``15..18`` missing while ``0..14`` and ``19..29`` are present),
which fails in ``consolidate_episodes_parquet`` and is refused by
``reconcile_resumed_dataset`` on resume. The default mode keeps every episode
whose data AND metadata are durably readable, renumbers them to a dense
``0..K-1`` range (rebuilding global frame indices, per-episode frame
offsets, ``info.json``, ``stats.json``, and the episode-meta files), and remaps
the eval ``results.json`` rollout records to match -- dropping the records for
the lost episodes so resume re-collects exactly those rounds and nothing else.

**Tail truncation (``--keep-episodes N``).** When the rollout loop ran on but the
episodes are *garbage* -- e.g. a failed scene reset that the operator didn't
catch, so the eval auto-advanced through manifest states with no real rollout --
the tail episodes are durably footered but invalid. This mode physically removes
every episode ``>= N`` (its data parquet + per-episode video files), rewrites the
episode-meta to keep only ``0..N-1``, rebuilds ``info.json`` / ``stats.json`` via
the same already-contiguous path as the gap repair, and drops the corresponding
``results.json`` rollout records so resume restarts at round ``N+1``. Removed
videos are MOVED into the backup dir (they are not otherwise copied); removed
data parquet are already covered by the backup copy.

**Surgical removal (``--drop-rounds`` / ``--drop-episodes``).** When *specific*
rounds in the middle of an otherwise-good eval are invalid -- e.g. a silent robot
fault that the protocol logged as ordinary failures -- or an unreferenced
restart-orphan episode needs purging, this mode physically removes exactly those
episodes (data parquet + per-episode videos), renumbers the survivors dense
(closing the resulting gap), rebuilds ``info.json`` / ``stats.json``, and drops
the removed episodes' ``results.json`` rollout records. ``--drop-rounds`` resolves
round numbers to episodes via results.json's recorded mapping;
``--drop-episodes`` takes explicit indices (use it for an orphan that has no
record). The operator then resumes the eval with ``--rerun-incomplete-rounds``
(its designed-for "redo a surgically-removed bad rollout in place" path), which
re-presents exactly the removed rounds with their preserved deterministic A/B
plans.

Usage::

    uv run python -m mulligan.tools.repair_eval_dataset --dataset-path ./data/<name>
    uv run python -m mulligan.tools.repair_eval_dataset --dataset-path ./data/<name> --results-path ./custom_results.json
    uv run python -m mulligan.tools.repair_eval_dataset --dataset-path ./data/<name> --keep-episodes 8
    uv run python -m mulligan.tools.repair_eval_dataset --dataset-path ./data/<name> --drop-rounds 20,21,22 --drop-episodes 44
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from mulligan.data.recording import renumber_dataset_to_durable_contiguous


def _backup(dataset_path: Path) -> Path:
    """Copy the parts the repair rewrites (meta/, data/, results.json) aside first.

    The renumber rewrites data + meta parquet IN PLACE and is not recoverable if it
    crashes partway, so snapshot the (small, video-excluded) mutated surface before
    touching anything. Videos are not copied -- they are never modified.
    """
    # Sibling of the dataset, NOT inside it: LeRobot's push uploads the whole
    # dataset folder, so a backup placed inside would be pushed to the Hub.
    backup_root = (
        dataset_path.parent / f"{dataset_path.name}.pre_repair_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    backup_root.mkdir(parents=True)
    for rel in ("meta", "data"):
        src = dataset_path / rel
        if src.exists():
            shutil.copytree(src, backup_root / rel)
    results = dataset_path / "results.json"
    if results.exists():
        shutil.copy2(results, backup_root / "results.json")
    print(f"Backed up meta/ + data/ + results.json to {backup_root}")
    return backup_root


def _truncate_to_episodes(dataset_path: Path, keep: int, backup_root: Path) -> list[int]:
    """Tail-truncation: physically remove every episode ``>= keep``.

    Thin wrapper over :func:`_drop_episodes` -- the tail case is just dropping the
    set ``{keep, keep+1, ..., total-1}``. Validates ``keep`` against the episode
    count first so the out-of-range / nothing-to-do messages stay specific.
    """
    import pandas as pd

    meta_files = sorted((dataset_path / "meta" / "episodes").glob("*/*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"{dataset_path}: no meta/episodes parquet to truncate")
    total = sum(len(pd.read_parquet(p, columns=["episode_index"])) for p in meta_files)
    if not 0 < keep <= total:
        raise ValueError(f"--keep-episodes={keep} out of range for {total} episodes")
    if keep == total:
        print(f"--keep-episodes={keep} == total episodes; nothing to truncate.")
        return []
    return _drop_episodes(dataset_path, set(range(keep, total)), backup_root)


def _drop_episodes(dataset_path: Path, drop_eps: set[int], backup_root: Path) -> list[int]:
    """Physically remove an arbitrary set of episodes; return the dropped indices.

    Generalizes tail-truncation to a *surgical mid-dataset removal* (e.g. specific
    rounds that hit a silent robot fault, or an unreferenced restart-orphan
    episode). Drops each dropped episode's OWN data parquet and per-episode video
    files (looked up by the ``(chunk_index, file_index)`` recorded in its meta
    row), then rewrites the episode-meta to keep the survivors with their ORIGINAL
    ``episode_index`` -- which is now *gapped*. A dropped file whose ``(chunk,
    file)`` is also referenced by a kept episode is refused rather than deleted
    (guards the not-one-file-per-episode layout). Video files are MOVED into
    *backup_root* (preserving their relative path) so the removal is reversible;
    data parquet are unlinked because the caller's backup already holds a full
    ``data/`` copy.

    The caller MUST have taken a backup first (the meta rewrite is in place).
    ``info.json`` / ``stats.json`` are NOT touched here -- the caller reruns
    :func:`renumber_dataset_to_durable_contiguous`, whose gap path renumbers the
    survivors dense and rebuilds info/stats (the tail case lands on its already-
    contiguous early-return instead). The returned ``index_map`` follows the
    renumber, and the results.json remap drops exactly the removed episodes'
    rollout records so resume re-collects those rounds (with
    ``--rerun-incomplete-rounds`` for a mid-dataset removal).
    """
    import pandas as pd

    root = dataset_path
    meta_files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"{root}: no meta/episodes parquet to drop from")
    df = (
        pd.concat([pd.read_parquet(p) for p in meta_files], ignore_index=True)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )
    total = len(df)
    eps = [int(e) for e in df["episode_index"].tolist()]
    if eps != list(range(total)):
        raise ValueError(
            f"{root}: meta episode_index is not contiguous 0..{total - 1} (got {eps}); "
            "run the default gap repair first."
        )
    if not drop_eps:
        print("No episodes to drop; nothing to do.")
        return []
    out_of_range = sorted(e for e in drop_eps if not 0 <= e < total)
    if out_of_range:
        raise ValueError(f"{root}: drop episodes {out_of_range} out of range for {total} episodes")
    if len(drop_eps) >= total:
        raise ValueError(
            f"{root}: refusing to drop all {total} episodes (drop set {sorted(drop_eps)})"
        )

    kept = df[~df["episode_index"].isin(drop_eps)].copy()
    dropped = df[df["episode_index"].isin(drop_eps)].copy()

    video_keys = [
        c[len("videos/") : -len("/chunk_index")]
        for c in df.columns
        if c.startswith("videos/") and c.endswith("/chunk_index")
    ]
    kept_data = {
        (int(r["data/chunk_index"]), int(r["data/file_index"])) for _, r in kept.iterrows()
    }
    kept_video = {
        k: {
            (int(r[f"videos/{k}/chunk_index"]), int(r[f"videos/{k}/file_index"]))
            for _, r in kept.iterrows()
        }
        for k in video_keys
    }

    for _, row in dropped.iterrows():
        ep = int(row["episode_index"])
        dci, dfi = int(row["data/chunk_index"]), int(row["data/file_index"])
        if (dci, dfi) in kept_data:
            raise ValueError(
                f"{root}: dropped episode {ep} shares data file (chunk {dci}, file {dfi}) "
                "with a kept episode; refusing to delete shared data."
            )
        dpath = root / "data" / f"chunk-{dci:03d}" / f"file-{dfi:03d}.parquet"
        if dpath.exists():
            dpath.unlink()  # already preserved in the caller's full data/ backup copy
        for k in video_keys:
            vci, vfi = int(row[f"videos/{k}/chunk_index"]), int(row[f"videos/{k}/file_index"])
            if (vci, vfi) in kept_video[k]:
                raise ValueError(
                    f"{root}: dropped episode {ep} shares video {k} (chunk {vci}, file {vfi}) "
                    "with a kept episode; refusing to delete shared video."
                )
            vpath = root / "videos" / k / f"chunk-{vci:03d}" / f"file-{vfi:03d}.mp4"
            if vpath.exists():
                dest = backup_root / vpath.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(vpath), str(dest))  # videos are not in the meta/data backup

    # Rewrite the episode-meta to the kept (possibly gapped) survivors as a single
    # consolidated file. The kept rows keep their original episode_index -- the
    # caller's renumber closes any gap -- but their self-referential meta pointers
    # may name a higher file than the single file-000 we now write; normalize them
    # to (0, 0) so they point at the consolidated file.
    for col, val in (("meta/episodes/chunk_index", 0), ("meta/episodes/file_index", 0)):
        if col in kept.columns:
            kept[col] = val
    for p in meta_files:
        p.unlink()
    out_path = meta_files[0].parent / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out_path, index=False)

    dropped_eps = sorted(int(e) for e in dropped["episode_index"])
    print(
        f"Removed {len(dropped_eps)} episode(s) {dropped_eps}, kept {len(kept)} "
        f"(videos moved to {backup_root})."
    )
    return dropped_eps


def _episodes_for_rounds(results_path: Path, rounds: set[int]) -> set[int]:
    """Resolve a set of round numbers to the episode indices recorded for them.

    Reads results.json's ``rollouts`` (the canonical round->episode mapping) and
    returns every ``episode_index`` whose record's round is in *rounds*. Raises if
    a requested round has no records (typo / already-removed) so a mistyped round
    fails loud instead of silently dropping nothing. Note this maps only *recorded*
    episodes -- an unreferenced restart-orphan in one of those rounds is NOT caught
    here; pass it explicitly via ``--drop-episodes``.
    """
    if not results_path.exists():
        raise FileNotFoundError(
            f"--drop-rounds needs the round->episode mapping in {results_path}, which is missing."
        )
    data = json.loads(results_path.read_text())
    by_round: dict[int, set[int]] = {}
    for rec in data.get("rollouts", []):
        rn = int(rec.get("round", rec.get("round_num")))
        by_round.setdefault(rn, set()).add(int(rec["episode_index"]))
    missing = sorted(r for r in rounds if r not in by_round)
    if missing:
        raise ValueError(
            f"--drop-rounds: round(s) {missing} have no rollout records in {results_path} "
            f"(present rounds: {sorted(by_round)[:5]}...{sorted(by_round)[-5:]})"
        )
    eps: set[int] = set()
    for r in rounds:
        eps |= by_round[r]
    return eps


def _remap_results_json(results_path: Path, index_map: dict[int, int]) -> dict:
    """Drop dropped-episode rollout records, remap survivors, recompute summary.

    Returns a small report dict ``{"kept": int, "dropped": int, "reopened_rounds": [..]}``.
    Every other top-level field is left untouched.
    """
    data = json.loads(results_path.read_text())
    rollouts = data.get("rollouts", [])
    kept, dropped_rounds = [], set()
    n_dropped = 0
    for rec in rollouts:
        old_idx = int(rec["episode_index"])
        if old_idx in index_map:
            rec = dict(rec)
            rec["episode_index"] = index_map[old_idx]
            kept.append(rec)
        else:
            n_dropped += 1
            dropped_rounds.add(int(rec.get("round", rec.get("round_num"))))
    data["rollouts"] = kept

    # Recompute the per-policy display summary from the surviving rollouts. The
    # eval also regenerates this on its next save; recomputing here just avoids
    # leaving a stale/misleading file in the meantime. Resume itself only reads
    # "rollouts", so this is display-only.
    summary = data.get("summary")
    if summary:
        by_policy: dict[int, list[dict]] = {}
        for rec in kept:
            by_policy.setdefault(int(rec["policy_id"]), []).append(rec)
        for entry in summary:
            recs = by_policy.get(int(entry["policy_id"]), [])
            num_rounds = len(recs)
            successes = sum(1 for r in recs if r.get("outcome") == "success")
            entry["num_rounds"] = num_rounds
            entry["successes"] = successes
            entry["failures"] = num_rounds - successes
            entry["success_rate"] = (successes / num_rounds) if num_rounds else 0.0

    results_path.write_text(json.dumps(data, indent=2))
    return {
        "kept": len(kept),
        "dropped": n_dropped,
        "reopened_rounds": sorted(dropped_rounds),
    }


def _verify(dataset_path: Path, expected_episodes: int, expected_frames: int) -> None:
    """Load the repaired dataset and assert it is internally consistent."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="repair/verify", root=str(dataset_path))
    if ds.meta.total_episodes != expected_episodes:
        raise RuntimeError(
            f"verify: total_episodes={ds.meta.total_episodes} != expected {expected_episodes}"
        )
    if ds.meta.total_frames != expected_frames:
        raise RuntimeError(
            f"verify: total_frames={ds.meta.total_frames} != expected {expected_frames}"
        )
    if len(ds) != expected_frames:
        raise RuntimeError(f"verify: len(ds)={len(ds)} != expected_frames {expected_frames}")
    # Touch the first and last frame to confirm the parquet/index wiring reads back.
    _ = ds[0]
    _ = ds[len(ds) - 1]
    print(
        f"[verify] OK: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, "
        "first/last frame readable."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=Path,
        help="Root directory of the eval dataset to repair (the dir containing meta/, data/).",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Path to results.json (default: <dataset-path>/results.json). Skipped if absent.",
    )
    parser.add_argument(
        "--keep-episodes",
        type=int,
        default=None,
        help=(
            "Tail-truncation mode: physically drop every episode >= this count "
            "(junk auto-advanced rollouts after a missed scene reset), keeping 0..N-1, "
            "and reopen the corresponding rounds in results.json."
        ),
    )
    parser.add_argument(
        "--drop-rounds",
        type=str,
        default=None,
        help=(
            "Surgical mid-dataset removal: comma-separated round numbers (e.g. 20,21,22) "
            "whose recorded episodes to physically remove. Resolved to episode indices via "
            "results.json; their records are dropped so resume re-collects exactly those "
            "rounds (operator resumes with --rerun-incomplete-rounds). Combine with "
            "--drop-episodes for unreferenced orphans. Mutually exclusive with --keep-episodes."
        ),
    )
    parser.add_argument(
        "--drop-episodes",
        type=str,
        default=None,
        help=(
            "Surgical mid-dataset removal: comma-separated explicit episode indices to "
            "physically remove (e.g. an unreferenced restart-orphan). Unioned with any "
            "--drop-rounds. Mutually exclusive with --keep-episodes."
        ),
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    print(f"Repairing dataset: {dataset_path}")

    surgical = args.drop_rounds is not None or args.drop_episodes is not None
    if args.keep_episodes is not None and surgical:
        parser.error(
            "--keep-episodes (tail) is mutually exclusive with --drop-rounds/--drop-episodes."
        )

    # Resolve the surgical drop set BEFORE the backup mutates anything (results.json
    # still holds the original round->episode mapping at this point).
    drop_eps: set[int] = set()
    if surgical:
        results_path_for_rounds = args.results_path or (dataset_path / "results.json")
        if args.drop_rounds:
            rounds = {int(r) for r in args.drop_rounds.split(",") if r.strip()}
            drop_eps |= _episodes_for_rounds(results_path_for_rounds, rounds)
            print(f"--drop-rounds {sorted(rounds)} -> episodes {sorted(drop_eps)}")
        if args.drop_episodes:
            explicit = {int(e) for e in args.drop_episodes.split(",") if e.strip()}
            drop_eps |= explicit
            print(f"--drop-episodes {sorted(explicit)}")
        print(f"Total drop set: {sorted(drop_eps)} ({len(drop_eps)} episode(s))")

    backup_root = _backup(dataset_path)
    if args.keep_episodes is not None:
        _truncate_to_episodes(dataset_path, args.keep_episodes, backup_root)
    elif surgical:
        _drop_episodes(dataset_path, drop_eps, backup_root)
    result = renumber_dataset_to_durable_contiguous(dataset_path)
    print(
        f"Rebuilt info/stats: {result.total_episodes} episodes, {result.total_frames} frames"
        + (
            f"; dropped phantom indices {result.dropped_episodes}."
            if result.dropped_episodes
            else "."
        )
    )

    results_path = args.results_path or (dataset_path / "results.json")
    if results_path.exists():
        report = _remap_results_json(results_path, result.index_map)
        print(
            f"results.json: kept {report['kept']} rollout record(s), dropped {report['dropped']}; "
            f"rounds reopened for re-collection: {report['reopened_rounds']}"
        )
    else:
        print(f"results.json not found at {results_path}; skipping rollout-record remap.")

    _verify(dataset_path, result.total_episodes, result.total_frames)

    print("Repair complete.")


if __name__ == "__main__":
    main()
