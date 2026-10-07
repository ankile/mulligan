#!/usr/bin/env python3
"""Protocol split of a real blind collection: one mixed dataset -> per-arm views.

Blind DAgger collections are split by the ``arm_key`` of each episode in the
protocol-quota ledger; teleop collections use the same LeRobot copy path but
key episodes by ``manifest_source`` in ``teleop_manifest_ledger.jsonl``. Each view
records its lineage (parent repo, ledger sha256, selected episodes) in
``meta/dataset_lineage.json``.

    python -m mulligan.real.data.split --help
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.split_copy import (
    build_frame_lookup_by_index,
    copy_episode,
    create_target_dataset,
    target_features,
)
from mulligan.data.task_names import validate_lerobot_task_name
from mulligan.tools.lerobot_fast_split import (
    fast_split_dataset,
    last_frame_success_by_episode,
    split_video_encoder,
)
from mulligan.tools.dataset_lineage import (
    make_parent_lineage,
    make_split_lineage,
    push_local_lineage,
    sha256_file,
    write_local_lineage,
)
from mulligan.tools.lerobot_hub import (
    push_lerobot_dataset_replacing_remote,
    refresh_lerobot_dataset_from_main,
)


def _parse_expected_counts(spec: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--expected-success-per-arm entries must be ARM=INT; got {item!r}"
            )
        arm, value = item.split("=", 1)
        counts[arm.strip()] = int(value)
    if not counts:
        raise argparse.ArgumentTypeError("--expected-success-per-arm must not be empty")
    return counts


def _parse_target(spec: str) -> tuple[str, str, str | None]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--target must be ARM=REPO[:ROOT]; got {spec!r}")
    arm, rhs = spec.split("=", 1)
    if ":" in rhs:
        repo, root = rhs.split(":", 1)
    else:
        repo, root = rhs, None
    arm = arm.strip()
    repo = repo.strip()
    root = root.strip() if root else None
    if not arm or not repo:
        raise argparse.ArgumentTypeError(f"--target has empty arm or repo: {spec!r}")
    return arm, repo, root


def _dataset_task_name(source: LeRobotDataset) -> str | None:
    tasks = getattr(source.meta, "tasks", None)
    if tasks is None or len(tasks) == 0:
        return None
    if "name" in tasks:
        return str(tasks.iloc[0]["name"])
    return str(tasks.iloc[0].name)


def _create_target_dataset(
    template: LeRobotDataset,
    repo_id: str,
    root: Path,
    *,
    drop_visual_features: bool,
) -> LeRobotDataset:
    features = target_features(template, drop_visual_features=drop_visual_features).copy()
    if "intervention" not in features and {"source", "success", "reward", "done"} <= set(features):
        features["intervention"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["intervention_flag"],
        }
    return create_target_dataset(
        template,
        repo_id,
        root,
        drop_visual_features=drop_visual_features,
        features=features,
        default_robot_type="franka",
    )


def _load_or_create_target_dataset(
    template: LeRobotDataset,
    repo_id: str,
    root: Path,
    *,
    drop_visual_features: bool,
    append_to_existing: bool,
) -> LeRobotDataset:
    if append_to_existing and (root / "meta" / "info.json").exists():
        print(f"  loading existing target dataset for append: repo={repo_id}, root={root}")
        # The bare constructor is read-only in the pinned LeRobot; resume() opens a writer.
        return LeRobotDataset.resume(repo_id=repo_id, root=str(root))
    return _create_target_dataset(
        template,
        repo_id,
        root,
        drop_visual_features=drop_visual_features,
    )


def _load_sidecar_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "episode_index" not in row or "source_episode_index" not in row:
            raise SystemExit(
                f"{path}:{line_no}: sidecar row must contain episode_index and source_episode_index"
            )
        rows.append(row)
    expected = list(range(len(rows)))
    actual = sorted(int(row["episode_index"]) for row in rows)
    if actual != expected:
        raise SystemExit(
            f"{path}: sidecar episode_index values must be contiguous 0..{len(rows) - 1}"
        )
    return rows


def _load_rows(
    path: Path,
    *,
    split_key: str = "arm_key",
    outcome_key: str | None = "outcome",
    default_outcome: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("episode_index", split_key, "success", "steps"):
            if key not in row:
                raise SystemExit(f"{path}:{line_no}: missing required key {key!r}")
        if outcome_key is not None and outcome_key not in row:
            if default_outcome is None:
                raise SystemExit(f"{path}:{line_no}: missing required key {outcome_key!r}")
            row[outcome_key] = default_outcome
        rows.append(row)
    if not rows:
        raise SystemExit(f"No ledger rows found in {path}")
    episode_idxs = sorted(int(row["episode_index"]) for row in rows)
    expected = list(range(len(rows)))
    if episode_idxs != expected:
        raise SystemExit(
            f"{path}: episode_index values must be contiguous 0..{len(rows) - 1}; "
            f"got first values {episode_idxs[:10]}"
        )
    return rows


def _unique_string_values(rows: list[dict], key: str) -> list[str]:
    return sorted(
        {
            str(row[key])
            for row in rows
            if key in row and row[key] is not None and str(row[key]).strip()
        }
    )


def _manifest_state_fields(row: dict) -> dict:
    keys = ("pen_x", "pen_y", "pen_yaw", "nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y")
    return {key: row[key] for key in keys if key in row}


def build_parser() -> argparse.ArgumentParser:
    """CLI surface, built separately so tests can assert on it without running a split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--split-key",
        default="arm_key",
        help=(
            "Ledger key used to assign episodes to targets. "
            "Use manifest_source for teleop collections."
        ),
    )
    parser.add_argument(
        "--outcome-key",
        default="outcome",
        help="Ledger outcome key copied into split sidecars.",
    )
    parser.add_argument(
        "--default-outcome",
        default=None,
        help="Outcome value to use when --outcome-key is absent from the source ledger.",
    )
    parser.add_argument(
        "--expected-success-per-arm",
        type=_parse_expected_counts,
        required=True,
    )
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        required=True,
        help="ARM=REPO[:ROOT]. Provide one per arm.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--drop-visual-features", action="store_true")
    parser.add_argument(
        "--append-to-existing",
        action="store_true",
        help=(
            "Append to target roots when they already contain finalized LeRobot datasets. "
            "Existing source_episode_index rows in the sidecar are skipped."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete local target roots before creating split datasets.",
    )
    parser.add_argument(
        "--replace-remote-codec",
        action="store_true",
        help=(
            "Allow --push to overwrite an existing Hub dataset whose video codec differs "
            "from the codec this split writes (LeRobot's AV1 default). Off by default: a "
            "codec change rewrites every remote pixel under the same repo id."
        ),
    )
    parser.add_argument(
        "--sidecar-name",
        default="real_blind_dagger_split_manifest.jsonl",
        help="Filename to write under each target dataset's meta directory.",
    )
    parser.add_argument(
        "--source-type",
        choices=["teleop", "dagger"],
        default=None,
        help="Dataset source type for lineage metadata. Defaults from --split-key.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Copy only successful episodes. Default copies all saved episodes for each arm.",
    )
    parser.add_argument(
        "--copy-backend",
        choices=["native", "frame"],
        default="native",
        help=(
            "Copy engine. 'native' (default) uses LeRobot's file-level parquet+video "
            "copy (fast, tolerant of null columns). 'frame' copies frame by frame through "
            "add_frame (slow; needed for --drop-visual-features or --append-to-existing)."
        ),
    )
    parser.add_argument(
        "--drop-columns",
        default=None,
        help=(
            "Comma-separated feature columns to drop from the split outputs (native backend). "
            "Use for analysis-only / partially-null columns training does not consume, e.g. "
            "telemetry.franka.motor_torques_external."
        ),
    )
    parser.add_argument(
        "--fill-null-columns",
        default=None,
        help=(
            "Comma-separated columns whose null cells are filled with NaN before the split "
            "(native backend). Preserves the column (real values where present, NaN where "
            "missing) and avoids the aggregate_stats null crash. Prefer this over "
            "--drop-columns when the partially-null data is worth keeping (e.g. "
            "telemetry.franka.motor_torques_external)."
        ),
    )
    parser.add_argument("--push", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.overwrite and args.append_to_existing:
        raise SystemExit("--overwrite and --append-to-existing are mutually exclusive")

    outcome_key = args.outcome_key if args.outcome_key else None
    rows = _load_rows(
        args.ledger,
        split_key=args.split_key,
        outcome_key=outcome_key,
        default_outcome=args.default_outcome,
    )
    targets_by_arm: dict[str, tuple[str, str | None]] = {}
    for arm, repo, root in args.target:
        if arm in targets_by_arm:
            raise SystemExit(f"Duplicate --target for arm {arm!r}")
        targets_by_arm[arm] = (repo, root)

    expected_arms = set(args.expected_success_per_arm)
    target_arms = set(targets_by_arm)
    ledger_arms = {str(row[args.split_key]) for row in rows}
    if target_arms != expected_arms:
        raise SystemExit(
            f"--target arms do not match --expected-success-per-arm arms. "
            f"targets={sorted(target_arms)}, expected={sorted(expected_arms)}"
        )
    if not ledger_arms <= expected_arms:
        raise SystemExit(
            f"Ledger contains arms without targets: {sorted(ledger_arms - expected_arms)}"
        )

    success_counts: Counter[str] = Counter()
    for row in rows:
        if bool(row["success"]):
            success_counts[str(row[args.split_key])] += 1
    bad_counts = {
        arm: success_counts.get(arm, 0)
        for arm, expected in args.expected_success_per_arm.items()
        if success_counts.get(arm, 0) != expected
    }
    if bad_counts:
        raise SystemExit(
            f"Ledger success counts do not match expected targets: {bad_counts}. "
            f"All success counts: {dict(success_counts)}"
        )
    print(f"Success counts verified: {dict(success_counts)}")

    ds = (
        LeRobotDataset(
            repo_id=args.source_repo,
            root=args.source_root,
            download_videos=not args.drop_visual_features,
        )
        if args.source_root
        else refresh_lerobot_dataset_from_main(
            repo_id=args.source_repo,
            download_videos=not args.drop_visual_features,
        )
    )
    if ds.num_episodes < len(rows):
        raise SystemExit(
            f"Source dataset has {ds.num_episodes} episodes but ledger has {len(rows)} rows"
        )
    # Resolve the task IDENTIFIER from the LEDGER (authoritative `task_name`), not the
    # LeRobot task string: per the v3 dataset spec the task string is free-form natural
    # language (and may be a real instruction for task-conditioned/VLA policies), so it must
    # not double as the identifier. Fall back to the dataset task string for a ledger
    # without a `task_name` column.
    ledger_task_names = {r.get("task_name") for r in rows if r.get("task_name")}
    if len(ledger_task_names) > 1:
        raise SystemExit(
            f"Ledger {args.ledger} has multiple task_names {sorted(ledger_task_names)}; "
            "expected a single task per split source."
        )
    task_name = ledger_task_names.pop() if ledger_task_names else _dataset_task_name(ds)
    if task_name is not None:
        task_name = validate_lerobot_task_name(
            task_name, context=f"{args.source_repo} ledger/dataset task name"
        )
    source_type = args.source_type or (
        "teleop" if args.split_key == "manifest_source" else "dagger"
    )
    ledger_sha256 = sha256_file(args.ledger)
    manifest_path = rows[0].get("manifest_snapshot_path")
    manifest_sha256 = rows[0].get("manifest_sha256")
    # Backend gating: the per-frame copy is needed only for the niche flags.
    drop_columns = (
        [c.strip() for c in args.drop_columns.split(",") if c.strip()]
        if args.drop_columns
        else None
    )
    fill_null_columns = (
        [c.strip() for c in args.fill_null_columns.split(",") if c.strip()]
        if args.fill_null_columns
        else None
    )
    if args.copy_backend == "native" and (args.append_to_existing or args.drop_visual_features):
        raise SystemExit(
            "--append-to-existing and --drop-visual-features require --copy-backend frame"
        )
    if (drop_columns or fill_null_columns) and args.copy_backend != "native":
        raise SystemExit(
            "--drop-columns / --fill-null-columns are only supported by --copy-backend native"
        )
    if drop_columns and fill_null_columns and (set(drop_columns) & set(fill_null_columns)):
        raise SystemExit(
            f"Columns cannot be both dropped and filled: {sorted(set(drop_columns) & set(fill_null_columns))}"
        )

    frame_lookup = build_frame_lookup_by_index(ds) if args.drop_visual_features else None
    # Null-safe terminal-success cross-check. Reads the success column without the torch
    # transform, so it tolerates partially-null columns (e.g. an all-None telemetry column
    # for early episodes) that the per-frame torch / frame_lookup path crashes on.
    ds_success = last_frame_success_by_episode(ds)
    for row in rows:
        ep_idx = int(row["episode_index"])
        if bool(row["success"]) != ds_success[ep_idx]:
            raise SystemExit(
                f"Ledger success mismatch for episode {ep_idx}: "
                f"ledger={bool(row['success'])}, dataset={ds_success[ep_idx]}"
            )

    args.output_root.mkdir(parents=True, exist_ok=True)

    def _sidecar_row(new_episode_index: int, row: dict) -> dict:
        return {
            "episode_index": new_episode_index,
            "source_episode_index": int(row["episode_index"]),
            "split_key": args.split_key,
            "arm_key": str(row[args.split_key]),
            "arm_id": int(row["arm_id"]) if "arm_id" in row else None,
            "model_id": row.get("model_id"),
            "manifest_idx": row.get("manifest_idx"),
            "manifest_source": row.get("manifest_source"),
            "manifest_source_index": row.get("manifest_source_index"),
            "manifest_snapshot_path": row.get("manifest_snapshot_path"),
            "manifest_sha256": row.get("manifest_sha256"),
            "initial_state_visualization_path": row.get("initial_state_visualization_path"),
            **_manifest_state_fields(row),
            "source_success": bool(row["success"]),
            "source_outcome": row[outcome_key] if outcome_key is not None else None,
            "source_steps": int(row["steps"]),
        }

    rows_to_copy = [row for row in rows if bool(row["success"])] if args.success_only else rows
    target_roots: dict[str, Path] = {
        arm: (Path(root_override) if root_override else args.output_root / repo.split("/")[-1])
        for arm, (repo, root_override) in targets_by_arm.items()
    }
    sidecar_paths: dict[str, Path] = {
        arm: root / "meta" / args.sidecar_name for arm, root in target_roots.items()
    }
    targets: dict[str, LeRobotDataset] = {}
    sidecars: dict[str, list[dict]] = {}
    copied_source_rows: dict[str, list[dict]] = {}

    if args.copy_backend == "native":
        # Assign episodes to arms in ledger order (= ascending source episode index).
        arm_rows: dict[str, list[dict]] = {arm: [] for arm in targets_by_arm}
        for row in rows_to_copy:
            arm_rows[str(row[args.split_key])].append(row)
        groups: dict[str, dict] = {}
        for arm, root in target_roots.items():
            if root.exists():
                if not args.overwrite:
                    raise SystemExit(f"Target root exists (use --overwrite): {root}")
                print(f"  removing existing target root for overwrite: {root}")
                shutil.rmtree(root)
            groups[arm] = {
                "repo_id": targets_by_arm[arm][0],
                "root": root,
                "episodes": [int(r["episode_index"]) for r in arm_rows[arm]],
            }
        print(
            f"Native split backend: encoder={split_video_encoder()}"
            + (f" fill_null={fill_null_columns}" if fill_null_columns else "")
            + (f" drop={drop_columns}" if drop_columns else "")
        )
        targets = fast_split_dataset(
            ds,
            groups,
            drop_columns=drop_columns,
            fill_null_columns=fill_null_columns,
        )
        sidecars = {
            arm: [_sidecar_row(i, r) for i, r in enumerate(arm_rows[arm])] for arm in targets
        }
        copied_source_rows = arm_rows
        written = Counter({arm: int(t.num_episodes) for arm, t in targets.items()})
        for arm in targets:
            if written[arm] != len(sidecars[arm]):
                raise SystemExit(
                    f"{arm}: native split wrote {written[arm]} episodes but assigned "
                    f"{len(sidecars[arm])} ledger rows"
                )
    else:
        for arm, root_path in target_roots.items():
            if args.overwrite and root_path.exists():
                print(f"  removing existing target root for overwrite: {root_path}")
                shutil.rmtree(root_path)
            targets[arm] = _load_or_create_target_dataset(
                ds,
                targets_by_arm[arm][0],
                root_path,
                drop_visual_features=args.drop_visual_features,
                append_to_existing=args.append_to_existing,
            )
            print(f"  target {arm}: repo={targets_by_arm[arm][0]}, root={root_path}")
        written = Counter({arm: int(target.num_episodes) for arm, target in targets.items()})
        sidecars = {
            arm: _load_sidecar_rows(sidecar_paths[arm]) if args.append_to_existing else []
            for arm in targets
        }
        existing_source_eps: dict[str, set[int]] = {
            arm: {int(r["source_episode_index"]) for r in srows} for arm, srows in sidecars.items()
        }
        for arm in targets:
            if args.append_to_existing and written[arm] != len(sidecars[arm]):
                raise SystemExit(
                    f"{arm}: existing dataset has {written[arm]} episodes but sidecar has "
                    f"{len(sidecars[arm])} rows"
                )
        copied_source_rows = {arm: [] for arm in targets}
        for row in rows_to_copy:
            arm = str(row[args.split_key])
            ep_idx = int(row["episode_index"])
            if ep_idx in existing_source_eps[arm]:
                continue
            copy_episode(
                ds,
                ep_idx,
                targets[arm],
                drop_visual_features=args.drop_visual_features,
                frame_lookup=frame_lookup,
                zero_missing_intervention=True,
            )
            copied_source_rows[arm].append(row)
            sidecars[arm].append(_sidecar_row(int(written[arm]), row))
            written[arm] += 1
            existing_source_eps[arm].add(ep_idx)
            if sum(written.values()) % 25 == 0:
                print(f"  copied {sum(written.values())} episodes")

    # Propagate the source's camera role->serial provenance into each split. The fast splitter
    # rebuilds split metadata from features/fps/robot_type only, so without this the splits lose
    # the physical-serial provenance the source carries (training keys cameras by the role-named
    # feature keys, which ARE preserved, so this is provenance-only — but it keeps the splits
    # symmetric with the source and self-describing).
    from mulligan.real.collect.dataset_features import (
        CAMERA_ROLE_SERIALS_SIDECAR,
        _load_camera_role_serials,
    )

    source_camera_role_serials = _load_camera_role_serials(ds.root)

    for arm, target_ds in targets.items():
        sidecar_path = sidecar_paths[arm]
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in sidecars[arm])
        )
        print(f"  wrote sidecar {sidecar_path}")
        if source_camera_role_serials:
            crs_path = Path(target_ds.root) / "meta" / CAMERA_ROLE_SERIALS_SIDECAR
            crs_path.parent.mkdir(parents=True, exist_ok=True)
            crs_path.write_text(json.dumps(source_camera_role_serials, indent=2, sort_keys=True))
            print(f"  wrote camera_role_serials {crs_path}")
        target_repo = targets_by_arm[arm][0]
        source_rows_for_arm = copied_source_rows[arm] or [
            row for row in rows_to_copy if str(row[args.split_key]) == arm
        ]
        producer_model_ids = _unique_string_values(source_rows_for_arm, "model_id")
        target_model_id = producer_model_ids[0] if len(producer_model_ids) == 1 else None
        lineage = make_split_lineage(
            repo_id=target_repo,
            task=task_name,
            source_type=source_type,
            parent_repo_id=args.source_repo,
            derivation_script="mulligan.real.data.split",
            sidecar_path=f"meta/{args.sidecar_name}",
            ledger_path=str(args.ledger),
            ledger_sha256=ledger_sha256,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            split_key=args.split_key,
            view_family_id=f"{args.source_repo}::{args.split_key}:{arm}",
            view_id="default",
            extra={
                "arm": arm,
                "target_arm_key": arm,
                "target_model_id": target_model_id,
                "producer_model_ids": producer_model_ids,
            },
        )
        lineage_path = write_local_lineage(target_ds.root, lineage)
        print(f"  wrote lineage {lineage_path}")
        print(f"{arm} split: {target_ds.num_episodes} episodes at {target_ds.root}")
        if args.copy_backend == "frame":
            # The native backend writes a complete dataset (info/stats) during the
            # file-level copy; finalize() is only needed for the add_frame path.
            target_ds.finalize()

    parent_lineage = make_parent_lineage(
        repo_id=args.source_repo,
        task=task_name,
        source_type=source_type,
        ledger_path=str(args.ledger),
        ledger_sha256=ledger_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        derived_repos=[repo for repo, _root in targets_by_arm.values()],
        derivation_script="mulligan.real.data.split",
    )
    parent_lineage_path = write_local_lineage(ds.root, parent_lineage)
    print(f"  wrote parent lineage {parent_lineage_path}")

    if args.push:
        print("\nPushing split datasets to HuggingFace Hub...")
        for arm, target_ds in targets.items():
            push_lerobot_dataset_replacing_remote(
                target_ds, replace_remote_codec=args.replace_remote_codec
            )
            print(f"  pushed {arm}: {targets_by_arm[arm][0]}")
        push_local_lineage(args.source_repo, ds.root)
        print(f"  pushed parent lineage: {args.source_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
