#!/usr/bin/env python3
"""Split a blind real-eval dataset into policy-specific training views.

The unsplit eval dataset remains the source of truth for the head-to-head
comparison. This tool creates derived LeRobot datasets for cases
where only one policy's own rollout episodes should be used for later training.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.split_copy import last_frame_success
from mulligan.data.task_names import validate_lerobot_task_name
from mulligan.real.data.split import _dataset_task_name
from mulligan.tools.dataset_lineage import (
    make_parent_lineage,
    make_split_lineage,
    push_local_lineage,
    write_local_lineage,
)
from mulligan.tools.lerobot_hub import (
    push_lerobot_dataset_replacing_remote,
    refresh_lerobot_dataset_from_main,
)
from mulligan.tools.lerobot_fast_split import fast_split_dataset, split_video_encoder


def _parse_target(spec: str) -> tuple[str, str, str | None]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--target must be POLICY_SELECTOR=REPO[:ROOT]; got {spec!r}"
        )
    selector, rhs = spec.split("=", 1)
    if ":" in rhs:
        repo, root = rhs.split(":", 1)
    else:
        repo, root = rhs, None
    selector = selector.strip()
    repo = repo.strip()
    root = root.strip() if root else None
    if not selector or not repo:
        raise argparse.ArgumentTypeError(f"--target has empty selector or repo: {spec!r}")
    return selector, repo, root


def _load_results_json(
    source_repo: str, source_root: Path | None, results_path: Path | None
) -> dict:
    if results_path is not None:
        path = results_path
    elif source_root is not None and (source_root / "results.json").exists():
        path = source_root / "results.json"
    else:
        path = Path(
            hf_hub_download(
                repo_id=source_repo,
                repo_type="dataset",
                filename="results.json",
            )
        )
    with path.open() as f:
        payload = json.load(f)
    if "rollouts" not in payload or "summary" not in payload:
        raise SystemExit(f"{path}: expected blind-eval results.json with summary and rollouts")
    return payload


def _policy_entries(results: dict) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    # Repeated models are valid arms; only their shared selectors are ambiguous.
    ambiguous: set[str] = set()
    policy_ids: set[int] = set()
    for entry in results["summary"]:
        policy_id = int(entry["policy_id"])
        if policy_id in policy_ids:
            raise SystemExit(f"Duplicate policy_id={policy_id} in results.json summary")
        policy_ids.add(policy_id)
        model_id = str(entry["model_id"])
        selectors = {str(policy_id), model_id}
        if entry.get("name"):
            selectors.add(str(entry["name"]))
        for selector in selectors:
            if selector in entries:
                entries.pop(selector)
                ambiguous.add(selector)
            elif selector not in ambiguous:
                entries[selector] = entry
    return entries


def _rollouts_by_episode(results: dict) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in results["rollouts"]:
        episode_index = int(row["episode_index"])
        if episode_index < 0:
            continue
        if episode_index in rows:
            raise SystemExit(f"results.json has duplicate episode_index={episode_index}")
        rows[episode_index] = row
    return rows


def _episode_policy_id(source: LeRobotDataset, ep_idx: int) -> int:
    start = int(source.meta.episodes[ep_idx]["dataset_from_index"])
    return int(np.asarray(source.hf_dataset[start]["policy_id"]).item())


def build_parser() -> argparse.ArgumentParser:
    """CLI surface, built separately so tests can assert on it without running a split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        required=True,
        help=(
            "POLICY_SELECTOR=REPO[:ROOT]. Selector can be policy_id, model_id, "
            "or policy name from results.json."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--drop-visual-features",
        action="store_true",
        help="Unsupported for policy-view splits; native splitting preserves videos.",
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
    parser.add_argument("--push", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.drop_visual_features:
        raise SystemExit(
            "split_real_eval_policy_episodes uses the native splitter and does not support "
            "--drop-visual-features. Preserve videos for policy-rollout training views."
        )

    source_root = Path(args.source_root) if args.source_root else None
    results = _load_results_json(args.source_repo, source_root, args.results_path)
    policy_entries = _policy_entries(results)
    rollouts_by_episode = _rollouts_by_episode(results)

    targets_by_policy_id: dict[int, tuple[str, str, str | None, dict[str, Any]]] = {}
    for selector, repo, root in args.target:
        if selector not in policy_entries:
            raise SystemExit(
                f"Unknown policy selector {selector!r}. Known selectors include: "
                f"{sorted(policy_entries)[:20]}"
            )
        entry = policy_entries[selector]
        policy_id = int(entry["policy_id"])
        if policy_id in targets_by_policy_id:
            raise SystemExit(f"Duplicate --target for policy_id={policy_id}")
        targets_by_policy_id[policy_id] = (selector, repo, root, entry)

    ds = (
        LeRobotDataset(
            repo_id=args.source_repo,
            root=str(source_root),
            download_videos=not args.drop_visual_features,
        )
        if source_root is not None
        else refresh_lerobot_dataset_from_main(
            repo_id=args.source_repo,
            download_videos=not args.drop_visual_features,
        )
    )
    task_name = _dataset_task_name(ds)
    if task_name is None:
        raise SystemExit(f"{args.source_repo}: missing LeRobot task metadata")
    task_name = validate_lerobot_task_name(
        task_name, context=f"{args.source_repo} LeRobot task metadata"
    )

    selected_rows: dict[int, list[dict[str, Any]]] = {pid: [] for pid in targets_by_policy_id}
    result_mismatches: Counter[int] = Counter()
    for ep_idx in range(ds.num_episodes):
        policy_id = _episode_policy_id(ds, ep_idx)
        if policy_id not in selected_rows:
            continue
        if ep_idx not in rollouts_by_episode:
            raise SystemExit(f"results.json is missing saved episode_index={ep_idx}")
        row = rollouts_by_episode[ep_idx]
        if int(row["policy_id"]) != policy_id:
            raise SystemExit(
                f"episode {ep_idx}: dataset policy_id={policy_id} but results.json has "
                f"policy_id={row['policy_id']}"
            )
        current_success = last_frame_success(ds, ep_idx, frame_lookup=None)
        result_success = str(row["outcome"]) == "success"
        if current_success != result_success:
            result_mismatches[policy_id] += 1
        selected_rows[policy_id].append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    targets: dict[int, LeRobotDataset] = {}
    split_sidecars: dict[int, list[dict[str, Any]]] = {pid: [] for pid in targets_by_policy_id}
    target_roots: dict[int, Path] = {}
    for policy_id, (_selector, repo, root_override, _entry) in targets_by_policy_id.items():
        target_roots[policy_id] = (
            Path(root_override) if root_override else args.output_root / repo.split("/")[-1]
        )

    sorted_selected_rows = {
        policy_id: sorted(rows, key=lambda r: int(r["episode_index"]))
        for policy_id, rows in selected_rows.items()
    }

    for policy_id, rows in sorted_selected_rows.items():
        for new_episode_index, row in enumerate(rows):
            source_ep_idx = int(row["episode_index"])
            current_success = last_frame_success(ds, source_ep_idx, frame_lookup=None)
            split_sidecars[policy_id].append(
                {
                    "episode_index": new_episode_index,
                    "source_episode_index": source_ep_idx,
                    "source_round": int(row["round"]),
                    "source_policy_id": int(row["policy_id"]),
                    "source_model_id": row["model_id"],
                    "source_policy_name": targets_by_policy_id[policy_id][3].get("name"),
                    "source_result_outcome": row["outcome"],
                    "source_current_success": bool(current_success),
                    "source_num_steps": int(row["num_steps"]),
                    "manifest_idx": row.get("manifest_idx"),
                    "pen_x": row.get("pen_x"),
                    "pen_y": row.get("pen_y"),
                    "pen_yaw": row.get("pen_yaw"),
                }
            )

    groups = {}
    for policy_id, (_selector, repo, _root_override, _entry) in targets_by_policy_id.items():
        root_path = target_roots[policy_id]
        if root_path.exists():
            if not args.overwrite:
                raise SystemExit(f"Target root exists (use --overwrite): {root_path}")
            print(f"  removing existing target root for overwrite: {root_path}")
            shutil.rmtree(root_path)
        groups[str(policy_id)] = {
            "repo_id": repo,
            "root": root_path,
            "episodes": [int(row["episode_index"]) for row in sorted_selected_rows[policy_id]],
        }
    print(f"Native split backend: encoder={split_video_encoder()}")
    native_targets = fast_split_dataset(ds, groups)
    targets = {int(policy_id): target for policy_id, target in native_targets.items()}

    for policy_id, target_ds in targets.items():
        selector, repo, _root, entry = targets_by_policy_id[policy_id]
        sidecar_path = target_ds.root / "meta" / "real_eval_policy_split_manifest.jsonl"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_sidecars[policy_id])
        )
        print(f"  wrote sidecar {sidecar_path}")
        lineage = make_split_lineage(
            repo_id=repo,
            task=task_name,
            source_type="eval",
            parent_repo_id=args.source_repo,
            derivation_script="mulligan.real.eval.split_policies",
            sidecar_path="meta/real_eval_policy_split_manifest.jsonl",
            ledger_path=None,
            ledger_sha256=None,
            manifest_path=results.get("args", {}).get("initial_states_manifest"),
            manifest_sha256=None,
            split_key="policy_id",
            view_family_id=f"{args.source_repo}::policy_id",
            view_id=str(policy_id),
            extra={
                "policy_selector": selector,
                "target_policy_id": policy_id,
                "target_model_id": entry["model_id"],
                "target_policy_name": entry.get("name"),
                "producer_model_ids": [entry["model_id"]],
            },
        )
        lineage_path = write_local_lineage(target_ds.root, lineage)
        print(f"  wrote lineage {lineage_path}")
        print(f"policy_id={policy_id} split: {target_ds.num_episodes} episodes at {target_ds.root}")
        if result_mismatches[policy_id]:
            print(
                f"  note: {result_mismatches[policy_id]} episode(s) differ from results.json; "
                "using current dataset labels"
            )

    parent_lineage = make_parent_lineage(
        repo_id=args.source_repo,
        task=task_name,
        source_type="eval",
        dataset_role="eval_session",
        trainable=False,
        ledger_path=None,
        ledger_sha256=None,
        manifest_path=results.get("args", {}).get("initial_states_manifest"),
        manifest_sha256=None,
        derived_repos=[repo for _selector, repo, _root, _entry in targets_by_policy_id.values()],
        derivation_script="mulligan.real.eval.split_policies",
    )
    parent_lineage_path = write_local_lineage(ds.root, parent_lineage)
    print(f"  wrote parent lineage {parent_lineage_path}")

    if args.push:
        print("\nPushing eval policy-view splits to HuggingFace Hub...")
        for policy_id, target_ds in targets.items():
            push_lerobot_dataset_replacing_remote(
                target_ds, replace_remote_codec=args.replace_remote_codec
            )
            print(f"  pushed policy_id={policy_id}: {targets_by_policy_id[policy_id][1]}")
        push_local_lineage(args.source_repo, ds.root)
        print(f"  pushed parent lineage: {args.source_repo}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
