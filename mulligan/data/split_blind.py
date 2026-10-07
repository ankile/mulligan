#!/usr/bin/env python3
"""Split a blinded list-sampler collection by manifest source label.

Each source episode is matched from its first-frame environment state to the
manifest state vectors, then copied into the target dataset configured for that
manifest row's ``source`` label.

Example:
    python -m mulligan.data.split_blind \\
        --source-repo <user>/square-narrow-blind-r1 \\
        --manifest path/to/manifest.json \\
        --target baseline_uniform=<user>/square-narrow-baseline-r1 \\
        --target sobol=<user>/square-narrow-sobol-r1 \\
        --output-root ./data/blind_split
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.data.split_copy import (
    build_frame_lookup_by_index,
    copy_episode,
    create_target_dataset,
    frame_at,
    last_frame_success,
)
from mulligan.sim.envs import SIM_TASK_ENV_NAMES
from mulligan.tools.lerobot_hub import (
    push_lerobot_dataset_replacing_remote,
    refresh_lerobot_dataset_from_main,
)
from mulligan.utils.manifest_matching import ManifestMatcher
from mulligan.utils.state_to_grid import extract_sampler_state_from_env_state

TASK_NAME_FOR_EXTRACTOR = SIM_TASK_ENV_NAMES


def _parse_expected(spec: str) -> dict[str, int]:
    expected: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected SOURCE=INT, got {item!r}")
        source, count = item.split("=", 1)
        expected[source.strip()] = int(count)
    return expected


def _parse_target(spec: str) -> tuple[str, str, str | None]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected SOURCE=REPO[:ROOT], got {spec!r}")
    source, rhs = spec.split("=", 1)
    if ":" in rhs:
        repo, root = rhs.split(":", 1)
    else:
        repo, root = rhs, None
    source = source.strip()
    repo = repo.strip()
    root = root.strip() if root else None
    if not source or not repo:
        raise argparse.ArgumentTypeError(f"empty source or repo in {spec!r}")
    return source, repo, root


def _load_manifest(path: Path) -> tuple[str, list[str], np.ndarray, list[str], float]:
    payload = json.loads(path.read_text())
    task = payload["task"]
    keys = payload["keys"]
    states = payload["states"]
    arr = np.asarray([[s[k] for k in keys] for s in states], dtype=np.float64)
    sources = [s["source"] for s in states]
    tol = float(payload.get("match_tolerance", 1e-3))
    return task, keys, arr, sources, tol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", action="append", type=_parse_target, required=True)
    parser.add_argument("--expected-per-source", type=_parse_expected, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-source-episodes",
        type=int,
        default=None,
        help="Only scan the first N source dataset episodes before matching and splitting.",
    )
    parser.add_argument("--drop-visual-features", action="store_true")
    parser.add_argument("--push", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_source_episodes is not None and args.max_source_episodes < 1:
        raise SystemExit(f"--max-source-episodes must be >= 1, got {args.max_source_episodes}")

    task, keys, manifest_arr, manifest_sources, tol = _load_manifest(args.manifest)
    extractor_task = TASK_NAME_FOR_EXTRACTOR[task]
    matcher = ManifestMatcher.from_keys(manifest_arr, keys)
    targets = {source: (repo, root) for source, repo, root in args.target}
    missing_targets = set(manifest_sources) - set(targets)
    if missing_targets:
        raise SystemExit(f"Missing --target entries for sources: {sorted(missing_targets)}")

    source_ds = (
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
    frame_lookup = build_frame_lookup_by_index(source_ds) if args.drop_visual_features else None
    assigned: list[tuple[int, int, str]] = []
    seen_manifest: dict[int, int] = {}
    duplicates: list[tuple[int, int, int]] = []

    max_episodes = (
        source_ds.num_episodes
        if args.max_source_episodes is None
        else min(args.max_source_episodes, source_ds.num_episodes)
    )
    print(f"Scanning first {max_episodes} / {source_ds.num_episodes} source episodes")

    for ep_idx in range(max_episodes):
        if not last_frame_success(source_ds, ep_idx, frame_lookup=frame_lookup):
            continue
        first = int(source_ds.meta.episodes[ep_idx]["dataset_from_index"])
        env_state = np.asarray(
            frame_at(
                source_ds,
                first,
                frame_lookup=frame_lookup,
            )["observation.environment_state"]
        )
        sampler_state = extract_sampler_state_from_env_state(env_state, task=extractor_task)
        try:
            manifest_idx, _ = matcher.query_within_tolerance(
                np.asarray(sampler_state, dtype=np.float64), tol
            )
        except ValueError as exc:
            raise SystemExit(f"Episode {ep_idx}: {exc}") from exc
        if manifest_idx in seen_manifest:
            duplicates.append((ep_idx, manifest_idx, seen_manifest[manifest_idx]))
            continue
        seen_manifest[manifest_idx] = ep_idx
        assigned.append((ep_idx, manifest_idx, manifest_sources[manifest_idx]))

    if duplicates:
        print(f"WARNING: dropping {len(duplicates)} duplicate manifest matches")
        for ep_idx, manifest_idx, kept_ep in duplicates[:20]:
            print(
                f"  drop episode {ep_idx}: manifest {manifest_idx} already kept by episode {kept_ep}"
            )

    counts = Counter(source for _, _, source in assigned)
    print(f"Assigned successful unique episodes: {dict(sorted(counts.items()))}")
    if args.expected_per_source:
        for source, expected_count in args.expected_per_source.items():
            actual = counts.get(source, 0)
            if actual != expected_count:
                raise SystemExit(f"Expected {expected_count} episodes for {source}, got {actual}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    target_datasets: dict[str, LeRobotDataset] = {}
    for source, (repo, root_override) in targets.items():
        root = Path(root_override) if root_override else args.output_root / repo.split("/")[-1]
        target_datasets[source] = create_target_dataset(
            source_ds,
            repo,
            root,
            drop_visual_features=args.drop_visual_features,
        )

    copied = Counter()
    for ep_idx, _manifest_idx, source in assigned:
        copy_episode(
            source_ds,
            ep_idx,
            target_datasets[source],
            drop_visual_features=args.drop_visual_features,
            frame_lookup=frame_lookup,
        )
        copied[source] += 1
        if sum(copied.values()) % 25 == 0:
            print(f"  copied {sum(copied.values())} episodes: {dict(sorted(copied.items()))}")

    for source, target in target_datasets.items():
        target.finalize()
        print(f"{source}: {target.num_episodes} episodes -> {target.repo_id}")
        if args.push:
            push_lerobot_dataset_replacing_remote(target)
            print(f"  pushed {target.repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
