"""Build the Arena's simulation statistics from the release's per-state point records.

    python -m mulligan.release.arena_sim_stats --release arena/data/release.json \
        --output sim-statistics.json [--cache DIR] [--local-cache DIR]

Reads each simulation task's evaluation bundle anonymously at the revision the Arena
release file pins. A parquet from ``--local-cache`` is used only after its SHA-256 matches
the metadata downloaded at that revision. Pairs require the same task, grid manifest,
training seed and initial-state ID. Different grid protocols never form pairs. Reused
baseline measurements remain explicit aliases, not additional independently collected
evidence.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def read_file(repo: str, revision: str, name: str, cache: Path | None) -> Path:
    """One file of a dataset repo at a pinned revision, read anonymously."""
    return Path(
        hf_hub_download(
            repo, name, repo_type="dataset", revision=revision, cache_dir=cache, token=False
        )
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(release, cache, local_cache):
    sessions, evidence = [], []
    for task in release["tasks"]:
        if task["domain"] != "sim":
            continue
        repo, revision = (task["evaluationDataset"][k] for k in ("id", "revision"))

        def remote(name):
            return read_file(repo, revision, name, cache)

        meta_path = remote("meta/mainline-evaluations.json")
        meta = json.loads(meta_path.read_text())
        indexed = {(r["arm"], r["round"], r["seed"]): r for r in meta}
        require(len(indexed) == len(meta), "Duplicate evaluation identity")
        grids, groups = {}, {}
        for policy in task["policies"]:
            seed_rows, outcomes, success_lengths = [], [], []
            require([s["seed"] for s in policy["seeds"]] == [1, 2, 3, 4, 5], "Expected five seeds")
            for seed in policy["seeds"]:
                row = indexed[(policy["arm"], int(policy["round"][1:]), seed["seed"])]
                require(row["task"] == task["id"], "Task mismatch")
                require(row["sourcePath"] == seed["sourcePath"], "Source mismatch")
                require(row["artifact"] == seed["artifact"], "Artifact mismatch")
                require(
                    seed["dataUrl"]
                    == f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{row['data']}",
                    "Unpinned point data",
                )
                grid_id = row["gridManifestHash"]
                if grid_id not in grids:
                    grid = json.loads(remote(f"grids/{grid_id}.json").read_text())
                    require(grid["manifest_hash"] == grid_id, "Grid identity mismatch")
                    grids[grid_id] = grid
                grid = grids[grid_id]
                path = local_cache / repo.split("/")[-1] / row["data"] if local_cache else None
                if path is None or not path.exists():
                    path = remote(row["data"])
                require(
                    hashlib.sha256(path.read_bytes()).hexdigest() == row["dataSha256"],
                    f"Point checksum mismatch: {path}",
                )
                frame = pq.read_table(path).to_pandas().sort_values("point_idx")
                n = grid["num_points"]
                require(len(frame) == n == row["episodes"], "Incomplete grid")
                require(
                    np.array_equal(frame.point_idx, np.arange(n)), "Duplicate or missing state IDs"
                )
                points = sorted(grid["points"], key=lambda p: p["point_idx"])
                for key in ("nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"):
                    if key in points[0]:
                        require(
                            np.array_equal(frame[key], [p[key] for p in points]),
                            f"Initial states differ: {policy['id']} {key}",
                        )
                require(frame.success.isin([0, 1]).all(), "Invalid success labels")
                require(
                    (frame.length > 0).all() and (frame.length % 1 == 0).all(),
                    "Invalid episode lengths",
                )
                success = frame.success.to_numpy(dtype=bool)
                lengths = frame.length.to_numpy(dtype=np.int64)
                require(int(success.sum()) == row["successes"], "Success count mismatch")
                require(int(lengths.sum()) == row["totalSteps"], "Step count mismatch")
                require(
                    abs(float(success.mean()) - seed["rate"]) <= 0.00000051,
                    "Released seed rate mismatch",
                )
                outcomes.append(success)
                success_lengths.append(int(lengths[success].sum()))
                seed_rows.append(
                    {
                        "seed": seed["seed"],
                        "url": seed["dataUrl"],
                        "sha256": row["dataSha256"],
                        "gridManifestHash": grid_id,
                        "episodes": n,
                        "successes": int(success.sum()),
                        "successSteps": success_lengths[-1],
                    }
                )
            require(
                len({r["gridManifestHash"] for r in seed_rows}) == 1, "Policy mixes grid protocols"
            )
            require(
                abs(float(np.mean([x.mean() for x in outcomes])) - policy["rate"]) <= 0.00000051,
                "Headline mismatch",
            )
            groups.setdefault(grid_id, []).append(
                (policy["id"], np.concatenate(outcomes), sum(success_lengths))
            )
            evidence.append({"policyId": policy["id"], "seeds": seed_rows})
        for grid_id, rows in groups.items():
            pairs = []
            for (a, x, _), (b, y, _) in itertools.combinations(rows, 2):
                require(x.shape == y.shape, "Pair grid shape mismatch")
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "winsA": int((x & ~y).sum()),
                        "winsB": int((y & ~x).sum()),
                        "draws": int((x == y).sum()),
                    }
                )
            sessions.append(
                {
                    "session_id": f"grid-{task['id']}-{grid_id}",
                    "creation_time": 0,
                    "session_mode": "fixed_grid",
                    "task": task["id"],
                    "rating_group": f"{task['id']}/{grid_id}",
                    "effective_status": "mainline",
                    "pairs": pairs,
                    "perPolicy": [
                        {
                            "policy_id": pid,
                            "rollouts": len(x),
                            "successes": int(x.sum()),
                            "successFramesSum": steps,
                            "successFramesCount": int(x.sum()),
                        }
                        for pid, x, steps in rows
                    ],
                }
            )
        print(
            f"Verified {task['id']}: {len(task['policies'])} policies, {len(grids)} grid protocols",
            flush=True,
        )
    return {
        "schemaVersion": 1,
        "selectionSha256": release["selectionSha256"],
        "releaseSha256": hashlib.sha256(json.dumps(release, sort_keys=True).encode()).hexdigest(),
        "method": "Binary success comparisons match task, grid manifest, training seed and point ID. All selected policies on the same grid are opponents, including other rounds. Draws receive half a win in Bradley-Terry; win rate excludes draws. Ratings are centered separately per task/grid. Mean steps uses successful episodes only. Reused R0 results are aliases, not independent measurements.",
        "sessions": sessions,
        "evidence": evidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=os.environ.get("MULLIGAN_HF_CACHE"),
        help="huggingface_hub cache_dir (default: $MULLIGAN_HF_CACHE, else the HF default)",
    )
    parser.add_argument(
        "--local-cache",
        type=Path,
        help="optional <dir>/<bundle name>/<data path> copies, used after a SHA-256 check",
    )
    args = parser.parse_args(argv)
    result = build(json.loads(args.release.read_text()), args.cache, args.local_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"Wrote {len(result['evidence'])} policy summaries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
