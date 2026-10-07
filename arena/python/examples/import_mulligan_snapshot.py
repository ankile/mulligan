"""Load the Mulligan real-robot evaluations into your own Policy Arena as example data.

Reads the frozen snapshot file ``arena/data/release.json`` and, for each evaluation block of the
three real-robot tasks, registers the block's ``mulligan/*`` evaluation dataset and submits one
evaluation session (one round per initial state, one result per policy). Videos then stream from
the public Hugging Face datasets. Simulation tasks carry no per-episode results and are skipped.

    export POLICY_ARENA_URL=https://<your-deployment>.convex.cloud
    export POLICY_ARENA_API_KEY=<key-id>.<secret>      # a key with the ingest scope
    python arena/python/examples/import_mulligan_snapshot.py arena/data/release.json

Each session uses the block id as its idempotency key, so a rerun returns the existing sessions
instead of adding duplicates. Policies are registered as ``mulligan-release/<policy id>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policy_arena import (
    DatasetInput,
    PolicyArenaClient,
    PolicyInput,
    RoundInput,
    RoundResultInput,
)


def sessions(release: dict):
    """Yield (task, block, policies, rounds) for every real-robot evaluation block."""
    for task in release["tasks"]:
        if task["domain"] != "real":
            continue
        by_id = {p["id"]: p for p in task["policies"]}
        for block in task["blocks"]:
            used = sorted({r["policyId"] for s in block["starts"] for r in s["results"]})
            policies = [
                PolicyInput(
                    name=by_id[pid]["id"],
                    model_id=f"mulligan-release/{pid}",
                    environment=task["id"],
                )
                for pid in used
            ]
            rounds = [
                RoundInput(
                    round_index=i,
                    results=[
                        RoundResultInput(
                            model_id=f"mulligan-release/{r['policyId']}",
                            success=r["success"],
                            episode_index=r["episode"],
                            num_frames=r["frames"],
                            num_subtask_marks=r["marks"],
                        )
                        for r in start["results"]
                    ],
                )
                for i, start in enumerate(block["starts"])
            ]
            yield task, block, policies, rounds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("release", type=Path, help="arena/data/release.json")
    ap.add_argument("--dry-run", action="store_true", help="print what would be submitted")
    args = ap.parse_args()
    release = json.loads(args.release.read_text())
    arena = None if args.dry_run else PolicyArenaClient()
    registered: set[str] = set()
    for task, block, policies, rounds in sessions(release):
        summary = f"{task['id']} {block['id']}: {len(policies)} policies, {len(rounds)} rounds"
        if arena is None:
            print("would submit", summary)
            continue
        if block["dataset"] not in registered:
            arena.register_dataset(
                DatasetInput(
                    repo_id=block["dataset"],
                    name=block["dataset"].split("/", 1)[1],
                    task=task["id"],
                    source_type="eval",
                    environment=task["id"],
                    dataset_role="eval_session",
                    trainable=False,
                )
            )
            registered.add(block["dataset"])
        session = arena.submit_eval_session(
            block["dataset"],
            policies,
            rounds,
            notes=f"Mulligan release {release['version']}: {block.get('label', block['id'])}",
            session_mode="fixed_grid",
            idempotency_key=f"mulligan-release-{block['id']}"[:200],
        )
        print("submitted", summary, "->", session)


if __name__ == "__main__":
    main()
