"""List HuggingFace dataset repo IDs from the Policy Arena registry.

Reads the deployment from $POLICY_ARENA_URL (see PolicyArenaClient).

Usage:
    python -m policy_arena.get_datasets
    python -m policy_arena.get_datasets --task franka_nut_assembly_square
    python -m policy_arena.get_datasets --source teleop rollout dagger
    python -m policy_arena.get_datasets --task franka_pick_cube --source teleop dagger
"""

import argparse

from policy_arena import PolicyArenaClient


def main():
    parser = argparse.ArgumentParser(description="List registered HF datasets")
    parser.add_argument("--task", type=str, default=None, help="Filter by task name")
    parser.add_argument(
        "--source",
        nargs="+",
        default=None,
        help="Filter by source type(s), e.g. teleop rollout dagger",
    )
    parser.add_argument(
        "--role",
        nargs="+",
        default=None,
        help="Filter by dataset role(s), e.g. training_view aggregate_parent eval_session",
    )
    parser.add_argument(
        "--trainable",
        choices=["true", "false"],
        default=None,
        help="Filter by whether the dataset is intended for training.",
    )
    args = parser.parse_args()

    client = PolicyArenaClient()
    trainable = None if args.trainable is None else args.trainable == "true"
    datasets = client.list_datasets(
        task=args.task,
        source_types=args.source,
        dataset_roles=args.role,
        trainable=trainable,
    )

    for d in datasets:
        print(d["repo_id"])


if __name__ == "__main__":
    main()
