#!/usr/bin/env python3
"""
Push a local LeRobot dataset to HuggingFace Hub.

Usage:
    python -m mulligan.tools.push_dataset --repo-id <namespace>/<name> --root <dataset-path> [options]

Examples:
    # Push a local dataset under a full repo id
    python -m mulligan.tools.push_dataset --repo-id my-org/my-demos --root ./data/my_demos

    # Bare name + explicit namespace, as a private dataset
    python -m mulligan.tools.push_dataset --repo-id my-demos --hf-namespace my-org \
        --root ./data/my_demos --private

    # Force push local changes (e.g. after an outcome review or other local edits)
    python -m mulligan.tools.push_dataset --repo-id my-org/my-demos --root ./data/my_demos --force

Without --force the dataset is loaded through ``LeRobotDataset``, which may sync files from the Hub
first; use --force to upload the local files as they are. --root is the dataset directory (containing
data/, meta/, videos/); --repo-id is the Hub name and may differ from the local folder name.
"""

import argparse
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mulligan.real.collect.hf_utils import (
    add_hf_namespace_arg,
    add_license_arg,
    resolve_push_repo_id,
)
from mulligan.tools.lerobot_hub import advance_lerobot_version_tag


def main():
    parser = argparse.ArgumentParser(description="Push a local LeRobot dataset to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help=(
            "Repository ID on HuggingFace Hub: 'namespace/dataset-name', or a bare "
            "'dataset-name' together with --hf-namespace"
        ),
    )
    add_hf_namespace_arg(parser)
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Full path to the dataset directory (should contain data/, meta/, videos/, etc.)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on HuggingFace Hub",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git branch to push to (default: uses dataset version)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Upload the local files as they are, without loading the dataset from the Hub first "
            "(use after local edits such as an outcome review). Creates a commit even if no file changed."
        ),
    )
    add_license_arg(parser)

    args = parser.parse_args()
    try:
        repo_id = resolve_push_repo_id(args.repo_id, args.hf_namespace)
    except ValueError as exc:
        parser.error(str(exc))
    hub_api = HfApi()

    # Resolve paths
    dataset_path = Path(args.root)

    print(f"Loading dataset from: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Verify it's a valid LeRobot dataset
    required_dirs = ["data", "meta"]
    for dir_name in required_dirs:
        if not (dataset_path / dir_name).exists():
            raise FileNotFoundError(
                f"Invalid LeRobot dataset: missing '{dir_name}/' directory in {dataset_path}"
            )

    # Load dataset info without triggering downloads
    # We use a local-only metadata loader to avoid overwriting local changes
    from lerobot.datasets.io_utils import load_info

    info = load_info(dataset_path)

    print("\nDataset info (from local files):")
    print(f"  Episodes: {info['total_episodes']}")
    print(f"  Frames: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")
    print(f"  Features: {list(info['features'].keys())}")

    # Push to hub
    print(f"\nPushing dataset to HuggingFace Hub: {repo_id}")
    print(f"  Private: {args.private}")
    print(f"  License: {args.license}")
    if args.branch:
        print(f"  Branch: {args.branch}")
    if args.force:
        print("  Force: True (will create commit even if files unchanged)")

    if args.force:
        # Create a timestamp file to force a new commit
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_file = dataset_path / ".last_push_timestamp"
        timestamp_file.write_text(f"Last pushed: {timestamp}\n")
        print("  Created timestamp file to force new commit")

        commit_message = f"Force update dataset - {timestamp}"

        # Create repo if it doesn't exist
        hub_api.create_repo(
            repo_id=repo_id,
            private=args.private,
            repo_type="dataset",
            exist_ok=True,
        )

        if args.branch:
            hub_api.create_branch(
                repo_id=repo_id,
                branch=args.branch,
                repo_type="dataset",
                exist_ok=True,
            )

        # Upload with custom commit message
        hub_api.upload_folder(
            repo_id=repo_id,
            folder_path=str(dataset_path),
            repo_type="dataset",
            revision=args.branch,
            ignore_patterns=["images/"],
            commit_message=commit_message,
        )

        # Update dataset card
        from lerobot.datasets.utils import create_lerobot_dataset_card

        card = create_lerobot_dataset_card(
            dataset_info=info,
            license=args.license,
        )
        card.push_to_hub(repo_id=repo_id, repo_type="dataset", revision=args.branch)

        # Create version tag (required for LeRobot to load the dataset)
        codebase_version = info.get("codebase_version")
        if codebase_version:
            advance_lerobot_version_tag(
                repo_id,
                revision=args.branch or "main",
                tag=codebase_version,
            )

        # Clean up timestamp file after push
        timestamp_file.unlink()
        print("  Cleaned up timestamp file")
    else:
        # Load dataset for standard push (this will trigger downloads if needed)
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=str(dataset_path),
        )
        dataset.push_to_hub(
            branch=args.branch,
            license=args.license,
            private=args.private,
        )

    print(f"\n✓ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
