"""HuggingFace Hub helpers for the dataset collectors and push tools."""

from __future__ import annotations

import argparse

from huggingface_hub.errors import RepositoryNotFoundError

# License written to the card of every pushed dataset (LeRobot's own default is apache-2.0).
DEFAULT_DATASET_LICENSE = "mit"


def add_hf_namespace_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--hf-namespace``, the Hub user/org that pushed datasets land under."""
    parser.add_argument(
        "--hf-namespace",
        default=None,
        help=(
            "HuggingFace user or organization for pushed datasets. Required with "
            "--push-to-hub when the dataset name has no NAMESPACE/ prefix."
        ),
    )


def add_license_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--license``, the license id written to a pushed dataset's card."""
    parser.add_argument(
        "--license",
        type=str,
        default=DEFAULT_DATASET_LICENSE,
        help="License id written to the dataset card (Hugging Face license identifier; "
        f"default: {DEFAULT_DATASET_LICENSE})",
    )


def resolve_push_repo_id(name: str, hf_namespace: str | None) -> str:
    """Return ``NAMESPACE/NAME`` for a Hub push.

    A name that already contains ``/`` is used as-is. A bare name needs an explicit
    namespace: there is no default user, so a push never lands in whichever account
    happens to be logged in.
    """
    if "/" in name:
        return name
    if not hf_namespace:
        raise ValueError(
            f"Dataset name {name!r} has no NAMESPACE/ prefix; pass --hf-namespace "
            "(or a full NAMESPACE/NAME repo id) to push it to the HuggingFace Hub."
        )
    return f"{hf_namespace}/{name}"


def ensure_dataset_repo(hub_api, repo_id: str, *, private: bool) -> bool:
    """Ensure a dataset repository exists.

    Returns True when a new repository was created. Auth, network, and other
    Hub errors intentionally propagate instead of being treated as "not found".
    """
    try:
        hub_api.repo_info(repo_id=repo_id, repo_type="dataset")
        return False
    except RepositoryNotFoundError:
        hub_api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
        )
        return True
