"""Dataset lineage helpers for blinded parent datasets and derived splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

LINEAGE_REL_PATH = "meta/dataset_lineage.json"


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_clean(v) for v in value if v is not None]
    return value


def write_local_lineage(dataset_root: Path, lineage: dict[str, Any]) -> Path:
    path = dataset_root / LINEAGE_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(lineage), indent=2, sort_keys=True) + "\n")
    return path


def push_local_lineage(repo_id: str, dataset_root: Path) -> None:
    from huggingface_hub import HfApi

    from mulligan.tools.lerobot_hub import advance_lerobot_version_tag

    path = dataset_root / LINEAGE_REL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing lineage file: {path}")
    HfApi().upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(path),
        path_in_repo=LINEAGE_REL_PATH,
    )
    advance_lerobot_version_tag(repo_id)


def make_parent_lineage(
    *,
    repo_id: str,
    task: str | None,
    source_type: str,
    ledger_path: str | None,
    ledger_sha256: str | None,
    manifest_path: str | None,
    manifest_sha256: str | None,
    derived_repos: list[str],
    derivation_script: str,
    dataset_role: str = "aggregate_parent",
    trainable: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "blinded_parent",
        "dataset_role": dataset_role,
        "trainable": trainable,
        "repo_id": repo_id,
        "task": task,
        "source_type": source_type,
        "ledger_path": ledger_path,
        "ledger_sha256": ledger_sha256,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "derived_repos": sorted(set(derived_repos)),
        "derivation_script": derivation_script,
    }


def make_split_lineage(
    *,
    repo_id: str,
    task: str | None,
    source_type: str,
    parent_repo_id: str,
    derivation_script: str,
    sidecar_path: str,
    ledger_path: str | None,
    ledger_sha256: str | None,
    manifest_path: str | None,
    manifest_sha256: str | None,
    split_key: str,
    view_family_id: str | None,
    view_id: str | None,
    dataset_role: str = "training_view",
    trainable: bool = True,
    mutually_exclusive_with: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lineage = {
        "schema_version": 1,
        "role": "derived_split",
        "dataset_role": dataset_role,
        "trainable": trainable,
        "repo_id": repo_id,
        "task": task,
        "source_type": source_type,
        "parent_repo_id": parent_repo_id,
        "derivation_script": derivation_script,
        "sidecar_path": sidecar_path,
        "ledger_path": ledger_path,
        "ledger_sha256": ledger_sha256,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "split_key": split_key,
        "view_family_id": view_family_id,
        "view_id": view_id,
        "mutually_exclusive_with": sorted(set(mutually_exclusive_with or [])),
    }
    if extra:
        lineage.update(extra)
    return lineage
