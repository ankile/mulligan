"""Initial-state manifest and round-plan helpers of the blind (manifest) eval.

The blind eval protocol: every policy is rolled out from the same manifest start
in a per-round shuffled order under anonymous labels (A, B, C, ...), so the
operator cannot track policy identity across rounds. The round plans are written
to ``results.json`` before any rollout, so an interrupted run resumes with the same
assignment. The entry point is :mod:`mulligan.real.eval.manifest_eval`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from mulligan.real.collect.initial_states import (
    InitialStateTarget,
    _manifest_keys,
    _target_value,
    load_manifest_targets,
)


def _load_eval_initial_state_manifest(
    path: Path, *, expected_task: str
) -> tuple[list[InitialStateTarget], dict]:
    # Route through the SHARED collection loader so blind eval inherits the
    # marker_d2 5-key (+ holder-on-grid registry validation) and square/marker
    # parsing for free, and the two parsers can never re-drift. The shared loader
    # validates each row's ``source`` against an arm whitelist; blind eval has no
    # ``--arm`` flag, so we synthesize that whitelist from the manifest's own
    # declared arms / row sources (see ``manifest_arm_keys``). The returned
    # ``(list[InitialStateTarget], payload)`` shape is exactly what blind eval's
    # call site consumes.
    return load_manifest_targets(
        path, expected_task=expected_task, model_id=lambda key: f"eval:{key}"
    )


def _copy_eval_initial_state_manifest(source_path: Path, dataset_path: Path) -> Path:
    target_path = dataset_path / "meta" / "initial_states_manifest.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        existing = json.loads(target_path.read_text())
        current = json.loads(source_path.read_text())
        if existing != current:
            raise RuntimeError(
                f"Existing dataset manifest snapshot {target_path} differs from "
                f"{source_path}; refusing to resume against a different eval target list."
            )
    else:
        shutil.copyfile(source_path, target_path)
    return target_path


def _load_round_plans(results_path: Path) -> list[dict]:
    """Round plans persisted in ``results.json`` (empty list for a fresh run)."""
    if not results_path.exists():
        return []
    data = json.loads(results_path.read_text())
    raw_round_plans = data.get("round_plans", [])
    if raw_round_plans is None:
        raw_round_plans = []
    if not isinstance(raw_round_plans, list):
        raise ValueError(f"{results_path}: round_plans must be a list")
    return raw_round_plans


def _round_plans_by_round(round_plans: list[dict]) -> dict[int, dict]:
    plans_by_round: dict[int, dict] = {}
    for plan in round_plans:
        if not isinstance(plan, dict):
            raise ValueError(f"round plan must be a JSON object, got {plan!r}")
        round_num = int(plan["round"])
        if round_num in plans_by_round:
            raise ValueError(f"duplicate round plan for round {round_num}")
        plans_by_round[round_num] = plan
    return plans_by_round


def _eval_initial_state_keys(manifest_meta: dict | None = None) -> list[str]:
    if manifest_meta is None:
        return []
    # Route through the shared key validator so the supported set (marker_d2, square_d2,
    # routing_d2) stays single-sourced and cannot drift.
    return _manifest_keys(manifest_meta)


def _eval_initial_state_features(manifest_meta: dict | None = None) -> dict:
    features = {
        "manifest_idx": {"dtype": "int64", "shape": (1,), "names": ["manifest_idx"]},
    }
    for key in _eval_initial_state_keys(manifest_meta):
        features[key] = {"dtype": "float32", "shape": (1,), "names": [key]}
    return features


def _eval_initial_state_frame_fields(
    target: InitialStateTarget, manifest_meta: dict | None = None
) -> dict[str, np.ndarray]:
    fields = {"manifest_idx": np.array([target.manifest_idx], dtype=np.int64)}
    for key in _eval_initial_state_keys(manifest_meta):
        # Reuse the shared per-key resolver (raw-backed) so holder_x/holder_y and
        # any future placement keys are saved without a second value-dispatch table.
        fields[key] = np.array([_target_value(target, key)], dtype=np.float32)
    return fields
