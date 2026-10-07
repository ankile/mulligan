"""Write a task's minimal stage-events CSV from its HF dataset.

Produces the per-episode events CSV the labeler needs (gripper hold/release/reopened +
episode_length), plus summarizer context (arm / strict outcome / state coords) parsed
from the dataset's canonical, outcome-edited ``results.json``. The gripper logic lives
in :mod:`mulligan.real.stage_labeling.events`; this CLI picks the task spec, resolves and
validates the rollout metadata, and writes the CSV.

    python -m mulligan.real.stage_labeling.prepare_events --task marker_d2 \\
        --dataset-repo-id mulligan/real-marker-d2-r00-eval \\
        --output outputs/real/stage_events/marker_d2_r0_heldout/minimal_stage_events.csv

The gripper threshold and FPS come from the spec. ``--dry-run`` prints the plan
without downloading.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from mulligan.real.eval.outcome_results import (
    DEFAULT_OUTCOME_OVERRIDES,
    canonicalize_results_payload,
    load_frame_outcomes_from_hf,
    load_hf_json,
    load_outcome_edit_record,
    subtask_frames_from_record,
    validate_results_against_frame_outcomes,
)
from mulligan.real.lifecycle.tasks import get_task_spec
from mulligan.real.stage_specs import get_label_task_spec


def load_rollout_meta(
    spec,
    arm_map: dict[str, str] | None,
    *,
    outcome_overrides_filename: str | None = DEFAULT_OUTCOME_OVERRIDES,
    require_outcome_overrides: bool = False,
) -> dict[int, dict[str, Any]] | None:
    """Per-episode meta from the canonical edited results: arm, strict outcome, the
    policy-phase ``num_steps``, and the task's manifest state columns.

    If an outcome-editor record is present, it is applied before parsing. The resolved
    outcomes/num_steps are validated against frame labels so a stale root ``results.json``
    cannot silently feed the stage pipeline. Returns None if the dataset has no
    ``results.json`` (a teleop collection; a gripper-only CSV is still valid).

    ``num_steps`` is the policy-phase boundary the events builder clips to: the recording
    keeps running through the operator's physical reset, and a gripper re-open in that
    tail is a reset artifact, not a policy release.
    """
    from huggingface_hub.errors import EntryNotFoundError

    try:
        raw_payload = load_hf_json(spec.dataset_repo_id, "results.json", revision="main")
    except EntryNotFoundError:
        print("  no results.json on the dataset; writing gripper-only events")
        return None
    record = None
    if outcome_overrides_filename is not None:
        record = load_outcome_edit_record(
            spec.dataset_repo_id,
            outcome_overrides_filename,
            revision="main",
            required=require_outcome_overrides,
        )
    payload, reconciliation = canonicalize_results_payload(
        raw_payload,
        outcome_edit_record=record,
        overrides_filename=outcome_overrides_filename or "",
    )
    # The recorded subtask reward frames let _detect_frame_outcome tolerate the operator's
    # mid-episode seat spikes (reward=1.0 at a labeled frame).
    validate_results_against_frame_outcomes(
        payload,
        load_frame_outcomes_from_hf(
            spec.dataset_repo_id,
            revision="main",
            subtask_frames_by_episode=subtask_frames_from_record(record),
        ),
    )
    if reconciliation is not None:
        print(
            "  outcome edits applied: "
            f"{reconciliation['episodes_reviewed']} reviewed, "
            f"{reconciliation['outcome_class_changes']} class changes, "
            f"{len(reconciliation['success_flips'])} success flips"
        )

    coord_keys = get_task_spec(spec.lifecycle_task).manifest_keys
    name_by_pid = {s["policy_id"]: s["name"] for s in payload.get("summary", [])}
    meta: dict[int, dict[str, Any]] = {}
    for r in payload.get("rollouts", []):
        name = name_by_pid.get(r["policy_id"], str(r["policy_id"]))
        row: dict[str, Any] = {
            "policy_short": (arm_map or {}).get(name, name),
            "original_outcome": str(r.get("outcome", "")),
            "num_steps": int(r["num_steps"]),
        }
        for key in coord_keys:
            if key in r:
                row[key] = r[key]
        meta[int(r["episode_index"])] = row
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="registered task, e.g. square_d2")
    parser.add_argument(
        "--dataset-repo-id", default=None, help="default: the spec's dataset_repo_id"
    )
    parser.add_argument("--output", type=Path, required=True, help="events CSV to write")
    parser.add_argument(
        "--arm-map", type=str, default=None, help="optional JSON {policy_name: arm_short}"
    )
    parser.add_argument(
        "--outcome-overrides-filename",
        default=DEFAULT_OUTCOME_OVERRIDES,
        help=(
            "outcome-editor record to apply when present; use 'none' to disable "
            f"(default: {DEFAULT_OUTCOME_OVERRIDES})"
        ),
    )
    parser.add_argument(
        "--require-outcome-overrides",
        action="store_true",
        help="fail if --outcome-overrides-filename is absent on the dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    spec = get_label_task_spec(args.task)
    if args.dataset_repo_id:
        spec = dataclasses.replace(spec, dataset_repo_id=args.dataset_repo_id)
    arm_map = json.loads(args.arm_map) if args.arm_map else None
    outcome_overrides_filename = args.outcome_overrides_filename
    if str(outcome_overrides_filename).lower() in {"", "0", "none", "null"}:
        outcome_overrides_filename = None

    if args.dry_run:
        print(
            f"[dry-run] task={spec.name}\n  dataset={spec.dataset_repo_id}\n"
            f"  gripper_col={spec.gripper_state_column} threshold={spec.gripper_close_threshold} "
            f"fps={spec.fps}\n  outcome_overrides={outcome_overrides_filename}\n"
            f"  -> {args.output}"
        )
        return

    import pandas as pd

    from mulligan.real.stage_labeling.events import (
        FULL_RECORDED_EPISODE,
        policy_steps_from_rollout_meta,
        write_minimal_events,
    )

    rollout_meta = load_rollout_meta(
        spec,
        arm_map,
        outcome_overrides_filename=outcome_overrides_filename,
        require_outcome_overrides=args.require_outcome_overrides,
    )
    # No results.json => no policy phase to clip to (a teleop demo collection, where the
    # recording IS the episode). Anything with rollouts is clipped to num_steps.
    policy_steps = (
        FULL_RECORDED_EPISODE
        if rollout_meta is None
        else policy_steps_from_rollout_meta(rollout_meta)
    )
    written = write_minimal_events(
        spec, args.output, rollout_meta=rollout_meta, policy_steps=policy_steps
    )
    df = pd.read_csv(written)
    print(f"{len(df)} episodes -> {written}")
    print(f"  reopened_at_end: {int(df['gripper_reopened_at_end'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
