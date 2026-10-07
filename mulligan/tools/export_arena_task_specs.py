"""Export the real-task specs a Policy Arena deployment reads, and optionally upload them.

The Arena's review screens read two tables that hold data exported from this package:

* ``taskSpecs``: per task, the camera layout and display crops of the outcome and stage
  review (from :class:`mulligan.real.lifecycle.tasks.RealTaskSpec` and the station camera
  config in :mod:`mulligan.real.robot.cameras`). Written by the ``taskSpecs:upsert`` mutation.
* ``stageTaskSpecs``: per task and taxonomy version, the stage-label vocabulary the stage
  review form and its consistency checks run on (from
  :class:`mulligan.real.stage_specs.StageLabelTaskSpec`). Written by ``stageTaskSpecs:upsert``.

Each exported row holds exactly the arguments of its mutation (without ``serviceToken``), as
plain JSON. Rows are keyed by the task name the released datasets carry, so they match the
``task`` of datasets registered in the Arena.

Usage::

    # Write the JSON without touching a deployment
    python -m mulligan.tools.export_arena_task_specs --out arena_task_specs.json

    # Upload through the policy-arena client (pip install ./arena/python); the key needs the
    # ingest scope and is read from $POLICY_ARENA_API_KEY or ~/.config/policy-arena/api_key
    python -m mulligan.tools.export_arena_task_specs --url https://<deployment>.convex.cloud

The camera keys come from the station config in effect (``$MULLIGAN_STATION_CONFIG``, else
the packaged example with placeholder serials); the released datasets name cameras by role,
which the Arena matches without the serials.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mulligan.real.lifecycle.tasks import RealTaskSpec, registered_task_specs
from mulligan.real.robot.cameras import (
    STATION_CAMERA_DEFAULT_CROPS,
    STATION_CAMERA_KEYS_BY_ROLE,
    STATION_STORED_FRAME_HW,
)
from mulligan.real.stage_specs import StageLabelTaskSpec, registered_label_specs
from mulligan.real.stage_specs.ladder import StageLevel
from mulligan.real.stage_specs.tasks import taxonomy_fingerprint

SOURCE = "mulligan.tools.export_arena_task_specs"

# Stage-label row fields the review form edits besides the stage, final state, failure
# mode and event fields.
_FREE_TEXT_FIELDS = ("notes",)


def arena_task_name(lifecycle_name: str) -> str:
    """The Arena ``task`` for a lifecycle key (the lifecycle key is the released name)."""
    return lifecycle_name


def task_spec_row(spec: RealTaskSpec, *, source: str = SOURCE) -> dict[str, Any]:
    """Arguments of ``taskSpecs:upsert`` for one lifecycle task."""
    crops = {**STATION_CAMERA_DEFAULT_CROPS, **spec.camera_crop_overrides}
    return {
        "task": arena_task_name(spec.name),
        "task_name": spec.task_name,
        "num_subtask_marks": int(spec.num_subtask_marks),
        "stored_frame_hw": [int(v) for v in STATION_STORED_FRAME_HW],
        "camera_keys_by_role": dict(STATION_CAMERA_KEYS_BY_ROLE),
        "crop_boxes": {role: [int(v) for v in box] for role, box in crops.items()},
        "review_camera_roles": list(spec.consumed_camera_roles),
        "source": source,
    }


def _gate_time_field(spec: StageLabelTaskSpec, level: StageLevel) -> str | None:
    """A single-field gate's timestamp: the declared one, else ``{gate_field}_time_s`` when
    the task has that time field. Multi-field and ungated rungs have none."""
    if level.gate_field is None:
        return None
    if level.gate_time_field is not None:
        return level.gate_time_field
    convention = f"{level.gate_field}_time_s"
    return convention if convention in spec.time_fields else None


def serialize_stage_spec(spec: StageLabelTaskSpec) -> dict[str, Any]:
    """The ``spec`` document of a ``stageTaskSpecs`` row (``ExportedStageSpec`` in
    ``arena/convex/stageConsistency.ts``)."""
    return {
        "task": arena_task_name(spec.name),
        "lifecycle_task": spec.lifecycle_task,
        "taxonomy_version": spec.taxonomy_version,
        "taxonomy_hash": taxonomy_fingerprint(spec),
        "ladder": {
            "header": spec.ladder.header,
            "success_level": spec.ladder.success_level,
            "max_stage": spec.ladder.max_stage,
            "levels": [
                {
                    "sid": level.sid,
                    "text": level.text,
                    "gate_field": level.gate_field,
                    "gate_time_field": _gate_time_field(spec, level),
                    "gate_any_of": list(level.gate_any_of),
                    "gate_all_of": list(level.gate_all_of),
                }
                for level in spec.ladder.levels
            ],
        },
        "failure_modes": list(spec.failure_modes),
        "final_states": list(spec.final_states),
        "success_final_state": spec.success_final_state,
        "released_field": spec.released_field,
        "stage_field": spec.stage_field,
        "final_state_field": spec.final_state_field,
        "failure_mode_field": spec.failure_mode_field,
        "event_fields": [
            {"name": f.name, "kind": f.kind, "description": f.description}
            for f in spec.event_fields
        ],
        "bool_fields": list(spec.bool_fields),
        "time_fields": list(spec.time_fields),
        "editable_fields": [
            spec.stage_field,
            spec.final_state_field,
            spec.failure_mode_field,
            *spec.bool_fields,
            *spec.time_fields,
            *_FREE_TEXT_FIELDS,
        ],
        "constraints": {
            "failure_mode_forbidden_stages": {
                mode: sorted(stages) for mode, stages in spec.failure_mode_forbidden_stages
            },
            "historical_event_failure_modes": dict(spec.historical_event_failure_modes),
            "final_state_requires_gates": {
                state: list(gates) for state, gates in spec.final_state_requires_gates
            },
            "final_state_requires_min_stage": dict(spec.final_state_requires_min_stage),
        },
        "fps": float(spec.fps),
    }


def stage_task_spec_row(
    spec: StageLabelTaskSpec, *, live: bool = True, source: str = SOURCE
) -> dict[str, Any]:
    """Arguments of ``stageTaskSpecs:upsert`` for one stage-label taxonomy version."""
    payload = serialize_stage_spec(spec)
    return {
        "task": payload["task"],
        "taxonomy_version": payload["taxonomy_version"],
        "taxonomy_hash": payload["taxonomy_hash"],
        "live": live,
        "spec": payload,
        "source": source,
    }


def export_task_specs(*, source: str = SOURCE) -> dict[str, list[dict[str, Any]]]:
    """Every registered task, as ``{"task_specs": [...], "stage_task_specs": [...]}``.

    Each registered stage spec is its task's live taxonomy version."""
    return {
        "task_specs": [task_spec_row(spec, source=source) for spec in registered_task_specs()],
        "stage_task_specs": [
            stage_task_spec_row(spec, source=source) for spec in registered_label_specs()
        ],
    }


def upload(export: dict[str, list[dict[str, Any]]], client: Any) -> list[str]:
    """Upsert every row through a ``policy_arena.PolicyArenaClient``; returns the row ids."""
    ids = []
    for row in export["task_specs"]:
        ids.append(client.upsert_task_spec(**row))
    for row in export["stage_task_specs"]:
        ids.append(client.upsert_stage_task_spec(**row))
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="write the exported rows to this JSON file")
    parser.add_argument(
        "--url",
        help="Convex deployment URL (https://<deployment>.convex.cloud) to upload the rows to",
    )
    parser.add_argument(
        "--api-url",
        help="machine API base for a self-hosted or local backend (default: derived from --url)",
    )
    args = parser.parse_args(argv)
    if args.out is None and args.url is None:
        parser.error("pass --out, --url or both")

    export = export_task_specs()
    if args.out is not None:
        args.out.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n")
        print(
            f"Wrote {len(export['task_specs'])} task specs and "
            f"{len(export['stage_task_specs'])} stage task specs to {args.out}"
        )
    if args.url is not None:
        try:
            from policy_arena import PolicyArenaClient
        except ImportError:
            print(
                "Uploading needs the policy-arena client: pip install ./arena/python",
                file=sys.stderr,
            )
            return 1
        client = PolicyArenaClient(args.url, api_url=args.api_url)
        client.whoami()
        ids = upload(export, client)
        print(f"Uploaded {len(ids)} rows to {args.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
