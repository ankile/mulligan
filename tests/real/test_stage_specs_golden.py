"""Golden outputs of the static stage-label specs (``mulligan.real.stage_specs``).

``data/golden/stage_specs.json`` pins, for the three paper tasks, the sha256 of every assembled
prompt variant, the taxonomy fingerprint, the response-schema field order, the sensor-rule names,
and a sensor-constraint battery (every stage x final state x jaw trace x bool pattern) hashed over
its outputs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import pytest

import mulligan.real.stage_specs as ss
from mulligan.real.stage_specs.schema import SchemaField, build_schema_fields, required_keys
from mulligan.real.stage_specs.sensor_constraints import (
    apply_sensor_constraints,
    cap_grasp_when_jaws_never_closed,
    invalidate_final_state_when_jaws_never_closed,
)
from mulligan.real.stage_specs.tasks import register_label_task_spec, taxonomy_fingerprint

GOLDEN = json.loads((Path(__file__).parent / "data/golden/stage_specs.json").read_text())
TASKS = ("marker_d2", "square_d2", "routing_d2")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _battery(spec) -> dict:
    fields = build_schema_fields(spec)
    bool_fields = list(spec.bool_fields)
    stage_levels = [lvl.sid for lvl in spec.ladder.levels]
    outs = []
    for stage, fs, close, reopened, bools in product(
        stage_levels, list(spec.final_states), [None, 3.5, 9.8], [True, False], [0, 1, 2]
    ):
        trace = {
            "jaw_close_time_s": close,
            "jaw_reopen_time_s": None if close is None else (6.0 if reopened else None),
            "jaws_reopened_before_episode_end": reopened,
            "episode_duration_s": 10.0,
            "gripper_min": 0.0,
            "gripper_max": 1.0,
        }
        parsed = {}
        for i, f in enumerate(fields):
            if f.name == spec.stage_field:
                parsed[f.name] = stage
            elif f.name == spec.final_state_field:
                parsed[f.name] = fs
            elif f.name == spec.failure_mode_field:
                parsed[f.name] = spec.failure_modes[(stage + i) % len(spec.failure_modes)]
            elif f.name in bool_fields:
                parsed[f.name] = bool((i + bools) % 2)
            elif f.kind == "number":
                parsed[f.name] = float(i % 10)
            elif f.name == "needs_human_review":
                parsed[f.name] = bool(bools == 2)
            elif f.name == "notes":
                parsed[f.name] = "" if bools else "prior note."
            elif f.kind == "boolean":
                parsed[f.name] = False
            elif f.kind == "integer":
                parsed[f.name] = 0
            else:
                parsed[f.name] = "high"
        try:
            outs.append(
                apply_sensor_constraints(spec.sensor_rules, {"sensor_trace": trace}, parsed)
            )
        except Exception as exc:  # the recorded battery stores the same error kinds
            outs.append({"error": type(exc).__name__})
    return {"n": len(outs), "sha256": _sha(json.dumps(outs, sort_keys=True))}


def _record(name: str) -> dict:
    spec = ss.get_label_task_spec(name)
    return {
        "taxonomy_version": spec.taxonomy_version,
        "taxonomy_fingerprint": taxonomy_fingerprint(spec),
        "default_prompt_variant": spec.default_prompt_variant,
        "prompt_sha256": {v: _sha(spec.system_prompt(v)) for v in spec.variants},
        "schema_fields": [f.name for f in build_schema_fields(spec)],
        "required_keys": required_keys(spec),
        "sensor_rules": [r.name for r in spec.sensor_rules],
        "failure_modes": list(spec.failure_modes),
        "final_states": list(spec.final_states),
        "ladder_success_level": spec.ladder.success_level,
        "side_camera_key": spec.side_camera_key,
        "wrist_camera_key": spec.wrist_camera_key,
        "extra_camera_keys": list(spec.extra_camera_keys),
        "gripper_close_threshold": spec.gripper_close_threshold,
        "release_final_abs_max": spec.release_final_abs_max,
        "release_plateau_margin": spec.release_plateau_margin,
        "sensor_battery": _battery(spec),
    }


@pytest.mark.parametrize("name", TASKS)
def test_stage_spec_matches_golden(name):
    got = json.loads(json.dumps(_record(name)))
    want = GOLDEN[name]
    assert set(got) == set(want)
    for key in want:
        assert got[key] == want[key], f"{name}.{key} drifted from the golden output"


def test_only_paper_tasks_registered():
    assert [s.name for s in ss.registered_label_specs()] == list(TASKS)
    for retired in ("insert_marker_d1", "Square_D1"):
        with pytest.raises(KeyError, match="unknown stage-label task"):
            ss.get_label_task_spec(retired)


def test_every_lifecycle_task_has_a_label_spec():
    from mulligan.real.lifecycle.tasks import registered_task_specs

    for spec in registered_task_specs():
        assert ss.get_label_task_spec(spec.name).lifecycle_task == spec.name


def test_stage_specs_import_no_labeler_or_genai():
    """``stage_specs`` is pure data: importing it pulls in neither the labeler nor genai."""
    import subprocess

    code = (
        "import sys, mulligan.real.stage_specs; "
        "bad = [m for m in sys.modules if 'stage_labeling' in m or m.startswith('google')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------------- #
# Registration guards.
# --------------------------------------------------------------------------- #

MARKER = "marker_d2"


def test_bool_and_time_fields_partition_event_fields():
    spec = ss.get_label_task_spec(MARKER)
    kinds = {f.name: f.kind for f in spec.event_fields}
    assert set(spec.bool_fields) == {n for n, k in kinds.items() if k == "boolean"}
    assert set(spec.time_fields) == {n for n, k in kinds.items() if k == "number"}
    assert not (set(spec.bool_fields) & set(spec.time_fields))


def test_register_rejects_bad_event_field_kind():
    spec = ss.get_label_task_spec(MARKER)
    bad = dataclasses.replace(
        spec,
        name="bad_kind_probe",
        event_fields=(*spec.event_fields, SchemaField("weird", "string")),
    )
    with pytest.raises(ValueError, match="event_fields must be"):
        register_label_task_spec(bad)


def test_register_rejects_duplicate_field_names():
    spec = ss.get_label_task_spec(MARKER)
    bad = dataclasses.replace(
        spec,
        name="dup_field_probe",
        event_fields=(*spec.event_fields, SchemaField("episode_index", "boolean")),
    )
    with pytest.raises(ValueError, match="duplicate schema field names"):
        register_label_task_spec(bad)


def test_register_rejects_sensor_rule_unknown_field():
    bad_rule = cap_grasp_when_jaws_never_closed(
        stage_field="max_stage_typo",
        grasp_field="grasp_acquired",
        seated_field="marker_fully_seated",
        released_field="marker_released",
    )
    spec = ss.get_label_task_spec(MARKER)
    bad = dataclasses.replace(spec, name="bad_rule_field_probe", sensor_rules=(bad_rule,))
    with pytest.raises(ValueError, match="references unknown schema"):
        register_label_task_spec(bad)


def test_register_rejects_sensor_rule_emit_not_in_final_states():
    spec = ss.get_label_task_spec(MARKER)
    rule = invalidate_final_state_when_jaws_never_closed(final_state_field="final_state")
    without_unclear = tuple(s for s in spec.final_states if s != "unclear")
    bad = dataclasses.replace(
        spec, name="bad_emit_probe", final_states=without_unclear, sensor_rules=(rule,)
    )
    with pytest.raises(ValueError, match="emits final_states"):
        register_label_task_spec(bad)


def test_register_rejects_taxonomy_edit_without_version_bump():
    spec = ss.get_label_task_spec(MARKER)
    bad = dataclasses.replace(
        spec, name="taxonomy_probe", failure_modes=(*spec.failure_modes, "new_mode")
    )
    with pytest.raises(ValueError, match="no taxonomy pins"):
        register_label_task_spec(bad)
