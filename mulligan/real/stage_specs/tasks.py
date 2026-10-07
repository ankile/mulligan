"""Task registry for VLM stage labeling.

A :class:`StageLabelTaskSpec` captures everything task-specific the otherwise
task-agnostic Gemini labeling pipeline needs: the data source (HF dataset +
events CSV + camera keys + FPS), the scoring contract (stage ladder, failure-mode
and final-state enums, the response-schema event fields), the prompt library
(base + append-only rulings), and the proprioceptive physics caps. Adding a task
= adding one ``StageLabelTaskSpec`` and registering it, mirroring
:class:`mulligan.real.lifecycle.tasks.RealTaskSpec` (which this cross-references via
``lifecycle_task`` for geometry/binning).

Concrete specs live in sibling modules (e.g. :mod:`mulligan.real.stage_specs.marker_d2`)
and register themselves on import; the package ``__init__`` imports them so
``get_label_task_spec`` sees a populated registry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from mulligan.real.stage_specs.ladder import StageLadder
from mulligan.real.stage_specs.prompts import PromptLibrary
from mulligan.real.stage_specs.schema import SchemaField, build_schema_fields
from mulligan.real.stage_specs.sensor_constraints import SensorConstraintRule


@dataclass(frozen=True)
class StageLabelTaskSpec:
    """Everything task-specific the VLM stage labeler needs for one real task."""

    name: str
    """Label-task key, e.g. ``"marker_d2"`` (matches the lifecycle key)."""

    lifecycle_task: str
    """Cross-reference into :func:`mulligan.real.lifecycle.get_task_spec` for geometry."""

    """Manipulated-object noun for prompt text, e.g. ``"marker"``."""

    # --- data source ---
    dataset_repo_id: str
    """Default HF dataset repo holding the episodes (overridable per run)."""

    events_csv: Path | None
    """Default per-episode sensor/arm events CSV (gripper hold/release/reopened,
    length). ``None`` for the released specs: the events CSV is a per-run pipeline
    intermediate, so callers pass it explicitly."""

    fps: float
    """Native dataset FPS (frame<->seconds, 1 Hz binning)."""

    side_camera_key: str
    """Fixed-view camera HF key, e.g. ``observation.images.side_1``."""

    wrist_camera_key: str
    """Wrist-mounted camera HF key (for a task with no wrist view this is a second
    fixed view, e.g. routing's ``side_2``; label it via ``wrist_camera_label``)."""

    gripper_state_column: str
    """Parquet column for the proprioceptive aperture, e.g.
    ``observation.state.gripper_position``."""

    gripper_close_threshold: float
    """Rising-edge threshold marking a jaw close (matches the events pipeline)."""

    # --- scoring contract ---
    taxonomy_version: str
    """Schema (taxonomy) version string identifying THIS ladder+enum+field
    vocabulary, e.g. routing's ``"s10_v1"``. Every stored label row (gold CSVs,
    ``.label_history.jsonl`` events) is keyed to a taxonomy version so labels survive taxonomy iteration (stage splits/
    removals) without silent remaps. Changing the ladder/enums/fields without
    bumping this string fails loudly at registration against the committed
    ``taxonomy_pins.json`` fingerprint."""

    ladder: StageLadder
    failure_modes: tuple[str, ...]
    final_states: tuple[str, ...]
    success_final_state: str
    """The final_state that full success (the ladder's success level + a true
    release) requires, e.g. ``"marker_fully_seated_released"``. Used by the
    downstream success-endpoint flag and scoring; must be in ``final_states``."""

    released_field: str
    """The event-boolean field meaning "object released" (e.g. ``"marker_released"``
    / ``"nut_released"``). The downstream release-vs-sensor contradiction flag keys
    off this; must be one of ``bool_fields``."""

    # --- response-schema field names + descriptions (task-flavored) ---
    stage_field: str
    stage_field_description: str
    final_state_field: str
    final_state_description: str
    failure_mode_field: str
    failure_mode_description: str
    event_fields: tuple[SchemaField, ...]
    """Ordered booleans/times between the stage integer and the final-state enum."""

    # --- prompts ---
    prompts: PromptLibrary
    default_prompt_variant: str

    # --- proprioceptive physics caps ---
    sensor_rules: tuple[SensorConstraintRule, ...]

    # --- camera-role labels shown in the prompt/asset text ---
    # The role tokens the labeler prints for the two camera streams (e.g. "SIDE
    # camera video:", the combo drawtext). Defaults reproduce the marker/square
    # wording byte-for-byte; a task whose two streams are not a side+wrist pair
    # (routing's two OPPOSING side views) overrides them, e.g. "SIDE-1"/"SIDE-2".
    side_camera_label: str = "SIDE"
    wrist_camera_label: str = "WRIST"

    # --- optional additional camera streams (dual mode only) ---
    # Extra synchronized views fed to the labeler AFTER side+wrist, each as a video
    # + final frame with its own role label. Empty by default (marker/square are
    # side+wrist only, unchanged). routing feeds its wrist_left here so the model
    # sees the close-up seat/orientation view the two opposing side views cannot
    # resolve (off_axis vs beside vs engaged). Keys/labels are positional pairs.
    extra_camera_keys: tuple[str, ...] = ()
    extra_camera_labels: tuple[str, ...] = ()

    # --- calibrated failure-mode <-> stage consistency (optional) ---
    # Declarative ``failure_mode -> forbidden max_stage set``: a row whose
    # ``failure_mode`` is paired with a ``max_stage`` in its forbidden set is
    # inconsistent (e.g. a seating-error mode cannot coexist with a
    # completed-seat rung). Kept as a tuple of ``(mode, frozenset)`` pairs so the
    # frozen spec stays hashable. Empty => no failure<->stage constraint (the
    # pre-calibration happy-path era).
    failure_mode_forbidden_stages: tuple[tuple[str, frozenset[int]], ...] = ()

    # Some terminal failure modes explicitly mean that a boolean state was
    # achieved earlier and then lost. In that narrow case the terminal bool is
    # False while its event timestamp remains present as historical evidence.
    # Each pair is ``(failure_mode, bool_field)``; a row with the active mode
    # carries exactly ``bool=False`` plus a present paired
    # ``{bool_field}_time_s``. Empty keeps the ordinary bool<=>time contract.
    historical_event_failure_modes: tuple[tuple[str, str], ...] = ()

    # A ``final_state`` that shows the object physically IN a seat at the last frame
    # implies that seat's gate bool is True (you cannot end in-clip without having
    # seated). Only the safe direction is encoded: states that DON'T show the object
    # seated (in gripper, free, at-clip-unseated) carry no requirement, because a
    # high-water seat that later falls out is legitimate. Tuple of
    # ``(final_state, (gate_field, ...))`` pairs for hashability. Empty => no check.
    final_state_requires_gates: tuple[tuple[str, tuple[str, ...]], ...] = ()

    # A ``final_state`` that shows the object AT/IN a target in the last frame implies
    # the ladder REACHED that target: the final frame is itself an >= min_stage state, so
    # the high-water ``max_stage`` cannot be below it (e.g. "rope at the first clip" =>
    # reached clip1 => max_stage >= S3). Tuple of ``(final_state, min_stage)`` pairs.
    # Empty => no check.
    final_state_requires_min_stage: tuple[tuple[str, int], ...] = ()

    # --- release (jaw-reopen) detection thresholds ---
    # Defaults reproduce the marker generator EXACTLY (the equivalence pin); a task
    # whose release is a brief partial-open-then-reclose (the Nut task: the gripper opens
    # just enough to drop a seated nut, then closes on air) relaxes these so a
    # shallow, recovering end-of-trace dip off the hold plateau still counts as a
    # release instead of being misread as held-to-end.
    release_final_abs_max: float = 0.5
    """A final-decline release requires the last frame below this absolute aperture."""

    release_plateau_margin: float = 0.3
    """A final-decline release requires the last frame this far below the hold plateau."""

    @property
    def variants(self) -> tuple[str, ...]:
        return self.prompts.variants

    def system_prompt(self, variant: str) -> str:
        return self.prompts.assemble(variant)

    @property
    def bool_fields(self) -> tuple[str, ...]:
        """Event boolean field names (consensus/UI/scoring ``BOOL_FIELDS``)."""
        return tuple(f.name for f in self.event_fields if f.kind == "boolean")

    @property
    def time_fields(self) -> tuple[str, ...]:
        """Event time field names (``TIME_FIELDS``)."""
        return tuple(f.name for f in self.event_fields if f.kind == "number")


TAXONOMY_PINS_PATH = Path(__file__).parent / "taxonomy_pins.json"


def taxonomy_fingerprint(spec: StageLabelTaskSpec) -> str:
    """Content hash of the spec's label VOCABULARY (its taxonomy/schema).

    Covers what a stored label row's meaning depends on: the ladder (rung ids,
    texts, gates), the enums, the field names AND their human-facing
    descriptions (the review form's guidance — a description edit moves the
    boundary a human labels against), ``released_field``, and the declarative
    consistency-constraint maps. Prompt text, camera keys, sensor rules, and
    dataset paths are labeling-PIPELINE concerns: a prompt ruling changes what
    a model predicts, not what a stored label means, so they are excluded.
    """
    canon = {
        "ladder": {
            "success_level": spec.ladder.success_level,
            "levels": [
                {
                    "sid": lvl.sid,
                    "text": lvl.text,
                    "gate_field": lvl.gate_field,
                    "gate_time_field": lvl.gate_time_field,
                    "gate_any_of": list(lvl.gate_any_of),
                    "gate_all_of": list(lvl.gate_all_of),
                }
                for lvl in spec.ladder.levels
            ],
        },
        "failure_modes": list(spec.failure_modes),
        "final_states": list(spec.final_states),
        "success_final_state": spec.success_final_state,
        "released_field": spec.released_field,
        "stage_field": spec.stage_field,
        "stage_field_description": spec.stage_field_description,
        "final_state_field": spec.final_state_field,
        "final_state_description": spec.final_state_description,
        "failure_mode_field": spec.failure_mode_field,
        "failure_mode_description": spec.failure_mode_description,
        "event_fields": [
            {"name": f.name, "kind": f.kind, "description": f.description}
            for f in spec.event_fields
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
    }
    payload = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=1)
def _taxonomy_pins() -> dict[str, dict[str, str]]:
    return json.loads(TAXONOMY_PINS_PATH.read_text())


def _check_taxonomy_pin(spec: StageLabelTaskSpec) -> None:
    """Fail loudly unless the spec's vocabulary hash matches its committed pin.

    The pin file maps ``task -> {taxonomy_version: fingerprint}``. Old versions'
    pins are permanent (they date historical label rows); a vocabulary edit
    therefore requires bumping ``taxonomy_version`` and appending a new pin —
    editing the ladder/enums in place under an existing version raises here at
    import time, before any label can be produced under a lying version string.
    """
    fingerprint = taxonomy_fingerprint(spec)
    task_pins = _taxonomy_pins().get(spec.name)
    if task_pins is None:
        raise ValueError(
            f"{spec.name}: no taxonomy pins recorded in {TAXONOMY_PINS_PATH.name}. "
            f"Append {{{spec.taxonomy_version!r}: {fingerprint!r}}} for this task."
        )
    pinned = task_pins.get(spec.taxonomy_version)
    if pinned is None:
        raise ValueError(
            f"{spec.name}: taxonomy_version {spec.taxonomy_version!r} has no pin in "
            f"{TAXONOMY_PINS_PATH.name}. If this is a deliberate new taxonomy version, "
            f"append its fingerprint {fingerprint!r}; never reuse or edit old pins."
        )
    if pinned != fingerprint:
        raise ValueError(
            f"{spec.name}: taxonomy fingerprint {fingerprint} does not match the pin "
            f"{pinned} for version {spec.taxonomy_version!r}. The ladder/enums/fields "
            "changed without a taxonomy_version bump — bump the version and append a "
            "new pin instead of mutating the existing vocabulary."
        )


_LABEL_SPECS: dict[str, StageLabelTaskSpec] = {}


# Event fields are partitioned into bool_fields / time_fields by kind; any other
# kind would be silently dropped from both, so the kinds are constrained here.
_EVENT_FIELD_KINDS = frozenset({"boolean", "number"})


def register_label_task_spec(spec: StageLabelTaskSpec) -> StageLabelTaskSpec:
    """Register a spec, validating it loudly. Returns the spec for module-level use."""
    if spec.name in _LABEL_SPECS:
        raise ValueError(f"duplicate StageLabelTaskSpec name {spec.name!r}")
    from mulligan.real.lifecycle import get_task_spec

    try:
        get_task_spec(spec.lifecycle_task)
    except KeyError as exc:
        raise ValueError(
            f"{spec.name}: lifecycle_task {spec.lifecycle_task!r} does not resolve via "
            "mulligan.real.lifecycle.get_task_spec"
        ) from exc
    if "none" not in spec.failure_modes:
        raise ValueError(f"{spec.name}: failure_modes must include 'none' for full success")
    if spec.success_final_state not in spec.final_states:
        raise ValueError(
            f"{spec.name}: success_final_state {spec.success_final_state!r} not in final_states"
        )
    if spec.released_field not in spec.bool_fields:
        raise ValueError(f"{spec.name}: released_field {spec.released_field!r} not in bool_fields")
    if spec.default_prompt_variant not in spec.variants:
        raise ValueError(
            f"{spec.name}: default_prompt_variant {spec.default_prompt_variant!r} "
            f"not in {spec.variants}"
        )
    bad_kinds = [(f.name, f.kind) for f in spec.event_fields if f.kind not in _EVENT_FIELD_KINDS]
    if bad_kinds:
        # A typo'd kind would otherwise vanish from bool_fields/time_fields with no error.
        raise ValueError(
            f"{spec.name}: event_fields must be 'boolean' or 'number'; got {bad_kinds}"
        )
    covered = set(spec.bool_fields) | set(spec.time_fields)
    all_event = {f.name for f in spec.event_fields}
    if covered != all_event:
        raise ValueError(
            f"{spec.name}: bool_fields|time_fields {sorted(covered)} do not cover "
            f"event_fields {sorted(all_event)}"
        )
    # Duplicate schema field names would silently overwrite each other in the
    # genai schema dict (and collapse in the sorted required-key set), losing a
    # field with no error. Fail loud at registration instead.
    names = [f.name for f in build_schema_fields(spec)]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{spec.name}: duplicate schema field names {dupes}")
    # Sensor-rule field references and emitted final_states are opaque inside the
    # rule closures, so each rule declares them; validate here so a typo'd field
    # name or an emit not in the enum fails loud rather than no-op'ing at run time.
    field_names = set(names)
    final_states = set(spec.final_states)
    for rule in spec.sensor_rules:
        bad_refs = [f for f in rule.field_refs if f not in field_names]
        if bad_refs:
            raise ValueError(
                f"{spec.name}: sensor rule {rule.name!r} references unknown schema "
                f"fields {bad_refs}"
            )
        bad_emits = [s for s in rule.emits_final_states if s not in final_states]
        if bad_emits:
            raise ValueError(
                f"{spec.name}: sensor rule {rule.name!r} emits final_states {bad_emits} "
                f"not in the task's final_states"
            )
    # An EXPLICIT gate_time_field must name a real time field, else the rung's
    # gate points at a timestamp the response schema never asks for.
    # Convention-resolved names are exempt: a single gate may
    # legitimately have no timestamp.
    for lvl in spec.ladder.gated_levels:
        if lvl.gate_time_field is not None and lvl.gate_time_field not in spec.time_fields:
            raise ValueError(
                f"{spec.name}: S{lvl.sid} gate_time_field {lvl.gate_time_field!r} is not "
                f"one of time_fields {list(spec.time_fields)}"
            )
    historical_modes = [mode for mode, _ in spec.historical_event_failure_modes]
    duplicate_historical_modes = sorted(
        {mode for mode in historical_modes if historical_modes.count(mode) > 1}
    )
    if duplicate_historical_modes:
        raise ValueError(
            f"{spec.name}: duplicate historical-event failure modes {duplicate_historical_modes}"
        )
    for mode, bool_field in spec.historical_event_failure_modes:
        if mode not in spec.failure_modes:
            raise ValueError(f"{spec.name}: historical-event mode {mode!r} is not in failure_modes")
        if bool_field not in spec.bool_fields:
            raise ValueError(
                f"{spec.name}: historical-event mode {mode!r} references unknown "
                f"bool field {bool_field!r}"
            )
        time_field = f"{bool_field}_time_s"
        if time_field not in spec.time_fields:
            raise ValueError(
                f"{spec.name}: historical-event bool {bool_field!r} has no paired "
                f"time field {time_field!r}"
            )
    # Last, after every structural check: the vocabulary must match its
    # committed taxonomy pin (see _check_taxonomy_pin) — an edited ladder/enum
    # under an unbumped taxonomy_version fails here at import time.
    _check_taxonomy_pin(spec)
    _LABEL_SPECS[spec.name] = spec
    return spec


def get_label_task_spec(name: str) -> StageLabelTaskSpec:
    """Look up a stage-label task spec by lifecycle key (e.g. ``"marker_d2"``)."""
    try:
        return _LABEL_SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown stage-label task {name!r}; registered: {sorted(_LABEL_SPECS)}"
        ) from None


def registered_label_specs() -> tuple[StageLabelTaskSpec, ...]:
    """All registered specs, in registration order."""
    return tuple(_LABEL_SPECS.values())
