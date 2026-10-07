"""Structured-output schema for VLM stage labeling, derived from a task spec.

The labeler asks Gemini for a single JSON object per episode, constrained by a
``responseSchema``. That schema is the contract every downstream stage of the
pipeline reads (consensus, adjudication, the correction UI, scoring), so it must
be the single source of truth for the label record's fields rather than being
re-declared per script.

This module is the pure-data layer (:class:`SchemaField` +
:func:`build_schema_fields`); it never imports ``google.genai``. The Gemini
materialization, :func:`mulligan.real.stage_labeling.genai_schema.to_genai_schema`,
lives with the labeler. Field order and attributes are pinned by
``tests/real/test_stage_specs_golden.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mulligan.real.stage_specs.tasks import StageLabelTaskSpec


# Field kinds, as plain strings (the labeler maps them to genai types).
_INTEGER = "integer"
_NUMBER = "number"
_BOOLEAN = "boolean"
_STRING = "string"


@dataclass(frozen=True)
class SchemaField:
    """One property of the response schema.

    ``nullable`` is only ever set on time (number) fields in the original schema;
    boolean/string/integer fields leave it unset (``None`` on the genai object).
    """

    name: str
    kind: str
    description: str | None = None
    nullable: bool = False
    enum: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None


# Structural fields shared by every stage-labeling task, in their canonical
# positions (episode_index leads; confidence/needs_human_review/notes trail).
# Descriptions are task-agnostic and match the marker schema verbatim.
_EPISODE_INDEX = SchemaField("episode_index", _INTEGER)
_CONFIDENCE = SchemaField("confidence", _STRING, enum=("low", "medium", "high"))
_NEEDS_REVIEW = SchemaField(
    "needs_human_review",
    _BOOLEAN,
    description="Set true when evidence was ambiguous or contradictory",
)
_NOTES = SchemaField(
    "notes",
    _STRING,
    description="Brief evidence summary; mention anything not verifiable from video",
)


def build_schema_fields(spec: StageLabelTaskSpec) -> tuple[SchemaField, ...]:
    """Assemble the ordered response-schema fields for a task.

    Order: ``episode_index``, the stage integer (bounded by the ladder), the
    task's event fields, the final-state enum, the failure-mode enum, then the
    shared ``confidence`` / ``needs_human_review`` / ``notes`` trailer.
    """
    stage = SchemaField(
        spec.stage_field,
        _INTEGER,
        description=spec.stage_field_description,
        minimum=0,
        maximum=spec.ladder.max_stage,
    )
    final_state = SchemaField(
        spec.final_state_field,
        _STRING,
        description=spec.final_state_description,
        enum=tuple(spec.final_states),
    )
    failure_mode = SchemaField(
        spec.failure_mode_field,
        _STRING,
        description=spec.failure_mode_description,
        enum=tuple(spec.failure_modes),
    )
    return (
        _EPISODE_INDEX,
        stage,
        *spec.event_fields,
        final_state,
        failure_mode,
        _CONFIDENCE,
        _NEEDS_REVIEW,
        _NOTES,
    )


def required_keys(spec: StageLabelTaskSpec) -> list[str]:
    """Sorted required-key list, matching the original ``required=sorted(...)``."""
    return sorted(field.name for field in build_schema_fields(spec))
