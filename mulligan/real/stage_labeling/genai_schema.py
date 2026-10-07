"""Gemini ``responseSchema`` materialization of a stage-label task spec.

Split out of :mod:`mulligan.real.stage_specs.schema` so the static specs carry no
Gemini code; ``google.genai`` is imported lazily, only when a schema is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mulligan.real.stage_specs.schema import build_schema_fields

if TYPE_CHECKING:
    from google.genai import types

    from mulligan.real.stage_specs.tasks import StageLabelTaskSpec


def to_genai_schema(spec: StageLabelTaskSpec) -> types.Schema:
    """Materialize the genai ``types.Schema`` (lazy-imports ``google.genai``)."""
    from google.genai import types

    kind_to_type = {
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
        "string": types.Type.STRING,
    }
    properties: dict[str, Any] = {}
    for field in build_schema_fields(spec):
        kwargs: dict[str, Any] = {"type": kind_to_type[field.kind]}
        if field.description is not None:
            kwargs["description"] = field.description
        if field.nullable:
            kwargs["nullable"] = True
        if field.enum is not None:
            kwargs["enum"] = list(field.enum)
        if field.minimum is not None:
            kwargs["minimum"] = field.minimum
        if field.maximum is not None:
            kwargs["maximum"] = field.maximum
        properties[field.name] = types.Schema(**kwargs)
    return types.Schema(
        type=types.Type.OBJECT,
        properties=properties,
        required=sorted(properties),
    )
