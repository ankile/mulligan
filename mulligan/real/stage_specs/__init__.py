"""Static stage-label task specs for the real-robot tasks.

A :class:`StageLabelTaskSpec` carries everything task-specific about the S0..SN
progress-stage labels of one real task: the scoring contract (ladder, enums,
response-schema fields), the prompt library (base + append-only rulings), the
proprioceptive physics caps, and the data-source defaults (camera keys, FPS).

These modules are pure data: no Gemini client, no labeler, no ``google.genai``
import. The labeler and the scoring tools in :mod:`mulligan.real.stage_labeling`
import their specs from here.

Registered tasks: ``marker_d2`` (Insert Marker), ``square_d2`` (Thread Nut) and
``routing_d2`` (Route Cable).
"""

from mulligan.real.stage_specs.tasks import (
    StageLabelTaskSpec,
    get_label_task_spec,
    register_label_task_spec,
    registered_label_specs,
)

# Importing the concrete spec modules registers them in the task registry.
from mulligan.real.stage_specs import marker_d2  # noqa: E402,F401  (registers marker_d2)
from mulligan.real.stage_specs import square_d2  # noqa: E402,F401  (registers square_d2)
from mulligan.real.stage_specs import routing_d2  # noqa: E402,F401  (registers routing_d2)

__all__ = [
    "StageLabelTaskSpec",
    "get_label_task_spec",
    "register_label_task_spec",
    "registered_label_specs",
]
