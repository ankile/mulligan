"""Stage ladder for a VLM-labeled real-robot task.

A task's progress is scored on an ordered ladder of stages S0..SN (the marker
task uses S0..S7: no-approach -> pregrasp -> grasp -> transport -> tip-in-hole ->
partial-release -> seated-held -> seated-released). The ladder is the spine of
the whole pipeline: it bounds the ``max_stage`` integer in the response schema,
defines the "full success" rung, and feeds the ``S0: ...`` description block of
every system prompt. Keeping it as data (not prose baked into one prompt) lets a
new task declare its own rungs once and have the schema and prompt scaffolding
follow.

This module is intentionally free of any ``google.genai`` dependency; the
genai ``types.Schema`` materialization lives in
:mod:`mulligan.real.stage_labeling.genai_schema`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageLevel:
    """One rung of the ladder. ``text`` is the description line shown in prompts
    (without the leading ``"- S{sid}: "`` marker, which :meth:`StageLadder.render`
    adds).

    A rung's gate is the load-bearing boolean evidence for having *reached* it. It
    comes in three shapes (at most one may be set):

    * ``gate_field`` — a SINGLE boolean field (e.g. marker S6 = ``marker_seated``).
      Its contract: ``max_stage >= sid  <=>  field is True`` AND the paired
      ``field is True  <=>  {field}_time_s present``.
    * ``gate_any_of`` — reached iff ANY of these bools is True (e.g. routing S6 =
      "one clip seated" = ``first_clip_seated`` OR ``second_clip_seated``).
    * ``gate_all_of`` — reached iff ALL of these bools are True (e.g. routing S10 =
      "both clips seated" = ``first_clip_seated`` AND ``second_clip_seated``).

    For the multi-field (any/all) shapes there is no single gate timestamp — each
    constituent bool is a physical fact carrying its OWN ``{field}_time_s``. Rungs with no distinct
    boolean (intermediate transport/align rungs) leave all unset and carry no gate."""

    sid: int
    text: str
    gate_field: str | None = None
    gate_time_field: str | None = None
    gate_any_of: tuple[str, ...] = ()
    gate_all_of: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        set_gates = sum(
            (self.gate_field is not None, bool(self.gate_any_of), bool(self.gate_all_of))
        )
        if set_gates > 1:
            raise ValueError(
                f"S{self.sid}: at most one of gate_field / gate_any_of / gate_all_of may be set"
            )
        if self.gate_time_field is not None and self.gate_field is None:
            raise ValueError(f"S{self.sid}: gate_time_field requires a single gate_field")

    @property
    def gated(self) -> bool:
        return self.gate_field is not None or bool(self.gate_any_of) or bool(self.gate_all_of)


@dataclass(frozen=True)
class StageLadder:
    """Ordered S0..SN ladder with a designated full-success rung."""

    levels: tuple[StageLevel, ...]
    success_level: int
    header: str = "Stage ladder (label the MAXIMUM stage reached):"

    def __post_init__(self) -> None:
        sids = [lvl.sid for lvl in self.levels]
        if sids != sorted(sids) or sids != list(range(len(sids))):
            raise ValueError(f"stage ids must be 0..N contiguous and sorted, got {sids}")
        if self.success_level not in sids:
            raise ValueError(f"success_level {self.success_level} not in ladder {sids}")

    @property
    def max_stage(self) -> int:
        return self.levels[-1].sid

    @property
    def gated_levels(self) -> tuple[StageLevel, ...]:
        """Rungs that declare a gate (single / any_of / all_of), in ascending ``sid``
        order — the load-bearing boolean gates of the ladder."""
        return tuple(lvl for lvl in self.levels if lvl.gated)

    def render(self) -> str:
        """Render the ladder block as it appears in a system prompt."""
        lines = [self.header]
        lines += [f"- S{lvl.sid}: {lvl.text}" for lvl in self.levels]
        return "\n".join(lines)
