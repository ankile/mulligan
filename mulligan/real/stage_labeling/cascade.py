"""Focused cascade / boosting-tree refinement nodes for stage labeling.

The monolithic labeler call (the *backbone*) handles the full task ladder in one
prompt. This module adds small boundary classifiers that re-decide only the
sub-judgments that have shown residual errors on reviewed data. Each node has a
narrow question, task-specific evidence, and an optional few-shot bank drawn from
reviewed boundary cases.

Current production nodes (the square_d2 nodes live in
:mod:`mulligan.real.stage_labeling.cascade_pipeline.square_d2`):

- ``marker_d2`` node H2: held marker at holder, strict S3/S4 hole engagement.
- ``marker_d2`` node T: marker ends on table after holder-area contact/release,
  strict S3/S4 hole engagement.

The cascade runner routes only labels near those contested boundaries through
the focused node, then applies an override gate based on sampled vote fraction.
``google.genai`` is imported lazily.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mulligan.real.stage_labeling.assets import task_build_dir
from mulligan.real.stage_labeling.labeler import (
    LabelerConfig,
    file_video_part,
    generate_with_retry,
    inline_video_part,
)

if TYPE_CHECKING:
    from mulligan.real.stage_specs.tasks import StageLabelTaskSpec


@dataclass(frozen=True)
class RefinementNode:
    """One focused boundary classifier.

    ``evidence`` is an ordered list of item keys to attach. ``*_path`` keys and
    ``*_crop``/``*_frame`` keys are resolved to video / image parts respectively.
    ``verdict_to_stage`` maps each allowed verdict to the stage it assigns.
    ``few_shot`` is the boosting bank: ``(episode_index, verdict, why)`` triples,
    each shown as ``exemplar_key`` for that episode with its correct verdict,
    before the target. Entries may include a fourth element with an explicit image
    path, used by calibration scripts that hand-pick a decisive still.
    """

    name: str
    question: str
    evidence: tuple[str, ...]
    verdict_to_stage: dict[str, int]
    few_shot: tuple[tuple[int, str, str], ...] = ()
    exemplar_key: str = "grasp_window_crop"
    samples: int = 5

    @property
    def verdicts(self) -> list[str]:
        return list(self.verdict_to_stage)


# --------------------------------------------------------------------------- #
# marker_d2 nodes. H2 is the active held-at-holder classifier used by MARKER_D2_NODES.
# --------------------------------------------------------------------------- #

_MARKER_D2_H2_WHY = {
    "definite_hole_engaged": (
        "Positive S4 reference: the held marker tip is visibly inside one of the intended "
        "circular holder holes, not merely touching or covering the holder face."
    ),
    "held_at_holder_no_hole": (
        "Hard S3 negative: the held marker reaches/presses the holder area, but the tip is "
        "not visibly inside an intended circular hole. Holder contact alone is S3."
    ),
    "wrong_hole_or_offtarget": (
        "S3 negative subtype: the marker interacts with the wrong/off-target holder opening "
        "or feature rather than the intended circular hole."
    ),
    "slipping_under_force_no_hole": (
        "S3 negative subtype: a tenuous grasp starts slipping or pulling out as force is "
        "applied at the holder; there is holder contact but no stable intended-hole entry."
    ),
    "ambiguous_no_credit": (
        "Conservative S3: the view is ambiguous, the marker occludes the holder face, or "
        "there is only intent/proximity without clear tip-in-hole evidence."
    ),
}

_MARKER_D2_H2_BANK = {
    8: "definite_hole_engaged",
    64: "definite_hole_engaged",
    67: "definite_hole_engaged",
    15: "held_at_holder_no_hole",
    39: "held_at_holder_no_hole",
    41: "held_at_holder_no_hole",
    65: "held_at_holder_no_hole",
    83: "held_at_holder_no_hole",
    92: "held_at_holder_no_hole",
    31: "wrong_hole_or_offtarget",
    82: "slipping_under_force_no_hole",
}

_MARKER_D2_NODE_H2 = RefinementNode(
    name="held_marker_hole_engagement_conservative",
    question=(
        "Focused marker_d2 held-at-holder classifier. The robot is holding the marker near "
        "the RED holder and never releases it. Decide ONLY whether this deserves S4 hole "
        "engagement credit, or which S3 no-credit subtype applies.\n"
        "\n"
        "S4 is STRICT: choose 'definite_hole_engaged' ONLY when the marker TIP is visibly "
        "inside / penetrating one of the intended circular holes in the holder while still "
        "held. The tip must appear to enter the hole opening, not just overlap it in the "
        "camera view.\n"
        "\n"
        "Default to S3 for every non-definite case:\n"
        "- 'held_at_holder_no_hole': the marker is held at, over, in front of, or pressing "
        "the holder/rim/top/face, but the tip is not clearly inside an intended circular hole.\n"
        "- 'wrong_hole_or_offtarget': the marker interacts with the wrong/off-target opening, "
        "edge, slot, holder feature, or non-intended hole.\n"
        "- 'slipping_under_force_no_hole': a weak/tenuous grasp starts slipping, pulling out, "
        "or losing control as the marker contacts the holder; no stable intended-hole entry.\n"
        "- 'ambiguous_no_credit': perspective overlap, occlusion, blur, or intent/proximity "
        "makes hole entry uncertain. Ambiguous is S3, not S4.\n"
        "\n"
        "Use the side video for depth/motion, the wrist video for the gripper-tip-holder "
        "relationship, and the final crops for the endpoint. In reasoning, first name the "
        "decisive moment and state whether the TIP enters an intended circular hole. Then "
        "give the verdict."
    ),
    evidence=("side_path", "wrist_path", "side_final_crop", "wrist_final_crop"),
    verdict_to_stage={
        "definite_hole_engaged": 4,
        "held_at_holder_no_hole": 3,
        "wrong_hole_or_offtarget": 3,
        "slipping_under_force_no_hole": 3,
        "ambiguous_no_credit": 3,
    },
    few_shot=tuple((ep, v, _MARKER_D2_H2_WHY[v]) for ep, v in _MARKER_D2_H2_BANK.items()),
    exemplar_key="wrist_final_crop",
    samples=5,
)

_MARKER_D2_FINAL_TABLE_WHY = {
    "surface_no_hole": (
        "The marker contacted or was released onto the red holder/table area, but the tip "
        "never visibly entered a circular hole. This is holder-area progress only: S3."
    ),
    "hole_entered_then_failed": (
        "Before ending on the table, the marker tip visibly entered a circular hole in the "
        "red holder. This earns S4 max-stage credit even though the insertion later failed."
    ),
}

_MARKER_D2_FINAL_TABLE_BANK = {
    17: "surface_no_hole",
    22: "surface_no_hole",
}

_MARKER_D2_NODE_T = RefinementNode(
    name="final_table_hole_engagement",
    question=(
        "Focused marker_d2 boundary classifier for episodes that END with the marker on the "
        "table after the gripper reached/released near the RED holder. Decide ONLY whether the "
        "marker tip visibly ENTERED one of the circular holes before the marker ended on the "
        "table.\n"
        "The red holder has dark circular hole openings on its vertical face. To count as "
        "hole entry, the marker tip must visibly go into one of those dark circular openings. "
        "Do NOT count the marker lying across the face, touching the rim/side/top of the holder, "
        "sliding under/behind the holder, or being released on top of the holder. A perspective "
        "overlap where the white marker covers a hole but stays outside the face is still "
        "'surface_no_hole'.\n"
        "Verdicts:\n"
        "- 'hole_entered_then_failed': the marker tip is visibly inside / penetrating a circular "
        "hole at some point before it falls or is pulled out. This is S4 max-stage credit.\n"
        "- 'surface_no_hole': the marker only contacts, presses, rests on, drags over, or is "
        "released onto the holder body/top/rim/table area, with no clear tip-in-hole moment. "
        "If the evidence is ambiguous, choose this. This is S3.\n"
        "Use the full side/wrist videos for motion/depth, the release-moment crop for what the "
        "gripper released, and the final crops for the endpoint. In 'reasoning', first say "
        "whether the tip entered a circular hole before the final table state, then give "
        "'verdict'."
    ),
    evidence=(
        "side_path",
        "wrist_path",
        "release_moment_crop",
        "side_final_crop",
        "wrist_final_crop",
    ),
    verdict_to_stage={"surface_no_hole": 3, "hole_entered_then_failed": 4},
    few_shot=tuple(
        (ep, v, _MARKER_D2_FINAL_TABLE_WHY[v]) for ep, v in _MARKER_D2_FINAL_TABLE_BANK.items()
    ),
    exemplar_key="combo_path",
    samples=5,
)

MARKER_D2_NODES: dict[str, RefinementNode] = {"H2": _MARKER_D2_NODE_H2, "T": _MARKER_D2_NODE_T}


_MARKER_D2_HELD_AT_HOLDER_STATES = (
    "marker_in_gripper_at_holder",
    "marker_partially_in_holder_held",
    "marker_fully_seated_held",
)


def marker_d2_route(label: dict[str, Any]) -> str | None:
    """Route a marker_d2 consensus label to its focused refinement node, if any."""
    stage = int(label["max_stage_v2"])
    if stage >= 4 and str(label["final_state"]) == "marker_on_table":
        return "T"
    if (
        stage in (3, 4)
        and not bool(label["marker_released"])
        and not bool(label["grasp_lost"])
        and str(label["final_state"]) in _MARKER_D2_HELD_AT_HOLDER_STATES
    ):
        return "H2"
    return None


def apply_marker_d2_node_result(
    label: dict[str, Any],
    node_key: str,
    result: dict[str, Any],
    *,
    override_min_frac: float = 0.8,
) -> tuple[dict[str, Any], bool]:
    """Apply one marker_d2 node result to a consensus-style parsed label."""
    votes = list(result["votes"])
    win_frac = max(Counter(votes).values()) / len(votes) if votes else 0.0
    if win_frac < override_min_frac:
        return dict(label), False

    out = dict(label)
    verdict = str(result["verdict"])
    before_failure_mode = str(out.get("early_failure_mode_v2", ""))
    if node_key in ("H", "H2", "H3") and verdict in ("hole_engaged", "definite_hole_engaged"):
        if win_frac < 1.0:
            verdict = "ambiguous_no_credit"
    elif node_key == "T" and verdict == "hole_entered_then_failed" and win_frac < 1.0:
        verdict = "surface_no_hole"
    # H/H3 are provenance keys from earlier marker_d2 cascade versions; only H2 is
    # currently routed in MARKER_D2_NODES.
    if node_key in ("H", "H2", "H3"):
        out["marker_released"] = False
        out["marker_released_time_s"] = None
        out["marker_fully_seated"] = False
        out["marker_fully_seated_time_s"] = None
        out["early_failure_mode_v2"] = "timeout_holding_marker"
        if verdict in ("hole_engaged", "definite_hole_engaged"):
            out["max_stage_v2"] = 4
            out["final_state"] = "marker_partially_in_holder_held"
            if out.get("hole_alignment_time_s") is None:
                out["hole_alignment_time_s"] = out.get("insertion_contact_time_s")
        elif verdict in (
            "held_at_holder_no_hole",
            "ambiguous_no_credit",
            "wrong_hole_or_offtarget",
            "slipping_under_force_no_hole",
        ):
            out["max_stage_v2"] = 3
            out["final_state"] = "marker_in_gripper_at_holder"
            out["hole_alignment_time_s"] = None
            if verdict == "wrong_hole_or_offtarget":
                out["early_failure_mode_v2"] = "wrong_hole_partial_insert"
        else:
            raise ValueError(f"unknown marker_d2 {node_key} verdict {verdict!r}")
    elif node_key == "T":
        out["marker_fully_seated"] = False
        out["marker_fully_seated_time_s"] = None
        out["marker_released"] = False
        out["final_state"] = "marker_on_table"
        if verdict == "hole_entered_then_failed":
            out["max_stage_v2"] = 4
            if out.get("hole_alignment_time_s") is None:
                out["hole_alignment_time_s"] = out.get("insertion_contact_time_s")
            if bool(out.get("grasp_lost")):
                out["early_failure_mode_v2"] = "marker_slipped_during_insertion"
            elif before_failure_mode in (
                "jammed_partial_insert",
                "marker_released_at_holder",
                "marker_slipped_during_insertion",
                "released_partial_insert_not_seated",
            ):
                out["early_failure_mode_v2"] = before_failure_mode
            else:
                out["early_failure_mode_v2"] = "released_partial_insert_not_seated"
        elif verdict == "surface_no_hole":
            out["max_stage_v2"] = 3
            if before_failure_mode in (
                "marker_released_at_holder",
                "marker_slipped_during_insertion",
                "marker_slipped_from_gripper",
            ):
                out["early_failure_mode_v2"] = before_failure_mode
            else:
                out["early_failure_mode_v2"] = "other"
            out["hole_alignment_time_s"] = None
        else:
            raise ValueError(f"unknown marker_d2 T verdict {verdict!r}")
    else:
        raise ValueError(f"unknown marker_d2 node {node_key!r}")

    out["needs_human_review"] = True
    out["notes"] = (
        str(out.get("notes", ""))
        + f" [marker_d2-cascade:{node_key} verdict={verdict} votes={votes}]"
    ).strip()
    return out, out != label


# --------------------------------------------------------------------------- #
# Execution.
# --------------------------------------------------------------------------- #


def _schema(node: RefinementNode) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "verdict": {"type": "string", "enum": node.verdicts},
        },
        "required": ["reasoning", "verdict"],
        "propertyOrdering": ["reasoning", "verdict"],
    }


def _media_part(types: Any, key: str, value: str, config: LabelerConfig, client: Any) -> Any:
    if key.endswith("_path") or key == "combo_path":  # a video
        if config.video_transport == "file":
            return file_video_part(client, Path(value), config.video_fps)
        return inline_video_part(Path(value), config.video_fps)
    return types.Part(  # a still / crop
        inline_data=types.Blob(data=Path(value).read_bytes(), mime_type="image/png")
    )


def _exemplar_path(
    spec: StageLabelTaskSpec, config: LabelerConfig, node: RefinementNode, episode: int
) -> Path:
    key_to_suffix = {
        "combo_path": "combo.mp4",
        "grasp_window_crop": "grasp_window_crop.png",
        "wrist_final_crop": "wrist_final_crop.png",
        "side_final_crop": "side_final_crop.png",
        "release_moment_crop": "release_crop.png",
    }
    try:
        suffix = key_to_suffix[node.exemplar_key]
    except KeyError:
        raise ValueError(f"unsupported exemplar_key {node.exemplar_key!r}") from None
    build_dir = config.few_shot_build_dir or config.build_dir
    return task_build_dir(spec, build_dir) / "assets" / f"episode_{episode:03d}_{suffix}"


def few_shot_media_paths(
    spec: StageLabelTaskSpec, config: LabelerConfig, node: RefinementNode
) -> list[Path]:
    """Media file of every few-shot entry of ``node`` (as :func:`build_node_parts` reads it)."""
    return [
        Path(entry[3]) if len(entry) > 3 else _exemplar_path(spec, config, node, entry[0])
        for entry in node.few_shot
    ]


def build_node_parts(
    spec: StageLabelTaskSpec, config: LabelerConfig, client: Any, node: RefinementNode, item: dict
) -> list[Any]:
    from google.genai import types

    parts: list[Any] = [types.Part(text=node.question)]
    if node.few_shot:
        parts.append(
            types.Part(
                text=(
                    "LABELED REFERENCE EXAMPLES (media + correct verdict) — use them as the "
                    "calibration standard for this exact boundary:"
                )
            )
        )
        for entry in node.few_shot:
            ep, verdict, why = entry[:3]
            # Optional 4th element: a custom exemplar image (e.g. a still extracted
            # at an operator-supplied DECISIVE timestamp) overriding the default
            # heuristic grasp_window_crop. Show, don't tell.
            media = Path(entry[3]) if len(entry) > 3 else _exemplar_path(spec, config, node, ep)
            parts.append(types.Part(text=f"REFERENCE — verdict={verdict}. {why}"))
            parts.append(_media_part(types, node.exemplar_key, str(media), config, client))
        parts.append(types.Part(text="END REFERENCES. Now judge the TARGET episode:"))
    for key in node.evidence:
        value = item.get(key)
        if value is None:
            continue
        parts.append(types.Part(text=f"{key}:"))
        if key in {"sensor_trace", "gripper_series_1hz"}:
            parts.append(types.Part(text=json.dumps(value, indent=2)))
            continue
        parts.append(_media_part(types, key, str(value), config, client))
    return parts


def run_node(
    spec: StageLabelTaskSpec, config: LabelerConfig, client: Any, node: RefinementNode, item: dict
) -> dict[str, Any]:
    """Self-consistency vote of one node; returns {verdict, stage, votes}."""
    from google.genai import types

    parts = build_node_parts(spec, config, client, node, item)
    contents = types.Content(role="user", parts=parts)
    gen_config = types.GenerateContentConfig(
        temperature=config.temperature,
        response_mime_type="application/json",
        response_schema=_schema(node),
        max_output_tokens=config.max_output_tokens,
        media_resolution=(
            types.MediaResolution(f"MEDIA_RESOLUTION_{config.media_resolution.upper()}")
            if config.media_resolution
            else None
        ),
    )

    def _one_sample(_: int) -> str:
        raw, _ = generate_with_retry(
            client,
            config.model,
            contents,
            gen_config,
            validate=lambda t: json.loads(t)["verdict"],
        )
        return json.loads(raw)["verdict"]

    # Self-consistency samples are independent -> run them concurrently.
    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        votes: list[str] = list(pool.map(_one_sample, range(node.samples)))
    verdict = Counter(votes).most_common(1)[0][0]
    return {"verdict": verdict, "stage": node.verdict_to_stage[verdict], "votes": votes}
