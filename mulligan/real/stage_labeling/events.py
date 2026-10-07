"""Derive a minimal stage-events CSV from a dataset's gripper trace (genai-free).

The labeler needs a per-episode events CSV with the proprioceptive gripper
columns (``gripper_hold_time_s`` / ``gripper_release_time_s`` /
``gripper_reopened_at_end`` / ``episode_length``). Those are derived from the
``observation.state.gripper_position`` channel by a rising-edge close detector
and a release-after-hold detector — logic that is identical across DROID tasks,
so it lives here rather than being re-cloned per task.

Only the close threshold and FPS are task parameters (from the spec); the
release heuristics (plateau detection, release-in-progress-at-episode-end) are
gripper-physics-general. The dataset-specific bits — the arm/outcome/state
columns parsed from a ``results.json`` — are passed in by the caller as
``rollout_meta``.

**Policy-phase boundary (the ``policy_steps`` contract).** A recorded real
episode does NOT end when the policy does: the operator's physical reset runs on
for up to ~49 s inside the same recording. Gripper events derived over the whole
recording therefore attribute the operator's reset re-open to the policy — the
``gripper_reopened_at_end`` flag the VLM labeler reads as "the robot released".
``build_minimal_events`` consequently REQUIRES an explicit ``policy_steps``
boundary: either a per-episode ``{episode_index: policy frame count}`` mapping,
or the explicit :data:`FULL_RECORDED_EPISODE` sentinel for datasets that have no
policy phase at all (teleop demo collections, where the recording IS the demo).
There is no default: an omitted boundary would silently produce
reset-contaminated events.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from mulligan.real.stage_labeling.assets import (
    frame_parquet_rel_paths,
    read_frame_parquet,
    repo_file,
    valid_frame_prefix,
)

if TYPE_CHECKING:
    from mulligan.real.stage_specs.tasks import StageLabelTaskSpec

#: Explicit opt-out of policy-phase truncation, for datasets whose recording has no
#: policy phase to clip to (teleop demo collections). Passing this is a deliberate
#: statement that the whole ``is_valid`` prefix IS the episode of interest — it is
#: NOT a default, and must never be used to paper over a missing boundary on an
#: eval-rollout dataset.
FULL_RECORDED_EPISODE = "full_recorded_episode"

#: A per-episode policy-phase boundary, or the explicit full-recording sentinel.
PolicySteps = Mapping[int, int] | str


def first_crossing(gripper: pd.Series, threshold: float) -> int | None:
    """First frame index where the aperture rises to/above ``threshold`` (jaw close)."""
    above = gripper[gripper >= threshold]
    return None if above.empty else int(above.index[0])


def release_after_hold(
    gripper: pd.Series,
    close_pos: int | None,
    threshold: float,
    final_abs_max: float = 0.5,
    plateau_margin: float = 0.3,
) -> int | None:
    """Frame index of a release after the hold at ``close_pos``, or None.

    A release is either the last sub-threshold crossing within the final 3 frames,
    or a final decline off the hold plateau that ends below ``final_abs_max``
    absolute and ``>= plateau_margin`` below the plateau (release-in-progress when
    recording stops before the jaws fully open). The defaults ``(0.5, 0.3)``
    reproduce the marker generator's logic exactly (pinned by the equivalence
    test); a task whose release is a shallow partial-open-then-reclose passes
    relaxed thresholds (from the spec) so that dip is not misread as held-to-end.
    """
    if close_pos is None:
        return None
    after = gripper.iloc[close_pos:]
    below = after[after < threshold]
    if not below.empty and int(below.index[-1]) >= len(gripper) - 3:
        return int(below.index[-1])
    g = gripper.to_numpy()
    hold_vals = g[close_pos:][g[close_pos:] > 0.5]
    if hold_vals.size < 5:
        return None
    plateau = float(pd.Series(hold_vals).median())
    final = float(g[-1])
    if final >= final_abs_max or (plateau - final) < plateau_margin:
        return None
    i = len(g) - 1
    while i > close_pos and g[i - 1] >= g[i] - 0.02 and g[i - 1] > final - 0.02:
        if g[i - 1] >= plateau - 0.2:
            break
        i -= 1
    return i


def derive_gripper_events(
    gripper: pd.Series,
    threshold: float,
    fps: float,
    final_abs_max: float = 0.5,
    plateau_margin: float = 0.3,
) -> dict[str, Any]:
    """The proprioceptive event columns for one episode's aperture series."""
    close_pos = first_crossing(gripper, threshold)
    release_pos = release_after_hold(gripper, close_pos, threshold, final_abs_max, plateau_margin)
    return {
        "episode_length": int(len(gripper)),
        "num_steps": int(len(gripper)),
        "gripper_hold_frame": close_pos,
        "gripper_hold_time_s": None if close_pos is None else round(close_pos / fps, 2),
        "gripper_release_frame": release_pos,
        "gripper_release_time_s": None if release_pos is None else round(release_pos / fps, 2),
        "gripper_reopened_at_end": release_pos is not None,
    }


def load_frame_gripper(spec: StageLabelTaskSpec) -> pd.DataFrame:
    """Per-frame ``episode_index / frame_index / <gripper col>`` from the dataset's
    frame parquets.

    Real eval repos can have non-contiguous data shard indices when a policy was
    dropped or an interrupted session was resumed. Enumerate actual HF files
    instead of scanning ``file-000, file-001, ...`` until the first gap.
    """
    col = spec.gripper_state_column
    frames = [
        read_frame_parquet(repo_file(spec, rel_path), [col])
        for rel_path in frame_parquet_rel_paths(spec)
    ]
    return pd.concat(frames, ignore_index=True)


def policy_steps_from_rollout_meta(rollout_meta: dict[int, dict[str, Any]]) -> dict[int, int]:
    """``{episode_index: num_steps}`` from a rollout-meta mapping that carries it.

    Fails loud when an episode's meta has no ``num_steps`` — that is precisely the
    silent-full-length hole this module's ``policy_steps`` contract closes.
    """
    steps: dict[int, int] = {}
    for episode, meta in rollout_meta.items():
        if "num_steps" not in meta or meta["num_steps"] is None:
            raise KeyError(
                f"rollout_meta episode {episode} has no num_steps; cannot derive the "
                "policy-phase boundary (pass FULL_RECORDED_EPISODE only for datasets "
                "with no policy phase)"
            )
        steps[int(episode)] = int(meta["num_steps"])
    return steps


def _episode_policy_steps(policy_steps: PolicySteps, episode_index: int, valid_len: int) -> int:
    """Resolve + validate the policy-phase frame count for one episode."""
    if policy_steps is FULL_RECORDED_EPISODE or policy_steps == FULL_RECORDED_EPISODE:
        return valid_len
    if not isinstance(policy_steps, Mapping):
        raise TypeError(
            "policy_steps must be a {episode_index: num_steps} mapping or the "
            f"FULL_RECORDED_EPISODE sentinel, got {type(policy_steps).__name__}"
        )
    if episode_index not in policy_steps:
        raise KeyError(
            f"policy_steps missing episode {episode_index}; every episode present in the "
            "dataset needs an explicit policy-phase boundary"
        )
    num_steps = int(policy_steps[episode_index])
    if num_steps <= 0:
        raise RuntimeError(
            f"episode {episode_index}: policy_steps must be positive, got {num_steps}"
        )
    if num_steps > valid_len:
        raise RuntimeError(
            f"episode {episode_index}: policy_steps={num_steps} exceeds dataset is_valid "
            f"prefix length {valid_len}"
        )
    return num_steps


def build_minimal_events(
    spec: StageLabelTaskSpec,
    rollout_meta: dict[int, dict[str, Any]] | None = None,
    *,
    policy_steps: PolicySteps,
) -> pd.DataFrame:
    """Build the minimal events DataFrame for ``spec``'s dataset.

    The gripper columns are derived from the frame parquets alone (enough for the
    labeler). If ``rollout_meta`` (e.g. arm / strict outcome / state coords parsed
    from a dataset ``results.json``) is given, it is merged per episode for the
    downstream summarizer; missing meta for a present episode fails loud.

    ``policy_steps`` is REQUIRED and carries the policy-phase boundary (see the
    module docstring): a ``{episode_index: num_steps}`` mapping, or
    :data:`FULL_RECORDED_EPISODE` for a dataset with no policy phase. When
    ``rollout_meta`` also carries ``num_steps``, the two must agree.
    """
    frames = load_frame_gripper(spec)
    col = spec.gripper_state_column
    rows = []
    for episode_index, group in frames.groupby("episode_index"):
        episode_index = int(episode_index)
        valid = valid_frame_prefix(group, episode_index=episode_index)
        num_steps = _episode_policy_steps(policy_steps, episode_index, len(valid))
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "policy_short": "",
            "original_outcome": "",
        }
        if rollout_meta is not None:
            if episode_index not in rollout_meta:
                raise KeyError(f"rollout_meta missing episode {episode_index}")
            meta_steps = rollout_meta[episode_index].get("num_steps")
            if meta_steps is not None and int(meta_steps) != num_steps:
                raise RuntimeError(
                    f"episode {episode_index}: rollout_meta num_steps={int(meta_steps)} "
                    f"disagrees with policy_steps={num_steps}"
                )
            row.update(rollout_meta[episode_index])
        valid = valid.iloc[:num_steps]
        gripper = valid.sort_values("frame_index")[col].reset_index(drop=True)
        row.update(
            derive_gripper_events(
                gripper,
                spec.gripper_close_threshold,
                spec.fps,
                spec.release_final_abs_max,
                spec.release_plateau_margin,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)


def write_minimal_events(
    spec: StageLabelTaskSpec,
    out_path: Path,
    rollout_meta: dict[int, dict[str, Any]] | None = None,
    *,
    policy_steps: PolicySteps,
) -> Path:
    df = build_minimal_events(spec, rollout_meta, policy_steps=policy_steps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
