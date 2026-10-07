"""A scripted DAgger operator that replays recorded human segments.

The sim DAgger collector (:mod:`mulligan.sim.collect.dagger`) reads all human
input through an :class:`~mulligan.sim.collect.utils.Operator`. This one plays
back a recorded DAgger collection (a ``mulligan/sim-*-dagger-*`` dataset), so a
round runs without a person, e.g. in CI:

    python -m mulligan.sim.collect.dagger ... --auto-save-on-success --headless \\
        --operator mulligan.sim.collect.replay_operator:make_replay_operator \\
        --operator-kwargs '{"repo_id": "mulligan/sim-square-narrow-c01-dagger-mixed",
                            "revision": "<40-hex revision>", "max_episodes": 2}'

How a recorded episode is used:

* It is chosen by start state: the nut pose (and the Square_D1 peg position) of
  its first frame is matched against the start the collector placed
  (:class:`~mulligan.sim.collect.utils.EpisodeStart`); the recorded episode must
  begin with the same controller (``source`` 1 = human for counterfactual,
  human-first episodes, 0 = policy otherwise). A counterfactual recording is used
  once. A policy-first recording is used once while the start has an unused one;
  after that the start's policy-first recordings are reused: the protocol quota
  redraws the start of a fresh episode that failed
  (:mod:`mulligan.sim.collect.quota`), and a replay can fail where the recording
  succeeded.
* Policy segments (``source`` 0): the live policy drives for as many steps as
  the recording's policy segment had, then the operator takes over (``h``).
* Human segments (``source`` 1): the recorded actions are replayed open loop, all
  of them recorded; afterwards control goes back to the policy (``h``) if the
  recording continues with the policy.
* End of the recording: a recorded failure is replayed as the same key (``0``
  recoverable, ``9`` terminal). A recorded success is not asserted: the collector
  must run with ``--auto-save-on-success`` (it refuses to start otherwise) and
  ends the episode when the env reports success; if the replay has not succeeded when the recorded steps run out, a
  last policy segment continues up to ``max_episode_steps`` and the episode is
  then saved as a recoverable failure (``0``).
* Between episodes it answers ``c`` when an unused counterfactual recording of
  the same start exists and ``c`` is offered, ``q`` after ``max_episodes``
  episodes, and ``n`` otherwise.

The replay reproduces the operator, not the episodes: the policy samples its own
actions (unseeded), so policy segments and the outcomes of
open-loop human segments differ from the recording. Human steps the collector did
not save (idle input) are absent from the recording and are not replayed.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mulligan.sim.collect.utils import EpisodeStart, HumanInput, Phase
from mulligan.utils.state_to_grid import (
    extract_nut_pose_from_env_state,
    extract_peg_pos_from_env_state,
)

POLICY_SOURCE = 0
HUMAN_SOURCE = 1


@dataclass(frozen=True)
class Segment:
    source: int
    actions: np.ndarray  # (T, 7)


@dataclass(frozen=True)
class RecordedEpisode:
    episode_index: int
    nut: tuple[float, float, float]
    peg: Optional[tuple[float, float]]
    segments: tuple[Segment, ...]
    success: bool
    done: bool

    @property
    def human_first(self) -> bool:
        return self.segments[0].source == HUMAN_SOURCE


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def load_recorded_episodes(data_dir: Path) -> list[RecordedEpisode]:
    """Read the frames of a LeRobot v3 DAgger dataset (``data/*/*.parquet``) into episodes.

    The collector pads every episode with one ``is_valid == 0`` frame (the final
    observation, no action taken); it is not replayed. The episode outcome
    (``success``, ``done``) is read from that last frame.
    """
    import pandas as pd

    files = sorted(glob.glob(str(Path(data_dir) / "data" / "*" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no data/*/*.parquet under {data_dir}")
    columns = [
        "episode_index",
        "frame_index",
        "source",
        "action",
        "observation.environment_state",
        "success",
        "done",
        "is_valid",
    ]
    frames = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    frames = frames.sort_values(["episode_index", "frame_index"], kind="stable")

    episodes = []
    for episode_index, all_rows in frames.groupby("episode_index", sort=True):
        valid = np.asarray([np.asarray(v).reshape(-1)[0] for v in all_rows["is_valid"]]) != 0
        rows = all_rows[valid]
        if rows.empty:
            raise ValueError(f"episode {episode_index}: no is_valid frames")
        env_state = np.asarray(rows["observation.environment_state"].iloc[0], dtype=np.float64)
        task = "Square_D1" if len(env_state) == 17 else "NutAssemblySquare"
        nut = extract_nut_pose_from_env_state(env_state, task=task)
        peg = extract_peg_pos_from_env_state(env_state)[:2] if task == "Square_D1" else None
        sources = np.asarray([np.asarray(v).reshape(-1)[0] for v in rows["source"]]).astype(int)
        actions = np.stack([np.asarray(a, dtype=np.float64) for a in rows["action"]])
        boundaries = np.flatnonzero(np.diff(sources)) + 1
        segments = tuple(
            Segment(source=int(src[0]), actions=act)
            for src, act in zip(np.split(sources, boundaries), np.split(actions, boundaries))
        )
        unknown = {s.source for s in segments} - {POLICY_SOURCE, HUMAN_SOURCE}
        if unknown:
            raise ValueError(f"episode {episode_index}: unknown source labels {sorted(unknown)}")
        episodes.append(
            RecordedEpisode(
                episode_index=int(episode_index),
                nut=nut,
                peg=peg,
                segments=segments,
                success=bool(np.asarray(all_rows["success"].iloc[-1]).reshape(-1)[0]),
                done=bool(np.asarray(all_rows["done"].iloc[-1]).reshape(-1)[0]),
            )
        )
    return episodes


class ReplayOperator:
    """Operator that replays recorded episodes; see the module docstring."""

    # Checked by the collector: a recorded success ends only through auto-save.
    requires_auto_save_on_success = True

    def __init__(
        self,
        episodes: list[RecordedEpisode],
        *,
        max_episodes: Optional[int] = None,
        position_tolerance: float = 5e-3,
        yaw_tolerance: float = 0.05,
        max_episode_steps: int = 400,
    ):
        if not episodes:
            raise ValueError("no recorded episodes to replay")
        self.episodes = episodes
        self.unused = {ep.episode_index for ep in episodes}
        self.max_episodes = max_episodes
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.max_episode_steps = max_episode_steps
        self.episodes_started = 0
        self.current: Optional[RecordedEpisode] = None
        self.current_start: Optional[EpisodeStart] = None
        self._segment_idx = 0
        self._step_in_segment = 0
        self._episode_steps = 0

    # ------------------------------------------------------------------ matching
    def _distance(self, episode: RecordedEpisode, start_state: tuple) -> Optional[float]:
        """Largest position error to ``start_state``, or None if outside the tolerances."""
        if episode.peg is None:
            nut, peg = start_state, None
        else:
            nut, peg = start_state
        errors = [abs(episode.nut[0] - nut[0]), abs(episode.nut[1] - nut[1])]
        if peg is not None:
            errors += [abs(episode.peg[0] - peg[0]), abs(episode.peg[1] - peg[1])]
        yaw_error = abs(_wrap_angle(episode.nut[2] - nut[2]))
        if max(errors) > self.position_tolerance or yaw_error > self.yaw_tolerance:
            return None
        return max(errors)

    def _find(
        self, start_state: tuple, human_first: bool, *, unused_only: bool = True
    ) -> Optional[RecordedEpisode]:
        best, best_distance = None, np.inf
        for episode in self.episodes:
            if episode.human_first != human_first:
                continue
            if unused_only and episode.episode_index not in self.unused:
                continue
            distance = self._distance(episode, start_state)
            if distance is not None and distance < best_distance:
                best, best_distance = episode, distance
        return best

    # ------------------------------------------------------------------ Operator
    def begin_episode(self, start: EpisodeStart) -> None:
        if start.start_state is None:
            raise ValueError(
                "the replay operator needs placed start states (run the collector with --sampler)"
            )
        episode = self._find(start.start_state, start.human_first)
        reused = False
        if episode is None and not start.human_first:
            # A start the quota redrew after a failed fresh episode.
            episode = self._find(start.start_state, False, unused_only=False)
            reused = episode is not None
        if episode is None:
            kind = "unused human-first" if start.human_first else "policy-first"
            raise LookupError(
                f"no {kind} recorded episode starts at {start.start_state} "
                f"(tolerance {self.position_tolerance} m, {self.yaw_tolerance} rad)"
            )
        self.unused.discard(episode.episode_index)
        self.current, self.current_start = episode, start
        self._segment_idx = 0
        self._step_in_segment = 0
        self._episode_steps = 0
        self.episodes_started += 1
        layout = " ".join(
            ("H" if seg.source else "P") + str(len(seg.actions)) for seg in episode.segments
        )
        print(
            f"[replay] episode {start.episode_index}: recorded episode {episode.episode_index} "
            f"({layout}){' (reused)' if reused else ''}"
        )

    def _segment(self) -> Optional[Segment]:
        segments = self.current.segments
        return segments[self._segment_idx] if self._segment_idx < len(segments) else None

    def _end_key(self) -> str:
        if self.current.success:
            return "0"  # the replay did not reproduce the recorded success
        return "9" if self.current.done else "0"

    def _advance(self, phase: Phase) -> Optional[str]:
        """Close the current segment; return the key that ends it."""
        self._segment_idx += 1
        self._step_in_segment = 0
        if self._segment() is not None:
            return "h"
        if self.current.success:
            # The recording succeeded but the replay has not (auto-save would have ended
            # the episode): the policy continues until success or the step budget.
            return "h" if phase == "human" else self._policy_overtime()
        return self._end_key()

    def _policy_overtime(self) -> Optional[str]:
        if self._episode_steps >= self.max_episode_steps:
            return self._end_key()
        self._episode_steps += 1
        return None

    def poll_key(self, phase: Phase) -> Optional[str]:
        if self.current is None:
            raise RuntimeError("poll_key called before begin_episode")
        expected = HUMAN_SOURCE if phase == "human" else POLICY_SOURCE
        segment = self._segment()
        if segment is None:
            # Past the recording: only a policy phase after a recorded success gets here.
            if phase != "policy":
                raise RuntimeError("human phase requested after the recording ended")
            return self._policy_overtime()
        if segment.source != expected:
            raise RuntimeError(
                f"recorded episode {self.current.episode_index}: collector is in the {phase} "
                f"phase but segment {self._segment_idx} is source {segment.source}"
            )
        if self._step_in_segment >= len(segment.actions):
            return self._advance(phase)
        if phase == "policy":
            self._step_in_segment += 1
            self._episode_steps += 1
        return None

    def begin_human_segment(self, gripper_action: float) -> None:
        segment = self._segment()
        if segment is None or segment.source != HUMAN_SOURCE:
            raise RuntimeError("takeover without a recorded human segment")

    def human_input(self) -> HumanInput:
        segment = self._segment()
        action = np.array(segment.actions[self._step_in_segment], dtype=np.float64)
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        action[6] = 1.0 if action[6] >= 0.0 else -1.0
        self._step_in_segment += 1
        self._episode_steps += 1
        return HumanInput(action=action, arm_active=True, strong=True)

    def choose(self, options: str) -> str:
        if self.max_episodes is not None and self.episodes_started >= self.max_episodes:
            return "q"
        if "c" in options and self.current_start is not None:
            if self._find(self.current_start.start_state, human_first=True) is not None:
                return "c"
        if "n" in options:
            return "n"
        return "q"

    def close(self) -> None:
        print(
            f"[replay] replayed {self.episodes_started} episodes, {len(self.unused)} recorded episodes unused"
        )


def make_replay_operator(
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    root: Optional[str] = None,
    max_episodes: Optional[int] = None,
    position_tolerance: float = 5e-3,
    yaw_tolerance: float = 0.05,
    max_episode_steps: int = 400,
) -> ReplayOperator:
    """Factory for ``--operator``: replay a local dataset (``root``) or an HF dataset at a pinned revision.

    Only the frame parquet files are downloaded (no videos).
    """
    if (repo_id is None) == (root is None):
        raise ValueError("pass exactly one of repo_id (with revision) or root")
    if repo_id is not None:
        if not revision:
            raise ValueError(f"replaying {repo_id} needs a pinned revision")
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["data/*/*.parquet", "meta/info.json"],
        )
    return ReplayOperator(
        load_recorded_episodes(Path(root)),
        max_episodes=max_episodes,
        position_tolerance=position_tolerance,
        yaw_tolerance=yaw_tolerance,
        max_episode_steps=max_episode_steps,
    )
