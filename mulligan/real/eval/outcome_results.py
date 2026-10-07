"""Canonical outcome reconciliation for real eval ``results.json`` payloads.

Outcome editing rewrites the LeRobot frame labels (``success``/``reward``/
``done``/``is_valid``) and records the operator decisions in
``.outcome_edit_progress.json``. Any consumer that reads root ``results.json``
must apply the same edit record and validate against frame data; otherwise stale
eval-time sidecars can silently disagree with the edited dataset.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import EntryNotFoundError

OUTCOME_ORDER = ("success", "timeout", "failure")
OUTCOME_SET = frozenset(OUTCOME_ORDER)
DEFAULT_OUTCOME_OVERRIDES = ".outcome_edit_progress.json"

# Keys of results.json that grow across a resumed eval. The ``arena_*`` keys only occur
# in results files written by the arena collection tooling (read, never written here).
_RESUMABLE_RESULTS_DYNAMIC_KEYS = frozenset(
    {
        "timestamp",
        "summary",
        "rollouts",
        "arena_submitted_round_indices",
        "arena_submitted_rollouts",
        "phase_stops",
    }
)


@dataclass(frozen=True)
class FrameOutcome:
    outcome: str
    expected_num_steps: int


def valid_prefix_length(is_valid_values: Any, *, episode_index: int | None = None) -> int:
    """Number of leading ``is_valid==1`` frames for an outcome-edited episode.

    Real LeRobot episodes store a valid prefix followed by an invalid suffix.
    At minimum the terminal padding frame is ``is_valid=0``; after the outcome
    editor's soft truncation, every post-outcome frame (retract / reset /
    operator handling junk) is ``is_valid=0`` too (see
    ``mulligan.tools.outcome_review.apply_outcome_edits``). All accounting over
    collected DAgger/teleop/rollout data must cut off at this boundary — only
    video rendering keeps the invalid suffix. Any other pattern (a valid frame
    after an invalid one) means corrupted flags and raises loudly rather than
    silently miscounting.

    Returns the length of the leading valid run, which is the effective episode
    length (frames ``0 .. outcome_frame`` inclusive when soft-truncated).
    """
    values = [int(v) for v in is_valid_values]
    if not values:
        raise RuntimeError(f"episode {episode_index}: empty is_valid sequence")
    seen_invalid = False
    prefix = 0
    for value in values:
        if value not in (0, 1):
            raise RuntimeError(f"episode {episode_index}: is_valid must be 0/1, got {value}")
        if value == 1:
            if seen_invalid:
                raise RuntimeError(
                    f"episode {episode_index}: is_valid is not a valid prefix followed by padding"
                )
            prefix += 1
        else:
            seen_invalid = True
    if prefix == 0:
        raise RuntimeError(f"episode {episode_index}: no valid frames")
    return prefix


def first_done_inclusive_length(done_values: Any, *, episode_index: int | None = None) -> int:
    """Length through the first ``done==1`` row, or the full input if none exists.

    Real outcome-edited episodes may retain a repeated terminal tail.  A value
    trajectory represents the task only through the first terminal row, inclusive.
    ``done`` returning to zero inside the supplied valid prefix is corruption and
    fails loudly.
    """
    values = [int(v) for v in done_values]
    for value in values:
        if value not in (0, 1):
            raise RuntimeError(f"episode {episode_index}: done must be 0/1, got {value}")
    terminal = [i for i, value in enumerate(values) if value == 1]
    if not terminal:
        return len(values)
    first = terminal[0]
    if any(value == 0 for value in values[first:]):
        raise RuntimeError(f"episode {episode_index}: done returns to zero after terminal frame")
    return first + 1


def task_effective_prefix_length(
    done_values: Any,
    is_valid_values: Any,
    *,
    episode_index: int | None = None,
) -> int:
    """Task-relevant prefix ending at first ``done`` or first invalid row.

    The first invalid row is excluded; the first terminal row is included.  The
    validity and terminal-tail contracts are both validated before returning.
    """
    done = [int(v) for v in done_values]
    valid = [int(v) for v in is_valid_values]
    if len(done) != len(valid):
        raise RuntimeError(
            f"episode {episode_index}: done/is_valid length mismatch ({len(done)} != {len(valid)})"
        )
    valid_length = valid_prefix_length(valid, episode_index=episode_index)
    return first_done_inclusive_length(done[:valid_length], episode_index=episode_index)


def load_hf_json(
    repo_id: str,
    path_in_repo: str,
    *,
    revision: str = "main",
    force_download: bool = True,
) -> dict[str, Any]:
    path = Path(
        hf_hub_download(
            repo_id,
            path_in_repo,
            repo_type="dataset",
            revision=revision,
            force_download=force_download,
        )
    )
    return json.loads(path.read_text())


def load_outcome_edit_record(
    repo_id: str,
    overrides_filename: str = DEFAULT_OUTCOME_OVERRIDES,
    *,
    revision: str = "main",
    required: bool = True,
    force_download: bool = True,
) -> dict[str, Any] | None:
    try:
        return load_hf_json(
            repo_id, overrides_filename, revision=revision, force_download=force_download
        )
    except EntryNotFoundError:
        if required:
            raise
        return None


def apply_outcome_edit_record(
    payload: dict[str, Any],
    record: dict[str, Any],
    *,
    overrides_filename: str = DEFAULT_OUTCOME_OVERRIDES,
) -> dict[str, Any]:
    """Apply an outcome-editor record to ``payload`` in place.

    Returns the reconciliation summary written into
    ``payload["_outcome_edit_reconciliation"]``. The function is idempotent:
    applying the same record to an already-reconciled payload leaves outcomes and
    counts unchanged.
    """
    existing_reconciliation = payload.get("_outcome_edit_reconciliation")
    changed = {int(k): v for k, v in record["changed_episodes"].items()}
    by_ep = _rollouts_by_episode(payload)
    unknown = sorted(set(changed) - set(by_ep))
    if unknown:
        raise RuntimeError(
            f"{overrides_filename} references episodes not in results.json: {unknown}"
        )

    class_changes = 0
    step_patches = 0
    flips: list[dict[str, Any]] = []
    for ep, entry in sorted(changed.items()):
        new = str(entry["new_outcome"])
        if new not in OUTCOME_SET:
            raise RuntimeError(f"episode {ep}: unknown edited outcome {new!r}")
        rollout = by_ep[ep]
        old = str(rollout["outcome"])
        if old != new:
            class_changes += 1
            if (old == "success") != (new == "success"):
                flips.append({"episode_index": ep, "old": old, "new": new})
            rollout["outcome"] = new
            if "success" in rollout:
                rollout["success"] = new == "success"

        frame = entry.get("outcome_frame")
        if frame is not None and (new in ("success", "failure") or entry.get("soft_truncate")):
            steps = int(frame) + 1
            if steps < 1:
                raise RuntimeError(f"episode {ep}: invalid outcome_frame {frame}")
            if steps != int(rollout["num_steps"]):
                rollout["num_steps"] = steps
                step_patches += 1

    recompute_summary_from_rollouts(payload)
    reconciliation = {
        "overrides_file": overrides_filename,
        "episodes_reviewed": len(changed),
        "outcome_class_changes": class_changes,
        "num_steps_patches": step_patches,
        "success_flips": flips,
    }
    if existing_reconciliation is not None and class_changes == 0 and step_patches == 0:
        payload["_outcome_edit_reconciliation"] = existing_reconciliation
        return existing_reconciliation
    payload["_outcome_edit_reconciliation"] = reconciliation
    return reconciliation


def recompute_summary_from_rollouts(payload: dict[str, Any]) -> None:
    _validate_summary_policy_ids(payload)
    for row in payload["summary"]:
        policy_id = int(row["policy_id"])
        sub = [r for r in payload["rollouts"] if int(r["policy_id"]) == policy_id]
        successes = sum(str(r["outcome"]) == "success" for r in sub)
        if "num_rounds" in row:
            row["num_rounds"] = len(sub)
        row["successes"] = successes
        row["failures"] = len(sub) - successes
        row["success_rate"] = successes / len(sub) if sub else 0.0


def _unique_ids(rows: list[dict[str, Any]], *, key: str, what: str) -> set[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for row in rows:
        value = int(row[key])
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise RuntimeError(f"{what} has duplicate {key} values: {sorted(duplicates)}")
    return seen


def _validate_summary_policy_ids(payload: dict[str, Any]) -> None:
    summary_policy_ids = _unique_ids(payload["summary"], key="policy_id", what="results summary")
    rollout_policy_ids = {int(row["policy_id"]) for row in payload["rollouts"]}
    missing_summary = sorted(rollout_policy_ids - summary_policy_ids)
    missing_rollouts = sorted(summary_policy_ids - rollout_policy_ids)
    errors = []
    if missing_summary:
        errors.append(f"missing summary rows for rollout policy_id values {missing_summary}")
    if missing_rollouts:
        errors.append(f"summary policy_id values with no rollouts {missing_rollouts}")
    if errors:
        raise RuntimeError(
            "results.json summary policy ids disagree with rollouts: " + "; ".join(errors)
        )


def _rollouts_by_episode(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["episode_index"]): row
        for row in _deduplicate_rows(
            payload["rollouts"], key="episode_index", what="results rollouts"
        )
    }


def _deduplicate_rows(
    rows: list[dict[str, Any]],
    *,
    key: str,
    what: str,
) -> list[dict[str, Any]]:
    _unique_ids(rows, key=key, what=what)
    return rows


def validate_summary_matches_rollouts(payload: dict[str, Any]) -> None:
    errors: list[str] = []
    _validate_summary_policy_ids(payload)
    for row in payload["summary"]:
        policy_id = int(row["policy_id"])
        sub = [r for r in payload["rollouts"] if int(r["policy_id"]) == policy_id]
        successes = sum(str(r["outcome"]) == "success" for r in sub)
        if "num_rounds" in row and int(row["num_rounds"]) != len(sub):
            errors.append(
                f"policy_id {policy_id}: summary num_rounds={row['num_rounds']} "
                f"but rollouts={len(sub)}"
            )
        if int(row["successes"]) != successes:
            errors.append(
                f"policy_id {policy_id}: summary successes={row['successes']} "
                f"but rollouts={successes}"
            )
        expected_failures = len(sub) - successes
        if int(row["failures"]) != expected_failures:
            errors.append(
                f"policy_id {policy_id}: summary failures={row['failures']} "
                f"but rollouts={expected_failures}"
            )
    if errors:
        raise RuntimeError(
            "results.json summary disagrees with rollouts:\n" + "\n".join(errors[:20])
        )


def canonicalize_results_payload(
    payload: dict[str, Any],
    *,
    outcome_edit_record: dict[str, Any] | None = None,
    overrides_filename: str = DEFAULT_OUTCOME_OVERRIDES,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    out = copy.deepcopy(payload)
    reconciliation = None
    if outcome_edit_record is not None:
        reconciliation = apply_outcome_edit_record(
            out,
            outcome_edit_record,
            overrides_filename=overrides_filename,
        )
    else:
        recompute_summary_from_rollouts(out)
    validate_summary_matches_rollouts(out)
    return out, reconciliation


def _meta_episode_files_from_hf(
    repo_id: str, *, revision: str, force_download: bool = True
) -> list[Path]:
    files = sorted(
        f
        for f in list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if f.startswith("meta/episodes/") and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"{repo_id}: no meta/episodes parquet files found")
    return [
        Path(
            hf_hub_download(
                repo_id,
                f,
                repo_type="dataset",
                revision=revision,
                force_download=force_download,
            )
        )
        for f in files
    ]


def _meta_episode_files_from_root(dataset_root: Path) -> list[Path]:
    files = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"{dataset_root}: no meta/episodes parquet files found")
    return files


def _read_episode_meta_from_files(files: list[Path]) -> pd.DataFrame:
    episode_meta = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    duplicate_mask = episode_meta["episode_index"].duplicated()
    if duplicate_mask.any():
        duplicates = sorted(
            episode_meta.loc[duplicate_mask, "episode_index"].astype(int).unique().tolist()
        )
        raise RuntimeError(f"meta/episodes has duplicate episode_index values: {duplicates}")
    return episode_meta


def _data_file_paths_from_meta_hf(
    repo_id: str,
    episode_meta: pd.DataFrame,
    *,
    revision: str,
    force_download: bool = True,
) -> dict[tuple[int, int], Path]:
    keys = sorted(
        {
            (int(r["data/chunk_index"]), int(r["data/file_index"]))
            for r in episode_meta[["data/chunk_index", "data/file_index"]].to_dict("records")
        }
    )
    return {
        key: Path(
            hf_hub_download(
                repo_id,
                f"data/chunk-{key[0]:03d}/file-{key[1]:03d}.parquet",
                repo_type="dataset",
                revision=revision,
                force_download=force_download,
            )
        )
        for key in keys
    }


def _data_file_paths_from_meta_root(
    dataset_root: Path, episode_meta: pd.DataFrame
) -> dict[tuple[int, int], Path]:
    keys = sorted(
        {
            (int(r["data/chunk_index"]), int(r["data/file_index"]))
            for r in episode_meta[["data/chunk_index", "data/file_index"]].to_dict("records")
        }
    )
    paths = {
        key: dataset_root / f"data/chunk-{key[0]:03d}/file-{key[1]:03d}.parquet" for key in keys
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "episode metadata references missing data files:\n" + "\n".join(map(str, missing[:20]))
        )
    return paths


def _read_data_file(path: Path) -> pd.DataFrame:
    columns = set(pq.read_schema(path).names)
    required = {"episode_index", "frame_index", "reward", "done"}
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(f"{path}: missing outcome columns {missing}")
    optional = [c for c in ("success", "is_valid") if c in columns]
    return pd.read_parquet(path, columns=[*sorted(required), *optional])


def _detect_frame_outcome(
    ep: pd.DataFrame,
    *,
    subtask_frames: tuple[int, ...] = (),
) -> FrameOutcome:
    """Classify one episode's frame trace into a ``FrameOutcome``.

    ``subtask_frames`` lists the frame indices at which the outcome editor
    deliberately placed a mid-episode reward spike (a recorded sub-goal reward
    with ``done=0``; see ``mulligan.tools.outcome_review``). Nonzero pre-terminal
    reward is tolerated ONLY at these explicitly-recorded frames, and only with
    the exact spike value the editor writes (``reward == 1.0``); any other
    mid-episode nonzero reward — or a corrupt value like 0.5/5.0 at a labeled
    frame — still raises loudly. ``done`` must stay ``0`` before the terminal
    frame regardless (a spike never sets ``done``), so the valid-prefix /
    ``expected_num_steps`` semantics are unchanged.
    """
    ep = ep.sort_values("frame_index")
    duplicate_frames = ep["frame_index"].duplicated()
    if duplicate_frames.any():
        duplicates = sorted(ep.loc[duplicate_frames, "frame_index"].astype(int).unique().tolist())
        episode_index = int(ep["episode_index"].iloc[0])
        raise RuntimeError(f"episode {episode_index}: duplicate frame_index values {duplicates}")
    subtask_set = {int(f) for f in subtask_frames}
    if "is_valid" in ep.columns:
        episode_index = int(ep["episode_index"].iloc[0])
        prefix = valid_prefix_length(ep["is_valid"].tolist(), episode_index=episode_index)
        valid = ep.iloc[:prefix]
    else:
        valid = ep
    if valid.empty:
        raise RuntimeError("episode has no valid frames")
    reward = valid["reward"].astype(float)
    done = valid["done"].astype(int)
    labeled = valid["frame_index"].astype(int).isin(subtask_set)
    terminal = done.eq(1)

    def _check_spike_values(frames: pd.DataFrame, frames_labeled: pd.Series) -> None:
        # The editor writes exactly reward=1.0 at recorded subtask frames; any
        # other value at a labeled frame is corruption, not a tolerated spike.
        bad_spike = frames["reward"].astype(float).ne(1.0) & frames_labeled
        if bad_spike.any():
            bad_frames = sorted(frames.loc[bad_spike, "frame_index"].astype(int).tolist())
            raise RuntimeError(
                f"recorded subtask frame(s) {bad_frames} have reward != 1.0; "
                "spike values must be exactly 1.0"
            )

    if not terminal.any():
        unlabeled_reward = reward.ne(0.0) & ~labeled
        if unlabeled_reward.any():
            raise RuntimeError("timeout episode has nonzero reward before terminal")
        _check_spike_values(valid, labeled)
        outcome = "timeout"
        expected_num_steps = len(valid)
    else:
        terminal_pos = int(terminal.to_numpy().argmax())
        pre = valid.iloc[:terminal_pos]
        tail = valid.iloc[terminal_pos:]
        if not pre.empty:
            if not pre["done"].astype(int).eq(0).all():
                raise RuntimeError("terminal episode has done=1 before the outcome frame")
            pre_labeled = pre["frame_index"].astype(int).isin(subtask_set)
            unlabeled_reward = pre["reward"].astype(float).ne(0.0) & ~pre_labeled
            if unlabeled_reward.any():
                raise RuntimeError("terminal episode has nonzero reward before outcome frame")
            _check_spike_values(pre, pre_labeled)
        tail_reward = tail["reward"].astype(float)
        tail_done = tail["done"].astype(int)
        if not tail_done.eq(1).all():
            raise RuntimeError("terminal episode has done=0 after the outcome frame")
        first_terminal_reward = float(tail_reward.iloc[0])
        if first_terminal_reward not in (0.0, 1.0):
            raise RuntimeError(f"unexpected terminal reward={first_terminal_reward}")
        if not tail_reward.eq(first_terminal_reward).all():
            raise RuntimeError("terminal episode changes reward after the outcome frame")
        outcome = "success" if first_terminal_reward == 1.0 else "failure"
        expected_num_steps = int(tail["frame_index"].min()) + 1

    if "success" in ep.columns:
        expected_success = 1 if outcome == "success" else 0
        bad = ep["success"].astype(int).ne(expected_success)
        if bad.any():
            raise RuntimeError(
                f"success column is not episode-constant {expected_success}; "
                f"saw {sorted(ep['success'].dropna().unique().tolist())}"
            )

    return FrameOutcome(outcome=outcome, expected_num_steps=expected_num_steps)


def subtask_mark_count_error(new_outcome: str, n_marks: int, subtask_marks: int) -> str | None:
    """Legality of ``n_marks`` mid-episode sub-goal marks for a ``new_outcome`` episode.

    Returns an error string if illegal, else ``None``. Single source of truth shared by
    the outcome editor's interactive confirm gate / ``apply_outcome_edits`` and the
    graded-score eval ingest (``mulligan.real.lifecycle.heldout_eval``) so they cannot diverge.

    A ``success`` reached EVERY sub-goal — the final one IS the terminal success — so it
    must carry exactly ``subtask_marks`` mid-episode marks. A ``failure``/``timeout``
    ended after reaching some PREFIX of the sub-goals (possibly none), so ``0..subtask_marks``
    marks are all legal (e.g. routing: 0 = never seated a clip, 1 = seated the first clip
    then fumbled/failed). ``subtask_marks == 0`` disables the machinery: any marks are
    rejected by the caller, none here.
    """
    if subtask_marks <= 0:
        return None
    if new_outcome == "success":
        if n_marks != subtask_marks:
            return (
                f"a SUCCESS episode must carry exactly {subtask_marks} subtask mark(s) "
                f"(the mid-episode sub-goal(s); the final sub-goal is the terminal success), "
                f"got {n_marks}"
            )
    elif n_marks > subtask_marks:
        return (
            f"a {new_outcome.upper()} episode may carry at most {subtask_marks} subtask "
            f"mark(s), got {n_marks}"
        )
    return None


def subtask_reviewed_mark_counts(record: dict[str, Any] | None) -> dict[int, int]:
    """Per-episode subtask-mark COUNTS for episodes REVIEWED in a subtask session.

    Key presence of ``subtask_frames`` in a changed-episode record — even as an
    empty list — is the reviewed signal (see ``mulligan.tools.outcome_review``
    ``episode_fully_processed``); a legitimate 0-mark failure review therefore
    appears here with count 0, while pre-subtask / unreviewed records are
    absent. Graded-score consumers must treat a missing episode as
    NOT-REVIEWED and fail loudly rather than silently scoring it 0.
    """
    if record is None:
        return {}
    out: dict[int, int] = {}
    for ep_str, entry in record.get("changed_episodes", {}).items():
        if "subtask_frames" in entry:
            # sorted(set(...)): hand-edited duplicates must not double-count.
            out[int(ep_str)] = len({int(f) for f in entry["subtask_frames"]})
    return out


def subtask_frames_from_record(record: dict[str, Any] | None) -> dict[int, list[int]]:
    """Extract per-episode recorded subtask reward frames from an edit record.

    The outcome-editor progress record stores an optional ``subtask_frames`` list
    per changed episode; an absent key means no spikes. Threading this into the
    frame-outcome loaders lets ``_detect_frame_outcome`` distinguish a RECORDED
    mid-episode reward spike from an unlabeled anomaly.
    """
    if record is None:
        return {}
    out: dict[int, list[int]] = {}
    for ep_str, entry in record.get("changed_episodes", {}).items():
        frames = entry.get("subtask_frames", [])
        if frames:
            # sorted(set(...)): hand-edited duplicates must not double-count.
            out[int(ep_str)] = sorted({int(f) for f in frames})
    return out


def live_subtask_frames_from_results(payload: dict[str, Any] | None) -> dict[int, list[int]]:
    """Per-episode LIVE (eval-time) subtask marks from a ``results.json`` payload.

    ``rollout_episode`` records the operator's live 'g' / numpad-'3' press as
    ``subtask_frames`` on the rollout row and ``finalize_episode_data`` writes the
    same reward=1.0 / done=0 spike the outcome editor writes, so an episode nobody
    has reviewed yet legitimately carries a mid-episode spike. Older results files
    have no key (no live marks).
    """
    if payload is None:
        return {}
    out: dict[int, list[int]] = {}
    for ep_idx, rollout in _rollouts_by_episode(payload).items():
        frames = rollout.get("subtask_frames") or []
        if frames:
            out[ep_idx] = sorted({int(f) for f in frames})
    return out


def subtask_frames_for_validation(
    record: dict[str, Any] | None,
    results_payload: dict[str, Any] | None,
) -> dict[int, list[int]]:
    """Subtask frames the frame validator must tolerate, per episode.

    Live eval-time marks (``results.json``) overridden per episode by the review
    record: an episode present in ``changed_episodes`` is authoritative even when
    its record carries no marks, because apply zeroed its pre-outcome reward and
    wrote exactly the recorded spikes. An UNREVIEWED episode keeps the live spike
    ``finalize_episode_data`` wrote. Without the live fallback a partial review of
    an in-progress eval (20 of 50 rounds reviewed, the rest still carrying live
    spikes) could never apply.
    """
    out = live_subtask_frames_from_results(results_payload)
    if record is not None:
        for ep_str, entry in record.get("changed_episodes", {}).items():
            frames = sorted({int(f) for f in entry.get("subtask_frames", [])})
            if frames:
                out[int(ep_str)] = frames
            else:
                out.pop(int(ep_str), None)
    return out


def load_frame_outcomes_from_hf(
    repo_id: str,
    *,
    revision: str = "main",
    subtask_frames_by_episode: dict[int, list[int]] | None = None,
    force_download: bool = True,
    episodes: Collection[int] | None = None,
) -> dict[int, FrameOutcome]:
    """Frame-label outcome of every episode of a Hub dataset, or of ``episodes`` only."""
    episode_meta = _read_episode_meta_from_files(
        _meta_episode_files_from_hf(repo_id, revision=revision, force_download=force_download)
    )
    if episodes is not None:
        episode_meta = episode_meta[episode_meta["episode_index"].isin(list(episodes))]
    paths = _data_file_paths_from_meta_hf(
        repo_id,
        episode_meta,
        revision=revision,
        force_download=force_download,
    )
    return _load_frame_outcomes_from_data_files(
        paths, subtask_frames_by_episode=subtask_frames_by_episode, episodes=episodes
    )


def load_frame_outcomes_from_root(
    dataset_root: Path,
    *,
    subtask_frames_by_episode: dict[int, list[int]] | None = None,
) -> dict[int, FrameOutcome]:
    episode_meta = _read_episode_meta_from_files(_meta_episode_files_from_root(dataset_root))
    paths = _data_file_paths_from_meta_root(dataset_root, episode_meta)
    return _load_frame_outcomes_from_data_files(
        paths, subtask_frames_by_episode=subtask_frames_by_episode
    )


def _load_frame_outcomes_from_data_files(
    paths: dict[tuple[int, int], Path],
    *,
    subtask_frames_by_episode: dict[int, list[int]] | None = None,
    episodes: Collection[int] | None = None,
) -> dict[int, FrameOutcome]:
    subtask_by_ep = subtask_frames_by_episode or {}
    frames = pd.concat((_read_data_file(path) for path in paths.values()), ignore_index=True)
    if episodes is not None:
        frames = frames[frames["episode_index"].isin(list(episodes))]
    outcomes: dict[int, FrameOutcome] = {}
    for episode_index, group in frames.groupby("episode_index"):
        ep_idx = int(episode_index)
        outcomes[ep_idx] = _detect_frame_outcome(
            group, subtask_frames=tuple(subtask_by_ep.get(ep_idx, ()))
        )
    return outcomes


def validate_results_against_frame_outcomes(
    payload: dict[str, Any],
    frame_outcomes: dict[int, FrameOutcome],
    *,
    check_num_steps: bool = True,
) -> None:
    errors: list[str] = []
    result_by_episode = _rollouts_by_episode(payload)
    result_episodes = set(result_by_episode)
    frame_episodes = set(frame_outcomes)
    missing_frames = sorted(result_episodes - frame_episodes)
    missing_results = sorted(frame_episodes - result_episodes)
    if missing_frames:
        errors.append(
            f"episodes present in results.json but missing from frame data: {missing_frames[:30]}"
        )
    if missing_results:
        errors.append(
            f"episodes present in frame data but missing from results.json: {missing_results[:30]}"
        )
    for ep, rollout in sorted(result_by_episode.items()):
        ep = int(rollout["episode_index"])
        if ep not in frame_outcomes:
            continue
        frame = frame_outcomes[ep]
        outcome = str(rollout["outcome"])
        if outcome != frame.outcome:
            errors.append(
                f"episode {ep}: results outcome={outcome!r} "
                f"but frame/meta outcome={frame.outcome!r}"
            )
        if check_num_steps and int(rollout["num_steps"]) != int(frame.expected_num_steps):
            errors.append(
                f"episode {ep}: results num_steps={rollout['num_steps']} "
                f"but frame/meta expected_num_steps={frame.expected_num_steps}"
            )
    if errors:
        preview = "\n".join(errors[:30])
        extra = "" if len(errors) <= 30 else f"\n... {len(errors) - 30} more"
        raise RuntimeError(f"results.json disagrees with frame/meta outcomes:\n{preview}{extra}")


def canonicalize_results_file(
    dataset_root: Path,
    *,
    results_filename: str = "results.json",
    overrides_filename: str = DEFAULT_OUTCOME_OVERRIDES,
    backup_filename: str = "results_eval_time.json",
    validate_with_frames: bool = True,
) -> tuple[Path | None, Path | None, dict[str, Any] | None]:
    """Canonicalize a local dataset-root ``results.json`` if present.

    Returns ``(results_path, backup_path, reconciliation)``. ``backup_path`` is
    non-``None`` only when a results file was replaced and the eval-time backup
    was created or preserved.
    """
    results_path = dataset_root / results_filename
    if not results_path.exists():
        return None, None, None
    overrides_path = dataset_root / overrides_filename
    record = json.loads(overrides_path.read_text()) if overrides_path.exists() else None
    payload = json.loads(results_path.read_text())
    canonical, reconciliation = canonicalize_results_payload(
        payload,
        outcome_edit_record=record,
        overrides_filename=overrides_filename,
    )
    if validate_with_frames:
        validate_results_against_frame_outcomes(
            canonical,
            load_frame_outcomes_from_root(
                dataset_root,
                subtask_frames_by_episode=subtask_frames_for_validation(record, canonical),
            ),
        )
    original_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    canonical_text = json.dumps(canonical, indent=2, sort_keys=True) + "\n"
    backup_path = None
    already_reconciled = "_outcome_edit_reconciliation" in payload
    existing_backup_path = dataset_root / backup_filename
    if already_reconciled and not existing_backup_path.exists():
        raise RuntimeError(
            f"{results_path} is already outcome-reconciled but {existing_backup_path} is missing; "
            "refusing to invent eval-time provenance"
        )
    if canonical_text != original_text:
        backup_path = existing_backup_path
        if existing_backup_path.exists():
            backup_text = (
                json.dumps(json.loads(existing_backup_path.read_text()), indent=2, sort_keys=True)
                + "\n"
            )
            if not already_reconciled and backup_text != original_text:
                backup_payload = json.loads(backup_text)
                if _is_resumed_eval_prefix(backup_payload, payload):
                    existing_backup_path.write_text(original_text)
                else:
                    raise RuntimeError(
                        f"{existing_backup_path} already exists and differs from the current raw "
                        "results payload; refusing to overwrite results.json with stale provenance"
                    )
        else:
            existing_backup_path.write_text(original_text)
        results_path.write_text(canonical_text)
    return results_path, backup_path, reconciliation


def _is_resumed_eval_prefix(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Whether ``previous`` is an exact earlier checkpoint of ``current``.

    Held-out evaluation can be outcome-edited before collection finishes, then
    resume and append more rollouts. In that case the old eval-time backup must
    advance to the completed raw payload before reconciliation. Require a stable
    run identity and exact rollout prefix; all non-progress fields must also be
    unchanged. This keeps unrelated or modified provenance loud failures.
    """
    identity_keys = ("dataset_name", "arena_session_id")
    if not any(key in previous and key in current for key in identity_keys):
        return False
    if any(previous.get(key) != current.get(key) for key in identity_keys):
        return False

    previous_rollouts = previous.get("rollouts")
    current_rollouts = current.get("rollouts")
    if not isinstance(previous_rollouts, list) or not isinstance(current_rollouts, list):
        return False
    if len(previous_rollouts) >= len(current_rollouts):
        return False
    if current_rollouts[: len(previous_rollouts)] != previous_rollouts:
        return False

    for key in ("arena_submitted_round_indices", "arena_submitted_rollouts", "phase_stops"):
        previous_submitted = previous.get(key)
        current_submitted = current.get(key)
        if previous_submitted is None and current_submitted is None:
            continue
        if not isinstance(previous_submitted, list) or not isinstance(current_submitted, list):
            return False
        if current_submitted[: len(previous_submitted)] != previous_submitted:
            return False

    stable_keys = (set(previous) | set(current)) - _RESUMABLE_RESULTS_DYNAMIC_KEYS - {"args"}
    if not all(previous.get(key) == current.get(key) for key in stable_keys):
        return False
    return _resumed_args_compatible(previous.get("args"), current.get("args"))


def _resumed_args_compatible(previous: Any, current: Any) -> bool:
    """``args`` may gain/lose CLI keys across a resume (the eval CLI evolves
    between sessions, e.g. a build that adds ``arena_session_status``); every key present in BOTH
    must match.
    Non-dict args fall back to strict equality."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return previous == current
    return all(previous[key] == current[key] for key in set(previous) & set(current))
