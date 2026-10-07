"""Immutable-snapshot helpers for held-out robot-eval analysis.

The standard held-out ingest reads a released dataset at its pin and any other repo at
mutable Hub ``main``.  This module is for retrospective analyses that must bind every
table and figure to one commit SHA and to the sha256 of each file they read.
It composes the shared held-out-eval and episode-length primitives instead of
copying their statistics or plotting code.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from mulligan.real.lifecycle import heldout_eval
from mulligan.real.lifecycle.episode_lengths import (
    EpisodeLengthConfig,
    build_episode_table,
    plot as plot_episode_lengths,
    recompute_lengths,
    write_sidecar_csv,
)
from mulligan.real.eval.outcome_results import (
    apply_outcome_edit_record,
    load_frame_outcomes_from_hf,
    subtask_frames_for_validation,
    subtask_reviewed_mark_counts,
    validate_results_against_frame_outcomes,
)
from mulligan.real.lifecycle.tasks import get_task_spec
from mulligan.tools.dataset_lineage import sha256_file


@dataclass(frozen=True)
class SnapshotSpec:
    repo_id: str
    revision: str
    expected_file_sha256: dict[str, str]
    results_filename: str = "results.json"
    outcome_record_filename: str = ".outcome_edit_progress.json"
    label_history_filename: str = ".label_history.jsonl"
    # (revision, sha256) of the outcome record written by an apply that predates
    # the label-history ledger.  Its episodes may be absent from the ledger only if
    # their final record entry is unchanged since that apply.
    pre_ledger_record: tuple[str, str] | None = None


@dataclass(frozen=True)
class PreparedSnapshot:
    spec: SnapshotSpec
    payload: dict[str, Any]
    outcome_record: dict[str, Any]
    source_paths: dict[str, Path]
    source_sha256: dict[str, str]
    reviewed_episode_indices: tuple[int, ...]
    unreviewed_episode_indices: tuple[int, ...]

    @property
    def review_coverage(self) -> dict[str, Any]:
        reviewed = self.reviewed_episode_indices
        unreviewed = self.unreviewed_episode_indices
        return {
            "reviewed_episode_count": len(reviewed),
            "reviewed_episode_indices": list(reviewed),
            "reviewed_episode_min": min(reviewed) if reviewed else None,
            "reviewed_episode_max": max(reviewed) if reviewed else None,
            "unreviewed_episode_count": len(unreviewed),
            "unreviewed_episode_indices": list(unreviewed),
            "unreviewed_episode_min": min(unreviewed) if unreviewed else None,
            "unreviewed_episode_max": max(unreviewed) if unreviewed else None,
            "label_rule": (
                "Use the human outcome-edit record where an episode is present; "
                "otherwise retain the pinned results.json outcome and live subtask marks."
            ),
        }


def _download_checked(
    spec: SnapshotSpec,
    filename: str,
    *,
    revision: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    revision = spec.revision if revision is None else revision
    expected = spec.expected_file_sha256[filename] if expected_sha256 is None else expected_sha256
    path = Path(
        hf_hub_download(
            spec.repo_id,
            filename,
            repo_type="dataset",
            revision=revision,
            force_download=True,
        )
    )
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{spec.repo_id}@{revision}:{filename} sha256={actual}; expected {expected}"
        )
    return path


def hybrid_subtask_mark_counts(
    payload: dict[str, Any],
    outcome_record: dict[str, Any],
    *,
    num_subtask_marks: int,
) -> tuple[dict[int, int], tuple[int, ...], tuple[int, ...]]:
    """Merge reviewed marks with untouched eval-time marks, episode by episode.

    Reviewed entries come from the outcome-editor record.  Episodes absent from
    that record retain their pinned live marks.  A terminal success must carry
    every subtask mark; one without them fails loudly (fix the label, not the
    analysis).
    """

    rollouts = {int(row["episode_index"]): row for row in payload["rollouts"]}
    if len(rollouts) != len(payload["rollouts"]):
        raise RuntimeError("results payload contains duplicate episode_index values")
    if num_subtask_marks == 0:
        # No mid-episode marks exist for the task, so the outcome review itself
        # (an entry in the record) is the reviewed signal.
        reviewed = tuple(sorted(int(ep) for ep in outcome_record["changed_episodes"]))
        unknown = sorted(set(reviewed) - set(rollouts))
        if unknown:
            raise RuntimeError(f"outcome record references unknown episodes: {unknown}")
        marked = sorted(ep for ep, rollout in rollouts.items() if rollout.get("subtask_frames"))
        if marked:
            raise RuntimeError(f"task has no subtask marks, but episodes {marked} carry some")
        unreviewed = tuple(sorted(set(rollouts) - set(reviewed)))
        return {ep: 0 for ep in rollouts}, reviewed, unreviewed

    reviewed_marks = subtask_reviewed_mark_counts(outcome_record)
    unknown = sorted(set(reviewed_marks) - set(rollouts))
    if unknown:
        raise RuntimeError(f"outcome record references unknown episodes: {unknown}")

    mark_counts: dict[int, int] = {}
    for episode_index, rollout in rollouts.items():
        if episode_index in reviewed_marks:
            marks = int(reviewed_marks[episode_index])
        else:
            if "subtask_frames" not in rollout:
                raise RuntimeError(
                    f"unreviewed episode {episode_index} has no live subtask_frames field"
                )
            marks = len(rollout["subtask_frames"])
        if not 0 <= marks <= num_subtask_marks:
            raise RuntimeError(
                f"episode {episode_index}: subtask mark count {marks} outside "
                f"0..{num_subtask_marks}"
            )
        mark_counts[episode_index] = marks

    success_missing_marks = sorted(
        episode_index
        for episode_index, rollout in rollouts.items()
        if rollout["outcome"] == "success" and mark_counts[episode_index] != num_subtask_marks
    )
    if success_missing_marks:
        raise RuntimeError(
            f"terminal-success episodes without all {num_subtask_marks} subtask mark(s): "
            f"{success_missing_marks}"
        )

    reviewed = tuple(sorted(reviewed_marks))
    unreviewed = tuple(sorted(set(rollouts) - set(reviewed)))
    if set(mark_counts) != set(rollouts):
        raise RuntimeError("hybrid mark merge did not cover every rollout")
    return mark_counts, reviewed, unreviewed


def stale_subtask_marks(payload: dict[str, Any], record: dict[str, Any]) -> tuple[int, ...]:
    """Reviewed episodes whose results.json ``subtask_frames`` differ from the record.

    The review record (and the parquet reward spikes its apply wrote) is the
    source of truth; the apply canonicalizes results.json to it, so any
    disagreement means the pinned results.json predates the apply.
    """

    changed = record["changed_episodes"]
    stale = []
    for rollout in payload["rollouts"]:
        entry = changed.get(str(int(rollout["episode_index"])))
        if entry is None or "subtask_frames" not in entry:
            continue
        if rollout.get("subtask_frames") != sorted({int(f) for f in entry["subtask_frames"]}):
            stale.append(int(rollout["episode_index"]))
    return tuple(sorted(stale))


def check_label_history(
    events: list[dict[str, Any]],
    outcome_record: dict[str, Any],
    *,
    reviewed_episodes: tuple[int, ...],
    pre_ledger_entries: dict[str, Any] | None = None,
) -> None:
    """Reconcile the append-only label ledger with the outcome-edit record.

    The ledger gains one event per apply, so a re-reviewed episode has several.
    Required: the ledger covers exactly the reviewed episodes, and each episode's
    latest event (file order is append order) carries the record's entry.  The
    one exception is ``pre_ledger_entries``: the record of an apply made before
    the ledger existed.  A reviewed episode may lack history only if that record
    holds it with the same entry as the final record.
    """

    latest: dict[int, dict[str, Any]] = {}
    for event in events:
        latest[int(event["episode_index"])] = event["payload"]
    changed = outcome_record["changed_episodes"]
    missing = sorted(set(reviewed_episodes) - set(latest))
    pre_ledger = pre_ledger_entries or {}
    unexplained = [ep for ep in missing if pre_ledger.get(str(ep)) != changed[str(ep)]]
    extra = sorted(set(latest) - set(reviewed_episodes))
    if unexplained or extra:
        raise RuntimeError(
            "label history episode coverage differs from the outcome record: "
            f"reviewed without history={unexplained}, history without review={extra}"
        )
    stale = sorted(
        episode for episode, payload in latest.items() if payload != changed[str(episode)]
    )
    if stale:
        raise RuntimeError(
            f"latest label-history event differs from the outcome record for episodes {stale}"
        )


def prepare_snapshot(
    spec: SnapshotSpec,
    *,
    task: str,
    expected_policy_names: tuple[str, ...],
    expected_rollouts: int,
    expected_pairing_groups: int,
    validate_frames: bool = True,
) -> PreparedSnapshot:
    """Download, hash, reconcile, and optionally frame-validate one snapshot."""

    resolved = HfApi().repo_info(spec.repo_id, repo_type="dataset", revision=spec.revision).sha
    if resolved != spec.revision:
        raise RuntimeError(
            f"revision {spec.revision!r} resolved to {resolved!r}; require one full immutable SHA"
        )

    source_paths = {name: _download_checked(spec, name) for name in spec.expected_file_sha256}
    source_sha256 = {name: sha256_file(path) for name, path in source_paths.items()}
    payload = json.loads(source_paths[spec.results_filename].read_text())
    record = json.loads(source_paths[spec.outcome_record_filename].read_text())

    stale = stale_subtask_marks(payload, record)
    if stale:
        raise RuntimeError(
            f"pinned results.json subtask_frames disagree with the review record for "
            f"{len(stale)} reviewed episodes: {list(stale)}"
        )
    replay = copy.deepcopy(payload)
    apply_outcome_edit_record(
        replay,
        record,
        overrides_filename=spec.outcome_record_filename,
    )
    if replay["rollouts"] != payload["rollouts"]:
        raise RuntimeError(
            "pinned results.json does not already contain the pinned outcome-edit record"
        )

    policy_names = tuple(row["name"] for row in payload["summary"])
    if policy_names != expected_policy_names:
        raise RuntimeError(
            f"policy roster mismatch: got {policy_names}, expected {expected_policy_names}"
        )
    if len(payload["rollouts"]) != expected_rollouts:
        raise RuntimeError(f"expected {expected_rollouts} rollouts, got {len(payload['rollouts'])}")
    groups: dict[int, list[dict[str, Any]]] = {}
    for rollout in payload["rollouts"]:
        groups.setdefault(int(rollout["manifest_idx"]), []).append(rollout)
    if len(groups) != expected_pairing_groups:
        raise RuntimeError(
            f"expected {expected_pairing_groups} manifest_idx groups, got {len(groups)}"
        )
    policy_name_by_id = {int(row["policy_id"]): str(row["name"]) for row in payload["summary"]}
    if len(policy_name_by_id) != len(expected_policy_names):
        raise RuntimeError("summary contains duplicate or missing policy_id values")
    expected_per_group = len(expected_policy_names)
    for manifest_idx, rows in groups.items():
        unknown_ids = sorted({int(row["policy_id"]) for row in rows} - set(policy_name_by_id))
        if unknown_ids:
            raise RuntimeError(f"manifest_idx {manifest_idx}: unknown policy ids {unknown_ids}")
        names = {policy_name_by_id[int(row["policy_id"])] for row in rows}
        if len(rows) != expected_per_group or names != set(expected_policy_names):
            raise RuntimeError(
                f"manifest_idx {manifest_idx}: got {len(rows)} rows / {sorted(names)}, "
                f"expected all {expected_per_group} policies"
            )

    task_spec = get_task_spec(task)
    mark_counts, reviewed, unreviewed = hybrid_subtask_mark_counts(
        payload,
        record,
        num_subtask_marks=task_spec.num_subtask_marks,
    )
    payload["_subtask_mark_counts"] = mark_counts
    payload["_subtask_skip_zero_episodes"] = []

    history_lines = source_paths[spec.label_history_filename].read_text().splitlines()
    pre_ledger_entries = None
    if spec.pre_ledger_record is not None:
        pre_revision, pre_sha256 = spec.pre_ledger_record
        pre_path = _download_checked(
            spec,
            spec.outcome_record_filename,
            revision=pre_revision,
            expected_sha256=pre_sha256,
        )
        pre_ledger_entries = json.loads(pre_path.read_text())["changed_episodes"]
    check_label_history(
        [json.loads(line) for line in history_lines],
        record,
        reviewed_episodes=reviewed,
        pre_ledger_entries=pre_ledger_entries,
    )

    if validate_frames:
        frame_outcomes = load_frame_outcomes_from_hf(
            spec.repo_id,
            revision=spec.revision,
            subtask_frames_by_episode=subtask_frames_for_validation(record, payload),
            force_download=False,
        )
        validate_results_against_frame_outcomes(payload, frame_outcomes)

    return PreparedSnapshot(
        spec=spec,
        payload=payload,
        outcome_record=record,
        source_paths=source_paths,
        source_sha256=source_sha256,
        reviewed_episode_indices=reviewed,
        unreviewed_episode_indices=unreviewed,
    )


def run_heldout_from_prepared(
    cfg: heldout_eval.HeldoutEvalConfig,
    prepared: PreparedSnapshot,
) -> dict[str, Any]:
    """Run the shared held-out battery from an already validated payload."""

    if cfg.eval_repo != prepared.spec.repo_id:
        raise ValueError(
            f"config eval_repo {cfg.eval_repo!r} != snapshot repo {prepared.spec.repo_id!r}"
        )
    spec = get_task_spec(cfg.task)
    manifest = heldout_eval.load_manifest(cfg, spec)
    payload = copy.deepcopy(prepared.payload)
    dataset = heldout_eval.resolve_eval_dataset(
        prepared.spec.repo_id, revision=prepared.spec.revision, session=cfg.eval_session
    )
    heldout_eval.validate_payload_contract(cfg, spec, payload, dataset.recorded_repo_ids)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    paired_df = heldout_eval.build_paired_rounds(cfg, spec, payload, manifest)
    policy_df = heldout_eval.build_policy_summary(cfg, payload, paired_df)
    pairwise_df = heldout_eval.build_pairwise_summary(cfg, paired_df)
    summary = heldout_eval.write_summary(cfg, payload, policy_df, paired_df, pairwise_df)
    reviewed = set(prepared.reviewed_episode_indices)
    coverage_by_policy = {}
    for policy_name in cfg.policy_names:
        prefix = cfg.policy_prefixes[policy_name]
        episode_indices = paired_df[f"{prefix}_episode_index"].astype(int)
        reviewed_count = int(episode_indices.isin(reviewed).sum())
        coverage_by_policy[policy_name] = {
            "episodes": len(episode_indices),
            "reviewed_episodes": reviewed_count,
            "unreviewed_episodes": len(episode_indices) - reviewed_count,
            "label_rule": ("outcome-editor record when reviewed; otherwise pinned eval-time label"),
        }
    summary["label_coverage_by_policy"] = coverage_by_policy
    summary["snapshot_review_coverage"] = prepared.review_coverage
    episode_columns = [f"{cfg.policy_prefixes[name]}_episode_index" for name in cfg.policy_names]
    jointly_reviewed = int(paired_df[episode_columns].astype(int).isin(reviewed).all(axis=1).sum())
    if "subtask_scoring" in summary:
        summary["subtask_scoring"]["scored_paired_rounds"] = len(paired_df)
        summary["subtask_scoring"]["reviewed_paired_rounds"] = jointly_reviewed
        summary["subtask_scoring"]["reviewed_paired_rounds_definition"] = (
            "starts whose configured-arm episodes are all present in the human review record"
        )
    (cfg.data_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    heldout_eval.plot(cfg, policy_df, paired_df, pairwise_df)
    return summary


def load_pinned_episode_lengths(
    repo_id: str,
    revision: str,
) -> tuple[dict[int, int], float]:
    """Load one immutable eval snapshot without videos and recompute all lengths."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=repo_id,
        revision=revision,
        force_cache_sync=True,
        download_videos=False,
    )
    terminal_columns = ["episode_index", "frame_index", "done", "is_valid"]
    frame_metadata = dataset.hf_dataset.select_columns(terminal_columns).with_format(None)
    return recompute_lengths(frame_metadata), float(dataset.meta.fps)


def run_episode_lengths_from_precomputed(
    cfg: EpisodeLengthConfig,
    *,
    lengths: dict[int, int],
    fps: float,
) -> dict[str, Path]:
    """Compose the shared episode-length table and plot around pinned lengths."""

    if not cfg.outcomes_csv.exists():
        raise FileNotFoundError(cfg.outcomes_csv)
    paired = pd.read_csv(cfg.outcomes_csv)
    if cfg.expected_num_starts is not None and len(paired) != cfg.expected_num_starts:
        raise ValueError(f"expected {cfg.expected_num_starts} paired starts, got {len(paired)}")
    if cfg.expected_eval_episodes is not None and len(lengths) != cfg.expected_eval_episodes:
        raise ValueError(f"expected {cfg.expected_eval_episodes} eval episodes, got {len(lengths)}")
    table = build_episode_table(cfg, lengths, paired)
    write_sidecar_csv(cfg, table, [])
    plot_episode_lengths(cfg, table, fps, [])
    return {"csv": cfg.out_csv, "svg": cfg.out_svg}


__all__ = [
    "PreparedSnapshot",
    "SnapshotSpec",
    "check_label_history",
    "hybrid_subtask_mark_counts",
    "load_pinned_episode_lengths",
    "prepare_snapshot",
    "run_episode_lengths_from_precomputed",
    "run_heldout_from_prepared",
    "sha256_file",
    "stale_subtask_marks",
]
