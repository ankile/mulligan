"""Consensus + adjudication over the blind VLM self-consistency labels.

The summarizer is task-invariant and parameterized by a
:class:`~mulligan.real.stage_specs.tasks.StageLabelTaskSpec`, so a new task is
summarized by picking a spec rather than forking the script.

The upstream blind labeler emits N self-consistency samples per episode (no v1
labels, no strict outcome, no human seeds). This module folds those samples into
one consensus row per episode and then adjudicates the consensus against priors
available in the per-episode events CSV — never overwriting the model output,
only *flagging* disagreements into a human review queue.

Adjudication rules:

Structural (task-invariant; always evaluated):
- ``sample_stage_disagreement``: the N samples did not agree on the stage.
- ``model_requested_review``: any sample set ``needs_human_review``.
- ``low_confidence``: consensus (min-over-samples) confidence is ``low``.
- ``success_endpoint_disagreement``: consensus stage == ladder success level vs
  a strict ``original_outcome`` == ``"success"`` (BOTH directions), only where a
  strict v1 outcome exists.
- ``large_stage_delta_vs_v1``: |consensus stage - v1 ``max_stage_achieved``| >= 2,
  evaluated only when the events CSV carries a ``max_stage_achieved`` column.

Object-physics (kept, but each GUARDED by events-CSV column presence so a task
whose events CSV lacks the column simply skips the rule instead of crashing):
- ``release_contradicts_sensor``: model claims a release but the proprioceptive
  gripper trace shows the jaws never reopened (needs ``gripper_reopened_at_end``).
- ``seated_contradicts_depth``: consensus seated-or-better (stage >= seated level)
  but recorded ``final_insert_depth`` is not ``"full"``.
- ``depth_contradicts_stage``: recorded depth ``"full"`` but consensus stage <
  the seated level.
- ``stage7_final_state_mismatch``: consensus stage == success level but
  ``final_state`` != ``spec.success_final_state``.

Where the success/seated rungs are needed, they come from the spec:
``spec.ladder.success_level`` (full success, e.g. S7) and ``success_level - 1``
(seated-held, e.g. S6) rather than hardcoded ``7``/``6``.

NOTE on marker R1 seed scoring: scoring agreement against held-out human seed
labels (``early_stage_v2_seed_labels.csv``) is intentionally not part of this
library. It depends on one specific marker R1 events file and episode-index space
and would attach wrong-round labels for any other run.
``summary["seed_summary"]`` is therefore always ``{}`` from this library; held-out
seed scoring is a per-run concern.

This module is genai-free: it never calls a VLM and imports nothing from
``google.genai``. It consumes the already-parsed per-sample label JSON.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from mulligan.real.stage_specs.tasks import StageLabelTaskSpec

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def _load_samples(run_dir: Path) -> list[dict[str, Any]]:
    """Load the per-sample raw labeler results from ``run_dir`` (loud on miss)."""
    for name in ["raw_results.json", "raw_results_partial.json"]:
        path = run_dir / name
        if path.exists():
            results = json.loads(path.read_text())
            if not results:
                raise ValueError(f"{path} is empty")
            return results
    raise FileNotFoundError(f"No raw results found under {run_dir}")


def _majority(values: list[Any]) -> tuple[Any, bool]:
    """Return (winner, unanimous). Ties break to the smallest value (conservative)."""
    counts = Counter(values)
    top = max(counts.values())
    winners = sorted(key for key, count in counts.items() if count == top)
    return winners[0], len(counts) == 1


def _mode_consensus(pool: list[dict[str, Any]], spec: StageLabelTaskSpec) -> tuple[str, bool]:
    """Consensus failure_mode over a sample pool: majority, ties broken by SUMMED sample
    confidence (not alphabetically), then sorted for determinism. Returns (mode, unanimous).

    Breaking ties alphabetically would give an episode whose samples split across modes at
    the same stage (e.g. regrasp_failed / stopped_after_first / timeout_holding, all S6) an
    arbitrary label. Callers should pass only the
    samples that AGREE with the consensus stage (the mode must describe THAT rung) and MUST
    surface ``not unanimous`` so a split mode is never trusted downstream."""
    field = spec.failure_mode_field
    modes = [(str(p[field]), CONFIDENCE_ORDER[str(p["confidence"])]) for p in pool]
    counts = Counter(m for m, _ in modes)
    top = max(counts.values())
    tied = sorted(m for m, c in counts.items() if c == top)
    if len(tied) == 1:
        return tied[0], len(counts) == 1
    conf = {m: sum(c for mm, c in modes if mm == m) for m in tied}
    return max(tied, key=lambda m: conf[m]), False


# Known issue kept for paper fidelity: booleans are majority-voted but times are the median of
# every sample that gave one, so they can disagree; see docs/reproduce.md, "Known issues".
def _consensus_episode(samples: list[dict[str, Any]], spec: StageLabelTaskSpec) -> dict[str, Any]:
    """Fold N self-consistency samples for one episode into a consensus row.

    Majority-vote the stage / final_state / failure_mode (ties -> lower), median
    each time field over present values, majority each boolean field, take the
    min (most conservative) confidence, and OR the model review request. The
    output column names mirror the spec field names so the merged frame keeps
    the task's vocabulary.
    """
    stage_field = spec.stage_field
    final_state_field = spec.final_state_field
    failure_mode_field = spec.failure_mode_field

    episode = samples[0]["episode_index"]
    parsed = [s["parsed"] for s in samples]
    stages = [int(p[stage_field]) for p in parsed]
    stage, stage_unanimous = _majority(stages)
    final_state, _ = _majority([str(p[final_state_field]) for p in parsed])
    # Vote the mode only among samples that AGREE with the consensus stage (the mode must
    # describe that rung), ties broken by confidence not alphabet; flag when they split.
    on_stage = [p for p in parsed if int(p[stage_field]) == stage] or parsed
    failure_mode, fm_unanimous = _mode_consensus(on_stage, spec)
    row: dict[str, Any] = {
        "episode_index": episode,
        "n_samples": len(parsed),
        stage_field: stage,
        "stage_min": min(stages),
        "stage_max": max(stages),
        "sample_stage_disagreement": not stage_unanimous,
        "sample_failure_mode_disagreement": not fm_unanimous,
        final_state_field: final_state,
        failure_mode_field: failure_mode,
        "gemini_confidence": min(
            (str(p["confidence"]) for p in parsed), key=lambda c: CONFIDENCE_ORDER[c]
        ),
        "model_requested_review": any(bool(p["needs_human_review"]) for p in parsed),
    }
    for field in spec.bool_fields:
        votes = [bool(p[field]) for p in parsed]
        row[field] = sum(votes) * 2 > len(votes)
    for field in spec.time_fields:
        # .get tolerates older run schemas that predate a time field (the current
        # labeler's parser enforces all required keys, so live labels always have
        # them); a present-but-null value is correctly treated as absent.
        values = [float(p[field]) for p in parsed if p.get(field) is not None]
        row[field] = round(median(values), 2) if values else None
    stage_notes = [str(p["notes"]) for p in parsed if int(p[stage_field]) == stage]
    row["notes"] = stage_notes[0] if stage_notes else str(parsed[0]["notes"])
    if not stage_unanimous:
        row["notes"] += f" [samples disagreed on stage: {sorted(stages)}]"
    if not fm_unanimous:
        split = sorted({str(p[failure_mode_field]) for p in on_stage})
        row["notes"] += f" [samples split on failure_mode: {split}]"
    return row


def build_consensus(samples: list[dict[str, Any]], spec: StageLabelTaskSpec) -> pd.DataFrame:
    """Group raw per-sample results by episode and build the consensus frame."""
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_episode.setdefault(int(sample["episode_index"]), []).append(sample)
    return pd.DataFrame([_consensus_episode(group, spec) for group in by_episode.values()])


def _flag(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Append ``reason;`` to ``review_reason`` for the masked rows (in place)."""
    df.loc[mask, "review_reason"] += f"{reason};"


def adjudicate(joined: pd.DataFrame, spec: StageLabelTaskSpec) -> pd.DataFrame:
    """Adjudicate the consensus<->events join, writing the ``review_reason`` column.

    Mutates and returns ``joined`` (which must already have the consensus columns
    merged onto the events rows). All marker/object-physics rules are guarded by
    events-CSV column presence so a leaner events CSV simply skips them.
    """
    stage_field = spec.stage_field
    final_state_field = spec.final_state_field
    success_level = spec.ladder.success_level
    seated_level = success_level - 1

    joined["gemini_success"] = joined[stage_field].astype(int).eq(success_level)
    joined["has_v1_outcome"] = ~joined["original_outcome"].astype(str).isin(["", "nan", "None"])
    joined["v1_success"] = joined["original_outcome"].astype(str).eq("success")
    joined["v1_success_when_available"] = joined["v1_success"].where(joined["has_v1_outcome"])
    # diagnostic vs the v1 visual-pass seed labels; minimal events CSVs have no
    # such column, in which case the delta is NA and the flag is skipped.
    if "max_stage_achieved" in joined.columns:
        joined["stage_delta_vs_v1"] = joined[stage_field].astype(int) - joined[
            "max_stage_achieved"
        ].astype(int)
    else:
        joined["stage_delta_vs_v1"] = pd.NA

    joined["review_reason"] = ""

    # --- object-physics rules: each guarded by events-CSV column presence ---
    # Both columns are guarded by presence: gripper_reopened_at_end may be absent from a
    # task's events CSV, and the released bool is task-specific, so key off
    # spec.released_field rather than a hardcoded "marker_released".
    released_field = spec.released_field if spec.released_field in joined.columns else None
    if released_field is not None and "gripper_reopened_at_end" in joined.columns:
        _flag(
            joined,
            joined[released_field] & ~joined["gripper_reopened_at_end"].astype(bool),
            "release_contradicts_sensor",
        )
    # depth cross-checks come from the v1 visual-pass seed columns; minimal
    # events CSVs have no final_insert_depth.
    if "final_insert_depth" in joined.columns:
        _flag(
            joined,
            joined[stage_field].astype(int).ge(seated_level)
            & joined["final_insert_depth"].astype(str).ne("full"),
            "seated_contradicts_depth",
        )
        _flag(
            joined,
            joined["final_insert_depth"].astype(str).eq("full")
            & joined[stage_field].astype(int).lt(seated_level),
            "depth_contradicts_stage",
        )

    # --- structural rules (task-invariant) ---
    # only adjudicate the endpoint where a strict v1 outcome exists; a 3-level
    # vocabulary (success/timeout/failure) treats any non-success as a failure
    # endpoint, and a blank/missing outcome means no strict prior to compare.
    _flag(
        joined,
        joined["has_v1_outcome"] & (joined["gemini_success"] != joined["v1_success"]),
        "success_endpoint_disagreement",
    )
    _flag(
        joined,
        joined["gemini_success"]
        & joined[final_state_field].astype(str).ne(spec.success_final_state),
        "stage7_final_state_mismatch",
    )
    _flag(joined, joined["sample_stage_disagreement"], "sample_stage_disagreement")
    if "sample_failure_mode_disagreement" in joined.columns:
        _flag(
            joined,
            joined["sample_failure_mode_disagreement"],
            "sample_failure_mode_disagreement",
        )
    _flag(joined, joined["model_requested_review"], "model_requested_review")
    _flag(joined, joined["gemini_confidence"].eq("low"), "low_confidence")
    if joined["stage_delta_vs_v1"].notna().any():
        _flag(joined, joined["stage_delta_vs_v1"].abs().ge(2), "large_stage_delta_vs_v1")

    return joined


def summarize(
    run_dir: str | Path,
    spec: StageLabelTaskSpec,
    events_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize one blind-labeler run: write the CSVs/JSON, return the summary dict.

    Loads the per-sample raw results from ``run_dir``, folds them to a consensus
    frame, joins against the per-episode events CSV (``events_csv`` or
    ``spec.events_csv`` by default), adjudicates the review flags, and writes
    ``consensus_labels.csv``, ``labels_joined.csv``, ``policy_summary.csv``,
    ``stage_counts.csv``, ``failure_mode_counts.csv``, ``review_queue.csv`` and
    ``gemini_label_summary.json`` into ``run_dir``. Returns the summary dict.
    """
    run_dir = Path(run_dir).resolve()
    if events_csv is None and spec.events_csv is None:
        raise ValueError(f"{spec.name}: pass events_csv (the released specs carry no default)")
    events_path = Path(events_csv) if events_csv is not None else Path(spec.events_csv)

    stage_field = spec.stage_field
    failure_mode_field = spec.failure_mode_field
    success_level = spec.ladder.success_level
    seated_level = success_level - 1

    samples = _load_samples(run_dir)
    consensus = build_consensus(samples, spec)

    events = pd.read_csv(events_path)
    joined = events.merge(consensus, on="episode_index", how="right", suffixes=("_v1", ""))
    if len(joined) != len(consensus):
        raise ValueError("Join changed row count; episode_index mapping is not one-to-one")

    joined = adjudicate(joined, spec)

    policy_summary = (
        joined.groupby("policy_short", dropna=False)
        .agg(
            episodes=("episode_index", "count"),
            # generic name (not the marker-flavored mean_stage_v2) so publish.py
            # reads the same column for every task.
            mean_stage=(stage_field, "mean"),
            s6_plus_rate=(stage_field, lambda s: float((s.astype(int) >= seated_level).mean())),
            gemini_success_rate=("gemini_success", "mean"),
            v1_success_rate=("v1_success_when_available", "mean"),
            flagged_rate=("review_reason", lambda s: float(s.astype(str).ne("").mean())),
            sample_disagreement_rate=("sample_stage_disagreement", "mean"),
        )
        .reset_index()
    )
    stage_counts = (
        joined.groupby(["policy_short", stage_field], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["policy_short", stage_field])
    )
    failure_mode_counts = (
        joined.groupby(["policy_short", failure_mode_field], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["policy_short", "count"], ascending=[True, False])
    )
    review_queue = joined[joined["review_reason"].astype(str).ne("")].sort_values(
        ["gemini_confidence", "episode_index"]
    )

    for name, df in [
        ("consensus_labels.csv", consensus),
        ("labels_joined.csv", joined),
        ("policy_summary.csv", policy_summary),
        ("stage_counts.csv", stage_counts),
        ("failure_mode_counts.csv", failure_mode_counts),
        ("review_queue.csv", review_queue),
    ]:
        df.to_csv(run_dir / name, index=False)

    has_v1_outcome = joined["has_v1_outcome"].astype(bool)
    endpoint_disagreements = joined[
        has_v1_outcome & (joined["gemini_success"] != joined["v1_success"])
    ]
    strict_outcome_count = int(has_v1_outcome.sum())
    v1_success_rate = (
        float(joined.loc[has_v1_outcome, "v1_success"].mean()) if strict_outcome_count else None
    )
    endpoint_agreement_rate = (
        float(
            (
                joined.loc[has_v1_outcome, "gemini_success"]
                == joined.loc[has_v1_outcome, "v1_success"]
            ).mean()
        )
        if strict_outcome_count
        else None
    )
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "episode_count": int(len(joined)),
        "samples_per_episode": int(consensus["n_samples"].max()),
        "overall_mean_stage_v2": float(joined[stage_field].mean()),
        "overall_s6_plus_rate": float((joined[stage_field].astype(int) >= seated_level).mean()),
        "overall_gemini_success_rate": float(joined["gemini_success"].mean()),
        "strict_outcome_count": strict_outcome_count,
        "overall_v1_success_rate": v1_success_rate,
        "success_endpoint_agreement_rate": endpoint_agreement_rate,
        "success_endpoint_disagreements": endpoint_disagreements["episode_index"]
        .astype(int)
        .tolist(),
        "sample_stage_disagreement_rate": float(joined["sample_stage_disagreement"].mean()),
        "release_contradicts_sensor_count": int(
            joined["review_reason"].str.contains("release_contradicts_sensor").sum()
        ),
        "seated_contradicts_depth_count": int(
            joined["review_reason"].str.contains("seated_contradicts_depth").sum()
        ),
        "review_queue_count": int(len(review_queue)),
        # seed scoring is a marker-R1-specific concern, not part of the task-invariant
        # library; see the module docstring.
        "seed_summary": {},
    }
    (run_dir / "gemini_label_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
