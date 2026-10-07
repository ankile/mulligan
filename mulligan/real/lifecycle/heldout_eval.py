"""Shared held-out paired-eval ingest + plot for real task lines.

Ingests a held-out paired eval of N >= 2 policy arms on any registered
:class:`~mulligan.real.lifecycle.tasks.RealTaskSpec`.
It reads ``results.json`` from an HF eval repo, snapshots the raw
payload, builds paired-round outcomes across all arms on a shared
initial-state manifest, computes per-arm Wilson CIs and pairwise paired-delta
/ bootstrap-CI / exact-McNemar comparisons, writes CSVs + a
``summary.json``, and renders the 4-panel result SVG.

Multi-sub-goal tasks (``RealTaskSpec.num_subtask_marks > 0``, e.g. routing_d2's
two successive clip seats) can additionally enable ``subtask_scoring``: each
episode is graded 0..(num_subtask_marks+1) from the outcome-editor record's
reviewed ``subtask_frames`` plus terminal success, and the ingest adds
mean-score / per-threshold (score >= k) paired statistics and a graded figure.
Binary success (== the top score) stays reported unchanged alongside.

Which repo and revision are read (:func:`resolve_eval_dataset`): a released
``mulligan/*`` evaluation dataset at its pin in ``release/revisions.json`` (a round
dataset that merges several sessions keeps each session's files under
``meta/sessions/<session>/``); a source repo id named in the release manifests (the
id a session's ``results.json`` records) as the released dataset that holds it; any
other repo (your own evaluation) at Hub ``main``, or at ``eval_revision``.

Convention — library-first: a per-round driver constructs a
:class:`HeldoutEvalConfig` in Python (round-specific repo ids, arms, labels,
seeds, output paths) and calls :func:`run`. Round-evolving choices stay in the
driver; everything here is round-invariant mechanics. Colors must come from
``mulligan.plotting.colors`` (the driver passes ``METHOD_COLORS[...]`` values
into ``policy_colors``); never hardcode hex. The CLI builds a binary-success
config for one session from the release manifests (:func:`config_from_release`):

    python -m mulligan.real.lifecycle.heldout_eval mulligan/real-square-d2-r02-eval \\
        --out outputs/real/heldout/square_d2_r02
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from matplotlib.patches import Patch

from mulligan.plotting.colors import METHOD_COLORS, stage_color
from mulligan.real.eval.outcome_results import (
    DEFAULT_OUTCOME_OVERRIDES,
    OUTCOME_ORDER,
    apply_outcome_edit_record,
    load_frame_outcomes_from_hf,
    load_hf_json,
    load_outcome_edit_record,
    subtask_frames_for_validation,
    subtask_mark_count_error,
    subtask_reviewed_mark_counts,
    validate_results_against_frame_outcomes,
)
from mulligan.real.lifecycle.plotting import REPO_ROOT
from mulligan.real.lifecycle.plotting import display_path as _display_path
from mulligan.real.lifecycle.plotting import write_svg
from mulligan.real.lifecycle.stats import (
    bootstrap_paired_delta,
    exact_mcnemar_pvalue,
    signflip_permutation_pvalue,
    wilson_ci,
)
from mulligan.real.lifecycle.tasks import (
    RealTaskSpec,
    find_task_spec_by_task_name,
    get_task_spec,
)
from mulligan.tools.dataset_lineage import sha256_file as _sha256_file

# The eval results writer records the manipulated object's pose under pen_*
# keys regardless of task (the tooling originated on the first marker line);
# spec.state_keys gives the matching manifest keys in the same (x, y, yaw)
# order.
RESULT_POSE_KEYS = ("pen_x", "pen_y", "pen_yaw")

OUTCOME_ORDER = list(OUTCOME_ORDER)
OUTCOME_COLORS = {
    "success": METHOD_COLORS["outcome_success"],
    "timeout": METHOD_COLORS["outcome_timeout"],
    "failure": METHOD_COLORS["outcome_failure"],
}
GRID_COLOR = METHOD_COLORS["gray_neutral"]


@dataclass(frozen=True)
class HeldoutEvalConfig:
    """Round-specific inputs for one held-out paired eval ingest."""

    task: str
    """RealTaskSpec lifecycle key, e.g. ``"square_d2"``."""

    eval_repo: str
    """HF dataset repo id holding the eval ``results.json``: a released ``mulligan/*``
    evaluation dataset, the source id of one, or your own repo
    (:func:`resolve_eval_dataset`)."""

    manifest_path: Path
    """Blind initial-state manifest shared by all arms."""

    expected_total_rounds: int
    """Number of paired rounds in the manifest / eval plan."""

    policy_names: tuple[str, ...]
    """Ordered arm names as recorded in ``results.json`` (first = baseline)."""

    policy_labels: dict[str, str]
    """Display label per arm; may contain ``\\n`` for two-line axis labels."""

    policy_colors: dict[str, str]
    """Color per arm — values MUST come from ``mulligan.plotting.colors``."""

    policy_prefixes: dict[str, str]
    """Short CSV column prefix per arm (e.g. ``"baseline"``, ``"mulligan"``)."""

    data_dir: Path
    """Output dir for results snapshot, CSVs, and summary.json."""

    plot_path: Path
    """Output SVG path."""

    plot_title: str
    """Figure suptitle."""

    plot_caption_prefix: str
    """Provenance caption; ``"; {completed}/{total} paired rounds complete."``
    is appended."""

    bootstrap_seed: int
    """Seed for the paired-delta bootstrap (must be pinned per round)."""

    expected_manifest_sha256: str | None = None
    """Optional sha256 pin for ``manifest_path``. Set this in durable per-round
    ingest scripts so placement/state corrections cannot drift silently."""

    expected_initial_states_manifest_name: str | None = None
    """Optional basename expected in the eval payload's
    ``args.initial_states_manifest``. This catches robot sessions pointed at the
    wrong manifest file even when the local analysis manifest exists."""

    n_boot: int = 20_000
    """Bootstrap resamples."""

    wilson_z: float = 1.96
    """z for per-arm Wilson CIs (1.96, not Z_95)."""

    comparison_pairs: tuple[tuple[str, str], ...] | None = None
    """Designated (a, b) comparisons; default = every arm vs the first arm."""

    legend_labels: dict[str, str] | None = None
    """Figure-legend label per arm; default = flattened ``policy_labels``."""

    expect_pure_dp: bool = True
    """Assert ``num_action_samples`` is unset (pure-DP arms, no re-ranking)."""

    manifest_index_keys: tuple[str, ...] = ("sobol_stream_index",)
    """Integer provenance columns carried from the manifest into the paired CSV."""

    outcome_overrides_filename: str | None = None
    """Optional outcome-editor record in the eval repo (e.g.
    ``".outcome_edit_progress.json"``). When set, each reviewed episode's
    ``new_outcome`` OVERRIDES the frozen ``results.json`` outcome — the
    outcome-edited dataset is the source of truth, and ``results.json`` is a
    snapshot from eval time. For terminal outcomes (success/failure) with a
    marked ``outcome_frame``, ``num_steps`` is set to ``outcome_frame + 1``
    (the done=1 point); timeouts keep raw lengths. Per-policy summary counts
    are recomputed from the reconciled rollouts, and the applied changes
    (including any success flips) are printed loudly and recorded in
    summary.json."""

    eval_revision: str | None = None
    """Revision to read; default the release pin of a released dataset, else ``main``."""

    eval_session: str | None = None
    """Session (``bNN``) of a released round dataset that merges several sessions."""

    extra_summary_fields: dict[str, Any] = field(default_factory=dict)
    """Extra provenance merged into summary.json (e.g. ``{"sobol_seed": ...}``)."""

    subtask_scoring: bool = False
    """Grade each episode 0..(num_subtask_marks+1) instead of binary success.

    For a multi-sub-goal task (e.g. routing_d2: seat the rope in two clips in
    succession, ``num_subtask_marks=1``) the per-episode score counts completed
    sub-goals: recorded mid-episode subtask marks from the outcome-editor record
    plus 1 for terminal success — routing scores 0 (no clip), 1 (first clip
    seated, then fumbled/failed/timed out), 2 (both clips = success). Requires
    ``outcome_overrides_filename`` (the reviewed edit record is the ONLY source
    of the marks) and a FULL subtask review: any configured-arm episode without
    a stamped ``subtask_frames`` key fails the ingest loudly. Adds
    ``{prefix}_score``/``{prefix}_subtask_marks`` paired columns, mean-score +
    per-threshold (score >= k) Wilson/McNemar/bootstrap stats, and switches the
    figure to the graded panels."""

    score_threshold_labels: dict[int, str] | None = None
    """Display label per score threshold k (1..max_score) for graded plots,
    e.g. ``{1: "first clip seated", 2: "both clips (success)"}``. Default
    ``"score >= k"``."""

    subtask_skip_means_zero: bool = False
    """Score an episode in the edit record's ``skipped_episodes`` as reviewed
    with 0 marks. The editor persists a skip ('n') as an explicit operator
    decision to keep the eval-time labels — which by construction carry no
    mid-episode reward spikes — so under a subtask-mark review protocol a skip
    IS a 0-mark review. Enable per round script ONLY when the record is known
    to come from a subtask-mark session (a pre-subtask session's skips are
    indistinguishable and would silently score 0). Episodes absent from BOTH
    ``changed_episodes`` and ``skipped_episodes`` still fail loudly, and a
    skipped SUCCESS still fails the mark-count legality gate (a success must
    carry exactly ``num_subtask_marks`` marks, which a skip cannot)."""

    allow_partial_subtask_review: bool = False
    # PRELIMINARY graded scoring from the eval's LIVE operator marks (results.json
    # ``subtask_frames``) when no outcome-editor review exists yet. Stamped loudly in
    # summary.json; the reviewed ingest replaces it once the outcome-editor pass lands.
    subtask_marks_from_live_records: bool = False
    # Live-mark ONLY: SUCCESS episodes the operator terminated without pressing the
    # mid-episode mark key. A terminal success forces every earlier sub-goal (routing:
    # the second clip cannot be seated before the first), so these score full marks.
    # The ingest raises unless the set of such episodes in results.json EXACTLY
    # matches this tuple — each one is an explicit, audited operator slip, never an
    # inferred fixup. Stamped in summary.json.
    live_success_zero_mark_episodes: tuple[int, ...] = ()
    # Analyze only rounds <= max_rounds (1-based round ids), e.g. a "first 20 starts"
    # snapshot of a block that keeps growing. None = every complete round.
    max_rounds: int | None = None
    """Allow a prefix of complete paired rounds to carry graded scores.

    Binary outcomes still use every completed paired round. Graded summaries
    use only rounds for which every configured arm has a subtask review; a
    round reviewed for only some arms fails loudly. The strict full-review
    contract remains the default and should be used for final ingests."""

    def __post_init__(self) -> None:
        if len(self.policy_names) < 2:
            raise ValueError(f"need >= 2 arms, got {self.policy_names!r}")
        if (
            self.subtask_scoring
            and self.outcome_overrides_filename is None
            and not self.subtask_marks_from_live_records
        ):
            raise ValueError(
                "subtask_scoring requires outcome_overrides_filename: graded scores are "
                "read from the reviewed outcome-editor record, never inferred"
            )
        if self.allow_partial_subtask_review and not self.subtask_scoring:
            raise ValueError("allow_partial_subtask_review requires subtask_scoring")
        if self.subtask_marks_from_live_records and not self.subtask_scoring:
            raise ValueError("subtask_marks_from_live_records requires subtask_scoring")
        if self.subtask_marks_from_live_records and self.outcome_overrides_filename is not None:
            raise ValueError(
                "subtask_marks_from_live_records is for the PRE-review read; drop it once an "
                "outcome_overrides_filename exists"
            )
        if self.live_success_zero_mark_episodes and not self.subtask_marks_from_live_records:
            raise ValueError(
                "live_success_zero_mark_episodes requires subtask_marks_from_live_records "
                "(a reviewed edit record must carry the marks itself)"
            )
        if self.max_rounds is not None and self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        for mapping, what in (
            (self.policy_labels, "policy_labels"),
            (self.policy_colors, "policy_colors"),
            (self.policy_prefixes, "policy_prefixes"),
        ):
            missing = [name for name in self.policy_names if name not in mapping]
            if missing:
                raise ValueError(f"{what} missing entries for {missing}")
        prefixes = [self.policy_prefixes[name] for name in self.policy_names]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError(f"policy_prefixes must be unique, got {prefixes}")
        for a, b in self.resolved_pairs:
            for name in (a, b):
                if name not in self.policy_names:
                    raise ValueError(f"comparison pair ({a}, {b}) references unknown arm {name!r}")

    @property
    def resolved_pairs(self) -> tuple[tuple[str, str], ...]:
        if self.comparison_pairs is not None:
            return self.comparison_pairs
        baseline = self.policy_names[0]
        return tuple((name, baseline) for name in self.policy_names[1:])

    def flat_label(self, name: str) -> str:
        return self.policy_labels[name].replace("\n", " ")

    def legend_label(self, name: str) -> str:
        if self.legend_labels is not None:
            return self.legend_labels[name]
        return self.flat_label(name)


def load_manifest(cfg: HeldoutEvalConfig, spec: RealTaskSpec) -> dict[int, dict[str, Any]]:
    manifest_sha = _sha256_file(cfg.manifest_path)
    if cfg.expected_manifest_sha256 is not None and manifest_sha != cfg.expected_manifest_sha256:
        raise RuntimeError(
            f"{cfg.manifest_path}: manifest sha256={manifest_sha}; "
            f"expected {cfg.expected_manifest_sha256}"
        )
    payload = json.loads(cfg.manifest_path.read_text())
    if payload["task"] != spec.task_name:
        raise RuntimeError(f"expected {spec.task_name} manifest, got {payload['task']!r}")
    required_keys = (*spec.manifest_keys, *cfg.manifest_index_keys)
    states: dict[int, dict[str, Any]] = {}
    for row in payload["states"]:
        idx = int(row["manifest_idx"])
        if idx in states:
            raise RuntimeError(f"{cfg.manifest_path}: duplicate manifest_idx {idx}")
        missing = [key for key in required_keys if key not in row]
        if missing:
            raise RuntimeError(f"{cfg.manifest_path}: manifest_idx {idx} missing keys {missing}")
        states[idx] = row
    if len(states) != cfg.expected_total_rounds:
        raise RuntimeError(
            f"{cfg.manifest_path}: expected {cfg.expected_total_rounds} states, got {len(states)}"
        )
    return states


@dataclass(frozen=True)
class EvalDataset:
    """Where the ingest reads one evaluation session."""

    repo_id: str
    revision: str
    recorded_repo_ids: tuple[str, ...]
    """Accepted ``args.hf_repo_id`` of the session's ``results.json``: the repo itself,
    plus the source repo id for released data (the id recorded at eval time)."""
    session: str | None = None
    session_dir: str = ""
    """Prefix of the session's ``results.json`` and outcome record in the repo; empty
    except in a round dataset that merges several sessions (``meta/sessions/<id>/``)."""
    released: bool = False

    def path(self, filename: str) -> str:
        return f"{self.session_dir}{filename}"

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "session": self.session,
            "released": self.released,
        }


@cache
def _release_eval_index() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Released evaluation rows of ``release/datasets.json`` and the round-dataset lock."""
    from mulligan.release.download import select_datasets
    from mulligan.release.round_counts import load_lock

    evals = {row["repo"]: row for row in select_datasets(role=["evaluation"])}
    lock = {row["id"]: row for row in load_lock()[0]["datasets"]}
    return evals, lock


def resolve_eval_dataset(
    repo: str, *, revision: str | None = None, session: str | None = None
) -> EvalDataset:
    """Repo, revision and session directory the ingest reads for ``repo``.

    A released evaluation dataset is read at its pin (``release/revisions.json``) unless
    ``revision`` is given. Any other repo is read at ``revision`` or Hub ``main``.
    """
    from mulligan.release.download import pinned_revision

    try:
        evals, lock = _release_eval_index()
    except FileNotFoundError:
        # Installed package without a release checkout: only unreleased repos resolve.
        if repo.startswith("mulligan/"):
            raise
        evals, lock = {}, {}
    if repo not in evals:
        if repo.startswith("mulligan/"):
            raise KeyError(
                f"{repo} is not a released evaluation dataset "
                "(release/datasets.json, role 'evaluation')"
            )
        if session is not None:
            raise ValueError(f"session {session!r} applies to released round datasets only")
        return EvalDataset(repo, revision or "main", (repo,))

    entry = lock.get(repo)
    sessions = {s["session_id"]: s for s in entry["sessions"]} if entry else {}
    if entry is not None and entry["build"] == "rebuild":
        if session not in sessions:
            raise ValueError(
                f"{repo} merges sessions {sorted(sessions)}; set the session "
                f"(eval_session / --session), got {session!r}"
            )
        recorded = (sessions[session]["source_repo"],)
        session_dir = f"meta/sessions/{session}/"
    else:
        if session is not None and session not in sessions:
            raise ValueError(f"{repo} holds sessions {sorted(sessions)}, not {session!r}")
        # the repo ids its results.json files record (the lock's sessions released here)
        recorded = tuple(
            sorted(
                {
                    s["source_repo"]
                    for e in lock.values()
                    for s in e["sessions"]
                    if e["id"] == repo or s.get("release_repo") == repo
                }
            )
        )
        session_dir = ""
        if session is None and len(sessions) == 1:
            session = next(iter(sessions))
    pin = pinned_revision(repo)
    if revision is not None and revision != pin:
        print(f"[heldout_eval] reading {repo} at {revision}, not at its release pin {pin}")
    return EvalDataset(
        repo,
        revision or pin,
        (repo, *recorded),
        session=session,
        session_dir=session_dir,
        released=True,
    )


def fetch_results_payload(dataset: EvalDataset) -> dict[str, Any]:
    """Download + parse a session's ``results.json`` (a branch is re-fetched fresh)."""
    return load_hf_json(
        dataset.repo_id,
        dataset.path("results.json"),
        revision=dataset.revision,
        force_download=not dataset.released,
    )


def _session_episode_map(dataset: EvalDataset) -> dict[int, int]:
    """Session episode index -> episode index in a merged round dataset."""
    path = hf_hub_download(
        dataset.repo_id,
        "meta/episode_provenance.parquet",
        repo_type="dataset",
        revision=dataset.revision,
    )
    prov = pd.read_parquet(path, columns=["episode_index", "session_id", "source_episode_index"])
    rows = prov[prov["session_id"] == dataset.session]
    if rows.empty or rows["source_episode_index"].duplicated().any():
        raise RuntimeError(
            f"{dataset.repo_id}: meta/episode_provenance.parquet has no unique episode rows "
            f"for session {dataset.session!r}"
        )
    return dict(
        zip(
            rows["source_episode_index"].astype(int).tolist(),
            rows["episode_index"].astype(int).tolist(),
            strict=True,
        )
    )


def validate_against_frames(
    cfg: HeldoutEvalConfig,
    dataset: EvalDataset,
    payload: dict[str, Any],
    record: dict[str, Any] | None,
) -> None:
    """Check the payload's outcomes and lengths against the dataset's frame labels.

    The subtask frames (live eval-time marks, overridden by the review record) are
    threaded into the frame-outcome loader: deliberate mid-episode reward=1.0 spikes
    would otherwise trip the nonzero-pre-terminal-reward guard. A merged round dataset
    renumbers the session's episodes and keeps only released policies' episodes: the
    check maps them through ``meta/episode_provenance.parquet`` and requires every
    episode of the configured arms.
    """
    subtask_frames = subtask_frames_for_validation(record, payload)
    if not dataset.session_dir:
        validate_results_against_frame_outcomes(
            payload,
            load_frame_outcomes_from_hf(
                dataset.repo_id,
                revision=dataset.revision,
                subtask_frames_by_episode=subtask_frames,
                force_download=not dataset.released,
            ),
        )
        return
    to_dataset = _session_episode_map(dataset)
    frame_outcomes = load_frame_outcomes_from_hf(
        dataset.repo_id,
        revision=dataset.revision,
        subtask_frames_by_episode={
            to_dataset[ep]: frames for ep, frames in subtask_frames.items() if ep in to_dataset
        },
        force_download=False,
        episodes=set(to_dataset.values()),
    )
    kept = [r for r in payload["rollouts"] if int(r["episode_index"]) in to_dataset]
    dropped = [r for r in payload["rollouts"] if int(r["episode_index"]) not in to_dataset]
    configured = {_pid(payload, name) for name in cfg.policy_names}
    missing = sorted(int(r["episode_index"]) for r in dropped if int(r["policy_id"]) in configured)
    if missing:
        raise RuntimeError(
            f"{dataset.repo_id} session {dataset.session} does not hold episodes {missing} of "
            "the configured arms (a round dataset keeps released policies' episodes only)"
        )
    if dropped:
        names = {int(row["policy_id"]): row["name"] for row in payload["summary"]}
        arms = sorted({names[int(r["policy_id"])] for r in dropped})
        print(
            f"[heldout_eval] {len(dropped)} episode(s) of unconfigured arms {arms} are not in "
            f"{dataset.repo_id}; frame-checked the {len(kept)} released episodes"
        )
    validate_results_against_frame_outcomes(
        {**payload, "rollouts": kept},
        {ep: frame_outcomes[to_dataset[ep]] for ep in to_dataset},
    )


def validate_payload_contract(
    cfg: HeldoutEvalConfig,
    spec: RealTaskSpec,
    payload: dict[str, Any],
    recorded_repo_ids: tuple[str, ...],
) -> None:
    """Check a results payload's eval args and arm roster against the config."""
    if payload["args"]["environment"] != spec.task_name:
        raise RuntimeError(
            f"expected {spec.task_name} eval, got {payload['args']['environment']!r}"
        )
    if payload["args"]["hf_repo_id"] not in recorded_repo_ids:
        raise RuntimeError(
            f"{recorded_repo_ids[0]}: expected results recorded for {list(recorded_repo_ids)}, "
            f"got {payload['args']['hf_repo_id']}"
        )
    if cfg.expected_initial_states_manifest_name is not None:
        actual_name = Path(str(payload["args"]["initial_states_manifest"])).name
        if actual_name != cfg.expected_initial_states_manifest_name:
            raise RuntimeError(
                "eval payload initial_states_manifest basename mismatch: "
                f"expected {cfg.expected_initial_states_manifest_name!r}, got {actual_name!r}"
            )
    # Held-out evals run from the manifest only. Results files of the original collection
    # tooling record ``random_arena_slots`` (always 0 for the paper evals); new ones omit it.
    if int(payload["args"].get("random_arena_slots", 0)) != 0:
        raise RuntimeError(
            f"held-out eval must not sample extra policy slots, got "
            f"random_arena_slots={payload['args']['random_arena_slots']}"
        )
    if cfg.expect_pure_dp and payload["args"]["num_action_samples"] is not None:
        raise RuntimeError("pure-DP held-out eval should not set num_action_samples")
    policy_names = {row["name"] for row in payload["summary"]}
    expected_names = set(cfg.policy_names)
    # The configured arms must be a SUBSET of the payload's arms, not an exact
    # match: a results.json may carry extra arms that were retired mid-session
    # via --drop-fixed-policy (e.g. an arm that ran only the first N rounds).
    # Analyzing the surviving arms means listing just those in cfg.policy_names;
    # the extra arm's rollouts are ignored downstream. Missing configured arms
    # are still a hard error.
    missing_names = expected_names - policy_names
    if missing_names:
        raise RuntimeError(
            f"configured arms {sorted(missing_names)} are absent from the eval payload "
            f"(payload arms: {sorted(policy_names)})"
        )
    if len(payload["round_plans"]) != cfg.expected_total_rounds:
        raise RuntimeError(
            f"expected {cfg.expected_total_rounds} round plans, got {len(payload['round_plans'])}"
        )


def load_results(cfg: HeldoutEvalConfig, spec: RealTaskSpec) -> dict[str, Any]:
    dataset = resolve_eval_dataset(
        cfg.eval_repo, revision=cfg.eval_revision, session=cfg.eval_session
    )
    payload = fetch_results_payload(dataset)
    validate_payload_contract(cfg, spec, payload, dataset.recorded_repo_ids)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    original_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    record = None
    if cfg.outcome_overrides_filename is not None:
        record = load_outcome_edit_record(
            dataset.repo_id,
            dataset.path(cfg.outcome_overrides_filename),
            revision=dataset.revision,
            required=True,
            force_download=not dataset.released,
        )
        apply_outcome_overrides(dataset, cfg.outcome_overrides_filename, payload, record)
    try:
        validate_against_frames(cfg, dataset, payload, record)
    except RuntimeError as exc:
        if record is None:
            raise
        raise RuntimeError(
            f"{exc}\nThe outcome record was applied on top of results.json. If results.json "
            "already carries it (mulligan/real-square-d2-r05-eval), ingest without it: "
            "outcome_overrides_filename=None / --no-outcome-record."
        ) from exc
    canonical_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if cfg.outcome_overrides_filename is not None and canonical_text != original_text:
        backup_path = cfg.data_dir / "results_eval_time.json"
        if backup_path.exists():
            backup_text = (
                json.dumps(json.loads(backup_path.read_text()), indent=2, sort_keys=True) + "\n"
            )
            if backup_text != original_text:
                raise RuntimeError(
                    f"{backup_path} already exists and differs from the fetched eval-time "
                    "results payload; refusing to overwrite provenance"
                )
        else:
            backup_path.write_text(original_text)
    (cfg.data_dir / "results.json").write_text(canonical_text)
    # Private keys are attached AFTER the snapshot writes so they never land in results.json.
    payload["_eval_dataset"] = dataset.summary
    if cfg.subtask_scoring and cfg.subtask_marks_from_live_records:
        # PRELIMINARY: the operator's live 'g' marks recorded at eval time, unreviewed.
        mark_counts = {
            int(r["episode_index"]): len(r.get("subtask_frames", [])) for r in payload["rollouts"]
        }
        print(
            "GRADED SCORING (PRELIMINARY): subtask marks taken from the eval's LIVE operator "
            f"marks for {len(mark_counts)} episodes; no outcome-editor review applied."
        )
        success_zero = sorted(
            int(r["episode_index"])
            for r in payload["rollouts"]
            if r["outcome"] == "success" and mark_counts[int(r["episode_index"])] == 0
        )
        accepted = sorted(cfg.live_success_zero_mark_episodes)
        if success_zero != accepted:
            raise RuntimeError(
                "LIVE marks: SUCCESS episodes with 0 mid-episode marks (operator missed the "
                f"mark key) = {success_zero}, but cfg.live_success_zero_mark_episodes = "
                f"{accepted}. Check each episode's video, then list exactly these episodes "
                "in the round script to score them as full successes."
            )
        for episode_index in success_zero:
            mark_counts[episode_index] = spec.num_subtask_marks
        if success_zero:
            print(
                f"GRADED SCORING (PRELIMINARY): {len(success_zero)} SUCCESS episode(s) with no "
                f"live mid-episode mark scored {spec.num_subtask_marks} mark(s) by explicit "
                f"config: {success_zero}"
            )
        payload["_subtask_mark_counts"] = mark_counts
        payload["_subtask_skip_zero_episodes"] = []
        payload["_live_success_zero_mark_episodes"] = success_zero
    elif cfg.subtask_scoring:
        # build_paired_rounds reads the mark counts for the graded scores.
        assert record is not None  # subtask_scoring requires the record
        mark_counts, skip_zero_episodes = graded_mark_counts(cfg, record)
        if skip_zero_episodes:
            print(
                f"GRADED SCORING: {len(skip_zero_episodes)} episode(s) score 0 marks from "
                "an explicit operator SKIP in the edit record (subtask_skip_means_zero): "
                f"{skip_zero_episodes}"
            )
        payload["_subtask_mark_counts"] = mark_counts
        payload["_subtask_skip_zero_episodes"] = skip_zero_episodes
    return payload


def graded_mark_counts(
    cfg: HeldoutEvalConfig, record: dict[str, Any]
) -> tuple[dict[int, int], list[int]]:
    """Per-episode graded mark counts from a subtask-reviewed edit record.

    Returns ``(mark_counts, skip_zero_episodes)``: counts for every REVIEWED
    episode, plus (with ``cfg.subtask_skip_means_zero``) the episodes whose
    0-mark review is an explicit operator skip rather than a stamped
    ``subtask_frames`` key.
    """
    mark_counts = subtask_reviewed_mark_counts(record)
    skip_zero_episodes: list[int] = []
    if cfg.subtask_skip_means_zero:
        for ep in record.get("skipped_episodes", []):
            ep = int(ep)
            if ep not in mark_counts:
                mark_counts[ep] = 0
                skip_zero_episodes.append(ep)
        skip_zero_episodes.sort()
    return mark_counts, skip_zero_episodes


def apply_outcome_overrides(
    dataset: EvalDataset,
    overrides_filename: str,
    payload: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Reconcile ``results.json`` outcomes with the outcome-editor record.

    Any on-disk ``results.json`` snapshot stays raw (eval-time truth); the
    in-memory payload used for ALL analysis carries the edited outcomes, and
    per-policy summary counts are recomputed to match.
    """
    reconciliation = apply_outcome_edit_record(
        payload,
        record,
        overrides_filename=overrides_filename,
    )
    print("=" * 72)
    print(
        f"OUTCOME OVERRIDES APPLIED from {dataset.repo_id}@{dataset.revision[:12]}:"
        f"{dataset.path(overrides_filename)}: "
        f"{reconciliation['episodes_reviewed']} episodes reviewed, "
        f"{reconciliation['outcome_class_changes']} outcome-class changes, "
        f"{reconciliation['num_steps_patches']} num_steps patch(es) from outcome_frame, "
        f"{len(reconciliation['success_flips'])} success flip(s): "
        f"{reconciliation['success_flips']}"
    )
    print("=" * 72)


def _pid(payload: dict[str, Any], name: str) -> int:
    for row in payload["summary"]:
        if row["name"] == name:
            return int(row["policy_id"])
    raise RuntimeError(f"policy {name} not in summary")


def _assert_pose_matches_manifest(
    spec: RealTaskSpec, row: dict[str, Any], manifest_row: dict[str, Any]
) -> None:
    # Pen / nut tasks record their 3-DOF manipulated pose under pen_x/pen_y/pen_yaw, so we
    # cross-check it against the manifest state to catch an operator/manifest desync. A non-pen
    # line (routing's 1-DOF rope_x) has no free pose: the eval writer leaves pen_* None (there is
    # nothing to record), so there is nothing to cross-check on the RESULT side — the
    # manifest_idx <-> round linkage binds the state, and the sampled-placement grid check below
    # still guards the recorded targets. Assert the writer really did leave the pose unset so a
    # future writer that starts emitting a routing pose is caught rather than silently ignored.
    if len(spec.state_keys) == len(RESULT_POSE_KEYS):
        for result_key, manifest_key in zip(RESULT_POSE_KEYS, spec.state_keys, strict=True):
            got = float(row[result_key])
            expected = float(manifest_row[manifest_key])
            if not math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(
                    f"manifest_idx {row['manifest_idx']} {result_key}={got} "
                    f"does not match {manifest_key}={expected}"
                )
    else:
        recorded = [row.get(key) for key in RESULT_POSE_KEYS]
        if any(value is not None for value in recorded):
            raise RuntimeError(
                f"non-pen task {spec.name!r} unexpectedly recorded a pen pose "
                f"{dict(zip(RESULT_POSE_KEYS, recorded))} at manifest_idx {row['manifest_idx']}; "
                "the ingest pose cross-check assumes non-pen lines leave pen_* unset"
            )
    # Sampled scene placements (e.g. marker_d2 holder_x/holder_y) have NO eval-side
    # robot-state record, so we CANNOT verify the operator physically set the right
    # spot. We DO validate the manifest's coordinates land on the registry grid (and
    # in-bounds) so a garbage value never rides silently into the paired analysis.
    # No-op for tasks without sampled placements.
    for placement_name, placement in spec.sampled_placements.items():
        kx, ky = placement.keys
        x = float(manifest_row[kx])
        y = float(manifest_row[ky])
        if not placement.is_on_grid(x, y):
            raise RuntimeError(
                f"manifest_idx {row['manifest_idx']} {placement_name} ({kx},{ky})="
                f"({x:.5f},{y:.5f}) is not an in-bounds grid point "
                f"(bounds {placement.bounds}, snap {placement.snap_m})"
            )


def graded_max_score(spec: RealTaskSpec) -> int:
    """Top of the graded 0..max score ladder: sub-goal marks + terminal success."""
    if spec.num_subtask_marks <= 0:
        raise RuntimeError(
            f"task {spec.name!r} has num_subtask_marks={spec.num_subtask_marks}; "
            "graded subtask scoring needs at least one mid-episode sub-goal"
        )
    return spec.num_subtask_marks + 1


def _score_threshold_label(cfg: HeldoutEvalConfig, k: int) -> str:
    if cfg.score_threshold_labels is not None:
        return cfg.score_threshold_labels[k]
    return f"score >= {k}"


def build_paired_rounds(
    cfg: HeldoutEvalConfig,
    spec: RealTaskSpec,
    payload: dict[str, Any],
    manifest: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    policy_ids = {name: _pid(payload, name) for name in cfg.policy_names}
    expected_ids = set(policy_ids.values())
    rollouts = pd.DataFrame(payload["rollouts"])
    rows: list[dict[str, Any]] = []

    mark_counts: dict[int, int] | None = None
    if cfg.subtask_scoring:
        graded_max_score(spec)  # fail early on a task without sub-goals
        if "_subtask_mark_counts" not in payload:
            raise RuntimeError(
                "subtask_scoring payload carries no reviewed subtask-mark counts; load it "
                "through load_results, or attach subtask_reviewed_mark_counts(record) as "
                "payload['_subtask_mark_counts']"
            )
        mark_counts = {int(k): int(v) for k, v in payload["_subtask_mark_counts"].items()}

    for round_id, group in sorted(rollouts.groupby("round"), key=lambda item: int(item[0])):
        if cfg.max_rounds is not None and int(round_id) > cfg.max_rounds:
            continue
        got_ids = set(group["policy_id"].astype(int))
        # Keep the round if every CONFIGURED arm ran it; a round may carry extra
        # arms (e.g. an arm retired later via --drop-fixed-policy that ran this
        # early round) whose rollouts we simply ignore. A round missing any
        # configured arm is incomplete for this analysis and skipped.
        if not expected_ids.issubset(got_ids):
            continue
        manifest_indices = set(group["manifest_idx"].astype(int))
        if len(manifest_indices) != 1:
            raise RuntimeError(f"round {round_id} has manifest indices {sorted(manifest_indices)}")
        manifest_idx = int(next(iter(manifest_indices)))
        if manifest_idx not in manifest:
            raise RuntimeError(f"round {round_id} references unknown manifest_idx {manifest_idx}")
        manifest_row = manifest[manifest_idx]

        pose_rows = group[list(RESULT_POSE_KEYS)].drop_duplicates()
        if len(pose_rows) != 1:
            raise RuntimeError(f"round {round_id} has mismatched policy poses")
        _assert_pose_matches_manifest(spec, group.iloc[0].to_dict(), manifest_row)

        row: dict[str, Any] = {"round": int(round_id), "manifest_idx": manifest_idx}
        round_reviewed: list[bool] = []
        for key in spec.manifest_keys:
            row[key] = float(manifest_row[key])
        for key in cfg.manifest_index_keys:
            row[key] = int(manifest_row[key])
        for policy_name in cfg.policy_names:
            policy_id = policy_ids[policy_name]
            sub = group[group["policy_id"] == policy_id]
            if len(sub) != 1:
                raise RuntimeError(f"round {round_id} policy {policy_name} has {len(sub)} rows")
            rollout = sub.iloc[0]
            prefix = cfg.policy_prefixes[policy_name]
            row[f"{prefix}_outcome"] = rollout["outcome"]
            row[f"{prefix}_success"] = rollout["outcome"] == "success"
            row[f"{prefix}_num_steps"] = int(rollout["num_steps"])
            row[f"{prefix}_episode_index"] = int(rollout["episode_index"])
            if mark_counts is not None:
                episode_index = int(rollout["episode_index"])
                if episode_index not in mark_counts:
                    if not cfg.allow_partial_subtask_review:
                        raise RuntimeError(
                            f"episode {episode_index} (round {round_id}, arm {policy_name}) has "
                            "no subtask review in the outcome-edit record; graded scoring "
                            "requires a FULL subtask-mark pass (mulligan.tools.outcome_review) over "
                            "the eval dataset. If the pass skipped ('n') no-edit episodes and "
                            "the session is KNOWN to be a subtask-mark session, set "
                            "subtask_skip_means_zero=True."
                        )
                    round_reviewed.append(False)
                    row[f"{prefix}_subtask_marks"] = np.nan
                    row[f"{prefix}_score"] = np.nan
                    continue
                round_reviewed.append(True)
                marks = mark_counts[episode_index]
                outcome = str(rollout["outcome"])
                err = subtask_mark_count_error(outcome, marks, spec.num_subtask_marks)
                if err is not None:
                    raise RuntimeError(
                        f"episode {episode_index} (round {round_id}, arm {policy_name}): {err}"
                    )
                row[f"{prefix}_subtask_marks"] = marks
                row[f"{prefix}_score"] = marks + (1 if outcome == "success" else 0)
        if mark_counts is not None and len(set(round_reviewed)) > 1:
            reviewed_arms = [
                name
                for name, reviewed in zip(cfg.policy_names, round_reviewed, strict=True)
                if reviewed
            ]
            missing_arms = [
                name
                for name, reviewed in zip(cfg.policy_names, round_reviewed, strict=True)
                if not reviewed
            ]
            raise RuntimeError(
                f"round {round_id} has a partial cross-arm subtask review; "
                f"reviewed={reviewed_arms}, missing={missing_arms}. Partial graded ingest "
                "requires complete paired-round reviews."
            )
        rows.append(row)

    if not rows:
        raise RuntimeError("no complete paired rounds found")
    df = pd.DataFrame(rows)
    if cfg.subtask_scoring and cfg.allow_partial_subtask_review:
        score_cols = [f"{cfg.policy_prefixes[name]}_score" for name in cfg.policy_names]
        if not df[score_cols].notna().all(axis=1).any():
            raise RuntimeError("partial subtask review contains no fully reviewed paired round")
    df.to_csv(cfg.data_dir / "paired_round_outcomes.csv", index=False)
    return df


def _mean_steps(rows: pd.DataFrame, outcome: str) -> float:
    vals = rows.loc[rows["outcome"] == outcome, "num_steps"]
    return float("nan") if len(vals) == 0 else round(float(vals.mean()), 3)


def build_policy_summary(
    cfg: HeldoutEvalConfig, payload: dict[str, Any], paired_df: pd.DataFrame
) -> pd.DataFrame:
    rollouts = pd.DataFrame(payload["rollouts"])
    rows = []
    for policy_name in cfg.policy_names:
        row = next(item for item in payload["summary"] if item["name"] == policy_name)
        n = int(row["num_rounds"])
        successes = int(row["successes"])
        sub = rollouts[rollouts["policy_id"] == int(row["policy_id"])]
        if cfg.max_rounds is not None:
            # results.json counts every round: check it against all of this arm's
            # rollouts, then analyze only the rounds <= max_rounds that
            # build_paired_rounds kept.
            if len(sub) != n or int((sub["outcome"] == "success").sum()) != successes:
                raise RuntimeError(
                    f"{policy_name} summary num_rounds={n}, successes={successes} disagree "
                    f"with its {len(sub)} raw rollout(s)."
                )
            sub = sub[sub["round"].astype(int) <= cfg.max_rounds]
            n = len(sub)
            successes = int((sub["outcome"] == "success").sum())
        if n != len(paired_df):
            raise RuntimeError(
                f"{policy_name} ran {n} rounds but the paired set has {len(paired_df)} rounds. "
                "Every configured arm must share full coverage of the paired rounds. This "
                "usually means an arm was retired mid-session (--drop-fixed-policy) and has "
                "partial coverage: list only the arms that ran ALL paired rounds in "
                "cfg.policy_names (the retired arm's earlier rounds are analyzed via the "
                "original full-arm ingest of those rounds)."
            )
        lo, hi = wilson_ci(successes, n, z=cfg.wilson_z)
        # The summary's n/successes must agree with the raw rollouts AND the paired
        # rounds: the outcome decomposition (timeouts/failures/mean-steps) below is
        # computed from ALL of this arm's raw rollouts, and success_rate from the
        # summary, so a drifting summary (or an arm with rollouts in rounds that were
        # dropped from paired_df) would silently mix denominators. Fail loudly instead.
        prefix = cfg.policy_prefixes[policy_name]
        paired_successes = int(paired_df[f"{prefix}_success"].astype(bool).sum())
        if len(sub) != n:
            raise RuntimeError(
                f"{policy_name} summary num_rounds={n} but has {len(sub)} raw rollout(s); "
                "summary and rollouts disagree (every rollout must be a paired round)."
            )
        if paired_successes != successes:
            raise RuntimeError(
                f"{policy_name} summary successes={successes} but paired rounds show "
                f"{paired_successes}; summary and paired outcomes disagree."
            )
        counts = Counter(sub["outcome"])
        out_row = {
            "policy_name": policy_name,
            "policy_label": cfg.flat_label(policy_name),
            "policy_id": int(row["policy_id"]),
            "model_id": row["model_id"],
            "episodes": n,
            "successes": successes,
            "success_rate": successes / n,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "timeouts": int(counts["timeout"]),
            "early_failures": int(counts["failure"]),
            "mean_success_steps": _mean_steps(sub, "success"),
            "mean_timeout_steps": _mean_steps(sub, "timeout"),
            "mean_failure_steps": _mean_steps(sub, "failure"),
        }
        if cfg.subtask_scoring:
            spec = get_task_spec(cfg.task)
            max_score = graded_max_score(spec)
            score_series = pd.to_numeric(paired_df[f"{prefix}_score"], errors="raise")
            reviewed = score_series.notna()
            scores = score_series.loc[reviewed].astype(int).to_numpy()
            graded_n = len(scores)
            if graded_n == 0:
                raise RuntimeError(f"{policy_name}: no reviewed graded episodes")
            # Only a success reaches the top rung (marks==num_subtask_marks AND
            # terminal success), so the graded ladder must agree with the binary
            # outcomes over the reviewed subset; a mismatch means corrupted marks.
            top_count = int((scores == max_score).sum())
            graded_successes = int(paired_df.loc[reviewed, f"{prefix}_success"].astype(bool).sum())
            if top_count != graded_successes:
                raise RuntimeError(
                    f"{policy_name}: {top_count} episode(s) at graded score {max_score} but "
                    f"the reviewed subset has {graded_successes} success(es); marks and "
                    "outcomes disagree"
                )
            out_row["max_score"] = max_score
            out_row["graded_episodes"] = graded_n
            out_row["graded_successes"] = graded_successes
            out_row["mean_score"] = float(scores.mean())
            for level in range(max_score + 1):
                out_row[f"score_{level}"] = int((scores == level).sum())
            for k in range(1, max_score + 1):
                ge_count = int((scores >= k).sum())
                ge_lo, ge_hi = wilson_ci(ge_count, graded_n, z=cfg.wilson_z)
                out_row[f"ge{k}_count"] = ge_count
                out_row[f"ge{k}_rate"] = ge_count / graded_n
                out_row[f"ge{k}_wilson_lo"] = ge_lo
                out_row[f"ge{k}_wilson_hi"] = ge_hi
        rows.append(out_row)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.data_dir / "policy_summary.csv", index=False)
    return df


def build_pairwise_summary(cfg: HeldoutEvalConfig, paired_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a_name, b_name in cfg.resolved_pairs:
        a = paired_df[f"{cfg.policy_prefixes[a_name]}_success"].astype(bool).to_numpy()
        b = paired_df[f"{cfg.policy_prefixes[b_name]}_success"].astype(bool).to_numpy()
        a_only = int(np.logical_and(a, ~b).sum())
        b_only = int(np.logical_and(~a, b).sum())
        both_success = int(np.logical_and(a, b).sum())
        both_fail = int(np.logical_and(~a, ~b).sum())
        delta, ci_lo, ci_hi = bootstrap_paired_delta(
            a, b, n_boot=cfg.n_boot, seed=cfg.bootstrap_seed
        )
        out_row = {
            "a": cfg.flat_label(a_name),
            "b": cfg.flat_label(b_name),
            "a_only_success": a_only,
            "b_only_success": b_only,
            "both_success": both_success,
            "both_fail": both_fail,
            "discordant": a_only + b_only,
            "a_success_rate": float(a.mean()),
            "b_success_rate": float(b.mean()),
            "paired_delta_a_minus_b": delta,
            "paired_delta_bootstrap_ci95_lo": ci_lo,
            "paired_delta_bootstrap_ci95_hi": ci_hi,
            "mcnemar_exact_pvalue": exact_mcnemar_pvalue(a_only, b_only),
        }
        if cfg.subtask_scoring:
            spec = get_task_spec(cfg.task)
            max_score = graded_max_score(spec)
            a_series = pd.to_numeric(
                paired_df[f"{cfg.policy_prefixes[a_name]}_score"], errors="raise"
            )
            b_series = pd.to_numeric(
                paired_df[f"{cfg.policy_prefixes[b_name]}_score"], errors="raise"
            )
            reviewed = a_series.notna() & b_series.notna()
            if int(reviewed.sum()) == 0:
                raise RuntimeError(f"{a_name} vs {b_name}: no jointly reviewed graded rounds")
            a_score = a_series.loc[reviewed].astype(int).to_numpy()
            b_score = b_series.loc[reviewed].astype(int).to_numpy()
            out_row["graded_paired_n"] = len(a_score)
            score_delta, score_lo, score_hi = bootstrap_paired_delta(
                a_score, b_score, n_boot=cfg.n_boot, seed=cfg.bootstrap_seed
            )
            out_row["a_mean_score"] = float(a_score.mean())
            out_row["b_mean_score"] = float(b_score.mean())
            out_row["mean_score_delta_a_minus_b"] = score_delta
            out_row["mean_score_delta_bootstrap_ci95_lo"] = score_lo
            out_row["mean_score_delta_bootstrap_ci95_hi"] = score_hi
            out_row["mean_score_signflip_pvalue"] = signflip_permutation_pvalue(
                a_score - b_score, n_perm=cfg.n_boot, seed=cfg.bootstrap_seed
            )
            # Each threshold k is a binary paired comparison in its own right:
            # ge{max_score} duplicates the full-success columns above by design
            # (uniform k loop; downstream readers use one naming scheme).
            for k in range(1, max_score + 1):
                a_ge = a_score >= k
                b_ge = b_score >= k
                ge_delta, ge_lo, ge_hi = bootstrap_paired_delta(
                    a_ge, b_ge, n_boot=cfg.n_boot, seed=cfg.bootstrap_seed
                )
                out_row[f"ge{k}_a_rate"] = float(a_ge.mean())
                out_row[f"ge{k}_b_rate"] = float(b_ge.mean())
                out_row[f"ge{k}_a_only"] = int(np.logical_and(a_ge, ~b_ge).sum())
                out_row[f"ge{k}_b_only"] = int(np.logical_and(~a_ge, b_ge).sum())
                out_row[f"ge{k}_paired_delta_a_minus_b"] = ge_delta
                out_row[f"ge{k}_paired_delta_bootstrap_ci95_lo"] = ge_lo
                out_row[f"ge{k}_paired_delta_bootstrap_ci95_hi"] = ge_hi
                out_row[f"ge{k}_mcnemar_exact_pvalue"] = exact_mcnemar_pvalue(
                    out_row[f"ge{k}_a_only"], out_row[f"ge{k}_b_only"]
                )
        rows.append(out_row)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.data_dir / "pairwise_summary.csv", index=False)
    return df


def write_summary(
    cfg: HeldoutEvalConfig,
    payload: dict[str, Any],
    policy_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> dict[str, Any]:
    summary = {
        "eval_repo": cfg.eval_repo,
        "eval_dataset": payload.get("_eval_dataset"),
        "dataset_name": payload["dataset_name"],
        "timestamp": payload["timestamp"],
        "completed_rounds": int(len(paired_df)),
        "expected_total_rounds": cfg.expected_total_rounds,
        "is_complete": len(paired_df) == cfg.expected_total_rounds,
        "max_rounds": cfg.max_rounds,
        "subtask_marks_from_live_records": cfg.subtask_marks_from_live_records,
        "script_contract": {
            "hf_repo_id": payload["args"]["hf_repo_id"],
            "local_dataset_name": payload["args"]["dataset_name"],
            "initial_states_manifest": payload["args"]["initial_states_manifest"],
            "local_manifest_path": _display_path(cfg.manifest_path),
            "local_manifest_sha256": _sha256_file(cfg.manifest_path),
            "expected_manifest_sha256": cfg.expected_manifest_sha256,
            "num_action_samples": payload["args"]["num_action_samples"],
            "n_action_steps": payload["args"]["n_action_steps"],
        },
        "policy_summary": policy_df.to_dict(orient="records"),
        "pairwise_summary": pairwise_df.to_dict(orient="records"),
        "outcome_edit_reconciliation": payload.get("_outcome_edit_reconciliation"),
        **cfg.extra_summary_fields,
    }
    if cfg.subtask_scoring:
        spec = get_task_spec(cfg.task)
        max_score = graded_max_score(spec)
        summary["subtask_scoring"] = {
            "num_subtask_marks": spec.num_subtask_marks,
            "max_score": max_score,
            "score_threshold_labels": {
                str(k): _score_threshold_label(cfg, k) for k in range(1, max_score + 1)
            },
            "subtask_skip_means_zero": cfg.subtask_skip_means_zero,
            "allow_partial_subtask_review": cfg.allow_partial_subtask_review,
            "reviewed_paired_rounds": int(policy_df["graded_episodes"].min()),
            "zero_mark_from_skipped_episodes": payload.get("_subtask_skip_zero_episodes", []),
            "live_success_zero_mark_episodes": payload.get("_live_success_zero_mark_episodes", []),
        }
    (cfg.data_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _write_svg(fig: plt.Figure, path: Path) -> None:
    write_svg(fig, path)
    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    print(f"wrote {rel}")


def plot(
    cfg: HeldoutEvalConfig,
    policy_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> None:
    if cfg.subtask_scoring:
        _plot_graded(cfg, policy_df, paired_df, pairwise_df)
    else:
        _plot_binary(cfg, policy_df, paired_df, pairwise_df)


def _plot_binary(
    cfg: HeldoutEvalConfig,
    policy_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> None:
    plt.rcParams.update({"svg.fonttype": "path"})
    completed = len(paired_df)
    complete_label = "Complete" if completed == cfg.expected_total_rounds else "Preliminary"
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))

    ax = axes[0, 0]
    x = np.arange(len(policy_df))
    rates = policy_df["success_rate"].to_numpy() * 100.0
    yerr = np.vstack(
        [
            rates - policy_df["wilson_lo"].to_numpy() * 100.0,
            policy_df["wilson_hi"].to_numpy() * 100.0 - rates,
        ]
    )
    for idx, row in policy_df.iterrows():
        name = row["policy_name"]
        ax.bar(
            idx,
            rates[idx],
            color=cfg.policy_colors[name],
            edgecolor="black",
            linewidth=0.7,
        )
    ax.errorbar(x, rates, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    for idx, row in policy_df.iterrows():
        ax.text(
            idx,
            min(96.0, rates[idx] + yerr[1, idx] + 2.2),
            f"{int(row.successes)}/{int(row.episodes)}\n{rates[idx]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8.7,
        )
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg.policy_labels[name] for name in policy_df["policy_name"]])
    ax.set_ylabel("success rate (%)")
    ax.set_title(
        f"{complete_label} paired success rate, {completed}/{cfg.expected_total_rounds} rounds"
    )
    _style_axis(ax)

    ax = axes[0, 1]
    # "minus baseline" only when the b arm IS the baseline prefix; a 2-arm DP-vs-rerank
    # session has no baseline and must name the DP arm instead.
    baseline_name = next(
        (n for n in cfg.policy_names if cfg.policy_prefixes[n] == "baseline"), None
    )
    ax.axhline(0, color="black", linewidth=0.8)
    pair_tick_labels = []
    for pair_idx, ((a_name, b_name), (_, row)) in enumerate(
        zip(cfg.resolved_pairs, pairwise_df.iterrows(), strict=True)
    ):
        delta = float(row["paired_delta_a_minus_b"]) * 100.0
        lo = float(row["paired_delta_bootstrap_ci95_lo"]) * 100.0
        hi = float(row["paired_delta_bootstrap_ci95_hi"]) * 100.0
        ax.errorbar(
            [pair_idx],
            [delta],
            yerr=[[delta - lo], [hi - delta]],
            fmt="o",
            color=cfg.policy_colors[a_name],
            ecolor="black",
            capsize=4,
            markersize=7,
        )
        a_short = cfg.policy_prefixes[a_name].capitalize()
        b_short = cfg.policy_prefixes[b_name].capitalize()
        ax.text(
            pair_idx,
            hi + 5.0,
            f"{a_short}-only {int(row.a_only_success)}\n"
            f"{b_short}-only {int(row.b_only_success)}\n"
            f"p={float(row.mcnemar_exact_pvalue):.3f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
        )
        minus = "baseline" if b_name == baseline_name else cfg.flat_label(b_name)
        pair_tick_labels.append(f"{cfg.flat_label(a_name)}\nminus {minus}")
    ax.set_xlim(-0.7, len(cfg.resolved_pairs) - 1 + 0.7)
    # Leave enough headroom for the three-line discordance annotation. Fixed
    # +70 clipped/overlapped the title when a paired CI reached the mid-50s.
    max_hi = float(pairwise_df["paired_delta_bootstrap_ci95_hi"].max()) * 100.0
    ax.set_ylim(-60, max(90.0, max_hi + 40.0))
    ax.set_xticks(np.arange(len(cfg.resolved_pairs)))
    ax.set_xticklabels(pair_tick_labels, fontsize=10.0 if len(cfg.resolved_pairs) <= 2 else 7.4)
    ax.set_ylabel("paired delta (%)")
    ax.set_title("Paired delta (bootstrap CI)")
    _style_axis(ax)

    ax = axes[1, 0]
    bottom = np.zeros(len(policy_df))
    for outcome in OUTCOME_ORDER:
        vals = []
        for _, row in policy_df.iterrows():
            if outcome == "success":
                vals.append(row["successes"] / row["episodes"] * 100.0)
            elif outcome == "timeout":
                vals.append(row["timeouts"] / row["episodes"] * 100.0)
            else:
                vals.append(row["early_failures"] / row["episodes"] * 100.0)
        vals_arr = np.array(vals)
        ax.bar(
            x,
            vals_arr,
            bottom=bottom,
            color=OUTCOME_COLORS[outcome],
            edgecolor="black",
            linewidth=0.7,
            label=outcome,
        )
        for idx, value in enumerate(vals_arr):
            if value >= 9.0:
                ax.text(
                    idx,
                    bottom[idx] + value / 2.0,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.4,
                )
        bottom += vals_arr
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg.policy_labels[name] for name in policy_df["policy_name"]])
    ax.set_ylabel("episodes (%)")
    ax.set_title("Outcome decomposition")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.18))
    _style_axis(ax)

    ax = axes[1, 1]
    round_x = paired_df["round"].to_numpy()
    for policy_name in cfg.policy_names:
        prefix = cfg.policy_prefixes[policy_name]
        successes = paired_df[f"{prefix}_success"].astype(int).cumsum().to_numpy()
        ax.plot(
            round_x,
            successes,
            marker="o",
            markersize=3.3,
            linewidth=1.9,
            color=cfg.policy_colors[policy_name],
            label=cfg.flat_label(policy_name),
        )
    ax.set_xlim(1, max(2, int(round_x.max())))
    ax.set_ylim(0, max(12, int(policy_df["successes"].max()) + 3))
    ax.set_xlabel("completed paired round")
    ax.set_ylabel("cumulative successes")
    ax.set_title("Cumulative successes")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    _style_axis(ax)

    handles = [
        Patch(
            facecolor=cfg.policy_colors[name],
            edgecolor="black",
            label=cfg.legend_label(name),
        )
        for name in cfg.policy_names
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(cfg.policy_names),
    )
    fig.suptitle(cfg.plot_title, fontsize=12, y=0.985)
    fig.text(
        0.5,
        0.018,
        f"{cfg.plot_caption_prefix}; "
        f"{completed}/{cfg.expected_total_rounds} paired rounds complete.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.88), h_pad=2.55, w_pad=2.0)
    _write_svg(fig, cfg.plot_path)


# Distinct threshold markers for the graded paired-delta panel (k=1 first).
_THRESHOLD_MARKERS = ("s", "o", "D", "^", "v")


def _plot_graded(
    cfg: HeldoutEvalConfig,
    policy_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> None:
    """Graded-score variant of the 4-panel figure.

    A: per-arm score distribution (stacked, stage ladder) with mean score.
    B: per-threshold paired deltas + mean-score delta / sign-flip p.
    C: per-threshold rates with Wilson CIs.
    D: cumulative score across paired rounds.
    """
    spec = get_task_spec(cfg.task)
    max_score = graded_max_score(spec)
    n_levels = max_score + 1
    if max_score > len(_THRESHOLD_MARKERS):
        raise RuntimeError(f"add markers to _THRESHOLD_MARKERS for max_score={max_score}")
    plt.rcParams.update({"svg.fonttype": "path"})
    completed = len(paired_df)
    graded_completed = int(policy_df["graded_episodes"].min())
    complete_label = "Complete" if completed == cfg.expected_total_rounds else "Preliminary"
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))
    x = np.arange(len(policy_df))

    ax = axes[0, 0]
    bottom = np.zeros(len(policy_df))
    for level in range(n_levels):
        vals = (
            policy_df[f"score_{level}"].to_numpy() / policy_df["graded_episodes"].to_numpy() * 100.0
        )
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=stage_color(level, n_levels),
            edgecolor="black",
            linewidth=0.7,
            label=f"score {level}",
        )
        for idx, value in enumerate(vals):
            if value >= 9.0:
                ax.text(
                    idx,
                    bottom[idx] + value / 2.0,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.4,
                )
        bottom += vals
    for idx, (_, row) in enumerate(policy_df.iterrows()):
        ax.text(
            idx,
            101.5,
            f"mean {row['mean_score']:.2f}/{max_score}",
            ha="center",
            va="bottom",
            fontsize=8.7,
        )
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xticks(x)
    ax.set_xticklabels([cfg.policy_labels[name] for name in policy_df["policy_name"]])
    ax.set_ylabel("episodes (%)")
    ax.set_title(
        f"{complete_label} graded score (0..{max_score}), "
        f"{graded_completed}/{completed} scored rounds"
    )
    ax.legend(
        frameon=False, fontsize=8.5, loc="upper center", ncol=n_levels, bbox_to_anchor=(0.5, -0.18)
    )
    _style_axis(ax)

    ax = axes[0, 1]
    # "minus baseline" only when the b arm IS the baseline prefix; a 2-arm DP-vs-rerank
    # session has no baseline and must name the DP arm instead.
    baseline_name = next(
        (n for n in cfg.policy_names if cfg.policy_prefixes[n] == "baseline"), None
    )
    ax.axhline(0, color="black", linewidth=0.8)
    pair_tick_labels = []
    for pair_idx, ((a_name, b_name), (_, row)) in enumerate(
        zip(cfg.resolved_pairs, pairwise_df.iterrows(), strict=True)
    ):
        for k in range(1, max_score + 1):
            offset = (k - (max_score + 1) / 2.0) * 0.22
            delta = float(row[f"ge{k}_paired_delta_a_minus_b"]) * 100.0
            lo = float(row[f"ge{k}_paired_delta_bootstrap_ci95_lo"]) * 100.0
            hi = float(row[f"ge{k}_paired_delta_bootstrap_ci95_hi"]) * 100.0
            ax.errorbar(
                [pair_idx + offset],
                [delta],
                yerr=[[delta - lo], [hi - delta]],
                fmt=_THRESHOLD_MARKERS[k - 1],
                color=cfg.policy_colors[a_name],
                ecolor="black",
                capsize=4,
                markersize=7,
            )
            ax.text(
                pair_idx + offset,
                hi + 3.0,
                f"p={float(row[f'ge{k}_mcnemar_exact_pvalue']):.3g}",
                ha="right" if k == 1 else "left",
                va="bottom",
                fontsize=7.5,
            )
        minus = "baseline" if b_name == baseline_name else cfg.flat_label(b_name)
        pair_tick_labels.append(f"{cfg.flat_label(a_name)}\nminus {minus}")
    first = pairwise_df.iloc[0]
    ax.text(
        0.5,
        0.03,
        (
            f"Δ mean score {float(first['mean_score_delta_a_minus_b']):+.2f} "
            f"[{float(first['mean_score_delta_bootstrap_ci95_lo']):+.2f}, "
            f"{float(first['mean_score_delta_bootstrap_ci95_hi']):+.2f}] of 0..{max_score}; "
            f"sign-flip p={float(first['mean_score_signflip_pvalue']):.3g}"
        )
        + ("" if len(pairwise_df) == 1 else " (first pair)"),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    ax.set_xlim(-0.7, len(cfg.resolved_pairs) - 1 + 0.7)
    ax.set_ylim(-60, 70)
    ax.set_xticks(np.arange(len(cfg.resolved_pairs)))
    ax.set_xticklabels(pair_tick_labels, fontsize=10.0 if len(cfg.resolved_pairs) <= 2 else 7.4)
    ax.set_ylabel("paired delta (%)")
    ax.set_title("Paired per-threshold delta (bootstrap CI)", y=1.15)
    ax.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker=_THRESHOLD_MARKERS[k - 1],
                linestyle="none",
                color="black",
                markersize=6,
                label=_score_threshold_label(cfg, k),
            )
            for k in range(1, max_score + 1)
        ],
        frameon=False,
        fontsize=8.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=max_score,
    )
    _style_axis(ax)

    ax = axes[1, 0]
    n_arms = len(policy_df)
    width = 0.8 / n_arms
    for arm_idx, (_, row) in enumerate(policy_df.iterrows()):
        name = row["policy_name"]
        xpos = np.arange(max_score) + (arm_idx - (n_arms - 1) / 2.0) * width
        rates = np.array([row[f"ge{k}_rate"] for k in range(1, max_score + 1)]) * 100.0
        los = np.array([row[f"ge{k}_wilson_lo"] for k in range(1, max_score + 1)]) * 100.0
        his = np.array([row[f"ge{k}_wilson_hi"] for k in range(1, max_score + 1)]) * 100.0
        ax.bar(xpos, rates, width, color=cfg.policy_colors[name], edgecolor="black", linewidth=0.7)
        ax.errorbar(
            xpos,
            rates,
            yerr=np.vstack([rates - los, his - rates]),
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1.0,
        )
        for k_idx, k in enumerate(range(1, max_score + 1)):
            ax.text(
                xpos[k_idx],
                min(96.0, his[k_idx] + 2.0),
                f"{int(row[f'ge{k}_count'])}/{int(row['graded_episodes'])}",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(max_score))
    ax.set_xticklabels(
        [_score_threshold_label(cfg, k).replace("; ", "\n") for k in range(1, max_score + 1)],
        fontsize=9.0,
    )
    ax.set_ylabel("episodes (%)")
    ax.set_title("Per-threshold rate (Wilson CI)")
    _style_axis(ax)

    ax = axes[1, 1]
    score_cols = [f"{cfg.policy_prefixes[name]}_score" for name in cfg.policy_names]
    reviewed_df = paired_df[paired_df[score_cols].notna().all(axis=1)]
    round_x = reviewed_df["round"].to_numpy()
    max_cum = 0
    for policy_name in cfg.policy_names:
        prefix = cfg.policy_prefixes[policy_name]
        cum_score = reviewed_df[f"{prefix}_score"].astype(int).cumsum().to_numpy()
        max_cum = max(max_cum, int(cum_score[-1]))
        ax.plot(
            round_x,
            cum_score,
            marker="o",
            markersize=3.3,
            linewidth=1.9,
            color=cfg.policy_colors[policy_name],
            label=cfg.flat_label(policy_name),
        )
    ax.set_xlim(1, cfg.expected_total_rounds)
    ax.set_ylim(0, max(12, max_cum + 3))
    ax.set_xlabel("completed paired round")
    ax.set_ylabel(f"cumulative score (0..{max_score} per round)")
    ax.set_title("Cumulative graded score")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    _style_axis(ax)

    handles = [
        Patch(
            facecolor=cfg.policy_colors[name],
            edgecolor="black",
            label=cfg.legend_label(name),
        )
        for name in cfg.policy_names
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(cfg.policy_names),
    )
    fig.suptitle(cfg.plot_title, fontsize=12, y=0.985)
    fig.text(
        0.5,
        0.018,
        f"{cfg.plot_caption_prefix}; "
        f"{completed}/{cfg.expected_total_rounds} paired rounds complete.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.88), h_pad=2.55, w_pad=2.0)
    _write_svg(fig, cfg.plot_path)


def run(cfg: HeldoutEvalConfig) -> dict[str, Any]:
    """Full ingest + plot; returns the summary dict written to summary.json."""
    spec = get_task_spec(cfg.task)
    manifest = load_manifest(cfg, spec)
    payload = load_results(cfg, spec)
    paired_df = build_paired_rounds(cfg, spec, payload, manifest)
    policy_df = build_policy_summary(cfg, payload, paired_df)
    pairwise_df = build_pairwise_summary(cfg, paired_df)
    summary = write_summary(cfg, payload, policy_df, paired_df, pairwise_df)
    plot(cfg, policy_df, paired_df, pairwise_df)
    return summary


# CLI: one session, arms and labels from the release manifests (binary success).
_METHOD_STYLE = {
    # method -> (CSV prefix, METHOD_COLORS key); the colors of the paper's real figures
    "HG-DAgger": ("baseline", "uniform"),
    "HG-DAgger+Mulligan": ("mulligan", "real_mulligan_with_cf"),
    "HG-DAgger+Mulligan no-CF": ("mulligan_no_cf", "real_mulligan_no_cf"),
    "HiL-IDQL+Mulligan": ("hil_idql", "real_iql_rerank"),
}
_FALLBACK_COLORS = (
    "uniform",
    "real_mulligan_with_cf",
    "real_iql_rerank",
    "real_mulligan_no_cf",
    "outcome_success",
    "accent_terracotta",
)


def _lock_session(dataset: EvalDataset) -> tuple[list[dict[str, Any]], str | None]:
    """Released arms (lock order) and manifest sha256 of a round-dataset session.

    Empty for a repo outside ``release/round-datasets.json``.
    """
    if not dataset.released:
        return [], None
    entry = _release_eval_index()[1].get(dataset.repo_id)
    if entry is None:
        return [], None
    arms = [
        p for p in entry["policies"] if p["session_id"] == dataset.session and not p.get("exclude")
    ]
    sha = next(
        s["manifest_sha256"] for s in entry["sessions"] if s["session_id"] == dataset.session
    )
    return arms, sha


def _parse_arm(value: str) -> tuple[str, str | None]:
    name, _, prefix = value.partition("=")
    return name, prefix or None


def config_from_release(
    repo: str,
    out_dir: Path,
    *,
    session: str | None = None,
    revision: str | None = None,
    arms: list[tuple[str, str | None]] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    manifest_path: Path | None = None,
    bootstrap_seed: int = 0,
    outcome_record: bool = True,
) -> HeldoutEvalConfig:
    """Binary-success config for one evaluation session.

    Arms default to the session's released policies (``release/round-datasets.json``),
    or to every arm of ``results.json`` for an unreleased repo; the first arm is the
    baseline of the default comparisons. The manifest defaults to the session's copy in
    the dataset (``meta/initial_states_manifest.json``), checked against the lock's
    sha256 for released data. The outcome-editor record is applied when present and
    ``outcome_record`` is set; turn it off for a ``results.json`` the record was already
    applied to with frame-level truncations (Nut R5, as in the paper's ingest).
    """
    dataset = resolve_eval_dataset(repo, revision=revision, session=session)
    payload = fetch_results_payload(dataset)
    spec = find_task_spec_by_task_name(payload["args"]["environment"])
    if spec is None:
        raise KeyError(f"{repo}: no real task registered for {payload['args']['environment']!r}")
    lock_arms, lock_manifest_sha256 = _lock_session(dataset)
    released = {p["name"]: p for p in lock_arms}
    if arms is None:
        names = list(released) or [row["name"] for row in payload["summary"]]
        arms = [(name, None) for name in names]
    names = [name for name, _ in arms]
    methods = [released.get(name, {}).get("method") for name in names]
    unique_methods = all(methods) and len(set(methods)) == len(methods)
    prefixes, labels, colors = {}, {}, {}
    for i, ((name, prefix), method) in enumerate(zip(arms, methods, strict=True)):
        if unique_methods:
            default_prefix, color_key = _METHOD_STYLE.get(method, (f"arm{i}", None))
            labels[name] = method
        else:
            default_prefix, color_key = f"arm{i}", None
            labels[name] = name
        prefixes[name] = prefix or default_prefix
        colors[name] = METHOD_COLORS[color_key or _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]]

    if manifest_path is None:
        manifest_path = Path(
            hf_hub_download(
                dataset.repo_id,
                dataset.path("meta/initial_states_manifest.json"),
                repo_type="dataset",
                revision=dataset.revision,
                force_download=not dataset.released,
            )
        )
    n_states = len(json.loads(manifest_path.read_text())["states"])
    record = None
    if outcome_record:
        record = load_outcome_edit_record(
            dataset.repo_id,
            dataset.path(DEFAULT_OUTCOME_OVERRIDES),
            revision=dataset.revision,
            required=False,
            force_download=not dataset.released,
        )
    if record is None:
        print(
            f"[heldout_eval] {repo}: outcomes as in results.json (no {DEFAULT_OUTCOME_OVERRIDES})"
        )
    where = f"{dataset.repo_id}@{dataset.revision[:12]}"
    if dataset.session:
        where += f" session {dataset.session}"
    return HeldoutEvalConfig(
        task=spec.name,
        eval_repo=dataset.repo_id,
        eval_revision=dataset.revision,
        eval_session=dataset.session if dataset.session_dir else None,
        manifest_path=manifest_path,
        expected_manifest_sha256=lock_manifest_sha256,
        expected_total_rounds=n_states,
        policy_names=tuple(names),
        policy_labels=labels,
        policy_colors=colors,
        policy_prefixes=prefixes,
        comparison_pairs=tuple(pairs) if pairs else None,
        data_dir=out_dir,
        plot_path=out_dir / "heldout_eval.svg",
        plot_title=f"{spec.name} held-out eval: {where}",
        plot_caption_prefix=f"{where}; paired by manifest_idx",
        bootstrap_seed=bootstrap_seed,
        expect_pure_dp=False,
        outcome_overrides_filename=DEFAULT_OUTCOME_OVERRIDES if record is not None else None,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Held-out paired eval ingest of one session: CSVs, summary.json and a figure.",
        epilog=(
            "Example: python -m mulligan.real.lifecycle.heldout_eval "
            "mulligan/real-square-d2-r02-eval --out outputs/real/heldout/square_d2_r02"
        ),
    )
    ap.add_argument(
        "repo", help="released mulligan/* evaluation dataset, its source id, or your eval repo"
    )
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--session", help="session (bNN) of a round dataset that merges several")
    ap.add_argument("--revision", help="default: the release pin, or main for an unreleased repo")
    ap.add_argument(
        "--arm",
        action="append",
        metavar="NAME[=PREFIX]",
        help="arm as named in results.json, first = baseline (repeatable); "
        "default: the session's released arms",
    )
    ap.add_argument(
        "--pair",
        action="append",
        metavar="A:B",
        help="comparison A minus B (repeatable); default: every arm against the first",
    )
    ap.add_argument("--manifest", type=Path, help="default: the dataset's manifest copy")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument(
        "--no-outcome-record",
        action="store_true",
        help=f"use results.json as is, without {DEFAULT_OUTCOME_OVERRIDES} "
        "(Nut R5: the record is already applied)",
    )
    args = ap.parse_args(argv)
    cfg = config_from_release(
        args.repo,
        args.out,
        session=args.session,
        revision=args.revision,
        arms=[_parse_arm(a) for a in args.arm] if args.arm else None,
        pairs=[tuple(p.split(":", 1)) for p in args.pair] if args.pair else None,
        manifest_path=args.manifest,
        bootstrap_seed=args.bootstrap_seed,
        outcome_record=not args.no_outcome_record,
    )
    summary = run(cfg)
    for row in summary["policy_summary"]:
        print(
            f"{row['policy_name']}: {row['successes']}/{row['episodes']} "
            f"(Wilson {row['wilson_lo']:.3f}-{row['wilson_hi']:.3f})"
        )
    for row in summary["pairwise_summary"]:
        print(
            f"{row['a']} - {row['b']}: {row['paired_delta_a_minus_b']:+.3f} "
            f"[{row['paired_delta_bootstrap_ci95_lo']:+.3f}, "
            f"{row['paired_delta_bootstrap_ci95_hi']:+.3f}], "
            f"McNemar p={row['mcnemar_exact_pvalue']:.4f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
