"""End-to-end routing_d2 held-out eval pipeline: BUILD a fresh Sobol eval manifest and INGEST
paired 2-arm results through the shared ``mulligan.real.lifecycle`` code.

routing_d2 is the first NON-PEN line (1-DOF rope_x + two oriented clips). These tests prove
the registry-driven eval pipeline works for it without a forked implementation:

* ``build_fresh_eval_manifest`` produces a physically-placeable manifest (clips >= 2 in apart via
  rejection sampling), resolves each clip's orientation index to its persisted angle, audits
  disjointness against a prior collection manifest, and the result loads through the real robot
  eval loader.
* ``build_paired_rounds`` / ``build_policy_summary`` / ``build_pairwise_summary`` / ``plot`` ingest
  a synthetic 2-arm results payload (the writer leaves pen_* None on a non-pen line) into paired
  Wilson CIs + exact McNemar + the 4-panel SVG.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import pytest

from mulligan.real.eval.blind_eval_helpers import _load_eval_initial_state_manifest
from mulligan.real.eval.eval_manifest import (
    FreshEvalManifestConfig,
    build_fresh_eval_manifest,
)
from mulligan.real.lifecycle.geometry import TaskGeometry
from mulligan.real.lifecycle.heldout_eval import (
    HeldoutEvalConfig,
    build_paired_rounds,
    build_pairwise_summary,
    build_policy_summary,
    load_manifest,
    plot,
)
from mulligan.real.lifecycle.tasks import get_task_spec

FAKE_EVAL_SEED = 9999123101  # obviously-fake test seed; never a real reserved eval seed
COUNT = 8

_REGISTRY_FIELDS = [
    "usage_id",
    "status",
    "source",
    "sobol_seed",
    "n_unique_stream_indices",
    "min_stream_index",
    "max_stream_index",
    "ranges",
]


def _write_registry(path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_REGISTRY_FIELDS, lineterminator="\n")
        writer.writeheader()
        # One unrelated pre-existing row so the file defines fieldnames for append.
        writer.writerow(
            {
                "usage_id": "unrelated_prior_claim",
                "status": "consumed",
                "source": "other",
                "sobol_seed": "1",
                "n_unique_stream_indices": "1",
                "min_stream_index": "0",
                "max_stream_index": "0",
                "ranges": "0..0",
            }
        )


def _write_prior_collection_manifest(path: Path) -> None:
    """A small routing manifest on a DIFFERENT Sobol seed, standing in for the R0 250-target
    collection manifest the eval block must be disjoint from."""
    spec = get_task_spec("routing_d2")
    tg = TaskGeometry(spec)
    pts, _idx = tg.sobol_feasible(2026062902, 40)
    states = []
    for i, point in enumerate(pts):
        row = {k: float(v) for k, v in zip(spec.sampling_keys, point)}
        spec.resolve_placement_orientations(row)
        row.update({"source": "mulligan_sobol", "sources": ["mulligan_sobol"], "manifest_idx": i})
        states.append(row)
    path.write_text(
        json.dumps(
            {"task": "routing_d2", "schema": "real_manual_initial_states_v1", "states": states}
        )
    )


def _build_eval_manifest(tmp_path: Path, *, seed: int = FAKE_EVAL_SEED) -> tuple[Path, dict]:
    registry = tmp_path / "sobol_stream_ranges.csv"
    prior = tmp_path / "routing_r0_collection.json"
    out = tmp_path / f"routing_d2_eval_heldout_sobol_seed{seed}.json"
    _write_registry(registry)
    _write_prior_collection_manifest(prior)
    cfg = FreshEvalManifestConfig(
        task="routing_d2",
        output_path=out,
        registry_path=registry,
        usage_id=f"routing_d2_r0_heldout_eval_TEST_seed{seed}",
        registry_source="routing_d2_eval_sobol_independent",
        source="eval_sobol_independent",
        sobol_seed=seed,
        count=COUNT,
        start_index=0,
        phase="R0_two_arm_heldout_eval",
        purpose="TEST routing eval manifest",
        disjoint_from="fresh independent Sobol stream audited vs the R0 collection manifest",
        audit_manifests=(prior,),
        audit_path_root=tmp_path,
        order_policy="source_index",
        order_policy_description="independent Sobol stream order (both clips move every episode)",
        placement_name=None,  # two clips: no single serpentine-ordering placement
    )
    manifest = build_fresh_eval_manifest(cfg, apply_registry=True)
    return out, manifest


def test_build_routing_eval_manifest_is_valid_and_loads(tmp_path):
    spec = get_task_spec("routing_d2")
    out, manifest = _build_eval_manifest(tmp_path)

    states = manifest["states"]
    assert len(states) == COUNT
    assert [int(s["manifest_idx"]) for s in states] == list(range(COUNT))
    assert manifest["units"]["clip_left_oidx"] == "index"
    assert manifest["units"]["clip_left_yaw"] == "rad"
    assert manifest["keys"] == list(spec.manifest_keys)

    for s in states:
        # Every required manifest key present, incl. the RESOLVED orientation angle.
        for key in spec.manifest_keys:
            assert key in s
        # Clips are physically placeable (>= 2 in apart) and orientation index/angle agree.
        assert spec.min_placement_separation(s) >= spec.placement_min_separation_m - 1e-9
        for clip in ("clip_left", "clip_right"):
            placement = spec.sampled_placements[clip]
            oidx = int(round(float(s[f"{clip}_oidx"])))
            assert placement.angle_for_index(oidx) == s[f"{clip}_yaw"]
            assert s[f"{clip}_yaw"] in [pytest.approx(a) for a in placement.orient_angles]

    # Disjointness audit ran against the prior collection manifest.
    assert manifest["cross_set_min_periodic_dist_to_prior_manifests"]
    assert all(d > 0 for d in manifest["cross_set_min_periodic_dist_to_prior_manifests"].values())

    # The registry claim was appended with the ACTUAL consumed (possibly non-contiguous) stream.
    registry_rows = list(csv.DictReader((tmp_path / "sobol_stream_ranges.csv").open()))
    claim = next(r for r in registry_rows if r["usage_id"].startswith("routing_d2_r0_heldout"))
    assert int(claim["n_unique_stream_indices"]) == COUNT

    # The manifest loads through the REAL robot eval loader (grid + orientation + separation
    # validation) — this is the same path the eval wrapper's preflight runs on the robot.
    targets, meta = _load_eval_initial_state_manifest(out, expected_task="routing_d2")
    assert len(targets) == COUNT
    assert meta["keys"] == list(spec.manifest_keys)
    assert all(t.pen_x is None and t.pen_y is None and t.pen_yaw is None for t in targets)


def test_rejection_sampling_drops_unplaceable_states_for_bad_seed():
    # Seed 9999123121's first 50 contiguous Sobol draws contain a <2 in clip pair that the robot
    # loader would reject; the feasible sampler MUST drop it (non-contiguous kept stream) while a
    # plain contiguous sobol draw still contains the violation.
    spec = get_task_spec("routing_d2")
    tg = TaskGeometry(spec)
    minsep = spec.placement_min_separation_m

    plain = tg.sobol(9999123121, 50)
    plain_viol = sum(
        spec.min_placement_separation({k: float(v) for k, v in zip(spec.sampling_keys, p)}) < minsep
        for p in plain
    )
    assert plain_viol > 0  # a naive contiguous draw would emit an unloadable state

    pts, idx = tg.sobol_feasible(9999123121, 50)
    assert idx != list(range(50))  # rejection happened -> non-contiguous stream indices
    assert len(pts) == 50
    for p in pts:
        row = {k: float(v) for k, v in zip(spec.sampling_keys, p)}
        assert spec.min_placement_separation(row) >= minsep - 1e-9


def _ingest_cfg(tmp_path: Path, manifest_path: Path) -> HeldoutEvalConfig:
    return HeldoutEvalConfig(
        task="routing_d2",
        eval_repo="org/routing-d2-r0-heldout-TEST",
        manifest_path=manifest_path,
        expected_total_rounds=COUNT,
        policy_names=("baseline_uniform_r0", "mulligan_sobol_r0"),
        policy_labels={"baseline_uniform_r0": "Baseline", "mulligan_sobol_r0": "Ours"},
        policy_colors={"baseline_uniform_r0": "#888888", "mulligan_sobol_r0": "#1f77b4"},
        policy_prefixes={"baseline_uniform_r0": "baseline", "mulligan_sobol_r0": "mulligan"},
        data_dir=tmp_path / "ingest",
        plot_path=tmp_path / "ingest" / "routing_eval.svg",
        plot_title="routing_d2 R0 held-out eval (TEST)",
        plot_caption_prefix="synthetic",
        bootstrap_seed=7,
        n_boot=200,
    )


# baseline vs ours per-round success (COUNT=8 rounds): construct a clean rescue signal.
_BASELINE = [True, False, True, False, False, True, False, False]
_OURS = [True, True, True, True, False, True, True, False]
_ARM_ID = {"baseline_uniform_r0": 0, "mulligan_sobol_r0": 1}


def _synthetic_payload(manifest: dict) -> dict:
    states = {int(s["manifest_idx"]): s for s in manifest["states"]}
    rollouts = []
    for round_id in range(1, COUNT + 1):
        manifest_idx = round_id - 1
        for arm, pattern in (("baseline_uniform_r0", _BASELINE), ("mulligan_sobol_r0", _OURS)):
            rollouts.append(
                {
                    "policy_id": _ARM_ID[arm],
                    "round": round_id,
                    "manifest_idx": manifest_idx,
                    "outcome": "success" if pattern[manifest_idx] else "failure",
                    "num_steps": 120 + manifest_idx,
                    "episode_index": round_id * 10 + _ARM_ID[arm],
                    # Non-pen line: the eval writer records no free pose.
                    "pen_x": None,
                    "pen_y": None,
                    "pen_yaw": None,
                }
            )
            _ = states  # states available if a future writer records rope_x
    summary = []
    for arm, pattern in (("baseline_uniform_r0", _BASELINE), ("mulligan_sobol_r0", _OURS)):
        successes = sum(pattern)
        summary.append(
            {
                "name": arm,
                "policy_id": _ARM_ID[arm],
                "num_rounds": COUNT,
                "successes": successes,
                "failures": COUNT - successes,
                "model_id": f"model://{arm}",
            }
        )
    return {"rollouts": rollouts, "summary": summary}


def test_ingest_routing_paired_two_arm_results(tmp_path):
    spec = get_task_spec("routing_d2")
    manifest_path, manifest = _build_eval_manifest(tmp_path)
    cfg = _ingest_cfg(tmp_path, manifest_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)  # load_results does this in the full pipeline
    loaded = load_manifest(cfg, spec)
    assert set(loaded) == set(range(COUNT))

    payload = _synthetic_payload(manifest)
    paired = build_paired_rounds(cfg, spec, payload, loaded)
    assert len(paired) == COUNT
    # Non-pen line: paired rows carry rope_x + clip columns from the manifest, no pen pose.
    assert "rope_x" in paired.columns
    assert "clip_left_yaw" in paired.columns

    policy = build_policy_summary(cfg, payload, paired)
    assert dict(zip(policy["policy_name"], policy["successes"])) == {
        "baseline_uniform_r0": sum(_BASELINE),
        "mulligan_sobol_r0": sum(_OURS),
    }
    assert set(policy["episodes"]) == {COUNT}

    pairwise = build_pairwise_summary(cfg, paired)
    assert len(pairwise) == 1
    row = pairwise.iloc[0]
    # ours rescues every round baseline failed except the shared fail at idx 4/7 -> discordant.
    assert row["b_only_success"] == 0  # baseline never wins a round ours loses
    assert row["a_only_success"] > 0  # ours rescues baseline failures
    assert 0.0 <= float(row["mcnemar_exact_pvalue"]) <= 1.0

    # The 4-panel SVG renders for a non-pen 2-arm line.
    plot(cfg, policy, paired, pairwise)
    assert cfg.plot_path.exists() and cfg.plot_path.stat().st_size > 0


def test_ingest_rejects_unexpected_pen_pose_on_non_pen_line(tmp_path):
    # Guard: if a future eval writer starts emitting a pen pose for routing, the ingest must fail
    # loudly rather than silently ignore it (the non-pen pose cross-check assumes pen_* stay None).
    spec = get_task_spec("routing_d2")
    manifest_path, manifest = _build_eval_manifest(tmp_path)
    cfg = _ingest_cfg(tmp_path, manifest_path)
    loaded = load_manifest(cfg, spec)
    payload = _synthetic_payload(manifest)
    for rollout in payload["rollouts"]:
        rollout["pen_x"] = 0.1  # a writer that wrongly records a pose
    with pytest.raises(RuntimeError, match="unexpectedly recorded a pen pose"):
        build_paired_rounds(cfg, spec, payload, loaded)


# ---------------------------------------------------------------------------
# Graded 0-2 clip scoring (subtask_scoring=True)
# ---------------------------------------------------------------------------
# routing_d2 seats the rope in two clips in succession (num_subtask_marks=1), so
# each episode grades 0..2: recorded first-clip subtask mark + terminal success.
# Per-round (outcome, first-clip marks) per arm; scores are marks + success.
_BASELINE_GRADED = [
    ("failure", 0),
    ("timeout", 0),
    ("failure", 1),
    ("timeout", 0),
    ("failure", 1),
    ("failure", 0),
    ("timeout", 0),
    ("success", 1),
]
_OURS_GRADED = [
    ("success", 1),
    ("failure", 1),
    ("timeout", 1),
    ("success", 1),
    ("failure", 0),
    ("timeout", 1),
    ("failure", 1),
    ("timeout", 0),
]
_BASELINE_SCORES = [0, 0, 1, 0, 1, 0, 0, 2]
_OURS_SCORES = [2, 1, 1, 2, 0, 1, 1, 0]


def _graded_cfg(tmp_path: Path, manifest_path: Path) -> HeldoutEvalConfig:
    return dataclasses.replace(
        _ingest_cfg(tmp_path, manifest_path),
        outcome_overrides_filename=".outcome_edit_progress.json",
        subtask_scoring=True,
        score_threshold_labels={1: "first clip seated", 2: "both clips (success)"},
    )


def _graded_payload_and_record(manifest: dict) -> tuple[dict, dict]:
    """Synthetic 2-arm graded payload + the matching outcome-editor record."""
    rollouts = []
    changed_episodes: dict[str, dict] = {}
    arms = (("baseline_uniform_r0", _BASELINE_GRADED), ("mulligan_sobol_r0", _OURS_GRADED))
    for round_id in range(1, COUNT + 1):
        idx = round_id - 1
        for arm, pattern in arms:
            outcome, marks = pattern[idx]
            num_steps = 120 + idx
            episode_index = round_id * 10 + _ARM_ID[arm]
            rollouts.append(
                {
                    "policy_id": _ARM_ID[arm],
                    "round": round_id,
                    "manifest_idx": idx,
                    "outcome": outcome,
                    "num_steps": num_steps,
                    "episode_index": episode_index,
                    "pen_x": None,
                    "pen_y": None,
                    "pen_yaw": None,
                }
            )
            # A full subtask review stamps EVERY episode with a subtask_frames
            # key (empty for a 0-mark failure review).
            changed_episodes[str(episode_index)] = {
                "new_outcome": outcome,
                "outcome_frame": num_steps - 1,
                "soft_truncate": True,
                "subtask_frames": [40] if marks == 1 else [],
            }
    summary = []
    for arm, pattern in arms:
        successes = sum(outcome == "success" for outcome, _ in pattern)
        summary.append(
            {
                "name": arm,
                "policy_id": _ARM_ID[arm],
                "num_rounds": COUNT,
                "successes": successes,
                "failures": COUNT - successes,
                "model_id": f"model://{arm}",
            }
        )
    payload = {"rollouts": rollouts, "summary": summary}
    record = {"changed_episodes": changed_episodes, "skipped_episodes": []}
    return payload, record


def _graded_setup(tmp_path: Path):
    from mulligan.real.eval.outcome_results import subtask_reviewed_mark_counts

    spec = get_task_spec("routing_d2")
    manifest_path, manifest = _build_eval_manifest(tmp_path)
    cfg = _graded_cfg(tmp_path, manifest_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_manifest(cfg, spec)
    payload, record = _graded_payload_and_record(manifest)
    payload["_subtask_mark_counts"] = subtask_reviewed_mark_counts(record)
    return spec, cfg, loaded, payload, record


def test_graded_scores_stats_and_plot(tmp_path):
    spec, cfg, loaded, payload, _record = _graded_setup(tmp_path)
    paired = build_paired_rounds(cfg, spec, payload, loaded)
    assert paired["baseline_score"].tolist() == _BASELINE_SCORES
    assert paired["mulligan_score"].tolist() == _OURS_SCORES
    assert paired["baseline_subtask_marks"].tolist() == [m for _, m in _BASELINE_GRADED]

    policy = build_policy_summary(cfg, payload, paired)
    by_name = {row["policy_name"]: row for _, row in policy.iterrows()}
    base, ours = by_name["baseline_uniform_r0"], by_name["mulligan_sobol_r0"]
    assert base["mean_score"] == pytest.approx(sum(_BASELINE_SCORES) / COUNT)
    assert ours["mean_score"] == pytest.approx(sum(_OURS_SCORES) / COUNT)
    assert (base["score_0"], base["score_1"], base["score_2"]) == (5, 2, 1)
    assert (ours["score_0"], ours["score_1"], ours["score_2"]) == (2, 4, 2)
    assert base["ge1_count"] == 3 and base["ge2_count"] == 1
    assert ours["ge1_count"] == 6 and ours["ge2_count"] == 2
    # ge{max} must equal the binary success readout.
    assert base["ge2_rate"] == pytest.approx(base["success_rate"])
    assert base["ge2_wilson_lo"] == pytest.approx(base["wilson_lo"])

    pairwise = build_pairwise_summary(cfg, paired)
    row = pairwise.iloc[0]  # (ours, baseline)
    assert row["mean_score_delta_a_minus_b"] == pytest.approx(1.0 - 0.5)
    assert (row["ge1_a_only"], row["ge1_b_only"]) == (5, 2)
    assert (row["ge2_a_only"], row["ge2_b_only"]) == (2, 1)
    assert row["ge2_a_only"] == row["a_only_success"]  # top threshold == binary columns
    assert row["ge2_mcnemar_exact_pvalue"] == pytest.approx(row["mcnemar_exact_pvalue"])
    assert 0.0 < float(row["mean_score_signflip_pvalue"]) <= 1.0

    plot(cfg, policy, paired, pairwise)
    assert cfg.plot_path.exists() and cfg.plot_path.stat().st_size > 0


def test_graded_zero_success_arms_still_show_first_clip_signal(tmp_path):
    # The actual routing R0 outcome shape: NO full success on either arm, but a
    # handful of first-clip seats -> nonzero graded signal, degenerate binary panel.
    spec, cfg, loaded, payload, record = _graded_setup(tmp_path)
    for rollout in payload["rollouts"]:
        if rollout["outcome"] == "success":
            rollout["outcome"] = "failure"  # keep the first-clip mark: score 2 -> 1
    for row in payload["summary"]:
        row["successes"] = 0
        row["failures"] = COUNT
    paired = build_paired_rounds(cfg, spec, payload, loaded)
    assert paired["baseline_success"].sum() == 0 and paired["mulligan_success"].sum() == 0
    assert paired["baseline_score"].sum() > 0 and paired["mulligan_score"].sum() > 0

    policy = build_policy_summary(cfg, payload, paired)
    assert set(policy["successes"]) == {0}
    assert policy["ge1_count"].tolist() == [3, 6]  # demoted successes keep their mark
    pairwise = build_pairwise_summary(cfg, paired)
    plot(cfg, policy, paired, pairwise)
    assert cfg.plot_path.exists() and cfg.plot_path.stat().st_size > 0


def test_graded_requires_full_subtask_review(tmp_path):
    spec, cfg, loaded, payload, _record = _graded_setup(tmp_path)
    del payload["_subtask_mark_counts"][11]  # one un-reviewed episode
    with pytest.raises(RuntimeError, match="no subtask review"):
        build_paired_rounds(cfg, spec, payload, loaded)
    payload.pop("_subtask_mark_counts")
    with pytest.raises(RuntimeError, match="no reviewed subtask-mark counts"):
        build_paired_rounds(cfg, spec, payload, loaded)


def test_partial_graded_review_uses_complete_reviewed_rounds(tmp_path):
    spec, cfg, loaded, payload, _record = _graded_setup(tmp_path)
    cfg = dataclasses.replace(cfg, allow_partial_subtask_review=True)
    # Keep only the first two paired rounds reviewed.
    payload["_subtask_mark_counts"] = {
        episode: marks
        for episode, marks in payload["_subtask_mark_counts"].items()
        if episode in {10, 11, 20, 21}
    }
    paired = build_paired_rounds(cfg, spec, payload, loaded)
    assert len(paired) == COUNT
    assert paired["baseline_score"].notna().sum() == 2
    assert paired["mulligan_score"].notna().sum() == 2

    policy = build_policy_summary(cfg, payload, paired)
    by_name = {row["policy_name"]: row for _, row in policy.iterrows()}
    assert by_name["baseline_uniform_r0"]["episodes"] == COUNT
    assert by_name["baseline_uniform_r0"]["graded_episodes"] == 2
    assert by_name["mulligan_sobol_r0"]["graded_episodes"] == 2
    assert by_name["mulligan_sobol_r0"]["ge1_count"] == 2

    pairwise = build_pairwise_summary(cfg, paired)
    assert int(pairwise.iloc[0]["graded_paired_n"]) == 2
    plot(cfg, policy, paired, pairwise)
    assert cfg.plot_path.exists() and cfg.plot_path.stat().st_size > 0


def test_partial_graded_review_rejects_cross_arm_partial_round(tmp_path):
    spec, cfg, loaded, payload, _record = _graded_setup(tmp_path)
    cfg = dataclasses.replace(cfg, allow_partial_subtask_review=True)
    # Round 1 has only baseline reviewed; all later rounds are unreviewed.
    payload["_subtask_mark_counts"] = {10: payload["_subtask_mark_counts"][10]}
    with pytest.raises(RuntimeError, match="partial cross-arm subtask review"):
        build_paired_rounds(cfg, spec, payload, loaded)


def test_graded_rejects_illegal_mark_counts(tmp_path):
    spec, cfg, loaded, payload, _record = _graded_setup(tmp_path)
    # ours round 1 (episode 11) is a success and must carry exactly 1 mark.
    payload["_subtask_mark_counts"][11] = 0
    with pytest.raises(RuntimeError, match="SUCCESS episode must carry exactly 1"):
        build_paired_rounds(cfg, spec, payload, loaded)
    payload["_subtask_mark_counts"][11] = 1
    # baseline round 1 (episode 10) is a failure: at most 1 mark.
    payload["_subtask_mark_counts"][10] = 2
    with pytest.raises(RuntimeError, match="FAILURE episode may carry at most 1"):
        build_paired_rounds(cfg, spec, payload, loaded)


def test_graded_config_requires_overrides_record(tmp_path):
    manifest_path, _manifest = _build_eval_manifest(tmp_path)
    with pytest.raises(ValueError, match="subtask_scoring requires outcome_overrides_filename"):
        dataclasses.replace(_ingest_cfg(tmp_path, manifest_path), subtask_scoring=True)


def test_reviewed_mark_counts_distinguish_empty_from_absent():
    from mulligan.real.eval.outcome_results import subtask_reviewed_mark_counts

    record = {
        "changed_episodes": {
            "3": {"new_outcome": "failure", "subtask_frames": []},
            "4": {"new_outcome": "failure", "subtask_frames": [10, 10, 25]},
            "5": {"new_outcome": "failure"},  # pre-subtask record: NOT reviewed
        }
    }
    counts = subtask_reviewed_mark_counts(record)
    assert counts == {3: 0, 4: 2}  # duplicates collapse; absent key excluded
    assert subtask_reviewed_mark_counts(None) == {}


def test_signflip_permutation_pvalue_basics():
    import numpy as np

    from mulligan.real.lifecycle.stats import signflip_permutation_pvalue

    assert signflip_permutation_pvalue(np.zeros(10)) == 1.0
    strong = signflip_permutation_pvalue(np.ones(20), n_perm=2000, seed=3)
    assert strong < 0.01
    balanced = signflip_permutation_pvalue(np.array([1.0, -1.0] * 10), n_perm=2000, seed=3)
    assert balanced > 0.5
    with pytest.raises(ValueError, match="paired-delta array"):
        signflip_permutation_pvalue(np.array([]))


def test_graded_skip_means_zero_scores_skipped_episodes(tmp_path):
    from mulligan.real.lifecycle.heldout_eval import graded_mark_counts

    spec, cfg, loaded, payload, record = _graded_setup(tmp_path)
    # Re-shape the record to the real R0 situation: 0-mark non-success episodes
    # skipped ('n') instead of confirmed. baseline round 1 (ep 10, failure/0)
    # and ours round 8 (ep 81, timeout/0) become explicit skips.
    for ep in (10, 81):
        del record["changed_episodes"][str(ep)]
        record["skipped_episodes"].append(ep)

    # Strict default: a skip is NOT a review.
    strict_counts, strict_skips = graded_mark_counts(cfg, record)
    assert 10 not in strict_counts and strict_skips == []
    payload["_subtask_mark_counts"] = strict_counts
    with pytest.raises(RuntimeError, match="no subtask review"):
        build_paired_rounds(cfg, spec, payload, loaded)

    # Opt-in: the recorded skip is a 0-mark review; scores are unchanged.
    cfg_skip = dataclasses.replace(cfg, subtask_skip_means_zero=True)
    skip_counts, skip_eps = graded_mark_counts(cfg_skip, record)
    assert skip_eps == [10, 81]
    assert skip_counts[10] == 0 and skip_counts[81] == 0
    payload["_subtask_mark_counts"] = skip_counts
    paired = build_paired_rounds(cfg_skip, spec, payload, loaded)
    assert paired["baseline_score"].tolist() == _BASELINE_SCORES
    assert paired["mulligan_score"].tolist() == _OURS_SCORES

    # A skipped SUCCESS episode still fails the mark-count legality gate:
    # ours round 1 (ep 11) is a success and cannot be reviewed via skip.
    del record["changed_episodes"]["11"]
    record["skipped_episodes"].append(11)
    counts_with_bad_skip, _ = graded_mark_counts(cfg_skip, record)
    payload["_subtask_mark_counts"] = counts_with_bad_skip
    with pytest.raises(RuntimeError, match="SUCCESS episode must carry exactly 1"):
        build_paired_rounds(cfg_skip, spec, payload, loaded)
