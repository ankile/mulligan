import json

import pytest

from mulligan.sim.collect.quota import ProtocolQuotaLedger


def _write_manifest(path):
    payload = {
        "task": "square_narrow",
        "keys": ["nut_x", "nut_y", "nut_yaw"],
        "match_tolerance": 1e-3,
        "states": [
            {"nut_x": 0.0, "nut_y": 0.0, "nut_yaw": 0.0, "sources": ["baseline"]},
            {"nut_x": 1.0, "nut_y": 0.0, "nut_yaw": 0.0, "sources": ["mulligan"]},
            {
                "nut_x": 2.0,
                "nut_y": 0.0,
                "nut_yaw": 0.0,
                "sources": ["baseline", "mulligan"],
            },
        ],
    }
    path.write_text(json.dumps(payload))


def _write_large_manifest(path):
    states = []
    for i in range(10):
        states.append(
            {
                "nut_x": float(i),
                "nut_y": 0.0,
                "nut_yaw": 0.0,
                "sources": ["baseline"],
            }
        )
    for i in range(10):
        states.append(
            {
                "nut_x": float(100 + i),
                "nut_y": 0.0,
                "nut_yaw": 0.0,
                "sources": ["mulligan"],
            }
        )
    states.append(
        {
            "nut_x": 200.0,
            "nut_y": 0.0,
            "nut_yaw": 0.0,
            "sources": ["baseline", "mulligan"],
        }
    )
    payload = {
        "task": "square_narrow",
        "keys": ["nut_x", "nut_y", "nut_yaw"],
        "match_tolerance": 1e-3,
        "states": states,
    }
    path.write_text(json.dumps(payload))


def _write_three_arm_tail_manifest(path):
    states = []
    for arm, base in [
        ("baseline", 0.0),
        ("arm_a", 100.0),
        ("arm_b", 200.0),
    ]:
        for i in range(100):
            states.append(
                {
                    "nut_x": base + float(i),
                    "nut_y": 0.0,
                    "nut_yaw": 0.0,
                    "sources": [arm],
                }
            )
    payload = {
        "task": "square_narrow",
        "keys": ["nut_x", "nut_y", "nut_yaw"],
        "match_tolerance": 1e-3,
        "states": states,
    }
    path.write_text(json.dumps(payload))


def _write_three_arm_asymmetric_manifest(path):
    states = []
    for i in range(4):
        states.append(
            {
                "nut_x": float(i),
                "nut_y": 0.0,
                "nut_yaw": 0.0,
                "sources": ["baseline"],
            }
        )
    for i in range(3):
        states.append(
            {
                "nut_x": float(100 + i),
                "nut_y": 0.0,
                "nut_yaw": 0.0,
                "sources": ["sobol"],
            }
        )
    for i in range(3):
        states.append(
            {
                "nut_x": float(200 + i),
                "nut_y": 0.0,
                "nut_yaw": 0.0,
                "sources": ["mulligan"],
            }
        )
    states.append(
        {
            "nut_x": 300.0,
            "nut_y": 0.0,
            "nut_yaw": 0.0,
            "sources": ["mulligan", "sobol"],
        }
    )
    payload = {
        "task": "square_narrow",
        "keys": ["nut_x", "nut_y", "nut_yaw"],
        "match_tolerance": 1e-3,
        "states": states,
    }
    path.write_text(json.dumps(payload))


def test_protocol_quota_matches_yaw_across_2pi_wrap(tmp_path):
    """Manifest yaws outside [-pi, pi] must match saved env_state yaws.

    Robosuite stores post-reset yaws wrapped to [-pi, pi]. Manifests that
    sample yaw uniformly on [0, 2*pi] (or any range outside [-pi, pi]) would
    otherwise fail KDTree matching when the splitter checks the first-frame
    env_state against the manifest entry.
    """
    import math

    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    payload = {
        "task": "square_broad",
        "keys": ["nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y"],
        "match_tolerance": 1e-3,
        "states": [
            {
                "nut_x": 0.0,
                "nut_y": 0.0,
                # 5.04 rad = -1.243 rad + 2*pi; robosuite saves the wrapped value.
                "nut_yaw": 5.04,
                "peg_x": 0.1,
                "peg_y": 0.1,
                "sources": ["baseline_uniform"],
            },
            {
                "nut_x": 1.0,
                "nut_y": 0.0,
                "nut_yaw": 0.5,
                "peg_x": 0.1,
                "peg_y": 0.1,
                "sources": ["sobol", "mulligan"],
            },
        ],
    }
    manifest.write_text(json.dumps(payload))

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 1, "with_cf": 1},
        ledger_path=ledger,
        arms_by_protocol={
            "no_cf": ["baseline_uniform", "sobol", "mulligan"],
            "with_cf": ["sobol", "mulligan"],
        },
    )

    wrapped_yaw = 5.04 - 2.0 * math.pi  # ~ -1.2432
    matched_idx, dist = q.match([0.0, 0.0, wrapped_yaw, 0.1, 0.1])
    assert matched_idx == 0
    assert dist < 1e-3

    matched_idx2, _ = q.match([1.0, 0.0, 0.5, 0.1, 0.1])
    assert matched_idx2 == 1


def test_protocol_quota_fresh_credits_both_cf_credits_free_only(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q.match([1.0, 0.0, 0.0])
    shared_idx, _ = q.match([2.0, 0.0, 0.0])

    row1 = q.credit_episode(
        manifest_idx=shared_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    assert row1["credited_protocol_arms"] == {
        "no_cf": ["baseline", "mulligan"],
        "with_cf": ["baseline", "mulligan"],
    }
    assert not q.is_fresh_state_eligible(shared_idx)
    assert q.can_accept_counterfactual(shared_idx)

    row2 = q.credit_episode(
        manifest_idx=shared_idx,
        episode_index=1,
        success=True,
        is_counterfactual=True,
    )
    assert row2["credited_protocol_arms"] == {
        "with_cf": ["baseline", "mulligan"],
    }
    assert q.is_protocol_complete("with_cf")
    assert not q.is_protocol_complete("no_cf")
    assert not q.can_accept_counterfactual(shared_idx)

    row3 = q.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=2,
        success=True,
        is_counterfactual=False,
    )
    row4 = q.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=3,
        success=True,
        is_counterfactual=False,
    )
    assert row3["credited_protocol_arms"] == {"no_cf": ["baseline"]}
    assert row4["credited_protocol_arms"] == {"no_cf": ["mulligan"]}
    assert q.is_complete()

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert rows[-1]["counts_after"] == {
        "no_cf": {"baseline": 2, "mulligan": 2},
        "with_cf": {"baseline": 2, "mulligan": 2},
    }
    progress = "\n".join(
        q.progress_lines(
            saved_episode_count=4,
            total_manifest_remaining=0,
            eligible_manifest_remaining=0,
        )
    )
    assert "With-CF quota: complete" in progress
    assert "No-CF quota: complete" in progress
    assert "Estimated remaining credited episodes: 0" in progress
    assert "ETA to all quotas full: complete" in progress


def test_protocol_quota_can_target_with_cf_to_subset_of_arms(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        arms_by_protocol={"no_cf": ["baseline", "mulligan"], "with_cf": ["mulligan"]},
        ledger_path=ledger,
        balance_slack=1,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q.match([1.0, 0.0, 0.0])
    shared_idx, _ = q.match([2.0, 0.0, 0.0])

    baseline_row = q.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    assert baseline_row["credited_protocol_arms"] == {"no_cf": ["baseline"]}
    assert not q.can_accept_counterfactual(baseline_idx)
    with pytest.raises(ValueError, match="would not credit any protocol quota"):
        q.credit_episode(
            manifest_idx=baseline_idx,
            episode_index=99,
            success=True,
            is_counterfactual=True,
        )

    mulligan_row = q.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=1,
        success=True,
        is_counterfactual=False,
    )
    assert mulligan_row["credited_protocol_arms"] == {
        "no_cf": ["mulligan"],
        "with_cf": ["mulligan"],
    }
    assert q.can_accept_counterfactual(mulligan_idx)

    cf_row = q.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=2,
        success=True,
        is_counterfactual=True,
    )
    assert cf_row["credited_protocol_arms"] == {"with_cf": ["mulligan"]}
    assert q.is_protocol_complete("with_cf")
    assert not q.is_protocol_complete("no_cf")

    shared_row = q.credit_episode(
        manifest_idx=shared_idx,
        episode_index=3,
        success=True,
        is_counterfactual=False,
    )
    assert shared_row["credited_protocol_arms"] == {"no_cf": ["baseline", "mulligan"]}
    assert q.is_complete()
    assert q.remaining() == {
        "no_cf": {"baseline": 0, "mulligan": 0},
        "with_cf": {"mulligan": 0},
    }


def test_protocol_quota_resume_rejects_cf_row_crediting_no_cf(tmp_path):
    """CORE-INTEGRITY: a counterfactual row that credits no_cf must fail loudly on
    resume — it would otherwise consume a no_cf slot and route a CF replay into a
    no_cf actor repo, contaminating the Ours-vs-baseline comparison. The live
    crediting path never produces such a row; this guards against a poisoned or
    foreign ledger."""
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    poisoned = {
        "episode_index": 0,
        "success": True,
        "is_counterfactual": True,
        "manifest_idx": 1,
        "credited_protocol_arms": {"no_cf": ["mulligan"]},
    }
    ledger.write_text(json.dumps(poisoned) + "\n")

    with pytest.raises(ValueError, match="CF replays must never"):
        ProtocolQuotaLedger(
            manifest_path=manifest,
            targets_by_protocol={"no_cf": 2, "with_cf": 2},
            arms_by_protocol={"no_cf": ["baseline", "mulligan"], "with_cf": ["mulligan"]},
            ledger_path=ledger,
        )


def test_protocol_quota_asymmetric_three_arm_sampling_with_half_cf(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_three_arm_asymmetric_manifest(manifest)

    arms_by_protocol = {
        "no_cf": ["baseline", "sobol", "mulligan"],
        "with_cf": ["sobol", "mulligan"],
    }
    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 4, "with_cf": 4},
        arms_by_protocol=arms_by_protocol,
        ledger_path=ledger,
        balance_slack=1,
    )
    planned = [(state["nut_x"], state["nut_y"], state["nut_yaw"]) for state in q.states]
    next_idx = 0
    episode_idx = 0
    cf_requests = 0

    while not q.is_complete():
        best_score = -1
        best_idx = None
        for idx, candidate in enumerate(planned):
            manifest_idx, _ = q.match(candidate)
            if not q.is_fresh_state_eligible(manifest_idx):
                continue
            score = q.state_score(manifest_idx)
            if score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None

        planned[next_idx], planned[best_idx] = planned[best_idx], planned[next_idx]
        manifest_idx, _ = q.match(planned[next_idx])
        next_idx += 1
        sources = q.sources_for(manifest_idx)
        fresh_row = q.credit_episode(
            manifest_idx=manifest_idx,
            episode_index=episode_idx,
            success=True,
            is_counterfactual=False,
        )
        episode_idx += 1

        no_cf_values = list(q.counts["no_cf"].values())
        assert max(no_cf_values) - min(no_cf_values) <= 1
        if sources == ["baseline"]:
            assert fresh_row["credited_protocol_arms"] == {"no_cf": ["baseline"]}
            assert not q.can_accept_counterfactual(manifest_idx)
            continue

        if cf_requests % 2 == 0 and q.can_accept_counterfactual(manifest_idx):
            cf_row = q.credit_episode(
                manifest_idx=manifest_idx,
                episode_index=episode_idx,
                success=True,
                is_counterfactual=True,
            )
            episode_idx += 1
            assert set(cf_row["credited_protocol_arms"]) == {"with_cf"}
        cf_requests += 1

    assert q.counts == {
        "no_cf": {"baseline": 4, "mulligan": 4, "sobol": 4},
        "with_cf": {"mulligan": 4, "sobol": 4},
    }
    assert episode_idx > len(planned)


def test_protocol_quota_cf_gate_uses_with_cf_balance_not_raw_capacity(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
        balance_slack=1,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q.match([1.0, 0.0, 0.0])
    shared_idx, _ = q.match([2.0, 0.0, 0.0])

    q.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    assert q.counts["with_cf"] == {"baseline": 1, "mulligan": 0}

    # CF availability follows balanced with-CF depletion, not merely local
    # residual source capacity.
    assert not q.can_accept_counterfactual(baseline_idx)
    assert q.can_accept_counterfactual(mulligan_idx)
    assert q.can_accept_counterfactual(shared_idx)

    with pytest.raises(ValueError, match="would not credit any protocol quota"):
        q.credit_episode(
            manifest_idx=baseline_idx,
            episode_index=1,
            success=True,
            is_counterfactual=True,
        )

    row = q.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=1,
        success=True,
        is_counterfactual=False,
    )
    assert row["credited_protocol_arms"] == {
        "no_cf": ["mulligan"],
        "with_cf": ["mulligan"],
    }
    assert q.counts["with_cf"] == {"baseline": 1, "mulligan": 1}

    row = q.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=2,
        success=True,
        is_counterfactual=True,
    )
    assert row["credited_protocol_arms"] == {"with_cf": ["mulligan"]}
    assert q.counts["with_cf"] == {"baseline": 1, "mulligan": 2}


def test_protocol_quota_fresh_gate_keeps_no_cf_synchronized(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
        balance_slack=1,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q.match([1.0, 0.0, 0.0])

    q.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    assert q.counts["no_cf"] == {"baseline": 1, "mulligan": 0}
    assert q.counts["with_cf"] == {"baseline": 1, "mulligan": 0}

    # Another baseline fresh rollout would make both protocols 2 vs 0.
    assert not q.is_fresh_state_eligible(baseline_idx)
    # The lagging arm can still receive a normal rollout.
    assert q.is_fresh_state_eligible(mulligan_idx)


def test_protocol_quota_soft_weighted_does_not_gate_on_count_spread(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    hard = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        balance_slack=1,
    )
    mulligan_idx, _ = hard.match([100.0, 0.0, 0.0])
    hard.counts["no_cf"]["baseline"] = 0
    hard.counts["no_cf"]["mulligan"] = 2
    hard.counts["with_cf"]["baseline"] = 0
    hard.counts["with_cf"]["mulligan"] = 2
    assert not hard.is_fresh_state_eligible(mulligan_idx)

    soft = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=tmp_path / "soft_ledger.jsonl",
        balance_slack=1,
        selection_mode="soft_weighted",
        softmax_beta=1.0,
        sampling_seed=123,
    )
    soft.counts["no_cf"]["baseline"] = 0
    soft.counts["no_cf"]["mulligan"] = 2
    soft.counts["with_cf"]["baseline"] = 0
    soft.counts["with_cf"]["mulligan"] = 2
    assert soft.is_fresh_state_eligible(mulligan_idx)


def test_protocol_quota_soft_weighted_probabilities_follow_remaining_queue(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        selection_mode="soft_weighted",
        softmax_beta=0.75,
        sampling_seed=123,
    )
    q.counts["no_cf"]["baseline"] = 0
    q.counts["no_cf"]["mulligan"] = 5
    q.counts["with_cf"]["baseline"] = 0
    q.counts["with_cf"]["mulligan"] = 5

    probs = q.soft_weighted_fresh_arm_probabilities()
    assert set(probs) == {"baseline", "mulligan"}
    assert probs["baseline"] > probs["mulligan"] > 0.0
    assert sum(probs.values()) == pytest.approx(1.0)


def test_protocol_quota_soft_weighted_preserves_per_arm_queue_order(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        selection_mode="soft_weighted",
        softmax_beta=1.0,
        sampling_seed=123,
    )
    q.counts["no_cf"]["mulligan"] = 10
    q.counts["with_cf"]["mulligan"] = 10

    assert q.select_fresh_manifest_idx() == 0
    q.credit_episode(
        manifest_idx=0,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    assert q.select_fresh_manifest_idx() == 1


def test_protocol_quota_soft_weighted_selection_resumes_reproducibly(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q1 = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        selection_mode="soft_weighted",
        softmax_beta=1.0,
        sampling_seed=123,
    )
    first = q1.select_fresh_manifest_idx()
    assert first is not None
    q1.credit_episode(
        manifest_idx=first,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )

    q2 = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        selection_mode="soft_weighted",
        softmax_beta=1.0,
        sampling_seed=123,
    )
    assert q1.select_fresh_manifest_idx() == q2.select_fresh_manifest_idx()


def test_protocol_quota_resume_retries_latest_uncredited_fresh_row(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
        selection_mode="soft_weighted",
        softmax_beta=1.0,
        sampling_seed=123,
    )
    mulligan_idx, _ = q.match([1.0, 0.0, 0.0])
    failed_row = q.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=0,
        success=False,
        is_counterfactual=False,
    )
    assert failed_row["credited_protocol_arms"] == {}
    assert q.pending_retry_fresh_manifest_idx == mulligan_idx
    assert q.select_fresh_manifest_idx() == mulligan_idx

    resumed = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
        selection_mode="soft_weighted",
        softmax_beta=1.0,
        sampling_seed=123,
    )
    assert resumed.select_fresh_manifest_idx() == mulligan_idx

    success_row = q.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=1,
        success=True,
        is_counterfactual=False,
    )
    assert success_row["credited_protocol_arms"] == {
        "no_cf": ["mulligan"],
        "with_cf": ["mulligan"],
    }
    assert q.pending_retry_fresh_manifest_idx is None
    assert q.select_fresh_manifest_idx() != mulligan_idx


def test_protocol_quota_fresh_credit_does_not_drop_collected_episode_when_imbalanced(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        balance_slack=1,
    )
    mulligan_idx, _ = q.match([100.0, 0.0, 0.0])

    q.counts["no_cf"]["baseline"] = 0
    q.counts["no_cf"]["mulligan"] = 2
    q.counts["with_cf"]["baseline"] = 0
    q.counts["with_cf"]["mulligan"] = 2
    assert not q.is_fresh_state_eligible(mulligan_idx)

    row = q.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    assert row["credited_protocol_arms"]["no_cf"] == ["mulligan"]
    assert q.counts["no_cf"]["mulligan"] == 3


def test_protocol_quota_fresh_success_still_credits_no_cf_when_selection_unbalanced(
    tmp_path,
):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        balance_slack=2,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])

    q.counts["no_cf"].update({"baseline": 7, "mulligan": 5})
    q.counts["with_cf"].update({"baseline": 7, "mulligan": 7})

    assert not q.is_fresh_state_eligible(baseline_idx)
    row = q.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    assert row["credited_protocol_arms"] == {
        "no_cf": ["baseline"],
        "with_cf": ["baseline"],
    }


def test_protocol_quota_fresh_gate_recovers_after_cf_imbalance(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        balance_slack=2,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q.match([100.0, 0.0, 0.0])
    shared_idx, _ = q.match([200.0, 0.0, 0.0])

    q.counts["no_cf"].update({"baseline": 3, "mulligan": 3})
    q.counts["with_cf"].update({"baseline": 3, "mulligan": 6})

    # A CF-heavy run can leave With-CF outside the nominal fresh-start slack.
    # Fresh starts for the lagging arm must remain eligible so the run can
    # recover instead of exhausting with quotas still remaining.
    assert q.is_fresh_state_eligible(baseline_idx)
    assert q.is_fresh_state_eligible(mulligan_idx)
    assert q.is_fresh_state_eligible(shared_idx)
    assert q.state_score(baseline_idx) > q.state_score(mulligan_idx)

    row = q.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    assert row["credited_protocol_arms"] == {
        "no_cf": ["baseline"],
        "with_cf": ["baseline"],
    }
    assert q.counts["with_cf"] == {"baseline": 4, "mulligan": 6}


def test_protocol_quota_cf_gate_targets_lagging_with_cf_arm(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        balance_slack=2,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q.match([100.0, 0.0, 0.0])

    q.counts["no_cf"].update({"baseline": 3, "mulligan": 3})
    q.counts["with_cf"].update({"baseline": 3, "mulligan": 6})

    assert q.can_accept_counterfactual(baseline_idx)
    assert not q.can_accept_counterfactual(mulligan_idx)

    with pytest.raises(ValueError, match="would not credit any protocol quota"):
        q.credit_episode(
            manifest_idx=mulligan_idx,
            episode_index=0,
            success=True,
            is_counterfactual=True,
        )

    row = q.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=1,
        success=True,
        is_counterfactual=True,
    )
    assert row["credited_protocol_arms"] == {"with_cf": ["baseline"]}
    assert q.counts["with_cf"] == {"baseline": 4, "mulligan": 6}


def test_protocol_quota_cf_gate_blocks_new_with_cf_imbalance(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        balance_slack=1,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q.match([100.0, 0.0, 0.0])

    q.counts["no_cf"].update({"baseline": 4, "mulligan": 4})
    q.counts["with_cf"].update({"baseline": 4, "mulligan": 5})

    assert q.can_accept_counterfactual(baseline_idx)
    assert not q.can_accept_counterfactual(mulligan_idx)


def test_protocol_quota_state_score_prioritizes_with_cf_balance_first(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 10, "with_cf": 10},
        ledger_path=ledger,
        balance_slack=2,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q.match([100.0, 0.0, 0.0])

    q.counts["no_cf"].update({"baseline": 5, "mulligan": 5})
    q.counts["with_cf"].update({"baseline": 4, "mulligan": 6})

    assert q.is_fresh_state_eligible(baseline_idx)
    assert q.is_fresh_state_eligible(mulligan_idx)
    assert q.state_score(baseline_idx) > q.state_score(mulligan_idx)


def test_protocol_quota_tail_does_not_require_same_arm_for_no_cf_and_with_cf(
    tmp_path,
):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_three_arm_tail_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 100, "with_cf": 100},
        ledger_path=ledger,
        balance_slack=2,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    arm_a_idx, _ = q.match([100.0, 0.0, 0.0])
    arm_b_idx, _ = q.match([200.0, 0.0, 0.0])

    q.counts["no_cf"].update({"baseline": 69, "arm_a": 67, "arm_b": 68})
    q.counts["with_cf"].update({"baseline": 92, "arm_a": 94, "arm_b": 94})

    # This mirrors the R2 tail failure: With-CF wants baseline, while no-CF
    # balance wants arm_a/arm_b. Fresh selection must follow no-CF, because a
    # fresh success consumes a no-CF start and can still credit with-CF later.
    assert not q.is_fresh_state_eligible(baseline_idx)
    assert q.is_fresh_state_eligible(arm_a_idx)
    assert q.is_fresh_state_eligible(arm_b_idx)


def test_protocol_quota_progress_marks_normal_only_after_with_cf_complete(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
    )
    shared_idx, _ = q.match([2.0, 0.0, 0.0])
    q.credit_episode(
        manifest_idx=shared_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    q.credit_episode(
        manifest_idx=shared_idx,
        episode_index=1,
        success=True,
        is_counterfactual=True,
    )

    assert q.is_protocol_complete("with_cf")
    assert not q.is_complete()
    assert not q.can_accept_counterfactual(shared_idx)
    progress = "\n".join(q.progress_lines(saved_episode_count=2))
    assert "With-CF quota: complete" in progress
    assert "No-CF quota: 2 / 4 credits" in progress
    assert "Mode: normal rollouts only" in progress


def test_protocol_quota_resume_preserves_counts_and_fresh_consumed(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q1 = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
    )
    shared_idx, _ = q1.match([2.0, 0.0, 0.0])
    q1.credit_episode(
        manifest_idx=shared_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    q1.credit_episode(
        manifest_idx=shared_idx,
        episode_index=1,
        success=True,
        is_counterfactual=True,
    )

    q2 = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
    )
    assert q2.counts["no_cf"] == {"baseline": 1, "mulligan": 1}
    assert q2.counts["with_cf"] == {"baseline": 2, "mulligan": 2}
    assert q2.last_successful_manifest_idx == shared_idx
    assert q2.last_successful_fresh_manifest_idx == shared_idx
    assert shared_idx in q2.fresh_consumed_manifest_idxs
    assert not q2.is_fresh_state_eligible(shared_idx)


def test_protocol_quota_failed_fresh_attempt_does_not_consume_state(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q1 = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 1, "with_cf": 1},
        ledger_path=ledger,
    )
    shared_idx, _ = q1.match([2.0, 0.0, 0.0])
    row = q1.credit_episode(
        manifest_idx=shared_idx,
        episode_index=0,
        success=False,
        is_counterfactual=False,
    )
    assert row["credited_protocol_arms"] == {}
    assert shared_idx not in q1.fresh_consumed_manifest_idxs
    assert q1.is_fresh_state_eligible(shared_idx)

    q2 = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 1, "with_cf": 1},
        ledger_path=ledger,
    )
    assert shared_idx not in q2.fresh_consumed_manifest_idxs
    assert q2.is_fresh_state_eligible(shared_idx)


def test_protocol_quota_resume_cf_prompt_uses_last_successful_row(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_large_manifest(manifest)

    q1 = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 5, "with_cf": 5},
        ledger_path=ledger,
    )
    baseline_idx, _ = q1.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q1.match([100.0, 0.0, 0.0])

    q1.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )
    q1.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=1,
        success=True,
        is_counterfactual=False,
    )
    q1.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=2,
        success=True,
        is_counterfactual=True,
    )

    q2 = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 5, "with_cf": 5},
        ledger_path=ledger,
    )
    assert q2.last_successful_manifest_idx == mulligan_idx
    assert q2.last_successful_fresh_manifest_idx == mulligan_idx
    assert q2.resume_counterfactual_manifest_idx() is None


def test_protocol_quota_rejects_duplicate_successful_fresh_rows(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q1 = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
    )
    baseline_idx, _ = q1.match([0.0, 0.0, 0.0])
    q1.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )

    row = {
        "episode_index": 1,
        "success": True,
        "is_counterfactual": False,
        "manifest_idx": baseline_idx,
        "manifest_sources": q1.sources_for(baseline_idx),
        "credited_protocol_arms": {"no_cf": ["baseline"]},
        "counts_after": q1.counts,
        "remaining_after": q1.remaining(),
        "matched_distance": 0.0,
        "manifest_hash": q1.manifest_hash,
    }
    q1.append_reserved_row(row)

    with pytest.raises(ValueError, match="already saved as a credited fresh episode"):
        ProtocolQuotaLedger(
            manifest_path=manifest,
            targets_by_protocol={"no_cf": 2, "with_cf": 2},
            ledger_path=ledger,
        )


def test_protocol_quota_rejects_manifest_hash_mismatch_on_resume(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
    )
    idx, _ = q.match([0.0, 0.0, 0.0])
    q.credit_episode(
        manifest_idx=idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )

    payload = json.loads(manifest.read_text())
    payload["states"][0]["nut_x"] = 0.5
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="manifest_hash mismatch"):
        ProtocolQuotaLedger(
            manifest_path=manifest,
            targets_by_protocol={"no_cf": 2, "with_cf": 2},
            ledger_path=ledger,
        )


def test_protocol_quota_rejects_manifest_source_mismatch_on_resume(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
    )
    idx, _ = q.match([2.0, 0.0, 0.0])
    row = q.credit_episode(
        manifest_idx=idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
        write_ledger=False,
    )
    row.pop("manifest_hash")
    row["manifest_sources"] = ["baseline"]
    q.append_reserved_row(row)

    with pytest.raises(ValueError, match="manifest_sources"):
        ProtocolQuotaLedger(
            manifest_path=manifest,
            targets_by_protocol={"no_cf": 2, "with_cf": 2},
            ledger_path=ledger,
        )


def test_protocol_quota_rejects_in_session_duplicate_fresh_success(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 2, "with_cf": 2},
        ledger_path=ledger,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    q.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )

    with pytest.raises(ValueError, match="already credited as a fresh"):
        q.credit_episode(
            manifest_idx=baseline_idx,
            episode_index=1,
            success=True,
            is_counterfactual=False,
        )


def test_protocol_quota_rejects_success_without_quota_credit(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 1, "with_cf": 1},
        ledger_path=ledger,
    )
    baseline_idx, _ = q.match([0.0, 0.0, 0.0])
    mulligan_idx, _ = q.match([1.0, 0.0, 0.0])
    q.credit_episode(
        manifest_idx=baseline_idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
    )

    with pytest.raises(ValueError, match="would not credit any protocol quota"):
        q.credit_episode(
            manifest_idx=baseline_idx,
            episode_index=1,
            success=True,
            is_counterfactual=True,
        )

    q.credit_episode(
        manifest_idx=mulligan_idx,
        episode_index=2,
        success=True,
        is_counterfactual=False,
    )
    assert q.is_complete()


def test_protocol_quota_can_credit_saved_failure_when_requested(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 1, "with_cf": 1},
        ledger_path=ledger,
    )
    idx, _ = q.match([0.0, 0.0, 0.0])
    row = q.credit_episode(
        manifest_idx=idx,
        episode_index=0,
        success=False,
        is_counterfactual=False,
        quota_credit=True,
    )

    assert row["success"] is False
    assert row["quota_credit"] is True
    assert row["credited_protocol_arms"] == {
        "no_cf": ["baseline"],
        "with_cf": ["baseline"],
    }
    assert q.counts["no_cf"]["baseline"] == 1
    assert idx in q.fresh_consumed_manifest_idxs

    resumed = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 1, "with_cf": 1},
        ledger_path=ledger,
    )
    assert resumed.counts["no_cf"]["baseline"] == 1
    assert idx in resumed.fresh_consumed_manifest_idxs


def test_protocol_quota_resume_rejects_noncontiguous_episode_index(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)
    row = {
        "episode_index": 1,
        "success": True,
        "is_counterfactual": False,
        "manifest_idx": 0,
        "manifest_sources": ["baseline"],
        "credited_protocol_arms": {"no_cf": ["baseline"], "with_cf": ["baseline"]},
    }
    ledger.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="episode_index values must be contiguous"):
        ProtocolQuotaLedger(
            manifest_path=manifest,
            targets_by_protocol={"no_cf": 1, "with_cf": 1},
            ledger_path=ledger,
        )


def test_protocol_quota_rejects_insufficient_manifest_capacity(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    with pytest.raises(ValueError, match="no_cf target requires"):
        ProtocolQuotaLedger(
            manifest_path=manifest,
            targets_by_protocol={"no_cf": 3, "with_cf": 3},
            ledger_path=ledger,
        )


def test_protocol_quota_can_delay_ledger_write_until_save_completes(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_manifest(manifest)

    q = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 1, "with_cf": 1},
        ledger_path=ledger,
    )
    idx, _ = q.match([2.0, 0.0, 0.0])
    row = q.credit_episode(
        manifest_idx=idx,
        episode_index=0,
        success=True,
        is_counterfactual=False,
        write_ledger=False,
    )

    assert q.counts["no_cf"] == {"baseline": 1, "mulligan": 1}
    assert q.counts["with_cf"] == {"baseline": 1, "mulligan": 1}
    assert q.is_complete()
    assert not ledger.exists()

    q.append_reserved_row(row)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert rows == [row]


def _write_capped_manifest(path, cap=None):
    """5 baseline-only + 5 mulligan-only starts; optionally a no_cf cap on 'mulligan'."""
    states = [
        {"nut_x": float(i), "nut_y": 0.0, "nut_yaw": 0.0, "sources": ["baseline"]} for i in range(5)
    ] + [
        {"nut_x": float(10 + i), "nut_y": 0.0, "nut_yaw": 0.0, "sources": ["mulligan"]}
        for i in range(5)
    ]
    payload = {
        "task": "square_narrow",
        "keys": ["nut_x", "nut_y", "nut_yaw"],
        "match_tolerance": 1e-3,
        "states": states,
    }
    if cap is not None:
        payload["protocol_arm_targets"] = {"no_cf": {"mulligan": cap}}
    path.write_text(json.dumps(payload))


def _capped_ledger(manifest, ledger):
    return ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 5, "with_cf": 5},
        ledger_path=ledger,
        arms_by_protocol={"no_cf": ["baseline", "mulligan"], "with_cf": ["mulligan"]},
        selection_mode="soft_weighted",
        softmax_beta=1.0,
        sampling_seed=7,
    )


def test_protocol_quota_arm_cap_resumes_and_keeps_interleave(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_capped_manifest(manifest)
    q = _capped_ledger(manifest, ledger)
    # halfway: 2 baseline fresh successes, 2 mulligan fresh successes (credit both), 1 CF replay
    for ep, x in enumerate([0.0, 10.0, 1.0, 11.0]):
        idx, _ = q.match([x, 0.0, 0.0])
        q.credit_episode(manifest_idx=idx, episode_index=ep, success=True, is_counterfactual=False)
    idx, _ = q.match([11.0, 0.0, 0.0])
    q.credit_episode(manifest_idx=idx, episode_index=4, success=True, is_counterfactual=True)
    assert q.counts["no_cf"]["mulligan"] == 2 and q.counts["with_cf"]["mulligan"] == 3

    # the ledgered amendment: cap mulligan's no_cf at what was collected (2); with_cf still 5
    _write_capped_manifest(manifest, cap=2)
    r = _capped_ledger(manifest, ledger)  # resume must load the old rows unchanged
    assert r.protocol_arm_targets == {"no_cf": {"mulligan": 2}}
    assert r.remaining() == {"no_cf": {"baseline": 3, "mulligan": 0}, "with_cf": {"mulligan": 2}}
    assert not r.is_complete()
    # both arms stay drawable: baseline owes no_cf, mulligan owes with_cf (union selection)
    probs = r.soft_weighted_fresh_arm_probabilities()
    assert set(probs) == {"baseline", "mulligan"}
    # a fresh mulligan success now credits with_cf only; a baseline success credits no_cf
    m_idx, _ = r.match([12.0, 0.0, 0.0])
    assert r.is_fresh_state_eligible(m_idx)
    row = r.credit_episode(
        manifest_idx=m_idx, episode_index=5, success=True, is_counterfactual=False
    )
    assert row["credited_protocol_arms"] == {"with_cf": ["mulligan"]}
    b_idx, _ = r.match([2.0, 0.0, 0.0])
    row = r.credit_episode(
        manifest_idx=b_idx, episode_index=6, success=True, is_counterfactual=False
    )
    assert row["credited_protocol_arms"] == {"no_cf": ["baseline"]}
    # completion = baseline no_cf 5 + mulligan with_cf 5, with mulligan no_cf frozen at 2
    for ep, x in enumerate([3.0, 4.0], start=7):
        idx, _ = r.match([x, 0.0, 0.0])
        r.credit_episode(manifest_idx=idx, episode_index=ep, success=True, is_counterfactual=False)
    idx, _ = r.match([13.0, 0.0, 0.0])
    r.credit_episode(manifest_idx=idx, episode_index=9, success=True, is_counterfactual=False)
    assert r.is_complete()
    assert r.counts["no_cf"]["mulligan"] == 2
    assert r.select_fresh_manifest_idx() is None


def test_protocol_quota_arm_cap_rejects_raising_and_unknown_arms(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_capped_manifest(manifest, cap=9)
    with pytest.raises(ValueError, match="may only lower"):
        _capped_ledger(manifest, ledger)
    payload = json.loads(manifest.read_text())
    payload["protocol_arm_targets"] = {"with_cf": {"baseline": 1}}
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not targeted by that protocol"):
        _capped_ledger(manifest, ledger)


def test_protocol_quota_arm_cap_refuses_ledger_above_cap(tmp_path):
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_capped_manifest(manifest)
    q = _capped_ledger(manifest, ledger)
    for ep, x in enumerate([10.0, 11.0, 12.0]):
        idx, _ = q.match([x, 0.0, 0.0])
        q.credit_episode(manifest_idx=idx, episode_index=ep, success=True, is_counterfactual=False)
    _write_capped_manifest(manifest, cap=2)  # below the 3 already credited
    with pytest.raises(ValueError, match="exceeds targets"):
        _capped_ledger(manifest, ledger)


def test_protocol_quota_cli_cap_collected_resolves_from_ledger(tmp_path):
    from mulligan.sim.collect.quota import parse_arm_target_caps

    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_capped_manifest(manifest)
    q = _capped_ledger(manifest, ledger)
    for ep, x in enumerate([0.0, 10.0, 11.0]):
        idx, _ = q.match([x, 0.0, 0.0])
        q.credit_episode(manifest_idx=idx, episode_index=ep, success=True, is_counterfactual=False)
    caps = parse_arm_target_caps("no_cf.mulligan=collected")
    assert caps == {"no_cf": {"mulligan": "collected"}}
    resumed = ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 5, "with_cf": 5},
        ledger_path=ledger,
        arms_by_protocol={"no_cf": ["baseline", "mulligan"], "with_cf": ["mulligan"]},
        selection_mode="soft_weighted",
        softmax_beta=1.0,
        sampling_seed=7,
        arm_target_caps=caps,
    )
    assert resumed.protocol_arm_targets == {"no_cf": {"mulligan": 2}}  # frozen at the collected 2
    assert resumed.remaining()["no_cf"] == {"baseline": 4, "mulligan": 0}
    assert set(resumed.soft_weighted_fresh_arm_probabilities()) == {"baseline", "mulligan"}
    # a fresh dataset (no ledger yet) cannot freeze at 'collected'
    with pytest.raises(ValueError, match="needs an existing ledger"):
        ProtocolQuotaLedger(
            manifest_path=manifest,
            targets_by_protocol={"no_cf": 5, "with_cf": 5},
            ledger_path=tmp_path / "missing.jsonl",
            arms_by_protocol={"no_cf": ["baseline", "mulligan"], "with_cf": ["mulligan"]},
            arm_target_caps=caps,
        ).target_for("no_cf", "mulligan")
    with pytest.raises(ValueError, match="PROTOCOL.ARM"):
        parse_arm_target_caps("no_cf=collected")


def test_protocol_quota_cap_end_to_end_resume_completes_with_interleave(tmp_path):
    """Half a session under the uncapped protocol (fresh retries + CF replays after successes),
    restart with the CLI cap, run to completion through the collector-shaped loop."""
    import random

    from mulligan.sim.collect.quota import parse_arm_target_caps

    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    states = [
        {"nut_x": float(i), "nut_y": 0.0, "nut_yaw": 0.0, "sources": ["baseline"]}
        for i in range(20)
    ] + [
        {"nut_x": float(100 + i), "nut_y": 0.0, "nut_yaw": 0.0, "sources": ["mulligan"]}
        for i in range(20)
    ]
    manifest.write_text(
        json.dumps(
            {
                "task": "square_narrow",
                "keys": ["nut_x", "nut_y", "nut_yaw"],
                "match_tolerance": 1e-3,
                "states": states,
            }
        )
    )
    targets = {"no_cf": 20, "with_cf": 20}
    arms = {"no_cf": ["baseline", "mulligan"], "with_cf": ["mulligan"]}

    def make(caps=None):
        return ProtocolQuotaLedger(
            manifest_path=manifest,
            targets_by_protocol=targets,
            ledger_path=ledger,
            arms_by_protocol=arms,
            selection_mode="soft_weighted",
            softmax_beta=1.0,
            sampling_seed=11,
            arm_target_caps=caps,
        )

    def drive(q, rng, ep, stop_after=None, cf_rate=0.5):
        cf_target = None
        seq = []
        while not q.is_complete() and (stop_after is None or ep < stop_after):
            if cf_target is not None:
                idx, is_cf, cf_target = cf_target, True, None
            else:
                idx, is_cf = q.select_fresh_manifest_idx(), False
                assert idx is not None, q.remaining()
            arm = q.sources_for(idx)[0]
            ok = True if is_cf else rng.random() < 0.8
            row = q.credit_episode(
                manifest_idx=idx, episode_index=ep, success=ok, is_counterfactual=is_cf
            )
            seq.append((arm, is_cf, ok, row["credited_protocol_arms"]))
            if (
                ok
                and not is_cf
                and arm == "mulligan"
                and q.can_accept_counterfactual(idx)
                and rng.random() < cf_rate
            ):
                cf_target = idx
            ep += 1
        return ep, seq

    rng = random.Random(3)
    q1 = make()
    ep, _ = drive(q1, rng, 0, stop_after=25)
    frozen = q1.counts["no_cf"]["mulligan"]
    assert 0 < frozen < 20 and not q1.is_complete()
    q2 = make(parse_arm_target_caps("no_cf.mulligan=collected"))
    assert q2.protocol_arm_targets == {"no_cf": {"mulligan": frozen}}
    ep, seq2 = drive(q2, rng, ep)
    assert q2.is_complete()
    assert q2.counts["no_cf"]["baseline"] == 20
    assert q2.counts["with_cf"]["mulligan"] == 20
    assert q2.counts["no_cf"]["mulligan"] == frozen
    fresh_arms = {arm for arm, is_cf, _, _ in seq2 if not is_cf}
    assert fresh_arms == {"baseline", "mulligan"}  # interleave survives the cap
    for arm, is_cf, ok, credited in seq2:
        if arm == "mulligan" and not is_cf and ok:
            assert credited == {"with_cf": ["mulligan"]}
    assert make(parse_arm_target_caps("no_cf.mulligan=collected")).is_complete()
