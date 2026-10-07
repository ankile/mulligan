"""Held-out ingest on the released evaluation datasets.

``mulligan.real.lifecycle.heldout_eval`` reads a released ``mulligan/*`` evaluation dataset
at its pin in ``release/revisions.json``, maps the source repo ids that the sessions'
``results.json`` record to the released dataset that holds them, and reads a session of a
merged round dataset from ``meta/sessions/<id>/``. Offline tests use the release manifests
of this checkout and stub every Hub read; the network test runs the CLI on one released
dataset at its pin and checks the paper's Nut R2 numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mulligan.plotting.colors import METHOD_COLORS
from mulligan.real.eval.outcome_results import FrameOutcome
from mulligan.real.lifecycle import heldout_eval
from mulligan.real.lifecycle.heldout_eval import (
    EvalDataset,
    HeldoutEvalConfig,
    config_from_release,
    load_results,
    resolve_eval_dataset,
)
from mulligan.real.lifecycle.tasks import get_task_spec
from mulligan.release.download import pinned_revision, select_datasets
from mulligan.release.round_counts import load_lock

EVALS = {row["repo"]: row for row in select_datasets(role=["evaluation"])}
LOCK = {row["id"]: row for row in load_lock()[0]["datasets"]}
SINGLE = "mulligan/real-square-d2-r02-eval"  # one session, files at the repo root
MERGED = "mulligan/real-marker-d2-r02-eval"  # sessions b01-b03 under meta/sessions/
OWN_B03 = "mulligan/real-marker-d2-r02-eval-b03"  # MERGED's b03, also released on its own


def _session(repo: str, session_id: str) -> dict:
    return next(s for s in LOCK[repo]["sessions"] if s["session_id"] == session_id)


def _released_arms(repo: str, session_id: str) -> list[dict]:
    return [
        p for p in LOCK[repo]["policies"] if p["session_id"] == session_id and not p.get("exclude")
    ]


def test_layout_assumptions():
    assert LOCK[SINGLE]["build"] == "move-and-sidecars"
    assert LOCK[MERGED]["build"] == "rebuild"
    assert OWN_B03 not in LOCK
    assert _session(MERGED, "b03")["release_repo"] == OWN_B03


def test_released_dataset_reads_at_its_pin():
    dataset = resolve_eval_dataset(SINGLE)
    assert dataset == EvalDataset(
        SINGLE,
        pinned_revision(SINGLE),
        (SINGLE, _session(SINGLE, "b01")["source_repo"]),
        session="b01",
        session_dir="",
        released=True,
    )
    assert dataset.path("results.json") == "results.json"
    assert resolve_eval_dataset(SINGLE, session="b01") == dataset
    with pytest.raises(ValueError, match="holds sessions"):
        resolve_eval_dataset(SINGLE, session="b02")


def test_explicit_revision_overrides_the_pin(capsys):
    dataset = resolve_eval_dataset(SINGLE, revision="0" * 40)
    assert dataset.revision == "0" * 40
    assert "not at its release pin" in capsys.readouterr().out


def test_merged_round_dataset_needs_a_session():
    with pytest.raises(ValueError, match="merges sessions"):
        resolve_eval_dataset(MERGED)
    dataset = resolve_eval_dataset(MERGED, session="b02")
    assert dataset.revision == pinned_revision(MERGED)
    assert dataset.session_dir == "meta/sessions/b02/"
    assert dataset.path(".outcome_edit_progress.json") == (
        "meta/sessions/b02/.outcome_edit_progress.json"
    )
    assert _session(MERGED, "b02")["source_repo"] in dataset.recorded_repo_ids


def test_every_session_records_a_released_repo():
    """results.json of every session records a released repo id (rehomed source ids)."""
    for entry in LOCK.values():
        for session in entry["sessions"]:
            assert session["source_repo"] in EVALS, (entry["id"], session["session_id"])
            sid = session["session_id"] if entry["build"] == "rebuild" else None
            dataset = resolve_eval_dataset(entry["id"], session=sid)
            assert session["source_repo"] in dataset.recorded_repo_ids


def test_unreleased_repo_reads_hub_main():
    dataset = resolve_eval_dataset("someone/my-eval")
    assert dataset == EvalDataset("someone/my-eval", "main", ("someone/my-eval",))
    assert resolve_eval_dataset("someone/my-eval", revision="abc1234").revision == "abc1234"
    with pytest.raises(ValueError, match="released round datasets only"):
        resolve_eval_dataset("someone/my-eval", session="b01")


def test_unknown_mulligan_repo_is_an_error():
    with pytest.raises(KeyError, match="not a released evaluation dataset"):
        resolve_eval_dataset("mulligan/real-square-d2-r02-eval-typo")
    with pytest.raises(KeyError, match="not a released evaluation dataset"):
        resolve_eval_dataset("mulligan/real-square-d2-r02-baseline-dp")


# load_results on a merged session, every Hub read stubbed


def _merged_payload(source: str, arms: list[str]) -> dict:
    summary = [
        {"name": name, "policy_id": i, "num_rounds": 1, "successes": 1, "failures": 0}
        for i, name in enumerate(arms)
    ]
    rollouts = [
        {"round": 1, "policy_id": i, "episode_index": i, "outcome": "success", "num_steps": 5}
        for i in range(len(arms))
    ]
    return {
        "args": {
            "environment": get_task_spec("marker_d2").task_name,
            "hf_repo_id": source,
            "num_action_samples": None,
            "initial_states_manifest": "manifest.json",
        },
        "round_plans": [{}],
        "summary": summary,
        "rollouts": rollouts,
    }


def _cfg(tmp_path: Path, eval_repo: str, names: tuple[str, ...], **kwargs) -> HeldoutEvalConfig:
    if eval_repo == MERGED:
        kwargs.setdefault("eval_session", "b02")
    return HeldoutEvalConfig(
        task="marker_d2",
        eval_repo=eval_repo,
        manifest_path=tmp_path / "manifest.json",
        expected_total_rounds=1,
        policy_names=names,
        policy_labels={n: n for n in names},
        policy_colors={n: METHOD_COLORS["uniform"] for n in names},
        policy_prefixes={n: f"p{i}" for i, n in enumerate(names)},
        data_dir=tmp_path / "out",
        plot_path=tmp_path / "out" / "plot.svg",
        plot_title="t",
        plot_caption_prefix="c",
        bootstrap_seed=1,
        outcome_overrides_filename=".outcome_edit_progress.json",
        **kwargs,
    )


@pytest.fixture
def merged_hub(monkeypatch):
    """Stub the Hub for MERGED session b02: arms a, b released (dataset episodes 10, 11), c not."""
    source = _session(MERGED, "b02")["source_repo"]
    calls: dict[str, list] = {"json": [], "record": [], "frames": []}
    state = {"payload": _merged_payload(source, ["a", "b", "c"])}

    def load_hf_json(repo_id, path, *, revision, force_download):
        calls["json"].append((repo_id, path, revision, force_download))
        return json.loads(json.dumps(state["payload"]))

    def load_record(repo_id, path, *, revision, required, force_download):
        calls["record"].append((repo_id, path, revision, force_download))
        return {"changed_episodes": {}}

    def load_frames(repo_id, *, revision, subtask_frames_by_episode, force_download, episodes):
        calls["frames"].append((repo_id, revision, set(episodes)))
        return {10: FrameOutcome("success", 5), 11: FrameOutcome("success", 5)}

    monkeypatch.setattr(heldout_eval, "load_hf_json", load_hf_json)
    monkeypatch.setattr(heldout_eval, "load_outcome_edit_record", load_record)
    monkeypatch.setattr(heldout_eval, "load_frame_outcomes_from_hf", load_frames)
    monkeypatch.setattr(heldout_eval, "_session_episode_map", lambda dataset: {0: 10, 1: 11})
    return source, calls, state


def test_load_results_reads_a_merged_session_at_the_pin(tmp_path, merged_hub, capsys):
    source, calls, _ = merged_hub
    spec = get_task_spec("marker_d2")
    payload = load_results(_cfg(tmp_path, MERGED, ("a", "b")), spec)
    pin = pinned_revision(MERGED)
    assert calls["json"] == [(MERGED, "meta/sessions/b02/results.json", pin, False)]
    assert calls["record"] == [
        (MERGED, "meta/sessions/b02/.outcome_edit_progress.json", pin, False)
    ]
    assert calls["frames"] == [(MERGED, pin, {10, 11})]
    assert payload["_eval_dataset"] == {
        "repo_id": MERGED,
        "revision": pin,
        "session": "b02",
        "released": True,
    }
    out = capsys.readouterr().out
    assert "1 episode(s) of unconfigured arms ['c'] are not in" in out
    # The snapshot keeps the session's full results.json, without the private keys.
    snapshot = json.loads((tmp_path / "out" / "results.json").read_text())
    assert len(snapshot["rollouts"]) == 3 and "_eval_dataset" not in snapshot


def test_load_results_needs_every_configured_episode(tmp_path, merged_hub):
    source, _, _ = merged_hub
    with pytest.raises(RuntimeError, match=r"does not hold episodes \[2\] of the configured"):
        load_results(_cfg(tmp_path, MERGED, ("a", "c")), get_task_spec("marker_d2"))


def test_frame_mismatch_after_the_record_names_the_opt_out(tmp_path, merged_hub, monkeypatch):
    source, _, _ = merged_hub
    monkeypatch.setattr(
        heldout_eval,
        "load_frame_outcomes_from_hf",
        lambda *a, **k: {10: FrameOutcome("success", 5), 11: FrameOutcome("success", 9)},
    )
    with pytest.raises(RuntimeError, match="--no-outcome-record"):
        load_results(_cfg(tmp_path, MERGED, ("a", "b")), get_task_spec("marker_d2"))


def test_load_results_rejects_results_of_another_repo(tmp_path, merged_hub):
    _, _, state = merged_hub
    state["payload"]["args"]["hf_repo_id"] = "someone/other-eval"
    cfg = _cfg(tmp_path, MERGED, ("a", "b"), eval_session="b02")
    with pytest.raises(RuntimeError, match="expected results recorded for"):
        load_results(cfg, get_task_spec("marker_d2"))


def test_unreleased_repo_is_fetched_fresh_from_main(tmp_path, monkeypatch):
    calls = []

    def load_hf_json(repo_id, path, *, revision, force_download):
        calls.append((repo_id, path, revision, force_download))
        return {}

    monkeypatch.setattr(heldout_eval, "load_hf_json", load_hf_json)
    heldout_eval.fetch_results_payload(resolve_eval_dataset("someone/my-eval"))
    assert calls == [("someone/my-eval", "results.json", "main", True)]


# CLI config from the release manifests


@pytest.fixture
def single_hub(monkeypatch, tmp_path):
    session = _session(SINGLE, "b01")
    payload = {
        "args": {"environment": get_task_spec("square_d2").task_name},
        "summary": [{"name": p["name"]} for p in _released_arms(SINGLE, "b01")],
    }
    manifest = tmp_path / "initial_states_manifest.json"
    manifest.write_text(json.dumps({"states": [{}] * session["n_starts"]}))
    downloads = []

    def hf_hub_download(repo_id, filename, *, repo_type, revision, force_download):
        downloads.append((repo_id, filename, revision))
        return str(manifest)

    monkeypatch.setattr(heldout_eval, "fetch_results_payload", lambda dataset: payload)
    monkeypatch.setattr(heldout_eval, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(
        heldout_eval, "load_outcome_edit_record", lambda *a, **k: {"changed_episodes": {}}
    )
    return session, downloads


def test_config_from_release_takes_arms_from_the_round_lock(tmp_path, single_hub):
    session, downloads = single_hub
    cfg = config_from_release(SINGLE, tmp_path / "out")
    arms = _released_arms(SINGLE, "b01")
    assert cfg.task == "square_d2"
    assert cfg.policy_names == tuple(p["name"] for p in arms)
    assert [cfg.policy_labels[p["name"]] for p in arms] == [p["method"] for p in arms]
    by_method = {p["method"]: p["name"] for p in arms}
    assert cfg.policy_prefixes[by_method["HG-DAgger"]] == "baseline"
    assert cfg.policy_prefixes[by_method["HG-DAgger+Mulligan"]] == "mulligan"
    assert (
        cfg.policy_colors[by_method["HG-DAgger+Mulligan"]] == METHOD_COLORS["real_mulligan_with_cf"]
    )
    assert cfg.policy_names[0] == by_method["HG-DAgger"]
    assert cfg.eval_revision == pinned_revision(SINGLE)
    assert cfg.expected_manifest_sha256 == session["manifest_sha256"]
    assert cfg.expected_total_rounds == session["n_starts"]
    assert cfg.outcome_overrides_filename == ".outcome_edit_progress.json"
    assert downloads == [(SINGLE, "meta/initial_states_manifest.json", pinned_revision(SINGLE))]


def test_config_from_release_explicit_arms_and_pairs(tmp_path, single_hub):
    names = [p["name"] for p in _released_arms(SINGLE, "b01")]
    cfg = config_from_release(
        SINGLE,
        tmp_path / "out",
        arms=[(names[2], "mulligan"), (names[0], None)],
        pairs=[(names[2], names[0])],
        bootstrap_seed=7,
    )
    assert cfg.policy_names == (names[2], names[0])
    assert cfg.policy_prefixes[names[2]] == "mulligan"
    assert cfg.resolved_pairs == ((names[2], names[0]),)
    assert cfg.bootstrap_seed == 7
    assert cfg.outcome_overrides_filename == ".outcome_edit_progress.json"
    as_is = config_from_release(SINGLE, tmp_path / "out", outcome_record=False)
    assert as_is.outcome_overrides_filename is None


def test_cli_parses_arms_and_pairs(tmp_path, monkeypatch):
    seen = {}

    def fake_config(repo, out_dir, **kwargs):
        seen.update(repo=repo, out_dir=out_dir, **kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(heldout_eval, "config_from_release", fake_config)
    with pytest.raises(SystemExit):
        heldout_eval.main(
            [SINGLE, "--out", str(tmp_path), "--arm", "x=base", "--arm", "y", "--pair", "y:x"]
        )
    assert seen["arms"] == [("x", "base"), ("y", None)]
    assert seen["pairs"] == [("y", "x")]
    assert seen["session"] is None and seen["revision"] is None


def test_cable_lineage_rounds_run_from_one_pinned_snapshot(monkeypatch):
    from mulligan.real.lifecycle import routing_d2_lineage as lineage

    seen = {}

    def prepare(spec, **kwargs):
        seen["spec"], seen["kwargs"] = spec, kwargs
        return "prepared"

    def run_prepared(cfg, prepared):
        assert prepared == "prepared"
        return {"policy_summary": [], "eval_repo": cfg.eval_repo}

    monkeypatch.setattr(lineage, "prepare_snapshot", prepare)
    monkeypatch.setattr(lineage, "run_heldout_from_prepared", run_prepared)
    out = lineage.run_rounds(validate_frames=False)
    assert list(out) == ["r0", "r1", "r2", "r3", "r4", "r5"]
    assert seen["spec"].repo_id == lineage.EVAL_REPO
    assert seen["spec"].revision == pinned_revision(lineage.EVAL_REPO)
    assert seen["kwargs"]["expected_policy_names"] == lineage.POLICY_ROSTER
    assert seen["kwargs"]["validate_frames"] is False


@pytest.mark.network
def test_nut_r2_heldout_numbers_from_the_released_dataset(tmp_path):
    """Nut (square_d2) R2: the paper's 18/50 vs 27/50, 15 vs 6 discordant, McNemar p=0.0784."""
    arms = {p["method"]: p["name"] for p in _released_arms(SINGLE, "b01")}
    baseline, no_cf, mulligan = (
        arms["HG-DAgger"],
        arms["HG-DAgger+Mulligan no-CF"],
        arms["HG-DAgger+Mulligan"],
    )
    out = tmp_path / "nut_r2"
    heldout_eval.main(
        [
            SINGLE,
            "--out",
            str(out),
            "--arm",
            f"{baseline}=baseline",
            "--arm",
            f"{no_cf}=mulligan_no_cf",
            "--arm",
            f"{mulligan}=mulligan_with_cf",
            "--bootstrap-seed",
            "20260629",
        ]
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["eval_dataset"]["revision"] == pinned_revision(SINGLE)
    assert summary["completed_rounds"] == 50
    successes = {row["policy_name"]: row["successes"] for row in summary["policy_summary"]}
    assert successes == {baseline: 18, no_cf: 20, mulligan: 27}
    pair = next(r for r in summary["pairwise_summary"] if r["a"] == "HG-DAgger+Mulligan")
    assert (pair["a_only_success"], pair["b_only_success"]) == (15, 6)
    assert pair["mcnemar_exact_pvalue"] == pytest.approx(0.0783538818359375, abs=0)
    assert (pair["paired_delta_bootstrap_ci95_lo"], pair["paired_delta_bootstrap_ci95_hi"]) == (
        0.0,
        0.36,
    )
    paired = pd.read_csv(out / "paired_round_outcomes.csv")
    assert list(paired["manifest_idx"]) == list(range(50))
