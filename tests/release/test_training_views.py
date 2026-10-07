"""release/training-views.json: every real checkpoint's data selectors reproduce the audited counts."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from mulligan.real.train.dataset_selectors import EpisodeSelector, resolve_episode_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWS = json.loads((REPO_ROOT / "release" / "training-views.json").read_text())
CHECKPOINTS = VIEWS["checkpoints"]
REAL_ROBOT_DOC = REPO_ROOT / "docs" / "real_robot.md"


def _ids(checkpoint: dict) -> str:
    return checkpoint["repo"] + (f"/{checkpoint['subfolder']}" if checkpoint["subfolder"] else "")


def test_covers_every_released_real_checkpoint():
    kinds = Counter(c["kind"] for c in CHECKPOINTS)
    assert kinds == {"dp-actor": 58, "idql-critic": 18}
    assert len({_ids(c) for c in CHECKPOINTS}) == len(CHECKPOINTS)
    assert VIEWS["problems"] == []


def test_round_dataset_lock_is_the_shipped_one():
    lock = (REPO_ROOT / "release" / "round-datasets.json").read_bytes()
    assert VIEWS["round_datasets_sha256"] == hashlib.sha256(lock).hexdigest()
    assert set(VIEWS) == {
        "schema_version",
        "rule",
        "round_datasets_sha256",
        "summary",
        "checkpoints",
        "problems",
    }


def test_round_dataset_sessions_name_the_session_their_repo_publishes():
    """A source read from a whole-session repo names the round dataset and session it equals."""
    lock = json.loads((REPO_ROOT / "release" / "round-datasets.json").read_text())
    sessions = {
        (d["id"], s["session_id"]): s["release_repo"]
        for d in lock["datasets"]
        for s in d["sessions"]
    }
    pins = json.loads((REPO_ROOT / "release" / "revisions.json").read_text())["repos"]
    found = []
    for checkpoint in CHECKPOINTS:
        for source in checkpoint["sources"]:
            session = source.get("round_dataset_session")
            if session is None:
                continue
            found.append(checkpoint["repo"])
            assert source["selector"] == "all"
            assert session["revision"] == pins[session["repo"]]["revision"]
            assert sessions[(session["repo"], session["session_id"])] == source["repo"]
            assert checkpoint["notes"], "a whole-session source must say why"
    assert len(found) == 5


def test_summary_matches_entries():
    assert VIEWS["summary"]["checkpoints"] == len(CHECKPOINTS)
    assert VIEWS["summary"]["by_class"] == dict(Counter(c["audit_class"] for c in CHECKPOINTS))
    assert VIEWS["summary"]["by_retrain"] == dict(Counter(c["retrain"] for c in CHECKPOINTS))


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=_ids)
def test_sources_are_pinned_public_selectors(checkpoint):
    assert checkpoint["sources"], "checkpoint without training sources"
    assert {s["role"] for s in checkpoint["sources"]} <= {"training", "validation"}
    for source in checkpoint["sources"]:
        assert set(source) - {"round_dataset_session"} == {
            "role",
            "repo",
            "revision",
            "selector",
            "expected_episodes",
            "source_at_run_start",
            "missing_episodes",
        }
        assert source["source_at_run_start"]["episodes"] == source["expected_episodes"]
        assert source["repo"].startswith("mulligan/"), source["repo"]
        assert len(source["revision"]) == 40, source
        assert source["expected_episodes"] > 0
        if source["selector"] != "all":
            EpisodeSelector.from_json(source["selector"])
        if checkpoint["retrain"] == "exact":
            assert source["missing_episodes"] == 0, source


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=_ids)
def test_retrain_flag_follows_audit(checkpoint):
    if checkpoint["audit_class"] != "exact":
        assert checkpoint["retrain"] == "approximate"
    if checkpoint["retrain"] == "approximate":
        assert checkpoint["notes"], "approximate checkpoints must say why"


def test_logged_train_episode_counts_match_public_selectors():
    for checkpoint in CHECKPOINTS:
        check = checkpoint.get("run_config_check") or {}
        if check.get("status") != "ok" or "logged_n_train_episodes" not in check:
            continue
        public = sum(
            s["expected_episodes"] for s in checkpoint["sources"] if s["role"] == "training"
        )
        assert check["public_n_train_episodes"] == public, _ids(checkpoint)
        if checkpoint["retrain"] == "exact":
            assert check["logged_n_train_episodes"] == public, _ids(checkpoint)


def test_approximate_checkpoints_listed_in_real_robot_docs():
    approximate = sorted(_ids(c) for c in CHECKPOINTS if c["retrain"] == "approximate")
    text = REAL_ROBOT_DOC.read_text()
    missing = [repo for repo in approximate if f"`{repo}`" not in text]
    assert not missing, f"docs/real_robot.md must list the approximate checkpoints: {missing}"


@cache
def _info(repo: str, revision: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo,
        "meta/info.json",
        repo_type="dataset",
        revision=revision,
        token=False,
        cache_dir=os.environ.get("MULLIGAN_HF_CACHE"),
    )
    return json.loads(Path(path).read_text())


def _selected_count(source: dict) -> int:
    total = _info(source["repo"], source["revision"])["total_episodes"]
    if source["selector"] == "all":
        return total
    selector = EpisodeSelector.from_json(source["selector"])
    indices = resolve_episode_indices(source["repo"], selector, source["revision"])
    assert all(0 <= i < total for i in indices), source["repo"]
    if "episode_index" in source:
        assert indices == sorted(source["episode_index"]), source["repo"]
    return len(indices)


@pytest.mark.network
@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=_ids)
def test_selectors_reproduce_audited_counts(checkpoint):
    for source in checkpoint["sources"]:
        assert _selected_count(source) == source["expected_episodes"], (
            source["role"],
            source["repo"],
        )
