"""`release/models.{json,csv}`: pins, training-data audit, `retrain` status, provenance fields."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

RELEASE = REPO / "release"
# Trained on session b03 of the Marker R2 eval, published whole as mulligan/real-marker-d2-r02-eval-b03.
B03_CRITICS = {
    "mulligan/real-marker-d2-c05-collector-idql-critic",
    "mulligan/real-marker-d2-r03-mulligan-idql-critic",
    "mulligan/real-marker-d2-r04-mulligan-idql-critic",
    "mulligan/real-marker-d2-r05-mulligan-idql-critic",
    "mulligan/real-marker-d2-r05-mulligan-idql-critic-screen-b02",
}
UNRELEASED_ENCODER_CRITIC = "mulligan/real-routing-d2-velocity-r05-mulligan-idql-critic"


def load(name: str):
    return json.loads((RELEASE / name).read_text())


def models() -> dict:
    return load("models.json")


def checkpoints():
    for m in models()["models"]:
        for c in m["checkpoints"]:
            yield m, c


def test_repos_and_revisions():
    pins = load("revisions.json")["repos"]
    rows = models()["models"]
    assert len(rows) == len({m["repo"] for m in rows}) == 180
    assert {m["repo"] for m in rows} == {k for k, v in pins.items() if v["type"] == "model"}
    for m in rows:
        assert m["revision"] == pins[m["repo"]]["revision"]
        assert m["tag"] == pins[m["repo"]]["tag"]
        assert "tag_pending" not in m


def test_checkpoint_counts():
    kinds = Counter(c["kind"] for _, c in checkpoints())
    assert kinds == {"idql-agent": 290, "divl-agent": 230, "dp-actor": 58, "idql-critic": 18}
    for m, c in checkpoints():
        assert c["files"] and all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) for f in c["files"])
        assert (c["subfolder"] is None) == (c["domain"] == "real"), m["repo"]


def test_retrain_flags_follow_the_training_data_audit():
    views = {(v["repo"], v["subfolder"]): v for v in load("training-views.json")["checkpoints"]}
    real = [(m, c) for m, c in checkpoints() if c["domain"] == "real"]
    assert len(real) == len(views) == 76
    for m, c in real:
        assert c["retrain"] == views[(m["repo"], c["subfolder"])]["retrain"]
    approximate = {m["repo"] for m, c in real if c["retrain"] == "approximate"}
    assert approximate == {UNRELEASED_ENCODER_CRITIC}
    assert all(c["retrain"] == "exact" for m, c in real if m["repo"] in B03_CRITICS)
    for m, c in checkpoints():
        if c["domain"] == "sim":
            assert "retrain" not in c


def test_exact_checkpoint_datasets_equal_training_view_selectors():
    """An exact checkpoint lists exactly the repos its training-views selectors read."""
    views = {(v["repo"], v["subfolder"]): v for v in load("training-views.json")["checkpoints"]}
    for m, c in checkpoints():
        if c.get("retrain") != "exact":
            continue
        view = views[(m["repo"], c["subfolder"])]
        for role in ("training", "validation"):
            sources = [s for s in view["sources"] if s["role"] == role]
            assert sorted(c[f"{role}_datasets"]) == sorted({s["repo"] for s in sources}), m["repo"]
            sessions = sorted(
                (s["repo"], session)
                for s in sources
                if s["selector"] != "all"
                for session in s["selector"].get("session_id", [])
            )
            got = sorted((s["dataset"], s["session"]) for s in c[f"{role}_round_sessions"])
            assert got == sessions, m["repo"]
    for m, c in checkpoints():
        if m["repo"] in B03_CRITICS:
            assert "mulligan/real-marker-d2-r02-eval-b03" in c["training_datasets"]
            assert "mulligan/real-marker-d2-r02-eval" not in c["training_datasets"]
    with (RELEASE / "models.csv").open() as f:
        rows = {r["repo"]: r for r in csv.DictReader(f) if r["repo"] in B03_CRITICS}
    assert len(rows) == len(B03_CRITICS)
    for row in rows.values():
        assert "mulligan/real-marker-d2-r02-eval-b03" in row["training_datasets"].split(";")
        assert "mulligan/real-marker-d2-r02-eval" not in row["training_datasets"].split(";")


def test_not_released_reason_names_the_whole_session_repo():
    (entry,) = [
        n
        for n in models()["not_released"]
        if "mulligan/real-marker-d2-r02-eval" in n["referenced_by"]
    ]
    assert "mulligan/real-marker-d2-r02-eval-b03" in entry["reason"]


def test_critic_dp_artifacts_resolve_to_released_dps():
    by_repo = {m["repo"]: m for m in models()["models"]}
    critics = [(m, c) for m, c in checkpoints() if c["kind"] == "idql-critic"]
    assert len(critics) == 18
    for m, c in critics:
        resolved = c["dp_artifact_resolved"]
        if m["repo"] == UNRELEASED_ENCODER_CRITIC:
            assert resolved is None and c["dp_artifact_note"]
            continue
        dp = by_repo[resolved["repo"]]
        assert resolved["revision"] == dp["revision"]
        assert dp["checkpoints"][0]["kind"] == "dp-actor"
        assert dp["checkpoints"][0]["task"] == c["task"]
    assert models()["summary"]["critics_dp_artifact_resolved"] == "17/18"


def test_dataset_references_are_pinned_and_released():
    pins = load("revisions.json")["repos"]
    released = {r["repo"] for r in load("datasets.json")["datasets"]}
    dr = models()["dataset_revisions"]
    for repo, rev in dr.items():
        assert rev == pins[repo]["revision"]
    for _, c in checkpoints():
        for d in c["training_datasets"] + c["validation_datasets"]:
            assert d in released and d in dr
        for s in c["training_round_sessions"] + c["validation_round_sessions"]:
            assert s["dataset"] in released
        for dep in c.get("deployments", []):
            assert dep["dataset"] in released
            assert dep["actor"]["revision"] == pins[dep["actor"]["repo"]]["revision"]


def test_csv_matches_json():
    with (RELEASE / "models.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 596
    want = {
        (m["repo"], c["subfolder"] or ""): (m["revision"], c["kind"], c.get("retrain") or "")
        for m, c in checkpoints()
    }
    got = {(r["repo"], r["subfolder"]): (r["revision"], r["kind"], r["retrain"]) for r in rows}
    assert got == want


@pytest.mark.network
def test_model_cards_at_the_pins_declare_mit(hf_cache):
    """[network] Each model card at its pin is the README.md in `extra_files` and declares MIT."""
    import hashlib
    import os
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

    def check(m: dict) -> str | None:
        (readme,) = [f for f in m["extra_files"] if f["path"] == "README.md"]
        path = hf_hub_download(
            m["repo"], "README.md", revision=m["revision"], token=False, cache_dir=hf_cache
        )
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != readme["sha256"]:
            return f"{m['repo']}: README.md at the pin differs from models.json"
        front = data.decode().split("---\n")[1]
        if not re.search(r"^license: mit$", front, re.M):
            return f"{m['repo']}: the card does not declare license: mit"
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        problems = [p for p in pool.map(check, models()["models"]) if p]
    assert not problems, problems
