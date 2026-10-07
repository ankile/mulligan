"""`release/datasets.{json,csv}` and the dataset-side manifests (docs/data.md)."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "release"
LOCK_SHA256 = "a626ad69dbe7ea155ee97f19857441b91b3790510a32e72a6aa0a72ec0eb29c3"
ROLES = {
    "training-view",
    "raw-collection",
    "validation-view",
    "evaluation",
    "policy-rollouts",
    "evaluation-bundle",
}
TASKS = {
    "real-marker-d2",
    "real-square-d2",
    "real-routing-d2",
    "sim-square-narrow",
    "sim-square-broad",
}


def load(name: str):
    return json.loads((RELEASE / name).read_text())


def datasets() -> list[dict]:
    return load("datasets.json")["datasets"]


def canonical() -> dict[str, dict]:
    return load("revisions.json")["repos"]


def test_one_row_per_dataset_pin():
    rows = datasets()
    repos = [r["repo"] for r in rows]
    assert len(repos) == len(set(repos)) == 223
    pins = {k: v for k, v in canonical().items() if v["type"] == "dataset"}
    assert set(repos) == set(pins)
    for r in rows:
        assert (r["revision"], r["tag"]) == (pins[r["repo"]]["revision"], pins[r["repo"]]["tag"])


def test_rows_are_clean_and_typed():
    for r in datasets():
        assert "state" not in r and "declared_license" not in r and "destination_repo" not in r
        assert r["task"] in TASKS, r["repo"]
        assert r["role"] in ROLES, r["repo"]
        assert r["repo"].startswith(f"mulligan/{r['task']}-"), r["repo"]
        assert isinstance(r["episodes"], int) and r["episodes"] > 0, r["repo"]
        if r["role"] == "evaluation-bundle":
            assert r["frames"] is None and r["cameras"] == []
            continue
        assert r["frames"] >= r["episodes"] and r["fps"] in (15, 20), r["repo"]
        assert all(c.startswith("observation.images.") for c in r["cameras"])
        assert r["codebase_version"] == "v3.0"
        assert not any(k.startswith("source_") for k in r), r["repo"]


def test_parents_are_released():
    repos = {r["repo"] for r in datasets()}
    for r in datasets():
        assert r["parent"] is None or r["parent"] in repos, (r["repo"], r["parent"])


def test_round_datasets_match_the_lock():
    lock = load("round-datasets.json")
    assert hashlib.sha256((RELEASE / "round-datasets.json").read_bytes()).hexdigest() == LOCK_SHA256
    rows = {r["repo"]: r for r in datasets() if "round_dataset" in r}
    assert set(rows) == {d["id"] for d in lock["datasets"]}
    kinds = Counter()
    for d in lock["datasets"]:
        rd = rows[d["id"]]["round_dataset"]
        kinds[rd["kind"]] += 1
        assert (rd["kind"], rd["round"]) == (d["kind"], d["round"])
        assert rd["sessions"] == [s["session_id"] for s in d["sessions"]]
        assert rows[d["id"]]["episodes"] == d["counts"]["included"] == rd["counts"]["included"]
    assert kinds == {"round": 12, "screen": 2, "cable": 1}


def test_csv_matches_json():
    with (RELEASE / "datasets.csv").open() as f:
        rows = list(csv.DictReader(f))
    by_repo = {r["repo"]: r for r in datasets()}
    assert [r["repo"] for r in rows] == sorted(by_repo)
    for r in rows:
        j = by_repo[r["repo"]]
        assert r["revision"] == j["revision"] and r["role"] == j["role"]
        assert r["episodes"] == str(j["episodes"])
        assert r["cameras"] == ";".join(j["cameras"])
        assert "source_repo" not in r


def test_paper_result_points():
    parity = load("paper-results.json")
    assert len(parity["points"]) == 110
    assert Counter(p["task"] for p in parity["points"]) == {
        "square_narrow": 32,
        "square_broad": 32,
        "marker_d2": 16,
        "square_d2": 15,
        "routing_d2": 15,
    }
