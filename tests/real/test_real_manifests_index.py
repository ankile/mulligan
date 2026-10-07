"""The locked real start manifests in ``data/real/manifests`` match their INDEX.csv pins."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mulligan.real.lifecycle.tasks import get_task_spec

ROOT = Path(__file__).resolve().parents[2] / "data" / "real" / "manifests"
ROWS = list(csv.DictReader((ROOT / "INDEX.csv").open()))


def test_index_lists_every_manifest_once():
    listed = sorted(row["path"] for row in ROWS)
    on_disk = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*.json"))
    assert listed == on_disk
    assert len(ROWS) == 53
    # The Cable 15-arm lineage eval manifest (all six released Cable rounds) is included.
    assert any(
        row["path"].startswith("routing_d2/lineage/") and row["public_round"] == "0-5"
        for row in ROWS
    )


@pytest.mark.parametrize("row", ROWS, ids=[row["path"] for row in ROWS])
def test_manifest_sha256_and_task(row):
    path = ROOT / row["path"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert len(raw) == int(row["bytes"])
    payload = json.loads(raw)
    assert payload.get("task", row["task"]) == row["task"]
    assert get_task_spec(row["task"]).name == row["task"]
    if row["round"]:
        assert path.parent.name == f"r{int(row['round']):02d}"


def test_routing_d2_lineage_manifest_matches_pinned_sha():
    from mulligan.real.lifecycle import routing_d2_lineage

    path = routing_d2_lineage.MANIFEST_PATH
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == routing_d2_lineage.MANIFEST_SHA256


def test_sampler_configs_pin_the_shipped_locked_manifests():
    configs = ROOT.parents[2] / "configs" / "real"
    locked = {}
    for path in sorted(configs.glob("*/r*_sampler.yaml")):
        raw = yaml.safe_load(path.read_text())
        pin = raw["locked"]["manifest"]
        rel = Path(pin["path"]).relative_to("data/real/manifests")
        locked[(raw["task"], int(raw["round"]))] = (str(rel), pin["sha256"])
    assert sorted(locked) == [(t, r) for t in ("marker_d2", "square_d2") for r in range(1, 6)]
    pins = {row["path"]: row["sha256"] for row in ROWS}
    for rel, sha in locked.values():
        assert pins[rel] == sha
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha


def test_sobol_registry_claims_every_shipped_manifest_point():
    """data/real/sobol_stream_ranges.csv claims every Sobol point a shipped manifest uses."""
    from collections import defaultdict

    from mulligan.real.eval.eval_manifest import _parse_ranges

    claimed: dict[int, set[int]] = defaultdict(set)
    for row in csv.DictReader((ROOT.parent / "sobol_stream_ranges.csv").open()):
        claimed[int(row["sobol_seed"])] |= _parse_ranges(row["ranges"])
    unclaimed = {}
    for path in sorted(ROOT.rglob("*.json")):
        states = json.loads(path.read_text()).get("states", [])
        # R0 manifests name the seed sampler_seed.
        points = [
            (int(s.get("sobol_seed", s.get("sampler_seed"))), int(s["sobol_stream_index"]))
            for s in states
            if "sobol_stream_index" in s
        ]
        missing = [(seed, idx) for seed, idx in points if idx not in claimed[seed]]
        if missing:
            unclaimed[str(path.relative_to(ROOT))] = missing[:5]
    assert not unclaimed
