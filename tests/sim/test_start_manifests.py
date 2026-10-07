"""The locked sim round inputs under data/sim/start_manifests match INDEX.csv."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
START_DIR = REPO_ROOT / "data/sim/start_manifests"
INDEX = START_DIR / "INDEX.csv"
UNINDEXED = {"INDEX.csv", "README.md"}
TASKS = {"square_narrow", "square_broad"}
ROLES = {
    "arm_starts",
    "blind_manifest",
    "collection_manifest",
    "collection_starts",
    "collection_starts_qpos",
    "coverage_reference",
    "eval_points",
    "points_manifest",
    "rollout_audit",
    "rollout_input",
    "rollout_input_qpos",
    "rollout_manifest",
    "sampler_candidate",
}
# The per-round autonomous-baseline start lists. The autonomous-baseline protocol pinned the
# sha256 of these files before their names were changed to the release names; the pins here are
# the sha256 of the files as released and the sha256 of their `states`, which the renaming did
# not touch (the value the protocol's pin covered).
AUTONOMOUS_BASELINE_PINS = {
    "square_narrow/r01/init_states/square_narrow_baseline_uniform_r1.json": (
        "981402807c63dd50b61bae7fc6c6da4d2060af3e26f4c48545e143db3109887a",
        "3e7b11ee340f02ab2e764faa819ecd8a399a188cdea9a7992a8373b363f04e63",
    ),
    "square_narrow/r02/init_states/square_narrow_baseline_uniform_r2.json": (
        "c833db028871725d1658f5b3c964694429c9c167b340ff509feae8f81cddf857",
        "5b7781768b5a78de7787730afecbd630495b1bc072a0bdad5c3376319f33606c",
    ),
    "square_narrow/r03/init_states/square_narrow_baseline_uniform_r3.json": (
        "f14b32ebfd0fa0a82fcd4dbab08b73e872a75855cbc27f143a5a4dac4978dfb0",
        "e1d2eea534465ddb977ec5b9b947ff813f72da2ad904da3c37125cf0775b31d7",
    ),
    "square_broad/r01/init_states/square_broad_baseline_uniform_r1.json": (
        "2c24475eebef75fd84835acdb273fa57409dd1f2a7eb220fb9600dca78a0cd2d",
        "62456fe01a9cf07cb3f84ccbc3db5720deaff868f78691cda38c917288b52c9f",
    ),
    "square_broad/r02/init_states/square_broad_baseline_uniform_r2.json": (
        "12ed465b0edccda01ad7b7676be3103dee0a5fd635e4de0ba397874c0de11d2b",
        "dfe6424efd699b6ef4a6df94352cced87adcfe7e6c310f91ac6e8614a2f8c170",
    ),
    "square_broad/r03/init_states/square_broad_baseline_uniform_r3.json": (
        "79fa88c419d4267e93dbeda489d39d8f935813c2a1008b9fd90732745d690f70",
        "c6ecb149db469be5a1433bd48632cbee2ded241906a915d8a74c5a6d49e90820",
    ),
}


def _rows() -> list[dict[str, str]]:
    with INDEX.open(newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ROWS = _rows()


def test_index_columns_and_values():
    assert {
        "task",
        "round",
        "path",
        "sha256",
        "states_sha256",
        "bytes",
        "role",
    } <= set(ROWS[0])
    paths = [row["path"] for row in ROWS]
    assert len(paths) == len(set(paths))
    for row in ROWS:
        assert row["task"] in TASKS
        assert row["role"] in ROLES
        assert 0 <= int(row["round"]) <= 3
        assert row["path"].startswith(
            f"data/sim/start_manifests/{row['task']}/r{int(row['round']):02d}/"
        )


@pytest.mark.parametrize("row", ROWS, ids=[row["path"].split("/", 3)[-1] for row in ROWS])
def test_indexed_file_matches(row: dict[str, str]):
    path = REPO_ROOT / row["path"]
    assert path.is_file(), path
    assert path.stat().st_size == int(row["bytes"])
    assert _sha256(path) == row["sha256"]


def test_no_unindexed_files():
    indexed = {row["path"] for row in ROWS}
    on_disk = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in START_DIR.rglob("*")
        if path.is_file() and path.relative_to(START_DIR).as_posix() not in UNINDEXED
    }
    assert on_disk == indexed


def _states_sha256(path: Path) -> str:
    states = json.loads(path.read_text())["states"]
    return hashlib.sha256(
        json.dumps(states, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_autonomous_baseline_pins():
    for rel, (digest, states_digest) in AUTONOMOUS_BASELINE_PINS.items():
        assert _sha256(START_DIR / rel) == digest, rel
        assert _states_sha256(START_DIR / rel) == states_digest, rel


@pytest.mark.parametrize("row", ROWS, ids=[row["path"].split("/", 3)[-1] for row in ROWS])
def test_states_sha256_matches(row: dict[str, str]):
    """``states_sha256`` identifies the start states independently of the names around them."""
    path = REPO_ROOT / row["path"]
    if path.suffix == ".npy":
        assert row["states_sha256"] == row["sha256"]
    elif "states" in json.loads(path.read_text()):
        assert row["states_sha256"] == _states_sha256(path)
    else:
        assert row["states_sha256"] == ""
