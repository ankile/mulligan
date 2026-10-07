"""Square samplers, and the round-0 start lists regenerated from them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mulligan.sampling.sobol import (
    SobolSampler,
    SquareD1ListSampler,
    SquareD1SobolSampler,
    SquareListSampler,
    UniformSquareD1Sampler,
    UniformSquareSampler,
    sample_square_sobol2d_uniform_x,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NARROW_R00 = REPO_ROOT / "data/sim/start_manifests/square_narrow/r00"
BROAD_R00 = REPO_ROOT / "data/sim/start_manifests/square_broad/r00"

NARROW_UNIFORM_SEED = 202605231
NARROW_SOBOL_SEED = 42
NARROW_SOBOL_X_SEED = 202605234
NARROW_SOBOL_POOL_SIZE = 1024
BROAD_UNIFORM_SEED = 202605232
BROAD_SOBOL_SEED = 42
BROAD_SOBOL_BATCH_SIZE = 2048


def _broad_state(nut, peg) -> dict[str, float]:
    return {
        "nut_x": float(nut[0]),
        "nut_y": float(nut[1]),
        "nut_yaw": float(nut[2]),
        "peg_x": float(peg[0]),
        "peg_y": float(peg[1]),
    }


def _narrow_uniform(n: int, seed: int) -> list[dict[str, float]]:
    sampler = UniformSquareSampler(seed=seed)
    states = []
    for _ in range(n):
        x, y, yaw = sampler.sample_next()
        states.append({"nut_x": x, "nut_y": y, "nut_yaw": yaw})
        sampler.record_state(x, y, yaw)
    return states


def _broad_stream(sampler, n: int) -> list[dict[str, float]]:
    states = []
    for _ in range(n):
        nut, peg = sampler.sample_next()
        states.append(_broad_state(nut, peg))
        sampler.record_state(nut, peg)
    return states


def _assert_regenerates(locked_path: Path, states: list[dict]) -> None:
    """States must equal the locked list exactly; with the locked ``_meta`` block
    (which carries the generation timestamp) the file must be byte-identical."""
    locked_bytes = locked_path.read_bytes()
    locked = json.loads(locked_bytes)
    assert states == locked["states"]
    payload = {"states": states, "_meta": locked["_meta"]}
    assert json.dumps(payload, indent=2).encode() == locked_bytes


def test_narrow_r00_baseline_uniform_regenerates():
    path = NARROW_R00 / "init_states/square_narrow_baseline_uniform_r0.json"
    meta = json.loads(path.read_text())["_meta"]
    assert meta["seed"] == NARROW_UNIFORM_SEED
    _assert_regenerates(path, _narrow_uniform(meta["n_states"], NARROW_UNIFORM_SEED))


def test_narrow_r00_sobol_regenerates():
    path = NARROW_R00 / "init_states/square_narrow_sobol_r0.json"
    meta = json.loads(path.read_text())["_meta"]
    assert meta["sampler"] == "sobol2d_uniformx"
    assert (meta["seed"], meta["uniform_x_seed"], meta["sobol_pool_size"]) == (
        NARROW_SOBOL_SEED,
        NARROW_SOBOL_X_SEED,
        NARROW_SOBOL_POOL_SIZE,
    )
    states = sample_square_sobol2d_uniform_x(
        meta["n_states"],
        sobol_seed=NARROW_SOBOL_SEED,
        x_seed=NARROW_SOBOL_X_SEED,
        start_index=meta["sobol_start_index"],
        pool_size=NARROW_SOBOL_POOL_SIZE,
    )
    _assert_regenerates(path, states)


def test_broad_r00_baseline_uniform_regenerates():
    path = BROAD_R00 / "init_states/square_broad_baseline_uniform_r0.json"
    meta = json.loads(path.read_text())["_meta"]
    assert meta["seed"] == BROAD_UNIFORM_SEED
    sampler = UniformSquareD1Sampler(seed=BROAD_UNIFORM_SEED)
    _assert_regenerates(path, _broad_stream(sampler, meta["n_states"]))


def test_broad_r00_sobol_regenerates():
    path = BROAD_R00 / "init_states/square_broad_sobol_r0.json"
    meta = json.loads(path.read_text())["_meta"]
    assert (meta["sampler"], meta["seed"]) == ("square_broad_sobol5d", BROAD_SOBOL_SEED)
    sampler = SquareD1SobolSampler(seed=BROAD_SOBOL_SEED, batch_size=BROAD_SOBOL_BATCH_SIZE)
    _assert_regenerates(path, _broad_stream(sampler, meta["n_states"]))


@pytest.mark.parametrize(
    ("r00", "task"),
    [(NARROW_R00, "square_narrow"), (BROAD_R00, "square_broad")],
)
def test_r00_blind_mix_is_the_two_arms(r00: Path, task: str):
    arms = {
        source: json.loads((r00 / f"init_states/{task}_{source}_r0.json").read_text())["states"]
        for source in ("baseline_uniform", "sobol")
    }
    manifest = json.loads((r00 / f"blind_inputs/{task}_r0_mixed_manifest.json").read_text())
    mixed = json.loads((r00 / f"blind_inputs/{task}_r0_mixed.json").read_text())
    keys = manifest["keys"]
    assert len(manifest["states"]) == sum(len(s) for s in arms.values())
    seen = set()
    for idx, (entry, blind) in enumerate(zip(manifest["states"], mixed["states"], strict=True)):
        assert entry["manifest_index"] == idx
        source_state = arms[entry["source"]][entry["source_index"]]
        assert {k: entry[k] for k in keys} == {k: source_state[k] for k in keys} == blind
        seen.add((entry["source"], entry["source_index"]))
    assert len(seen) == len(manifest["states"])


def test_sobol_sampler_boundary_and_budget():
    sampler = SobolSampler(seed=0, batch_size=8)
    corners = {(p[0], p[1]) for p in sampler.planned_points[:4]}
    assert corners == {(x, y) for x in sampler.x_range for y in sampler.y_range}
    for _ in range(8):
        sampler.record_state(*sampler.sample_next())
    with pytest.raises(RuntimeError, match="budget exhausted"):
        sampler.sample_next()


def test_square_broad_sobol_rejects_collisions():
    sampler = SquareD1SobolSampler(seed=0, batch_size=256, min_clearance=0.13263)
    nut = np.array([p[0][:2] for p in sampler.planned_points])
    peg = np.array([p[1] for p in sampler.planned_points])
    assert 0 < len(sampler.planned_points) < 256
    assert np.all(np.linalg.norm(nut - peg, axis=1) > 0.13263)


def test_uniform_samplers_are_seeded():
    a = [UniformSquareSampler(seed=3).sample_next() for _ in range(2)]
    b = [UniformSquareSampler(seed=3).sample_next() for _ in range(2)]
    assert a == b
    assert (
        UniformSquareD1Sampler(seed=3).sample_next() == UniformSquareD1Sampler(seed=3).sample_next()
    )


def test_sobol2d_uniform_x_slices_one_stream():
    full = sample_square_sobol2d_uniform_x(64, pool_size=64)
    tail = sample_square_sobol2d_uniform_x(32, start_index=32, pool_size=64)
    assert full[32:] == tail
    assert [s["sobol_stream_index"] for s in tail] == list(range(32, 64))
    with pytest.raises(ValueError, match="pool_size"):
        sample_square_sobol2d_uniform_x(10, start_index=60, pool_size=64)
    with pytest.raises(ValueError, match="power of two"):
        sample_square_sobol2d_uniform_x(10, pool_size=100)


def test_list_samplers_replay_locked_round_lists():
    narrow = REPO_ROOT / (
        "data/sim/start_manifests/square_narrow/r01/blind_inputs/square_narrow_r1.json"
    )
    broad = REPO_ROOT / (
        "data/sim/start_manifests/square_broad/r01/blind_inputs/square_broad_r1.json"
    )
    narrow_states = json.loads(narrow.read_text())["states"]
    broad_states = json.loads(broad.read_text())["states"]

    narrow_sampler = SquareListSampler(narrow, shuffle=False)
    assert narrow_sampler.sample_next() == tuple(
        narrow_states[0][k] for k in ("nut_x", "nut_y", "nut_yaw")
    )
    assert len(narrow_sampler.planned_points) == len(narrow_states)
    assert narrow_sampler._planned_idx_for(*narrow_sampler.planned_points[5]) == 5

    broad_sampler = SquareD1ListSampler(broad, shuffle=False)
    nut, peg = broad_sampler.sample_next()
    assert _broad_state(nut, peg) == {k: broad_states[0][k] for k in _broad_state(nut, peg)}
    assert broad_sampler._planned_idx_for(*nut, *peg) == 0
    assert broad_sampler._planned_idx_for(nut[0] + 0.01, nut[1], nut[2], *peg) is None
