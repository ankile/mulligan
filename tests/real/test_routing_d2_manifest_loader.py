"""routing_d2 blind-teleop manifest validates + round-trips through the collection loader.

routing_d2 is the first NON-PEN line: the loader (mulligan.real.collect.blind_dagger) must accept
its 9-key manifest, validate the rope against the registry bounds and each clip against the
registry grid + orientation set, and round-trip every column. These tests build a small
manifest from the routing_d2 TaskGeometry sampler (the same path the manifest builder uses) and
assert the loader accepts it and rejects the off-grid / off-orientation corruptions the
registry guard exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path


from mulligan.real.collect.initial_states import (
    ArmSpec,
    ROUTING_D2_INITIAL_STATE_KEYS,
    _load_initial_state_manifest,
)
from mulligan.real.lifecycle.geometry import TaskGeometry
from mulligan.real.lifecycle.tasks import get_task_spec

ARM = "mulligan_sobol"


def _build_manifest(n: int = 24) -> dict:
    spec = get_task_spec("routing_d2")
    tg = TaskGeometry(spec)
    # Oversample and keep only states whose clips satisfy the >= 2 in separation (the loader
    # now enforces it), mirroring the manifest builder's rejection sampling.
    points = tg.sobol(2026062902, n * 8)
    states = []
    for point in points:
        row = {k: float(v) for k, v in zip(spec.sampling_keys, point)}
        for placement in spec.sampled_placements.values():
            idx = int(round(row[placement.orient_key]))
            row[placement.orient_key] = float(idx)
            row[placement.orient_angle_key] = placement.angle_for_index(idx)
        if spec.min_placement_separation(row) < spec.placement_min_separation_m - 1e-9:
            continue
        i = len(states)
        row.update({"source": ARM, "sources": [ARM], "source_index": i, "manifest_idx": i})
        states.append(row)
        if len(states) == n:
            break
    assert len(states) == n, "not enough feasible states drawn for the test"
    return {
        "task": "routing_d2",
        "schema": "real_manual_initial_states_v1",
        "keys": list(spec.manifest_keys),
        "states": states,
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "routing_manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_routing_manifest_loads_and_round_trips(tmp_path):
    payload = _build_manifest()
    path = _write(tmp_path, payload)
    arms = [ArmSpec(ARM, "wandb://e/p/m:v0")]
    targets, loaded = _load_initial_state_manifest(path, arms, expected_task="routing_d2")
    assert loaded["keys"] == ROUTING_D2_INITIAL_STATE_KEYS
    assert len(targets) == len(payload["states"])
    spec = get_task_spec("routing_d2")
    for tgt, row in zip(targets, payload["states"]):
        # Non-pen line: no pen pose is set on the target.
        assert tgt.pen_x is None and tgt.pen_y is None and tgt.pen_yaw is None
        assert tgt.rope_x == row["rope_x"]
        for clip in ("clip_left", "clip_right"):
            assert getattr(tgt, f"{clip}_x") == row[f"{clip}_x"]
            assert getattr(tgt, f"{clip}_y") == row[f"{clip}_y"]
            assert getattr(tgt, f"{clip}_yaw") == row[f"{clip}_yaw"]
            oidx = int(tgt.raw[f"{clip}_oidx"])
            # the resolved angle agrees with the persisted index
            assert spec.sampled_placements[clip].angle_for_index(oidx) == row[f"{clip}_yaw"]


def test_routing_manifest_rejects_off_grid_clip(tmp_path):
    payload = _build_manifest()
    payload["states"][0]["clip_left_x"] += 0.003  # nudge off the 1-inch dot grid
    path = _write(tmp_path, payload)
    arms = [ArmSpec(ARM, "wandb://e/p/m:v0")]
    try:
        _load_initial_state_manifest(path, arms, expected_task="routing_d2")
    except ValueError as exc:
        assert "not an in-bounds grid point" in str(exc)
    else:
        raise AssertionError("expected an off-grid clip to be rejected")


def test_routing_manifest_rejects_off_set_orientation(tmp_path):
    payload = _build_manifest()
    payload["states"][0]["clip_right_yaw"] = 0.0  # 0 rad is not in {-45, +45, +90} deg
    path = _write(tmp_path, payload)
    arms = [ArmSpec(ARM, "wandb://e/p/m:v0")]
    try:
        _load_initial_state_manifest(path, arms, expected_task="routing_d2")
    except ValueError as exc:
        assert "not within" in str(exc)
    else:
        raise AssertionError("expected an off-set orientation angle to be rejected")


def test_routing_manifest_rejects_index_angle_disagreement(tmp_path):
    payload = _build_manifest()
    row = payload["states"][0]
    row["clip_left_oidx"] = float(
        (int(row["clip_left_oidx"]) + 1) % 3
    )  # index no longer matches angle
    path = _write(tmp_path, payload)
    arms = [ArmSpec(ARM, "wandb://e/p/m:v0")]
    try:
        _load_initial_state_manifest(path, arms, expected_task="routing_d2")
    except ValueError as exc:
        assert "disagrees with angle" in str(exc)
    else:
        raise AssertionError("expected an index/angle disagreement to be rejected")


def test_routing_manifest_rejects_clips_too_close(tmp_path):
    payload = _build_manifest()
    spec = get_task_spec("routing_d2")
    # Force clip_right onto clip_left's x and just 1 inch away in y (on the dot grid) so the
    # pair is 1 in < 2 in apart, in a y-range valid for clip_right.
    row = payload["states"][0]
    row["clip_left_x"] = 0.5 * 0.0254
    row["clip_left_y"] = 0.5 * 0.0254
    row["clip_right_x"] = 0.5 * 0.0254
    row["clip_right_y"] = -0.5 * 0.0254
    assert spec.min_placement_separation(row) < spec.placement_min_separation_m  # sanity
    path = _write(tmp_path, payload)
    arms = [ArmSpec(ARM, "wandb://e/p/m:v0")]
    try:
        _load_initial_state_manifest(path, arms, expected_task="routing_d2")
    except ValueError as exc:
        assert "below the required" in str(exc) and "minimum" in str(exc)
    else:
        raise AssertionError("expected clips closer than 2 in to be rejected")


def test_routing_manifest_rejects_out_of_bounds_rope(tmp_path):
    payload = _build_manifest()
    payload["states"][0]["rope_x"] = 99.0  # absurd rope_x (meters)
    path = _write(tmp_path, payload)
    arms = [ArmSpec(ARM, "wandb://e/p/m:v0")]
    try:
        _load_initial_state_manifest(path, arms, expected_task="routing_d2")
    except ValueError as exc:
        assert "out of bounds" in str(exc)
    else:
        raise AssertionError("expected an out-of-bounds rope_x to be rejected")
