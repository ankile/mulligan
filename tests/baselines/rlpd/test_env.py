"""The RLPD env on the main stack: action space, controller semantics, the
observation key order shared with the hdf5 demos, early kill by key, and a
recorded reset/step trace."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

from mulligan.baselines.rlpd.datasets import (  # noqa: E402
    OBS_KEYS,
    PH_ENV_META,
    env_meta_for,
    load_robomimic_low_dim,
)

pytest.importorskip("robosuite")
pytest.importorskip("gymnasium")

TRACE = Path(__file__).resolve().parent / "env_trace.json"
TRACE_SEED = 10000
TRACE_STEPS = 20
# robomimic `object` layout -> robosuite 1.5.2 observation keys
OBJECT_KEYS = (
    "SquareNut_pos",
    "SquareNut_quat",
    "SquareNut_to_robot0_eef_pos",
    "SquareNut_to_robot0_eef_quat",
)
ENV_METAS = {
    "NutAssemblySquare": PH_ENV_META,
    "Square_D1": env_meta_for("Square_D1"),
}


@pytest.fixture(scope="module", params=sorted(ENV_METAS))
def env(request):
    from mulligan.baselines.rlpd.env import RLPDEnv

    e = RLPDEnv(ENV_METAS[request.param])
    yield request.param, e
    e.close()


def record_trace(env_name: str) -> dict:
    from mulligan.baselines.rlpd.env import RLPDEnv

    e = RLPDEnv(ENV_METAS[env_name])
    rng = np.random.default_rng(0)
    obs, _ = e.reset(seed=TRACE_SEED)
    out = {"obs": [obs.tolist()], "reward": []}
    for _ in range(TRACE_STEPS):
        obs, reward, terminated, truncated, _ = e.step(rng.uniform(-1, 1, 7))
        assert not (terminated or truncated)
        out["obs"].append(obs.tolist())
        out["reward"].append(reward)
    e.close()
    return out


def test_spaces_and_controller(env):
    name, e = env
    assert e.action_space.shape == (7,)
    assert np.all(e.action_space.low == -1) and np.all(e.action_space.high == 1)
    assert e.observation_space.shape == ({"NutAssemblySquare": 23, "Square_D1": 26}[name],)
    base = e.env.unwrapped
    arm = base.robots[0].part_controllers["right"]
    assert base.control_freq == 20
    np.testing.assert_allclose(arm.output_max, [0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(arm.input_max, 1.0)
    assert arm.input_type == "delta"


def test_controller_mismatch_fails_loudly(env):
    from mulligan.baselines.rlpd.env import check_controller_semantics

    name, e = env
    meta = json.loads(json.dumps(ENV_METAS[name]))
    meta["env_kwargs"]["controller_configs"]["output_max"] = [0.1] * 6
    meta["env_kwargs"]["control_freq"] = 10
    with pytest.raises(ValueError, match="output_max.*\n.*|control_freq"):
        check_controller_semantics(e.env, meta)


def test_obs_key_order_matches_hdf5_layout(env):
    """obs = eef_pos, eef_quat, gripper_qpos, object (robomimic order), read by key."""
    name, e = env
    obs, _ = e.reset(seed=3)
    obs, *_ = e.step(np.zeros(7))
    raw = e.env._last_raw
    parts = [raw["robot0_eef_pos"], raw["robot0_eef_quat"], raw["robot0_gripper_qpos"]]
    parts += [raw[k] for k in OBJECT_KEYS]
    if name == "Square_D1":
        parts.append(raw["object-state"][14:17])  # peg_pos, appended by MimicGen's Square_D1
    np.testing.assert_array_equal(obs, np.concatenate(parts).astype(np.float32))
    # early kill reads the nut position by key; it is this obs slice
    np.testing.assert_array_equal(e.env.nut_pos().astype(np.float32), obs[9:12])


def test_hdf5_reader_concatenates_obs_keys_in_order(tmp_path):
    path = tmp_path / "low_dim_v141.hdf5"
    dims = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2, "object": 14}
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(PH_ENV_META)
        # demo_10 must come after demo_2 (numeric order), as in robomimic
        for i, n in ((10, 3), (2, 2)):
            g = data.create_group(f"demo_{i}")
            g.attrs["num_samples"] = n
            for grp, off in (("obs", 0.0), ("next_obs", 0.5)):
                og = g.create_group(grp)
                for j, k in enumerate(OBS_KEYS):
                    og.create_dataset(k, data=np.full((n, dims[k]), 100 * i + 10 * j + off))
            g.create_dataset("actions", data=np.zeros((n, 7)))
            g.create_dataset("rewards", data=np.zeros(n))
            g.create_dataset("dones", data=np.zeros(n, dtype=np.int64))
    d = load_robomimic_low_dim(path)
    assert d["observations"].dtype == np.float32 and d["observations"].shape == (5, 23)
    expected_row = np.concatenate([np.full(dims[k], 200 + 10 * j) for j, k in enumerate(OBS_KEYS)])
    np.testing.assert_array_equal(d["observations"][0], expected_row)
    np.testing.assert_array_equal(d["observations"][2], expected_row + 800)
    np.testing.assert_array_equal(d["next_observations"][0], expected_row + 0.5)


@pytest.mark.parametrize("env_name", sorted(ENV_METAS))
def test_recorded_reset_step_trace(env_name):
    """reset(seed=10000) + 20 fixed random actions reproduce the recorded trace
    (robosuite 85abee22 / mujoco 3.3.7)."""
    expected = json.loads(TRACE.read_text())[env_name]
    got = record_trace(env_name)
    np.testing.assert_allclose(
        np.array(got["obs"]), np.array(expected["obs"]), rtol=1e-5, atol=1e-6
    )
    assert got["reward"] == expected["reward"]


@pytest.mark.parametrize(("env_step", "expected_len"), [(0, 150), (100_000, 300)])
def test_early_kill_ends_a_zero_action_episode_at_the_no_lift_cutoff(env_step, expected_len):
    """With zero actions the nut is never lifted: the episode is killed as a failure at
    KILL_NO_LIFT_STEP_EARLY before KILL_EARLY_UNTIL training steps, at KILL_NO_LIFT_STEP after."""
    from mulligan.baselines.rlpd import env as env_mod

    assert (env_mod.KILL_NO_LIFT_STEP_EARLY, env_mod.KILL_NO_LIFT_STEP) == (150, 300)
    e = env_mod.RLPDEnv(PH_ENV_META, early_kill=True)
    e.env_step = env_step
    e.reset(seed=TRACE_SEED)
    for t in range(1, 401):
        _, reward, terminated, truncated, info = e.step(np.zeros(7, np.float32))
        if terminated or truncated:
            break
    e.close()
    assert terminated and not truncated and t == expected_len
    assert reward == 0.0 and not info["success"]
    assert info["episode"] == {
        "return": 0.0,
        "length": expected_len,
        "success": False,
        "early_killed": True,
    }


def test_early_kill_is_square_narrow_only():
    from mulligan.baselines.rlpd.env import RLPDEnv

    with pytest.raises(ValueError, match="calibrated for NutAssemblySquare"):
        RLPDEnv(ENV_METAS["Square_D1"], early_kill=True)


if __name__ == "__main__":
    # regenerate the trace: python -m tests.baselines.rlpd.test_env
    TRACE.write_text(json.dumps({k: record_trace(k) for k in sorted(ENV_METAS)}, indent=0) + "\n")
    print(f"wrote {TRACE}")
