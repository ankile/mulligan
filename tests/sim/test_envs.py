"""Env factory, MimicGen shims and vec-env seeding (no rendering, CPU only)."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from mulligan.sim.envs import (
    SIM_TASK_ENV_NAMES,
    create_robosuite_env,
    register_square_environments,
    resolve_env_name,
    unwrap_env,
)
from mulligan.sim.vec_env import AsyncVectorEnv, SyncVectorEnv


def _make(task: str, seed=None):
    return create_robosuite_env(task, use_render_wrapper=False, seed=seed)


def _qpos(env) -> np.ndarray:
    return np.array(env.sim.data.qpos, copy=True)


def _start(env) -> np.ndarray:
    """qpos plus, on Square_D1, the peg position (a model body position, not in qpos)."""
    base = unwrap_env(env)
    peg = base.sim.model.body_pos[base.peg1_body_id][:2] if hasattr(base, "peg1_body_id") else []
    return np.concatenate([_qpos(env), np.array(peg, dtype=float)])


def test_public_names_map_to_paper_envs():
    assert SIM_TASK_ENV_NAMES == {"square_narrow": "NutAssemblySquare", "square_broad": "Square_D1"}
    assert resolve_env_name("square_narrow") == "NutAssemblySquare"
    assert resolve_env_name("Square_D0") == "Square_D0"
    with pytest.raises(ValueError, match="Unsupported sim task"):
        resolve_env_name("Threading_D0")


def test_shims_installed_and_square_registered():
    import sys

    from robosuite.environments.base import REGISTERED_ENVS
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

    register_square_environments()
    shim = sys.modules["robosuite.environments.manipulation.single_arm_env"]
    assert shim.SingleArmEnv is ManipulationEnv
    assert getattr(ManipulationEnv._get_rel_obj_eef_sensor, "_patched", False)
    for name in ("NutAssemblySquare", "Square_D0", "Square_D1"):
        assert name in REGISTERED_ENVS


@pytest.mark.parametrize("task, dim", [("square_narrow", 14), ("square_broad", 17)])
def test_first_observation_has_relative_nut_pose(task, dim):
    env = _make(task)
    try:
        obs = env.reset()
        assert obs["object-state"].shape == (dim,)
        # object-state[:3] is nut_to_eef_pos, which was all zeros on the first observation before the sensor fix.
        assert np.linalg.norm(obs["object-state"][:3]) > 1e-3
        _, _, _, info = env.step(np.zeros(env.action_dim))
        assert info["success"] is False
    finally:
        env.close()


def test_square_broad_peg_moves_on_every_soft_reset():
    env = _make("square_broad")
    try:
        base = unwrap_env(env)
        assert base.hard_reset is False
        pegs = []
        for _ in range(4):
            env.reset()
            pegs.append(np.array(base.sim.data.body_xpos[base.peg1_body_id][:2]))
        assert len({tuple(np.round(p, 6)) for p in pegs}) == 4
    finally:
        env.close()


@pytest.mark.parametrize("task", ["square_narrow", "square_broad"])
def test_seeded_env_resets_are_reproducible(task):
    a, b, c = _make(task, seed=11), _make(task, seed=11), _make(task, seed=12)
    try:
        for _ in range(2):
            a.reset(), b.reset(), c.reset()
            np.testing.assert_array_equal(_start(a), _start(b))
            assert not np.allclose(_start(a), _start(c))
            if task == "square_broad":
                base_a, base_c = unwrap_env(a), unwrap_env(c)
                peg = base_a.sim.model.body_pos[base_a.peg1_body_id][:2]
                assert not np.allclose(peg, base_c.sim.model.body_pos[base_c.peg1_body_id][:2])
    finally:
        for env in (a, b, c):
            env.close()


@pytest.mark.parametrize("task", ["square_narrow", "square_broad"])
def test_seeded_sync_and_async_vec_envs_agree(task):
    make = functools.partial(_make, task)
    sync = SyncVectorEnv([make, make], seed=3)
    asyn = AsyncVectorEnv([make, make], start_method="spawn", seed=3)
    try:
        for _ in range(2):
            obs_sync, qpos_sync, _ = sync.reset_with_sim_state()
            obs_async, qpos_async, _ = asyn.reset_with_sim_state()
            for q_sync, q_async in zip(qpos_sync, qpos_async):
                np.testing.assert_array_equal(q_sync, q_async)
            assert not np.allclose(qpos_sync[0], qpos_sync[1])
            # object-state carries the nut pose and, on Square_D1, the peg position [14:16].
            for o_sync, o_async in zip(obs_sync, obs_async):
                np.testing.assert_array_equal(o_sync["object-state"], o_async["object-state"])
            if task == "square_broad":
                pegs = [o["object-state"][14:16] for o in obs_sync]
                assert not np.allclose(pegs[0], pegs[1])
    finally:
        sync.close()
        asyn.close()


def test_reset_with_placements_places_nut_and_peg():
    make = functools.partial(_make, "square_broad")
    vec_env = SyncVectorEnv([make])
    try:
        base = unwrap_env(vec_env.envs[0])
        joint_id = base.sim.model.joint_name2id(base.nuts[0].joints[0])
        start = int(base.sim.model.jnt_qposadr[joint_id])
        nut_qpos = np.array([0.05, -0.1, 0.89, 1.0, 0.0, 0.0, 0.0])
        obs, qpos, _ = vec_env.reset_with_placements(
            [[("body_pos", "peg1", [0.1, 0.05]), ("qpos", start, nut_qpos)]]
        )
        np.testing.assert_allclose(obs[0]["object-state"][14:16], [0.1, 0.05])
        np.testing.assert_allclose(qpos[0][start : start + 7], nut_qpos)
        with pytest.raises(ValueError, match="Unknown placement kind"):
            vec_env.reset_with_placements([[("site", "x", [0, 0])]])
    finally:
        vec_env.close()


def test_fixed_eval_state_reset_accepts_public_task_names():
    from mulligan.training.evaluation import make_fixed_eval_initial_states

    spec = make_fixed_eval_initial_states("square_broad", 1, 20260524)[0]
    assert spec == make_fixed_eval_initial_states("Square_D1", 1, 20260524)[0]
    vec_env = SyncVectorEnv([functools.partial(_make, "square_broad")])
    try:
        (obs,) = vec_env.reset_to_eval_initial_states([{**spec, "env_name": "square_broad"}])
        np.testing.assert_allclose(obs["object-state"][14:16], spec["peg"])
        base = unwrap_env(vec_env.envs[0])
        np.testing.assert_allclose(base.sim.model.body_pos[base.peg1_body_id][:2], spec["peg"])
    finally:
        vec_env.close()


def test_unseeded_forked_workers_draw_different_pegs():
    """Forked workers must not share the parent's global NumPy stream (Square_D1 peg).

    NumPy seeds its global generator on first use, so the parent touches it first,
    as ``--seed`` in rollouts.py and train.py's seeding do before building workers.
    """
    make = functools.partial(_make, "square_broad")
    parent_state = np.random.get_state()
    np.random.seed(0)
    try:
        vec_env = AsyncVectorEnv([make, make, make], start_method="fork")
    finally:
        np.random.set_state(parent_state)
    try:
        for _ in range(2):
            obs = vec_env.reset()
            pegs = {tuple(np.round(o["object-state"][14:16], 8)) for o in obs}
            assert len(pegs) == 3
    finally:
        vec_env.close()


def test_eval_reset_seed_keeps_samplers_on_env_rng():
    """A seeded eval reset reseeds env.rng in place: the placement samplers, built
    with rng=env.rng, keep sharing the env's generator."""
    from mulligan.training.evaluation import make_fixed_eval_initial_states

    spec = make_fixed_eval_initial_states("square_narrow", 1, 0)[0]
    vec_env = SyncVectorEnv([functools.partial(_make, "square_narrow")])
    try:
        base = unwrap_env(vec_env.envs[0])
        rng = base.rng
        vec_env.reset_to_eval_initial_states([spec])
        assert base.rng is rng
        samplers = list(base.placement_initializer.samplers.values())
        assert samplers and all(s.rng is base.rng for s in samplers)
    finally:
        vec_env.close()
