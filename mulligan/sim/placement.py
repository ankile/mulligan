"""Place the Square nut and peg at explicit poses (teleop, DAgger, rollouts)."""

from __future__ import annotations

import numpy as np

from mulligan.sim.envs import unwrap_env
from mulligan.utils.quaternion import z_rot_to_quat_wxyz

# The nut rests on the table: table top (0.82 m) plus the nut's half-height.
TABLE_OFFSET_Z = 0.82
NUT_Z_OFFSET = 0.07


def nut_qpos(x: float, y: float, yaw: float) -> np.ndarray:
    """Free-joint qpos [x, y, z, qw, qx, qy, qz] of the nut resting at (x, y, yaw).

    The yaw is wrapped to [0, 2*pi) as robosuite's own placement samplers do.
    ``mulligan.sampling.sobol`` keeps its own ``to_qpos`` (same arithmetic, outside this package).
    """
    position = np.array([x, y, TABLE_OFFSET_Z + NUT_Z_OFFSET], dtype=np.float64)
    quat_wxyz = z_rot_to_quat_wxyz(yaw % (2.0 * np.pi))
    return np.concatenate([position, quat_wxyz])


def _refresh_observations(base_env) -> dict:
    base_env.sim.forward()
    # Clear the base env's cache (a wrapper attribute would shadow it) so relative
    # sensors are recomputed from the new state.
    base_env._obs_cache = {}
    return base_env._get_observations(force_update=True)


def place_nut(env, qpos: np.ndarray) -> dict:
    """Set the nut's free joint to ``qpos`` = [x, y, z, qw, qx, qy, qz]; return fresh observations."""
    base_env = unwrap_env(env)
    nut = base_env.nuts[0]
    joint_id = base_env.sim.model.joint_name2id(nut.joints[0])
    start_idx = base_env.sim.model.jnt_qposadr[joint_id]
    base_env.sim.data.qpos[start_idx : start_idx + 7] = qpos
    return _refresh_observations(base_env)


def place_peg(env, peg_x: float, peg_y: float) -> dict:
    """Move ``peg1`` to (x, y). Pegs are static bodies, so this edits ``sim.model.body_pos``."""
    base_env = unwrap_env(env)
    peg1_body_id = base_env.peg1_body_id
    base_env.sim.model.body_pos[peg1_body_id][0] = peg_x
    base_env.sim.model.body_pos[peg1_body_id][1] = peg_y
    return _refresh_observations(base_env)


def place_square_broad_state(env, sampler, nut_state, peg_state) -> dict:
    """Place peg then nut for Square_D1 (the nut observations depend on the peg).

    Args:
        env: robosuite env (may be wrapped).
        sampler: a ``SquareD1SobolSampler`` (for ``nut_to_qpos``).
        nut_state: (x, y, yaw).
        peg_state: (x, y).
    """
    peg_x, peg_y = peg_state
    place_peg(env, peg_x, peg_y)
    return place_nut(env, sampler.nut_to_qpos(*nut_state))
