"""The single observation builder shared by demos (``demos.py``), the live env
(``env.py``) and eval. One function, so demo-time and rollout-time observations
cannot drift (the main silent-failure risk of a state-based HiL-SERL setup).

Layout (float32, robomimic ``OBS_KEYS`` order, robomimic object order):

    eef_pos(3)  eef_quat xyzw(4)  gripper_qpos(2)
    nut_pos(3)  nut_quat(4)  nut_to_eef_pos(3)  nut_to_eef_quat(4)  [peg_pos(3)]

23-D on ``square_narrow`` (NutAssemblySquare), 26-D on ``square_broad`` (Square_D1, which
appends ``peg_pos``). The robosuite 1.5.2 env (and therefore the LeRobot
R0 datasets' ``observation.environment_state``) orders the object block as
``nut_to_eef_pos, nut_to_eef_quat, nut_pos, nut_quat[, peg_pos]``; it is reordered
with ``concat(es[7:14], es[0:7], es[14:])``.
"""

from __future__ import annotations

import numpy as np

PROPRIO_DIM = 9
NUT_DIM = 14  # nut pose + nut-to-eef pose; anything after it (peg_pos) keeps its place
OBJECT_DIMS = (14, 17)  # square_narrow, square_broad
ACTION_DIM = 7
ROBOT_NAME = "Panda"
HORIZON = 400  # episode length of both tasks; train and eval share it


def object_state_to_robomimic_order(env_state: np.ndarray) -> np.ndarray:
    """robosuite 1.5.2 ``object-state`` (nut_to_eef first) -> robomimic ``object`` (nut pose first)."""
    env_state = np.asarray(env_state)
    if env_state.shape[-1] not in OBJECT_DIMS:
        raise ValueError(
            f"object-state must be one of {OBJECT_DIMS}-D (Square), got {env_state.shape}"
        )
    return np.concatenate(
        [env_state[..., 7:NUT_DIM], env_state[..., 0:7], env_state[..., NUT_DIM:]], axis=-1
    )


def assemble_obs(eef_pos, eef_quat, gripper_qpos, object_state_rs15_order) -> np.ndarray:
    """Build the observation from robosuite 1.5.2-order components (float32).

    Works on single rows or stacked ``(N, ...)`` arrays."""
    obs = np.concatenate(
        [
            np.asarray(eef_pos, dtype=np.float64),
            np.asarray(eef_quat, dtype=np.float64),
            np.asarray(gripper_qpos, dtype=np.float64),
            object_state_to_robomimic_order(np.asarray(object_state_rs15_order, dtype=np.float64)),
        ],
        axis=-1,
    )
    if obs.shape[-1] - PROPRIO_DIM not in OBJECT_DIMS:
        raise ValueError(f"assembled obs has shape {obs.shape}")
    return obs.astype(np.float32)


def assemble_obs_from_robosuite(raw_obs: dict) -> np.ndarray:
    """Build the observation from a robosuite observation dict."""
    return assemble_obs(
        raw_obs["robot0_eef_pos"],
        raw_obs["robot0_eef_quat"],
        raw_obs["robot0_gripper_qpos"],
        raw_obs["object-state"],
    )
