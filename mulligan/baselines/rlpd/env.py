# Based on the EXPO source code (https://github.com/pd-perry/EXPO) and the RLPD source code
# (https://github.com/ikostrikov/rlpd), on which EXPO builds.
# Vendored from EXPO's expo/data/robomimic_datasets.py (RobosuiteGymWrapper, get_robomimic_env).
# Both projects are MIT-licensed; their notices follow.
#
# ---- EXPO ----
# MIT License
#
# Copyright (c) 2025 pd-perry
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---- RLPD (https://github.com/ikostrikov/rlpd, LICENCE) ----
# MIT License
#
# Copyright (c) 2022 Ilya Kostrikov, Philip J. Ball, Laura Smith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The RLPD train/eval env: the HiL-SERL env (:class:`mulligan.baselines.hilserl.env.HilSerlEnv`)
with RLPD's episode semantics. The offline data's robomimic ``env_args`` select the task
(``NutAssemblySquare`` -> Square-Narrow, 23-D observations; MimicGen ``Square_D1`` ->
Square-Broad, 26-D) and are checked against the live controller.

- reward 1.0 at success, else 0.0; the episode ends at the first success or at 400 steps;
- the timeout is reported as ``truncated``; the train loop stores mask 0 for every
  episode-ending transition, timeouts included;
- ``--early_kill`` (training env, ``NutAssemblySquare`` only) ends doomed episodes early as
  failures (``terminated``), reading the nut position by observation key (``SquareNut_pos``).
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from mulligan.baselines.hilserl.env import HilSerlEnv
from mulligan.baselines.hilserl.obs import ACTION_DIM

# robosuite env class named by env_args["env_name"] -> HiL-SERL task
ENV_NAME_TO_TASK = {"NutAssemblySquare": "square_narrow", "Square_D1": "square_broad"}

# --early_kill rule (training env only; eval keeps the full horizon). The episode ends as a
# failure (done -> mask 0, like the timeout) when the nut was never lifted KILL_LIFT_M by step
# KILL_NO_LIFT_STEP, or when, after a lift, it has rested within KILL_REST_M of its start height
# more than KILL_PEG_FAR_M from the peg for KILL_REST_STEPS consecutive steps. Until
# KILL_EARLY_UNTIL env steps of training the no-lift cutoff is KILL_NO_LIFT_STEP_EARLY instead.
# The training loop sets env_step before every step so the switch survives a resume. Heights are
# relative to the nut's z at reset, its 0.89 m spawn height before it drops onto the table
# (~0.83 m): a "lift" is ~8 cm above the table and "rest" is anything below ~6.5 cm above it.
KILL_NO_LIFT_STEP = 300
KILL_NO_LIFT_STEP_EARLY = 150
KILL_EARLY_UNTIL = 100_000
KILL_REST_STEPS = 100
KILL_LIFT_M = 0.02
KILL_REST_M = 0.005
KILL_PEG_FAR_M = 0.03


def check_controller_semantics(env: HilSerlEnv, env_meta: dict) -> None:
    """Fail unless the live env's controller does what ``env_meta``'s robomimic
    ``env_kwargs`` describe (OSC_POSE: delta input in [-1, 1], output_max [0.05]*3 + [0.5]*3,
    kp 150, damping 1, 20 Hz)."""
    kw = env_meta["env_kwargs"]
    cc = kw["controller_configs"]
    base = env.unwrapped
    robot = base.robots[0]
    arm = robot.part_controllers["right"]
    problems = []

    def expect(name, live, want):
        if not np.allclose(np.asarray(live, dtype=float), np.asarray(want, dtype=float)):
            problems.append(f"{name}: env {live!r} != env_args {want!r}")

    if kw["robots"] != [robot.name]:
        problems.append(f"robots: env {[robot.name]} != env_args {kw['robots']}")
    if cc["type"] != "OSC_POSE" or type(arm).__name__ != "OperationalSpaceController":
        problems.append(f"controller: env {type(arm).__name__} vs env_args {cc['type']}")
    if not arm.use_ori:
        problems.append("controller: env OSC controls position only")
    expect("control_freq", base.control_freq, kw["control_freq"])
    expect("input_max", arm.input_max, np.full(6, cc["input_max"]))
    expect("input_min", arm.input_min, np.full(6, cc["input_min"]))
    expect("output_max", arm.output_max, cc["output_max"])
    expect("output_min", arm.output_min, cc["output_min"])
    expect("kp", arm.kp, np.full(6, cc["kp"]))
    expect("damping", arm.kd, 2 * np.sqrt(arm.kp) * cc["damping"])
    expect("kp_limits", [arm.kp_min[0], arm.kp_max[0]], cc["kp_limits"])
    expect(
        "damping_limits", [arm.damping_ratio_min[0], arm.damping_ratio_max[0]], cc["damping_limits"]
    )
    if arm.impedance_mode != cc["impedance_mode"]:
        problems.append(f"impedance_mode: env {arm.impedance_mode} != {cc['impedance_mode']}")
    if (arm.input_type == "delta") != bool(cc["control_delta"]):
        problems.append(f"control_delta: env input_type {arm.input_type} vs {cc['control_delta']}")
    if bool(arm.uncoupling) != bool(cc["uncouple_pos_ori"]):
        problems.append(f"uncouple_pos_ori: env {arm.uncoupling} vs {cc['uncouple_pos_ori']}")
    if cc["interpolation"] is not None or arm.interpolator_pos is not None:
        problems.append(f"interpolation: env {arm.interpolator_pos} vs {cc['interpolation']}")
    if cc["position_limits"] is not None or cc["orientation_limits"] is not None:
        problems.append("position/orientation limits are set in env_args")
    if not np.allclose(robot.base_ori, np.eye(3)):
        # robosuite 1.5.2 applies delta actions in the robot base frame, 1.4.1 in the world frame
        problems.append("robot base frame is rotated against the world frame")
    if kw["reward_shaping"]:
        problems.append("env_args ask for shaped rewards; RLPD uses the sparse success reward")
    if problems:
        raise ValueError(
            f"{env_meta['env_name']}: controller differs from the dataset's env_args:\n  "
            + "\n  ".join(problems)
        )


class RLPDEnv(gym.Env):
    """gymnasium view of the task env with RLPD's episode semantics."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, env_meta: dict, horizon: int = 400, early_kill: bool = False):
        name = env_meta["env_name"]
        if name not in ENV_NAME_TO_TASK:
            raise ValueError(f"env_args name {name!r}; supported: {sorted(ENV_NAME_TO_TASK)}")
        if early_kill and name != "NutAssemblySquare":
            raise ValueError(f"--early_kill is calibrated for NutAssemblySquare, got {name}")
        self.env = HilSerlEnv(ENV_NAME_TO_TASK[name], horizon=horizon)
        check_controller_semantics(self.env, env_meta)
        obs_dim = self.env.task.obs_dim
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.horizon = horizon
        self.early_kill = early_kill
        self.env_step: Optional[int] = None  # global training step, set by the loop when early_kill
        self.returns = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs = self.env.reset(seed=seed)
        self.returns = 0.0
        if self.early_kill:
            sim = self.env.sim
            self._peg_xy = sim.data.body_xpos[self.env.unwrapped.peg1_body_id][:2].copy()
            self._nut_z0 = float(self.env.nut_pos()[2])
            self._lifted = False
            self._rest_steps = 0
        return obs, {}

    def _doomed(self) -> bool:
        nut = self.env.nut_pos()
        lift = nut[2] - self._nut_z0
        self._lifted = self._lifted or lift >= KILL_LIFT_M
        away = np.linalg.norm(nut[:2] - self._peg_xy) > KILL_PEG_FAR_M
        resting = self._lifted and lift < KILL_REST_M and away
        self._rest_steps = self._rest_steps + 1 if resting else 0
        no_lift_step = (
            KILL_NO_LIFT_STEP_EARLY if self.env_step < KILL_EARLY_UNTIL else KILL_NO_LIFT_STEP
        )
        return (self.env.t >= no_lift_step and not self._lifted) or (
            self._rest_steps >= KILL_REST_STEPS
        )

    def step(self, action):
        res = self.env.step(np.asarray(action))
        self.returns += res.reward
        terminated = res.success
        truncated = res.truncated
        killed = False
        if self.early_kill and not (terminated or truncated):
            killed = self._doomed()
            terminated = killed
        info: dict[str, Any] = {"success": res.success}
        if terminated or truncated:
            info["episode"] = {"return": self.returns, "length": res.t, "success": res.success}
            if self.early_kill:
                info["episode"]["early_killed"] = killed
        return res.obs, res.reward, terminated, truncated, info

    def close(self):
        self.env.close()
