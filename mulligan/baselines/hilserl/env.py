"""The task's robosuite 1.5.2 env (``NutAssemblySquare`` for ``square_narrow``,
MimicGen ``Square_D1`` for ``square_broad``) behind the RLPD transition contract.

Built on the env from ``mulligan.sim.envs.create_robosuite_env``:

- 23-D (square_narrow) / 26-D (square_broad) observation from
  ``mulligan.baselines.hilserl.obs.assemble_obs_from_robosuite`` (the one builder demos also use);
- sparse reward ``1.0`` at success, else ``0.0``;
- episode ends at first success or at ``HORIZON`` (400) steps;
- ``mask`` (the TD bootstrap coefficient) is 0 at success; at the timeout it is
  0 too by default (RLPD's convention) or 1 with
  ``truncation_bootstrap=True``;
- ``reset(seed=k)`` reseeds the env's robot-noise generator, every placement
  sampler's generator and ``np.random`` (the Square_D1 peg re-randomization draws
  from it) so the native eval's fixed seeds 10000..10049 give reproducible initial
  states.

``offscreen=True`` + ``frame()`` renders camera images for videos.

Extras the HiL loop needs: MuJoCo snapshot/restore (redo an intervention),
nut translational speed (idle filter), flattened sim state for the episode log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mulligan.baselines.hilserl.obs import (
    ACTION_DIM,
    HORIZON,
    ROBOT_NAME,
    assemble_obs_from_robosuite,
)
from mulligan.baselines.hilserl.tasks import get_task


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool  # episode over (success or timeout)
    mask: float  # TD bootstrap coefficient stored in the buffer
    success: bool
    truncated: bool  # timeout without success
    t: int  # steps taken in this episode (after this step)


@dataclass
class Snapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    time: float
    t: int


class HilSerlEnv:
    def __init__(
        self,
        task: str,
        render: bool = False,
        horizon: int = HORIZON,
        truncation_bootstrap: bool = False,
        offscreen: bool = False,
    ):
        from mulligan.sim.envs import configure_viewer_shadows, create_robosuite_env

        self.task = get_task(task)
        self.env = create_robosuite_env(
            env_name=self.task.env_name,
            robot_name=ROBOT_NAME,
            has_renderer=render,
            has_offscreen_renderer=offscreen,
            control_freq=20,
            reward_shaping=False,
            use_success_wrapper=True,
            use_render_wrapper=False,
        )
        if render:
            # Viewer-only, as in the teleop collector: light shadows + hidden helper geoms.
            # Touches no physics and no offscreen camera render.
            configure_viewer_shadows(self.env)
        self.render_enabled = render
        self.horizon = int(horizon)
        self.truncation_bootstrap = bool(truncation_bootstrap)
        self.t = 0
        self._last_raw: Optional[dict] = None
        self.action_dim = ACTION_DIM

    # --- robosuite plumbing ---------------------------------------------
    @property
    def sim(self):
        return self.env.sim

    @property
    def unwrapped(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            # The robosuite 1.5.2 env draws placement + robot-joint noise from its own
            # Generator (base.py `self.rng`, shared by the placement sampler and passed
            # to robot.reset), so reseed that one in place as well as np.random.
            np.random.seed(int(seed))
            self.unwrapped.rng.bit_generator.state = np.random.default_rng(
                int(seed)
            ).bit_generator.state
            # robosuite's NutAssemblySquare hands env.rng to its placement sampler;
            # MimicGen's Square_D1 builds its nut sampler with rng=None, i.e. its own
            # unseeded default_rng(), so the nut pose ignored the seed above. Reseed
            # every other placement generator too, each from its own stream.
            for i, gen in enumerate(self._placement_generators()):
                gen.bit_generator.state = np.random.default_rng(
                    [int(seed), i + 1]
                ).bit_generator.state
        # robosuite's reset() always destroys the mjviewer and the next step()
        # relaunches it; under mjpython (Mac) the relaunch races the closing
        # window's UI loop and dies with "another MuJoCo viewer is already open".
        # Resets are soft (hard_reset=False: the same MjModel/MjData survive), so
        # the open viewer stays valid -- hide it from reset() and put it back.
        base = self.unwrapped
        if base.hard_reset:
            raise RuntimeError(
                "HilSerlEnv keeps the viewer across resets; requires hard_reset=False"
            )
        renderer = base.viewer
        base.viewer = None
        raw = self.env.reset()
        if renderer is not None:
            base.viewer = renderer
        self.t = 0
        return self._obs(raw)

    def _placement_generators(self) -> list:
        """Distinct placement-sampler generators other than ``env.rng``."""
        base = self.unwrapped
        seen = {id(base.rng)}
        gens = []
        stack = [base.placement_initializer]
        while stack:
            sampler = stack.pop(0)
            if id(sampler.rng) not in seen:
                seen.add(id(sampler.rng))
                gens.append(sampler.rng)
            stack.extend(getattr(sampler, "samplers", {}).values())
        return gens

    def _obs(self, raw: dict) -> np.ndarray:
        self._last_raw = raw
        obs = assemble_obs_from_robosuite(raw)
        if obs.shape != (self.task.obs_dim,):
            raise ValueError(
                f"{self.task.env_name} gave a {obs.shape} observation, expected ({self.task.obs_dim},)"
            )
        return obs

    def step(self, action: np.ndarray, count: bool = True) -> StepResult:
        """Step the sim. ``count=False`` steps physics without advancing the
        episode clock (idle human-bout steps that are not recorded)."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action must be {ACTION_DIM}-D, got {action.shape}")
        raw, _env_reward, _env_done, info = self.env.step(action)
        success = bool(info["success"])
        if count:
            self.t += 1
        obs = self._obs(raw)
        timeout = self.t >= self.horizon
        done = success or timeout
        truncated = timeout and not success
        if success:
            mask = 0.0
        elif truncated:
            mask = 1.0 if self.truncation_bootstrap else 0.0
        else:
            mask = 1.0
        return StepResult(
            obs=obs,
            reward=1.0 if success else 0.0,
            done=done,
            mask=mask,
            success=success,
            truncated=truncated,
            t=self.t,
        )

    def render(self) -> None:
        if self.render_enabled:
            self.env.render()

    def frame(self, camera: str = "agentview", size: int = 256) -> np.ndarray:
        """RGB camera image (``offscreen=True`` only); does not touch the physics."""
        img = self.sim.render(camera_name=camera, width=size, height=size)
        return np.ascontiguousarray(img[::-1])

    def close(self) -> None:
        self.env.close()

    # --- HiL extras -------------------------------------------------------
    def snapshot(self) -> Snapshot:
        st = self.sim.get_state()
        return Snapshot(qpos=np.copy(st.qpos), qvel=np.copy(st.qvel), time=float(st.time), t=self.t)

    def restore(self, snap: Snapshot) -> np.ndarray:
        """Restore a snapshot taken in this episode; returns the observation."""
        self.sim.data.time = snap.time
        self.sim.data.qpos[:] = snap.qpos
        self.sim.data.qvel[:] = snap.qvel
        self.sim.forward()
        self.t = snap.t
        raw = self.unwrapped._get_observations(force_update=True)
        return self._obs(raw)

    def sim_state_flat(self) -> np.ndarray:
        return np.asarray(self.sim.get_state().flatten(), dtype=np.float64)

    def nut_speed(self) -> float:
        """Translational speed (m/s) of the nut body."""
        nut = self.unwrapped.nuts[0]
        return float(np.linalg.norm(self.sim.data.get_body_xvelp(nut.root_body)))

    def nut_pos(self) -> np.ndarray:
        """World position of the square nut, read by observation key."""
        if self._last_raw is None:
            raise RuntimeError("reset() before reading nut_pos")
        return np.asarray(self._last_raw[f"{self.unwrapped.nuts[0].name}_pos"], dtype=np.float64)
