"""Environment factory for the simulated Square tasks.

This is the one place that builds robosuite environments. Two tasks ship:

=================  ====================  =====================================
public name        robosuite env         notes
=================  ====================  =====================================
``square_narrow``  ``NutAssemblySquare``  robosuite's own task; the released
                                          Square-Narrow datasets record the
                                          LeRobot task ``NutAssemblySquare_Panda``
``square_broad``   ``Square_D1``          MimicGen; datasets record
                                          ``Square_D1_Panda``
=================  ====================  =====================================

MimicGen's ``Square_D0`` (MimicGen's own reset bounds for the Square task) is
registered as well, for MimicGen source data. :func:`register_square_environments` installs
the robosuite/MimicGen patches from :mod:`mulligan.sim._patches` explicitly;
:func:`create_robosuite_env` calls it, so no import side effect is needed.

Note: importing the ``mimicgen`` package runs its ``__init__``, which defines
(and, through robosuite's metaclass, registers) every MimicGen task class. Only
the names in :data:`SUPPORTED_ENV_NAMES` are accepted here.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SIM_TASK_ENV_NAMES: dict[str, str] = {
    "square_narrow": "NutAssemblySquare",
    "square_broad": "Square_D1",
}
SUPPORTED_ENV_NAMES: tuple[str, ...] = ("NutAssemblySquare", "Square_D0", "Square_D1")

_SQUARE_REGISTERED = False


def resolve_env_name(name: str) -> str:
    """Map a public task name (``square_narrow``) or a robosuite env ID to the env ID."""
    env_name = SIM_TASK_ENV_NAMES.get(name, name)
    if env_name not in SUPPORTED_ENV_NAMES:
        known = sorted(SIM_TASK_ENV_NAMES) + list(SUPPORTED_ENV_NAMES)
        raise ValueError(f"Unsupported sim task {name!r}; expected one of {known}")
    return env_name


def register_square_environments() -> None:
    """Install the sim patches and register ``Square_D0`` / ``Square_D1`` with robosuite.

    Idempotent. Also needed in ``spawn`` worker processes, which do not inherit
    the parent's patched modules.
    """
    global _SQUARE_REGISTERED
    if _SQUARE_REGISTERED:
        return

    from mulligan.sim._patches import apply_all_patches

    apply_all_patches()

    from mimicgen.envs.robosuite.nut_assembly import Square_D0, Square_D1
    from robosuite.environments.base import REGISTERED_ENVS, register_env

    for env_cls in (Square_D0, Square_D1):
        if env_cls.__name__ not in REGISTERED_ENVS:
            register_env(env_cls)
    _SQUARE_REGISTERED = True


class RobosuiteSuccessWrapper:
    """Add ``info["success"]`` (from ``env._check_success()``) to every step."""

    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["success"] = bool(self.env._check_success())
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class RobosuiteRenderWrapper:
    """Add ``render()`` returning an RGB frame (H, W, 3, uint8) for video recording."""

    def __init__(self, env, render_size=(240, 320), render_camera="agentview"):
        self.env = env
        self.render_size = render_size
        self.render_camera = render_camera

    def render(self):
        frame = self.env.sim.render(
            camera_name=self.render_camera,
            height=self.render_size[0],
            width=self.render_size[1],
        )
        # OpenGL images are bottom-up.
        return frame[::-1]

    def __getattr__(self, name):
        return getattr(self.env, name)


def unwrap_env(env):
    """Return the base robosuite env below the wrappers."""
    while hasattr(env, "env"):
        env = env.env
    return env


def seed_env(env, seed: int) -> None:
    """Seed every random stream a Square reset draws from.

    Environments are unseeded by default. This reseeds,
    in place, the env's ``rng`` (robot init noise, NutAssemblySquare nut
    placement) and every placement-sampler generator that is not ``env.rng``
    (MimicGen builds its nut sampler with its own generator), and gives the env
    a private generator for the Square_D1 peg (``mulligan_peg_rng``, read by the
    peg fix in ``_patches``; unseeded envs draw the peg from ``np.random``).
    """
    base = unwrap_env(env)
    base.mulligan_peg_rng = np.random.RandomState(int(seed))
    base.rng.bit_generator.state = np.random.default_rng(int(seed)).bit_generator.state
    seen = {id(base.rng)}
    stack = [base.placement_initializer]
    stream = 0
    while stack:
        sampler = stack.pop(0)
        if id(sampler.rng) not in seen:
            seen.add(id(sampler.rng))
            stream += 1
            sampler.rng.bit_generator.state = np.random.default_rng(
                [int(seed), stream]
            ).bit_generator.state
        stack.extend(getattr(sampler, "samplers", {}).values())


def create_robosuite_env(
    env_name: str,
    robot_name: str = "Panda",
    camera_names: Optional[list[str]] = None,
    camera_height: int = 256,
    camera_width: int = 256,
    controller: Optional[str] = None,
    render_camera: str = "agentview",
    has_renderer: bool = False,
    has_offscreen_renderer: Optional[bool] = None,
    visual_aids: bool = False,
    control_freq: int = 20,
    reward_shaping: bool = False,
    render_size: tuple[int, int] = (240, 320),
    use_success_wrapper: bool = True,
    use_render_wrapper: bool = True,
    seed: Optional[int] = None,
):
    """Create a Square environment with the settings of the paper's experiments.

    Args:
        env_name: public task name (``square_narrow`` / ``square_broad``) or
            robosuite env ID (``NutAssemblySquare``, ``Square_D0``, ``Square_D1``).
        robot_name: robosuite robot (the paper uses ``Panda``).
        camera_names: cameras for image observations; ``None`` disables them.
        camera_height, camera_width: camera resolution.
        controller: robosuite controller name (default: the robot's default
            composite controller, OSC pose for the Panda arm).
        render_camera: camera for the viewer and for ``RobosuiteRenderWrapper``.
        has_renderer: open the MuJoCo viewer (``mjviewer``).
        has_offscreen_renderer: default: on when cameras or video are requested.
        visual_aids: wrap with robosuite's ``VisualizationWrapper``.
        control_freq: control frequency in Hz.
        reward_shaping: dense reward instead of sparse success.
        render_size: (height, width) of video frames.
        use_success_wrapper: add ``info["success"]``.
        use_render_wrapper: add ``render()`` for video frames.
        seed: seed all reset randomness (see :func:`seed_env`). ``None`` leaves
            resets unseeded.

    The env ignores ``done`` (``ignore_done=True``) and uses soft resets
    (``hard_reset=False``); callers enforce the horizon.
    """
    import robosuite as suite
    from robosuite import load_composite_controller_config
    from robosuite.wrappers import VisualizationWrapper

    env_name = resolve_env_name(env_name)
    register_square_environments()

    controller_config = load_composite_controller_config(controller=controller, robot=robot_name)
    use_camera_obs = camera_names is not None and len(camera_names) > 0
    if has_offscreen_renderer is None:
        has_offscreen_renderer = use_camera_obs or use_render_wrapper

    env_config = {
        "env_name": env_name,
        "robots": robot_name,
        "controller_configs": controller_config,
        "has_renderer": has_renderer,
        "has_offscreen_renderer": has_offscreen_renderer,
        "render_camera": render_camera,
        "ignore_done": True,
        "use_camera_obs": use_camera_obs,
        "use_object_obs": True,
        "reward_shaping": reward_shaping,
        "control_freq": control_freq,
        "hard_reset": False,
    }
    # MimicGen tasks default to the OpenCV "mujoco" renderer; teleop needs the MuJoCo viewer.
    if has_renderer:
        env_config["renderer"] = "mjviewer"
    if use_camera_obs:
        env_config["camera_names"] = camera_names
        env_config["camera_heights"] = camera_height
        env_config["camera_widths"] = camera_width
        env_config["camera_depths"] = False
        logger.info(
            "Camera observations enabled: %s at %dx%d", camera_names, camera_width, camera_height
        )

    env = suite.make(**env_config)
    if seed is not None:
        seed_env(env, seed)

    if use_success_wrapper:
        env = RobosuiteSuccessWrapper(env)
    if visual_aids:
        env = VisualizationWrapper(env, indicator_configs=None)
    if use_render_wrapper:
        env = RobosuiteRenderWrapper(env, render_size=render_size, render_camera=render_camera)
    return env


def configure_viewer_shadows(env) -> None:
    """Shadows in the MuJoCo viewer without changing offscreen camera renders.

    Enables shadow casting on the shared model and turns shadows off again in the
    offscreen renderer, so recorded camera observations are unchanged. Walls,
    invisible helper geoms (plumb-line guides) and the gripper's 10 m
    ``grip_site_cylinder`` site (which casts a line shadow across the table) move
    to hidden viewer groups (geom group 4, site group 5) and stay visible
    offscreen. Viewer-only: no physics change.

    Call after :func:`create_robosuite_env` and before the first ``env.step()``.
    """
    import mujoco

    base_env = unwrap_env(env)
    model = base_env.sim.model._model

    model.light_castshadow[:] = 1
    # shadowclip scales the directional-light shadow frustum by the model extent
    # (16.5 on Square_D1: floor + walls); 0.15 keeps the whole robot shadow
    # (0.07 clips it) and roughly triples the texel density on the table vs 0.5.
    model.vis.quality.shadowsize = 8192
    model.vis.map.shadowclip = 0.15

    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        name_lower = name.lower() if name else ""
        alpha = float(model.geom_rgba[i, 3])
        is_wall_visual = "wall_" in name_lower and "_visual" in name_lower
        is_plumb_helper = "plumb" in name_lower or "plum" in name_lower
        is_invisible_helper = alpha <= 1e-3
        if is_wall_visual or is_plumb_helper or is_invisible_helper:
            model.geom_group[i] = 4
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and "grip_site_cylinder" in name:
            model.site_group[i] = 5

    offscreen = base_env.sim._render_context_offscreen
    if offscreen is not None:
        offscreen.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        offscreen.vopt.geomgroup[4] = 1
        offscreen.vopt.sitegroup[5] = 1

    renderer = base_env.viewer
    if renderer is None:
        raise RuntimeError("configure_viewer_shadows requires an initialized mjviewer renderer")

    def _apply_viewer_options():
        viewer = renderer.viewer
        if viewer is None:
            raise RuntimeError("mjviewer update completed without creating a viewer handle")
        viewer.opt.geomgroup[4] = 0
        viewer.opt.sitegroup[5] = 0

    if renderer.viewer is not None:
        _apply_viewer_options()
    else:
        original_update = renderer.update

        def update_with_shadow_options(*args, **kwargs):
            result = original_update(*args, **kwargs)
            _apply_viewer_options()
            return result

        renderer.update = update_with_shadow_options
