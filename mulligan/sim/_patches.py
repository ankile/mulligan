# The CGL render-context initializer in apply_cgl_rendering_patch is adapted from robosuite
# (https://github.com/ARISE-Initiative/robosuite, commit 85abee2 = 1.5.2),
# MjRenderContext.__init__ in robosuite/utils/binding_utils.py. Modified by the Mulligan
# authors: MUJOCO_GL=cgl selects a CGL context (macOS); the GL backend is read from the
# environment at construction time. The rest of the file is by the Mulligan authors.
#
# robosuite's MIT notice:
#
# MIT License
#
# Copyright (c) 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab
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

"""Runtime patches for robosuite 1.5.2 and MimicGen, installed explicitly.

``mulligan.sim.envs.register_square_environments`` calls :func:`apply_all_patches`
before it builds any environment, and ``mulligan.apply_runtime_patches`` calls it
for entry points that construct robosuite envs directly. Every installer is
idempotent.

- :func:`install_mimicgen_compatibility_shim`: MimicGen imports
  ``robosuite.environments.manipulation.single_arm_env.SingleArmEnv``, which
  robosuite 1.5 removed; alias it to ``ManipulationEnv``.
- :func:`install_relative_sensor_fix`: relative object sensors (``nut_to_eef_pos``)
  returned zeros on the first observation after a reset.
- :func:`install_square_broad_peg_randomization_fix`: re-sample the Square_D1 peg on
  every soft reset (MimicGen only places it in ``_load_model``).
- :func:`apply_cgl_rendering_patch`: CGL offscreen rendering on macOS.
"""

import os
import sys
from types import ModuleType

import numpy as np


def install_mimicgen_compatibility_shim() -> None:
    """Provide ``SingleArmEnv`` (robosuite 1.4) as an alias of ``ManipulationEnv`` (1.5).

    Must run before the first ``import mimicgen``.
    """
    module_name = "robosuite.environments.manipulation.single_arm_env"
    if module_name in sys.modules:
        return

    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

    shim_module = ModuleType(module_name)
    shim_module.SingleArmEnv = ManipulationEnv
    shim_module.__doc__ = "Compatibility shim for MimicGen (SingleArmEnv -> ManipulationEnv)"
    sys.modules[module_name] = shim_module


def install_relative_sensor_fix() -> None:
    """Compute object poses from the sim when the relative sensor runs before them.

    ``ManipulationEnv._get_rel_obj_eef_sensor`` read ``{obj}_pos`` / ``{obj}_quat``
    from ``obs_cache`` and returned zeros when they were missing, which is the
    case on the first observation after a reset (relative sensors are sampled
    before the absolute ones).
    """
    from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
    from robosuite.utils.observables import sensor
    import robosuite.utils.transform_utils as T

    if hasattr(ManipulationEnv._get_rel_obj_eef_sensor, "_patched"):
        return

    def _get_rel_obj_eef_sensor_fixed(self, prefix, obj_key, fn_name, new_key_prefix, modality):
        @sensor(modality=modality)
        def fn(obs_cache):
            obj_pos_key = f"{obj_key}_pos"
            obj_quat_key = f"{obj_key}_quat"
            world_pose_key = f"world_pose_in_{prefix}gripper"

            if obj_pos_key in obs_cache:
                obj_pos = obs_cache[obj_pos_key]
            else:
                obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_key]])
                obs_cache[obj_pos_key] = obj_pos

            if obj_quat_key in obs_cache:
                obj_quat = obs_cache[obj_quat_key]
            else:
                obj_quat = T.convert_quat(
                    self.sim.data.body_xquat[self.obj_body_id[obj_key]], to="xyzw"
                )
                obs_cache[obj_quat_key] = obj_quat

            # world_pose_in_gripper is registered before the object sensors; robosuite's
            # own sensor returns zeros when it is missing, and so does this one.
            if world_pose_key not in obs_cache:
                return np.zeros(3)

            obj_pose = T.pose2mat((obj_pos, obj_quat))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache[world_pose_key])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_key}_to_{new_key_prefix}eef_quat"] = rel_quat
            return rel_pos

        fn.__name__ = fn_name
        return fn

    ManipulationEnv._get_rel_obj_eef_sensor = _get_rel_obj_eef_sensor_fixed
    ManipulationEnv._get_rel_obj_eef_sensor._patched = True


def install_square_broad_peg_randomization_fix() -> None:
    """Re-randomize the Square_D1 peg on every ``env.reset()``.

    MimicGen's Square_D1 places the peg only in ``_load_model``. With
    ``hard_reset=False`` (used everywhere for speed) that runs once per env
    instance, so each env keeps one peg position across resets. The fix samples
    the peg before the original ``_reset_internal``, whose nut rejection loop
    then sees the new peg (MimicGen's original order: the peg is placed freely,
    the nut avoids it). The peg is drawn from the global ``np.random`` stream, as
    in MimicGen's ``_load_model``, unless ``mulligan.sim.envs.seed_env`` gave the
    env its own ``mulligan_peg_rng``.

    Requires :func:`install_mimicgen_compatibility_shim` first.
    """
    from mimicgen.envs.robosuite.nut_assembly import Square_D1

    if getattr(Square_D1._reset_internal, "_peg_patched", False):
        return

    orig_reset_internal = Square_D1._reset_internal

    def _reset_internal_with_peg(self):
        # deterministic_reset restores exact saved states; nothing to re-sample.
        if not self.deterministic_reset:
            rng = getattr(self, "mulligan_peg_rng", None) or np.random
            peg_bounds = self._get_initial_placement_bounds()["peg"]
            peg1_id = self.sim.model.body_name2id("peg1")
            ref_x = peg_bounds["reference"][0]
            ref_y = peg_bounds["reference"][1]
            sample_x = rng.uniform(low=peg_bounds["x"][0], high=peg_bounds["x"][1])
            sample_y = rng.uniform(low=peg_bounds["y"][0], high=peg_bounds["y"][1])
            self.sim.model.body_pos[peg1_id][0] = ref_x + sample_x
            self.sim.model.body_pos[peg1_id][1] = ref_y + sample_y

            # Square_D1 peg z_rot bounds are (0, 0); re-sample only if a subclass widens them.
            rot_lo, rot_hi = peg_bounds["z_rot"][0], peg_bounds["z_rot"][1]
            if rot_hi > rot_lo:
                sample_z_rot = rng.uniform(low=rot_lo, high=rot_hi)
                self.sim.model.body_quat[peg1_id][0] = np.cos(sample_z_rot / 2.0)
                self.sim.model.body_quat[peg1_id][1] = 0.0
                self.sim.model.body_quat[peg1_id][2] = 0.0
                self.sim.model.body_quat[peg1_id][3] = np.sin(sample_z_rot / 2.0)

            # The original nut loop reads body_xpos of the peg.
            self.sim.forward()

        orig_reset_internal(self)

    _reset_internal_with_peg._peg_patched = True
    Square_D1._reset_internal = _reset_internal_with_peg


def apply_cgl_rendering_patch() -> None:
    """Use a CGL offscreen context for robosuite renders on macOS (no-op elsewhere).

    CGL contexts are not tied to a Cocoa window, so offscreen camera renders work
    next to the MuJoCo viewer. Sets ``MUJOCO_GL=cgl`` if unset.
    """
    if sys.platform != "darwin":
        return

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "cgl"

    import robosuite.utils.binding_utils as binding_utils

    if getattr(binding_utils.MjRenderContext.__init__, "_cgl_patched", False):
        return

    from mulligan.sim.render_cgl import CGLGLContext

    def patched_init(self, sim, offscreen=True, device_id=-1, max_width=640, max_height=480):
        import mujoco

        mujoco_gl = os.environ.get("MUJOCO_GL", "").lower()
        if mujoco_gl not in ("disable", "disabled", "off", "false", "0"):
            valid = ("enable", "enabled", "on", "true", "1", "glfw", "", "cgl")
            if mujoco_gl not in valid:
                raise RuntimeError(f"invalid value for environment variable MUJOCO_GL: {mujoco_gl}")
            if mujoco_gl == "cgl":
                GLContext = CGLGLContext
            else:
                from robosuite.renderers.context.glfw_context import GLFWGLContext as GLContext

        assert offscreen, "only offscreen supported for now"
        self.sim = sim
        self.offscreen = offscreen
        self.device_id = device_id

        self.gl_ctx = GLContext(
            max_width=max_width, max_height=max_height, device_id=self.device_id
        )
        self.gl_ctx.make_current()

        # Make sure there is something to render and that the sim knows this context.
        sim.forward()
        sim.add_render_context(self)

        self.model = sim.model
        self.data = sim.data

        # maxgeom 10k supports large scenes.
        self.scn = mujoco.MjvScene(sim.model._model, maxgeom=10000)

        self.cam = mujoco.MjvCamera()
        self.cam.fixedcamid = 0
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        self.vopt = mujoco.MjvOption()

        self.pert = mujoco.MjvPerturb()
        self.pert.active = 0
        self.pert.select = 0
        self.pert.skinselect = -1

        self.con = mujoco.MjrContext(self.model._model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

    patched_init._cgl_patched = True
    binding_utils.MjRenderContext.__init__ = patched_init


def apply_all_patches() -> None:
    """Install every robosuite / MimicGen patch (idempotent).

    Order matters: the SingleArmEnv shim must exist before MimicGen is imported
    by the peg fix.
    """
    apply_cgl_rendering_patch()
    install_mimicgen_compatibility_shim()
    install_relative_sensor_fix()
    install_square_broad_peg_randomization_fix()
