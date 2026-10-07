"""
Tests for the robosuite runtime monkey-patching layer.

`mulligan` extends robosuite without modifying its source: `apply_runtime_patches()`
(mulligan/__init__.py) installs the patches in `mulligan/sim/_patches.py`, which
swap robosuite's `MjRenderContext` over to the CGL offscreen GL context in
`mulligan/sim/render_cgl.py` on macOS.

This suite verifies that:
1. The CGL rendering patch installs and produces working camera observations on macOS.
2. Stock robosuite environments still render and step correctly once patched.
"""

import sys

import numpy as np
import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="CGL only on macOS")
def test_cgl_rendering_patch():
    """Test that CGL rendering works on macOS."""
    import os

    import robosuite

    import mulligan

    mulligan.apply_runtime_patches()

    # Verify MUJOCO_GL is set to CGL
    assert os.environ.get("MUJOCO_GL") == "cgl"

    # Create environment with camera observations
    env = robosuite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        horizon=10,
        control_freq=20,
    )

    obs = env.reset()

    # Verify camera observations work
    assert "agentview_image" in obs
    assert obs["agentview_image"].shape == (84, 84, 3)
    assert obs["agentview_image"].dtype == np.uint8

    env.close()


def test_base_environment_with_camera_obs():
    """Test that stock robosuite environments still work with camera observations."""
    import robosuite

    import mulligan

    mulligan.apply_runtime_patches()

    env = robosuite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        horizon=10,
        control_freq=20,
    )

    obs = env.reset()

    # Verify standard robosuite behavior
    assert "agentview_image" in obs
    assert obs["robot0_eef_pos"].shape == (3,)

    # Take a step
    action = np.random.randn(env.action_dim) * 0.01
    obs, reward, done, info = env.step(action)

    assert "agentview_image" in obs
    assert isinstance(reward, (int, float))

    env.close()
