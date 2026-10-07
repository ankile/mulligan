"""DROID ZED camera configuration.

Import this module before creating a RobotEnv or calling gather_zed_cameras().
Ensures the aruco API shim is applied before any DROID camera imports.

Camera FPS is left at the DROID default (60fps for HD720). The ZED SDK's
grab() is a blocking call, so 60fps ensures frames are always available
for the 15Hz control loop without blocking or introducing motion blur.
"""

import mulligan.real.robot.droid_compat  # noqa: F401  (aruco API shim — must come before droid imports)
import droid.camera_utils.camera_readers.zed_camera as _zed_mod  # noqa: F401
