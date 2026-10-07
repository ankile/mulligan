"""Compatibility shims for importing DROID in different environments.

DROID depends on opencv's legacy aruco API (< 4.7), which opencv 4.7+ renamed.
Import this module **before** ``droid.robot_env`` to patch the difference:

    from mulligan.real.robot.droid_compat import RobotEnv  # applies shims automatically
"""

import cv2

from mulligan.real.robot.station import require_station_env

# Station identity (nuc_ip, robot_ip, robot_type, serial) reaches DROID from the
# per-machine file ~/.config/droid/station.env, read by droid.misc.parameters at
# import time. Validate it here, before any robot motion: missing/incomplete
# file, a NUC-only secret on the workstation, or a droid module already bound
# to different values all raise with the fix.
require_station_env()

# --------------------------------------------------------------------------- #
# OpenCV aruco API rename shim (opencv 4.8+)
# --------------------------------------------------------------------------- #
# droid/droid/misc/parameters.py calls aruco.Dictionary_get() which was renamed
# to aruco.getPredefinedDictionary() in opencv 4.7.  Patch it back so droid
# works without source modifications.

_aruco = getattr(cv2, "aruco", None)
if _aruco is not None and not hasattr(_aruco, "Dictionary_get"):
    _aruco.Dictionary_get = _aruco.getPredefinedDictionary
    _aruco.DetectorParameters_create = _aruco.DetectorParameters
    _orig_CharucoBoard = _aruco.CharucoBoard

    def _charuco_board_create(squaresX, squaresY, squareLength, markerLength, dictionary):
        return _orig_CharucoBoard((squaresX, squaresY), squareLength, markerLength, dictionary)

    _aruco.CharucoBoard_create = _charuco_board_create

# --------------------------------------------------------------------------- #
# Re-export RobotEnv (safe to import after shims are applied)
# --------------------------------------------------------------------------- #
from droid.robot_env import RobotEnv  # noqa: E402, F401

__all__ = ["RobotEnv"]
