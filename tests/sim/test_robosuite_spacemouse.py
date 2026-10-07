"""SpaceMouse driver helpers (no device needed).

With a device attached, ``python -m mulligan.teleop.spacemouse`` prints live readings.
"""

import numpy as np

from mulligan.teleop import spacemouse
from mulligan.teleop.spacemouse import RobosuiteSpaceMouse


def test_hid_bytes_to_control():
    assert spacemouse.to_int16(0x10, 0x00) == 16
    assert spacemouse.to_int16(0xFF, 0xFF) == -1
    assert spacemouse.to_int16(0x00, 0x80) == -32768
    assert spacemouse.convert(0x5E, 0x01) == 350 / 350.0
    assert spacemouse.convert(0xFF, 0x7F) == 1.0  # clipped
    assert spacemouse.convert(0x00, 0x80) == -1.0


def test_prefers_the_direct_spacemouse_over_the_receiver():
    receiver = {"product_id": spacemouse.SPACEMOUSE_PRODUCT_ID, "product_string": "Receiver"}
    direct = {"product_id": spacemouse.SPACEMOUSE_DIRECT_USB_PRODUCT_ID, "product_string": "x"}
    named = {"product_id": 0x1234, "product_string": "SpaceMouse Wireless"}
    assert spacemouse._prefer_direct_spacemouse([receiver, direct]) is direct
    assert spacemouse._prefer_direct_spacemouse([receiver, named]) is named
    assert spacemouse._prefer_direct_spacemouse([receiver]) is None


def test_device_path_normalization():
    assert spacemouse._normalize_device_path(None) is None
    assert spacemouse._normalize_device_path("/dev/hidraw3") == b"/dev/hidraw3"
    assert spacemouse._normalize_device_path(b"/dev/hidraw3") == b"/dev/hidraw3"


def test_control_properties_without_a_device():
    device = RobosuiteSpaceMouse.__new__(RobosuiteSpaceMouse)
    device._control = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    device.gripper_closed = False
    np.testing.assert_allclose(device.control, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert device.control_gripper == 0
    device.gripper_closed = True
    assert device.control_gripper == 1
    device.reset_gripper()
    assert device.control_gripper == 0
