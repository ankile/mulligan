# Adapted from robosuite's SpaceMouse input device
# (https://github.com/ARISE-Initiative/robosuite, robosuite/devices/spacemouse.py at
# 85abee228d1c43ab1939bce33028099945d453b4). robosuite's license:
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

"""Minimal 3Dconnexion SpaceMouse driver for teleoperation (sim and real collectors).

Adapted from robosuite's SpaceMouse device. It opens the device with ``hid``
(hidapi, from the ``teleop`` extra); if that fails it tries ``easyhid`` and,
on Linux, the kernel joystick device (``/dev/input/js*``). Set
``MULLIGAN_SPACEMOUSE_DEVICE_PATH`` to open a specific HID path.

Check a device with ``python -m mulligan.teleop.spacemouse``.
"""

import glob
import os
import select
import struct
import threading
import time

import numpy as np

try:
    from easyhid import Enumeration as EasyHIDEnumeration
    from easyhid import HIDException as EasyHIDException
except ModuleNotFoundError:
    # easyhid is an optional second HID backend.
    EasyHIDEnumeration = None
    EasyHIDException = None

# SpaceMouse vendor/product IDs
SPACEMOUSE_VENDOR_ID = 0x256F  # 3Dconnexion
SPACEMOUSE_PRODUCT_ID = 0xC652  # 3Dconnexion Universal Receiver (SpaceMouse Wireless)
SPACEMOUSE_DIRECT_USB_PRODUCT_ID = 0xC63A  # SpaceMouse Wireless BT over direct USB
SPACEMOUSE_DEVICE_PATH_ENV = "MULLIGAN_SPACEMOUSE_DEVICE_PATH"
DIRECT_SPACEMOUSE_PRODUCT_IDS = {
    SPACEMOUSE_DIRECT_USB_PRODUCT_ID,
}

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_STRUCT = struct.Struct("IhBB")


def _hid():
    try:
        import hid
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The SpaceMouse needs hidapi: install the teleop extra (uv sync --extra teleop)"
        ) from exc
    return hid


def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))


def _enumerate_spacemouse_devices(vendor_id):
    try:
        return [device for device in _hid().enumerate() if device.get("vendor_id") == vendor_id]
    except OSError as err:
        # Enumeration can fail without device permissions; opening by ID may still work.
        print(f"Warning: HID enumeration failed ({err})")
        return []


def _format_device_info(device):
    manufacturer = device.get("manufacturer_string", "")
    product = device.get("product_string", "")
    vendor_id = device.get("vendor_id")
    product_id = device.get("product_id")
    path = device.get("path")
    return (
        f"vendor=0x{vendor_id:04X} product=0x{product_id:04X} "
        f"manufacturer={manufacturer!s} product={product!s} path={path!s}"
    )


def _find_spacemouse_joystick_path(vendor_id=SPACEMOUSE_VENDOR_ID):
    """Return a Linux joydev path for a 3Dconnexion SpaceMouse if available."""
    for path in sorted(glob.glob("/dev/input/by-id/*3Dconnexion*joystick")):
        if "event-joystick" in os.path.basename(path):
            continue
        return os.path.realpath(path)

    for js_path in sorted(glob.glob("/sys/class/input/js*")):
        device_dir = os.path.join(js_path, "device")
        vendor_path = os.path.join(device_dir, "id", "vendor")
        try:
            with open(vendor_path) as f:
                detected_vendor = int(f.read().strip(), 16)
        except FileNotFoundError:
            continue
        if detected_vendor == vendor_id:
            js_name = os.path.basename(js_path)
            return f"/dev/input/{js_name}"


def _normalize_device_path(path):
    if path is None:
        return None
    if isinstance(path, (bytes, bytearray)):
        return bytes(path)
    return os.fsencode(path)


def _prefer_direct_spacemouse(devices):
    """Prefer the actual SpaceMouse interface over a passive universal receiver."""
    for device in devices:
        if device.get("product_id") in DIRECT_SPACEMOUSE_PRODUCT_IDS:
            return device
        product = str(device.get("product_string", "")).lower()
        if "spacemouse" in product and "receiver" not in product:
            return device
    return None


class RobosuiteSpaceMouse:
    """
    Minimal SpaceMouse driver for Robosuite teleoperation.

    This class provides direct HID access to the SpaceMouse without
    requiring the full Robosuite Device interface.

    Args:
        vendor_id (int): USB vendor ID
        product_id (int): USB product ID
        pos_sensitivity (float): Position control sensitivity multiplier
        rot_sensitivity (float): Rotation control sensitivity multiplier
    """

    def __init__(
        self,
        vendor_id=SPACEMOUSE_VENDOR_ID,
        product_id=SPACEMOUSE_PRODUCT_ID,
        pos_sensitivity=1.0,
        rot_sensitivity=1.0,
        device_path=None,
    ):
        print("Opening SpaceMouse device")
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = None
        self._joystick_fd = None
        self._joystick_path = None
        self._backend = "hid"
        if device_path is None:
            device_path = os.environ.get(SPACEMOUSE_DEVICE_PATH_ENV)
        device_path = _normalize_device_path(device_path)

        hid = _hid()

        def _open_by_ids(vendor=None, product=None):
            self.device = hid.device()
            self.device.open(
                self.vendor_id if vendor is None else vendor,
                self.product_id if product is None else product,
            )

        def _open_by_path(path):
            self.device = hid.device()
            self.device.open_path(path)

        def _open_enumerated_device(device):
            vendor = device.get("vendor_id", self.vendor_id)
            product = device.get("product_id", self.product_id)
            path = device.get("path")
            if path:
                try:
                    _open_by_path(path)
                except OSError:
                    _open_by_ids(vendor, product)
            else:
                _open_by_ids(vendor, product)
            self.vendor_id = vendor
            self.product_id = product

        def _open_with_easyhid(path=None):
            if EasyHIDEnumeration is None:
                return False

            enum = EasyHIDEnumeration()
            candidates = enum.find(vid=self.vendor_id, pid=self.product_id)
            if path is not None:
                path_str = path.decode() if isinstance(path, (bytes, bytearray)) else str(path)
                candidates = [d for d in candidates if d.path == path_str]

            for candidate in candidates:
                try:
                    candidate.open()
                    # Match hid.device.read() behavior used in the loop below.
                    candidate.set_nonblocking(False)
                    self.device = candidate
                    self._backend = "easyhid"
                    self.vendor_id = candidate.vendor_id
                    self.product_id = candidate.product_id
                    return True
                except EasyHIDException as err:
                    print(f"easyhid could not open {candidate.path}: {err}")
                    continue
            return False

        def _open_with_joystick():
            joystick_path = _find_spacemouse_joystick_path(self.vendor_id)
            if joystick_path is None:
                return False

            self._joystick_fd = os.open(joystick_path, os.O_RDONLY | os.O_NONBLOCK)
            self._joystick_path = joystick_path
            self._backend = "joystick"
            return True

        opened = False
        devices = []
        open_error = None
        try:
            devices = _enumerate_spacemouse_devices(self.vendor_id)
            preferred_device = _prefer_direct_spacemouse(devices) if device_path is None else None
            if device_path:
                _open_by_path(device_path)
            elif (
                preferred_device is not None
                and preferred_device.get("product_id") in DIRECT_SPACEMOUSE_PRODUCT_IDS
                and os.name == "posix"
            ):
                print("Direct USB SpaceMouse detected; trying Linux joystick backend")
                opened = _open_with_joystick()
                if not opened:
                    _open_enumerated_device(preferred_device)
            elif preferred_device is not None:
                _open_enumerated_device(preferred_device)
            else:
                _open_by_ids()
            opened = True
        except OSError as err:
            open_error = err
            if devices and device_path:
                for device in devices:
                    if _normalize_device_path(device.get("path")) != device_path:
                        continue
                    try:
                        _open_enumerated_device(device)
                        opened = True
                        break
                    except OSError:
                        continue

            if devices and not opened and not device_path:
                for device in devices:
                    path = device.get("path")
                    if not path:
                        continue
                    try:
                        _open_enumerated_device(device)
                        opened = True
                        break
                    except OSError:
                        continue

            if not opened:
                print("Falling back to easyhid backend")
                opened = _open_with_easyhid(device_path)

            if not opened and os.name == "posix":
                print("Falling back to Linux joystick backend")
                try:
                    opened = _open_with_joystick()
                except OSError as err:
                    open_error = err

            if not opened:
                print(
                    "Failed to open SpaceMouse device. "
                    "Consider killing other processes that may be using the device:\n"
                    "  killall 3DconnexionHelper\n"
                    "Also ensure Input Monitoring permissions are granted (macOS) "
                    "or proper USB permissions (Linux)."
                )
                if devices:
                    print("Detected 3Dconnexion devices:")
                    for device in devices:
                        print(f"  - {_format_device_info(device)}")
                if EasyHIDEnumeration is None:
                    print("easyhid not available. Install with: pip install easyhid")
                raise open_error if open_error is not None else OSError("open failed")

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        if self._backend == "hid":
            print(f"Manufacturer: {self.device.get_manufacturer_string()}")
            print(f"Product: {self.device.get_product_string()}")
        elif self._backend == "easyhid":
            print(f"Manufacturer: {self.device.get_manufacture_string()}")
            print(f"Product: {self.device.get_product_string()}")
        else:
            print("Manufacturer: 3Dconnexion")
            print(f"Product: Linux joystick device {self._joystick_path}")

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        # Button state
        self.gripper_closed = False

        # Control state
        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._enabled = False

        # Launch listener thread
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def _display_controls():
        """Print control instructions."""
        print("\nSpaceMouse Controls:")
        print("  Left button (click)     - toggle gripper open/close")
        print("  Move mouse laterally    - move arm horizontally in x-y plane")
        print("  Move mouse vertically   - move arm vertically")
        print("  Twist mouse about axis  - rotate arm about corresponding axis")
        print("  Control+C               - quit")
        print()

    def start_control(self):
        """Enable control."""
        self._enabled = True

    def _run(self):
        """Listener thread that continuously reads from the device."""
        if self._backend == "joystick":
            self._run_joystick()
            return

        t_last_click = -1
        read_errors = (OSError, ValueError)
        if self._backend == "easyhid":
            read_errors = read_errors + (EasyHIDException,)

        while True:
            try:
                if self._backend == "easyhid":
                    d = self.device.read(13, timeout=50)
                else:
                    d = self.device.read(13)
            except read_errors:
                # The device was closed (close()) or disconnected.
                break

            if d and self._enabled:
                if self.product_id == 50741:
                    # Older SpaceMouse model - separate messages for pos/rot
                    if d[0] == 1:  # Position data
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0
                    elif d[0] == 2:  # Rotation data
                        self.roll = convert(d[1], d[2])
                        self.pitch = convert(d[3], d[4])
                        self.yaw = convert(d[5], d[6])
                        self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
                else:
                    # Modern SpaceMouse - all 6-DOF in one message
                    if d[0] == 1:
                        self.y = convert(d[1], d[2])
                        self.x = convert(d[3], d[4])
                        self.z = convert(d[5], d[6]) * -1.0
                        self.roll = convert(d[7], d[8])
                        self.pitch = convert(d[9], d[10])
                        self.yaw = convert(d[11], d[12])
                        self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

                # Button handling
                if d[0] == 3:
                    # Left button - toggle gripper
                    if d[1] == 1:
                        t_click = time.time()
                        elapsed_time = t_click - t_last_click
                        t_last_click = t_click
                        # Debounce
                        if elapsed_time > 0.3 or elapsed_time < 0:
                            self.gripper_closed = not self.gripper_closed

    def _run_joystick(self):
        """Listener thread for Linux /dev/input/js* SpaceMouse events."""
        t_last_click = -1
        axis_specs = {
            0: ("y", 1.0),
            1: ("x", 1.0),
            2: ("z", -1.0),
            3: ("roll", 1.0),
            4: ("pitch", 1.0),
            5: ("yaw", 1.0),
        }

        while True:
            fd = self._joystick_fd
            if fd is None:
                break

            try:
                readable, _, _ = select.select([fd], [], [], 0.05)
            except (OSError, ValueError):
                break
            if not readable:
                continue

            try:
                data = os.read(fd, JS_EVENT_STRUCT.size * 32)
            except BlockingIOError:
                continue
            except OSError:
                break

            complete_length = len(data) - (len(data) % JS_EVENT_STRUCT.size)
            for offset in range(0, complete_length, JS_EVENT_STRUCT.size):
                _timestamp, value, event_type, number = JS_EVENT_STRUCT.unpack_from(data, offset)
                event_type = event_type & ~JS_EVENT_INIT

                if not self._enabled:
                    continue

                if event_type == JS_EVENT_AXIS:
                    if number not in axis_specs:
                        continue
                    axis_attr, axis_scale = axis_specs[number]
                    scaled_value = float(np.clip(axis_scale * value / 32767.0, -1.0, 1.0))
                    if abs(scaled_value) < 0.002:
                        scaled_value = 0.0
                    setattr(self, axis_attr, scaled_value)
                    self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
                elif event_type == JS_EVENT_BUTTON and number == 0 and value == 1:
                    t_click = time.time()
                    elapsed_time = t_click - t_last_click
                    t_last_click = t_click
                    if elapsed_time > 0.3 or elapsed_time < 0:
                        self.gripper_closed = not self.gripper_closed

    @property
    def control(self):
        """
        Get current 6-DOF control values.

        Returns:
            np.array: [x, y, z, roll, pitch, yaw]
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Get gripper state.

        Returns:
            int: 1 for closed, 0 for open
        """
        return 1 if self.gripper_closed else 0

    def reset_gripper(self):
        """Reset gripper to open position."""
        self.gripper_closed = False

    def close(self):
        """Close the device connection."""
        self._enabled = False
        if self.device is not None:
            self.device.close()
        if self._joystick_fd is not None:
            os.close(self._joystick_fd)
            self._joystick_fd = None


if __name__ == "__main__":
    # Simple test
    space_mouse = RobosuiteSpaceMouse()
    space_mouse.start_control()

    print("Testing SpaceMouse. Move the device and press buttons. Ctrl+C to stop.\n")

    try:
        while True:
            control = space_mouse.control
            gripper = space_mouse.control_gripper
            print(f"Control: {control}  Gripper: {gripper}", end="\r")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        space_mouse.close()
