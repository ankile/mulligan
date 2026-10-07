# SpaceMouse

Human teleoperation and DAgger corrections use a 3Dconnexion SpaceMouse (the paper used a
SpaceMouse Wireless) through `mulligan.teleop.spacemouse`, a small driver adapted from
robosuite's. It is used by the simulated collectors (`mulligan.sim.collect.teleop`,
`mulligan.sim.collect.dagger`) and by the real-robot collectors (`mulligan.real.collect.*`).

## Install

The `teleop` extra adds `hidapi` (the `hid` module), `easyhid` (a fallback HID backend) and `pynput`:

```bash
uv sync --frozen --extra teleop     # main environment (simulation)
bash scripts/sync_robot_env.sh      # robot workstation: the robot project includes the extra
```

On Linux the `hidapi` wheel bundles the HID library. On macOS, `easyhid` loads it through
ctypes, so install it with Homebrew (below).

## How the driver finds the device

1. `MULLIGAN_SPACEMOUSE_DEVICE_PATH` (a hidraw path such as `/dev/hidraw3`, or the path
   `hid.enumerate()` reports) pins one device when several 3Dconnexion devices are
   attached.
2. Otherwise it enumerates 3Dconnexion devices (vendor `0x256f`) and opens the universal
   receiver (`0xc652`); a SpaceMouse Wireless plugged in directly by USB (`0xc63a`) is
   read through the Linux joystick device instead.
3. If `hid` cannot open the device, it falls back to `easyhid` (if installed) and then,
   on Linux, to the kernel joystick device (`/dev/input/by-id/*3Dconnexion*joystick`).

It fails loudly if none of these works.

## Linux

Grant your user access to the device, then re-plug it:

```bash
# /etc/udev/rules.d/99-spacemouse.rules
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0660", GROUP="plugdev"
SUBSYSTEM=="input", ATTRS{idVendor}=="256f", MODE="0660", GROUP="plugdev"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG plugdev "$USER"     # log out and in again
```

If another process holds the device (for example `spacenavd`), stop it before
collecting.

## macOS

1. `brew install hidapi` and make the library visible to Python:
   `export DYLD_LIBRARY_PATH="$(brew --prefix hidapi)/lib:$DYLD_LIBRARY_PATH"`.
2. On Apple Silicon, `easyhid` needs a patched build:
   `uv pip install git+https://github.com/bglopez/python-easyhid.git` (optional; only the
   fallback path uses it).
3. Grant Input Monitoring to your terminal: System Settings -> Privacy & Security ->
   Input Monitoring.
4. Quit the 3Dconnexion driver if it is installed: `killall 3DconnexionHelper`.
5. The simulated collectors open a MuJoCo viewer, which on macOS needs `mjpython`:
   `uv run mjpython -m mulligan.sim.collect.teleop --env NutAssemblySquare --robot Panda`.

## Check

```bash
uv run python -c "
import time
from mulligan.teleop.spacemouse import RobosuiteSpaceMouse
d = RobosuiteSpaceMouse(); d.start_control()
for _ in range(50): print(d.control, d.control_gripper); time.sleep(0.1)
"
```

Move the cap and press the buttons; the six-axis control and the gripper value change.
`uv run python -m mulligan.sim.collect.teleop --help` lists the collector
options (`--pos-sensitivity`, `--rot-sensitivity`). The robot-side key bindings (success / failure /
timeout, sub-goal marks, intervention) are printed by each collector at startup.
