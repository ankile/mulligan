"""Human takeover for the actor: spacemouse -> 7-D action, the intervention
rule, latch, idle filter, and operator keys.

Device: ``mulligan.teleop.spacemouse.RobosuiteSpaceMouse``
unchanged. Action mapping = the teleop collector's (``mulligan.sim.collect.teleop``,
the mapping the R0 demos were recorded with; gripper toggle -> ``+1 = close``),
NOT the DAgger collector's sign-inverted one.

Intervention rule (hil-serl ``SpacemouseIntervention`` + a latch): the human
drives while any raw puck axis exceeds ``deadzone`` or the gripper toggle
changed, plus ``latch_s`` seconds after the last deflection so a bout is one
contiguous segment; ``h`` force-holds the takeover.

Idle filter (the R0 demos' active filter, applied ONLY inside human bouts):
a bout step with no puck input, no gripper change/settling and a still nut is
not recorded and does not advance the episode clock; the env is still stepped
with a zero arm command + the gripper command so physics never freezes.

Keys (pynput):  h hold takeover  |  r redo the current/last bout  |
x end episode as failure  |  p pause  |  t toggle real-time pacing  |
q graceful stop (finish this episode, flush, exit). On macOS only keys typed
while the MuJoCo viewer window is focused count (see ``KeyQueue``); on Linux
the listener is global.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# Teleop mapping constants (mulligan.sim.collect.teleop)
TELEOP_BASE_SCALE = 0.0055
POS_POSTSCALE = 125.0
ROT_POSTSCALE = 50.0
DEFAULT_DEADZONE = 0.025  # raw puck units in [-1, 1] (teleop `has_input` threshold)
DEFAULT_LATCH_S = 0.5
DEFAULT_OBJECT_VEL_THRESH = 0.01  # m/s (teleop --object-vel-threshold default)
GRIPPER_SETTLE_STEPS = 6  # keep recording this many steps after a gripper toggle


def teleop_action(
    control: np.ndarray, gripper_closed: bool, pos_sensitivity: float, rot_sensitivity: float
) -> np.ndarray:
    """6-D raw puck control + gripper toggle -> robosuite 7-D action (teleop mapping)."""
    control = np.asarray(control, dtype=np.float64).reshape(6)
    dpos = control[:3] * TELEOP_BASE_SCALE * pos_sensitivity
    raw_drot = control[3:6] * TELEOP_BASE_SCALE * rot_sensitivity  # [roll, pitch, yaw]
    drot = raw_drot[[1, 0, 2]].copy()  # [pitch, roll, yaw]
    drot[2] = -drot[2]  # [pitch, roll, -yaw]
    dpos = np.clip(dpos * POS_POSTSCALE, -1, 1)
    drot = np.clip(drot * ROT_POSTSCALE, -1, 1)
    return np.concatenate([dpos, drot, [1.0 if gripper_closed else -1.0]]).astype(np.float32)


@dataclass
class DeviceReading:
    control: np.ndarray  # raw 6-D in [-1, 1]
    gripper_closed: bool


class SpacemouseDevice:
    """Thin adapter over RobosuiteSpaceMouse (hid / joystick fallback)."""

    def __init__(self, pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.0):
        from mulligan.teleop.spacemouse import RobosuiteSpaceMouse

        self.device = RobosuiteSpaceMouse(
            pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity
        )
        self.device.start_control()

    def read(self) -> DeviceReading:
        return DeviceReading(
            control=np.asarray(self.device.control, dtype=np.float64),
            gripper_closed=bool(self.device.control_gripper == 1),
        )

    def reset_gripper(self) -> None:
        self.device.reset_gripper()

    def set_gripper(self, closed: bool) -> None:
        """Overwrite the device's toggle (it otherwise remembers the last button press forever)."""
        self.device.gripper_closed = bool(closed)

    def close(self) -> None:
        self.device.close()


def _darwin_focused_listener(base, on_press, wanted: set[str]):
    """pynput listener for macOS: key-down/up only, and only keys addressed to this process.

    Key-down/up only: pynput's default tap also takes flags-changed and system-defined
    events and converts the latter to an NSEvent on the tap thread; on macOS 26 a Caps
    Lock press then trips HIToolbox's main-queue assert and SIGTRAPs the whole process.

    This-process only: the CGEvent carries the pid of the app it is delivered to, so a
    key counts only while the MuJoCo viewer window is focused, and the wanted keys are
    swallowed there (active tap) so the viewer's own key bindings do not fire. Typing in a chat or a
    terminal does not drive the actor; wanted keys typed elsewhere are logged
    (rate-limited) so a wrong filter cannot fail silently.
    """
    import os

    import Quartz

    pid = os.getpid()
    last_ignored = [0.0]

    class FocusedKeyListener(base):
        _EVENTS = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) | Quartz.CGEventMaskBit(
            Quartz.kCGEventKeyUp
        )

        def _handle_message(self, proxy, event_type, event, refcon, injected):
            target = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventTargetUnixProcessID)
            if target == pid:
                super()._handle_message(proxy, event_type, event, refcon, injected)
                return
            if event_type != Quartz.kCGEventKeyDown:
                return
            try:
                key = self._event_to_key(event)
            except IndexError:  # pynput's documented raise for an invalid keycode
                return
            ch = getattr(key, "char", None)
            if ch in wanted and time.time() - last_ignored[0] > 5.0:
                last_ignored[0] = time.time()
                print(
                    f"[keys] ignored '{ch}': typed into another app (pid {target}); "
                    "click the MuJoCo viewer window first",
                    flush=True,
                )

    def intercept(listener, event_type, event):
        # Swallow wanted keys addressed to this process so the MuJoCo viewer's own
        # bindings never fire (q/x/h/t/r/p all toggle render flags); all else passes.
        target = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventTargetUnixProcessID)
        if target != pid or event_type not in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
            return event
        try:
            key = listener._event_to_key(event)
        except IndexError:  # pynput's documented raise for an invalid keycode
            return event
        return None if getattr(key, "char", None) in wanted else event

    listener = FocusedKeyListener(
        on_press=on_press, darwin_intercept=lambda et, ev: intercept(listener, et, ev)
    )
    return listener


class KeyQueue:
    """Keyboard listener (pynput) feeding a set of pending keys.

    macOS: only keys typed while the MuJoCo viewer window is focused, and the tap never
    sees modifier / system-defined events (``_darwin_focused_listener``). Linux: pynput's
    global listener.
    """

    def __init__(self, keys: Sequence[str] = ("h", "r", "x", "p", "t", "q")):
        from pynput import keyboard

        self._wanted = set(keys)
        self._pending: list[str] = []
        if sys.platform == "darwin":
            self._listener = _darwin_focused_listener(
                keyboard.Listener, self._on_press, self._wanted
            )
        else:
            self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key) -> None:
        ch = getattr(key, "char", None)
        if ch and ch in self._wanted:
            self._pending.append(ch)

    def drain(self) -> list[str]:
        keys, self._pending = self._pending, []
        return keys

    def close(self) -> None:
        self._listener.stop()


class ScriptedKeys:
    def __init__(self, presses: Optional[dict[int, Sequence[str]]] = None):
        self.presses = presses or {}
        self.calls = 0

    def drain(self) -> list[str]:
        keys = list(self.presses.get(self.calls, ()))
        self.calls += 1
        return keys

    def close(self) -> None:
        pass


@dataclass
class TakeoverDecision:
    intervening: bool
    action: Optional[np.ndarray]  # human 7-D action when intervening
    idle: bool  # step is an idle bout step: execute no-op arm, do not record
    exec_action: Optional[np.ndarray]  # action to execute when idle (zero arm + gripper)
    bout_started: bool  # this step begins a new bout


@dataclass
class TakeoverState:
    hold: bool = False
    last_input_time: float = -1.0
    prev_gripper: Optional[bool] = None
    gripper_settle: int = 0
    in_bout: bool = False
    stats: dict = field(default_factory=lambda: {"bouts": 0, "human_steps": 0, "idle_dropped": 0})


class SpacemouseTakeover:
    def __init__(
        self,
        device,
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        deadzone: float = DEFAULT_DEADZONE,
        latch_s: float = DEFAULT_LATCH_S,
        object_vel_thresh: float = DEFAULT_OBJECT_VEL_THRESH,
        idle_filter: bool = True,
        clock=time.monotonic,
    ):
        self.device = device
        self.pos_sensitivity = float(pos_sensitivity)
        self.rot_sensitivity = float(rot_sensitivity)
        self.deadzone = float(deadzone)
        self.latch_s = float(latch_s)
        self.object_vel_thresh = float(object_vel_thresh)
        self.idle_filter = bool(idle_filter)
        self.clock = clock
        self.state = TakeoverState()

    def set_hold(self, hold: bool) -> None:
        self.state.hold = bool(hold)

    def toggle_hold(self) -> bool:
        self.state.hold = not self.state.hold
        return self.state.hold

    def end_episode(self) -> None:
        """Reset per-episode bout state (the toggle is re-synced to the policy on the next step)."""
        self.state.in_bout = False
        self.state.gripper_settle = 0
        self.state.last_input_time = -1.0

    def decide(self, nut_speed: float, gripper_closed_now: bool) -> TakeoverDecision:
        """``gripper_closed_now`` is the gripper command executed on the previous step."""
        st = self.state
        reading = self.device.read()
        now = self.clock()
        puck_active = bool(np.any(np.abs(reading.control) > self.deadzone))
        if not st.in_bout:
            # Between bouts the policy owns the gripper. The device toggle only remembers the
            # last button press, so without this a bout that starts from puck motion carried a
            # stale toggle in and flipped the gripper, mostly to "close". Keep the toggle
            # equal to the policy's command, and read a press as "flip the gripper the policy
            # has", not "flip the stale toggle".
            pressed = st.prev_gripper is not None and reading.gripper_closed != st.prev_gripper
            synced = (not gripper_closed_now) if pressed else bool(gripper_closed_now)
            self.device.set_gripper(synced)
            reading = DeviceReading(control=reading.control, gripper_closed=synced)
            st.prev_gripper = bool(gripper_closed_now)
        if st.prev_gripper is None:
            st.prev_gripper = reading.gripper_closed
        gripper_changed = reading.gripper_closed != st.prev_gripper
        st.prev_gripper = reading.gripper_closed
        if gripper_changed:
            st.gripper_settle = GRIPPER_SETTLE_STEPS
        has_input = puck_active or gripper_changed
        if has_input:
            st.last_input_time = now
        latched = st.last_input_time >= 0 and (now - st.last_input_time) <= self.latch_s
        intervening = has_input or latched or st.hold or st.gripper_settle > 0
        if not intervening:
            st.in_bout = False
            st.gripper_settle = 0
            return TakeoverDecision(
                intervening=False, action=None, idle=False, exec_action=None, bout_started=False
            )
        bout_started = not st.in_bout
        if bout_started:
            st.stats["bouts"] += 1
        st.in_bout = True
        action = teleop_action(
            reading.control, reading.gripper_closed, self.pos_sensitivity, self.rot_sensitivity
        )
        settling = st.gripper_settle > 0
        if settling and not gripper_changed:  # GRIPPER_SETTLE_STEPS steps AFTER the toggle step
            st.gripper_settle -= 1
        object_moving = nut_speed > self.object_vel_thresh
        idle = self.idle_filter and not has_input and not settling and not object_moving
        if idle:
            st.stats["idle_dropped"] += 1
            exec_action = np.concatenate([np.zeros(6), [action[6]]]).astype(np.float32)
            return TakeoverDecision(
                intervening=True,
                action=None,
                idle=True,
                exec_action=exec_action,
                bout_started=bout_started,
            )
        st.stats["human_steps"] += 1
        return TakeoverDecision(
            intervening=True, action=action, idle=False, exec_action=None, bout_started=bout_started
        )
