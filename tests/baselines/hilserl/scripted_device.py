"""A deterministic stand-in for the SpaceMouse device, for the takeover and actor tests."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mulligan.baselines.hilserl.takeover import DeviceReading


class ScriptedDevice:
    """Deterministic stand-in: a list of (control, gripper_closed) readings, one per call.

    ``gripper_closed`` may be ``None`` to keep the toggle where ``set_gripper`` last put it,
    which is how the real device behaves between button presses.
    """

    def __init__(self, readings: Sequence[tuple]):
        self._readings = list(readings)
        self.calls = 0
        self.gripper_closed = False

    def read(self) -> DeviceReading:
        if self.calls < len(self._readings):
            c, g = self._readings[self.calls]
        else:
            c, g = np.zeros(6), None
        self.calls += 1
        if g is not None:
            self.gripper_closed = bool(g)
        return DeviceReading(
            control=np.asarray(c, dtype=np.float64), gripper_closed=self.gripper_closed
        )

    def reset_gripper(self) -> None:
        self.gripper_closed = False

    def set_gripper(self, closed: bool) -> None:
        self.gripper_closed = bool(closed)

    def close(self) -> None:
        pass
