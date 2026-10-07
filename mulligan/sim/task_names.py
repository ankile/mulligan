"""Canonical reader-facing names for the simulation tasks.

The task keys are ``square_narrow`` / ``square_broad``; they map to the
robosuite environment IDs ``NutAssemblySquare`` / ``Square_D1`` in
:mod:`mulligan.sim.envs`.  Plot titles and prose should use the display names
here.
"""

from __future__ import annotations

from typing import Final

SIM_TASK_DISPLAY_NAMES: Final[dict[str, str]] = {
    "square_narrow": "Square-Narrow",
    "square_broad": "Square-Broad",
}


def sim_task_display_name(task_key: str) -> str:
    """Return the public task name and fail on an unknown internal key."""
    return SIM_TASK_DISPLAY_NAMES[task_key]
