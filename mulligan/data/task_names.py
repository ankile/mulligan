"""Canonical task names used in LeRobot datasets."""

from __future__ import annotations


# LeRobot task strings of the released datasets: the simulation collectors store
# ``ENV_ROBOT`` (e.g. ``Square_D1_Panda``) or the robosuite env id, the real collectors the
# task's collection name.
VALID_LEROBOT_TASK_NAMES = frozenset(
    {
        "NutAssemblySquare",
        "NutAssemblySquare_Panda",
        "Square_D1",
        "Square_D1_Panda",
        "marker_d2",
        "routing_d2",
        "square_d2",
    }
)


def validate_lerobot_task_name(task_name: str, *, context: str = "task") -> str:
    """Return ``task_name`` if it is registered, otherwise fail loudly."""
    if task_name not in VALID_LEROBOT_TASK_NAMES:
        valid = ", ".join(sorted(VALID_LEROBOT_TASK_NAMES))
        raise ValueError(f"{context} must be one of [{valid}], got {task_name!r}")
    return task_name
