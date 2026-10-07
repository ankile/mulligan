"""Cross-registry consistency for canonical LeRobot task names.

These pin the spelling of task names so the three independent registries that
gate real-world collection cannot silently drift:
  - ``mulligan.data.task_names.VALID_LEROBOT_TASK_NAMES`` (the argparse allowlist that
    every collection script validates against), and
  - ``mulligan.real.lifecycle.tasks`` ``RealTaskSpec.task_name`` (the analysis-side spec).

A new real task (e.g. ``marker_d2``) must be present in the allowlist before any
collection script will accept ``--task-name``.
"""

from __future__ import annotations

import pytest

from mulligan.data.task_names import (
    VALID_LEROBOT_TASK_NAMES,
    validate_lerobot_task_name,
)


def test_validate_accepts_registered_name():
    assert validate_lerobot_task_name("marker_d2") == "marker_d2"


def test_validate_rejects_unknown_name():
    with pytest.raises(ValueError, match="must be one of"):
        validate_lerobot_task_name("definitely_not_a_task")


def test_marker_d2_is_registered():
    # Task series used for collection.
    assert "marker_d2" in VALID_LEROBOT_TASK_NAMES


def test_square_d2_is_registered():
    # The final square series.
    assert "square_d2" in VALID_LEROBOT_TASK_NAMES


def test_every_real_task_spec_name_is_a_valid_lerobot_task_name():
    # The lifecycle registry's collection/LeRobot task_name must always be a member
    # of the allowlist, or the spec describes a task no collection script can name.
    from mulligan.real.lifecycle.tasks import registered_task_specs

    for spec in registered_task_specs():
        assert spec.task_name in VALID_LEROBOT_TASK_NAMES, (
            f"RealTaskSpec {spec.name!r} has task_name {spec.task_name!r} "
            "which is not in VALID_LEROBOT_TASK_NAMES"
        )
