"""Golden test for ``get_environment_state_names``.

The names are written into dataset metadata (``observation.environment_state``), so the
exact strings, their order and the shape checks are part of the released data format.
``data/env_state_names_golden.json`` holds the expected names over a cross product of the
Square environment names and shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mulligan.data.env_state_names import get_environment_state_names


GOLDEN = json.loads((Path(__file__).parent / "data" / "env_state_names_golden.json").read_text())
CASES = [(env, shape) for env in GOLDEN["env_names"] for shape in GOLDEN["shapes"]]
ERROR = (
    "Environment state names not defined for environment '{env}' with shape {shape}. "
    "Please add a case for this environment in mulligan/data/env_state_names.py to ensure "
    "proper semantic naming of environment state dimensions."
)


@pytest.mark.parametrize(("env_name", "shape"), CASES)
def test_matches_golden(env_name: str, shape: int) -> None:
    schema_id = GOLDEN["supported"][env_name].get(str(shape))
    if schema_id is None:
        with pytest.raises(ValueError) as exc_info:
            get_environment_state_names(env_name, shape)
        assert str(exc_info.value) == ERROR.format(env=env_name, shape=shape)
        return
    assert get_environment_state_names(env_name, shape) == GOLDEN["schemas"][schema_id]


def test_golden_covers_every_schema() -> None:
    used = {sid for per_env in GOLDEN["supported"].values() for sid in per_env.values()}
    assert used == set(GOLDEN["schemas"])
    assert len(GOLDEN["schemas"]) == 2
    for names in GOLDEN["schemas"].values():
        assert len(names) == len(set(names))


def test_returns_fresh_list() -> None:
    first = get_environment_state_names("Square_D1", 17)
    first.append("mutated")
    assert get_environment_state_names("Square_D1", 17)[-1] == "peg_pos_z"
