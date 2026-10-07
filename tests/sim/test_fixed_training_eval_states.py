import math

import pytest

from mulligan.training.evaluation import (
    _reset_eval_batch,
    make_fixed_eval_initial_states,
)


class FakeVecEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.reset_calls = 0
        self.fixed_batches = []

    def __len__(self):
        return self.num_envs

    def reset(self):
        self.reset_calls += 1
        return [{"mode": "random"} for _ in range(self.num_envs)]

    def reset_to_eval_initial_states(self, state_specs):
        self.fixed_batches.append(list(state_specs))
        return [{"mode": "fixed", "state": spec} for spec in state_specs]


def test_fixed_square_narrow_eval_states_are_deterministic_and_in_bounds():
    states = make_fixed_eval_initial_states("NutAssemblySquare", 400, 20260524)
    states_again = make_fixed_eval_initial_states("NutAssemblySquare", 400, 20260524)

    assert states == states_again
    assert len(states) == 400
    assert len({state["reset_seed"] for state in states}) == 400

    first = states[0]
    assert set(first) == {"env_name", "nut", "reset_seed"}
    assert first["env_name"] == "NutAssemblySquare"

    for state in states:
        nut_x, nut_y, nut_yaw = state["nut"]
        assert -0.115 <= nut_x <= -0.11
        assert 0.11 <= nut_y <= 0.225
        assert -math.pi <= nut_yaw <= math.pi
        assert 0 <= state["reset_seed"] <= 2**32 - 1


def test_fixed_eval_states_accept_public_task_names():
    for task, env_name in (("square_narrow", "NutAssemblySquare"), ("square_broad", "Square_D1")):
        states = make_fixed_eval_initial_states(task, 8, 3)
        assert states == make_fixed_eval_initial_states(env_name, 8, 3)
        assert {state["env_name"] for state in states} == {env_name}


def test_fixed_square_broad_eval_states_are_deterministic_valid_and_collision_free():
    states = make_fixed_eval_initial_states("Square_D1", 400, 20260524)
    states_again = make_fixed_eval_initial_states("Square_D1", 400, 20260524)

    assert states == states_again
    assert len(states) == 400
    assert len({state["reset_seed"] for state in states}) == 400

    first = states[0]
    assert set(first) == {"env_name", "nut", "peg", "reset_seed"}
    assert first["env_name"] == "Square_D1"

    for state in states:
        nut_x, nut_y, nut_yaw = state["nut"]
        peg_x, peg_y = state["peg"]
        assert -0.115 <= nut_x <= 0.115
        assert -0.255 <= nut_y <= 0.255
        assert -math.pi <= nut_yaw <= math.pi
        assert -0.1 <= peg_x <= 0.3
        assert -0.2 <= peg_y <= 0.2
        assert math.dist((nut_x, nut_y), (peg_x, peg_y)) > 0.13263
        assert 0 <= state["reset_seed"] <= 2**32 - 1


def test_reset_eval_batch_uses_sequential_fixed_states_and_ignores_extra_round_slots():
    vec_env = FakeVecEnv(num_envs=4)
    states = [{"idx": idx} for idx in range(10)]

    obs = _reset_eval_batch(
        vec_env=vec_env,
        fixed_initial_states=states,
        completed_episodes=8,
        num_episodes=10,
    )

    assert vec_env.reset_calls == 0
    assert [spec["idx"] for spec in vec_env.fixed_batches[0]] == [8, 9, 9, 9]
    assert [item["mode"] for item in obs] == ["fixed"] * 4


def test_reset_eval_batch_falls_back_to_random_reset_when_disabled():
    vec_env = FakeVecEnv(num_envs=3)

    obs = _reset_eval_batch(
        vec_env=vec_env,
        fixed_initial_states=None,
        completed_episodes=0,
        num_episodes=5,
    )

    assert vec_env.reset_calls == 1
    assert vec_env.fixed_batches == []
    assert [item["mode"] for item in obs] == ["random"] * 3


def test_fixed_eval_state_validation_fails_loudly():
    with pytest.raises(ValueError, match="num_episodes must be >= 1"):
        make_fixed_eval_initial_states("NutAssemblySquare", 0, 1)
    with pytest.raises(ValueError, match="seed must be >= 0"):
        make_fixed_eval_initial_states("NutAssemblySquare", 1, -1)
    with pytest.raises(ValueError, match="Unsupported sim task"):
        make_fixed_eval_initial_states("UnsupportedTask", 1, 1)
    with pytest.raises(ValueError, match="not implemented"):
        make_fixed_eval_initial_states("Square_D0", 1, 1)
    with pytest.raises(ValueError, match="Expected at least 4 fixed eval states"):
        _reset_eval_batch(FakeVecEnv(num_envs=2), [{"idx": 0}], 0, 4)
