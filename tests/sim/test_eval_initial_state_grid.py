import sys
import types

import numpy as np
import pytest
import torch

from mulligan.sim.eval import grid_eval
from mulligan.training.normalization import Normalizer

ROBOT_DIM = 9
OBJECT_DIM = 14


def _obs(value: float) -> dict[str, np.ndarray]:
    return {
        "robot0_eef_pos": np.full(3, value),
        "robot0_eef_quat": np.full(4, value),
        "robot0_gripper_qpos": np.full(2, value),
        "object-state": np.full(OBJECT_DIM, value),
    }


class ScriptedVecEnv:
    """Env ``i`` succeeds on step ``success_step[i]`` (``None``: never)."""

    def __init__(self, success_step: list[int | None]):
        self.success_step = success_step
        self.steps = 0
        self.actions: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.success_step)

    def step(self, actions):
        self.steps += 1
        self.actions.append(np.stack(actions))
        infos = [{"success": step is not None and self.steps >= step} for step in self.success_step]
        rewards = [1.0 if info["success"] else 0.0 for info in infos]
        return [_obs(0.0) for _ in self.success_step], rewards, [False] * len(infos), infos


@pytest.fixture
def fake_idql(monkeypatch):
    module = types.ModuleType("mulligan.agents.idql")
    module.IDQLPolicy = type("IDQLPolicy", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.IDQLPolicy


def _normalizer() -> Normalizer:
    dim = ROBOT_DIM + OBJECT_DIM
    return Normalizer(
        state_mean=torch.ones(dim),
        state_std=torch.full((dim,), 2.0),
        action_min=torch.full((7,), -2.0),
        action_max=torch.full((7,), 2.0),
    )


def test_idql_rollouts_normalize_split_state_and_stop_per_env(fake_idql) -> None:
    seen: list[dict[str, torch.Tensor]] = []

    class Policy(fake_idql):
        def reset(self) -> None:
            pass

        def select_action(self, batch):
            seen.append(batch)
            return torch.full((batch["observation.state"].shape[0], 7), 0.5)

    vec_env = ScriptedVecEnv([2, None])
    successes, rewards, lengths = grid_eval.step_rollouts_from_obs(
        vec_env=vec_env,
        policy=Policy(),
        preprocessor=_normalizer(),
        postprocessor=_normalizer(),
        obs_list=[_obs(3.0), _obs(3.0)],
        max_steps=4,
        device="cpu",
    )

    assert successes == [True, False]
    assert lengths == [2, 4]
    assert rewards == [1.0, 0.0]
    first = seen[0]
    assert first["observation.state"].shape == (2, ROBOT_DIM)
    assert first["observation.environment_state"].shape == (2, OBJECT_DIM)
    torch.testing.assert_close(first["observation.state"], torch.ones(2, ROBOT_DIM))
    # Actions in [-1, 1] map to [action_min, action_max].
    np.testing.assert_allclose(vec_env.actions[0], np.full((2, 7), 1.0))


def test_normalizer_preprocessing_rejects_non_idql_policy(fake_idql) -> None:
    class Policy:
        def reset(self) -> None:
            pass

    with pytest.raises(TypeError, match="only defined for IDQL"):
        grid_eval.step_rollouts_from_obs(
            vec_env=ScriptedVecEnv([None]),
            policy=Policy(),
            preprocessor=_normalizer(),
            postprocessor=_normalizer(),
            obs_list=[_obs(0.0)],
            max_steps=1,
            device="cpu",
        )
