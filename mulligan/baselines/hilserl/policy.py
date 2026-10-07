"""CPU-side actor policy: the learner's actor network + params, sampling with
exactly the learner's ``sample_actions`` / ``eval_actions`` code paths
(``mulligan.baselines.rlpd.agent.base``). Used by the actor process and the eval workers."""

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np

from mulligan.baselines.hilserl import require_jax
from mulligan.baselines.hilserl.config import HilSerlConfig
from mulligan.baselines.hilserl.obs import ACTION_DIM


class ActorPolicy:
    def __init__(self, cfg: HilSerlConfig, rng_seed: int, params: Optional[dict] = None):
        require_jax()
        import jax

        from mulligan.baselines.rlpd.agent.distributions import TanhNormal
        from mulligan.baselines.rlpd.agent.mlp import MLP

        actor_base_cls = partial(MLP, hidden_dims=cfg.hidden_dims, activate_final=True)
        self.actor_def = TanhNormal(actor_base_cls, ACTION_DIM)
        self.apply_fn = self.actor_def.apply
        self.rng = jax.random.PRNGKey(int(rng_seed))
        if params is None:
            _, init_key = jax.random.split(self.rng)
            params = self.actor_def.init(init_key, np.zeros(cfg.task_spec.obs_dim, np.float32))[
                "params"
            ]
        self.params = params
        self.version = -1

    def set_params(self, params: dict, version: int) -> None:
        self.params = params
        self.version = int(version)

    def sample(self, obs: np.ndarray) -> np.ndarray:
        from mulligan.baselines.rlpd.agent.base import _sample_actions

        actions, self.rng = _sample_actions(
            self.rng, self.apply_fn, self.params, np.asarray(obs, np.float32)
        )
        return np.asarray(actions, dtype=np.float32)

    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        from mulligan.baselines.rlpd.agent.base import _eval_actions

        return np.asarray(
            _eval_actions(self.apply_fn, self.params, np.asarray(obs, np.float32)), dtype=np.float32
        )


def actor_params_to_numpy(params) -> dict:
    import jax

    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), params)
