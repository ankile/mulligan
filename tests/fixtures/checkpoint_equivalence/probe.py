"""Fixed-input probe of a loaded IDQL/DIVL agent (golden regression).

This file does not import ``mulligan``: it only calls ``IDQLPolicy`` methods on an already
loaded policy, so the same code records the references and checks them.
"""

from __future__ import annotations

import torch

BATCH = 4
INPUT_SEED = 20260930
SAMPLING_SEED = 1234
NUM_THREADS = 4


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(NUM_THREADS)


def make_inputs(state_dim: int, robot_state_dim: int, chunk_size: int, action_dim: int):
    g = torch.Generator().manual_seed(INPUT_SEED)
    robot = torch.randn(BATCH, robot_state_dim, generator=g)
    env = torch.randn(BATCH, state_dim - robot_state_dim, generator=g)
    action = torch.rand(BATCH, chunk_size * action_dim, generator=g) * 2.0 - 1.0
    return robot, env, action


@torch.no_grad()
def probe(policy) -> dict[str, torch.Tensor]:
    """Return every output compared by the equivalence test (all CPU float tensors)."""
    policy.eval()
    # Sim checkpoints do not record robot_state_dim; the actor's state feature has it.
    robot_dim = policy.actor.config.robot_state_feature.shape[0]
    robot, env, action = make_inputs(
        policy.state_dim, robot_dim, policy.chunk_size, policy.action_dim
    )
    state = torch.cat([robot, env], dim=-1)
    out: dict[str, torch.Tensor] = {
        "input.robot_state": robot,
        "input.env_state": env,
        "input.action": action,
        "critic.q_per_network": torch.stack([c(state, action) for c in policy.critics]),
        "critic.target_q_per_network": torch.stack(
            [c(state, action) for c in policy.target_critics]
        ),
        "critic.q_aggregated": policy.compute_q_value(robot, action, env_state=env),
        "value.v": policy.compute_v_value(state),
    }
    if hasattr(policy.value, "num_atoms"):
        out["value.logits"] = policy.value(state)
        out["value.target_logits"] = policy.target_value(state)
    else:
        out["value.target_v"] = policy.target_value(state)

    obs = {"observation.state": robot, "observation.environment_state": env}
    torch.manual_seed(SAMPLING_SEED)
    sampled, q_values, _, _ = policy._sample_and_evaluate_actions(obs)
    out["bon.sampled_actions"] = sampled
    out["bon.q_values"] = q_values

    torch.manual_seed(SAMPLING_SEED)
    policy.reset()
    steps, qs, vs = [], [], []
    for _ in range(policy.chunk_size):
        a, q, v = policy.select_action(obs, return_values=True)
        steps.append(a)
        qs.append(q)
        vs.append(v)
    out["select_action.chunk"] = torch.stack(steps, dim=1)
    out["select_action.q"] = torch.stack(qs, dim=1)
    out["select_action.v"] = torch.stack(vs, dim=1)
    return {k: v.detach().cpu().contiguous() for k, v in out.items()}
