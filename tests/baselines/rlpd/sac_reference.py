"""Replay the agent reference: identical synthetic batches + seeds through
``mulligan.baselines.rlpd.agent.SACLearner`` must give bitwise the parameters and update infos
recorded in ``rlpd_sac_parity.npz`` (five UTD-20 updates of the EXPO agent code, jax / flax /
optax 0.6.2 / 0.10.7 / 0.2.6, JAX CPU backend).

    JAX_PLATFORMS=cpu python -m tests.baselines.rlpd.sac_reference REF.npz

The batches are synthetic (seeded Gaussian observations, uniform actions,
Bernoulli rewards/masks) so the reference does not depend on any demo file.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

OBS_DIM, ACT_DIM = 23, 7
AGENT_KW = dict(
    hidden_dims=(256, 256, 256),
    num_qs=10,
    num_min_qs=2,
    critic_layer_norm=True,
    discount=0.99,
    tau=0.005,
    actor_lr=3e-4,
    critic_lr=3e-4,
    temp_lr=3e-4,
    init_temperature=1.0,
    backup_entropy=False,
    target_entropy=None,
)
SEED = 7
UTD = 20
BATCH = 256
N_CALLS = 5


def synthetic_batches(
    n_calls: int = N_CALLS, utd: int = UTD, batch: int = BATCH, seed: int = 123
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_calls):
        n = utd * batch
        out.append(
            dict(
                observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
                actions=rng.uniform(-0.99, 0.99, (n, ACT_DIM)).astype(np.float32),
                rewards=(rng.random(n) < 0.05).astype(np.float32),
                masks=(rng.random(n) > 0.05).astype(np.float32),
                dones=(rng.random(n) < 0.05),
                next_observations=rng.standard_normal((n, OBS_DIM)).astype(np.float32),
            )
        )
    return out


def flatten_params(agent) -> dict:
    import jax

    flat = {}
    for name in ("actor", "critic", "target_critic", "temp"):
        params = getattr(agent, name).params
        leaves, _ = jax.tree_util.tree_flatten_with_path(params)
        for path, leaf in leaves:
            flat[f"{name}/{jax.tree_util.keystr(path)}"] = np.asarray(leaf)
    return flat


def run(agent_cls, batches: list[dict]) -> tuple[dict, list[dict]]:
    agent = agent_cls.create(
        SEED, np.zeros(OBS_DIM, np.float32), np.zeros(ACT_DIM, np.float32), **AGENT_KW
    )
    infos = []
    for b in batches:
        agent, info = agent.update(b, UTD)
        infos.append({k: float(v) for k, v in info.items()})
    return flatten_params(agent), infos


def check(ref_path: str) -> int:
    from mulligan.baselines.rlpd.agent import SACLearner

    ref = np.load(ref_path)
    params, infos = run(SACLearner, synthetic_batches())
    bad = 0
    for k, v in params.items():
        r = ref[f"param/{k}"]
        if r.shape != v.shape or r.dtype != v.dtype or not np.array_equal(r, v):
            bad += 1
            print(
                f"MISMATCH {k}: shape {r.shape} vs {v.shape}, maxabs {np.abs(r.astype(np.float64) - v.astype(np.float64)).max() if r.shape == v.shape else 'n/a'}"
            )
    for i, info in enumerate(infos):
        for k, v in info.items():
            r = float(ref[f"info/{i}/{k}"])
            if r != v:
                bad += 1
                print(f"MISMATCH info[{i}][{k}]: {r} vs {v}")
    print(
        f"{'REFERENCE OK' if bad == 0 else 'REFERENCE MISMATCH'}: {len(params)} params, {len(infos)} update calls, {bad} mismatches"
    )
    return 0 if bad == 0 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("reference", metavar="REF.npz")
    return check(p.parse_args(argv).reference)


if __name__ == "__main__":
    sys.exit(main())
