"""HiL-SERL on Square-Narrow (square_narrow / NutAssemblySquare) and Square-Broad
(square_broad / MimicGen Square_D1); ``tasks.py`` holds the per-task env + demos.

Protocols: docs/baselines.md. The learner is the RLPD agent
(``mulligan.baselines.rlpd.agent``), the env is the task's
robosuite 1.5.2 env from ``mulligan.sim.envs`` (``env.py``), and the split
actor/learner pair talks over agentlace (``actor.py`` / ``learner.py``).

Everything JAX-dependent is imported lazily; ``require_jax()`` raises with the
install hint when the JAX stack is missing.
"""

from __future__ import annotations

INSTALL_HINT = "install the project environment with: uv sync --frozen"


def require_jax() -> None:
    """Fail loudly (with the install hint) when the JAX stack is missing."""
    try:
        import flax  # noqa: F401
        import jax  # noqa: F401
        import optax  # noqa: F401
        import tensorflow_probability.substrates.jax  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only in broken envs
        raise ImportError(
            f"mulligan.baselines.hilserl needs jax/flax/optax/tfp: {exc}. {INSTALL_HINT}"
        ) from exc
