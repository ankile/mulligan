"""
Agents: IDQL / DIVL (diffusion actor + IQL critics with best-of-N reranking).

Agents take observation dicts with separate keys:
- "observation.state": Robot state (normalized)
- "observation.environment_state": Environment state (normalized)
and concatenate them internally where needed.
"""

from mulligan.agents.idql import IDQLPolicy
from mulligan.agents.iql_utils import expectile_loss
from mulligan.agents.processors import (
    IDQLPreprocessor,
    IQLPostprocessor,
    make_idql_pre_post_processors,
)

__all__ = [
    "IDQLPolicy",
    "IDQLPreprocessor",
    "IQLPostprocessor",
    "make_idql_pre_post_processors",
    "expectile_loss",
]
