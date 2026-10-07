"""
Configuration system for training (draccus dataclasses).

Usage:
    from mulligan.configs import TrainConfig

    @draccus.wrap()
    def train(cfg: TrainConfig):
        ...

The policy config is polymorphic: ``--policy.type idql`` or ``idql_divl``.
"""

from mulligan.configs.dataset import DatasetConfig
from mulligan.configs.env import EnvConfig
from mulligan.configs.eval import EvalConfig
from mulligan.configs.policy import IDQLDIVLConfig, IDQLPolicyConfig, PolicyConfig
from mulligan.configs.system import SystemConfig
from mulligan.configs.train import TrainConfig
from mulligan.configs.training import TrainingConfig
from mulligan.configs.wandb import WandBConfig

__all__ = [
    "TrainConfig",
    "DatasetConfig",
    "PolicyConfig",
    "IDQLPolicyConfig",
    "IDQLDIVLConfig",
    "TrainingConfig",
    "EvalConfig",
    "EnvConfig",
    "WandBConfig",
    "SystemConfig",
]
