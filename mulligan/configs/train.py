"""Top-level training configuration composing all sub-configs."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import draccus

from mulligan.configs.dataset import DatasetConfig
from mulligan.configs.env import EnvConfig
from mulligan.configs.eval import EvalConfig
from mulligan.configs.policy import IDQLDIVLConfig, IDQLPolicyConfig, PolicyConfig
from mulligan.configs.system import SystemConfig
from mulligan.configs.training import TrainingConfig
from mulligan.configs.wandb import WandBConfig


@dataclass
class TrainConfig:
    """
    Top-level configuration for training.

    Composes all sub-configurations into a single hierarchical config.
    Use with @draccus.wrap() for automatic CLI parsing.

    Example CLI usage:
        python -m mulligan.training.train \\
            --dataset.repo_ids mulligan/sim-square-narrow-c00-teleop-sobol \\
            --policy.type idql \\
            --training.batch_size 1024 \\
            --env.name NutAssemblySquare \\
            --env.robot Panda
    """

    # Required configs
    dataset: DatasetConfig = field(
        metadata={"help": "Dataset configuration"},
    )
    env: EnvConfig = field(
        metadata={"help": "Environment configuration"},
    )

    # Policy config (polymorphic via ChoiceRegistry)
    policy: PolicyConfig = field(
        default_factory=IDQLPolicyConfig,
        metadata={"help": "Policy configuration (use --policy.type to select)"},
    )

    # Training config
    training: TrainingConfig = field(
        default_factory=TrainingConfig,
        metadata={"help": "Training hyperparameters"},
    )

    # Optional configs
    eval: EvalConfig = field(
        default_factory=EvalConfig,
        metadata={"help": "Evaluation configuration"},
    )
    wandb: WandBConfig = field(
        default_factory=WandBConfig,
        metadata={"help": "Weights & Biases configuration"},
    )
    system: SystemConfig = field(
        default_factory=SystemConfig,
        metadata={"help": "System configuration"},
    )

    # Pretrained model loading
    pretrained_artifact: str | None = field(
        default=None,
        metadata={
            "help": "Pretrained IDQL checkpoint to start from: a local directory, "
            "hf://<org>/<repo>[@<revision>]/<subfolder>, or a W&B artifact"
        },
    )

    # Runtime values (populated during training, not from CLI)
    state_dim: int | None = field(
        default=None,
        metadata={"help": "(Runtime) State dimension computed from dataset"},
    )
    action_dim: int | None = field(
        default=None,
        metadata={"help": "(Runtime) Action dimension computed from dataset"},
    )
    num_frames: int | None = field(
        default=None,
        metadata={"help": "(Runtime) Total frames in dataset"},
    )
    dataset_info: dict | None = field(
        default=None,
        metadata={"help": "(Runtime) Per-dataset metadata"},
    )

    def __post_init__(self):
        """Validate configuration."""
        # Validate sub-configs are properly instantiated
        if self.dataset is None:
            raise ValueError("dataset config is required")
        if self.env is None:
            raise ValueError("env config is required")
        if self.training.update_components == "critic_value_only":
            if not isinstance(self.policy, IDQLDIVLConfig):
                raise ValueError(
                    "training.update_components='critic_value_only' requires policy.type=idql_divl"
                )
            if self.pretrained_artifact is None:
                raise ValueError(
                    "training.update_components='critic_value_only' requires pretrained_artifact"
                )

    def generate_run_name(self) -> str:
        """Generate a unique run name based on config."""
        import hashlib

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_ids = self.dataset.get_repo_id_list()
        dataset_names = "+".join([rid.split("/")[-1] for rid in repo_ids])
        dataset_hash = hashlib.md5(dataset_names.encode()).hexdigest()[:8]

        # Get policy type from the config
        policy_type = self.policy.get_choice_name(type(self.policy))

        if len(dataset_names) <= 40:
            return f"{policy_type}_{dataset_names}_{timestamp}"
        else:
            first_dataset = repo_ids[0].split("/")[-1]
            return (
                f"{policy_type}_{first_dataset}+{len(repo_ids) - 1}more_{dataset_hash}_{timestamp}"
            )

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use draccus encoding for proper serialization
        config_dict = draccus.encode(self)

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        """Load configuration from JSON file."""
        with draccus.config_type("json"):
            return draccus.parse(cls, str(path), args=[])

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return draccus.encode(self)
