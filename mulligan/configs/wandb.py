"""Weights & Biases configuration for experiment tracking."""

from dataclasses import dataclass, field


@dataclass
class WandBConfig:
    """
    Configuration for Weights & Biases logging.

    Controls whether to enable W&B, project/entity settings, and run metadata.
    """

    # Enable/disable
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable Weights & Biases logging"},
    )

    # Project settings
    project: str = field(
        default="mulligan",
        metadata={"help": "W&B project name"},
    )
    entity: str | None = field(
        default=None,
        metadata={"help": "W&B entity/team name (optional)"},
    )

    # Run metadata
    run_name: str | None = field(
        default=None,
        metadata={"help": "W&B run name (optional, auto-generated if None)"},
    )
    group: str | None = field(
        default=None,
        metadata={"help": "W&B group name for organizing runs"},
    )
    notes: str | None = field(
        default=None,
        metadata={"help": "W&B run notes/description"},
    )
    log_freq: int = field(
        default=5_000,
        metadata={"help": "Log training metrics to W&B every N steps"},
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.log_freq < 1:
            raise ValueError(f"log_freq must be >= 1, got {self.log_freq}")
