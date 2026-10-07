"""Evaluation configuration for policy assessment during training."""

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """
    Configuration for policy evaluation during training.

    Controls evaluation frequency, number of episodes, and video saving.
    """

    # Evaluation frequency
    freq: int | None = field(
        default=None,
        metadata={"help": "Evaluate every N steps (None = no evaluation)"},
    )

    # Evaluation settings
    episodes: int = field(
        default=10,
        metadata={"help": "Number of episodes for evaluation"},
    )
    max_steps: int = field(
        default=400,
        metadata={"help": "Maximum steps per episode during evaluation"},
    )

    # Parallelization
    num_envs: int = field(
        default=4,
        metadata={"help": "Number of parallel environments for evaluation"},
    )
    sync_envs: bool = field(
        default=False,
        metadata={"help": "Use synchronous (sequential) vectorized environments"},
    )

    # Multi-N evaluation for IDQL (evaluate with different num_action_samples)
    eval_num_action_samples: list[int] | None = field(
        default=None,
        metadata={
            "help": "List of num_action_samples values to evaluate with. "
            "If set, runs eval once per value and logs metrics with /nX suffix. "
            "Best model is selected from the LAST value in the list (typically the largest). "
            "Example: [1, 128] evaluates both BC (n1) and IDQL (n128)."
        },
    )

    fixed_initial_states: bool = field(
        default=False,
        metadata={
            "help": (
                "Evaluate every checkpoint/model on the same deterministic Sobol "
                "initial-state list. Supported for NutAssemblySquare and Square_D1."
            )
        },
    )
    fixed_initial_state_seed: int = field(
        default=20260524,
        metadata={"help": "Seed for the fixed Sobol evaluation initial-state list."},
    )
    # Video saving
    save_video: bool = field(
        default=False,
        metadata={"help": "Save videos of evaluation rollouts"},
    )
    max_video_episodes: int = field(
        default=20,
        metadata={
            "help": "Only record video for first N episodes (0 = all). Reduces memory usage."
        },
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.freq is not None and self.freq < 1:
            raise ValueError(f"freq must be >= 1 or None, got {self.freq}")
        if self.episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {self.episodes}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")
        if self.fixed_initial_state_seed < 0:
            raise ValueError(
                f"fixed_initial_state_seed must be >= 0, got {self.fixed_initial_state_seed}"
            )
        if self.max_video_episodes < 0:
            raise ValueError(f"max_video_episodes must be >= 0, got {self.max_video_episodes}")
        if self.eval_num_action_samples is not None and (
            not self.eval_num_action_samples or min(self.eval_num_action_samples) < 1
        ):
            raise ValueError(
                "eval_num_action_samples must be None or a non-empty list of values >= 1, "
                f"got {self.eval_num_action_samples}"
            )
