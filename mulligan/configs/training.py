"""Training configuration for optimization and gradient settings."""

from dataclasses import dataclass, field

from mulligan.training.precision import AMP_DTYPE_CHOICES


@dataclass
class TrainingConfig:
    """
    Configuration for training hyperparameters.

    Controls batch size, learning rates, gradient clipping, and training duration.
    """

    seed: int | None = field(
        default=None,
        metadata={
            "help": "Optional training RNG seed used for Python, NumPy, and Torch. "
            "None leaves initialization nondeterministic."
        },
    )

    # Batch size
    batch_size: int = field(
        default=256,
        metadata={"help": "Batch size for training"},
    )

    # Learning rates (separate for each component)
    lr_actor: float = field(
        default=3e-4,
        metadata={"help": "Actor learning rate"},
    )
    lr_critic: float = field(
        default=3e-4,
        metadata={"help": "Critic learning rate"},
    )
    lr_value: float = field(
        default=3e-4,
        metadata={"help": "Value network learning rate"},
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay for optimizer"},
    )

    # Gradient clipping
    max_grad_norm_actor: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for actor (0 = no clipping)"},
    )
    max_grad_norm_critic: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for critic"},
    )
    max_grad_norm_value: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for value network"},
    )

    # Precision
    amp_dtype: str = field(
        default="none",
        metadata={"help": "CUDA autocast dtype for policy updates: 'none' or 'bfloat16'."},
    )
    enable_tf32: bool = field(
        default=True,
        metadata={"help": "Enable TF32 matmul/cuDNN kernels on CUDA devices."},
    )
    compile_actor: bool = field(
        default=True,
        metadata={
            "help": "Compile the IDQL diffusion actor loss on CUDA with torch.compile. "
            "This has first-step compile cost and GPU/model-dependent throughput gains."
        },
    )
    compile_mode: str = field(
        default="default",
        metadata={"help": "torch.compile mode for compile_actor. Passed through to torch.compile."},
    )

    # Training duration
    training_steps: int = field(
        default=50_000,
        metadata={"help": "Number of training steps"},
    )
    update_components: str = field(
        default="all",
        metadata={
            "help": "Which IDQL components to optimize: 'all' updates actor, critic, and "
            "value; 'critic_value_only' freezes the pretrained actor and trains fresh "
            "critic/value networks (DIVL)."
        },
    )

    # Checkpointing
    periodic_checkpoint_freq: int = field(
        default=0,
        metadata={"help": "Save periodic checkpoints every N steps (0 = never)"},
    )
    resume_checkpoint_freq: int = field(
        default=5_000,
        metadata={
            "help": "Save the rolling local resume checkpoint every N steps (0 disables "
            "resume). Resume is keyed by wandb.run_name, which names the run."
        },
    )
    save_final_model: bool = field(
        default=True,
        metadata={"help": "Save final model at end of training"},
    )
    save_best_model: bool = field(
        default=True,
        metadata={"help": "Save best model checkpoints during evaluation"},
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.seed is not None and self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.training_steps < 1:
            raise ValueError(f"training_steps must be >= 1, got {self.training_steps}")
        if self.update_components not in ("all", "critic_value_only"):
            raise ValueError(
                "update_components must be 'all' or 'critic_value_only', "
                f"got {self.update_components!r}"
            )
        if self.amp_dtype not in AMP_DTYPE_CHOICES:
            raise ValueError(
                f"amp_dtype must be one of {AMP_DTYPE_CHOICES}, got {self.amp_dtype!r}"
            )
        if not isinstance(self.compile_mode, str) or not self.compile_mode:
            raise ValueError(f"compile_mode must be a non-empty string, got {self.compile_mode!r}")
        if self.resume_checkpoint_freq < 0:
            raise ValueError(
                f"resume_checkpoint_freq must be >= 0, got {self.resume_checkpoint_freq}"
            )
