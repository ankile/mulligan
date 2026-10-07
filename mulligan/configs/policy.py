"""Policy configuration for training using draccus ChoiceRegistry for polymorphism."""

from dataclasses import dataclass, field

import draccus

# The IDQL diffusion actor conditions on one observation step and samples with DDIM;
# both are fixed (the batch builders and select_action feed a single step).
IDQL_N_OBS_STEPS = 1
IDQL_NOISE_SCHEDULER = "DDIM"


@dataclass
class PolicyConfig(draccus.ChoiceRegistry):
    """
    Base policy configuration class.

    Subclasses register themselves using the @PolicyConfig.register_subclass decorator.
    This enables CLI polymorphism via --policy.type=<name>.

    Registered types: ``idql`` and ``idql_divl``.
    """

    # Common architecture parameters
    hidden_dims: list[int] = field(
        default_factory=lambda: [256, 256, 256],
        metadata={"help": "Comma-separated hidden layer dimensions"},
    )
    use_layer_norm: bool = field(
        default=False,
        metadata={"help": "Use layer normalization in networks"},
    )

    def print_config_info(self) -> None:
        """Print policy-specific configuration info. Override in subclasses."""
        print(f"Hidden dims: {self.hidden_dims}")


@PolicyConfig.register_subclass("idql")
@dataclass
class IDQLPolicyConfig(PolicyConfig):
    """
    Configuration for IDQL policy (Implicit Diffusion Q-Learning).

    This policy uses:
    - IQL for value learning (V via expectile regression, Q via Q-learning)
    - Diffusion Policy as the actor
    - N-sample Q-max selection at inference time

    Diffusion policy settings have sensible defaults for state-based IDQL:
    - Single observation step (n_obs_steps=1)
    - Small U-Net (down_dims=[512]) for fast inference
    - DDIM scheduler with 8 inference steps for speed
    """

    # IQL hyperparameters
    expectile: float = field(
        default=0.7,
        metadata={"help": "Expectile for V-network regression"},
    )
    gamma: float = field(
        default=0.99,
        metadata={"help": "Discount factor for Q-learning"},
    )
    tau: float = field(
        default=0.005,
        metadata={"help": "Polyak averaging coefficient for target networks"},
    )

    # Action selection at inference
    num_action_samples: int = field(
        default=10,
        metadata={"help": "Number of actions to sample for Q-max selection"},
    )
    # Double Q-learning: Q targets and action selection take the minimum over the ensemble.
    num_q_networks: int = field(
        default=1,
        metadata={"help": "Number of Q-networks for ensemble (aggregated by min)"},
    )

    # Target clipping
    clip_targets_to_range: bool = field(
        default=False,
        metadata={"help": "Clip Q-value targets to empirical reward range"},
    )

    # Diffusion policy architecture (common settings with good defaults)
    chunk_size: int = field(
        default=1,
        metadata={"help": "Action chunk size (horizon for diffusion). 1 = single-step actions."},
    )
    n_action_steps: int | None = field(
        default=None,
        metadata={
            "help": "Number of action steps to execute per chunk (queue length). "
            "None = equal to chunk_size (default, predict==execute). "
            "Setting < chunk_size lets the actor predict longer chunks but execute fewer steps "
            "(e.g. predict 8, execute 5 — useful when the actor benefits from longer prediction "
            "horizons but the critic input shape should match a shorter execution chunk)."
        },
    )
    down_dims: list[int] = field(
        default_factory=lambda: [512],
        metadata={
            "help": "U-Net encoder channel dimensions. [512] = small/fast, [512,1024,2048] = large."
        },
    )

    # Diffusion scheduler settings (DDIM)
    num_train_timesteps: int = field(
        default=100,
        metadata={"help": "Number of diffusion timesteps during training."},
    )
    num_inference_steps: int = field(
        default=8,
        metadata={
            "help": "Number of diffusion timesteps during inference (DDIM only). Lower = faster."
        },
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.n_action_steps is not None:
            if self.n_action_steps < 1:
                raise ValueError(f"n_action_steps must be >= 1 when set, got {self.n_action_steps}")
            # LeRobot diffusion policies consume one observation step inside the
            # horizon window, so the maximum executable action queue is the
            # number of action slots remaining in that window.
            max_action_steps = self.chunk_size - IDQL_N_OBS_STEPS + 1
            if self.n_action_steps > max_action_steps:
                raise ValueError(
                    f"n_action_steps ({self.n_action_steps}) must be <= "
                    f"the diffusion horizon's available action slots "
                    f"(chunk_size - n_obs_steps + 1 = {max_action_steps})"
                )
        if self.num_action_samples < 1:
            raise ValueError(f"num_action_samples must be >= 1, got {self.num_action_samples}")
        if self.num_q_networks < 1:
            raise ValueError(f"num_q_networks must be >= 1, got {self.num_q_networks}")
        if not (0 <= self.expectile <= 1):
            raise ValueError(f"expectile must be in [0, 1], got {self.expectile}")
        if not (0 <= self.gamma <= 1):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not (0 < self.tau <= 1):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")

    def get_diffusion_overrides(self) -> dict:
        """Build dict of diffusion settings to pass to LeRobot's DiffusionConfig."""
        return {
            "down_dims": self.down_dims,
            "n_obs_steps": IDQL_N_OBS_STEPS,
            "noise_scheduler_type": IDQL_NOISE_SCHEDULER,
            "num_train_timesteps": self.num_train_timesteps,
            "num_inference_steps": self.num_inference_steps,
        }

    def print_config_info(self) -> None:
        """Print IDQL specific configuration info."""
        print(f"Expectile: {self.expectile}, Gamma: {self.gamma}, Tau: {self.tau}")
        print(f"Clip targets to range: {self.clip_targets_to_range}")
        print(f"Num action samples: {self.num_action_samples}")
        print(f"Num Q-networks: {self.num_q_networks}")
        print(
            "Action horizons: "
            f"predict={self.chunk_size}, execute={self.n_action_steps or self.chunk_size}"
        )
        print(f"Diffusion: {IDQL_NOISE_SCHEDULER}, {self.num_inference_steps} inference steps")


@PolicyConfig.register_subclass("idql_divl")
@dataclass
class IDQLDIVLConfig(IDQLPolicyConfig):
    """
    Configuration for DIVL (Distributional Implicit Value Learning) policy.

    Identical to IDQL except the value network V(s) is *distributional* (a
    categorical distribution over a fixed support of action-values) rather than
    scalar. The TD bootstrap statistic is a tau-quantile extracted from that
    distribution, with optional entropy-adaptive tau. The critic Q(s,a) stays
    scalar (clipped double-Q) and inference (sample-and-rerank on scalar Q) is
    unchanged — DIVL is a drop-in upgrade to training-time value learning only.

    Defaults reproduce scalar-IQL behavior at the optimum: ``tau_base`` equals
    the IQL ``expectile`` (0.7) and ``tau_entropy_alpha=0`` (fixed tau), so only
    the distributional form of V differs from IDQL.

    Reference: AgiBot Finch, "Learning while Deploying: Fleet-Scale RL for
    Generalist Robot Policies", arXiv:2605.00416 (May 2026).
    """

    # Categorical value support.
    num_atoms: int = field(
        default=101,
        metadata={"help": "Number of categorical atoms for the distributional V-network."},
    )
    v_min: float | None = field(
        default=None,
        metadata={
            "help": "Lower edge of the value support. None = derive from the "
            "empirical return range at policy construction (stored back into "
            "config for eval-time reconstruction)."
        },
    )
    v_max: float | None = field(
        default=None,
        metadata={"help": "Upper edge of the value support. None = derive from data (see v_min)."},
    )
    hl_gauss_sigma_ratio: float = field(
        default=0.75,
        metadata={
            "help": "HL-Gauss smoothing sigma as a multiple of the atom spacing "
            "(sigma = ratio * atom_width)."
        },
    )

    # Quantile / adaptive-tau settings.
    tau_base: float = field(
        default=0.7,
        metadata={"help": "Base quantile level for the TD bootstrap (mirrors IQL expectile)."},
    )
    tau_min: float = field(
        default=0.5,
        metadata={"help": "Lower clip bound for the adaptive quantile level."},
    )
    tau_max: float = field(
        default=0.95,
        metadata={"help": "Upper clip bound for the adaptive quantile level."},
    )
    tau_entropy_alpha: float = field(
        default=0.0,
        metadata={
            "help": "Entropy sensitivity for adaptive tau: "
            "tau = clip(tau_base - alpha * norm_entropy, tau_min, tau_max). "
            "0 = fixed tau (default); >0 enables entropy-adaptive tau."
        },
    )

    def __post_init__(self):
        """Run IDQL validators, then add DIVL-specific validation."""
        super().__post_init__()
        if self.num_atoms < 2:
            raise ValueError(f"num_atoms must be >= 2, got {self.num_atoms}")
        if self.v_min is not None and self.v_max is not None and not self.v_max > self.v_min:
            raise ValueError(f"v_max ({self.v_max}) must be greater than v_min ({self.v_min})")
        if (self.v_min is None) != (self.v_max is None):
            raise ValueError(
                "v_min and v_max must both be set or both be None "
                f"(got v_min={self.v_min}, v_max={self.v_max})"
            )
        if self.hl_gauss_sigma_ratio <= 0:
            raise ValueError(f"hl_gauss_sigma_ratio must be > 0, got {self.hl_gauss_sigma_ratio}")
        if not (0 <= self.tau_min <= self.tau_max <= 1):
            raise ValueError(
                f"require 0 <= tau_min <= tau_max <= 1, got "
                f"tau_min={self.tau_min}, tau_max={self.tau_max}"
            )
        if not (0 <= self.tau_base <= 1):
            raise ValueError(f"tau_base must be in [0, 1], got {self.tau_base}")
        if self.tau_entropy_alpha < 0:
            raise ValueError(f"tau_entropy_alpha must be >= 0, got {self.tau_entropy_alpha}")

    def print_config_info(self) -> None:
        """Print DIVL-specific configuration info."""
        super().print_config_info()
        support = (
            "derived-from-data" if self.v_min is None else f"[{self.v_min:.3f}, {self.v_max:.3f}]"
        )
        print(
            f"DIVL: num_atoms={self.num_atoms}, support={support}, "
            f"hl_gauss_sigma_ratio={self.hl_gauss_sigma_ratio}"
        )
        if self.tau_entropy_alpha > 0:
            print(
                f"  Adaptive tau: base={self.tau_base}, alpha={self.tau_entropy_alpha}, "
                f"clip=[{self.tau_min}, {self.tau_max}]"
            )
        else:
            print(f"  Fixed tau: {self.tau_base}")
