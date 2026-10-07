"""Dataset configuration for training."""

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """
    Configuration for dataset loading and filtering.

    Attributes:
        repo_ids: Comma-separated list of dataset repo IDs.
            Example: "mulligan/sim-square-narrow-c00-teleop-sobol,mulligan/sim-square-narrow-c01-dagger-mulligan"
        root: Root directory containing datasets.
            If None, downloads from HuggingFace Hub.
        intervention_negative_reward: Apply negative reward penalty when human intervention
            occurs. When set, adds this value (typically negative, e.g., -1.0) to the reward
            at timesteps where intervention=1.
    """

    # Required: dataset identifiers
    repo_ids: str = field(metadata={"help": "Comma-separated list of dataset repo IDs"})

    # Optional: local root directory
    root: str | None = field(
        default=None,
        metadata={"help": "Root directory for datasets. None = download from Hub."},
    )
    revisions: dict[str, str] | None = field(
        default=None,
        metadata={
            "help": "Optional repo_id-to-Hub-revision mapping. When provided, every "
            "repo_id must have an entry and training loads that exact revision."
        },
    )

    # Reward modification
    intervention_negative_reward: float | None = field(
        default=-1.0,
        metadata={"help": "Negative reward penalty for intervention timesteps"},
    )
    reward_shift: float = field(
        default=0.0,
        metadata={
            "help": "Constant added to all rewards after intervention_negative_reward. "
            "E.g., -1.0 shifts [0,1] rewards to [-1,0]. Typically paired with "
            "policy.clip_targets_to_range=False so the critic can learn the "
            "shifted negative range."
        },
    )

    # Straddled sampling: the critic draws 50/50 from human-success and DAgger data;
    # policy_data_mode selects the data the actor is trained on.
    policy_data_mode: str = field(
        default="human_only",
        metadata={
            "help": "Policy training data mode. Options: "
            "'human_only' = source=HUMAN AND success=SUCCESS only (default), "
            "'straddled_auto_success' = 50% human-success + 50% pure autonomous successes "
            "(episodes where ALL frames have source=AUTONOMOUS AND success=SUCCESS), "
            "'straddled_all' = 50% human-success + 50% all DAgger data "
            "(mirrors critic sampling; human corrections in DAgger appear in both strata)"
        },
    )
    strict_policy_data_mode: bool = field(
        default=False,
        metadata={
            "help": "Fail instead of changing a requested straddled actor or critic "
            "data mode when its secondary stratum is too small. Enable for locked "
            "experiments where a fallback would invalidate the comparison."
        },
    )
    critic_sampling_mode: str = field(
        default="straddled",
        metadata={
            "help": "Critic/value sampling mode. Options: "
            "'straddled' = 50% human-success + 50% DAgger (default), "
            "'flat' = uniform sampling across all data (no straddling), "
            "'human_only' = uniform over valid chunks containing only human-success actions, "
            "'balanced_outcome' = 50% success-episode + 50% fail-or-intervened "
            "transitions (balances the value net's view of the return support so "
            "the failure tail isn't washed out at high SR; DIVL sampler ablation)"
        },
    )

    def __post_init__(self):
        """Validate configuration."""
        if not self.repo_ids:
            raise ValueError("repo_ids cannot be empty")
        if self.revisions is not None:
            repo_ids = set(self.get_repo_id_list())
            revision_ids = set(self.revisions)
            if revision_ids != repo_ids:
                raise ValueError(
                    "dataset.revisions keys must exactly match dataset.repo_ids; "
                    f"missing={sorted(repo_ids - revision_ids)}, "
                    f"extra={sorted(revision_ids - repo_ids)}"
                )
            empty = sorted(repo_id for repo_id, revision in self.revisions.items() if not revision)
            if empty:
                raise ValueError(f"dataset.revisions contains empty revisions for {empty}")

        valid_policy_modes = ["human_only", "straddled_auto_success", "straddled_all"]
        if self.policy_data_mode not in valid_policy_modes:
            raise ValueError(
                f"policy_data_mode must be one of {valid_policy_modes}, "
                f"got '{self.policy_data_mode}'"
            )

        valid_critic_modes = ["straddled", "flat", "balanced_outcome", "human_only"]
        if self.critic_sampling_mode not in valid_critic_modes:
            raise ValueError(
                f"critic_sampling_mode must be one of {valid_critic_modes}, "
                f"got '{self.critic_sampling_mode}'"
            )

    def get_repo_id_list(self) -> list[str]:
        """Parse comma-separated repo_ids into a list."""
        return [rid.strip() for rid in self.repo_ids.split(",")]
