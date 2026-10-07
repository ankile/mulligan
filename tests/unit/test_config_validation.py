"""Validation contract tests for IDQLPolicyConfig and DatasetConfig.

These tests pin the fixed diffusion settings of IDQLPolicyConfig and the
validators that gate `policy_data_mode` and `critic_sampling_mode`.
"""

import pytest


# =============================================================================
# IDQLPolicyConfig: fixed diffusion settings
# =============================================================================


class TestFixedDiffusionSettings:
    """One observation step and DDIM are fixed, not options."""

    def test_diffusion_overrides(self):
        from mulligan.configs.policy import IDQLPolicyConfig

        overrides = IDQLPolicyConfig().get_diffusion_overrides()
        assert overrides["n_obs_steps"] == 1
        assert overrides["noise_scheduler_type"] == "DDIM"

    @pytest.mark.parametrize(
        "removed", ["n_obs_steps", "noise_scheduler_type", "q_ensemble_aggregation", "dropout_rate"]
    )
    def test_removed_options_are_rejected(self, removed):
        from mulligan.configs.policy import IDQLPolicyConfig

        with pytest.raises(TypeError):
            IDQLPolicyConfig(**{removed: None})


# =============================================================================
# DatasetConfig: critic_sampling_mode
# =============================================================================


class TestCriticSamplingModeValidator:
    """Pin the critic_sampling_mode contract."""

    def test_default_is_straddled(self):
        """Default must remain 'straddled' — flipping the default would
        change every existing run silently."""
        from mulligan.configs.dataset import DatasetConfig

        cfg = DatasetConfig(repo_ids="example-org/foo")
        assert cfg.critic_sampling_mode == "straddled"

    def test_flat_mode_validates(self):
        from mulligan.configs.dataset import DatasetConfig

        DatasetConfig(repo_ids="example-org/foo", critic_sampling_mode="flat")

    def test_straddled_mode_validates(self):
        from mulligan.configs.dataset import DatasetConfig

        DatasetConfig(repo_ids="example-org/foo", critic_sampling_mode="straddled")

    def test_invalid_mode_raises(self):
        from mulligan.configs.dataset import DatasetConfig

        with pytest.raises(ValueError, match="critic_sampling_mode must be one of"):
            DatasetConfig(repo_ids="example-org/foo", critic_sampling_mode="bogus")

    @pytest.mark.parametrize("mode", ["human_only", "straddled_auto_success", "straddled_all"])
    def test_policy_data_modes(self, mode):
        from mulligan.configs.dataset import DatasetConfig

        DatasetConfig(repo_ids="example-org/foo", policy_data_mode=mode)

    def test_invalid_policy_data_mode_raises(self):
        from mulligan.configs.dataset import DatasetConfig

        with pytest.raises(ValueError, match="policy_data_mode must be one of"):
            DatasetConfig(repo_ids="example-org/foo", policy_data_mode="all_data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
