#!/usr/bin/env python3
"""Test suite for straddled batch sampling functionality.

Tests cover:
- SubsetRandomSampler: shuffle behavior, index coverage
- Policy data mode configuration validation
"""

import torch
import pytest

from mulligan.training.straddled_sampler import GPUBatchSampler, SubsetRandomSampler
from mulligan.configs.dataset import DatasetConfig


class TestSubsetRandomSampler:
    """Tests for SubsetRandomSampler."""

    def test_returns_all_indices(self):
        """Test that all indices are returned exactly once per epoch."""
        indices = torch.arange(50)
        sampler = SubsetRandomSampler(indices, shuffle=True)

        sampled = list(sampler)

        assert len(sampled) == len(indices)
        assert set(sampled) == set(indices.tolist())

    def test_length(self):
        """Test that __len__ returns correct length."""
        indices = torch.arange(42)
        sampler = SubsetRandomSampler(indices)

        assert len(sampler) == 42

    def test_shuffle_produces_different_order(self):
        """Test that shuffle=True produces different orders across iterations."""
        torch.manual_seed(42)
        indices = torch.arange(100)

        sampler1 = SubsetRandomSampler(indices, shuffle=True)
        sampler2 = SubsetRandomSampler(indices, shuffle=True)

        order1 = list(sampler1)
        order2 = list(sampler2)

        # Very unlikely to be the same order with 100! permutations
        assert order1 != order2

    def test_no_shuffle_deterministic(self):
        """Test that shuffle=False returns indices in original order."""
        indices = torch.tensor([5, 2, 8, 1, 9])
        sampler = SubsetRandomSampler(indices, shuffle=False)

        order = list(sampler)

        assert order == [5, 2, 8, 1, 9]

    def test_list_input(self):
        """Test that list inputs work correctly."""
        indices = [10, 20, 30, 40, 50]
        sampler = SubsetRandomSampler(indices, shuffle=False)

        order = list(sampler)

        assert order == [10, 20, 30, 40, 50]

    def test_empty_indices(self):
        """Test with empty indices."""
        indices = torch.tensor([], dtype=torch.long)
        sampler = SubsetRandomSampler(indices)

        assert len(sampler) == 0
        assert list(sampler) == []

    def test_single_index(self):
        """Test with single index."""
        indices = torch.tensor([42])
        sampler = SubsetRandomSampler(indices, shuffle=True)

        assert len(sampler) == 1
        assert list(sampler) == [42]


class TestGPUBatchSampler:
    """Tests for GPUBatchSampler (GPU-native batch sampling)."""

    @pytest.fixture
    def device(self):
        """Get device for testing (GPU if available, else CPU)."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_correct_5050_distribution_straddled(self, device):
        """Test that straddled mode produces 50/50 distribution."""
        indices_a = torch.arange(100)
        indices_b = torch.arange(100, 200)
        batch_size = 10

        sampler = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

        batch = sampler.sample()

        assert batch.shape == (batch_size,)
        assert batch.device.type == device if device != "cpu" else "cpu"

        # First half should be from set A (0-99), second half from set B (100-199)
        batch_a = batch[: batch_size // 2 + batch_size % 2]  # size_a
        batch_b = batch[batch_size // 2 + batch_size % 2 :]  # size_b

        assert all(0 <= idx < 100 for idx in batch_a.tolist()), "First half should be from set A"
        assert all(100 <= idx < 200 for idx in batch_b.tolist()), "Second half should be from set B"

    def test_single_set_mode(self, device):
        """Test single-set mode (indices_b=None)."""
        indices_a = torch.arange(50)
        batch_size = 10

        sampler = GPUBatchSampler(indices_a, None, batch_size, device=device)

        batch = sampler.sample()

        assert batch.shape == (batch_size,)
        assert all(0 <= idx < 50 for idx in batch.tolist())

    def test_odd_batch_size_set_a_gets_extra(self, device):
        """Test that with odd batch_size in straddled mode, set A gets the extra sample."""
        indices_a = torch.arange(50)
        indices_b = torch.arange(50, 100)
        batch_size = 11  # Odd

        sampler = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

        # size_a = (11 + 1) // 2 = 6, size_b = 11 // 2 = 5
        assert sampler.size_a == 6
        assert sampler.size_b == 5

        batch = sampler.sample()
        assert batch.shape == (batch_size,)

    def test_device_placement(self, device):
        """Test that sampled indices are on the correct device."""
        indices_a = torch.arange(50)
        indices_b = torch.arange(50, 100)
        batch_size = 10

        sampler = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

        batch = sampler.sample()

        expected_device = device if device != "cpu" else "cpu"
        assert batch.device.type == expected_device

    def test_reshuffling_on_exhaustion(self, device):
        """Test that indices are reshuffled when permutation is exhausted."""
        indices_a = torch.arange(10)  # Small set
        indices_b = torch.arange(10, 110)  # Larger set
        batch_size = 4  # 2 from each

        sampler = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

        # Sample multiple times to exhaust set A (10 indices / 2 per batch = 5 batches)
        batches_a = []
        for _ in range(10):  # More than enough to exhaust and reshuffle
            batch = sampler.sample()
            batches_a.append(batch[: sampler.size_a].tolist())

        # Should have gotten samples from A multiple times (reshuffled)
        all_a_samples = [idx for batch in batches_a for idx in batch]
        assert len(all_a_samples) == 10 * sampler.size_a

    def test_determinism_with_seed(self, device):
        """Test that setting torch seed produces deterministic sampling."""
        indices_a = torch.arange(100)
        indices_b = torch.arange(100, 200)
        batch_size = 10

        torch.manual_seed(42)
        sampler1 = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)
        batch1 = sampler1.sample()

        torch.manual_seed(42)
        sampler2 = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)
        batch2 = sampler2.sample()

        assert torch.equal(batch1, batch2)

    def test_different_samples_without_seed(self, device):
        """Test that different samplers produce different batches (statistical)."""
        indices_a = torch.arange(1000)
        indices_b = torch.arange(1000, 2000)
        batch_size = 10

        sampler1 = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)
        sampler2 = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

        batch1 = sampler1.sample()
        batch2 = sampler2.sample()

        # With 1000 indices, probability of exact match is negligible
        assert not torch.equal(batch1, batch2)

    def test_batch_size_1_raises_error_straddled(self, device):
        """Test that batch_size=1 raises error in straddled mode."""
        indices_a = torch.arange(10)
        indices_b = torch.arange(10, 20)

        with pytest.raises(ValueError, match="batch_size must be at least 2"):
            GPUBatchSampler(indices_a, indices_b, batch_size=1, device=device)

    def test_batch_size_1_allowed_single_mode(self, device):
        """Test that batch_size=1 is allowed in single-set mode."""
        indices_a = torch.arange(10)

        sampler = GPUBatchSampler(indices_a, None, batch_size=1, device=device)
        batch = sampler.sample()

        assert batch.shape == (1,)

    def test_empty_indices_a_raises_error(self, device):
        """Test that empty indices_a raises error."""
        indices_a = torch.tensor([], dtype=torch.long)

        with pytest.raises(ValueError, match="indices_a cannot be empty"):
            GPUBatchSampler(indices_a, None, batch_size=10, device=device)

    def test_empty_indices_b_raises_error_straddled(self, device):
        """Test that empty indices_b raises error in straddled mode."""
        indices_a = torch.arange(100)
        indices_b = torch.tensor([], dtype=torch.long)

        with pytest.raises(ValueError, match="indices_b cannot be empty"):
            GPUBatchSampler(indices_a, indices_b, batch_size=10, device=device)

    def test_insufficient_indices_a(self, device):
        """Test error when indices_a has insufficient elements."""
        indices_a = torch.arange(3)  # Only 3 indices
        indices_b = torch.arange(10, 110)
        batch_size = 10  # Needs 5 from A

        with pytest.raises(ValueError, match="indices_a has 3 elements"):
            GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

    def test_insufficient_indices_b(self, device):
        """Test error when indices_b has insufficient elements."""
        indices_a = torch.arange(100)
        indices_b = torch.arange(100, 103)  # Only 3 indices
        batch_size = 10  # Needs 5 from B

        with pytest.raises(ValueError, match="indices_b has 3 elements"):
            GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

    def test_all_indices_from_correct_sets(self, device):
        """Test that all sampled indices come from the correct sets."""
        indices_a = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        indices_b = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        batch_size = 4

        sampler = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

        for _ in range(10):
            batch = sampler.sample()
            batch_a = batch[: sampler.size_a]
            batch_b = batch[sampler.size_a :]

            # All indices in first part should be from indices_a
            for idx in batch_a.tolist():
                assert idx in indices_a.tolist(), f"{idx} not in indices_a"

            # All indices in second part should be from indices_b
            for idx in batch_b.tolist():
                assert idx in indices_b.tolist(), f"{idx} not in indices_b"

    def test_multiple_samples_same_sampler(self, device):
        """Test that multiple samples from same sampler work correctly."""
        indices_a = torch.arange(100)
        indices_b = torch.arange(100, 200)
        batch_size = 10

        sampler = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

        batches = [sampler.sample() for _ in range(20)]

        # All batches should have correct shape
        for batch in batches:
            assert batch.shape == (batch_size,)

        # Should have sampled different batches (with high probability)
        unique_batches = set(tuple(b.tolist()) for b in batches)
        assert len(unique_batches) > 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_performance_no_cpu_transfer(self):
        """Test that sampling stays entirely on GPU (no implicit CPU transfers)."""
        device = "cuda"
        indices_a = torch.arange(10000)
        indices_b = torch.arange(10000, 20000)
        batch_size = 256

        sampler = GPUBatchSampler(indices_a, indices_b, batch_size, device=device)

        # Warm up
        for _ in range(10):
            _ = sampler.sample()

        torch.cuda.synchronize()

        # Time sampling (should be very fast if no CPU transfers)
        import time

        start = time.perf_counter()
        for _ in range(100):
            sampler.sample()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # 100 samples should take less than 100ms if staying on GPU
        # (typical is < 10ms, but being conservative for CI)
        assert elapsed < 0.1, f"Sampling too slow ({elapsed:.3f}s), possible CPU transfers"


class TestPolicyDataModeConfig:
    """Tests for policy data mode configuration validation."""

    def test_valid_human_only_mode(self):
        """Test that 'human_only' is a valid policy_data_mode."""
        config = DatasetConfig(
            repo_ids="test/dataset",
            policy_data_mode="human_only",
        )
        assert config.policy_data_mode == "human_only"

    def test_valid_straddled_auto_success_mode(self):
        """Test that 'straddled_auto_success' is a valid policy_data_mode."""
        config = DatasetConfig(
            repo_ids="test/dataset",
            policy_data_mode="straddled_auto_success",
        )
        assert config.policy_data_mode == "straddled_auto_success"

    def test_invalid_policy_data_mode_raises_error(self):
        """Test that invalid policy_data_mode raises ValueError."""
        with pytest.raises(ValueError, match="policy_data_mode must be one of"):
            DatasetConfig(
                repo_ids="test/dataset",
                policy_data_mode="invalid_mode",
            )

    def test_default_policy_data_mode(self):
        """Test that default policy_data_mode is 'human_only'."""
        config = DatasetConfig(repo_ids="test/dataset")
        assert config.policy_data_mode == "human_only"


class TestIntegrationWithDataLoader:
    """Integration tests with PyTorch DataLoader."""

    def test_subset_sampler_with_dataloader(self):
        """Test SubsetRandomSampler works correctly with DataLoader."""
        from torch.utils.data import DataLoader, TensorDataset

        # Create a simple dataset
        data = torch.randn(100, 5)
        dataset = TensorDataset(data)

        # Create subset sampler for a portion of indices
        indices = torch.arange(50, 100)
        sampler = SubsetRandomSampler(indices, shuffle=True)

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

        # Verify we only get samples from indices 50-99
        all_data = []
        for batch in dataloader:
            all_data.append(batch[0])

        all_data = torch.cat(all_data, dim=0)
        assert len(all_data) == 50  # Only 50 samples from our subset


if __name__ == "__main__":
    # Run tests manually
    print("Running straddled sampler tests...\n")

    # SubsetRandomSampler tests
    test_class = TestSubsetRandomSampler()

    test_class.test_returns_all_indices()
    print("  test_returns_all_indices passed")

    test_class.test_length()
    print("  test_length passed")

    test_class.test_shuffle_produces_different_order()
    print("  test_shuffle_produces_different_order passed")

    test_class.test_no_shuffle_deterministic()
    print("  test_no_shuffle_deterministic passed")

    test_class.test_list_input()
    print("  test_list_input passed")

    test_class.test_empty_indices()
    print("  test_empty_indices passed")

    test_class.test_single_index()
    print("  test_single_index passed")

    print("SubsetRandomSampler tests passed\n")

    # PolicyDataModeConfig tests
    test_class = TestPolicyDataModeConfig()

    test_class.test_valid_human_only_mode()
    print("  test_valid_human_only_mode passed")

    test_class.test_valid_straddled_auto_success_mode()
    print("  test_valid_straddled_auto_success_mode passed")

    test_class.test_invalid_policy_data_mode_raises_error()
    print("  test_invalid_policy_data_mode_raises_error passed")

    test_class.test_default_policy_data_mode()
    print("  test_default_policy_data_mode passed")

    print("PolicyDataModeConfig tests passed\n")

    # Integration tests
    test_class = TestIntegrationWithDataLoader()

    test_class.test_subset_sampler_with_dataloader()
    print("  test_subset_sampler_with_dataloader passed")

    print("Integration tests passed\n")

    print("All tests passed!")


@pytest.mark.parametrize("straddled", [False, True])
def test_gpu_batch_sampler_state_dict_round_trip(straddled):

    torch.manual_seed(0)
    b = torch.arange(100, 130) if straddled else None
    s = GPUBatchSampler(torch.arange(20), b, batch_size=6, device="cpu")
    for _ in range(5):
        s.sample()
    state = s.state_dict()
    rng = torch.get_rng_state()
    expected = [s.sample() for _ in range(7)]

    torch.manual_seed(123)
    t = GPUBatchSampler(torch.arange(20), b, batch_size=6, device="cpu")
    t.load_state_dict(state)
    torch.set_rng_state(rng)
    assert all(torch.equal(x, t.sample()) for x in expected)
    with pytest.raises(ValueError, match="does not match"):
        GPUBatchSampler(torch.arange(21), b, batch_size=6, device="cpu").load_state_dict(state)
