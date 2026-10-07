"""Tests for InfiniteSubsetRandomSampler (mulligan.training.straddled_sampler).

The infinite sampler feeds step-based training loops (cycle(dataloader)) with
back-to-back full permutation passes so the DataLoader iterator never raises
StopIteration and never rebuilds its prefetch pipeline at epoch boundaries.
Pinned contract:

  1. The stream is exactly concatenated FULL permutations of the subset —
     every window of len(indices) items covers each index exactly once.
  2. Deterministic per base_seed, independent of global RNG state.
  3. Pass 0 reproduces the permutation the finite SubsetRandomSampler draws
     immediately after ``torch.manual_seed(base_seed)`` (old first epoch).
"""

import itertools

import torch

from mulligan.training.straddled_sampler import (
    InfiniteSubsetRandomSampler,
    SubsetRandomSampler,
)

INDICES = [3, 7, 11, 12, 20, 21, 22, 40, 41, 99]


def _take(sampler, n):
    return list(itertools.islice(iter(sampler), n))


def test_stream_is_concatenated_full_permutations():
    n = len(INDICES)
    stream = _take(InfiniteSubsetRandomSampler(INDICES, base_seed=17), 5 * n)
    for p in range(5):
        window = stream[p * n : (p + 1) * n]
        assert sorted(window) == sorted(INDICES), f"pass {p} is not a full permutation"
    # Sanity: consecutive passes are (overwhelmingly likely) different orders.
    assert stream[:n] != stream[n : 2 * n]


def test_deterministic_per_seed_and_global_rng_independent():
    n = len(INDICES)
    a = _take(InfiniteSubsetRandomSampler(INDICES, base_seed=5), 3 * n)
    torch.manual_seed(123456)  # perturb global RNG between constructions
    torch.randperm(1000)
    b = _take(InfiniteSubsetRandomSampler(INDICES, base_seed=5), 3 * n)
    assert a == b
    c = _take(InfiniteSubsetRandomSampler(INDICES, base_seed=6), 3 * n)
    assert a != c


def test_pass0_matches_finite_sampler_first_epoch():
    seed = 42
    torch.manual_seed(seed)
    old_first_epoch = list(iter(SubsetRandomSampler(INDICES)))
    infinite = _take(InfiniteSubsetRandomSampler(INDICES, base_seed=seed), len(INDICES))
    assert infinite == old_first_epoch


def test_len_reports_one_pass():
    assert len(InfiniteSubsetRandomSampler(INDICES, base_seed=0)) == len(INDICES)


def test_tensor_indices_accepted():
    t = torch.tensor(INDICES, dtype=torch.long)
    n = len(INDICES)
    stream = _take(InfiniteSubsetRandomSampler(t, base_seed=1), 2 * n)
    assert sorted(stream[:n]) == sorted(INDICES)
    assert all(isinstance(i, int) for i in stream)


def test_never_raises_stopiteration_across_many_passes():
    n = len(INDICES)
    it = iter(InfiniteSubsetRandomSampler(INDICES, base_seed=9))
    seen = [next(it) for _ in range(25 * n + 3)]  # deep into pass 25 without exhaustion
    assert len(seen) == 25 * n + 3
