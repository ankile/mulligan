#!/usr/bin/env python3
"""
Straddled batch sampler for HiL-SERL style training.

This module provides a batch sampler that yields batches with 50% of samples
from one index set (e.g., demo buffer) and 50% from another (e.g., RL buffer).
"""

from typing import Iterator

import torch
from torch.utils.data import Sampler


class GPUBatchSampler:
    """
    Fast GPU-native batch sampler for direct tensor indexing.

    Unlike DataLoader-based sampling, this keeps everything on GPU and returns
    index tensors directly for fast tensor indexing. Avoids Python loops and
    CPU-GPU transfers entirely.

    Supports two modes:
    - Straddled: 50% from index set A, 50% from index set B
    - Single: Sample from a single index set

    Args:
        indices_a: First set of indices (on GPU)
        indices_b: Optional second set for straddled sampling (on GPU)
        batch_size: Total batch size
        device: GPU device
    """

    def __init__(
        self,
        indices_a: torch.Tensor,
        indices_b: torch.Tensor | None,
        batch_size: int,
        device: str | torch.device,
    ):
        self.device = device
        self.batch_size = batch_size
        self.straddled = indices_b is not None

        # Input validation
        if len(indices_a) == 0:
            raise ValueError("indices_a cannot be empty")

        if self.straddled:
            if batch_size < 2:
                raise ValueError(
                    f"batch_size must be at least 2 for straddled mode, got {batch_size}"
                )
            if indices_b is not None and len(indices_b) == 0:
                raise ValueError("indices_b cannot be empty in straddled mode")

            # 50/50 split
            self.size_a = (batch_size + 1) // 2
            self.size_b = batch_size // 2

            # Validate sufficient indices
            if len(indices_a) < self.size_a:
                raise ValueError(
                    f"indices_a has {len(indices_a)} elements but need at least {self.size_a} "
                    f"for batch_size={batch_size}"
                )
            if indices_b is not None and len(indices_b) < self.size_b:
                raise ValueError(
                    f"indices_b has {len(indices_b)} elements but need at least {self.size_b} "
                    f"for batch_size={batch_size}"
                )
        else:
            if batch_size < 1:
                raise ValueError(f"batch_size must be at least 1, got {batch_size}")
            if len(indices_a) < batch_size:
                raise ValueError(
                    f"indices_a has {len(indices_a)} elements but need at least {batch_size}"
                )
            self.size_a = batch_size
            self.size_b = 0

        # Move indices to GPU
        self.indices_a = indices_a.to(device)
        self.indices_b = indices_b.to(device) if indices_b is not None else None

        # Initialize permutation state
        self._reset_permutations()

    def _reset_permutations(self):
        """Reset/reshuffle permutations."""
        self.perm_a = torch.randperm(len(self.indices_a), device=self.device)
        self.ptr_a = 0

        if self.straddled:
            self.perm_b = torch.randperm(len(self.indices_b), device=self.device)
            self.ptr_b = 0

    def state_dict(self) -> dict:
        """Permutations and read pointers, for a bitwise resume."""
        state = {"perm_a": self.perm_a.cpu(), "ptr_a": self.ptr_a}
        if self.straddled:
            state.update(perm_b=self.perm_b.cpu(), ptr_b=self.ptr_b)
        return state

    def load_state_dict(self, state: dict) -> None:
        if ("perm_b" in state) != self.straddled or len(state["perm_a"]) != len(self.indices_a):
            raise ValueError("sampler state does not match this sampler's index sets")
        self.perm_a = state["perm_a"].to(self.device)
        self.ptr_a = int(state["ptr_a"])
        if self.straddled:
            if len(state["perm_b"]) != len(self.indices_b):
                raise ValueError("sampler state does not match this sampler's index sets")
            self.perm_b = state["perm_b"].to(self.device)
            self.ptr_b = int(state["ptr_b"])

    def sample(self) -> torch.Tensor:
        """
        Sample a batch of indices (on GPU).

        Returns:
            torch.Tensor: Batch indices on GPU, shape (batch_size,)
        """
        # Check if we need to reshuffle set A
        if self.ptr_a + self.size_a > len(self.indices_a):
            self.perm_a = torch.randperm(len(self.indices_a), device=self.device)
            self.ptr_a = 0

        # Get indices from set A
        idx_a = self.indices_a[self.perm_a[self.ptr_a : self.ptr_a + self.size_a]]
        self.ptr_a += self.size_a

        if not self.straddled:
            return idx_a

        # Check if we need to reshuffle set B
        if self.ptr_b + self.size_b > len(self.indices_b):
            self.perm_b = torch.randperm(len(self.indices_b), device=self.device)
            self.ptr_b = 0

        # Get indices from set B
        idx_b = self.indices_b[self.perm_b[self.ptr_b : self.ptr_b + self.size_b]]
        self.ptr_b += self.size_b

        # Concatenate on GPU
        return torch.cat([idx_a, idx_b])


class SubsetRandomSampler(Sampler[int]):
    """
    Sampler that samples from a subset of indices with shuffling.

    Unlike torch.utils.data.SubsetRandomSampler, this one reshuffles
    every epoch and provides proper __len__.

    Args:
        indices: Indices to sample from
        shuffle: Whether to shuffle (default: True)
    """

    def __init__(self, indices: torch.Tensor | list[int], shuffle: bool = True):
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long)
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            perm = torch.randperm(len(self.indices))
            indices = self.indices[perm]
        else:
            indices = self.indices
        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self.indices)


class InfiniteSubsetRandomSampler(Sampler[int]):
    """Infinite stream of back-to-back full random permutations of a subset.

    Unlike :class:`SubsetRandomSampler`, iteration NEVER raises StopIteration:
    when one full permutation of ``indices`` is exhausted the next one starts
    immediately. Fed to a DataLoader, the epoch boundary disappears — a
    ``cycle(dataloader)`` consumer never exhausts the underlying iterator, so
    the DataLoader never tears down and recreates its prefetch pipeline (the
    multi-second epoch-boundary drain in step-based training loops).

    Pass ``p`` is permuted by a dedicated generator seeded with
    ``base_seed + p``: the stream is deterministic given ``base_seed``
    (independent of global RNG consumption), and pass 0 reproduces exactly the
    permutation :class:`SubsetRandomSampler` draws immediately after
    ``torch.manual_seed(base_seed)``. The sampler is iterated in the main
    process (map-style DataLoader), so worker sharding is correct by
    construction.

    Args:
        indices: Indices to sample from
        base_seed: Seed for pass 0; pass p uses ``base_seed + p``
    """

    def __init__(self, indices: torch.Tensor | list[int], base_seed: int):
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long)
        self.indices = indices
        self.base_seed = base_seed

    def __iter__(self) -> Iterator[int]:
        pass_idx = 0
        g = torch.Generator()
        while True:
            g.manual_seed(self.base_seed + pass_idx)
            perm = torch.randperm(len(self.indices), generator=g)
            yield from self.indices[perm].tolist()
            pass_idx += 1

    def __len__(self) -> int:
        # Length of ONE pass, so len(DataLoader) still reports batches-per-epoch.
        return len(self.indices)
