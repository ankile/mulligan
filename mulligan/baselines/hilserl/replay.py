# Based on the EXPO source code (https://github.com/pd-perry/EXPO) and the RLPD source code
# (https://github.com/ikostrikov/rlpd), on which EXPO builds.
# Adapted from EXPO's expo/data/dataset.py, expo/data/replay_buffer.py and train_robo.py (combine).
# Both projects are MIT-licensed; their notices follow.
#
# ---- EXPO ----
# MIT License
#
# Copyright (c) 2025 pd-perry
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---- RLPD (https://github.com/ikostrikov/rlpd, LICENCE) ----
# MIT License
#
# Copyright (c) 2022 Ilya Kostrikov, Philip J. Ball, Laura Smith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Replay buffers + batch combination with RLPD's semantics.

Adapted from EXPO ``expo/data/dataset.py`` (``Dataset``),
``expo/data/replay_buffer.py`` (``RoboReplayBuffer``) and ``train_robo.py``
(``combine``):

- uniform i.i.d. sampling **with replacement** from a ``numpy.random.Generator``
  seeded like ``gym.utils.seeding.np_random(seed)`` (``np.random.default_rng``
  is the same PCG64/SeedSequence construction);
- ``combine(offline, online)`` interleaves rows ``[off0, on0, off1, on1, ...]``
  so that ``SACLearner.update``'s ``utd_ratio`` slices each contain exactly
  ``batch_size/2`` demo + ``batch_size/2`` online rows;
- the six transition keys ``observations, actions, rewards, masks, dones,
  next_observations`` and their dtypes.

No gym dependency; ``insert_dataset`` starts the ring at ``size``; ``state()`` /
``load_state()`` support preemption-safe resume.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

TRANSITION_KEYS = ("observations", "actions", "rewards", "masks", "dones", "next_observations")

Batch = Dict[str, np.ndarray]


def _check_lengths(dataset_dict: Batch) -> int:
    lengths = {k: len(v) for k, v in dataset_dict.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent item lengths in the dataset: {lengths}")
    return next(iter(lengths.values()))


class Dataset:
    """A dict of equal-length arrays with seeded uniform sampling (EXPO ``Dataset``)."""

    def __init__(self, dataset_dict: Batch, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)
        self._np_random: Optional[np.random.Generator] = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> None:
        # gym.utils.seeding.np_random(seed) == Generator(PCG64(SeedSequence(seed)))
        self._np_random = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> Batch:
        if indx is None:
            indx = self.np_random.integers(len(self), size=batch_size)
        if keys is None:
            keys = self.dataset_dict.keys()
        return {k: self.dataset_dict[k][indx] for k in keys}


class ReplayBuffer(Dataset):
    """Fixed-capacity ring buffer over the six transition keys (EXPO ``RoboReplayBuffer``)."""

    def __init__(self, example_observation: np.ndarray, example_action: np.ndarray, capacity: int):
        example_observation = np.asarray(example_observation)
        example_action = np.asarray(example_action)
        dataset_dict = dict(
            observations=np.empty(
                (capacity, *example_observation.shape), dtype=example_observation.dtype
            ),
            next_observations=np.empty(
                (capacity, *example_observation.shape), dtype=example_observation.dtype
            ),
            actions=np.empty((capacity, *example_action.shape), dtype=example_action.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )
        super().__init__(dataset_dict)
        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: Batch) -> None:
        if set(data_dict.keys()) != set(TRANSITION_KEYS):
            raise KeyError(f"transition keys {sorted(data_dict)} != {sorted(TRANSITION_KEYS)}")
        for k in TRANSITION_KEYS:
            self.dataset_dict[k][self._insert_index] = data_dict[k]
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def insert_dataset(self, dataset: Batch) -> None:
        """Preload an offline dataset into an EMPTY buffer (row order preserved)."""
        if self._size != 0:
            raise RuntimeError("insert_dataset requires an empty buffer")
        n = _check_lengths({k: dataset[k] for k in TRANSITION_KEYS})
        if n > self._capacity:
            raise ValueError(f"dataset ({n}) exceeds buffer capacity ({self._capacity})")
        for k in TRANSITION_KEYS:
            self.dataset_dict[k][:n] = dataset[k]
        self._size = n
        self._insert_index = n % self._capacity

    # --- resume support -------------------------------------------------
    def state(self) -> dict:
        """Filled rows + counters + sampler RNG state (the payload of
        ``mulligan.baselines.rlpd.train.save_resume_state``); safe to pickle across numpy versions."""
        return {
            "data": {k: self.dataset_dict[k][: self._size].copy() for k in TRANSITION_KEYS},
            "size": int(self._size),
            "insert_index": int(self._insert_index),
            "rng_state": self.np_random.bit_generator.state,
        }

    def load_state(self, state: dict) -> None:
        n = state["size"]
        if n > self._capacity:
            raise ValueError(f"saved buffer ({n}) exceeds capacity ({self._capacity})")
        for k in TRANSITION_KEYS:
            self.dataset_dict[k][:n] = state["data"][k]
        self._size = n
        self._insert_index = state["insert_index"]
        self.np_random.bit_generator.state = state["rng_state"]


def combine(one_dict: Batch, other_dict: Batch) -> Batch:
    """Interleave two equal-size batches row-wise (EXPO ``train_robo.combine``)."""
    combined = {}
    for k, v in one_dict.items():
        other = other_dict[k]
        tmp = np.empty((v.shape[0] + other.shape[0], *v.shape[1:]), dtype=v.dtype)
        tmp[0::2] = v
        tmp[1::2] = other
        combined[k] = tmp
    return combined


def make_transition(obs, action, reward, mask, done, next_obs) -> Batch:
    return dict(
        observations=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(action, dtype=np.float32),
        rewards=np.float32(reward),
        masks=np.float32(mask),
        dones=bool(done),
        next_observations=np.asarray(next_obs, dtype=np.float32),
    )
