# Based on the EXPO source code (https://github.com/pd-perry/EXPO) and the RLPD source code
# (https://github.com/ikostrikov/rlpd), on which EXPO builds.
# Vendored from EXPO's expo/data/dataset.py. Both projects are MIT-licensed; their notices follow.
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

# Edits: `gym.utils.seeding.np_random(seed)` -> `np.random.default_rng(seed)` (gym 0.23
# builds the same PCG64(SeedSequence(seed)) generator); only `sample` is kept (the
# jax/split/filter helpers are unused by the RLPD loop); flat arrays only.
from typing import Dict, Iterable, Optional

import numpy as np
from flax.core import frozen_dict

DatasetDict = Dict[str, np.ndarray]


def _check_lengths(dataset_dict: DatasetDict) -> int:
    lengths = {len(v) for v in dataset_dict.values()}
    assert len(lengths) == 1, "Inconsistent item lengths in the dataset."
    return lengths.pop()


class Dataset(object):
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)
        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> None:
        self._np_random = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> frozen_dict.FrozenDict:
        if indx is None:
            indx = self.np_random.integers(len(self), size=batch_size)
        if keys is None:
            keys = self.dataset_dict.keys()
        return frozen_dict.freeze({k: self.dataset_dict[k][indx] for k in keys})
