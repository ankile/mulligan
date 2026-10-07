"""Shared RNG seeding helper for real-world trainers.

Used by both the policy and the critic trainer so they seed identically from a
single source of truth.
"""

import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
