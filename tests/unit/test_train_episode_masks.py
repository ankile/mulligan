"""train.py's one-pass episode grouping reproduces the per-episode mask loop it replaced."""

from __future__ import annotations

import torch

from mulligan.data.constants import DataSource, EpisodeOutcome
from mulligan.training.train import _episodes


def _frames(seed: int):
    gen = torch.Generator().manual_seed(seed)
    ds, ep, success, source = [], [], [], []
    for d in range(3):
        for e in range(int(torch.randint(1, 12, (1,), generator=gen))):
            n = int(torch.randint(1, 9, (1,), generator=gen))
            outcome = int(torch.randint(0, 2, (1,), generator=gen))
            human = torch.rand(n, generator=gen) < (0.0 if e % 3 == 0 else 0.2)
            ds += [d] * n
            ep += [e] * n
            success += [outcome] * n
            source += [int(DataSource.HUMAN) if h else int(DataSource.AUTONOMOUS) for h in human]
    order = torch.randperm(len(ds), generator=gen)  # frames need not be contiguous
    return tuple(torch.tensor(x)[order] for x in (ds, ep, success, source))


def _loop_pure(ds, ep, success, source):
    mask = torch.zeros(len(ds), dtype=torch.bool)
    for d, e in torch.unique(torch.stack([ds, ep], dim=1), dim=0):
        m = (ds == d) & (ep == e)
        if success[m][0].item() != EpisodeOutcome.SUCCESS:
            continue
        if not (source[m] == DataSource.HUMAN).any().item():
            mask |= m
    return mask


def test_episode_grouping_matches_the_per_episode_loops():
    for seed in range(5):
        ds, ep, success, source = _frames(seed)
        episodes = _episodes(ds, ep, success, source)

        pure = episodes.pure_autonomous_success
        assert torch.equal(pure[episodes.frame_episode], _loop_pure(ds, ep, success, source))
