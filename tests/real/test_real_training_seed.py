import random

import numpy as np
import torch


def _draw_rng_triplet():
    return random.random(), float(np.random.rand()), torch.rand(4)


def test_temporary_seed_reproducible_and_restores_rng_streams():
    from mulligan.real.train.policy import _temporary_seed

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected_after = _draw_rng_triplet()

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    with _temporary_seed(7):
        seeded_draw_a = _draw_rng_triplet()
    actual_after = _draw_rng_triplet()

    with _temporary_seed(7):
        seeded_draw_b = _draw_rng_triplet()

    assert seeded_draw_a[0] == seeded_draw_b[0]
    assert seeded_draw_a[1] == seeded_draw_b[1]
    torch.testing.assert_close(seeded_draw_a[2], seeded_draw_b[2])
    assert actual_after[0] == expected_after[0]
    assert actual_after[1] == expected_after[1]
    torch.testing.assert_close(actual_after[2], expected_after[2])
