"""Terminal/valid-prefix contracts for cached Vision-IQL evaluation."""

from __future__ import annotations

import torch

from mulligan.real.train.iql_eval import (
    task_prefix_mask,
    truncate_holdout_to_task_prefix,
)


def _holdout() -> dict[str, torch.Tensor]:
    return {
        "dataset_indices": torch.tensor([0] * 6 + [0] * 5),
        "episode_indices": torch.tensor([10] * 6 + [11] * 5),
        "frame_indices": torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]),
        "done": torch.tensor([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
        "is_valid": torch.tensor([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]),
        "states": torch.arange(22).reshape(11, 2),
        "success": torch.tensor([1] * 6 + [0] * 5),
    }


def test_task_prefix_mask_includes_first_done_and_excludes_first_invalid() -> None:
    mask = task_prefix_mask(_holdout())
    assert mask.tolist() == [True, True, True, False, False, False, True, True, True, False, False]


def test_truncate_holdout_filters_every_row_aligned_tensor() -> None:
    holdout = _holdout()
    holdout["scalar_config"] = torch.tensor(7)
    filtered, stats = truncate_holdout_to_task_prefix(holdout)
    assert filtered["states"].shape == (6, 2)
    assert filtered["frame_indices"].tolist() == [0, 1, 2, 0, 1, 2]
    assert filtered["scalar_config"].item() == 7
    assert stats == {"rows_before": 11, "rows_after": 6, "rows_dropped": 5}


def test_task_prefix_mask_rejects_noncontiguous_cached_trace() -> None:
    holdout = _holdout()
    holdout["frame_indices"][1] = 7
    try:
        task_prefix_mask(holdout)
    except ValueError as exc:
        assert "not contiguous" in str(exc)
    else:
        raise AssertionError("noncontiguous cached trajectory did not fail")
