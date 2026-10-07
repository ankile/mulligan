"""Eligibility for the human-data-only critic ablation."""

import torch


def human_chunk_indices(
    human_success: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> torch.Tensor:
    """Select chunks whose every executed action is human-controlled and successful.

    Source labels belong to individual frames, not whole episodes. Testing only
    the anchor would admit autonomous actions after a human-to-policy handoff.
    Padding is ignored. The critic's existing terminal/truncation loss weights
    still apply, so this sampler does not remove terminal success supervision.
    """
    n, horizon = action_is_pad.shape
    assert human_success.shape == (n,)
    indices = torch.arange(n, device=human_success.device)[:, None]
    indices = indices + torch.arange(horizon, device=human_success.device)[None, :]
    within_data = indices < n
    human_actions = human_success[indices.clamp(max=n - 1)] & within_data
    eligible = (human_actions | action_is_pad).all(dim=1)
    eligible &= ~action_is_pad[:, 0]
    return eligible.nonzero(as_tuple=True)[0]
