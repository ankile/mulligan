#!/usr/bin/env python3
"""
Fast dataset loading using PyArrow for vectorized operations.

This module provides fast preprocessing of LeRobot datasets by directly accessing
the underlying Arrow tables and using vectorized operations instead of iterating
through each item one at a time.

Key insight: The Arrow table stores RAW data (one frame per row), while LeRobot's
delta_timestamps logic is applied dynamically in __getitem__. We must handle
current/next state computation ourselves, respecting episode boundaries.
"""

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _fix_repeated_success_rewards(
    rewards: np.ndarray,
    dones: np.ndarray,
    episode_indices: np.ndarray,
    is_valid: np.ndarray,
) -> int:
    """Fix episodes where reward=1.0 is given at every step after task success.

    Robosuite returns reward=1.0 on every step where _check_success() is True
    but does NOT terminate on success. MimicGen-converted data thus has multiple
    consecutive reward=1.0 steps at the end of successful episodes.

    Fix: keep reward only at the first success step, set done=1 there, zero the rest.

    DAgger datasets already have done=1 at the first success step (at most 1
    positive-reward step), so this is a no-op for them.

    Modifies rewards and dones arrays in-place.

    Returns:
        Number of episodes that were fixed.
    """
    n_fixed = 0
    for ep_idx in np.unique(episode_indices):
        ep_mask = (episode_indices == ep_idx) & (is_valid == 1)
        ep_indices = np.where(ep_mask)[0]
        if len(ep_indices) == 0:
            continue

        ep_rewards = rewards[ep_indices]
        positive_mask = ep_rewards > 0
        if positive_mask.sum() <= 1:
            continue

        # Zero out all positive rewards after the first one
        first_positive = np.argmax(positive_mask)
        for j in range(first_positive + 1, len(ep_indices)):
            if ep_rewards[j] > 0:
                rewards[ep_indices[j]] = 0.0
        # Mark the first success step as terminal
        dones[ep_indices[first_positive]] = 1.0
        n_fixed += 1
    return n_fixed


def load_dataset_fast(
    lerobot_dataset: LeRobotDataset,
    intervention_negative_reward: float | None = None,
    reward_shift: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Fast loading of LeRobot dataset into RL transition tensors using PyArrow.

    This is 10-100x faster than iterating a per-sample torch Dataset because:
    1. Direct access to Arrow table columns (vectorized reads)
    2. Vectorized filtering with PyArrow compute
    3. Batch numpy operations for current/next state computation
    4. Single torch.from_numpy conversion

    Only includes transitions with is_valid=1.

    Args:
        lerobot_dataset: LeRobotDataset instance
        intervention_negative_reward: If set, add this negative reward to intervention frames
        reward_shift: Constant added to all rewards after intervention modification (default 0.0)

    Returns:
        Dict with tensors:
            - states: (N, state_dim) current states (robot + env concatenated)
            - actions: (N, action_dim) actions
            - rewards: (N,) rewards
            - next_states: (N, state_dim) next states
            - dones: (N,) done flags
            - source: (N,) data source (0=policy, 1=human)
            - success: (N,) episode success flags
            - intervention: (N,) intervention flags
            - episode_indices: (N,) episode indices
    """
    # Access underlying Arrow table
    table = lerobot_dataset.hf_dataset.data
    n_total = len(table)
    columns = set(table.column_names)

    # Define required columns - fail loudly if missing
    required_columns = [
        "observation.state",
        "observation.environment_state",
        "action",
        "episode_index",
        "is_valid",
        "reward",
        "done",
        "source",
        "success",
    ]

    missing = [col for col in required_columns if col not in columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Available columns: {sorted(columns)}\n"
            f"All datasets must have: {required_columns}"
        )

    # Extract all columns to numpy (vectorized bulk read)
    # Observation columns are fixed_size_list, so to_numpy gives object array of 1D arrays
    obs_state_raw = table.column("observation.state").to_numpy(zero_copy_only=False)
    obs_env_raw = table.column("observation.environment_state").to_numpy(zero_copy_only=False)
    actions_raw = table.column("action").to_numpy(zero_copy_only=False)
    episode_indices = table.column("episode_index").to_numpy(zero_copy_only=False).astype(np.int64)

    # Extract required scalar columns
    is_valid = table.column("is_valid").to_numpy(zero_copy_only=False).astype(np.int64)
    rewards = table.column("reward").to_numpy(zero_copy_only=False).astype(np.float32)
    dones = table.column("done").to_numpy(zero_copy_only=False).astype(np.float32)
    source = table.column("source").to_numpy(zero_copy_only=False).astype(np.int64)
    success = table.column("success").to_numpy(zero_copy_only=False).astype(np.int64)

    # Intervention is optional (only present in DAgger datasets)
    if "intervention" in columns:
        intervention = table.column("intervention").to_numpy(zero_copy_only=False).astype(np.int64)
    else:
        # This is the ONLY optional column - explicitly create zeros
        intervention = np.zeros(n_total, dtype=np.int64)

    # === Fix repeated success rewards in MimicGen-converted data ===
    # Robosuite returns reward=1.0 at every step where _check_success() is True
    # but does NOT terminate on success (only at horizon). MimicGen demos thus have
    # multiple consecutive reward=1.0 steps at episode end. Fix: keep reward only at
    # the first success step, set done=1 there, zero the rest.
    n_fixed = _fix_repeated_success_rewards(rewards, dones, episode_indices, is_valid)
    if n_fixed > 0:
        print(f"  Fixed {n_fixed} episodes with repeated success rewards")

    # Stack observation arrays into 2D matrices (N, state_dim)
    obs_state = np.stack(obs_state_raw).astype(np.float32)  # (N, robot_state_dim)
    obs_env = np.stack(obs_env_raw).astype(np.float32)  # (N, env_state_dim)
    actions = np.stack(actions_raw).astype(np.float32)  # (N, action_dim)

    # Validate dataset format: each episode must end with is_valid=0 (padded frame)
    #
    # Expected format (written by mulligan.sim.collect.teleop):
    # -----------------------------------------------------------------
    # For an episode with T real timesteps, the dataset has T+1 frames:
    #   - Frames 0 to T-1: Real (s, a, r, done) with is_valid=1
    #   - Frame T: Padded frame with the terminal observation s_T and is_valid=0
    #
    # The padded frame exists solely to provide the correct next_state for frame T-1.
    # This ensures: next_state[T-1] = s_T (the actual terminal state)
    #
    # The padded frame has the SAME episode_index as the real frames, so shifting
    # via np.roll gives the correct next_state without any episode boundary logic.
    unique_episodes = np.unique(episode_indices)
    for ep_idx in unique_episodes:
        ep_mask = episode_indices == ep_idx
        ep_is_valid = is_valid[ep_mask]
        if ep_is_valid[-1] != 0:
            raise ValueError(
                f"Episode {ep_idx} does not end with is_valid=0 (padded frame). "
                f"All datasets must use the padded frame format where the last frame "
                f"of each episode has is_valid=0 to provide next_state for the final transition. "
                f"See mulligan.sim.collect.teleop for the expected format."
            )

    # Create next_obs arrays by shifting
    # For frame i, next_state comes from frame i+1
    # The padded frame (is_valid=0) ensures the last real frame gets correct next_state
    next_obs_state = np.roll(obs_state, -1, axis=0)
    next_obs_env = np.roll(obs_env, -1, axis=0)

    # Concatenate robot and environment states
    states = np.concatenate([obs_state, obs_env], axis=1)
    next_states = np.concatenate([next_obs_state, next_obs_env], axis=1)

    # Valid transitions are simply those with is_valid=1
    # Padded frames (is_valid=0) are excluded - they exist only to provide next_state
    valid_mask = is_valid == 1
    n_valid = valid_mask.sum()
    print(f"  Filtered: {n_total:,} -> {n_valid:,} valid transitions")

    # Apply mask to all arrays
    states = states[valid_mask]
    actions = actions[valid_mask]
    rewards = rewards[valid_mask]
    next_states = next_states[valid_mask]
    dones = dones[valid_mask]
    source = source[valid_mask]
    success = success[valid_mask]
    intervention = intervention[valid_mask]
    episode_indices = episode_indices[valid_mask]

    # Apply intervention-based negative reward if enabled
    if intervention_negative_reward is not None:
        intervention_mask = intervention == 1
        rewards[intervention_mask] += intervention_negative_reward

    # Apply global reward shift (e.g., -1.0 to shift [0,1] rewards to [-1,0])
    if reward_shift != 0.0:
        rewards += reward_shift

    # Convert to torch tensors (contiguous for efficiency)
    return {
        "states": torch.from_numpy(np.ascontiguousarray(states)).float(),
        "actions": torch.from_numpy(np.ascontiguousarray(actions)).float(),
        "rewards": torch.from_numpy(np.ascontiguousarray(rewards)).float(),
        "next_states": torch.from_numpy(np.ascontiguousarray(next_states)).float(),
        "dones": torch.from_numpy(np.ascontiguousarray(dones)).float(),
        "source": torch.from_numpy(np.ascontiguousarray(source)).long(),
        "success": torch.from_numpy(np.ascontiguousarray(success)).long(),
        "intervention": torch.from_numpy(np.ascontiguousarray(intervention)).long(),
        "episode_indices": torch.from_numpy(np.ascontiguousarray(episode_indices)).long(),
    }


def load_multiple_datasets_fast(
    lerobot_datasets: list[LeRobotDataset],
    intervention_negative_reward: float | None = None,
    reward_shift: float = 0.0,
) -> dict[str, torch.Tensor]:
    """
    Load multiple LeRobot datasets and concatenate them.

    Args:
        lerobot_datasets: List of LeRobotDataset instances
        intervention_negative_reward: If set, add this negative reward to intervention frames
        reward_shift: Constant added to all rewards after intervention modification (default 0.0)

    Returns:
        Dict with concatenated tensors (same keys as load_dataset_fast)
        Plus: dataset_indices tensor indicating which dataset each sample came from
    """
    all_data = []

    for idx, dataset in enumerate(lerobot_datasets):
        repo_id = dataset.repo_id
        print(f"  Loading dataset {idx + 1}/{len(lerobot_datasets)}: {repo_id}...")
        data = load_dataset_fast(
            dataset,
            intervention_negative_reward=intervention_negative_reward,
            reward_shift=reward_shift,
        )
        # Add dataset index
        data["dataset_indices"] = torch.full((len(data["states"]),), idx, dtype=torch.long)
        all_data.append(data)

    # Concatenate all datasets
    result = {}
    for key in all_data[0].keys():
        result[key] = torch.cat([d[key] for d in all_data], dim=0)

    return result


def prepare_chunked_data(
    data: dict[str, torch.Tensor],
    chunk_size: int,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """
    Prepare data for chunked Q-learning following the QC paper approach.

    For each timestep t, this function:
    1. Creates action chunk [a_t, a_{t+1}, ..., a_{t+chunk_size-1}] with padding
    2. Computes chunk-aligned next_state = state at t+chunk_size
    3. Computes cumulative discounted reward over the chunk
    4. Computes masks (AND of all (1-done) in chunk - 0 if any done)
    5. Computes valid flag (0 if chunk crosses episode boundary)

    The Q-learning target becomes:
        Q(s_t, chunk) = Σ(γ^i * r_{t+i}) + γ^k * masks * V(s_{t+k})

    Chunks crossing episode boundaries are marked invalid (valid=0) and should
    be excluded from critic loss via weighting.

    Args:
        data: Dict from load_multiple_datasets_fast() containing:
            - states: (N, state_dim) current states
            - actions: (N, action_dim) flat actions
            - rewards: (N,) rewards
            - next_states: (N, state_dim) single-step next states (will be replaced)
            - dones: (N,) done flags
            - episode_indices: (N,) episode membership
            - dataset_indices: (N,) dataset membership (for multi-dataset)
            - other tensors that pass through unchanged
        chunk_size: Number of future actions per chunk (horizon k)
        gamma: Discount factor for cumulative reward computation

    Padded actions of a terminal chunk (one that contains done=1) are uniform random in
    [-1, 1], so the Q-function learns invariance to post-terminal actions; padded actions of
    a chunk truncated at the episode end repeat the last valid action.

    Returns:
        Dict with:
            - actions: (N, chunk_size, action_dim) chunked actions
            - action_is_pad: (N, chunk_size) boolean mask (True = padded)
            - next_states: (N, state_dim) chunk-aligned (state at t+chunk_size)
            - rewards: (N, 1) cumulative discounted reward over chunk
            - masks: (N, 1) AND of all (1-done) in chunk (0 if any done)
            - chunk_valid: (N, 1) 0 if chunk crosses episode boundary, 1 otherwise
            - other tensors unchanged (states, source, success, etc.)
    """
    if chunk_size <= 1:
        # No chunking needed, just add temporal dimension
        result = dict(data)
        result["actions"] = data["actions"].unsqueeze(1)  # (N, 1, action_dim)
        result["action_is_pad"] = torch.zeros((len(data["actions"]), 1), dtype=torch.bool)
        # masks = 1 - done (for single step), shape (N, 1)
        result["masks"] = (1.0 - data["dones"]).unsqueeze(-1)
        # rewards stay as-is but with shape (N, 1)
        result["rewards"] = data["rewards"].unsqueeze(-1)
        # All single-step transitions are valid, shape (N, 1)
        result["chunk_valid"] = torch.ones((len(data["actions"]), 1), dtype=torch.float32)
        return result

    actions = data["actions"]  # (N, action_dim)
    states = data["states"]  # (N, state_dim)
    next_states = data["next_states"]  # (N, state_dim) - single-step next states
    rewards = data["rewards"]  # (N,)
    dones = data["dones"]  # (N,)
    episode_indices = data["episode_indices"]  # (N,)
    dataset_indices = data["dataset_indices"]  # (N,)

    N, action_dim = actions.shape
    state_dim = states.shape[1]

    # Pre-allocate output tensors
    chunked_actions = torch.zeros((N, chunk_size, action_dim), dtype=actions.dtype)
    action_is_pad = torch.zeros((N, chunk_size), dtype=torch.bool)
    chunk_next_states = torch.zeros((N, state_dim), dtype=states.dtype)
    chunk_rewards = torch.zeros(N, dtype=rewards.dtype)
    chunk_masks = torch.ones(N, dtype=torch.float32)  # Start with 1 (should bootstrap)
    chunk_valid = torch.ones(N, dtype=torch.float32)  # Start with 1 (valid)

    # Precompute discount factors: [1, gamma, gamma^2, ..., gamma^(k-1)]
    discount_powers = torch.tensor([gamma**i for i in range(chunk_size)], dtype=torch.float32)

    # Process each unique (dataset, episode) pair
    # This is needed because episode_indices may repeat across datasets
    unique_pairs = torch.unique(torch.stack([dataset_indices, episode_indices], dim=1), dim=0)

    for ds_idx, ep_idx in unique_pairs:
        # Get mask for this episode
        ep_mask = (dataset_indices == ds_idx) & (episode_indices == ep_idx)
        ep_indices = ep_mask.nonzero(as_tuple=True)[0]
        ep_len = len(ep_indices)

        # Get episode data
        ep_actions = actions[ep_indices]  # (ep_len, action_dim)
        ep_states = states[ep_indices]  # (ep_len, state_dim)
        ep_next_states = next_states[ep_indices]  # (ep_len, state_dim) - single-step next states
        ep_rewards = rewards[ep_indices]  # (ep_len,)
        ep_dones = dones[ep_indices]  # (ep_len,)
        # next_done[t]: first frame >= t with done=1 (ep_len if none)
        done_frames = np.flatnonzero(ep_dones.cpu().numpy() > 0)
        next_done = np.append(done_frames, ep_len)[np.searchsorted(done_frames, np.arange(ep_len))]

        # For each timestep in episode, create chunked data
        for local_t in range(ep_len):
            global_t = ep_indices[local_t].item()

            # Determine how many valid future steps we have within this episode
            remaining = ep_len - local_t  # includes current timestep
            n_valid = min(chunk_size, remaining)
            n_pad = chunk_size - n_valid

            # === 1. Chunk actions (with padding) ===
            chunked_actions[global_t, :n_valid] = ep_actions[local_t : local_t + n_valid]
            if n_pad > 0:
                action_is_pad[global_t, n_valid:] = True

                # Check if this chunk is terminal (contains done=1)
                chunk_dones_slice = ep_dones[local_t : local_t + n_valid]
                is_terminal_chunk = chunk_dones_slice.max() == 1

                if is_terminal_chunk:
                    # Random uniform [-1, 1] for terminal chunks
                    # This helps Q-function learn invariance to post-terminal actions
                    chunked_actions[global_t, n_valid:] = torch.rand(n_pad, action_dim) * 2 - 1
                else:
                    # Truncated chunk: repeat the last valid action
                    last_valid_action = ep_actions[local_t + n_valid - 1]
                    chunked_actions[global_t, n_valid:] = last_valid_action

            # === 2. Chunk-aligned next_state ===
            # next_state should be state at t + chunk_size
            # If chunk extends beyond episode, use the terminal state from next_states
            if local_t + chunk_size < ep_len:
                # Chunk ends within episode - use state at t+k
                chunk_next_states[global_t] = ep_states[local_t + chunk_size]
            else:
                # Chunk extends to or beyond episode end - use terminal state
                # The terminal state is stored in next_states[last_timestep]
                chunk_next_states[global_t] = ep_next_states[-1]

            # === 3. Cumulative discounted reward over chunk ===
            # reward = sum(gamma^i * r[t+i]) for i up to and including the first
            # done in the chunk (rewards after a terminal belong to no return).
            n_reward = min(n_valid, int(next_done[local_t]) - local_t + 1)
            chunk_rewards_slice = ep_rewards[local_t : local_t + n_reward]
            chunk_rewards[global_t] = (chunk_rewards_slice * discount_powers[:n_reward]).sum()

            # === 4. Masks: AND of all (1-done) in chunk ===
            # If any timestep in chunk has done=1, masks=0 (don't bootstrap)
            chunk_dones_slice = ep_dones[local_t : local_t + n_valid]
            # masks = 0 if any done in chunk, else 1
            chunk_masks[global_t] = 1.0 - chunk_dones_slice.max()

            # === 5. Valid: check if chunk crosses episode boundary ===
            # A chunk is valid if it can be fully executed within the episode
            # i.e., we have at least chunk_size timesteps remaining
            # If n_valid < chunk_size, the chunk is truncated (crosses boundary)
            if n_valid < chunk_size:
                chunk_valid[global_t] = 0.0

    # Create result dict with all chunked data
    # rewards, masks, chunk_valid are (N, 1) for consistent shapes in policy code
    result = dict(data)
    result["actions"] = chunked_actions
    result["action_is_pad"] = action_is_pad
    result["next_states"] = chunk_next_states
    result["rewards"] = chunk_rewards.unsqueeze(-1)  # (N, 1)
    result["masks"] = chunk_masks.unsqueeze(-1)  # (N, 1)
    result["chunk_valid"] = chunk_valid.unsqueeze(-1)  # (N, 1)

    return result


def refresh_random_padding_for_batch(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Refresh random padding for terminal chunks in a batch.

    This should be called at each training step to ensure the Q-function sees
    different random actions for padded positions, helping it learn true invariance
    to post-terminal actions rather than overfitting to specific random values.

    Args:
        batch: Dict containing at minimum:
            - action: (B, chunk_size, action_dim) chunked actions
            - action_is_pad: (B, chunk_size) boolean mask (True = padded)
            - masks: (B, 1) bootstrap mask (0 = terminal chunk, 1 = should bootstrap)

    Returns:
        Same batch dict with refreshed random padding for terminal chunks.
    """
    # Handle both "action" (batch convention) and "actions" (dataset convention)
    action_key = "action" if "action" in batch else "actions"
    if action_key not in batch:
        return batch

    actions = batch[action_key]  # (B, chunk_size, action_dim)

    # Check if we have the required keys
    if "action_is_pad" not in batch or "masks" not in batch:
        return batch

    action_is_pad = batch["action_is_pad"]  # (B, chunk_size)
    masks = batch["masks"]  # (B, 1) or (B,)

    # Ensure masks is 1D for broadcasting
    if masks.ndim > 1:
        masks = masks.squeeze(-1)  # (B,)

    # Terminal chunks have masks=0
    # We want to refresh padding for positions where:
    # - action_is_pad=True (this position is padded)
    # - masks=0 (this is a terminal chunk, not truncated)
    is_terminal = masks == 0  # (B,)

    # Create mask for positions to refresh: terminal chunk AND padded position
    # Broadcast: is_terminal (B,) -> (B, 1) to match action_is_pad (B, chunk_size)
    refresh_mask = is_terminal.unsqueeze(-1) & action_is_pad  # (B, chunk_size)

    if refresh_mask.any():
        # Generate fresh random actions for all positions that need refresh
        # actions shape: (B, chunk_size, action_dim)
        # refresh_mask shape: (B, chunk_size)
        # We need to expand refresh_mask to (B, chunk_size, action_dim) for indexing
        refresh_mask_expanded = refresh_mask.unsqueeze(-1).expand_as(actions)

        # Generate random values and apply only to refresh positions
        random_actions = torch.rand_like(actions) * 2 - 1  # uniform [-1, 1]
        actions = torch.where(refresh_mask_expanded, random_actions, actions)
        batch[action_key] = actions

    return batch
