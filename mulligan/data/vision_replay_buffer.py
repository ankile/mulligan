#!/usr/bin/env python3
"""
Vision Replay Buffer: Budget-based in-memory cache with async refresh.

Allocates a fixed-capacity buffer in RAM (sized by a memory budget), fills it
from a DataLoader, and serves training batches via fast vectorized indexing.
A background DataLoader continuously decodes new samples, replacing old entries
so that the buffer gradually covers the full dataset.

When the dataset fits entirely in the buffer, all frames are eventually cached
and no further MP4 decoding is needed. When the dataset exceeds the buffer,
it acts as a sliding window that amortizes decode cost over many training steps.

Key properties:
  - O(1) batch retrieval via vectorized indexing (~1ms on GPU, ~50ms on CPU)
  - Images stored as uint8 (4x less memory than float32)
  - Buffer can be moved to GPU after filling for 50-300x faster sampling
  - Refresh is pipelined: DataLoader workers decode in the background while
    the GPU trains, so refresh latency is fully hidden
  - No assumptions about fitting the full dataset in memory

Usage:
    buffer = VisionReplayBuffer.from_budget_gb(10.0, camera_keys, ...)
    buffer.fill(dataloader, max_samples=buffer.capacity)
    buffer.to_device(torch.device("cuda"))  # Move to GPU for fast sampling
    for step in range(training_steps):
        batch = buffer.sample(batch_size=256)  # ~1ms on GPU
        # ... train ...
        buffer.refresh(refresh_batch)  # CPU->GPU transfer handled internally
"""

import time

import torch

from mulligan.real.policy.image_preprocess import quantize_float01_to_uint8


# The released critics budgeted 16 bytes per sample for three per-sample slots the trainer never
# read (an MC return, a replay weight and a replay bucket). The slots are gone; the bytes stay in
# the estimate so a given --buffer-capacity-gb holds the same number of samples as in those runs.
_BUDGETED_SLOT_BYTES = 4 + 4 + 8


class VisionReplayBuffer:
    """Fixed-capacity replay buffer for decoded video frames + scalar data.

    Each entry is a self-contained training sample with delta timestamps already
    applied (current + next images, action/reward/done chunks, states, metadata).
    No episode boundary logic needed at sampling time. Sampling is uniform.
    """

    def __init__(
        self,
        capacity: int,
        camera_keys: list[str],
        n_image_timestamps: int,
        img_h: int,
        img_w: int,
        action_chunk_size: int,
        action_dim: int,
        state_dim: int,
        n_state_timestamps: int,
        reward_horizon_size: int | None = None,
    ):
        self.capacity = capacity
        self.camera_keys = camera_keys
        self.size = 0
        self._write_ptr = 0
        self.device = torch.device("cpu")

        # Image storage: uint8 for 4x memory savings
        self.images: dict[str, torch.Tensor] = {}
        for cam in camera_keys:
            self.images[cam] = torch.zeros(
                capacity, n_image_timestamps, 3, img_h, img_w, dtype=torch.uint8
            )

        # Scalar storage
        reward_horizon_size = (
            action_chunk_size if reward_horizon_size is None else reward_horizon_size
        )
        if reward_horizon_size < 1:
            raise ValueError(f"reward_horizon_size must be positive, got {reward_horizon_size}")
        self.actions = torch.zeros(capacity, action_chunk_size, action_dim)
        self.rewards = torch.zeros(capacity, reward_horizon_size)
        self.dones = torch.zeros(capacity, reward_horizon_size)
        self.states = torch.zeros(capacity, n_state_timestamps, state_dim)

        # Metadata (per-sample, optional — filled only if present in data)
        self.success = torch.zeros(capacity, dtype=torch.long)
        self.source = torch.zeros(capacity, dtype=torch.long)
        self.episode_index = torch.zeros(capacity, dtype=torch.long)

        self._bytes_per_sample = self._compute_bytes_per_sample()

    def _compute_bytes_per_sample(self) -> int:
        """Compute memory per sample for reporting."""
        total = 0
        for cam in self.images.values():
            total += cam[0].numel() * cam[0].element_size()
        total += self.actions[0].numel() * self.actions.element_size()
        total += self.rewards[0].numel() * self.rewards.element_size()
        total += self.dones[0].numel() * self.dones.element_size()
        total += self.states[0].numel() * self.states.element_size()
        total += 3 * 8  # success, source, episode_index (long = 8 bytes)
        return total

    @classmethod
    def from_budget_gb(
        cls,
        budget_gb: float,
        camera_keys: list[str],
        n_image_timestamps: int,
        img_h: int,
        img_w: int,
        action_chunk_size: int,
        action_dim: int,
        state_dim: int,
        n_state_timestamps: int,
        reward_horizon_size: int | None = None,
    ) -> "VisionReplayBuffer":
        """Create a buffer sized to fit within a memory budget."""
        # Estimate bytes per sample
        img_bytes = len(camera_keys) * n_image_timestamps * 3 * img_h * img_w  # uint8
        reward_horizon_size = (
            action_chunk_size if reward_horizon_size is None else reward_horizon_size
        )
        scalar_bytes = (
            action_chunk_size * action_dim * 4  # actions float32
            + reward_horizon_size * 4  # rewards float32
            + reward_horizon_size * 4  # dones float32
            + n_state_timestamps * state_dim * 4  # states float32
            + 3 * 8  # metadata longs
            + _BUDGETED_SLOT_BYTES
        )
        bytes_per_sample = img_bytes + scalar_bytes
        budget_bytes = int(budget_gb * (1024**3))
        capacity = budget_bytes // bytes_per_sample
        if capacity <= 0:
            raise ValueError(
                f"Replay buffer budget {budget_gb:.3f} GB is too small for one sample "
                f"({bytes_per_sample:,} bytes per sample)"
            )

        print(f"  Replay buffer: {budget_gb:.1f} GB budget")
        print(f"  Bytes per sample: {bytes_per_sample:,} ({bytes_per_sample / 1024:.1f} KB)")
        print(f"  Capacity: {capacity:,} samples")

        return cls(
            capacity=capacity,
            camera_keys=camera_keys,
            n_image_timestamps=n_image_timestamps,
            img_h=img_h,
            img_w=img_w,
            action_chunk_size=action_chunk_size,
            action_dim=action_dim,
            state_dim=state_dim,
            n_state_timestamps=n_state_timestamps,
            reward_horizon_size=reward_horizon_size,
        )

    def to_device(self, device: torch.device) -> "VisionReplayBuffer":
        """Move all buffer tensors to the given device (e.g. GPU).

        Call after fill() to enable fast GPU-side sampling (~1ms vs ~300ms on CPU).
        Refresh batches from CPU DataLoaders are automatically transferred.
        """
        t0 = time.perf_counter()
        self.device = device
        for cam in list(self.images.keys()):
            self.images[cam] = self.images[cam].to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.dones = self.dones.to(device)
        self.states = self.states.to(device)
        self.success = self.success.to(device)
        self.source = self.source.to(device)
        self.episode_index = self.episode_index.to(device)
        elapsed = time.perf_counter() - t0
        print(f"  Buffer moved to {device} in {elapsed:.1f}s ({self.memory_gb:.1f} GB)")
        return self

    def _to_buf_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move tensor to buffer device if needed.

        non_blocking is a no-op for pageable CPU sources; for pinned sources it
        lets the copy + slice writes enqueue without blocking the caller on
        already-queued GPU work (all ops stay ordered on the buffer's stream,
        so readers on the same stream observe completed writes).
        """
        if t.device != self.device:
            return t.to(self.device, non_blocking=True)
        return t

    @staticmethod
    def pack_images_uint8(imgs: torch.Tensor) -> torch.Tensor:
        """Convert float32 [0,1] images to the buffer's packed uint8 format.

        The quantization formula itself is single-sourced in
        ``mulligan.real.policy.image_preprocess.quantize_float01_to_uint8`` (also used by the
        DP worker-side post-aug quantize); producers may pre-pack on a background
        thread so the buffer write path skips the (large) CPU convert and
        transfers 4x fewer bytes.
        """
        if imgs.dtype == torch.float32:
            return quantize_float01_to_uint8(imgs)
        if imgs.dtype != torch.uint8:
            raise TypeError(f"VisionReplayBuffer images must be uint8 or float32, got {imgs.dtype}")
        return imgs

    def _prep_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert float32 [0,1] images to uint8 and move to buffer device."""
        return self._to_buf_device(self.pack_images_uint8(imgs))

    def _write_batch(self, batch: dict[str, torch.Tensor]) -> int:
        """Write a batch into the circular buffer. Returns number written."""
        n = batch["action"].shape[0]
        if n == 0:
            return 0
        if n > self.capacity:
            raise ValueError(
                f"VisionReplayBuffer _write_batch batch size {n} exceeds capacity {self.capacity}"
            )

        end = self._write_ptr + n
        if end <= self.capacity:
            sl = slice(self._write_ptr, end)
            for cam in self.camera_keys:
                self.images[cam][sl] = self._prep_images(batch[cam])
            self.actions[sl] = self._to_buf_device(batch["action"])
            self.rewards[sl] = self._to_buf_device(batch["reward"])
            self.dones[sl] = self._to_buf_device(batch["done"])
            self.states[sl] = self._to_buf_device(batch["observation.state"])
            self.success[sl] = self._to_buf_device(batch["success"]).long()
            self.source[sl] = self._to_buf_device(batch["source"]).long()
            self.episode_index[sl] = self._to_buf_device(batch["episode_index"]).long()
        else:
            first_n = self.capacity - self._write_ptr
            sl1 = slice(self._write_ptr, self.capacity)
            sl2 = slice(0, n - first_n)

            for cam in self.camera_keys:
                imgs_u8 = self._prep_images(batch[cam])
                self.images[cam][sl1] = imgs_u8[:first_n]
                self.images[cam][sl2] = imgs_u8[first_n:]
            for key, buf in [
                ("action", self.actions),
                ("reward", self.rewards),
                ("done", self.dones),
                ("observation.state", self.states),
            ]:
                val = self._to_buf_device(batch[key])
                buf[sl1] = val[:first_n]
                buf[sl2] = val[first_n:]
            for key, buf in [
                ("success", self.success),
                ("source", self.source),
                ("episode_index", self.episode_index),
            ]:
                val = self._to_buf_device(batch[key]).long()
                buf[sl1] = val[:first_n]
                buf[sl2] = val[first_n:]

        self._write_ptr = end % self.capacity
        self.size = min(self.size + n, self.capacity)
        return n

    def fill(self, dataloader, max_samples: int | None = None) -> float:
        """Fill the buffer from a DataLoader. Returns elapsed seconds.

        Args:
            dataloader: PyTorch DataLoader producing batches with delta_timestamps.
            max_samples: Stop after this many samples (default: fill to capacity).
        """
        if max_samples is None:
            max_samples = self.capacity

        target = min(max_samples, self.capacity)
        t0 = time.perf_counter()
        filled = 0

        for batch in dataloader:
            n = self._write_batch(batch)
            filled += n
            report_every = max(1024, n)
            if filled % report_every < n:
                elapsed = time.perf_counter() - t0
                print(
                    f"    Buffer fill: {filled:,}/{target:,} samples "
                    f"({elapsed:.1f}s, {filled / max(elapsed, 1e-6):.0f} sps)",
                    flush=True,
                )
            if filled >= target:
                break

        elapsed = time.perf_counter() - t0
        print(
            f"  Buffer filled: {self.size:,} samples in {elapsed:.1f}s "
            f"({self.size / max(elapsed, 1e-6):.0f} sps)"
        )
        return elapsed

    def refresh(self, batch: dict[str, torch.Tensor]) -> int:
        """Add a batch of newly decoded samples, replacing oldest entries.

        Call this each training step with a small batch from the DataLoader
        to gradually rotate new data into the buffer.

        Returns number of samples added.
        """
        return self._write_batch(batch)

    def _build_batch_from_indices(
        self,
        idx: torch.Tensor,
        *,
        image_format: str,
    ) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}

        for cam in self.camera_keys:
            imgs = self.images[cam][idx]  # (B, 2, 3, H, W) uint8
            if image_format == "float32":
                imgs = imgs.float().div_(255.0)
            batch[cam] = imgs

        batch["action"] = self.actions[idx]
        batch["reward"] = self.rewards[idx]
        batch["done"] = self.dones[idx]
        batch["observation.state"] = self.states[idx]
        batch["success"] = self.success[idx]
        batch["source"] = self.source[idx]
        batch["episode_index"] = self.episode_index[idx]

        return batch

    def _validate_sample_request(self, batch_size: int, image_format: str) -> None:
        if image_format not in {"float32", "uint8"}:
            raise ValueError(f"image_format must be 'float32' or 'uint8', got {image_format!r}")
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty VisionReplayBuffer")
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from replay buffer with "
                f"only {self.size} filled sample(s). Fill more data before training."
            )

    def sample(self, batch_size: int, *, image_format: str = "float32") -> dict[str, torch.Tensor]:
        """Sample a random batch from the buffer.

        Samples with replacement, as replay buffers conventionally do. A batch
        larger than the filled buffer is rejected so under-filled buffers cannot
        silently dominate a batch with repeated frames.

        Returns dict matching DataLoader output format, on the buffer's device.
        Camera images default to normalized float32 so training cannot
        accidentally consume uint8 0-255 values. Pass image_format="uint8" only
        for callers that explicitly want packed images.
        """
        self._validate_sample_request(batch_size, image_format)
        idx = torch.randint(self.size, (batch_size,), device=self.device)
        return self._build_batch_from_indices(idx, image_format=image_format)

    @property
    def memory_gb(self) -> float:
        """Current memory usage in GB."""
        total = 0
        for cam in self.images.values():
            total += cam.numel() * cam.element_size()
        total += self.actions.numel() * self.actions.element_size()
        total += self.rewards.numel() * self.rewards.element_size()
        total += self.dones.numel() * self.dones.element_size()
        total += self.states.numel() * self.states.element_size()
        total += self.success.numel() * self.success.element_size()
        total += self.source.numel() * self.source.element_size()
        total += self.episode_index.numel() * self.episode_index.element_size()
        return total / (1024**3)

    def compute_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        """Compute per-dimension mean/std for states and actions from filled buffer entries.

        Returns dict with 'state' and 'action' keys, each containing 'mean' and 'std' tensors.
        Stats are computed on CPU regardless of buffer device.
        """
        n = self.size
        if n == 0:
            raise RuntimeError("Cannot compute stats from empty buffer")

        # States: (n, n_timestamps, state_dim) → flatten timestamps
        states_flat = self.states[:n].reshape(-1, self.states.shape[-1]).float().cpu()
        # Actions: (n, chunk_size, action_dim) → flatten chunk steps
        actions_flat = self.actions[:n].reshape(-1, self.actions.shape[-1]).float().cpu()

        return {
            "state": {
                "mean": states_flat.mean(dim=0),
                "std": states_flat.std(dim=0).clamp(min=1e-6),
            },
            "action": {
                "mean": actions_flat.mean(dim=0),
                "std": actions_flat.std(dim=0).clamp(min=1e-6),
            },
        }

    @property
    def fill_pct(self) -> float:
        """Fraction of buffer that is filled."""
        return self.size / self.capacity if self.capacity > 0 else 0.0
