"""Horizon-independent frozen-encoder trajectory cache for real-world IQL."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


FLAT_ENCODED_CACHE_SCHEMA_VERSION = 3

# Metadata keys that encode the *dynamic* horizon/reward-shaping recipe rather
# than the frozen-encoder provenance the flat cache is keyed on. They are
# stripped when deriving the schema-v3 provenance contract so that a single flat
# cache can back multiple horizon/reward configurations. Single-sourced here and
# consumed by mulligan.real.train.critic.flat_cache_expected_metadata.
FLAT_CACHE_DYNAMIC_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "action_horizon",
        "td_horizon",
        "gamma",
        "intervention_negative_reward",
        "reward_shift",
        # Batch-time chunk relativization (--action-mode relative): the cache stores
        # the raw source columns unchanged; cross-mode misuse is caught by the STATIC
        # action_columns/action_dim keys.
        "action_mode",
    }
)


FLAT_CACHE_REQUIRED_FRAME_FIELDS: frozenset[str] = frozenset(
    {
        "proprio",
        "action",
        "reward",
        "done",
        "is_valid",
        "success",
        "source",
        "intervention",
        "dataset_index",
        "episode_index",
        "frame_index",
    }
)


def derive_flat_cache_rows(
    *,
    source_anchor_indices: torch.Tensor,
    target_anchor_indices: torch.Tensor | None = None,
    frame_table: dict[str, torch.Tensor],
    action_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Physical-frame row set + anchor-eligibility mask for a flat cache.

    Single source of truth for *which physical frames the flat cache holds*: every
    source anchor plus every within-episode successor at ``anchor + action_horizon``
    that is not itself already an anchor. ``anchor_eligible[i]`` marks whether
    ``rows[i]`` is a *target* anchor.

    Returns ``(rows, anchor_eligible, boundary_successors_added)`` where ``rows`` is
    a sorted ``LongTensor`` of global frame indices.
    """
    missing_frame = sorted(FLAT_CACHE_REQUIRED_FRAME_FIELDS - set(frame_table))
    if missing_frame:
        raise ValueError(f"flat-cache frame table is missing {missing_frame}")
    if target_anchor_indices is None:
        target_anchor_indices = source_anchor_indices
    source_anchor_indices = source_anchor_indices.to(dtype=torch.long)
    target_anchor_indices = target_anchor_indices.to(dtype=torch.long)
    n_anchors = int(source_anchor_indices.numel())
    if torch.unique(source_anchor_indices).numel() != n_anchors:
        raise ValueError("source anchor indices contain duplicates")
    if torch.unique(target_anchor_indices).numel() != target_anchor_indices.numel():
        raise ValueError("target flat-cache anchor indices contain duplicates")
    frame_count = next(iter(frame_table.values())).shape[0]
    bad_frame_fields = {
        name: tuple(value.shape)
        for name, value in frame_table.items()
        if value.shape[0] != frame_count
    }
    if bad_frame_fields:
        raise ValueError(f"flat-cache frame-table lengths disagree: {bad_frame_fields}")
    if (source_anchor_indices < 0).any() or (source_anchor_indices >= frame_count).any():
        raise IndexError("source anchor index is outside the frame table")
    if (target_anchor_indices < 0).any() or (target_anchor_indices >= frame_count).any():
        raise IndexError("target flat-cache anchor index is outside the frame table")
    source_anchor_set = set(source_anchor_indices.tolist())
    target_anchor_set = set(target_anchor_indices.tolist())
    removed_source_anchors = sorted(source_anchor_set - target_anchor_set)
    if removed_source_anchors:
        raise ValueError(
            "target flat-cache contract removes source anchors: "
            f"count={len(removed_source_anchors)}, examples={removed_source_anchors[:10]}"
        )
    if not frame_table["is_valid"][source_anchor_indices].bool().all():
        raise ValueError("source anchor set contains an invalid frame")
    if not frame_table["is_valid"][target_anchor_indices].bool().all():
        raise ValueError("target flat-cache anchor set contains an invalid frame")

    dataset_index = frame_table["dataset_index"]
    episode_index = frame_table["episode_index"]
    done = frame_table["done"]
    present = set(source_anchor_set)
    boundary_added = 0
    for global_idx in source_anchor_indices.tolist():
        successor = global_idx + action_horizon
        same_episode = (
            successor < frame_count
            and int(dataset_index[successor]) == int(dataset_index[global_idx])
            and int(episode_index[successor]) == int(episode_index[global_idx])
        )
        if not bool(done[global_idx]) and same_episode and successor not in present:
            present.add(successor)
            boundary_added += 1

    rows = torch.tensor(sorted(present), dtype=torch.long)
    missing_target_states = sorted(target_anchor_set - set(rows.tolist()))
    if missing_target_states:
        raise ValueError(
            "physical-frame row set cannot cover the target anchor contract without "
            f"re-encoding: count={len(missing_target_states)}, "
            f"examples={missing_target_states[:10]}"
        )
    anchor_eligible = torch.tensor(
        [int(row) in target_anchor_set for row in rows], dtype=torch.bool
    )
    return rows, anchor_eligible, boundary_added


def _ensure_transitions_constructible(
    cache: "FlatEncodedTrajectoryCache", action_horizon: int
) -> None:
    """Fail loud if any eligible anchor cannot build its H-step transition."""
    for start in range(0, cache.anchor_rows.numel(), 8192):
        try:
            cache.build_transitions(
                cache.anchor_rows[start : start + 8192],
                action_horizon=action_horizon,
                td_horizon=action_horizon,
                current_view=0,
            )
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "flat-cache state union cannot construct every target-contract H-step "
                "transition; re-encoding is required"
            ) from exc


def build_flat_cache_from_physical_frames(
    *,
    visual: torch.Tensor,
    rows: torch.Tensor,
    anchor_eligible: torch.Tensor,
    frame_table: dict[str, torch.Tensor],
    action_horizon: int,
) -> "FlatEncodedTrajectoryCache":
    """Assemble a flat cache from a physical-frame visual bank aligned to ``rows``.

    ``visual`` holds one row per entry of ``rows`` (each unique physical frame
    encoded exactly once, clean + augmented views). Immutable scalar/action data
    comes from the revision-pinned frame table. Fails loud if any eligible anchor
    cannot build its H-step transition.
    """
    if int(visual.shape[0]) != int(rows.numel()):
        raise ValueError(
            f"physical visual bank has {int(visual.shape[0])} rows but the flat frame "
            f"set has {int(rows.numel())}"
        )
    cache = FlatEncodedTrajectoryCache(
        visual=visual,
        proprio=frame_table["proprio"][rows],
        action=frame_table["action"][rows],
        reward=frame_table["reward"][rows],
        done=frame_table["done"][rows],
        is_valid=frame_table["is_valid"][rows],
        anchor_eligible=anchor_eligible,
        success=frame_table["success"][rows],
        source=frame_table["source"][rows],
        intervention=frame_table["intervention"][rows],
        dataset_index=frame_table["dataset_index"][rows],
        episode_index=frame_table["episode_index"][rows],
        frame_index=frame_table["frame_index"][rows],
    )
    _ensure_transitions_constructible(cache, action_horizon)
    return cache


class FlatEncodedTrajectoryCache:
    """Canonical per-frame data with episode-local transition construction.

    Training samples eligible anchors uniformly.
    """

    _TENSOR_NAMES = (
        "visual",
        "proprio",
        "action",
        "reward",
        "done",
        "is_valid",
        "anchor_eligible",
        "success",
        "source",
        "intervention",
        "dataset_index",
        "episode_index",
        "frame_index",
    )

    def __init__(
        self,
        *,
        visual: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        is_valid: torch.Tensor,
        anchor_eligible: torch.Tensor,
        success: torch.Tensor,
        source: torch.Tensor,
        intervention: torch.Tensor,
        dataset_index: torch.Tensor,
        episode_index: torch.Tensor,
        frame_index: torch.Tensor,
    ) -> None:
        n = int(visual.shape[0])
        if visual.ndim not in (2, 3):
            raise ValueError(f"flat visual must have shape (N,D) or (N,V,D), got {visual.shape}")
        if proprio.ndim != 2 or action.ndim != 2:
            raise ValueError("flat proprio/action must have shape (N,D)")
        fields = {
            "proprio": proprio,
            "action": action,
            "reward": reward,
            "done": done,
            "is_valid": is_valid,
            "anchor_eligible": anchor_eligible,
            "success": success,
            "source": source,
            "intervention": intervention,
            "dataset_index": dataset_index,
            "episode_index": episode_index,
            "frame_index": frame_index,
        }
        bad = {name: tuple(value.shape) for name, value in fields.items() if value.shape[0] != n}
        if bad:
            raise ValueError(f"flat cache leading dimensions differ from N={n}: {bad}")
        scalar_names = set(self._TENSOR_NAMES) - {"visual", "proprio", "action"}
        nonscalar = {
            name: tuple(fields[name].shape) for name in scalar_names if fields[name].ndim != 1
        }
        if nonscalar:
            raise ValueError(f"flat cache metadata fields must be one-dimensional: {nonscalar}")
        if n == 0:
            raise ValueError("flat cache cannot be empty")
        all_values = {"visual": visual, **fields}
        for name in ("visual", "proprio", "action", "reward"):
            if not torch.isfinite(all_values[name]).all():
                raise ValueError(f"flat cache field {name!r} contains non-finite values")
        keys = torch.stack((dataset_index.long(), episode_index.long(), frame_index.long()), dim=1)
        if torch.unique(keys, dim=0).shape[0] != n:
            raise ValueError("flat cache contains duplicate (dataset, episode, frame) keys")

        order = sorted(
            range(n),
            key=lambda i: (int(dataset_index[i]), int(episode_index[i]), int(frame_index[i])),
        )
        order_t = torch.tensor(order, dtype=torch.long, device=visual.device)
        for name in self._TENSOR_NAMES:
            value = all_values[name]
            setattr(self, name, value[order_t].contiguous())
        self.size = n
        self.capacity = n
        self.device = self.visual.device
        self.num_visual_views = int(self.visual.shape[1]) if self.visual.ndim == 3 else 1
        self._rebuild_episode_index()

    def _rebuild_episode_index(self) -> None:
        episode_keys = torch.stack((self.dataset_index.long(), self.episode_index.long()), dim=1)
        changes = torch.ones(self.size, dtype=torch.bool, device=self.device)
        changes[1:] = (episode_keys[1:] != episode_keys[:-1]).any(dim=1)
        starts = torch.nonzero(changes, as_tuple=False).flatten()
        ends = torch.cat(
            [starts[1:], torch.tensor([self.size], device=self.device, dtype=torch.long)]
        )
        self.episode_start = torch.empty(self.size, dtype=torch.long, device=self.device)
        self.episode_end = torch.empty(self.size, dtype=torch.long, device=self.device)
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            frames = self.frame_index[start:end]
            if frames.numel() > 1 and not torch.equal(frames[1:], frames[:-1] + 1):
                key = tuple(int(x) for x in episode_keys[start])
                raise ValueError(f"flat cache episode {key} has noncontiguous frame indices")
            self.episode_start[start:end] = start
            self.episode_end[start:end] = end
        self.anchor_rows = torch.nonzero(self.anchor_eligible.bool(), as_tuple=False).flatten()
        if self.anchor_rows.numel() == 0:
            raise ValueError("flat cache has no eligible anchors")

    @property
    def fill_pct(self) -> float:
        # Static in-memory cache: always fully populated.
        return 1.0

    @property
    def memory_gb(self) -> float:
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in self._TENSOR_NAMES
        ) / (1024**3)

    def tensor_payload(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).detach().cpu() for name in self._TENSOR_NAMES}

    @classmethod
    def from_tensor_payload(cls, tensors: dict[str, torch.Tensor]) -> "FlatEncodedTrajectoryCache":
        missing = sorted(set(cls._TENSOR_NAMES) - set(tensors))
        extra = sorted(set(tensors) - set(cls._TENSOR_NAMES))
        if missing or extra:
            raise ValueError(f"flat cache tensor keys mismatch: missing={missing}, extra={extra}")
        return cls(**tensors)

    def to_device(self, device: str | torch.device) -> None:
        for name in self._TENSOR_NAMES:
            setattr(self, name, getattr(self, name).to(device))
        self.device = torch.device(device)
        self._rebuild_episode_index()

    def compute_stats(self, *, action_horizon: int = 6) -> dict[str, dict[str, torch.Tensor]]:
        # Normalization population: both proprio states and the canonicalized
        # action chunk of every eligible anchor.
        batch = self.build_transitions(
            self.anchor_rows,
            action_horizon=action_horizon,
            td_horizon=action_horizon,
            current_view=0,
        )
        state = batch["observation.state"].reshape(-1, self.proprio.shape[-1]).float().cpu()
        action = batch["action"].reshape(-1, self.action.shape[-1]).float().cpu()
        return {
            "state": {"mean": state.mean(0), "std": state.std(0).clamp(min=1e-6)},
            "action": {"mean": action.mean(0), "std": action.std(0).clamp(min=1e-6)},
        }

    def _select_visual(self, rows: torch.Tensor, view: torch.Tensor | int | None) -> torch.Tensor:
        if self.visual.ndim == 2:
            return self.visual[rows]
        if view is None:
            view = torch.randint(self.num_visual_views, (rows.numel(),), device=self.device)
        if isinstance(view, int):
            return self.visual[rows, view]
        return self.visual[rows, view]

    def _select_visual_samples(
        self,
        rows: torch.Tensor,
        views: torch.Tensor,
    ) -> torch.Tensor:
        """Gather K sampled feature banks for each row or row/horizon position."""
        if self.visual.ndim != 3:
            raise ValueError("sampled target views require a multi-view visual cache")
        if views.shape[:-1] != rows.shape:
            raise ValueError(
                "sampled target-view prefix must match row shape, got "
                f"rows={tuple(rows.shape)} views={tuple(views.shape)}"
            )
        expanded_rows = rows.unsqueeze(-1).expand_as(views)
        return self.visual[expanded_rows, views]

    def build_transitions(
        self,
        rows: torch.Tensor,
        *,
        action_horizon: int,
        td_horizon: int,
        current_view: torch.Tensor | int | None = None,
        successor_view: torch.Tensor | int | None = None,
        include_chunk_bootstraps: bool = False,
        independent_target_samples: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Gather episode-local transitions and canonicalize post-terminal steps."""
        if action_horizon < 1 or td_horizon < action_horizon:
            raise ValueError("transition horizons require td_horizon >= action_horizon >= 1")
        if independent_target_samples < 0:
            raise ValueError("independent_target_samples must be >= 0")
        if independent_target_samples > self.num_visual_views:
            raise ValueError(
                "independent_target_samples cannot exceed the cache view count: "
                f"samples={independent_target_samples}, views={self.num_visual_views}"
            )
        if independent_target_samples and self.visual.ndim != 3:
            raise ValueError("independent target views require a multi-view cache")
        rows = rows.to(device=self.device, dtype=torch.long)
        if rows.ndim != 1 or rows.numel() == 0:
            raise ValueError("transition rows must be a nonempty one-dimensional tensor")
        if (rows < 0).any() or (rows >= self.size).any():
            raise IndexError("transition row is outside the flat cache")
        if not self.anchor_eligible[rows].bool().all():
            raise ValueError("transition request contains an ineligible anchor")
        offsets = torch.arange(td_horizon, device=self.device)
        raw_rows = rows[:, None] + offsets[None, :]
        episode_end = self.episode_end[rows, None]
        in_episode = raw_rows < episode_end
        done_window = torch.where(
            in_episode,
            self.done[raw_rows.clamp(max=self.size - 1)].bool(),
            torch.zeros_like(in_episode),
        )
        has_done = done_window.any(dim=1)
        first_done_offset = done_window.long().argmax(dim=1)
        remaining = self.episode_end[rows] - rows - 1
        valid_steps = torch.where(has_done, first_done_offset + 1, remaining).clamp(max=td_horizon)
        if ((~has_done) & (valid_steps < action_horizon)).any():
            raise ValueError("eligible anchor lacks a complete action chunk")
        step_limit = torch.where(has_done, first_done_offset + 1, valid_steps)
        canonical_offsets = torch.minimum(offsets[None, :], (step_limit - 1)[:, None])
        gather_rows = rows[:, None] + canonical_offsets
        if not self.is_valid[gather_rows].bool().all():
            raise ValueError("transition canonicalization reached an invalid frame")

        action_rows = gather_rows[:, :action_horizon]
        reward_rows = gather_rows
        successor_offset = valid_steps
        successor_rows = rows + successor_offset
        terminal_rows = rows + first_done_offset
        successor_rows = torch.where(has_done, terminal_rows, successor_rows)
        if (successor_rows >= self.episode_end[rows]).any():
            raise ValueError("timeout successor is missing from the flat cache")

        if current_view is None and self.visual.ndim == 3:
            current_view = torch.randint(self.num_visual_views, (rows.numel(),), device=self.device)
        if successor_view is None:
            successor_view = current_view
        out = {
            "visual_curr": self._select_visual(rows, current_view),
            "visual_next": self._select_visual(successor_rows, successor_view),
            "observation.state": torch.stack(
                (self.proprio[rows], self.proprio[successor_rows]), dim=1
            ),
            "action": self.action[action_rows],
            "reward": self.reward[reward_rows],
            "intervention": self.intervention[reward_rows],
            "done": self.done[reward_rows].long(),
            "valid_steps": valid_steps,
            "success": self.success[rows],
            "source": self.source[rows],
            "episode_index": self.episode_index[rows],
            "dataset_index": self.dataset_index[rows],
            "frame_index": self.frame_index[rows],
        }
        if independent_target_samples:
            # These IID draws intentionally use replacement: K is a Monte Carlo
            # estimate of the finite-bank nuisance expectation. Target-Q current,
            # H6 successor, and every TD(lambda) future position are independent
            # from the active current view and from one another.
            target_curr_views = torch.randint(
                self.num_visual_views,
                (rows.numel(), independent_target_samples),
                device=self.device,
            )
            target_next_views = torch.randint(
                self.num_visual_views,
                (rows.numel(), independent_target_samples),
                device=self.device,
            )
            out["visual_curr_target"] = self._select_visual_samples(rows, target_curr_views)
            out["visual_next_target"] = self._select_visual_samples(
                successor_rows, target_next_views
            )
            out["target_curr_view_indices"] = target_curr_views
            out["target_next_view_indices"] = target_next_views
        if include_chunk_bootstraps:
            horizons = torch.arange(
                action_horizon,
                td_horizon + 1,
                action_horizon,
                device=self.device,
            )
            # Only complete chunk-aligned horizons participate. H=action_horizon
            # remains valid for an eligible anchor even when a terminal occurs
            # inside that first action chunk; its done mask removes bootstrap.
            valid_chunks = torch.div(valid_steps, action_horizon, rounding_mode="floor").clamp(
                min=1
            )
            horizon_valid = (
                torch.arange(1, horizons.numel() + 1, device=self.device)[None, :]
                <= valid_chunks[:, None]
            )
            done_by_horizon = has_done[:, None] & (first_done_offset[:, None] < horizons[None, :])
            bounded_offsets = torch.minimum(horizons[None, :], valid_steps[:, None])
            bootstrap_offsets = torch.where(
                done_by_horizon,
                first_done_offset[:, None],
                bounded_offsets,
            )
            bootstrap_rows = rows[:, None] + bootstrap_offsets
            if (bootstrap_rows >= self.episode_end[rows, None]).any():
                raise ValueError("chunk bootstrap successor is missing from the flat cache")
            if current_view is None and self.visual.ndim == 3:
                raise RuntimeError("current_view must be resolved before chunk bootstrap gathering")
            flat_bootstrap_rows = bootstrap_rows.reshape(-1)
            if isinstance(successor_view, torch.Tensor):
                bootstrap_view = successor_view[:, None].expand_as(bootstrap_rows).reshape(-1)
            else:
                bootstrap_view = successor_view
            bootstrap_visual = self._select_visual(flat_bootstrap_rows, bootstrap_view)
            out["visual_bootstrap"] = bootstrap_visual.reshape(rows.numel(), horizons.numel(), -1)
            out["observation.bootstrap_state"] = self.proprio[bootstrap_rows]
            out["td_horizons"] = horizons
            out["td_horizon_valid"] = horizon_valid
            if independent_target_samples:
                target_bootstrap_views = torch.randint(
                    self.num_visual_views,
                    (*bootstrap_rows.shape, independent_target_samples),
                    device=self.device,
                )
                out["visual_bootstrap_target"] = self._select_visual_samples(
                    bootstrap_rows, target_bootstrap_views
                )
                out["target_bootstrap_view_indices"] = target_bootstrap_views
        return out

    def _sample_rows(self, batch_size: int) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return self.anchor_rows[
            torch.randint(self.anchor_rows.numel(), (batch_size,), device=self.device)
        ]

    def sample(
        self,
        batch_size: int,
        *,
        action_horizon: int,
        td_horizon: int,
        include_chunk_bootstraps: bool = False,
        independent_target_samples: int = 0,
    ) -> dict:
        return self.build_transitions(
            self._sample_rows(batch_size),
            action_horizon=action_horizon,
            td_horizon=td_horizon,
            include_chunk_bootstraps=include_chunk_bootstraps,
            independent_target_samples=independent_target_samples,
        )


def save_flat_encoded_cache(
    path: Path,
    cache: FlatEncodedTrajectoryCache,
    metadata: dict,
    *,
    normalization_action_horizon: int = 6,
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = {
        "schema_version": FLAT_ENCODED_CACHE_SCHEMA_VERSION,
        "metadata": metadata,
        "tensors": cache.tensor_payload(),
    }
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    stats = cache.compute_stats(action_horizon=normalization_action_horizon)
    manifest = {
        "schema_version": FLAT_ENCODED_CACHE_SCHEMA_VERSION,
        "cache_path": str(path),
        "size_bytes": path.stat().st_size,
        "memory_gb": cache.memory_gb,
        "frames": cache.size,
        "eligible_anchors": int(cache.anchor_rows.numel()),
        "visual_views": cache.num_visual_views,
        "normalization_action_horizon": normalization_action_horizon,
        "normalization": {
            group: {name: value.tolist() for name, value in values.items()}
            for group, values in stats.items()
        },
        "metadata": metadata,
    }
    manifest_path = path.with_suffix(path.suffix + ".json")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    try:
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(manifest_tmp, manifest_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)


# Keys recorded only when set; a cache that has one must not satisfy a run that lacks it (a cache
# of an episode subset is not a cache of the whole repo).
FLAT_CACHE_SYMMETRIC_METADATA_KEYS = ("dataset_episodes",)


def validate_flat_encoded_cache_metadata(actual: dict, expected: dict) -> None:
    """Validate the complete frozen-cache provenance contract with per-key errors."""
    keys = set(expected) | {k for k in FLAT_CACHE_SYMMETRIC_METADATA_KEYS if k in actual}
    mismatch = {
        key: {"found": actual.get(key), "expected": expected.get(key)}
        for key in sorted(keys)
        if actual.get(key) != expected.get(key)
    }
    if mismatch:
        raise ValueError(f"flat encoded cache metadata mismatch: {mismatch}")


def load_flat_encoded_cache_payload(
    path: Path,
) -> tuple[FlatEncodedTrajectoryCache, dict]:
    """Load and structurally validate one schema-v3 cache without external datasets."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != FLAT_ENCODED_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"flat encoded cache schema mismatch: found={payload.get('schema_version')!r}, "
            f"expected={FLAT_ENCODED_CACHE_SCHEMA_VERSION}"
        )
    actual = payload.get("metadata")
    if not isinstance(actual, dict):
        raise ValueError("flat encoded cache is missing metadata")
    tensors = payload.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("flat encoded cache is missing tensors")
    return FlatEncodedTrajectoryCache.from_tensor_payload(tensors), actual


def load_flat_encoded_cache(path: Path, expected_metadata: dict) -> FlatEncodedTrajectoryCache:
    cache, actual = load_flat_encoded_cache_payload(path)
    validate_flat_encoded_cache_metadata(actual, expected_metadata)
    return cache
