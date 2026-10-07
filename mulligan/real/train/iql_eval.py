"""Evaluation and visualization utilities for real-world Vision IQL training."""

from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from mulligan.data.transforms import raw_metadata_column
from mulligan.networks.distributional_v import DistributionalVNetwork
from mulligan.networks.q_network import QNetwork
from mulligan.networks.v_network import VNetwork
from mulligan.real.train.iql_returns import discounted_return_target
from mulligan.real.train.run_logging import MediaFile
from mulligan.real.eval.outcome_results import (
    task_effective_prefix_length,
)
from mulligan.training.visualization import create_qv_trajectory_plots


TRAJECTORY_HOLDOUT_KEYS = ("done", "is_valid", "frame_indices")


def task_prefix_mask(holdout_data: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mask rows after first ``done`` or at/after first ``is_valid=False`` per episode."""
    required = {"dataset_indices", "episode_indices", *TRAJECTORY_HOLDOUT_KEYS}
    missing = sorted(required - set(holdout_data))
    if missing:
        raise ValueError(f"task-prefix masking requires holdout keys {missing}")
    n = int(holdout_data["episode_indices"].shape[0])
    for key in required:
        value = holdout_data[key]
        if not isinstance(value, torch.Tensor) or value.shape[0] != n:
            raise ValueError(f"holdout key {key!r} must be a tensor with leading dim {n}")

    dataset = holdout_data["dataset_indices"].long().cpu()
    episode = holdout_data["episode_indices"].long().cpu()
    frame = holdout_data["frame_indices"].long().cpu()
    done = holdout_data["done"].bool().cpu()
    valid = holdout_data["is_valid"].bool().cpu()
    keep = torch.zeros(n, dtype=torch.bool)
    pair_keys = torch.stack([dataset, episode], dim=1)
    for pair in torch.unique(pair_keys, dim=0):
        rows = torch.nonzero((pair_keys == pair).all(dim=1)).squeeze(1)
        order = torch.argsort(frame[rows], stable=True)
        rows = rows[order]
        ordered_frame = frame[rows]
        if len(rows) > 1 and not torch.all(torch.diff(ordered_frame) == 1):
            raise ValueError(
                f"dataset {int(pair[0])} episode {int(pair[1])}: cached frame_index "
                "is not contiguous and strictly increasing"
            )
        effective = task_effective_prefix_length(
            done[rows].tolist(),
            valid[rows].tolist(),
            episode_index=int(pair[1]),
        )
        keep[rows[:effective]] = True
    if not keep.any():
        raise RuntimeError("task-prefix masking removed every holdout row")
    return keep


def truncate_holdout_to_task_prefix(
    holdout_data: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Filter every row-aligned tensor through :func:`task_prefix_mask`."""
    mask = task_prefix_mask(holdout_data)
    n = len(mask)
    filtered: dict[str, torch.Tensor] = {}
    for key, value in holdout_data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n:
            filtered[key] = value[mask.to(value.device)]
        else:
            filtered[key] = value
    stats = {
        "rows_before": n,
        "rows_after": int(mask.sum()),
        "rows_dropped": int((~mask).sum()),
    }
    return filtered, stats


def loaded_frame_index(sub_datasets) -> list[torch.Tensor]:
    """Absolute LeRobot ``index`` of every loaded row, per sub-dataset.

    The per-frame lookups (MC returns, intervention, replay plan) are positional over a
    sub-dataset's loaded rows, while ``batch['index']`` is LeRobot's absolute frame index. The two
    differ when ``--dataset-episodes`` loads an episode subset; :func:`local_frame_index` maps them.
    """
    out = []
    for sub_ds in sub_datasets:
        n = len(sub_ds)
        if "index" not in sub_ds.hf_dataset.column_names:
            # Prebuilt-cache metadata adapters: batches come from the cache, never from LeRobot.
            out.append(torch.arange(n))
            continue
        index = torch.as_tensor(
            raw_metadata_column(sub_ds.hf_dataset, "index"), dtype=torch.long
        ).reshape(-1)[:n]
        if len(index) != n or (n > 1 and not bool((index[1:] > index[:-1]).all())):
            raise ValueError(
                f"{getattr(sub_ds, 'repo_id', '?')}: 'index' column is not strictly increasing "
                f"over the {n} loaded rows"
            )
        out.append(index)
    return out


def local_frame_index(
    index: torch.Tensor,
    dataset_index: torch.Tensor,
    frame_index_by_dataset: list[torch.Tensor],
) -> torch.Tensor:
    """Map absolute ``batch['index']`` values to row positions in their sub-dataset (1-D, CPU)."""
    if index.ndim > 1:
        index = index.reshape(index.shape[0], -1)[:, 0]
    if dataset_index.ndim > 1:
        dataset_index = dataset_index.reshape(dataset_index.shape[0], -1)[:, 0]
    index = index.long().cpu()
    dataset_index = dataset_index.long().cpu()
    local = torch.empty_like(index)
    handled = torch.zeros(index.shape[0], dtype=torch.bool)
    for ds_i, table in enumerate(frame_index_by_dataset):
        rows = dataset_index == ds_i
        if not rows.any():
            continue
        wanted = index[rows]
        pos = torch.searchsorted(table, wanted).clamp(max=max(len(table) - 1, 0))
        if len(table) == 0 or not torch.equal(table[pos], wanted):
            bad = wanted[table[pos] != wanted][:10].tolist() if len(table) else wanted[:10].tolist()
            raise IndexError(f"frame index(es) {bad} are not loaded rows of dataset_index={ds_i}")
        local[rows] = pos
        handled |= rows
    if not handled.all():
        missing = sorted({int(v) for v in dataset_index[~handled].tolist()})
        raise IndexError(f"batch has unknown dataset_index value(s) {missing}")
    return local


def create_augmented_image_grid(
    curr_images: torch.Tensor,
    num_samples: int = 8,
) -> torch.Tensor | None:
    """Build an interleaved camera grid from one batch of current-frame images."""
    if curr_images.ndim != 5:
        return None

    batch_size, n_cameras = curr_images.shape[:2]
    if batch_size == 0 or n_cameras == 0:
        return None

    n = min(num_samples, batch_size)
    interleaved_images = []
    for sample_i in range(n):
        for cam_i in range(n_cameras):
            interleaved_images.append(curr_images[sample_i, cam_i])
    if not interleaved_images:
        return None

    samples_per_row = max(1, n // 2)
    nrow = n_cameras * samples_per_row
    grid = make_grid(
        torch.stack(interleaved_images, dim=0).detach().cpu(),
        nrow=nrow,
        normalize=True,
        value_range=(0, 1),
    )
    return grid


def create_annotated_video(
    q_values: torch.Tensor,
    v_values: torch.Tensor,
    advantages: torch.Tensor,
    episode_indices: torch.Tensor,
    success: torch.Tensor,
    dataset_indices: torch.Tensor,
    original_frame_indices: torch.Tensor,
    repo_ids: list[str],
    camera_keys: list[str],
    step: int,
    output_dir: Path,
    max_episodes: int = 50,
    fps: int = 10,
    chunk_size: int = 1,
    root: Path | None = None,
    episodes_by_repo: dict[str, list[int] | None] | None = None,
    revisions: dict[str, str] | None = None,
) -> str | None:
    """Create a single concatenated annotated video with Q/V/A overlays for holdout episodes.

    All selected episodes are written sequentially into one video file.
    When multiple cameras are available, frames are shown side-by-side.
    ``original_frame_indices`` index the loaded (possibly episode-subset) dataset, so
    ``episodes_by_repo`` must match the ``--dataset-episodes`` selection used in training.
    """
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    unique_episodes = torch.unique(episode_indices).tolist()

    # Select episodes: try to get a mix of success and failure
    success_eps = [ep for ep in unique_episodes if success[episode_indices == ep][0].item() == 1]
    failure_eps = [ep for ep in unique_episodes if success[episode_indices == ep][0].item() == 0]
    rng = np.random.RandomState(step)

    selected = []
    n_fail = min(len(failure_eps), max_episodes // 2)
    n_succ = min(len(success_eps), max_episodes - n_fail)
    if failure_eps:
        selected.extend(rng.choice(failure_eps, size=n_fail, replace=False).tolist())
    if success_eps:
        selected.extend(rng.choice(success_eps, size=n_succ, replace=False).tolist())

    if not selected:
        return None

    # Cache loaded datasets
    loaded_datasets: dict[int, LeRobotDataset] = {}

    video_dir = output_dir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(video_dir / f"annotated_step{step}.mp4")

    cam_size = 320
    n_cams = len(camera_keys)
    canvas_w = cam_size * n_cams
    bar_h = 70
    canvas_h = cam_size + bar_h

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            font_small = font
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_small = font

    writer = imageio.get_writer(video_path, fps=fps)
    total_frames = 0

    for sel_idx, ep_idx in enumerate(selected):
        ep_mask = episode_indices == ep_idx
        ep_q = q_values[ep_mask].numpy()
        ep_v = v_values[ep_mask].numpy()
        ep_a = advantages[ep_mask].numpy()
        ep_ds_indices = dataset_indices[ep_mask]
        ep_frame_indices = original_frame_indices[ep_mask]
        ep_success = success[ep_mask][0].item() == 1
        outcome_str = "SUCCESS" if ep_success else "FAIL"
        status_color = (0, 200, 0) if ep_success else (200, 0, 0)

        current_q, current_v, current_a = ep_q[0], ep_v[0], ep_a[0]

        for i in range(len(ep_q)):
            if i % chunk_size == 0:
                current_q, current_v, current_a = ep_q[i], ep_v[i], ep_a[i]

            ds_idx = ep_ds_indices[i].item()
            frame_idx = ep_frame_indices[i].item()

            if ds_idx not in loaded_datasets:
                ds_root = root / repo_ids[ds_idx] if root else None
                episodes = (episodes_by_repo or {}).get(repo_ids[ds_idx])
                loaded_datasets[ds_idx] = LeRobotDataset(
                    repo_ids[ds_idx],
                    root=ds_root,
                    episodes=episodes,
                    revision=(revisions or {}).get(repo_ids[ds_idx]),
                )

            ds = loaded_datasets[ds_idx]
            sample = ds[frame_idx]

            canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)

            for cam_idx, cam_key in enumerate(camera_keys):
                img_tensor = sample[cam_key]
                if img_tensor.ndim == 4:
                    img_tensor = img_tensor[0]
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np).resize((cam_size, cam_size))
                canvas.paste(img_pil, (cam_idx * cam_size, bar_h))

                cam_label = cam_key.split(".")[-1]
                draw.text(
                    (cam_idx * cam_size + 5, bar_h + cam_size - 18),
                    cam_label,
                    fill=(200, 200, 200),
                    font=font_small,
                )

            is_chunk_boundary = i % chunk_size == 0
            value_color = (255, 255, 255) if is_chunk_boundary else (180, 180, 180)
            chunk_marker = " |" if is_chunk_boundary else ""
            draw.text(
                (8, 5),
                f"Q={current_q:.3f}  V={current_v:.3f}  A={current_a:.3f}{chunk_marker}",
                fill=value_color,
                font=font,
            )
            draw.text(
                (8, 26),
                f"Ep {sel_idx + 1}/{len(selected)} [{outcome_str}]  t={i}/{len(ep_q)}",
                fill=status_color,
                font=font,
            )
            draw.text(
                (8, 47),
                f"train_step={step}  chunk={chunk_size}",
                fill=(140, 140, 140),
                font=font_small,
            )

            writer.append_data(np.array(canvas))
            total_frames += 1

    writer.close()

    if total_frames > 0:
        print(f"  Created annotated video: {len(selected)} episodes, {total_frames} frames")
        return video_path
    return None


@torch.no_grad()
def encode_holdout_images(
    holdout_dl: DataLoader,
    encoder,
    camera_keys: list[str],
    separate_encoders: bool,
    device: str,
    proprio_mean: torch.Tensor | None = None,
    proprio_std: torch.Tensor | None = None,
    image_norm_mean: torch.Tensor | None = None,
    image_norm_std: torch.Tensor | None = None,
    encoder_autocast_bf16: bool = False,
    channels_last: bool = False,
) -> torch.Tensor:
    """Encode holdout images through the (possibly updated) encoder.

    Returns (N, visual_dim + proprio_dim) states tensor on *device*.
    If proprio_mean/std are provided, z-score normalizes the proprio component.
    If image_norm_mean/std are provided, applies the mirrored DP image
    normalization to the raw [0,1] frames before encoding (None = identity).
    encoder_autocast_bf16/channels_last must match the TRAINING-time encoder
    numerics/layout so eval scores the same feature distribution the Q/V heads
    were trained on.
    """
    if (image_norm_mean is None) != (image_norm_std is None):
        raise ValueError("image_norm_mean and image_norm_std must be provided together")
    if image_norm_mean is not None:
        image_norm_mean = image_norm_mean.reshape(1, 3, 1, 1).float().to(device)
        image_norm_std = image_norm_std.reshape(1, 3, 1, 1).float().to(device)

    states_list = []
    for batch in holdout_dl:
        features = []
        for cam_key in camera_keys:
            cam_views = batch[cam_key].to(device, non_blocking=True)[:, 0]
            if image_norm_mean is not None:
                cam_views = (cam_views - image_norm_mean) / image_norm_std
            if channels_last:
                cam_views = cam_views.contiguous(memory_format=torch.channels_last)
            cam_encoder = (
                encoder[cam_key.removeprefix("observation.images.")]
                if separate_encoders
                else encoder
            )
            if encoder_autocast_bf16 and cam_views.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feat = cam_encoder(cam_views)
                feat = feat.float()
            else:
                feat = cam_encoder(cam_views)
            features.append(feat)
        visual = torch.cat(features, dim=-1)

        state_gpu = batch["observation.state"].to(device, non_blocking=True)[:, 0]
        if proprio_mean is not None and proprio_std is not None:
            state_gpu = (state_gpu - proprio_mean) / proprio_std
        states = torch.cat([visual, state_gpu], dim=-1)
        states_list.append(states)

    return torch.cat(states_list, dim=0)


@torch.no_grad()
def evaluate_on_holdout(
    q_networks: list[QNetwork],
    v_network: VNetwork,
    holdout_data: dict[str, torch.Tensor],
    device: str,
    step: int,
    output_dir: Path,
    repo_ids: list[str] | None = None,
    camera_keys: list[str] | None = None,
    max_videos: int = 10,
    chunk_size: int = 1,
    root: Path | None = None,
    gamma: float = 0.99,
    n_random_actions: int = 16,
    episodes_by_repo: dict[str, list[int] | None] | None = None,
    revisions: dict[str, str] | None = None,
) -> dict:
    """Evaluate Q/V on held-out episodes and compute diagnostics.

    holdout_data should contain pre-encoded states with visual features.
    Uses min(Q1, Q2) for Q-value aggregation (consistent with IDQLPolicy).
    Returns a dict of scalar metrics plus :class:`MediaFile` plots/videos (logged to
    W&B by ``RunLogger.log`` when enabled).

    ``output_dir`` is the run/analysis ROOT, NOT a plot dir: this function writes
    ``output_dir/eval_plots/`` and (with ``repo_ids``) ``output_dir/eval_videos/``
    itself. Callers must pass the parent — handing it an already-suffixed
    ``.../eval_plots`` yields ``eval_plots/eval_plots/``.
    """
    for qn in q_networks:
        qn.eval()
    v_network.eval()

    holdout_data, prefix_stats = truncate_holdout_to_task_prefix(holdout_data)

    states = holdout_data["states"].to(device)
    actions = holdout_data["actions"].to(device)
    episode_indices = holdout_data["episode_indices"]
    frame_indices = holdout_data["frame_indices"]
    success = holdout_data["success"]

    q_all = torch.stack([qn(states, actions) for qn in q_networks], dim=0)
    q_values = q_all.min(dim=0).values.squeeze(-1)
    # A distributional (DIVL) V returns categorical logits [N, num_atoms]; the
    # scalar value is its distribution mean. Scalar VNetwork returns [N, 1].
    if isinstance(v_network, DistributionalVNetwork):
        v_values = v_network.expected_value(states).squeeze(-1)
    else:
        v_values = v_network(states).squeeze(-1)
    advantages = q_values - v_values

    q_values_cpu = q_values.cpu()
    v_values_cpu = v_values.cpu()
    advantages_cpu = advantages.cpu()

    # ---- Per-episode trajectory analysis ----
    unique_episodes = torch.unique(episode_indices).tolist()
    q_trajectories = []
    v_trajectories = []
    a_trajectories = []
    successes = []
    v_s0_success = []
    v_s0_failure = []

    for ep_idx in unique_episodes:
        ep_mask = episode_indices == ep_idx
        ep_q = q_values_cpu[ep_mask].tolist()
        ep_v = v_values_cpu[ep_mask].tolist()
        ep_a = advantages_cpu[ep_mask].tolist()
        ep_success = success[ep_mask][0].item() == 1

        q_trajectories.append(ep_q)
        v_trajectories.append(ep_v)
        a_trajectories.append(ep_a)
        successes.append(ep_success)

        if ep_v:
            if ep_success:
                v_s0_success.append(ep_v[0])
            else:
                v_s0_failure.append(ep_v[0])

    # ---- Metrics ----
    metrics = {
        "eval/task_prefix_rows": prefix_stats["rows_after"],
        "eval/post_task_rows_dropped": prefix_stats["rows_dropped"],
    }

    metrics["eval/v_s0_success_mean"] = np.mean(v_s0_success) if v_s0_success else float("nan")
    metrics["eval/v_s0_success_std"] = np.std(v_s0_success) if v_s0_success else float("nan")
    metrics["eval/v_s0_failure_mean"] = np.mean(v_s0_failure) if v_s0_failure else float("nan")
    metrics["eval/v_s0_failure_std"] = np.std(v_s0_failure) if v_s0_failure else float("nan")
    if v_s0_success and v_s0_failure:
        metrics["eval/v_s0_gap"] = np.mean(v_s0_success) - np.mean(v_s0_failure)
    else:
        metrics["eval/v_s0_gap"] = float("nan")

    metrics["eval/q_mean"] = q_values_cpu.mean().item()
    metrics["eval/q_std"] = q_values_cpu.std().item()
    metrics["eval/v_mean"] = v_values_cpu.mean().item()
    metrics["eval/v_std"] = v_values_cpu.std().item()
    metrics["eval/advantage_mean"] = advantages_cpu.mean().item()
    metrics["eval/advantage_std"] = advantages_cpu.std().item()

    metrics["eval/q_max"] = q_values_cpu.max().item()
    metrics["eval/q_min"] = q_values_cpu.min().item()
    metrics["eval/v_max"] = v_values_cpu.max().item()
    metrics["eval/v_min"] = v_values_cpu.min().item()

    success_mask_cpu = success == 1
    failure_mask_cpu = success == 0
    if success_mask_cpu.any():
        metrics["eval/q_mean_success"] = q_values_cpu[success_mask_cpu].mean().item()
        metrics["eval/v_mean_success"] = v_values_cpu[success_mask_cpu].mean().item()
    if failure_mask_cpu.any():
        metrics["eval/q_mean_failure"] = q_values_cpu[failure_mask_cpu].mean().item()
        metrics["eval/v_mean_failure"] = v_values_cpu[failure_mask_cpu].mean().item()

    n_success = sum(successes)
    n_failure = len(successes) - n_success
    metrics["eval/n_holdout_episodes"] = len(successes)
    metrics["eval/n_holdout_success"] = n_success
    metrics["eval/n_holdout_failure"] = n_failure

    # ---- Metric 1: Q-value rank correlation with return-to-go ----
    from scipy.stats import spearmanr

    all_q_flat = []
    all_rtg_flat = []
    for i, ep_idx in enumerate(unique_episodes):
        ep_mask = episode_indices == ep_idx
        ep_q = q_values_cpu[ep_mask]
        ep_frame = frame_indices[ep_mask].long().cpu().numpy()
        ep_success = successes[i]
        if ep_success and not bool(holdout_data["done"][ep_mask][-1]):
            raise RuntimeError(
                f"successful holdout episode {ep_idx} has no cached first-terminal row; "
                "cannot compute terminal-anchored return-to-go"
            )
        steps_to_go = ep_frame[-1] - ep_frame
        rtg = torch.from_numpy(
            discounted_return_target(
                steps_to_go,
                np.full(len(ep_frame), ep_success),
                gamma=gamma,
                n_action_steps=chunk_size,
            )
        )
        all_q_flat.append(ep_q)
        all_rtg_flat.append(rtg)

    all_q_flat = torch.cat(all_q_flat).numpy()
    all_rtg_flat = torch.cat(all_rtg_flat).numpy()
    if len(all_q_flat) > 2:
        rho, p_val = spearmanr(all_q_flat, all_rtg_flat)
        metrics["eval/q_rtg_spearman_rho"] = rho
        metrics["eval/q_rtg_spearman_pval"] = p_val

    # ---- Metric 2: Late-trajectory V separation ----
    late_frac = 0.25
    v_late_success = []
    v_late_failure = []
    for i in range(len(successes)):
        traj = v_trajectories[i]
        if not traj:
            continue
        cutoff = max(1, int(len(traj) * (1 - late_frac)))
        late_vals = traj[cutoff:]
        if successes[i]:
            v_late_success.extend(late_vals)
        else:
            v_late_failure.extend(late_vals)

    if v_late_success and v_late_failure:
        late_succ_mean = np.mean(v_late_success)
        late_fail_mean = np.mean(v_late_failure)
        metrics["eval/v_late_success_mean"] = late_succ_mean
        metrics["eval/v_late_failure_mean"] = late_fail_mean
        metrics["eval/v_late_gap"] = late_succ_mean - late_fail_mean
        pooled_std = np.sqrt(
            (
                np.var(v_late_success) * (len(v_late_success) - 1)
                + np.var(v_late_failure) * (len(v_late_failure) - 1)
            )
            / (len(v_late_success) + len(v_late_failure) - 2)
        )
        if pooled_std > 1e-8:
            metrics["eval/v_late_cohens_d"] = (late_succ_mean - late_fail_mean) / pooled_std

    # ---- Metric 3: Advantage AUROC for classifying success vs failure ----
    from sklearn.metrics import roc_auc_score

    per_frame_labels = success.numpy()
    per_frame_advantages = advantages_cpu.numpy()
    if len(np.unique(per_frame_labels)) == 2:
        metrics["eval/advantage_auroc"] = roc_auc_score(per_frame_labels, per_frame_advantages)

    # Late-trajectory AUROC (last 25% of each episode)
    late_labels = []
    late_advs = []
    for i, ep_idx in enumerate(unique_episodes):
        ep_mask = episode_indices == ep_idx
        ep_a = advantages_cpu[ep_mask].numpy()
        ep_s = success[ep_mask].numpy()
        cutoff = max(1, int(len(ep_a) * (1 - late_frac)))
        late_labels.append(ep_s[cutoff:])
        late_advs.append(ep_a[cutoff:])
    late_labels = np.concatenate(late_labels)
    late_advs = np.concatenate(late_advs)
    if len(np.unique(late_labels)) == 2:
        metrics["eval/advantage_auroc_late"] = roc_auc_score(late_labels, late_advs)

    # ---- Metric 4: Counterfactual Q-spread ----
    max_states_for_spread = min(2048, states.shape[0])
    spread_indices = torch.randperm(states.shape[0])[:max_states_for_spread]
    spread_states = states[spread_indices]
    spread_actions_real = actions[spread_indices]

    with torch.no_grad():
        q_real = (
            torch.stack([qn(spread_states, spread_actions_real) for qn in q_networks], dim=0)
            .min(dim=0)
            .values.squeeze(-1)
        )

        q_randoms = []
        for _ in range(n_random_actions):
            perm = torch.randperm(actions.shape[0])[:max_states_for_spread]
            random_actions = actions[perm].to(device)
            q_rand = (
                torch.stack([qn(spread_states, random_actions) for qn in q_networks], dim=0)
                .min(dim=0)
                .values.squeeze(-1)
            )
            q_randoms.append(q_rand)

        q_all_actions = torch.stack([q_real] + q_randoms, dim=1)
        q_spread_per_state = q_all_actions.std(dim=1)

    metrics["eval/q_spread_mean"] = q_spread_per_state.mean().item()
    metrics["eval/q_spread_std"] = q_spread_per_state.std().item()
    metrics["eval/q_spread_median"] = q_spread_per_state.median().item()

    q_rand_mean = torch.stack(q_randoms, dim=1).mean(dim=1)
    metrics["eval/q_real_vs_random_frac"] = (q_real > q_rand_mean).float().mean().item()

    # ---- Value trajectory plots ----
    plot_path = output_dir / "eval_plots" / f"qv_trajectories_step{step}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    create_qv_trajectory_plots(
        q_trajectories=q_trajectories,
        v_trajectories=v_trajectories,
        successes=successes,
        output_path=plot_path,
        step=step,
        a_trajectories=a_trajectories,
    )
    metrics["eval/qv_trajectories"] = MediaFile(str(plot_path), "image")

    # ---- V(s_0) histogram ----
    if v_s0_success or v_s0_failure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        if v_s0_success:
            ax.hist(
                v_s0_success,
                bins=20,
                alpha=0.6,
                label=f"Success ({len(v_s0_success)})",
                color="green",
            )
        if v_s0_failure:
            ax.hist(
                v_s0_failure,
                bins=20,
                alpha=0.6,
                label=f"Failure ({len(v_s0_failure)})",
                color="red",
            )
        ax.set_xlabel("V(s_0)")
        ax.set_ylabel("Count")
        ax.set_title(f"Initial State Value Distribution (Step {step})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        hist_path = output_dir / "eval_plots" / f"v_s0_histogram_step{step}.png"
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()
        metrics["eval/v_s0_histogram"] = MediaFile(str(hist_path), "image")

    # ---- Value progression ----
    if q_trajectories:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        n_bins = 20

        for is_success, color, label in [(True, "green", "Success"), (False, "red", "Failure")]:
            matching_v = [
                v_trajectories[i] for i in range(len(successes)) if successes[i] == is_success
            ]
            if not matching_v:
                continue

            binned_values = [[] for _ in range(n_bins)]
            for traj in matching_v:
                ep_len = len(traj)
                for t, val in enumerate(traj):
                    bin_idx = min(int(t / ep_len * n_bins), n_bins - 1)
                    binned_values[bin_idx].append(val)

            bin_means = []
            bin_stds = []
            bin_centers = []
            for i in range(n_bins):
                if binned_values[i]:
                    bin_means.append(np.mean(binned_values[i]))
                    bin_stds.append(np.std(binned_values[i]))
                    bin_centers.append((i + 0.5) / n_bins)

            bin_means = np.array(bin_means)
            bin_stds = np.array(bin_stds)
            bin_centers = np.array(bin_centers)

            n_matching = len(matching_v)
            ax.plot(bin_centers, bin_means, color=color, label=f"{label} ({n_matching})")
            ax.fill_between(
                bin_centers, bin_means - bin_stds, bin_means + bin_stds, color=color, alpha=0.15
            )

        ax.set_xlabel("Normalized Episode Time")
        ax.set_ylabel("V(s)")
        ax.set_title(f"Value Progression Over Episode (Step {step})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        prog_path = output_dir / "eval_plots" / f"value_progression_step{step}.png"
        plt.savefig(prog_path, dpi=150, bbox_inches="tight")
        plt.close()
        metrics["eval/value_progression"] = MediaFile(str(prog_path), "image")

    # ---- Annotated video ----
    if repo_ids is not None:
        video_path = create_annotated_video(
            q_values=q_values_cpu,
            v_values=v_values_cpu,
            advantages=advantages_cpu,
            episode_indices=episode_indices,
            success=success,
            dataset_indices=holdout_data["dataset_indices"],
            original_frame_indices=holdout_data["original_frame_indices"],
            repo_ids=repo_ids,
            camera_keys=camera_keys,
            step=step,
            output_dir=output_dir,
            max_episodes=max_videos,
            chunk_size=chunk_size,
            root=root,
            episodes_by_repo=episodes_by_repo,
            revisions=revisions,
        )
        if video_path:
            metrics["eval/annotated_video"] = MediaFile(video_path, "video", fps=10)

    for qn in q_networks:
        qn.train()
    v_network.train()

    return metrics
