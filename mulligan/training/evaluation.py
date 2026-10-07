"""Policy evaluation utilities."""

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mulligan.agents.idql import IDQLPolicy
from mulligan.sim.envs import resolve_env_name
from mulligan.sim.vec_env import AsyncVectorEnv, SyncVectorEnv
from mulligan.training.normalization import Normalizer
from mulligan.training.visualization import create_qv_trajectory_plots


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError(f"value must be >= 1, got {value}")
    return 1 << (value - 1).bit_length()


def _episode_reset_seed(seed: int, episode_idx: int) -> int:
    if episode_idx < 0:
        raise ValueError(f"episode_idx must be >= 0, got {episode_idx}")
    return int(np.random.SeedSequence([seed, episode_idx]).generate_state(1)[0])


def make_fixed_eval_initial_states(
    env_name: str,
    num_episodes: int,
    seed: int,
) -> list[dict[str, object]]:
    """Create a deterministic Sobol initial-state list for paired evaluations.

    ``env_name`` is a public task name (``square_narrow``) or a robosuite env ID.
    """
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    env_name = resolve_env_name(env_name)

    if env_name == "NutAssemblySquare":
        from mulligan.sampling.sobol import SobolSampler

        sampler = SobolSampler(
            seed=seed,
            batch_size=_next_power_of_two(num_episodes),
            include_boundary=False,
        )
        return [
            {
                "env_name": env_name,
                "nut": tuple(point),
                "reset_seed": _episode_reset_seed(seed, idx),
            }
            for idx, point in enumerate(sampler.planned_points[:num_episodes])
        ]

    if env_name == "Square_D1":
        from mulligan.sampling.sobol import SquareD1SobolSampler

        batch_size = _next_power_of_two(num_episodes * 2)
        while True:
            sampler = SquareD1SobolSampler(seed=seed, batch_size=batch_size)
            if len(sampler.planned_points) >= num_episodes:
                break
            batch_size *= 2
        return [
            {
                "env_name": env_name,
                "nut": tuple(nut),
                "peg": tuple(peg),
                "reset_seed": _episode_reset_seed(seed, idx),
            }
            for idx, (nut, peg) in enumerate(sampler.planned_points[:num_episodes])
        ]

    raise ValueError(
        f"Fixed Sobol eval initial states are not implemented for env_name={env_name!r}"
    )


def _reset_eval_batch(
    vec_env: Union[AsyncVectorEnv, SyncVectorEnv],
    fixed_initial_states: Optional[list[dict[str, object]]],
    completed_episodes: int,
    num_episodes: int,
) -> List[dict]:
    if fixed_initial_states is None:
        return vec_env.reset()

    if len(fixed_initial_states) < num_episodes:
        raise ValueError(
            f"Expected at least {num_episodes} fixed eval states, got {len(fixed_initial_states)}"
        )

    state_specs = []
    for env_idx in range(len(vec_env)):
        episode_idx = min(completed_episodes + env_idx, num_episodes - 1)
        state_specs.append(fixed_initial_states[episode_idx])
    return vec_env.reset_to_eval_initial_states(state_specs)


class StreamingVideoWriter:
    """
    Writes video frames to disk as episodes complete, avoiding memory accumulation.

    Annotates frames with episode outcome, step number, and optional Q/V values,
    then immediately writes them to the video file.
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 20,
        max_episodes: int = 0,  # 0 = unlimited
        num_total_episodes: int = 0,  # Total episodes being evaluated (for display)
    ):
        """
        Initialize streaming video writer.

        Args:
            output_path: Path to output video file
            fps: Frames per second for video
            max_episodes: Maximum number of episodes to record (0 = unlimited)
            num_total_episodes: Total number of episodes being evaluated (for annotation)
        """
        self.output_path = output_path
        self.fps = fps
        self.max_episodes = max_episodes
        self.num_total_episodes = num_total_episodes
        self.episodes_written = 0
        self.frames_written = 0
        self.writer = None
        self._font = None

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_font(self):
        """Load font lazily."""
        if self._font is None:
            try:
                self._font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except (OSError, IOError):
                # Try Linux font paths
                try:
                    self._font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
                    )
                except (OSError, IOError):
                    self._font = ImageFont.load_default()
        return self._font

    def _ensure_writer_open(self):
        """Open video writer if not already open."""
        if self.writer is None:
            self.writer = imageio.get_writer(self.output_path, fps=self.fps)

    def should_record(self) -> bool:
        """Check if we should record more episodes."""
        if self.max_episodes == 0:
            return True
        return self.episodes_written < self.max_episodes

    def write_episode_frames(
        self,
        frames: List[tuple],
        episode_idx: int,
        is_success: bool,
        has_qv_values: bool = False,
    ):
        """
        Annotate and write all frames for a completed episode.

        Args:
            frames: List of frame tuples. Format depends on has_qv_values:
                - If False: (frame, step_num, q_value), or (frame, step_num) for
                  policies without a critic (no Q line)
                - If True: (frame, step_num, q_value, v_value, advantage)
            episode_idx: 0-indexed episode number
            is_success: Whether episode succeeded
            has_qv_values: Whether frames include V and advantage values
        """
        if not self.should_record():
            return

        if not frames:
            return

        self._ensure_writer_open()
        font = self._get_font()

        # Determine display numbers
        episode_display = episode_idx + 1
        total_display = self.num_total_episodes if self.num_total_episodes > 0 else "?"

        # Status text and color
        status_text = "SUCCESS" if is_success else "FAIL"
        status_color = (0, 255, 0) if is_success else (255, 0, 0)

        for frame_data in frames:
            if has_qv_values:
                frame, step_num, q_value, v_value, advantage = frame_data
            else:
                frame, step_num, *rest = frame_data
                q_value = rest[0] if rest else None
                v_value = None
                advantage = None

            # Convert frame to PIL Image
            frame_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame_pil)

            # Draw multi-line text overlay
            x = 10
            y = 10
            dy = 20  # Line spacing

            draw.text(
                (x, y),
                f"Episode {episode_display}/{total_display}",
                fill=(255, 255, 255),
                font=font,
            )
            y += dy
            draw.text((x, y), f"Step {step_num}", fill=(255, 255, 255), font=font)
            y += dy
            draw.text((x, y), status_text, fill=status_color, font=font)
            if q_value is not None:
                y += dy
                draw.text((x, y), f"Q-value: {q_value:.3f}", fill=(0, 0, 0), font=font)

            if has_qv_values and v_value is not None:
                y += dy
                draw.text((x, y), f"V-value: {v_value:.3f}", fill=(0, 0, 0), font=font)
                y += dy
                draw.text((x, y), f"A-value: {advantage:.3f}", fill=(0, 0, 0), font=font)

            # Write frame to video
            self.writer.append_data(np.array(frame_pil))
            self.frames_written += 1

        self.episodes_written += 1

    def close(self):
        """Close the video writer and return path if frames were written."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            if self.frames_written > 0:
                return self.output_path
        return None


@dataclass
class EpisodeLog:
    """Log data for a single episode for computing value function diagnostics."""

    rewards: list[float]  # len T, rewards r_0..r_{T-1}
    q_values: list[float]  # len T, Q(s_t, a_t) for executed actions
    v_values: list[float]  # len T, V(s_t)
    states: list[np.ndarray]  # len T+1, states s_0..s_T
    success: bool  # Whether episode succeeded
    done_reason: str  # "success" | "timeout" | "failure"


def get_robot_state_keys(obs: dict) -> list[str]:
    """
    Get the appropriate robot state keys based on whether the task is bimanual.

    Args:
        obs: Observation dictionary from the environment

    Returns:
        List of state keys to extract from observations
    """
    state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    if "robot1_eef_pos" in obs:
        # Bimanual task - add robot1 keys
        state_keys.extend(["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"])
    return state_keys


def compute_mc_returns_for_episode(
    ep: EpisodeLog,
    policy: IDQLPolicy,
    normalizer: Normalizer,
    gamma: float,
    device: str,
) -> np.ndarray:
    """
    Compute Monte Carlo returns for an episode with proper bootstrapping.

    Args:
        ep: Episode log with rewards, states, and done_reason
        policy: IQL policy for computing bootstrap values
        normalizer: State normalizer
        gamma: Discount factor
        device: Device for torch computation

    Returns:
        Array of MC returns G[t] for each timestep t
    """
    T = len(ep.rewards)
    G = np.zeros(T, dtype=np.float32)

    # Compute bootstrap value for timeout episodes
    boot_final = 0.0
    if ep.done_reason == "timeout":
        s_T = ep.states[-1]  # Final state
        with torch.no_grad():
            s_T_tensor = torch.from_numpy(s_T).float().unsqueeze(0).to(device)
            s_T_norm = normalizer.normalize_state(s_T_tensor)
            v_T = policy.compute_v_value(s_T_norm)
        boot_final = float(v_T.cpu().numpy())

    # Compute return for final timestep
    if ep.done_reason == "success" or ep.done_reason == "failure":
        # Terminal state: no bootstrap
        G[-1] = ep.rewards[-1]
    elif ep.done_reason == "timeout":
        # Non-terminal: bootstrap from final state value
        G[-1] = ep.rewards[-1] + gamma * boot_final

    # Backward recursion for all other timesteps
    for t in reversed(range(T - 1)):
        G[t] = ep.rewards[t] + gamma * G[t + 1]

    return G


def calibration_stats(q_preds: np.ndarray, G_vals: np.ndarray, num_bins: int = 10) -> dict:
    """
    Compute calibration statistics for Q-values vs realized returns.

    Args:
        q_preds: Array of predicted Q-values
        G_vals: Array of realized returns (MC targets)
        num_bins: Number of bins for calibration

    Returns:
        Dict with calibration metrics including ECE (Expected Calibration Error)
    """
    if len(q_preds) == 0:
        return {
            "q_mean": np.array([]),
            "G_mean": np.array([]),
            "counts": np.array([]),
            "ece_q": 0.0,
        }

    bins = np.linspace(q_preds.min(), q_preds.max(), num_bins + 1)
    indices = np.digitize(q_preds, bins) - 1  # 0..num_bins-1
    indices = np.clip(indices, 0, num_bins - 1)  # Ensure valid range

    q_mean = []
    G_mean = []
    counts = []

    for k in range(num_bins):
        mask = indices == k
        if not np.any(mask):
            continue
        q_mean.append(q_preds[mask].mean())
        G_mean.append(G_vals[mask].mean())
        counts.append(mask.sum())

    q_mean = np.array(q_mean)
    G_mean = np.array(G_mean)
    counts = np.array(counts)

    if len(counts) == 0:
        ece_q = 0.0
    else:
        N = counts.sum()
        ece_q = float(np.sum(counts / N * np.abs(q_mean - G_mean)))

    return {
        "q_mean": q_mean,
        "G_mean": G_mean,
        "counts": counts,
        "ece_q": ece_q,
    }


def compute_q_diagnostics(
    episode_logs: list[EpisodeLog],
    policy: IDQLPolicy,
    normalizer: Normalizer,
    gamma: float,
    device: str,
) -> dict:
    """
    Compute comprehensive Q-value diagnostics from episode logs.

    Computes:
    - MC-based over/underestimation and error metrics
    - TD/Bellman error statistics
    - Calibration (Expected Calibration Error)
    - Start-state overestimation
    - Stability statistics

    Args:
        episode_logs: List of episode logs
        policy: IQL policy for computing values
        normalizer: State normalizer
        gamma: Discount factor
        device: Device for torch computation

    Returns:
        Dict with all diagnostic metrics and arrays for histograms
    """
    if len(episode_logs) == 0:
        # Return empty metrics if no episodes
        return {
            # MC-based metrics
            "mc/bias": 0.0,
            "mc/over_rate": 0.0,
            "mc/over_mag": 0.0,
            "mc/under_rate": 0.0,
            "mc/under_mag": 0.0,
            "mc/mse": 0.0,
            "mc/rmse": 0.0,
            "mc/mae": 0.0,
            # TD error metrics
            "td/mean": 0.0,
            "td/mse": 0.0,
            "td/mae": 0.0,
            # Calibration metrics
            "calibration/ece": 0.0,
            # Start-state metrics
            "start_state/bias": 0.0,
            "start_state/abs_bias": 0.0,
            # Stability metrics
            "stability/max_abs": 0.0,
            "stability/mean": 0.0,
            "stability/std": 0.0,
            # Candidate-ranking metrics
            "ranking/q0_success_auroc": float("nan"),
            "ranking/adv0_success_auroc": float("nan"),
            "ranking/v0_success_auroc": float("nan"),
            "ranking/q0_return_spearman": float("nan"),
            # Histogram arrays
            "histograms/mc_deltas": np.array([]),
            "histograms/td_errors": np.array([]),
            "histograms/q_values": np.array([]),
            "histograms/mc_returns": np.array([]),
            # Per-episode arrays
            "per_episode/q0": np.array([]),
            "per_episode/v0": np.array([]),
            "per_episode/g0": np.array([]),
            "per_episode/success": np.array([]),
        }

    # 1. Compute MC returns and Q-MC deltas for all episodes
    all_deltas = []
    all_q = []
    all_G = []
    init_deltas = []
    init_G = []

    for ep in episode_logs:
        G = compute_mc_returns_for_episode(ep, policy, normalizer, gamma, device)
        q_pred = np.asarray(ep.q_values, dtype=np.float32)
        delta = q_pred - G

        all_deltas.append(delta)
        all_q.append(q_pred)
        all_G.append(G)

        # Start-state overestimation
        init_delta = ep.q_values[0] - G[0]
        init_deltas.append(init_delta)
        init_G.append(G[0])

    all_deltas = np.concatenate(all_deltas)
    all_q = np.concatenate(all_q)
    all_G = np.concatenate(all_G)
    init_deltas = np.asarray(init_deltas, dtype=np.float32)

    # MC-based over/underestimation statistics
    over_rate = float(np.mean(all_deltas > 0))
    under_rate = float(np.mean(all_deltas < 0))
    over_mag = float(np.mean(np.maximum(all_deltas, 0.0)))
    under_mag = float(np.mean(np.maximum(-all_deltas, 0.0)))
    q_mc_bias = float(np.mean(all_deltas))

    # MC error statistics
    mse_mc = float(np.mean(all_deltas**2))
    rmse_mc = float(np.sqrt(mse_mc))
    mae_mc = float(np.mean(np.abs(all_deltas)))

    # Start-state statistics
    init_bias = float(np.mean(init_deltas))
    init_abs_bias = float(np.mean(np.abs(init_deltas)))

    # Stability statistics
    q_max_abs = float(np.max(np.abs(all_q)))
    q_mean = float(np.mean(all_q))
    q_std = float(np.std(all_q))

    # 2. Compute TD errors
    td_errors = []

    for ep in episode_logs:
        T = len(ep.rewards)
        states_np = np.stack(ep.states, axis=0)  # (T+1, state_dim)

        with torch.no_grad():
            s_tensor = torch.from_numpy(states_np).float().to(device)
            s_norm = normalizer.normalize_state(s_tensor)
            # Compute V(s_t) for all states
            v_all = policy.compute_v_value(s_norm).cpu().numpy().reshape(-1)

        q_pred = np.asarray(ep.q_values, dtype=np.float32)

        for t in range(T):
            r_t = ep.rewards[t]
            if t == T - 1:
                # Final timestep
                if ep.done_reason == "timeout":
                    target = r_t + gamma * v_all[t + 1]  # Bootstrap from V(s_T)
                else:  # success/failure terminal
                    target = r_t
            else:
                # Intermediate timestep
                target = r_t + gamma * v_all[t + 1]  # Bootstrap from V(s_{t+1})

            td_errors.append(target - q_pred[t])

    td_errors = np.asarray(td_errors, dtype=np.float32)

    # TD error statistics
    td_mean = float(np.mean(td_errors))
    td_mse = float(np.mean(td_errors**2))
    td_mae = float(np.mean(np.abs(td_errors)))

    # 3. Calibration statistics
    calib = calibration_stats(all_q, all_G, num_bins=10)
    ece_q = calib["ece_q"]

    # 4. Candidate-ranking diagnostics: does the value's score of
    # the EXECUTED first chunk rank episodes by realized outcome? Across episodes,
    # AUROC(q0 -> success) conflates state difficulty with action choice;
    # AUROC(q0 - v0 -> success) controls for state difficulty via the value's own
    # baseline; AUROC(v0 -> success) is the state-difficulty-only reference.
    q0 = np.asarray([ep.q_values[0] for ep in episode_logs], dtype=np.float32)
    v0 = np.asarray([ep.v_values[0] for ep in episode_logs], dtype=np.float32)
    g0 = np.asarray(init_G, dtype=np.float32)
    ep_success = np.asarray([ep.success for ep in episode_logs], dtype=np.int64)

    def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        from scipy.stats import rankdata

        ranks = rankdata(scores)
        rank_sum_pos = float(ranks[labels == 1].sum())
        return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    from scipy.stats import spearmanr

    if len(q0) >= 3 and np.std(q0) > 0 and np.std(g0) > 0:
        rho, _ = spearmanr(q0, g0)
        q0_return_spearman = float(rho)
    else:
        q0_return_spearman = float("nan")

    return {
        # MC-based metrics (Q vs Monte Carlo returns)
        "mc/bias": q_mc_bias,
        "mc/over_rate": over_rate,
        "mc/over_mag": over_mag,
        "mc/under_rate": under_rate,
        "mc/under_mag": under_mag,
        "mc/mse": mse_mc,
        "mc/rmse": rmse_mc,
        "mc/mae": mae_mc,
        # TD error metrics (Q vs one-step Bellman target)
        "td/mean": td_mean,
        "td/mse": td_mse,
        "td/mae": td_mae,
        # Calibration metrics
        "calibration/ece": ece_q,
        # Start-state metrics
        "start_state/bias": init_bias,
        "start_state/abs_bias": init_abs_bias,
        # Stability metrics
        "stability/max_abs": q_max_abs,
        "stability/mean": q_mean,
        "stability/std": q_std,
        # Candidate-ranking metrics (per-episode start-state scores vs outcome)
        "ranking/q0_success_auroc": _auroc(q0, ep_success),
        "ranking/adv0_success_auroc": _auroc(q0 - v0, ep_success),
        "ranking/v0_success_auroc": _auroc(v0, ep_success),
        "ranking/q0_return_spearman": q0_return_spearman,
        # Histogram arrays
        "histograms/mc_deltas": all_deltas,
        "histograms/td_errors": td_errors,
        "histograms/q_values": all_q,
        "histograms/mc_returns": all_G,
        # Per-episode arrays (episode-index aligned; for offline metric analysis)
        "per_episode/q0": q0,
        "per_episode/v0": v0,
        "per_episode/g0": g0,
        "per_episode/success": ep_success,
    }


def evaluate_iql_policy_with_qv_values(
    policy: IDQLPolicy,
    vec_env: Union[AsyncVectorEnv, SyncVectorEnv],
    normalizer: Normalizer,
    num_episodes: int = 10,
    max_steps: int = 400,
    device: str = "cpu",
    gamma: float = 0.99,
    save_video: bool = False,
    save_qv_plots: bool = True,
    output_dir: str = "./videos",
    step: Optional[int] = None,
    max_video_episodes: int = 20,
    fixed_initial_states: Optional[list[dict[str, object]]] = None,
) -> tuple[dict, Optional[str], Optional[str]]:
    """
    Evaluate an IDQL/DIVL policy in vectorized environments, tracking Q and V values.

    All environments run in parallel until they all complete their episodes.
    Then all environments are batch reset together. This ensures clean episode boundaries.

    Uses streaming video writing to avoid OOM when evaluating many episodes - frames are
    annotated and written to disk immediately after each episode completes, rather than
    accumulating all frames in memory.

    Args:
        policy: IDQL policy to evaluate (includes actor, critic, and value networks)
        vec_env: Vectorized robosuite environment (AsyncVectorEnv or SyncVectorEnv)
        normalizer: Normalizer for state normalization and action denormalization
        num_episodes: Total number of episodes to evaluate
        max_steps: Maximum steps per episode
        device: Device to run policy on
        save_video: Whether to save rollout videos (concatenated into a single file)
        save_qv_plots: Whether to create Q/V-trajectory plots
        output_dir: Directory to save videos and plots
        step: Current training step (for video naming)
        max_video_episodes: Only record video for first N episodes (0 = all). Reduces memory.

    Returns:
        tuple: (metrics dict, video_path, plot_path)
            - metrics: Dict with success_rate, avg_reward, avg_length, avg_success_length
            - video_path: Path to saved video (None if save_video=False or no frames collected)
            - plot_path: Path to Q/V-trajectory plots (None if save_qv_plots=False)
    """
    policy.eval()

    num_envs = len(vec_env)

    # Tracking for completed episodes
    successes = []
    total_rewards = []
    episode_lengths = []
    success_lengths = []

    # Q/V/A-trajectory tracking for plots
    all_q_trajectories = []  # List of Q-value trajectories (one per episode)
    all_v_trajectories = []  # List of V-value trajectories (one per episode)
    all_advantage_trajectories = []  # List of Advantage (Q-V) trajectories (one per episode)

    # Episode logs for value function diagnostics
    episode_logs: list[EpisodeLog] = []

    # Streaming video writer - writes frames to disk as episodes complete
    video_writer: Optional[StreamingVideoWriter] = None
    if save_video:
        # Create output directory and video path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        video_name = f"eval_step_{step if step is not None else 'final'}.mp4"
        video_path_str = str(output_path / video_name)

        # Determine effective max episodes for video
        effective_max = max_video_episodes if max_video_episodes > 0 else num_episodes

        video_writer = StreamingVideoWriter(
            output_path=video_path_str,
            fps=20,
            max_episodes=effective_max,
            num_total_episodes=num_episodes,
        )
        print(f"  Video will record first {effective_max} of {num_episodes} episodes")

    # State extraction keys - will be detected from first observation
    state_keys: list[str] = []
    env_state_keys = ["object-state"]

    # Calculate number of rounds needed (each round runs num_envs episodes in parallel)
    num_rounds = (num_episodes + num_envs - 1) // num_envs  # Ceiling division
    total_episodes_to_run = num_rounds * num_envs

    print(f"  Evaluating with {num_envs} parallel environments")
    print(
        f"  Running {num_rounds} rounds ({total_episodes_to_run} total episodes, will use first {num_episodes})"
    )
    if fixed_initial_states is not None:
        print(f"  Using {num_episodes} fixed Sobol initial states")

    completed_episodes = 0

    # Run rounds of parallel episodes
    robot_state_dim = None  # Will be computed from first observation
    for round_idx in range(num_rounds):
        print(f"\n  Round {round_idx + 1}/{num_rounds}")

        # Batch reset all environments
        obs_list = _reset_eval_batch(
            vec_env,
            fixed_initial_states,
            completed_episodes,
            num_episodes,
        )

        # Reset policy state (clears action queue for policies like IDQL/Diffusion)
        # This MUST be called after env reset to prevent using stale actions
        policy.reset()

        # Detect bimanual from first observation and set state_keys
        if not state_keys and obs_list:
            state_keys = get_robot_state_keys(obs_list[0])

        # Extract initial states and compute robot_state_dim from first observation
        states = []
        for obs in obs_list:
            robot_state = np.concatenate([obs[k].flatten() for k in state_keys if k in obs])
            if robot_state_dim is None:
                robot_state_dim = len(robot_state)
            env_state_parts = [obs[k].flatten() for k in env_state_keys if k in obs]
            if env_state_parts:
                env_state = np.concatenate(env_state_parts)
                state = np.concatenate([robot_state, env_state])
            else:
                state = robot_state
            states.append(state)

        # Per-environment tracking for this round
        env_episode_rewards = [0.0] * num_envs
        env_episode_lengths = [0] * num_envs
        env_done = [False] * num_envs
        env_success = [False] * num_envs
        env_q_trajectories = [[] for _ in range(num_envs)]
        env_v_trajectories = [[] for _ in range(num_envs)]
        env_advantage_trajectories = [[] for _ in range(num_envs)]

        # Per-environment tracking for diagnostics
        env_rewards_traj = [[] for _ in range(num_envs)]
        env_state_traj = [[] for _ in range(num_envs)]
        env_done_reason = [None] * num_envs

        # Append initial states to state trajectories
        for env_idx, state in enumerate(states):
            env_state_traj[env_idx].append(state.copy())

        # Per-environment frame buffers for this round only
        # Only collect frames if video writer needs more episodes
        should_collect_frames = video_writer is not None and video_writer.should_record()
        round_frame_buffers: Optional[List[List[tuple]]] = (
            [[] for _ in range(num_envs)] if should_collect_frames else None
        )

        # Run until all environments complete their episodes
        step_count = 0
        while not all(env_done):
            # Prepare batch states
            batch_states = []
            for env_idx, (state, done) in enumerate(zip(states, env_done)):
                if not done:
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    batch_states.append(state_tensor)
                else:
                    # For completed envs, add dummy state (won't be used)
                    batch_states.append(torch.zeros(1, len(state), device=device))

            # Stack into batch
            batch_state_tensor = torch.cat(batch_states, dim=0)

            # Normalize the full concatenated state
            state_normalized = normalizer.normalize_state(batch_state_tensor)

            # Split normalized state into robot state and environment state for unified interface
            robot_state_normalized = state_normalized[:, :robot_state_dim]
            env_state_normalized = state_normalized[:, robot_state_dim:]

            # Create batch dict with separate state keys for unified select_action interface
            obs_batch = {
                "observation.state": robot_state_normalized,
                "observation.environment_state": env_state_normalized,
            }

            # Get actions, Q-values, and V-values from policy
            # Use unified select_action interface for all IQL-style policies
            with torch.no_grad():
                # Use unified select_action interface (policy concatenates internally)
                actions_normalized, q_values, v_values = policy.select_action(
                    obs_batch, return_values=True
                )

            # Denormalize actions
            actions = normalizer.denormalize_action(actions_normalized).cpu().numpy()

            # Step all environments
            obs_list, rewards, dones, infos = vec_env.step(list(actions))

            # Capture frames from all environments (only if we're still recording video)
            if round_frame_buffers is not None:
                frames = vec_env.render()  # Returns (num_envs, H, W, 3) stacked array
                for env_idx in range(num_envs):
                    if not env_done[env_idx]:  # Only collect frames while episode is running
                        advantage = float(q_values[env_idx]) - float(v_values[env_idx])
                        round_frame_buffers[env_idx].append(
                            (
                                frames[env_idx],
                                env_episode_lengths[env_idx] + 1,
                                float(q_values[env_idx]),
                                float(v_values[env_idx]),
                                advantage,
                            )
                        )

            # Update states and tracking
            for env_idx in range(num_envs):
                if env_done[env_idx]:
                    continue  # Skip already completed environments

                # Extract next state
                obs = obs_list[env_idx]
                robot_state = np.concatenate([obs[k].flatten() for k in state_keys if k in obs])
                env_state_parts = [obs[k].flatten() for k in env_state_keys if k in obs]
                if env_state_parts:
                    env_state = np.concatenate(env_state_parts)
                    state = np.concatenate([robot_state, env_state])
                else:
                    state = robot_state
                states[env_idx] = state

                # Update tracking
                env_episode_rewards[env_idx] += rewards[env_idx]
                env_episode_lengths[env_idx] += 1
                env_q_trajectories[env_idx].append(float(q_values[env_idx]))
                env_v_trajectories[env_idx].append(float(v_values[env_idx]))
                advantage = float(q_values[env_idx]) - float(v_values[env_idx])
                env_advantage_trajectories[env_idx].append(advantage)

                # Update diagnostic tracking
                env_rewards_traj[env_idx].append(float(rewards[env_idx]))
                env_state_traj[env_idx].append(state.copy())

                # Check for success - fail loudly if "success" not in info dict
                # (the environment wrapper should guarantee this key exists)
                success_this_step = bool(infos[env_idx]["success"])

                if success_this_step:
                    env_success[env_idx] = True
                    env_done[env_idx] = True

                # Check for done or max steps
                if dones[env_idx] or env_episode_lengths[env_idx] >= max_steps:
                    env_done[env_idx] = True

            step_count += 1

        # Store results for this round
        for env_idx in range(num_envs):
            if completed_episodes < num_episodes:
                # Determine done reason
                if env_success[env_idx]:
                    env_done_reason[env_idx] = "success"
                elif env_episode_lengths[env_idx] >= max_steps:
                    env_done_reason[env_idx] = "timeout"
                else:
                    env_done_reason[env_idx] = "failure"

                successes.append(float(env_success[env_idx]))
                total_rewards.append(env_episode_rewards[env_idx])
                episode_lengths.append(env_episode_lengths[env_idx])
                if env_success[env_idx]:
                    success_lengths.append(env_episode_lengths[env_idx])

                # Store Q/V/A-trajectories
                all_q_trajectories.append(env_q_trajectories[env_idx])
                all_v_trajectories.append(env_v_trajectories[env_idx])
                all_advantage_trajectories.append(env_advantage_trajectories[env_idx])

                # Store episode log for diagnostics
                ep_log = EpisodeLog(
                    rewards=env_rewards_traj[env_idx],
                    q_values=env_q_trajectories[env_idx],
                    v_values=env_v_trajectories[env_idx],
                    states=env_state_traj[env_idx],
                    success=env_success[env_idx],
                    done_reason=env_done_reason[env_idx],
                )
                episode_logs.append(ep_log)

                # Write frames for this completed episode to video (streaming)
                if round_frame_buffers is not None and video_writer is not None:
                    video_writer.write_episode_frames(
                        frames=round_frame_buffers[env_idx],
                        episode_idx=completed_episodes,
                        is_success=env_success[env_idx],
                        has_qv_values=True,
                    )
                    # Clear frame buffer to free memory
                    round_frame_buffers[env_idx] = []

                # Print episode result
                status = "✓ SUCCESS" if env_success[env_idx] else "✗ FAILURE"
                cumulative_success_rate = (sum(successes) / len(successes)) * 100
                print(
                    f"  Episode {completed_episodes + 1}/{num_episodes} (env {env_idx}) - {status} "
                    f"(reward: {env_episode_rewards[env_idx]:.2f}, length: {env_episode_lengths[env_idx]}) "
                    f"[Cumulative: {cumulative_success_rate:.1f}%]"
                )

                completed_episodes += 1

        # Run garbage collection after each round to free memory
        gc.collect()

    policy.train()

    # Close streaming video writer and get video path
    video_path = None
    if video_writer is not None:
        video_path = video_writer.close()
        if video_path:
            print(
                f"\n  ✓ Video saved to {video_path} ({video_writer.episodes_written} episodes, {video_writer.frames_written} frames)"
            )

    # Create Q/V-trajectory plots if requested
    plot_path = None
    if save_qv_plots and all_q_trajectories and all_v_trajectories:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        plot_name = f"eval_qv_trajectories_step_{step if step is not None else 'final'}.png"
        plot_path = str(output_path / plot_name)

        print("\n  Creating Q/V/A-trajectory plots...")
        create_qv_trajectory_plots(
            q_trajectories=all_q_trajectories[:num_episodes],  # Only plot requested num_episodes
            v_trajectories=all_v_trajectories[:num_episodes],
            successes=[bool(s) for s in successes[:num_episodes]],
            output_path=Path(plot_path),
            step=step,
            a_trajectories=all_advantage_trajectories[:num_episodes],
        )

    # Compute value function diagnostics
    print("\n  Computing Q-value diagnostics...")
    q_metrics = compute_q_diagnostics(
        episode_logs[:num_episodes],
        policy,
        normalizer,
        gamma,
        device,
    )

    metrics = {
        "success_rate": np.mean(successes[:num_episodes]) * 100,
        "avg_reward": np.mean(total_rewards[:num_episodes]),
        "avg_length": np.mean(episode_lengths[:num_episodes]),
        "avg_success_length": (
            np.mean(success_lengths) if len(success_lengths) > 0 else float("nan")
        ),
        "episode_lengths": episode_lengths[:num_episodes],  # Raw list for histogram logging
        **q_metrics,  # Spread Q-value diagnostics directly into metrics
    }

    return metrics, video_path, plot_path
