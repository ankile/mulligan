"""Trainer harness: deterministic tiny CPU end-to-end run of the Vision-IQL trainer.

Runs ``mulligan.real.train.critic.main()`` (the RAW-IMAGE branch, i.e.
``use_embedding_cache is False``) for a handful of steps on a synthetic tiny
LeRobot v3.0 dataset and a synthetic tiny DiffusionPolicy encoder checkpoint,
and dumps a bitwise-exact per-step loss/grad-norm/LR trace to JSON.

Two captures of the same code compare equal, key for key, ``float.hex()`` for
``float.hex()``, so a trace taken before and after a change to the trainer shows
whether its training math moved.

Contract, deliberately:

* **No production source is edited.** Everything the harness needs is installed
  as a monkeypatch around ``main()`` and torn down afterwards.
* **Everything that perturbs numerics is pinned via CLI flags or monkeypatches
  that apply identically to every run** — see :func:`build_argv`,
  :data:`_RUN_ENV`, and :func:`_patched_run`.
* The tiny dataset comes from the SHARED builder
  ``tests.real.tiny_real_dataset.build_tiny_real_dataset`` (also used by the DP
  harness), so both traces are taken over the same production-schema bytes.
  IQL's extra column requirements (``source``, ``success``, ``reward``, ``done``)
  are all satisfied by that builder's production writer. The DP *encoder*
  checkpoint (``--encoder-artifact``) is IQL-specific and is built here.

Determinism levers (all of them matter; removing any one de-randomizes the run):

* ``--no-async-buffer-refresh`` — the default-ON async refresh thread rotates
  decoded batches into the replay buffer on a *background thread*, so buffer
  contents at step N depend on thread scheduling. The sync path is deterministic.
* ``--num-workers 0`` + ``--multiprocessing-context none`` — no worker RNG forks.
  (This also makes ``--uint8-native-images`` fail its precondition and downgrade
  loudly to the float image path, which is fine: the downgrade is deterministic
  and identical across runs.)
* ``torch.set_num_threads(1)`` — intra-op thread count changes float reduction
  order on CPU.
* CPU only: ``CUDA_VISIBLE_DEVICES=""`` plus a ``torch.cuda.is_available``
  monkeypatch, because ``main()`` picks its device from ``torch.cuda.is_available()``.
* ``torch._dynamo.config.disable = True`` (and ``TORCH_COMPILE_DISABLE=1``) —
  ``main()`` unconditionally ``torch.compile``s the online encoder and the GPU
  augmentation fn on the raw-image branch, with no CLI flag to turn either off. Disabling dynamo makes both a transparent eager
  passthrough. torch 2.11 reads ``TORCH_COMPILE_DISABLE`` (not the older
  ``TORCHDYNAMO_DISABLE``) at ``torch._dynamo.config`` import time, so the
  harness sets the config attribute directly as well.
* ``--no-dataset-sync`` — skips the forced HuggingFace Hub dataset-cache refresh,
  which would otherwise try to reach the network.

Usage::

    from tests.real.trainer_harness_iql import run_iql_trace
    run_iql_trace(Path("iql_trace.json"),
                  data_root=Path("/tmp/trainer_iql"))

or::

    python -m tests.real.trainer_harness_iql --out <json> --data-root <dir>

Comparing two runs: compare ``trace["steps"]`` and ``trace["config"]["argv"]`` (the
argv with the scratch data root replaced by ``<DATA_ROOT>``) for exact equality. Loss
values are serialized with ``float.hex()`` (a NaN becomes the string ``"nan"``), so
string equality is the right comparison and NaN != NaN never bites. Repeated runs,
including one that rebuilds the fixture in a fresh directory, produce bitwise
identical ``steps`` arrays.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from tests.real import tiny_real_dataset

# ---------------------------------------------------------------------------
# Tiny dataset + tiny DP-encoder fixture
# ---------------------------------------------------------------------------

REPO_ID = "tiny/iql-tiny"
CAMERAS = tiny_real_dataset.DEFAULT_CAMERAS
FRAME_H, FRAME_W = tiny_real_dataset.DEFAULT_FRAME_HW
STATE_DIM = 7  # cartesian pose (6) + gripper position (1)
ACTION_DIM = 7  # cartesian velocity (6) + gripper velocity (1)
# 4 episodes so the episode holdout split is non-empty on BOTH sides (main()
# raises otherwise); 20 frames each so the k=2 action/TD horizon always fits.
N_EPISODES = 4
EP_LEN = 20

# Sub-directory of ``data_root`` holding the synthetic DiffusionPolicy checkpoint
# consumed by ``--encoder-artifact`` (``load_frozen_encoder_from_dp`` accepts a
# local path as well as a W&B artifact id).
ENCODER_DIRNAME = "_trainer_dp_encoder"

# Bumped whenever the encoder checkpoint contents would change, so a stale cached
# checkpoint is rebuilt instead of silently reused.
ENCODER_VERSION = 1


def _build_tiny_dp_encoder(path: Path, *, seed: int = 0) -> Path:
    """Save a deterministic tiny DiffusionPolicy checkpoint usable as --encoder-artifact.

    ``main()`` requires ``--encoder-artifact``; ``load_frozen_encoder_from_dp``
    pulls ``policy.diffusion.rgb_encoder`` (plus the image-normalization stats and
    the ``action_target`` / ``cartesian_action_frame`` I/O contract) out of a saved
    DiffusionPolicy, so the fixture must be a real DP checkpoint directory — just
    a tiny, randomly-initialized one (no pretrained backbone download).
    """
    import torch

    import mulligan.real.policy.lerobot_patches  # noqa: F401  (adds action_target / crop fields)
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    cfg = DiffusionConfig(
        device="cpu",
        n_obs_steps=1,
        horizon=8,
        n_action_steps=4,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            **{
                cam_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, FRAME_H, FRAME_W))
                for cam_key in tiny_real_dataset.camera_feature_keys(CAMERAS)
            },
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        crop_shape=None,
        spatial_softmax_num_keypoints=8,
        down_dims=(64, 128),
        use_separate_rgb_encoder_per_camera=True,
        # Vision-IQL only supports this DP action contract.
        action_target="cartesian_velocity",
        cartesian_action_frame="base",
    )
    policy = DiffusionPolicy(cfg)

    def _vec_stats(dim):
        return {
            "mean": np.zeros(dim, dtype=np.float32),
            "std": np.ones(dim, dtype=np.float32),
            "min": -3.0 * np.ones(dim, dtype=np.float32),
            "max": 3.0 * np.ones(dim, dtype=np.float32),
        }

    stats = {"observation.state": _vec_stats(STATE_DIM), "action": _vec_stats(ACTION_DIM)}
    for cam_key in tiny_real_dataset.camera_feature_keys(CAMERAS):
        stats[cam_key] = {
            "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1),
            "std": np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1),
            "min": np.zeros((3, 1, 1), dtype=np.float32),
            "max": np.ones((3, 1, 1), dtype=np.float32),
        }

    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    policy.save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    return path


def ensure_fixture(data_root: Path, *, seed: int = 0) -> tuple[Path, Path]:
    """Build (or reuse) the dataset + encoder fixture under ``data_root``.

    Both halves are cached: an existing fixture whose stamp matches this harness
    revision + seed is reused verbatim. That matters for the parity protocol —
    two runs should read literally the same encoded video
    bytes, and video re-encoding is the one step whose byte output depends on the
    local SVT-AV1 build rather than on this repo.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    ds_path = tiny_real_dataset.build_tiny_real_dataset(
        data_root,
        repo_id=REPO_ID,
        n_episodes=N_EPISODES,
        ep_len=EP_LEN,
        cameras=CAMERAS,
        seed=seed,
    )

    enc_path = data_root / ENCODER_DIRNAME
    stamp_path = data_root / ".trainer_iql_encoder.json"
    stamp = {
        "version": ENCODER_VERSION,
        "seed": seed,
        "cameras": list(CAMERAS),
        "frame_hw": [FRAME_H, FRAME_W],
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
    }
    if not (
        enc_path.exists() and stamp_path.is_file() and json.loads(stamp_path.read_text()) == stamp
    ):
        enc_path = _build_tiny_dp_encoder(enc_path, seed=seed)
        stamp_path.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")
    return ds_path, enc_path


# ---------------------------------------------------------------------------
# argv
# ---------------------------------------------------------------------------


def build_argv(
    *,
    data_root: Path,
    encoder_path: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    extra_argv: list[str] | None = None,
):
    """The exact argv handed to the critic trainer's ``main()``."""
    return [
        "mulligan.real.train.critic",
        "--repo-ids",
        REPO_ID,
        "--root",
        str(data_root),
        "--camera-keys",
        ",".join(CAMERAS),
        "--image-height",
        str(FRAME_H),
        "--image-width",
        str(FRAME_W),
        "--encoder-artifact",
        str(encoder_path),
        # --- tiny model / tiny optimization ---
        "--hidden-dims",
        "32,32",
        # The fixture's rewards are all zero, so the DIVL value support cannot be derived
        # from the return range; pin it like the released configs do.
        "--v-min",
        "-0.05",
        "--v-max",
        "1.05",
        "--chunk-size",
        "2",
        "--n-action-steps",
        "2",
        "--training-steps",
        str(steps),
        "--batch-size",
        "4",
        "--buffer-capacity-gb",
        "0.05",
        "--buffer-refresh-rate",
        "4",
        "--lr",
        "3e-4",
        # Non-constant critic LR schedule so the captured per-step q/v LRs also
        # pin the "schedule applied BEFORE optimizer.step()" ordering.
        "--critic-lr-schedule",
        "warmup_cosine",
        "--critic-lr-warmup-steps",
        "3",
        "--critic-lr-min-frac",
        "0.1",
        "--seed",
        str(seed),
        "--holdout-pct",
        "0.25",
        # --- everything that would touch network / disk / nondeterminism ---
        "--num-workers",
        "0",
        "--multiprocessing-context",
        "none",
        "--no-persistent-workers",
        "--no-async-buffer-refresh",
        "--video-backend",
        "pyav",
        "--eval-freq",
        "0",
        "--eval-video-freq",
        "0",
        "--log-freq",
        "1",
        "--checkpoint-freq",
        "0",
        "--no-auto-resume",
        "--no-dataset-sync",
        "--output-dir",
        str(out_dir),
        *(extra_argv or []),
    ]


# ---------------------------------------------------------------------------
# Trace capture
# ---------------------------------------------------------------------------


def _hex(value) -> str:
    return float(value).hex()


class _Recorder:
    """Accumulates one record per training step, keyed off the VisionIQL forward."""

    def __init__(self):
        self.steps: list[dict] = []
        self._pending: dict | None = None

    def _flush(self):
        if self._pending is not None:
            self.steps.append(self._pending)
            self._pending = None

    def on_forward(self, losses: dict):
        import torch

        self._flush()
        scalars = {}
        for key, val in losses.items():
            if isinstance(val, torch.Tensor) and val.numel() == 1:
                scalars[key] = _hex(val.detach().reshape(()).item())
        self._pending = {
            "step": len(self.steps),
            "losses": scalars,
            "grad_norm_hex": None,
            "lrs": {},
            "target_updates": 0,
        }

    def on_grad_norm(self, norm):
        if self._pending is None:
            return
        import torch

        value = norm.item() if isinstance(norm, torch.Tensor) else float(norm)
        self._pending["grad_norm_hex"] = _hex(value)

    def on_optimizer_step(self, param_groups):
        if self._pending is None:
            return
        self._pending["lrs"] = {
            str(group.get("name", f"group{i}")): _hex(group["lr"])
            for i, group in enumerate(param_groups)
        }

    def on_update_targets(self):
        if self._pending is None:
            return
        self._pending["target_updates"] += 1

    def finish(self) -> list[dict]:
        self._flush()
        return self.steps


@contextlib.contextmanager
def _patched_run(recorder: _Recorder):
    """Install every monkeypatch the harness needs, and remove them afterwards."""
    import torch

    from mulligan.networks.vision_iql import VisionIQL

    saved = {}

    # -- CPU-only ----------------------------------------------------------
    saved["cuda_is_available"] = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    saved["num_threads"] = torch.get_num_threads()
    torch.set_num_threads(1)
    # The critic trainer's main() calls configure_torch_precision(), which sets the
    # PROCESS-GLOBAL fp32 matmul mode to "high" and never restores it. Left
    # leaked it shifts a later in-process DP trace by a couple of ULP, making
    # the parity comparison test-order dependent.
    saved["matmul_precision"] = torch.get_float32_matmul_precision()

    # -- torch.compile -> eager passthrough --------------------------------
    import torch._dynamo.config as dynamo_config

    saved["dynamo_disable"] = dynamo_config.disable
    dynamo_config.disable = True

    # -- VisionIQL.forward -------------------------------------------------
    saved["forward"] = VisionIQL.forward

    def traced_forward(self, *args, **kwargs):
        losses = saved["forward"](self, *args, **kwargs)
        recorder.on_forward(losses)
        return losses

    VisionIQL.forward = traced_forward

    # -- VisionIQL.update_targets -----------------------------------------
    saved["update_targets"] = VisionIQL.update_targets

    def traced_update_targets(self, *args, **kwargs):
        recorder.on_update_targets()
        return saved["update_targets"](self, *args, **kwargs)

    VisionIQL.update_targets = traced_update_targets

    # -- clip_grad_norm_ ---------------------------------------------------
    saved["clip"] = torch.nn.utils.clip_grad_norm_

    def traced_clip(*args, **kwargs):
        norm = saved["clip"](*args, **kwargs)
        recorder.on_grad_norm(norm)
        return norm

    torch.nn.utils.clip_grad_norm_ = traced_clip
    torch.nn.utils.clip_grad.clip_grad_norm_ = traced_clip

    # -- AdamW.step (param-group LRs prove the critic-schedule ordering) -----
    saved["adamw_step"] = torch.optim.AdamW.step

    def traced_adamw_step(self, *args, **kwargs):
        recorder.on_optimizer_step(self.param_groups)
        return saved["adamw_step"](self, *args, **kwargs)

    torch.optim.AdamW.step = traced_adamw_step

    try:
        yield
    finally:
        torch.cuda.is_available = saved["cuda_is_available"]
        torch.set_num_threads(saved["num_threads"])
        torch.set_float32_matmul_precision(saved["matmul_precision"])
        dynamo_config.disable = saved["dynamo_disable"]
        VisionIQL.forward = saved["forward"]
        VisionIQL.update_targets = saved["update_targets"]
        torch.nn.utils.clip_grad_norm_ = saved["clip"]
        torch.nn.utils.clip_grad.clip_grad_norm_ = saved["clip"]
        torch.optim.AdamW.step = saved["adamw_step"]


# Environment pinned for the duration of the run. Applied identically pre- and
# every run, so it cannot bias the comparison.
_RUN_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONHASHSEED": "0",
    "TORCH_COMPILE_DISABLE": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    # Fast reader / node-local staging are off so the harness reads the plain
    # on-disk dataset exactly as written.
    "MULLIGAN_REAL_STAGE_LOCAL": "0",
}


@contextlib.contextmanager
def _patched_env():
    saved = {key: os.environ.get(key) for key in _RUN_ENV}
    os.environ.update(_RUN_ENV)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_iql_trace(
    out_json: Path,
    *,
    data_root: Path,
    steps: int = 12,
    seed: int = 0,
    extra_argv: list[str] | None = None,
) -> dict:
    """Run the tiny deterministic Vision-IQL training and write a full-precision trace.

    Args:
        out_json: Destination JSON path (parent dirs created).
        data_root: Scratch dir for the synthetic dataset + DP-encoder fixture.
        steps: ``--training-steps``.
        seed: ``--seed`` (also seeds the fixture builders).

    Returns:
        The trace dict that was written.
    """
    import time

    out_json = Path(out_json)
    data_root = Path(data_root)

    with _patched_env():
        _, encoder_path = ensure_fixture(data_root, seed=seed)
        out_dir = data_root / "_trainer_outputs"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        argv = build_argv(
            data_root=data_root,
            encoder_path=encoder_path,
            out_dir=out_dir,
            steps=steps,
            seed=seed,
            extra_argv=extra_argv,
        )

        from mulligan.real.train import critic as critic_trainer

        recorder = _Recorder()
        saved_argv = sys.argv
        t0 = time.perf_counter()
        try:
            sys.argv = list(argv)
            with _patched_run(recorder):
                critic_trainer.main()
        finally:
            sys.argv = saved_argv
        wall_s = time.perf_counter() - t0

    captured = recorder.finish()
    # The raw argv embeds absolute scratch paths, which differ between
    # runs. Record the DATA_ROOT-normalized argv as the
    # comparison surface so the WHOLE json can be diffed, and keep the concrete
    # paths in a clearly non-comparable side field.
    argv_normalized = [
        arg.replace(str(data_root), "<DATA_ROOT>") if isinstance(arg, str) else arg for arg in argv
    ]
    trace = {
        "config": {
            "argv": argv_normalized,
            "steps": steps,
            "seed": seed,
            "repo_id": REPO_ID,
            "cameras": list(CAMERAS),
            "env": dict(_RUN_ENV),
            "use_embedding_cache": False,
            "dataset_builder": "tests.real.tiny_real_dataset.build_tiny_real_dataset",
            "n_episodes": N_EPISODES,
            "ep_len": EP_LEN,
        },
        "steps": captured,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    print(
        f"[trainer-iql] wrote {out_json} — {len(captured)} step record(s) in {wall_s:.1f}s",
        flush=True,
    )
    return trace


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extra",
        action="append",
        default=None,
        help="Extra argv token appended to the training command (repeatable).",
    )
    ns = parser.parse_args()
    run_iql_trace(
        ns.out,
        data_root=ns.data_root,
        steps=ns.steps,
        seed=ns.seed,
        extra_argv=ns.extra,
    )


if __name__ == "__main__":
    _cli()
