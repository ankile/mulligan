"""Trainer harness: deterministic tiny CPU run of the Diffusion-Policy trainer.

NOT a pytest module (no ``test_`` prefix, so pytest never collects it).

Runs a deterministic, tiny, CPU-only, few-step end-to-end training of the
Diffusion-Policy monolith ``mulligan.real.train.policy.train`` over the synthetic
fixture from :mod:`tests.real.tiny_real_dataset`, and dumps a full-precision
per-step loss / grad-norm / lr trace to JSON.

Two captures of the same code can be diffed to check that a change to the trainer
leaves its training math unchanged. Every scalar is stored as ``float.hex()`` so
the comparison is bitwise.

Nothing in ``mulligan/real/`` is modified. The trace is captured purely by
monkeypatching, at the three points the training loop actually touches:

* ``DiffusionPolicy.forward``      -> per-step loss + loss_dict
* ``torch.nn.utils.clip_grad_norm_`` -> per-step grad norm (the loop calls it
  through that exact attribute path, so the module-level patch is seen)
* ``torch.optim.Optimizer.zero_grad`` -> ``param_groups[0]["lr"]`` at the moment
  of the update (``zero_grad`` is defined on the base ``Optimizer`` and is not
  overridden by Adam/AdamW, and the loop calls it immediately before
  ``backward``/``clip``/``step`` with the LR the scheduler has set for this step)

Usage
-----
    python -m tests.real.trainer_harness_dp capture --out trace.json
    python -m tests.real.trainer_harness_dp verify        # double-run bitwise check
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Force CPU before torch initializes a CUDA context. The DP trainer picks its
# device with `torch.cuda.is_available()` (there is no --device flag), so the
# harness must make CUDA invisible; `run_dp_trace` additionally monkeypatches
# `torch.cuda.is_available` in case the process already saw a GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Single-threaded BLAS: thread-count-dependent reduction order is the classic
# source of last-ulp drift between two "identical" CPU runs.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

REPO_ID = "tiny/real-tiny"
N_EPISODES = 3
EP_LEN = 16
CAMERAS = ("cam_a_left", "cam_b_left")
FIXTURE_SEED = 0

DEFAULT_TRACE_PATH = Path("dp_trace.json")


DATA_ROOT_TOKEN = "<DATA_ROOT>"
OUTPUT_DIR_TOKEN = "<OUTPUT_DIR>"


def _hex(value: float) -> str:
    return float(value).hex()


def _redact(argv: list[str], *, data_root: Path, output_dir: Path) -> list[str]:
    """Replace the two run-scratch paths with stable tokens.

    The committed baseline trace must be byte-stable across machines and across
    the two verification runs, so the recorded argv carries placeholders; the
    real paths are kept under ``config.paths`` (which parity comparison ignores).
    """
    subs = {str(data_root): DATA_ROOT_TOKEN, str(output_dir): OUTPUT_DIR_TOKEN}
    return [subs.get(tok, tok) for tok in argv]


def dp_argv(
    *,
    data_root: Path,
    output_dir: Path,
    steps: int,
    seed: int,
    extra_argv: list[str] | None = None,
) -> list[str]:
    """The exact argv (after argv[0]) handed to ``parse_args()``.

    Smallest configuration that still exercises the production DP loop: real
    multi-camera dataset, per-camera replacement crop + resize, standard image
    augmentation, separate per-camera encoders (the trainer default), MIN_MAX
    action normalization, cosine LR schedule, grad clipping, EMA off.
    """
    return [
        # --- dataset ---
        "--repo-ids",
        REPO_ID,
        "--dataset-root",
        str(data_root),
        "--camera-keys",
        ",".join(CAMERAS),
        "--video-backend",
        "torchcodec",
        # --- policy geometry (tiny) ---
        "--chunk-size",
        "8",
        "--n-action-steps",
        "6",
        "--down-dims",
        "32,64",
        "--spatial-softmax-num-keypoints",
        "8",
        # Random-init ResNet18 + GroupNorm: no ImageNet weight download, so the
        # harness runs offline; per-dataset image normalization.
        "--pretrained-backbone-weights",
        "none",
        "--visual-normalization",
        "dataset",
        # Every frame of the fixture, DAgger episodes included.
        "--no-filter-dagger",
        "--image-height",
        "64",
        "--image-width",
        "64",
        "--side-crop",
        "cam_a_left=4,2,60,62",
        "--drop-n-last-frames",
        "auto",
        # --- training ---
        "--training-steps",
        str(steps),
        "--batch-size",
        "2",
        "--eval-batch-size",
        "2",
        "--eval-freq",
        "0",
        "--save-freq",
        "1000000",
        "--log-freq",
        "1",
        "--num-workers",
        "0",
        "--no-uint8-native-images",
        "--no-decoded-frame-cache",
        "--no-resume",
        # The fixture is purely local; a sync would reach the HF Hub.
        "--no-dataset-sync",
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        *(extra_argv or []),
    ]


class _TraceRecorder:
    """Collects the per-call scalars the training loop produces."""

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.loss_dicts: list[dict] = []
        self.grad_norms: list[float] = []
        self.lrs: list[float] = []

    def steps(self, n_expected: int) -> list[dict]:
        if not (len(self.losses) == len(self.grad_norms) == len(self.lrs) == n_expected):
            raise RuntimeError(
                "trace capture is misaligned: "
                f"expected {n_expected} of each, got losses={len(self.losses)}, "
                f"grad_norms={len(self.grad_norms)}, lrs={len(self.lrs)}. "
                "A hook fired a different number of times than the training loop "
                "stepped (e.g. validation ran, or the loop changed shape)."
            )
        return [
            {
                "step": i + 1,
                "loss_hex": _hex(self.losses[i]),
                "loss": self.losses[i],
                "loss_dict": {k: _hex(v) for k, v in self.loss_dicts[i].items()},
                "grad_norm_hex": _hex(self.grad_norms[i]),
                "grad_norm": self.grad_norms[i],
                "lr_hex": _hex(self.lrs[i]),
                "lr": self.lrs[i],
            }
            for i in range(n_expected)
        ]


@contextlib.contextmanager
def _instrumented(recorder: _TraceRecorder):
    """Install the three capture hooks (and the offline/CPU guards); restore after."""
    import torch

    import mulligan.real.train.policy as trp
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    orig_forward = DiffusionPolicy.forward
    orig_clip = torch.nn.utils.clip_grad_norm_
    orig_zero_grad = torch.optim.Optimizer.zero_grad
    orig_cuda_available = torch.cuda.is_available

    def traced_forward(self, batch, *args, **kwargs):
        loss, loss_dict = orig_forward(self, batch, *args, **kwargs)
        recorder.losses.append(loss.detach().double().item())
        flat = trp.flatten_loss_dict(loss_dict) if loss_dict is not None else {}
        recorder.loss_dicts.append({k: float(v) for k, v in flat.items()})
        return loss, loss_dict

    def traced_clip(*args, **kwargs):
        out = orig_clip(*args, **kwargs)
        value = out.detach().double().item() if isinstance(out, torch.Tensor) else float(out)
        recorder.grad_norms.append(value)
        return out

    def traced_zero_grad(self, *args, **kwargs):
        recorder.lrs.append(float(self.param_groups[0]["lr"]))
        return orig_zero_grad(self, *args, **kwargs)

    DiffusionPolicy.forward = traced_forward
    torch.nn.utils.clip_grad_norm_ = traced_clip
    torch.optim.Optimizer.zero_grad = traced_zero_grad
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        DiffusionPolicy.forward = orig_forward
        torch.nn.utils.clip_grad_norm_ = orig_clip
        torch.optim.Optimizer.zero_grad = orig_zero_grad
        torch.cuda.is_available = orig_cuda_available


def run_dp_trace(
    out_json: Path,
    *,
    data_root: Path,
    steps: int = 12,
    seed: int = 0,
    output_dir: Path | None = None,
    capture_stdout: bool = True,
    log_path: Path | None = None,
    extra_argv: list[str] | None = None,
) -> dict:
    """Run the DP trainer's ``train`` for ``steps`` steps and write the trace JSON.

    ``data_root`` must be the parent directory holding ``<repo_id>/`` (i.e. what
    ``--dataset-root`` expects); build it with
    :func:`tests.real.tiny_real_dataset.build_tiny_real_dataset`.
    """
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    torch.use_deterministic_algorithms(True)
    # Pin the fp32 matmul mode to the torch default. This is a PROCESS-GLOBAL
    # knob and the critic trainer's `main()` sets it to "high" (via
    # mulligan.training.precision.configure_torch_precision) without restoring it —
    # so running the IQL trace first in the same interpreter would otherwise
    # shift this run's losses by a couple of ULP and make the parity comparison
    # test-order dependent. Pinning is a no-op in a fresh process, which is how
    # the reference traces were captured.
    saved_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(seed)
    np.random.seed(seed)

    import mulligan.real.train.policy as trp

    tmp_out_ctx: tempfile.TemporaryDirectory | None = None
    if output_dir is None:
        tmp_out_ctx = tempfile.TemporaryDirectory(prefix="trainer_dp_out_")
        output_dir = Path(tmp_out_ctx.name)

    argv = dp_argv(
        data_root=Path(data_root),
        output_dir=Path(output_dir),
        steps=steps,
        seed=seed,
        extra_argv=extra_argv,
    )
    recorder = _TraceRecorder()
    buf = io.StringIO()
    orig_argv = sys.argv
    try:
        sys.argv = ["mulligan.real.train.policy", *argv]
        with _instrumented(recorder):
            args = trp.parse_args()
            if capture_stdout:
                try:
                    with contextlib.redirect_stdout(buf):
                        trp.train(args)
                except BaseException:
                    # Never swallow: surface the monolith's own output first.
                    sys.stdout.write(buf.getvalue())
                    sys.stdout.flush()
                    raise
            else:
                trp.train(args)
    finally:
        sys.argv = orig_argv
        torch.set_float32_matmul_precision(saved_matmul_precision)
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text(buf.getvalue())
        if tmp_out_ctx is not None:
            tmp_out_ctx.cleanup()

    trace = {
        "config": {
            "argv": _redact(argv, data_root=Path(data_root), output_dir=Path(output_dir)),
            "paths": {
                "data_root": str(Path(data_root)),
                "output_dir": str(Path(output_dir)),
            },
            "steps": steps,
            "seed": seed,
            "repo_id": REPO_ID,
            "fixture": {
                "n_episodes": N_EPISODES,
                "ep_len": EP_LEN,
                "cameras": list(CAMERAS),
                "seed": FIXTURE_SEED,
            },
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        "steps": recorder.steps(steps),
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(trace, indent=2) + "\n")
    return trace


def ensure_fixture(data_root: Path, *, seed: int = FIXTURE_SEED) -> Path:
    """Build (or reuse) the tiny fixture dataset under ``data_root``."""
    from tests.real.tiny_real_dataset import build_tiny_real_dataset

    return build_tiny_real_dataset(
        Path(data_root),
        repo_id=REPO_ID,
        n_episodes=N_EPISODES,
        ep_len=EP_LEN,
        cameras=list(CAMERAS),
        seed=seed,
    )


def _comparable(trace: dict) -> str:
    """Canonical string of everything that must be bitwise stable across runs."""
    return json.dumps(
        {
            "argv": trace["config"]["argv"],
            "steps": [
                {
                    "step": s["step"],
                    "loss_hex": s["loss_hex"],
                    "loss_dict": s["loss_dict"],
                    "grad_norm_hex": s["grad_norm_hex"],
                    "lr_hex": s["lr_hex"],
                }
                for s in trace["steps"]
            ],
        },
        sort_keys=True,
    )


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="Build the fixture and dump one trace JSON.")
    cap.add_argument("--out", type=Path, default=DEFAULT_TRACE_PATH)
    cap.add_argument("--data-root", type=Path, required=True)
    cap.add_argument("--steps", type=int, default=12)
    cap.add_argument("--seed", type=int, default=0)
    cap.add_argument("--log", type=Path, default=None)
    cap.add_argument(
        "--extra",
        action="append",
        default=None,
        help="Extra argv token appended to the training command (repeatable).",
    )

    ver = sub.add_parser(
        "verify", help="Run the trace TWICE in fresh subprocesses; assert bitwise identity."
    )
    ver.add_argument("--data-root", type=Path, required=True)
    ver.add_argument("--steps", type=int, default=12)
    ver.add_argument("--seed", type=int, default=0)
    ver.add_argument("--workdir", type=Path, default=None)

    ns = parser.parse_args()

    if ns.cmd == "capture":
        ensure_fixture(ns.data_root)
        trace = run_dp_trace(
            ns.out,
            data_root=ns.data_root,
            steps=ns.steps,
            seed=ns.seed,
            log_path=ns.log,
            extra_argv=ns.extra,
        )
        print(f"wrote {ns.out} ({len(trace['steps'])} steps)")
        print(f"first loss {trace['steps'][0]['loss']!r}  last loss {trace['steps'][-1]['loss']!r}")
        return 0

    # verify: two fresh interpreters, so no in-process RNG/cache carry-over can
    # mask nondeterminism.
    ensure_fixture(ns.data_root)
    workdir = Path(ns.workdir) if ns.workdir else Path(tempfile.mkdtemp(prefix="trainer_verify_"))
    workdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for i in (1, 2):
        out = workdir / f"dp_trace_run{i}.json"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.real.trainer_harness_dp",
                "capture",
                "--out",
                str(out),
                "--data-root",
                str(ns.data_root),
                "--steps",
                str(ns.steps),
                "--seed",
                str(ns.seed),
            ],
            check=True,
            env=env,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        outs.append(out)

    a = json.loads(outs[0].read_text())
    b = json.loads(outs[1].read_text())
    if _comparable(a) != _comparable(b):
        if a["config"]["argv"] != b["config"]["argv"]:
            print(f"ARGV DIFFERS:\n  run1={a['config']['argv']}\n  run2={b['config']['argv']}")
        for sa, sb in zip(a["steps"], b["steps"]):
            keys = ("loss_hex", "loss_dict", "grad_norm_hex", "lr_hex")
            if any(sa[k] != sb[k] for k in keys):
                print(f"FIRST DIVERGENCE at step {sa['step']}:")
                for k in keys:
                    mark = "  " if sa[k] == sb[k] else "!!"
                    print(f"  {mark} {k}: run1={sa[k]!r} run2={sb[k]!r}")
                break
        raise SystemExit("NON-DETERMINISTIC: the two runs differ")
    print(f"DETERMINISTIC: {len(a['steps'])} steps bitwise identical across two fresh processes")
    print(f"  run1={outs[0]}\n  run2={outs[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
