"""CLI argument parsing for real-world Vision IQL training."""

import argparse
import sys
from pathlib import Path

from mulligan.real.train.hub_data import add_dataset_revision_args
from mulligan.real.train.iql_recipes import apply_recipe, recipe_names


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the real critic trainer (no side effects).

    ``--iql-recipe`` is resolved by :func:`parse_args`, not by the parser itself.
    """
    parser = argparse.ArgumentParser(
        prog="python -m mulligan.real.train.critic",
        description="Train IQL Q/V networks on real-world vision data",
        # --iql-recipe detects explicit flags by their full name in argv; an abbreviated
        # flag would be parsed but then overwritten by the recipe.
        allow_abbrev=False,
    )

    # Data
    parser.add_argument(
        "--repo-ids",
        type=str,
        required=True,
        help="Comma-separated HuggingFace dataset repo IDs",
    )
    parser.add_argument(
        "--eval-repo-ids",
        type=str,
        default=None,
        help=(
            "Optional comma-separated HuggingFace dataset repo IDs used exclusively for "
            "evaluation. When set, --repo-ids are used 100%% for training and "
            "--holdout-pct is ignored."
        ),
    )
    add_dataset_revision_args(parser)
    parser.add_argument(
        "--no-dataset-sync",
        action="store_true",
        help=(
            "Read the local dataset copies under --root as they are: no Hub sync (which prunes "
            "local parquet shards and re-downloads the pinned revision). For datasets that are "
            "not on the Hub."
        ),
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local data root path for faster I/O (datasets at root/repo_id/)",
    )
    parser.add_argument(
        "--camera-filter",
        type=str,
        default="_left",
        help="Suffix to filter camera names (default: '_left')",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=None,
        help=(
            "Optional comma-separated visual feature keys to use instead of "
            "--camera-filter. Entries may be full keys like "
            "'observation.images.side_1' or raw camera names like 'side_1'."
        ),
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=224,
        help="Resize images to this height (default: 224)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=224,
        help="Resize images to this width (default: 224)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help=(
            "Peak DataLoader worker budget, split evenly across the two overlapping "
            "persistent loader pools (default: 20)"
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor (default: 2)",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="torchcodec",
        help="Video decode backend (default: torchcodec). Options: torchcodec, pyav",
    )
    parser.add_argument(
        "--multiprocessing-context",
        type=str,
        choices=["fork", "spawn", "forkserver", "none"],
        default="spawn",
        help="DataLoader multiprocessing context (default: spawn). Use 'none' to let PyTorch choose.",
    )
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs (default: enabled when num_workers > 0).",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    parser.set_defaults(persistent_workers=True)
    parser.add_argument(
        "--buffer-capacity-gb",
        type=float,
        default=24.0,
        help="Replay-buffer memory budget in GB (default: 24).",
    )
    parser.add_argument(
        "--buffer-refresh-rate",
        type=int,
        default=16,
        help="Number of new samples decoded per training step (default: 16).",
    )

    # Encoder
    parser.add_argument(
        "--encoder-artifact",
        type=str,
        required=True,
        help=(
            "Frozen DP encoder source: hf://<repo>[@<revision>] (e.g. "
            "hf://mulligan/real-marker-d2-r05-mulligan-dp), a local DP checkpoint directory, "
            "or (with the optional wandb package) a W&B artifact [wandb://]<entity>/<project>/<name>:<v>."
        ),
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="gpu",
        choices=["none", "gpu"],
        help="Image augmentation mode. 'gpu' (default) uses torch.compile'd batched GPU ops "
        "(per-channel brightness, contrast, saturation, sharpness, affine; ~10ms for 512 imgs).",
    )
    parser.add_argument(
        "--aug-shift-frac",
        type=float,
        default=0.05,
        help="Max random translation in the gpu-augmentation affine, as a fraction of "
        "image size (default: 0.05; DrQ-strong ~0.10-0.12).",
    )

    # IQL hyperparams
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=12,
        help="DP actor prediction horizon, recorded as protocol metadata (default: 12).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=6,
        help=(
            "Executed action horizon for IQL Q/V targets and action chunks; at most "
            "--chunk-size (default: 6)."
        ),
    )
    parser.add_argument(
        "--td-horizon-steps",
        type=int,
        default=None,
        help=(
            "Reward/bootstrap horizon in robot steps while keeping the Q action input at "
            "--n-action-steps. Defaults to --n-action-steps. Non-default values must be "
            "integer multiples of --n-action-steps (for example 12 or 18 with a 6-step "
            "ranked action chunk)."
        ),
    )
    parser.add_argument(
        "--td-lambda-curriculum",
        action="store_true",
        help=(
            "Train Q against a deterministic truncated TD(lambda) mixture of every "
            "chunk-aligned horizon from --n-action-steps through --td-horizon-steps. "
            "Requires a schema-v3 flat embedding cache."
        ),
    )
    parser.add_argument(
        "--td-lambda-initial",
        type=float,
        default=0.95,
        help="Initial TD(lambda) lambda value before cosine decay (default: 0.95).",
    )
    parser.add_argument(
        "--td-lambda-hold-steps",
        type=int,
        default=25_000,
        help="Training steps to hold the initial TD(lambda) value (default: 25000).",
    )
    parser.add_argument(
        "--td-lambda-cosine-end-step",
        type=int,
        default=112_500,
        help="Step where cosine decay reaches lambda=0 (default: 112500).",
    )
    parser.add_argument(
        "--num-action-samples",
        type=int,
        default=16,
        help="Default number of diffusion candidates scored by the exported policy (default: 16).",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument(
        "--tau", type=float, default=0.005, help="Target network Polyak rate (default: 0.005)"
    )
    parser.add_argument(
        "--intervention-negative-reward",
        type=float,
        default=0.0,
        help=(
            "Reward penalty added at intervention frames before TD target construction "
            "(default: 0.0). Use -1.0 for DAgger-style intervention penalties."
        ),
    )
    parser.add_argument(
        "--reward-shift",
        type=float,
        default=0.0,
        help="Constant added to rewards after intervention penalties (default: 0.0).",
    )
    # Network architecture
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="1024,1024,1024",
        help="Hidden layer dims for Q/V networks (default: '1024,1024,1024')",
    )

    # DIVL (distributional value learning): categorical V(s) over --num-atoms atoms.
    parser.add_argument(
        "--iql-recipe",
        type=str,
        default=None,
        help="Apply a named value-head recipe from mulligan.real.train.iql_recipes as the "
        "effective config; explicit CLI flags still win. Default None = argparse defaults. "
        f"Known: {recipe_names()}.",
    )
    parser.add_argument(
        "--num-atoms",
        type=int,
        default=101,
        help="Number of categorical support atoms for the distributional V (default: 101).",
    )
    parser.add_argument(
        "--v-min",
        type=float,
        default=None,
        help="Lower edge of the distributional value support. Default None = "
        "derive from the empirical return range (lo - 0.05*span).",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=None,
        help="Upper edge of the distributional value support. Default None = "
        "derive from the empirical return range (hi + 0.05*span).",
    )
    parser.add_argument(
        "--hl-gauss-sigma-ratio",
        type=float,
        default=0.75,
        help="HL-Gauss smoothing sigma as a fraction of the atom spacing (default: 0.75).",
    )
    parser.add_argument(
        "--tau-base",
        type=float,
        default=0.7,
        help="Base DIVL quantile fraction for the Q TD bootstrap (the "
        "distributional analogue of the IQL expectile; default: 0.7). NOT the "
        "Polyak rate --tau.",
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=0.5,
        help="Lower clip bound for the (adaptive) DIVL quantile fraction (default: 0.5).",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=0.95,
        help="Upper clip bound for the (adaptive) DIVL quantile fraction (default: 0.95).",
    )
    parser.add_argument(
        "--tau-entropy-alpha",
        type=float,
        default=0.4,
        help="Entropy sensitivity for the adaptive DIVL quantile: "
        "tau = clip(tau_base - alpha*norm_entropy, tau_min, tau_max) (default: 0.4). "
        "0.0 = fixed tau_base.",
    )

    # Observation-conditioned learned action scaling ("action FiLM"): a small MLP on the
    # fused critic state emits per-arm-dim (mean, log_std), and Q scores the transformed
    # z-scored action. The final layer is zero-initialized (identity at init). The executed
    # action is unchanged; the deploy path applies the same head to a critic trained with it.
    parser.add_argument(
        "--action-film-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the observation-conditioned learned action-scaling (FiLM) head on "
        "the critic action input (default: off).",
    )
    parser.add_argument(
        "--action-film-hidden",
        type=int,
        default=256,
        help="Hidden width of the FiLM head MLP (default: 256). Only used when --action-film-head.",
    )
    parser.add_argument(
        "--action-film-arm-dims",
        type=int,
        default=6,
        help="Number of leading per-step action dims the FiLM head scales "
        "(default: 6 = the arm dims; the trailing gripper dim is left untouched). "
        "Only used when --action-film-head.",
    )

    # Training
    parser.add_argument(
        "--training-steps", type=int, default=50000, help="Total training steps (default: 50000)"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument(
        "--critic-lr-schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
        help=(
            "LR schedule for the Q/V parameter groups. 'constant' keeps the base "
            "learning rate; "
            "'warmup_cosine' ramps linearly from 0 over --critic-lr-warmup-steps "
            "then cosine-decays to --critic-lr-min-frac * --lr at "
            "--training-steps (default: constant)."
        ),
    )
    parser.add_argument(
        "--critic-lr-warmup-steps",
        type=int,
        default=2000,
        help="Linear warmup steps for --critic-lr-schedule warmup_cosine (default: 2000).",
    )
    parser.add_argument(
        "--critic-lr-min-frac",
        type=float,
        default=0.1,
        help=(
            "Cosine floor as a fraction of --lr for --critic-lr-schedule "
            "warmup_cosine (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW (default: 1e-4)"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for Q/V (default: 1.0, 0 = no clipping)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    # Target clipping
    parser.add_argument(
        "--clip-targets-min",
        type=float,
        default=None,
        help="Min value for clipping V-target (default: None = disabled)",
    )
    parser.add_argument(
        "--clip-targets-max",
        type=float,
        default=None,
        help="Max value for clipping V-target (default: None = disabled)",
    )

    # Evaluation
    parser.add_argument(
        "--holdout-pct",
        type=float,
        default=0.1,
        help="Fraction of episodes to hold out (default: 0.1)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluate every N steps (default: 10000)",
    )
    parser.add_argument(
        "--max-eval-videos",
        type=int,
        default=50,
        help="Max episodes to annotate with videos (default: 50)",
    )
    parser.add_argument(
        "--eval-video-freq",
        type=int,
        default=10000,
        help="Create annotated videos every N steps (default: 10000). 0 = disable.",
    )
    parser.add_argument(
        "--encoding-batch-size",
        type=int,
        default=512,
        help="Batch size for holdout clean-encoding pass (default: 512).",
    )
    parser.add_argument(
        "--precompute-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Precompute all train-set frozen-encoder visual embeddings once, keep the "
            "flat per-frame trajectory cache in memory, and train only Q/V MLPs. "
            "With --augmentation none, caches one clean "
            "view. With --augmentation gpu, requires --embedding-cache-augmented-views "
            "> 0 and caches one clean plus that many independently augmented views."
        ),
    )
    parser.add_argument(
        "--embedding-cache-augmented-views",
        type=int,
        default=0,
        help=(
            "Number of independently GPU-augmented frozen-encoder feature views to cache "
            "per physical state in addition to one clean view (default: 0). Requires "
            "--precompute-embeddings and --augmentation gpu."
        ),
    )
    parser.add_argument(
        "--embedding-cache-input",
        type=Path,
        default=None,
        help="Load a schema-v3 flat encoded trajectory cache instead of recomputing it.",
    )
    parser.add_argument(
        "--target-view-sampling",
        choices=("matched", "independent"),
        default="matched",
        help=(
            "Frozen-cache target-view treatment. 'matched' reuses the active current "
            "view-bank index for the successor and every TD(lambda) bootstrap. "
            "'independent' samples label-producing target-Q/current and "
            "online-V successor views independently from the active current view and "
            "independently at every TD(lambda) horizon. Requires a multi-view schema-v3 "
            "cache."
        ),
    )
    parser.add_argument(
        "--target-view-samples",
        type=int,
        default=1,
        help=(
            "Number of independently drawn cached banks (with replacement) sampled and "
            "averaged on every label-producing "
            "target evaluation when --target-view-sampling independent (default: 1). "
            "Target Q/V scalar outputs are averaged, never the frozen features. Must be 1 "
            "for matched training and cannot exceed the cache view count."
        ),
    )
    parser.add_argument(
        "--embedding-cache-output",
        type=Path,
        default=None,
        help="Atomically save the encoded replay cache to this path.",
    )
    parser.add_argument(
        "--embedding-cache-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Build and save the encoded replay cache, then exit before Q/V creation. "
            "Requires --precompute-embeddings and --embedding-cache-output."
        ),
    )
    parser.add_argument(
        "--embedding-eval-cache-input",
        type=Path,
        default=None,
        help="Load clean pre-encoded holdout states and metadata for evaluation.",
    )
    parser.add_argument(
        "--embedding-eval-cache-output",
        type=Path,
        default=None,
        help="Atomically save clean pre-encoded holdout states and metadata.",
    )
    parser.add_argument(
        "--embedding-eval-cache-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Build and save the clean holdout embedding cache, then exit before Q/V "
            "training. Requires --precompute-embeddings, --embedding-cache-input, and "
            "--embedding-eval-cache-output."
        ),
    )

    # Logging
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--wandb-project", type=str, default="mulligan", help="W&B project (default: mulligan)"
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-notes", type=str, default=None, help="W&B run notes")
    parser.add_argument(
        "--log-freq", type=int, default=5000, help="Log every N steps (default: 5000)"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/real_iql", help="Output directory"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25_000,
        help=(
            "Save a checkpoint every N steps (default: 25000; 0 = final checkpoint only). "
            "The final checkpoint is always saved."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint-freq",
        type=int,
        default=5000,
        help=(
            "Save local auto-resume state every N steps unless --no-auto-resume (default: 5000)."
        ),
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help=(
            "Disable local auto-resume. By default a run keeps resume state under "
            "<output-dir>/_resume/ keyed by its run name (--wandb-run-name, else the "
            "name derived from the flags), so rerunning the same command resumes."
        ),
    )

    # Action representation
    parser.add_argument(
        "--action-mode",
        type=str,
        default="absolute",
        choices=["absolute", "relative"],
        help=(
            "Critic action representation. 'absolute' (default): score the raw dataset "
            "action chunk (e.g. 7D cartesian_velocity). "
            "'relative': score the UMI proprio-anchored physical relative-pose chunk (10D "
            "[rel_trans(3), rel_r6(6), grip(1)] per step, relativized against the "
            "anchor frame's proprio pose) — the SAME representation an "
            "--action-mode relative DP emits after its per-timestep un-normalization, "
            "so the critic can rerank a relative DP's candidates directly. Built from "
            "the action.cartesian_position and action.gripper_position columns; "
            "requires the flat-cache regime (--precompute-embeddings)."
        ),
    )

    # Proprio dropout
    parser.add_argument(
        "--proprio-dropout",
        type=float,
        default=0.5,
        help="Dropout rate for proprioception input (0.0=full proprio, 1.0=vision-only; "
        "default: 0.5). Zeros the entire proprio vector per sample with this probability "
        "during training.",
    )

    # Throughput flags (on by default; opt out with the --no-<flag> twins). tf32,
    # encoder bf16 and uint8-native images change float rounding only.
    parser.add_argument(
        "--async-buffer-refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Decode + shape refresh batches on a background thread and rotate them "
            "into the replay buffer via a non-blocking poll, instead of blocking every "
            "training step on the refresh DataLoader. Refresh-batch content and "
            "shaping are identical; only the rotation cadence decouples from the step "
            "loop (it drops to whatever decode throughput sustains)."
        ),
    )
    parser.add_argument(
        "--enable-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable TF32 matmul (set_float32_matmul_precision('high')). Note cudnn "
            "conv TF32 is already torch's default-ON everywhere, including runs "
            "without this flag; the flag adds the matmul side (numerics-affecting)."
        ),
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cudnn.benchmark autotune (fixed shapes; algorithmically neutral).",
    )
    parser.add_argument(
        "--encoder-autocast-bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the frozen encoder forward (online + target identically) under bf16 "
            "autocast, casting features back to fp32 before the Q/V heads "
            "(numerics-affecting)."
        ),
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use channels_last memory format for encoder convs (layout-only).",
    )
    parser.add_argument(
        "--uint8-native-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Return RAW uint8 native frames from DataLoader workers and run the "
            "crop+antialias-resize+float preprocessing batched on GPU in the main "
            "process (image_preprocess.GpuImagePreprocessor), instead of decoding "
            "float32 + cropping/resizing per-sample inside each worker. The buffer "
            "and end-to-end 224^2 pixel format are unchanged; requires --num-workers>0, "
            "--persistent-workers, and DP camera_crop_boxes (crop proxy installed) — "
            "when a precondition is unmet the trainer WARNs loudly and downgrades to "
            "the float path (perf knob; the experiment is identical either way). GPU "
            "vs CPU resize rounds differently (<=1 uint8 LSB); numerics-affecting."
        ),
    )

    # Debug
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run for 100 steps with 1 eval to verify everything works",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` and apply ``--iql-recipe`` (explicit flags win over the recipe)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    apply_recipe(args, argv)
    return args
