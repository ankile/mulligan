"""Blind Gemini stage labeling (the backbone run) for any registered real task.

Builds the per-episode media assets and calls the labeler engine; the output run
directory is the input of :mod:`mulligan.real.stage_labeling.apply_cascade` and of the
eval battery. The events CSV comes from :mod:`mulligan.real.stage_labeling.prepare_events`.

    python -m mulligan.real.stage_labeling.label --task marker_d2 \\
        --dataset-repo-id mulligan/real-marker-d2-r00-eval \\
        --events-csv outputs/real/stage_events/marker_d2_r0_heldout/minimal_stage_events.csv \\
        --run-name marker_d2_r0_heldout --samples 3 --input-mode dual --send-final-crops

The Gemini route is configured from the environment (``GEMINI_API_KEY`` for the
Developer API, or ``MULLIGAN_GEMINI_ROUTE=vertex`` with ``GOOGLE_CLOUD_PROJECT`` /
``GOOGLE_CLOUD_LOCATION``). Labels are not byte-reproducible across model versions; the
paper's frozen labels ship as data. ``--dry-run`` validates the wiring without any
network call.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from mulligan.real.stage_labeling.cascade_pipeline.common import (
    DEFAULT_STAGE_LABELING_BUILD_DIR,
    DEFAULT_STAGE_LABELING_RUNS_DIR,
)
from mulligan.real.stage_labeling.labeler import LabelerConfig
from mulligan.real.stage_specs import get_label_task_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="registered task, e.g. square_d2")
    parser.add_argument(
        "--dataset-repo-id", default=None, help="default: the spec's dataset_repo_id"
    )
    parser.add_argument(
        "--events-csv", type=Path, required=True, help="per-episode events CSV of that dataset"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--prompt-variant", default=None, help="default: the spec's default variant"
    )
    parser.add_argument(
        "--model", default=None, help="default: LabelerConfig default (gemini-3.5-flash)"
    )
    parser.add_argument(
        "--media-resolution",
        choices=["low", "medium", "high"],
        default=None,
        help="tokens-per-frame budget; 'high' preserves small-object detail (default: SDK default)",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--input-mode", choices=["dual", "combo"], default="dual")
    parser.add_argument("--video-transport", choices=["inline", "file"], default="inline")
    parser.add_argument("--send-final-crops", action="store_true")
    parser.add_argument(
        "--send-grasp-crops",
        action="store_true",
        help="attach the zoomed grasp-window wrist montage (decisive for the S0/S1 line)",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--episode", type=int, action="append", dest="episodes")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STAGE_LABELING_RUNS_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_STAGE_LABELING_BUILD_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate wiring, no network/API")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    spec = get_label_task_spec(args.task)
    spec = dataclasses.replace(
        spec,
        dataset_repo_id=args.dataset_repo_id or spec.dataset_repo_id,
        events_csv=args.events_csv,
    )
    variant = args.prompt_variant or spec.default_prompt_variant
    if variant not in spec.variants:
        raise SystemExit(
            f"unknown prompt variant {variant!r} for {spec.name}; have {spec.variants}"
        )

    config_kwargs = dict(
        run_name=args.run_name,
        output_dir=args.output_dir,
        build_dir=args.build_dir,
        prompt_variant=variant,
        samples=args.samples,
        temperature=args.temperature,
        input_mode=args.input_mode,
        video_transport=args.video_transport,
        send_final_crops=args.send_final_crops,
        send_grasp_crops=args.send_grasp_crops,
        media_resolution=args.media_resolution,
        workers=args.workers,
        resume=args.resume,
        episodes=tuple(args.episodes) if args.episodes else (),
    )
    if args.model:
        config_kwargs["model"] = args.model
    config = LabelerConfig(**config_kwargs)

    if args.dry_run:
        print(
            f"[dry-run] task={spec.name} variant={variant} samples={args.samples} "
            f"model={config.model} input_mode={args.input_mode}\n"
            f"  dataset={spec.dataset_repo_id}\n"
            f"  events_csv={spec.events_csv} (exists={spec.events_csv.exists()})\n"
            f"  cameras=side:{spec.side_camera_key} wrist:{spec.wrist_camera_key} fps={spec.fps}\n"
            f"  output={args.output_dir / args.run_name}"
        )
        return

    # Imported here so --dry-run needs neither HF nor google.genai.
    from mulligan.real.stage_labeling.assets import build_items
    from mulligan.real.stage_labeling.cascade_pipeline.common import configure_gemini
    from mulligan.real.stage_labeling.labeler import run_labeler

    configure_gemini()
    episodes = set(args.episodes) if args.episodes else None
    items = build_items(spec, args.build_dir, episodes=episodes)
    print(f"built {len(items)} episode(s) x {args.samples} sample(s) for {spec.name}")
    run_labeler(spec, config, items)


if __name__ == "__main__":
    main()
