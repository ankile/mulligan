"""Apply task-specific cascade refinements to a stage-labeling run.

Example:
  python -m mulligan.real.stage_labeling.apply_cascade \
      --task marker_d2 \
      --input-run-name marker_d2_r0_heldout_v4p16d2_g35 \
      --output-run-name marker_d2_r0_heldout_cascade \
      --dataset-repo-id mulligan/real-marker-d2-r00-eval \
      --events-csv outputs/real/stage_events/marker_d2_r0_heldout/minimal_stage_events.csv

  python -m mulligan.real.stage_labeling.apply_cascade \
      --task square_d2 \
      --input-run-name square_d2_r0_v0p5_g35 \
      --output-run-name square_d2_r0_v0p5_s1subtype_g35 \
      --dataset-repo-id mulligan/real-square-d2-c00-teleop-mixed \
      --events-csv outputs/real/stage_events/square_d2_r0/minimal_stage_events.csv

The events CSV (per-episode gripper close/reopen times) comes from
:mod:`mulligan.real.stage_labeling.events`; the input run is a backbone labeler run
under ``--runs-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mulligan.real.stage_labeling.cascade_pipeline.common import (
    DEFAULT_STAGE_LABELING_BUILD_DIR,
    DEFAULT_STAGE_LABELING_RUNS_DIR,
    CascadeRunOutputs,
    MarkerD2CascadeConfig,
    SquareD2CascadeConfig,
)
from mulligan.real.stage_labeling.cascade_pipeline.marker_d2 import (
    apply_marker_d2_cascade,
)
from mulligan.real.stage_labeling.cascade_pipeline.square_d2 import (
    apply_square_d2_cascade,
)


def _print_outputs(outputs: CascadeRunOutputs) -> None:
    print(f"run_dir: {outputs.run_dir}")
    print(f"sample_labels_csv: {outputs.sample_labels_csv}")
    print(f"labels_joined_csv: {outputs.labels_joined_csv}")
    print(f"provenance_json: {outputs.provenance_json}")
    print(f"summary_json: {outputs.summary_json}")


def _parse_episode_set(text: str | None) -> frozenset[int]:
    if text is None or text.strip() == "":
        return frozenset()
    episodes: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"invalid descending episode range {part!r}")
            episodes.update(range(lo, hi + 1))
        else:
            episodes.add(int(part))
    return frozenset(episodes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="marker_d2",
        choices=["marker_d2", "square_d2"],
        help="registered task cascade to apply",
    )
    parser.add_argument("--input-run-name", required=True)
    parser.add_argument("--output-run-name", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_STAGE_LABELING_RUNS_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_STAGE_LABELING_BUILD_DIR)
    parser.add_argument(
        "--exemplar-dataset-repo-id",
        default=None,
        help="marker_d2 only: repository that owns the fixed reviewed few-shot episode ids",
    )
    parser.add_argument(
        "--exemplar-build-dir",
        type=Path,
        default=None,
        help="marker_d2 only: asset cache built from --exemplar-dataset-repo-id",
    )
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--media-resolution", choices=["low", "medium", "high"], default="high")
    parser.add_argument(
        "--episode-indices",
        default="",
        help=(
            "square_d2 only: comma/range list of source episode indices to process, "
            "e.g. 0-30 or 10,18,22"
        ),
    )
    parser.add_argument("--override-min-frac", type=float, default=0.8)
    parser.add_argument(
        "--table-override-min-frac",
        type=float,
        default=0.6,
        help="override gate for marker_d2 final-on-table node T",
    )
    parser.add_argument(
        "--square-d2-transport-override-min-frac",
        type=float,
        default=0.6,
        help="square_d2 only: override gate for S2/S3 transport-boundary verdicts",
    )
    parser.add_argument(
        "--square-d2-transport-s4-override-min-frac",
        type=float,
        default=1.0,
        help="square_d2 only: override gate for definite S4 peg-hole engagement verdicts",
    )
    parser.add_argument(
        "--square-d2-carry-endpoint-override-min-frac",
        type=float,
        default=0.8,
        help="square_d2 only: override gate for endpoint-action verdicts",
    )
    parser.add_argument(
        "--square-d2-peg-arrival-override-min-frac",
        type=float,
        default=0.8,
        help="square_d2 only: override gate for strict peg-arrival verdicts",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.task != "marker_d2" and (
        args.exemplar_dataset_repo_id is not None or args.exemplar_build_dir is not None
    ):
        raise SystemExit(
            "--exemplar-dataset-repo-id/--exemplar-build-dir are only supported for marker_d2"
        )
    if args.task == "marker_d2":
        if args.square_d2_transport_override_min_frac != 0.6:
            raise SystemExit(
                "--square-d2-transport-override-min-frac is only supported for square_d2"
            )
        if args.square_d2_transport_s4_override_min_frac != 1.0:
            raise SystemExit(
                "--square-d2-transport-s4-override-min-frac is only supported for square_d2"
            )
        if args.square_d2_carry_endpoint_override_min_frac != 0.8:
            raise SystemExit(
                "--square-d2-carry-endpoint-override-min-frac is only supported for square_d2"
            )
        if args.square_d2_peg_arrival_override_min_frac != 0.8:
            raise SystemExit(
                "--square-d2-peg-arrival-override-min-frac is only supported for square_d2"
            )
        if args.episode_indices:
            raise SystemExit("--episode-indices is only supported for square_d2")
        outputs = apply_marker_d2_cascade(
            MarkerD2CascadeConfig(
                input_run_name=args.input_run_name,
                output_run_name=args.output_run_name,
                dataset_repo_id=args.dataset_repo_id,
                events_csv=args.events_csv,
                runs_dir=args.runs_dir,
                build_dir=args.build_dir,
                exemplar_dataset_repo_id=args.exemplar_dataset_repo_id,
                exemplar_build_dir=args.exemplar_build_dir,
                model=args.model,
                workers=args.workers,
                media_resolution=args.media_resolution,
                override_min_frac=args.override_min_frac,
                table_override_min_frac=args.table_override_min_frac,
            )
        )
    elif args.task == "square_d2":
        if args.override_min_frac != 0.8:
            raise SystemExit("--override-min-frac is only supported for node cascades")
        if args.table_override_min_frac != 0.6:
            raise SystemExit("--table-override-min-frac is only supported for marker_d2")
        outputs = apply_square_d2_cascade(
            SquareD2CascadeConfig(
                input_run_name=args.input_run_name,
                output_run_name=args.output_run_name,
                dataset_repo_id=args.dataset_repo_id,
                events_csv=args.events_csv,
                runs_dir=args.runs_dir,
                build_dir=args.build_dir,
                model=args.model,
                workers=args.workers,
                media_resolution=args.media_resolution,
                included_episodes=_parse_episode_set(args.episode_indices),
                transport_override_min_frac=args.square_d2_transport_override_min_frac,
                transport_s4_override_min_frac=args.square_d2_transport_s4_override_min_frac,
                carry_endpoint_override_min_frac=args.square_d2_carry_endpoint_override_min_frac,
                peg_arrival_override_min_frac=args.square_d2_peg_arrival_override_min_frac,
            )
        )
    else:  # argparse choices keep this unreachable unless a new task is added incorrectly.
        raise ValueError(f"unsupported cascade task {args.task!r}")
    _print_outputs(outputs)


if __name__ == "__main__":
    main()
