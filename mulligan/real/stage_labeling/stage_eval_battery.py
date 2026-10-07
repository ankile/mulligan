"""Generate the standard stage-label eval plot battery.

Example:
  python -m mulligan.real.stage_labeling.stage_eval_battery \
      --task marker_d2 \
      --labels-csv outputs/real/stage_labeling_runs/<run>/labels_joined.csv \
      --plot-dir outputs/real/stage_labels/marker_d2_r0_heldout/plots \
      --csv-dir outputs/real/stage_labels/marker_d2_r0_heldout/csv \
      --paired-rounds-csv <heldout ingest data dir>/paired_round_outcomes.csv \
      --prefix marker_d2_r0_heldout \
      --title "marker_d2 R0 held-out eval"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mulligan.real.stage_labeling.eval_battery import run_stage_eval_battery
from mulligan.real.stage_specs import get_label_task_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="registered stage-label task")
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--labels-csv", type=Path, help="joined labels CSV of a stage-label run")
    labels.add_argument(
        "--run-dir", type=Path, help="stage-label run dir; reads RUN_DIR/labels_joined.csv"
    )
    parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument(
        "--paired-rounds-csv",
        type=Path,
        default=None,
        help="optional paired eval map with *_episode_index columns for exact McNemar tests",
    )
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--title", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    labels_csv = (
        args.labels_csv if args.labels_csv is not None else args.run_dir / "labels_joined.csv"
    )
    outputs = run_stage_eval_battery(
        spec=get_label_task_spec(args.task),
        labels_csv=labels_csv,
        plot_dir=args.plot_dir,
        csv_dir=args.csv_dir,
        prefix=args.prefix,
        title=args.title,
        paired_rounds_csv=args.paired_rounds_csv,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
