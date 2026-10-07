"""Render (and optionally show) the operator target cards of an initial-state manifest.

Run this before a collection or eval day to check the cards a manifest will produce, on
any machine, without the robot stack:

    python -m mulligan.real.operator_ui.preview --manifest data/real/manifests/<task>/rNN/<manifest>.json
    python -m mulligan.real.operator_ui.preview --manifest M --idx 0 --idx 7 --show
    python -m mulligan.real.operator_ui.preview --manifest M --collection-style --out-dir /tmp/cards

``--dashboard`` also renders the eval session panels (setup / running / reset) and
``--collection-style`` the collector's panels (setup / policy / human / saving / choose)
around the card, with explicitly simulated timing and quota numbers. ``--show`` opens each
card in the same OpenCV window the entrypoints use (needs a display) and waits for a key
between cards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from mulligan.real.collect.initial_states import (
    _load_manifest_payload,
    load_manifest_targets,
)
from mulligan.real.lifecycle.tasks import find_task_spec_by_task_name
from mulligan.real.operator_ui.cards import CardStyle
from mulligan.real.operator_ui.session import OperatorUI
from mulligan.real.operator_ui.progress import (
    CollectionScene,
    CollectionStatus,
    EvalScene,
    SessionProgress,
)
from mulligan.real.operator_ui.panel import compose_panel, fit_diagram


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--task",
        default=None,
        help="Expected task name (default: the manifest's own `task` field).",
    )
    parser.add_argument(
        "--idx",
        type=int,
        action="append",
        default=None,
        help="Manifest index to render (repeatable). Default: the first row.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Also render setup/running/reset session panels with explicitly simulated timing.",
    )
    parser.add_argument(
        "--collection-style",
        action="store_true",
        help=(
            "Collection-style card (start label, FIRST ROLLOUT) plus the collector's "
            "session panels with simulated quota / pace numbers."
        ),
    )
    parser.add_argument("--show", action="store_true", help="Open each card in an OpenCV window.")
    return parser.parse_args(argv)


def _write_collection_panels(ui, card_path, out_dir, idx, target, subtask_marks) -> None:
    """The collector's panel in each phase, with sample numbers marked as such."""
    elapsed = 37 * 60.0
    ui.progress = SessionProgress(
        clock=lambda: elapsed,
        started=0.0,
        scene=CollectionScene(f"start {idx + 1:03d}", "First rollout", 13, target=target),
        status=CollectionStatus(
            metrics=(
                ("WITH-CF QUOTA", "12 / 50"),
                ("NO-CF QUOTA", "20 / 50"),
                ("EST. TIME LEFT", "1h 42m"),
            ),
            fraction=0.32,
            summary="Credited 7   |   4m 12s / ep   |   ~27 left",
        ),
    )
    diagram = fit_diagram(cv2.imread(str(card_path)))
    phases = {
        "setup": ("Set up the target", "Place the clips and rope to match the diagram."),
        "policy": ("Policy running", ""),
        "human": ("Human correction", ""),
        "saving": (
            "Saving episode",
            "Episode 13: SUCCESS (412 steps, 3 segments). Encoding video...",
        ),
        "choose": ("Choose what happens next", "Episode saved"),
    }
    for phase, (title, detail) in phases.items():
        if phase in {"policy", "human"}:
            ui.begin_rollout(0, subtask_marks, phase=title)
            ui.progress.step = 412 if phase == "human" else 187
            ui.progress.marks = 1 if (phase == "human" and subtask_marks) else 0
            ui.progress.rollout_started = elapsed - 61
        else:
            ui.progress.rollout_started = None
            ui.set_phase(title, detail)
        ui.progress.phase += "  [PREVIEW - sample numbers]"
        output = out_dir / f"target_{idx:04d}_collect_{phase}.png"
        if not cv2.imwrite(str(output), compose_panel(ui.progress, diagram)):
            raise RuntimeError(f"Could not write {output}")
        print(f"  collection panel preview: {output}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task_name = args.task or str(_load_manifest_payload(args.manifest)["task"])
    task_spec = find_task_spec_by_task_name(task_name)
    subtask_marks = 0 if task_spec is None else task_spec.num_subtask_marks
    targets, meta = load_manifest_targets(
        args.manifest, expected_task=task_name, model_id=lambda key: f"preview:{key}"
    )
    by_idx = {t.manifest_idx: t for t in targets}
    wanted = args.idx if args.idx else [targets[0].manifest_idx]
    missing = [i for i in wanted if i not in by_idx]
    if missing:
        raise SystemExit(f"{args.manifest}: no rows with manifest_idx {missing}")
    out_dir = args.out_dir or Path("/tmp") / f"{args.manifest.stem}_cards"
    ui = OperatorUI(show_card_window=args.show, card_dir=out_dir, monitor_camera_keys=None)
    print(f"{args.manifest}: task={task_name} rows={len(targets)}")
    for idx in wanted:
        target = by_idx[idx]
        if args.collection_style:
            style = CardStyle(
                mode_label="FIRST ROLLOUT",
                display_label=f"start {idx + 1:03d}",
                show_coordinates=False,
            )
        else:
            style = CardStyle(mode_label="PREVIEW")
        print(f"  manifest_idx={idx}: {json.dumps(target.raw, sort_keys=True)}")
        path = ui.show_card(target, meta, task_name=task_name, style=style)
        if args.collection_style:
            _write_collection_panels(ui, path, out_dir, idx, target, subtask_marks)
        if args.dashboard:
            # Sample timing is explicitly marked in the image, never presented as
            # evidence from a real eval. The drawing code is the operational panel.
            finished = 2 * targets.index(target) + 1
            elapsed = finished * 42.0
            ui.progress = SessionProgress(
                total=2 * len(targets),
                completed=finished,
                clock=lambda: elapsed,
                started=0.0,
                session_completed=finished,
                scene=EvalScene(targets.index(target) + 1, len(targets), "B", 2, 2, target),
            )
            diagram = fit_diagram(cv2.imread(str(path)))
            for phase in ("setup", "running", "reset"):
                if phase == "running":
                    ui.begin_rollout(850, subtask_marks)
                    ui.progress.step = 324
                    ui.progress.rollout_started = elapsed - 22
                else:
                    ui.progress.rollout_started = None
                    ui.set_phase(
                        "Set up the target" if phase == "setup" else "Resetting robot",
                        "Place objects to match the diagram."
                        if phase == "setup"
                        else "Next target shown. Keep clear of the arm during reset.",
                        controls=("Any key starts", "r / numpad - re-home   |   q / numpad / quit")
                        if phase == "setup"
                        else (),
                    )
                ui.progress.phase += "  [PREVIEW - sample timing]"
                output = out_dir / f"target_{idx:04d}_{phase}.png"
                if not cv2.imwrite(str(output), compose_panel(ui.progress, diagram)):
                    raise RuntimeError(f"Could not write {output}")
                print(f"  dashboard preview: {output}")
        if args.show:
            ui.gate(prompt="Card shown", any_key_starts=True)
    ui.close()


if __name__ == "__main__":
    main()
