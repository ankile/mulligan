"""One-command rebuild of every figure of the paper and its appendix.

The registry below is the inventory of the manuscript's figures: every file the frozen
manuscript includes (``paper/reference/manuscript.json``) is produced by one entry.

- ``built``: matplotlib figures at true print size through :mod:`mulligan.plotting.paper`.
- ``rendered``: the HTML teaser (headless Chrome), the task-sequence strips and the
  initial-state photos (released HF videos) and the reset-range overlays (pinned photo
  composites).
- ``restored``: authored illustrations copied byte-for-byte from the paper evidence.

Everything is written to ``paper/build/figs`` (``$MULLIGAN_PAPER_BUILD/figs`` if set).
``--check`` validates a finished build against the bundled manifest
``paper/figures_manifest.json`` and the manuscript inventory.

    python -m paper.figures                     # build everything, stop at the first failure
    python -m paper.figures --keep-going        # build what can be built, then list the failures
    python -m paper.figures --list              # entries, and which read the paper evidence
    python -m paper.figures --only headline_real_sim
    python -m paper.figures --exclude teaser    # no Chrome on this machine
    python -m paper.figures --check
    python -m paper.figures --update-manifest   # rebuild, then rewrite the bundled manifest

Inputs come from ``paper/data`` (git), the pinned paper evidence (``$MULLIGAN_PAPER_EVIDENCE``
or the ``mulligan/paper-evidence`` HF dataset) and, for the task sequences, the released
``mulligan/*`` HF datasets. No AWS or W&B access is needed. Whether an entry reads the paper
evidence is observed by ``--update-manifest`` (every read goes through
:func:`paper.appendix.artifacts.evidence_file`) and recorded as the manifest's ``evidence``
flag; every later build re-checks it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

from mulligan.plotting import paper  # noqa: E402
from paper.appendix import artifacts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "paper/figures_manifest.json"
MANIFEST_SCHEMA = "mulligan.paper.figures.v1"
MANUSCRIPT_PATH = ROOT / "paper/reference/manuscript.json"
# Platform-dependent rasters (FFmpeg colour conversion, JPEG/PNG encoding, resampling) vs the
# manuscript's copy: mean absolute channel difference (0-255) and the fraction of pixels that
# move by more than compare_reference.PIXEL_TOLERANCE. Measured on the release workstation:
# task sequences <= 1.06 and 1.8e-6, reset ranges <= 1e-4 and 0.
RASTER_TOLERANCE = {"mean_abs_diff": 2.0, "differing_fraction": 1e-3}

Output = paper.FigureRecord | Path
BuildFn = Callable[[], "Output | list[Output]"]


@dataclass(frozen=True)
class PaperFigure:
    """One registry entry: a builder and the manuscript files it writes.

    ``width_frac`` is set for matplotlib figures (checked against the saved record);
    ``deterministic=False`` marks outputs whose bytes depend on the local platform, so
    ``--check`` does not compare their hash: with ``pixel_check`` it compares their pixels
    with the frozen manuscript's copy (``paper/reference/figs``) within ``RASTER_TOLERANCE``,
    otherwise (the Chrome-printed teaser) it verifies their presence only.
    """

    name: str
    part: str  # main | appendix
    kind: str  # built | rendered | restored
    outputs: tuple[str, ...]
    build: BuildFn
    width_frac: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    deterministic: bool = True
    pixel_check: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.pixel_check and self.deterministic:
            raise ValueError(f"{self.name}: pixel_check is for non-deterministic outputs")


# --- builders (lazy imports: data are read only when a figure is built) ---------


def _headline_combined():
    from paper.plotting import sim_paper_headline_combined as mod

    return mod.build_records()


def _state_rlpd_compact():
    from paper.plotting import state_rlpd_learning_curves as mod

    return mod.build_paper_records()


def _appendix(package: str, name: str) -> BuildFn:
    def build():
        from importlib import import_module

        return import_module(f"paper.appendix.{package}.plot").build_records(name=name)

    return build


def _teaser():
    from paper.teaser import build_teaser

    return build_teaser.build()


def _task_sequences():
    from paper import fig_tasks

    return fig_tasks.build_task_sequences()


def _initial_states():
    from paper import fig_initial_states

    return fig_initial_states.build_initial_states()


def _eval_protocol():
    from paper import fig_eval_protocol

    return fig_eval_protocol.build()


def _reset_card():
    from paper import fig_reset_ranges

    return fig_reset_ranges.build_card_record()


def _reset_ranges():
    from paper import fig_reset_ranges

    return fig_reset_ranges.build_range_overlays()


def _restored_reference():
    from paper.appendix.reference_data.prepare import materialize_assets

    return materialize_assets()


def _restored_state_evolution():
    from paper.appendix.simulation.prepare import materialize_assets

    return materialize_assets()


def _built(name: str, part: str, width_frac: float, build: BuildFn, note: str = "") -> PaperFigure:
    return PaperFigure(name, part, "built", (f"{name}.pdf",), build, width_frac, note=note)


def _appendix_built(package: str, name: str, width_frac: float = 1.0) -> PaperFigure:
    return _built(name, "appendix", width_frac, _appendix(package, name))


REGISTRY: tuple[PaperFigure, ...] = (
    # --- main text ---
    PaperFigure(
        "overview_teaser",
        "main",
        "rendered",
        ("overview_teaser.pdf",),
        _teaser,
        tags=("teaser",),
        deterministic=False,
        note="Fig. 1: authored HTML teaser printed by headless Chrome; bucket counts are digitized",
    ),
    _built(
        "sim_state_rlpd_vs_mulligan_compact",
        "main",
        0.36,
        _state_rlpd_compact,
        note="Fig. 2: state-based RLPD vs HiL-IDQL+Mulligan vs HiL-SERL",
    ),
    PaperFigure(
        "task_sequences",
        "main",
        "rendered",
        ("task_sequence_marker.jpg", "task_sequence_nut.jpg", "task_sequence_cable.jpg"),
        _task_sequences,
        tags=("network",),
        deterministic=False,
        pixel_check=True,
        note="Fig. 3: frames of released final-round episodes (HF videos)",
    ),
    _built(
        "real_world_square_d2_reset_card",
        "main",
        0.30,
        _reset_card,
        note="Fig. 4 left: Thread Nut reset card at print scale",
    ),
    PaperFigure(
        "reset_ranges",
        "main",
        "rendered",
        (
            "real_world_square_d2_init_ranges_side1.png",
            "real_world_marker_d2_init_ranges_side1.png",
        ),
        _reset_ranges,
        deterministic=False,
        pixel_check=True,
        note="Fig. 4 middle/right: annotated dense reset composites",
    ),
    _built(
        "headline_real_sim",
        "main",
        1.0,
        _headline_combined,
        note="Fig. 5: three real task panels over Square-Narrow/Broad SR + log failure",
    ),
    _built(
        "real_world_success_throughput_headline",
        "main",
        1.0,
        _appendix("productivity", "real_world_success_throughput_headline"),
        note="Fig. 6: throughput for Marker, Nut and Cable",
    ),
    _built(
        "ablations_sim_real",
        "main",
        1.0,
        _appendix("component_panels", "ablations_sim_real"),
        note="Fig. 7: sim sampling + actor data + Nut and Marker CF",
    ),
    _built(
        "real_world_dagger_collection_success_square",
        "main",
        0.30,
        _appendix("productivity", "real_world_dagger_collection_success_square"),
        note="Fig. 8: Thread Nut collection success",
    ),
    _built(
        "collection_burden_real_sim",
        "main",
        0.68,
        _appendix("component_panels", "collection_burden_real_sim"),
        note="Fig. 9: Nut burden over rounds + Square-Narrow per-sampler intervention",
    ),
    # --- appendix ---
    _appendix_built("real_results", "real_world_headline_mulligan_vs_baseline_detailed"),
    _appendix_built("real_results", "real_world_success_speed"),
    _appendix_built("real_results", "real_world_success_throughput"),
    _appendix_built("real_results", "real_world_cf_ablation"),
    _appendix_built("real_results", "real_world_marker_d2_substage"),
    _appendix_built("real_results", "real_world_square_d2_substage"),
    _appendix_built("real_results", "real_world_burden_progression"),
    _appendix_built("real_results", "real_world_dagger_collection_success_detailed"),
    _appendix_built("productivity", "real_world_dagger_collection_success"),
    _appendix_built("data_ledger", "real_world_data_composition"),
    _appendix_built("value_learning", "real_world_value_fqe"),
    _appendix_built("value_learning", "real_world_value_training"),
    _appendix_built("hilserl", "sim_hilserl_sessions"),
    _appendix_built("simulation", "sim_success_speed"),
    _appendix_built("simulation", "sim_success_throughput"),
    _appendix_built("simulation", "square_narrow_init_distribution_sobol_vs_uniform"),
    _appendix_built("simulation", "square_narrow_r2_bucket_success_sorted"),
    _appendix_built("simulation", "value_learning_bar_chart"),
    PaperFigure(
        "initial_states",
        "appendix",
        "rendered",
        (
            "initial_state_marker_typical.jpg",
            "initial_state_marker_hardest.jpg",
            "initial_state_nut_typical.jpg",
            "initial_state_nut_hardest.jpg",
            "initial_state_nut_reorientation.jpg",
        ),
        _initial_states,
        tags=("network",),
        deterministic=False,
        pixel_check=True,
        note="App. C: typical vs hardest round-0 starts (released teleop videos)",
    ),
    _built(
        "overview_eval_protocol",
        "appendix",
        1.0,
        _eval_protocol,
        note="App. D: blinded, paired evaluation schematic (illustrative outcomes)",
    ),
    PaperFigure(
        "restored_reference_data",
        "appendix",
        "restored",
        (
            "overview_funnel.pdf",
            "real_world_marker_d2_teleop_card.png",
            "real_world_square_d2_teleop_card.png",
            "real_world_routing_d2_teleop_card.png",
        ),
        _restored_reference,
        note="authored funnel illustration and operator reset cards",
    ),
    PaperFigure(
        "restored_state_evolution",
        "appendix",
        "restored",
        ("overview_state_evolution.pdf",),
        _restored_state_evolution,
        note="authored state-evolution illustration",
    ),
)


def _by_name() -> dict[str, PaperFigure]:
    return {entry.name: entry for entry in REGISTRY}


def _select(only: list[str] | None, exclude: list[str], parts: list[str]) -> list[PaperFigure]:
    known = set(_by_name()) | {tag for entry in REGISTRY for tag in entry.tags}
    unknown = sorted(set(only or ()) - set(_by_name())) + sorted(set(exclude) - known)
    if unknown:
        raise SystemExit(f"unknown figure names: {unknown}; try --list")
    entries = [_by_name()[name] for name in only] if only else list(REGISTRY)
    return [
        e
        for e in entries
        if e.part in parts and e.name not in exclude and not set(e.tags) & set(exclude)
    ]


def _chrome_missing() -> str | None:
    """None if Chrome is available, else why the teaser cannot be printed."""
    from paper.teaser.build_teaser import ChromeNotFound, chrome_binary

    try:
        chrome_binary()
    except ChromeNotFound as error:
        return f"{error}. The teaser (Fig. 1) needs Chrome; rerun with --exclude teaser to skip it."
    return None


def _outputs(result: Output | list[Output]) -> list[Output]:
    return result if isinstance(result, list) else [result]


def _build_entry(entry: PaperFigure) -> dict[str, Output]:
    results = _outputs(entry.build())
    by_file = {
        (r.pdf_path.name if isinstance(r, paper.FigureRecord) else r.name): r for r in results
    }
    if set(by_file) != set(entry.outputs):
        raise RuntimeError(
            f"{entry.name}: builder wrote {sorted(by_file)}, registry declares "
            f"{sorted(entry.outputs)}"
        )
    for result in by_file.values():
        if isinstance(result, paper.FigureRecord) and (
            entry.width_frac is None or abs(result.width_frac - entry.width_frac) > 1e-9
        ):
            raise RuntimeError(
                f"{entry.name}: registry declares width_frac={entry.width_frac} but the "
                f"builder saved width_frac={result.width_frac}"
            )
    return by_file


def _yes_no(flag: bool | None) -> str:
    return {True: "yes", False: "no", None: "?"}[flag]


def _one_line(error: BaseException) -> str:
    lines = str(error).strip().splitlines()
    return f"{type(error).__name__}: {lines[0]}" if lines else type(error).__name__


@dataclass
class BuildReport:
    files: dict[str, Output] = field(default_factory=dict)  # output file -> record or path
    built: list[str] = field(default_factory=list)  # entry names, in build order
    evidence: dict[str, bool] = field(default_factory=dict)  # built entry -> read the evidence
    failures: dict[str, str] = field(default_factory=dict)  # entry -> one-line reason


def build_entries(
    entries: list[PaperFigure], *, keep_going: bool = False, record_evidence: bool = False
) -> BuildReport:
    """Build ``entries``. Without ``keep_going`` the first failure raises; with it, each
    failure is recorded in the report and the build moves on to the next entry.

    An entry whose use of the paper evidence differs from the manifest's ``evidence`` flag
    fails, unless ``record_evidence`` (``--update-manifest`` rewrites the flag).
    """
    recorded = {} if record_evidence else recorded_evidence()
    no_chrome = _chrome_missing() if any("teaser" in e.tags for e in entries) else None
    if no_chrome and not keep_going:
        raise SystemExit(no_chrome)
    report = BuildReport()
    for entry in entries:
        print(f"── building {entry.name} ({entry.part}, {entry.kind})", flush=True)
        requests = len(artifacts.REQUESTED)
        try:
            if no_chrome and "teaser" in entry.tags:
                raise RuntimeError(no_chrome)
            by_file = _build_entry(entry)
            used = len(artifacts.REQUESTED) > requests
            expected = recorded.get(entry.name)
            if expected is not None and used != expected:
                raise RuntimeError(
                    f"{entry.name}: the manifest records evidence: {_yes_no(expected)} but this "
                    f"build {'read' if used else 'did not read'} the paper evidence; rerun with "
                    "--update-manifest"
                )
        except (Exception, SystemExit) as error:
            if not keep_going:
                raise
            report.failures[entry.name] = _one_line(error)
            print(f"FAILED {entry.name}: {report.failures[entry.name]}", flush=True)
            continue
        report.files.update(by_file)
        report.built.append(entry.name)
        report.evidence[entry.name] = used
    return report


def run_builds(entries: list[PaperFigure]) -> dict[str, Output]:
    """Build ``entries``, stopping at the first failure; return {output file name:
    FigureRecord or Path}."""
    return build_entries(entries).files


def print_summary(report: BuildReport) -> None:
    print(f"── summary: {len(report.built)} built, {len(report.failures)} failed")
    for name in report.built:
        print(f"built   {name}")
    for name, reason in report.failures.items():
        print(f"FAILED  {name}: {reason}")
    if report.failures:
        print("Rerun a failed entry with --only NAME (without --keep-going) for its traceback.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"schema": MANIFEST_SCHEMA, "figures": {}}
    payload = json.loads(MANIFEST_PATH.read_text())
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(f"{MANIFEST_PATH}: unexpected schema {payload.get('schema')!r}")
    return payload


def recorded_evidence(manifest: dict | None = None) -> dict[str, bool]:
    """{entry name: its build reads the paper evidence}, from the manifest's ``evidence`` flags."""
    rows = (load_manifest() if manifest is None else manifest)["figures"]
    flags: dict[str, set[bool]] = {}
    for row in rows.values():
        if "evidence" in row:
            flags.setdefault(row["entry"], set()).add(row["evidence"])
    conflicting = sorted(name for name, values in flags.items() if len(values) > 1)
    if conflicting:
        raise RuntimeError(f"{MANIFEST_PATH}: conflicting evidence flags for {conflicting}")
    return {name: values.pop() for name, values in flags.items()}


def _tracked_source(source: str) -> bool:
    """Repository files only: evidence copies in the build cache are pinned by the
    packages' ``inputs.json`` locks, which are sources themselves."""
    path = (ROOT / source).resolve()
    return path.is_relative_to(ROOT) and not path.is_relative_to(paper.BUILD_DIR.resolve())


def update_manifest(report: BuildReport) -> None:
    manifest = load_manifest()
    entries = {name: entry.name for entry in REGISTRY for name in entry.outputs}
    for name, result in sorted(report.files.items()):
        path = result.pdf_path if isinstance(result, paper.FigureRecord) else result
        row: dict[str, object] = {
            "sha256": _sha256(path),
            "entry": entries[name],
            "evidence": report.evidence[entries[name]],
        }
        if isinstance(result, paper.FigureRecord):
            row.update(
                width_pt=round(result.width_pt, 3),
                height_pt=round(result.height_pt, 3),
                width_frac=result.width_frac,
                # Content hashes of the input locks, frozen data and rendering code.
                sources=[
                    {"path": source, "sha256": _sha256(ROOT / source)}
                    for source in sorted(set(result.sources))
                    if _tracked_source(source)
                ],
            )
        manifest["figures"][name] = row
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")


def check_inventory(entries: tuple[PaperFigure, ...] = REGISTRY) -> list[str]:
    """Registry outputs vs the frozen manuscript's included figures, both directions."""
    manuscript = json.loads(MANUSCRIPT_PATH.read_text())
    failures = []
    for part in ("main", "appendix"):
        expected = set(manuscript[f"{part}_figures"])
        registered = {name for e in entries if e.part == part for name in e.outputs}
        for name in sorted(expected - registered):
            failures.append(f"{part}: included by the manuscript but not registered: {name}")
        for name in sorted(registered - expected):
            failures.append(f"{part}: registered but not included by the manuscript: {name}")
    return failures


def check_pixels(path: Path) -> str | None:
    """Pixel comparison of a platform-dependent raster with the manuscript's; None if within
    ``RASTER_TOLERANCE``, else the failure message."""
    from paper import compare_reference

    with tempfile.TemporaryDirectory(prefix="paper-figures-check-") as scratch:
        c = compare_reference.compare(
            path, compare_reference.REFERENCE / path.name, None, Path(scratch)
        )
    if not c.size_match:
        return f"{path.name}: image size differs from the manuscript's"
    if (
        c.mean_abs_diff > RASTER_TOLERANCE["mean_abs_diff"]
        or c.differing_fraction > RASTER_TOLERANCE["differing_fraction"]
    ):
        return (
            f"{path.name}: pixels differ from the manuscript's (mean abs diff "
            f"{c.mean_abs_diff:.3f}, {100 * c.differing_fraction:.4f}% of pixels > "
            f"{compare_reference.PIXEL_TOLERANCE}/255; tolerance {RASTER_TOLERANCE})"
        )
    return None


def check(only: list[str] | None = None, exclude: list[str] = ()) -> int:
    """No-render validation of ``paper/build/figs`` against the bundled manifest."""
    failures = check_inventory() if only is None else []
    manifest = load_manifest()["figures"]
    entries = _select(only, list(exclude), ["main", "appendix"])
    for entry in entries:
        for name in entry.outputs:
            path = paper.FIGS_DIR / name
            recorded = manifest.get(name)
            if not path.is_file():
                failures.append(f"{name}: not built (run python -m paper.figures)")
                continue
            if recorded is None:
                failures.append(
                    f"{name}: no manifest entry (python -m paper.figures --update-manifest)"
                )
                continue
            if "evidence" not in recorded:
                failures.append(
                    f"{name}: no evidence flag in the manifest (python -m paper.figures "
                    "--update-manifest)"
                )
            if entry.deterministic and _sha256(path) != recorded["sha256"]:
                failures.append(f"{name}: built bytes differ from the manifest")
            if entry.pixel_check and (message := check_pixels(path)):
                failures.append(message)
            for source in recorded.get("sources", []):
                source_path = ROOT / source["path"]
                if not source_path.is_file():
                    failures.append(f"{name}: declared source missing: {source['path']}")
                elif _sha256(source_path) != source["sha256"]:
                    failures.append(f"{name}: source changed since the manifest: {source['path']}")
            if entry.width_frac is not None:
                expected_w = paper.DOC_TEXTWIDTH_IN["main"] * entry.width_frac * 72.0
                if abs(recorded["width_pt"] - expected_w) > 1.0:
                    failures.append(
                        f"{name}: manifest width {recorded['width_pt']}pt != declared "
                        f"{expected_w:.1f}pt"
                    )
    for message in failures:
        print(f"FAIL  {message}")
    files = sum(len(entry.outputs) for entry in entries)
    outcome = f"{len(failures)} failures" if failures else "all match"
    print(f"Checked {len(entries)} entries ({files} files): {outcome}")
    return 1 if failures else 0


def list_entries() -> None:
    evidence = recorded_evidence()
    for entry in REGISTRY:
        width = f" {entry.width_frac}x" if entry.width_frac else ""
        tags = f" [{', '.join(entry.tags)}]" if entry.tags else ""
        note = f" — {entry.note}" if entry.note else ""
        flag = _yes_no(evidence.get(entry.name))
        print(
            f"{entry.name:48s} {entry.part:8s} {entry.kind:8s} evidence: {flag:3s}"
            f"{width}{tags}{note}"
        )
    free = [entry.name for entry in REGISTRY if evidence.get(entry.name) is False]
    print(
        f"{len(REGISTRY)} entries; {len(free)} build without the paper evidence "
        f"(${artifacts.EVIDENCE_ENV}): {', '.join(free)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--list",
        action="store_true",
        help="list registry entries, and whether they read the paper evidence, and exit",
    )
    parser.add_argument("--only", nargs="+", metavar="NAME", help="build only these entries")
    parser.add_argument(
        "--exclude", nargs="+", default=[], metavar="NAME", help="skip entries (names or tags)"
    )
    parser.add_argument(
        "--part", nargs="+", choices=["main", "appendix"], default=["main", "appendix"]
    )
    parser.add_argument("--check", action="store_true", help="validate a finished build")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="build every entry that can be built, then list the failures (exit 1 if any)",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="after building, record the outputs in paper/figures_manifest.json",
    )
    args = parser.parse_args()
    if args.list:
        list_entries()
        return 0
    if args.check:
        return check(only=args.only, exclude=args.exclude)
    entries = _select(args.only, args.exclude, args.part)
    if not entries:
        raise SystemExit("nothing selected to build")
    report = build_entries(
        entries, keep_going=args.keep_going, record_evidence=args.update_manifest
    )
    if args.update_manifest:
        update_manifest(report)
    print(f"built {len(report.files)} file(s) in {paper.FIGS_DIR}")
    if args.keep_going:
        print_summary(report)
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
