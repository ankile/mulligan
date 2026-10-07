"""Tests for the paper-figure style contract (mulligan.plotting.paper) and the
figure driver registry (paper.figures)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from mulligan.plotting import paper
from paper.plotting import sim_paper_headline
from tests.paper.evidence import reads_evidence

ROOT = Path(__file__).resolve().parents[2]


def _config_values(key: str) -> dict[str, object]:
    """{task_key: value of ``key``} over the frozen paper line-headline configs."""
    config = json.loads((ROOT / "paper/appendix/real_results/config.json").read_text())
    out = {}

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "LineHeadlineConfig":
                fields = node["fields"]
                value = fields[key]
                out[fields["task_key"]] = (
                    value.get("tuple", value) if isinstance(value, dict) else value
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(config)
    return out


def test_paper_rc_restores_rcparams_including_hashsalt():
    before_size = matplotlib.rcParams["font.size"]
    before_salt = matplotlib.rcParams["svg.hashsalt"]
    with paper.paper_rc():
        assert matplotlib.rcParams["font.size"] == paper.PAPER_RC["font.size"]
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        # Simulate the line-headline plots mutating the hashsalt mid-block.
        matplotlib.rcParams["svg.hashsalt"] = "scribbled-by-a-plot"
    assert matplotlib.rcParams["font.size"] == before_size
    assert matplotlib.rcParams["svg.hashsalt"] == before_salt


def test_paper_rc_never_uses_tight_bbox():
    with paper.paper_rc():
        assert matplotlib.rcParams["savefig.bbox"] is None


def test_save_paper_figure_rejects_wrong_width():
    fig = plt.figure(figsize=(4.0, 2.0))  # not 5.5 * 1.0
    try:
        with pytest.raises(ValueError, match="width_frac"):
            paper.save_paper_figure(fig, "bogus", width_frac=1.0)
    finally:
        plt.close(fig)


def test_save_paper_figure_writes_pdf_and_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PROOF_DIR", tmp_path)
    monkeypatch.setattr(paper, "FIGS_DIR", tmp_path)
    fig = plt.figure(figsize=paper.fig_size(0.5, height_in=1.0))
    record = paper.save_paper_figure(fig, "unit_test_figure", width_frac=0.5, sources=("a.csv",))
    assert record.pdf_path.exists()
    assert record.proof_png.exists()
    assert record.width_pt == pytest.approx(0.5 * 5.5 * 72.0)
    assert record.sources == ("a.csv",)
    assert len(record.sha256) == 64


def test_save_paper_figure_rejects_title_outside_canvas(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PROOF_DIR", tmp_path)
    monkeypatch.setattr(paper, "FIGS_DIR", tmp_path)
    fig, axis = plt.subplots(figsize=paper.fig_size(1.0, height_in=1.0))
    axis.set_title("Clipped title", x=1.5)
    try:
        with pytest.raises(ValueError, match="panel title extends outside"):
            paper.save_paper_figure(fig, "clipped_title", width_frac=1.0)
    finally:
        plt.close(fig)


def test_save_paper_figure_checks_restyled_task_title_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "PROOF_DIR", tmp_path)
    monkeypatch.setattr(paper, "FIGS_DIR", tmp_path)
    fig, axes = plt.subplots(1, 3, figsize=paper.fig_size(1.0, height_in=1.0))
    axes[-1].set_title("Square-Narrow failure rate (sim)")
    try:
        with pytest.raises(ValueError, match="panel title extends outside"):
            paper.save_paper_figure(fig, "clipped_task_title", width_frac=1.0)
    finally:
        plt.close(fig)


def test_fig_size_matches_textwidth():
    width, height = paper.fig_size(0.9, height_in=2.6)
    assert width == pytest.approx(0.9 * paper.DOC_TEXTWIDTH_IN["main"])
    assert height == 2.6


def test_darken_returns_hex():
    darker = paper.darken(paper.OURS)
    assert darker.startswith("#") and len(darker) == 7
    assert darker != paper.OURS


def test_y_axis_break_has_fixed_printed_width_across_unequal_panels():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0), gridspec_kw={"width_ratios": [2.0, 1.0]})
    try:
        for axis in axes:
            paper.y_axis_break(axis)
        widths_pt = []
        for axis in axes:
            slash = axis.lines[-1]
            x0, x1 = slash.get_xdata()
            axis_width_pt = axis.get_position().width * fig.get_figwidth() * 72.0
            widths_pt.append((x1 - x0) * axis_width_pt)
        assert widths_pt == pytest.approx([4.0, 4.0])
    finally:
        plt.close(fig)


def test_sampler_ablation_panels_share_natural_success_rate_ceiling():
    from paper.plotting import square_narrow_r1_sampler as sampler

    cells = {
        ("no_cf", "human_only", "baseline_uniform"): [85.0, 86.0, 84.5, 87.0, 86.0],
        ("no_cf", "human_only", "sobol"): [89.0, 90.0, 88.5, 91.0, 90.0],
        ("no_cf", "human_only", "mulligan"): [92.0, 93.0, 92.5, 91.5, 93.0],
        ("with_cf", "human_only", "mulligan"): [93.0, 94.0, 93.5, 92.5, 94.0],
        ("no_cf", "straddled_auto_success", "baseline_uniform"): [82.0, 83.0, 81.5, 84.0, 82.0],
    }
    fig, axes = plt.subplots(1, 2)
    try:
        sampler.draw_sampler_panels(*axes, by_grid=cells)
        assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
        assert axes[0].get_ylim() == pytest.approx(sampler.SHARED_YLIM)
        assert axes[1].get_ylim() == pytest.approx(sampler.SHARED_YLIM)
        assert tuple(axes[0].get_yticks()) == pytest.approx(sampler.SHARED_YTICKS)
        assert not any(label.get_visible() for label in axes[1].get_yticklabels())
    finally:
        plt.close(fig)


@reads_evidence
def test_combined_ablation_shows_both_cf_tasks_and_omits_sobol():
    from paper.appendix.component_panels.plot import ablations
    from paper.appendix.component_panels.prepare import prepare

    groups, paths, _, _ = prepare()
    with paper.paper_rc():
        fig = ablations(groups, paths)
    axes = fig.axes
    assert len(axes) == 4
    assert [label.get_text() for label in axes[0].get_xticklabels()] == [
        "Uniform",
        "Guided",
        "Guided\n+ CF",
    ]
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    assert axes[2].get_shared_y_axes().joined(axes[2], axes[3])
    assert [axis.get_title() for axis in axes[2:]] == ["Thread Nut", "Insert Marker"]
    assert [label.get_text() for label in axes[2].get_xticklabels()] == [f"R{r}" for r in range(5)]
    assert [label.get_text() for label in axes[3].get_xticklabels()] == [f"R{r}" for r in range(3)]
    assert axes[2].get_ylim() == pytest.approx((0, 100))
    plt.close(fig)


def test_sim_headline_uses_student_t_ci95_over_seeds():
    mean, half_width = sim_paper_headline._mean_ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == pytest.approx(3.0)
    assert half_width == pytest.approx(1.963243, rel=1e-6)


def test_real_paper_default_is_boundary_safe_wilson_one_se():
    from paper.real_headline import PAPER_WHISKERS
    from mulligan.real.lifecycle.stats import wilson_1se

    assert PAPER_WHISKERS[0] == "wilson1se"
    assert wilson_1se(0, 50) == pytest.approx((0.0, 1.0 / 51.0))
    assert wilson_1se(50, 50) == pytest.approx((50.0 / 51.0, 1.0))


def test_sim_headline_rejects_partial_rounds():
    rows = [
        {
            "task_key": "square_narrow",
            "series_key": "mulligan_n32",
            "round": "3",
            "seed": str(seed),
            "n_planned_seeds": "5",
            "point_status": "complete",
        }
        for seed in range(1, 5)
    ]
    with pytest.raises(RuntimeError, match="INCOMPLETE SIM ROUND"):
        sim_paper_headline._validate_complete_seed_groups(rows, source="test.csv")


def test_task_titles_cover_all_line_configs():
    task_keys = _config_values("task_key")
    assert set(task_keys) == {"marker_d2", "square_d2", "routing_d2"}
    for task_key in task_keys:
        assert task_key in paper.TASK_TITLES, task_key
        # Verb form: starts with an imperative verb, no task_key jargon.
        assert "_" not in paper.TASK_TITLES[task_key]


def test_policy_arm_labels_cover_headline_arms():
    for task_key, arms in _config_values("headline_arms").items():
        for arm in arms:
            assert arm in paper.POLICY_ARM_LABELS, (task_key, arm)


def test_paper_labels_name_method_and_deployment_operator():
    assert paper.POLICY_ARM_LABELS["baseline"] == "HG-DAgger"
    assert paper.POLICY_ARM_LABELS["final_iql"] == "HiL-IDQL+Mulligan"
    assert paper.STATIC_BC_LABEL == "Static BC (R0; no further data)"
    assert paper.static_bc_legend_handle().get_label() == paper.STATIC_BC_LABEL


def test_registry_covers_the_manuscript_figures():
    from paper.figures import check_inventory

    assert check_inventory() == []


def test_registry_names_and_outputs_are_unique():
    from paper.figures import REGISTRY

    names = [entry.name for entry in REGISTRY]
    outputs = [name for entry in REGISTRY for name in entry.outputs]
    assert len(names) == len(set(names))
    assert len(outputs) == len(set(outputs))


def test_missing_chrome_is_a_hard_error_naming_the_exclusion(monkeypatch):
    from paper import figures
    from paper.teaser import build_teaser

    monkeypatch.setenv(build_teaser.CHROME_ENV, "/nonexistent/chrome")
    with pytest.raises(SystemExit, match="--exclude teaser"):
        figures.build_entries(figures._select(None, [], ["main"]))
    assert not any("teaser" in e.tags for e in figures._select(None, ["teaser"], ["main"]))


def test_platform_dependent_rasters_are_pixel_checked():
    from paper.figures import REGISTRY

    pixel = {name for e in REGISTRY if e.pixel_check for name in e.outputs}
    assert pixel == {
        "task_sequence_marker.jpg",
        "task_sequence_nut.jpg",
        "task_sequence_cable.jpg",
        "real_world_square_d2_init_ranges_side1.png",
        "real_world_marker_d2_init_ranges_side1.png",
        "initial_state_marker_typical.jpg",
        "initial_state_marker_hardest.jpg",
        "initial_state_nut_typical.jpg",
        "initial_state_nut_hardest.jpg",
        "initial_state_nut_reorientation.jpg",
    }
    assert {e.name for e in REGISTRY if not e.deterministic and not e.pixel_check} == {
        "overview_teaser"
    }


def test_pixel_check_tolerates_noise_and_rejects_changes(tmp_path):
    import numpy as np
    from PIL import Image

    from paper import compare_reference
    from paper.figures import check_pixels

    name = "real_world_marker_d2_init_ranges_side1.png"
    image = np.asarray(Image.open(compare_reference.REFERENCE / name).convert("RGB"), dtype=int)
    noisy = tmp_path / "noisy" / name
    noisy.parent.mkdir()
    Image.fromarray(np.clip(image + 1, 0, 255).astype(np.uint8)).save(noisy)
    assert check_pixels(noisy) is None
    changed = tmp_path / "changed" / name
    changed.parent.mkdir()
    edited = image.copy()
    edited[: image.shape[0] // 4] = 255 - edited[: image.shape[0] // 4]
    Image.fromarray(edited.astype(np.uint8)).save(changed)
    assert "pixels differ" in check_pixels(changed)


# --- evidence flags, --keep-going, --list, --check count --------------------------


@pytest.fixture
def cold_build(tmp_path, monkeypatch):
    """An empty build directory (no warm evidence cache), no local mirror, and no Hub."""
    import huggingface_hub

    from paper.appendix import artifacts

    def offline(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(paper, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(paper, "FIGS_DIR", tmp_path / "build/figs")
    monkeypatch.setattr(paper, "PROOF_DIR", tmp_path / "build/proofs")
    monkeypatch.delenv(artifacts.EVIDENCE_ENV, raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    return tmp_path / "build"


def _write(name: str):
    def build():
        paper.FIGS_DIR.mkdir(parents=True, exist_ok=True)
        path = paper.FIGS_DIR / name
        path.write_text(name)
        return path

    return build


def _read_evidence(tmp_path, monkeypatch):
    """A builder that reads one pinned file from a local mirror, then writes ``read.pdf``."""
    import hashlib

    from paper.appendix import artifacts

    data = b"seed,success\n1,0.5\n"
    (tmp_path / "mirror/sim/study").mkdir(parents=True)
    (tmp_path / "mirror/sim/study/seeds.csv").write_bytes(data)
    monkeypatch.setenv(artifacts.EVIDENCE_ENV, str(tmp_path / "mirror"))

    def build():
        artifacts.evidence_file(
            "sim/study/seeds.csv",
            digest=hashlib.sha256(data).hexdigest(),
            size=len(data),
            destination=tmp_path / "cache/seeds.csv",
        )
        return _write("read.pdf")()

    return build


def _boom():
    raise RuntimeError("boom\nsecond line")


def test_manifest_records_evidence_for_every_entry():
    from paper import figures

    manifest = figures.load_manifest()
    assert all("evidence" in row for row in manifest["figures"].values())
    assert set(figures.recorded_evidence(manifest)) == {entry.name for entry in figures.REGISTRY}


def test_evidence_flags_agree_with_the_recorded_sources():
    """Figures that read the evidence pin its reader (artifacts.py) among their sources."""
    from paper import figures

    for name, row in figures.load_manifest()["figures"].items():
        if "sources" in row:
            reader = "paper/appendix/artifacts.py" in {s["path"] for s in row["sources"]}
            assert row["evidence"] == reader, name


def test_entries_without_evidence_build_from_a_cold_cache(cold_build):
    """Offline half of the flag's contract; the networked task sequences are left out."""
    from paper import figures

    free = [
        e
        for e in figures.REGISTRY
        if figures.recorded_evidence()[e.name] is False and "network" not in e.tags
    ]
    assert {e.name for e in free} >= {"sim_state_rlpd_vs_mulligan_compact"}
    report = figures.build_entries(free)
    assert report.evidence == {e.name: False for e in free}
    assert all((cold_build / "figs" / name).is_file() for e in free for name in e.outputs)


def test_entries_with_evidence_fail_without_it(cold_build):
    from paper import figures

    evidence = figures.recorded_evidence()
    needed = [e for e in figures.REGISTRY if evidence[e.name]]
    report = figures.build_entries(needed, keep_going=True)
    assert report.built == []
    assert set(report.failures) == {e.name for e in needed}
    for name, reason in report.failures.items():
        # Without Chrome, the teaser fails before it reaches the evidence.
        assert "MULLIGAN_PAPER_EVIDENCE" in reason or "Chrome" in reason, (name, reason)


def test_build_records_evidence_use_and_rejects_a_stale_flag(tmp_path, monkeypatch, cold_build):
    from paper import figures

    entries = [
        figures.PaperFigure(
            "reads", "main", "restored", ("read.pdf",), _read_evidence(tmp_path, monkeypatch)
        ),
        figures.PaperFigure("local", "main", "restored", ("local.pdf",), _write("local.pdf")),
    ]
    monkeypatch.setattr(figures, "load_manifest", lambda: {"figures": {}})
    report = figures.build_entries(entries, record_evidence=True)
    assert report.evidence == {"reads": True, "local": False}
    # A second read of the (now warm) cache still counts as reading the evidence.
    assert figures.build_entries(entries[:1]).evidence == {"reads": True}

    stale = {"local.pdf": {"entry": "local", "evidence": True, "sha256": ""}}
    monkeypatch.setattr(figures, "load_manifest", lambda: {"figures": stale})
    with pytest.raises(RuntimeError, match="manifest records evidence: yes"):
        figures.build_entries(entries)
    report = figures.build_entries(entries, keep_going=True)
    assert report.built == ["reads"]
    assert "--update-manifest" in report.failures["local"]


def test_keep_going_builds_the_rest_and_exits_nonzero(monkeypatch, capsys, cold_build):
    from paper import figures

    entries = (
        figures.PaperFigure("broken", "main", "restored", ("broken.pdf",), _boom),
        figures.PaperFigure("fine", "appendix", "restored", ("fine.pdf",), _write("fine.pdf")),
    )
    monkeypatch.setattr(figures, "REGISTRY", entries)
    monkeypatch.setattr(figures, "load_manifest", lambda: {"figures": {}})
    monkeypatch.setattr("sys.argv", ["figures", "--keep-going"])
    assert figures.main() == 1
    out = capsys.readouterr().out
    assert "── summary: 1 built, 1 failed" in out
    assert "built   fine" in out
    assert "FAILED  broken: RuntimeError: boom\n" in out
    assert (cold_build / "figs/fine.pdf").is_file()

    # Without the flag, the first failure stops the build.
    (cold_build / "figs/fine.pdf").unlink()
    monkeypatch.setattr("sys.argv", ["figures"])
    with pytest.raises(RuntimeError, match="boom"):
        figures.main()
    assert not (cold_build / "figs/fine.pdf").exists()


def test_keep_going_records_a_missing_chrome(monkeypatch, cold_build):
    from paper import figures
    from paper.teaser import build_teaser

    monkeypatch.setenv(build_teaser.CHROME_ENV, "/nonexistent/chrome")
    entries = [
        figures.PaperFigure("teaser", "main", "rendered", ("t.pdf",), _boom, tags=("teaser",)),
        figures.PaperFigure("fine", "main", "restored", ("fine.pdf",), _write("fine.pdf")),
    ]
    monkeypatch.setattr(figures, "load_manifest", lambda: {"figures": {}})
    report = figures.build_entries(entries, keep_going=True)
    assert report.built == ["fine"]
    assert "--exclude teaser" in report.failures["teaser"]


def test_list_shows_the_evidence_flag(capsys):
    from paper import figures

    figures.list_entries()
    lines = capsys.readouterr().out.splitlines()
    by_name = {line.split()[0]: line for line in lines[:-1]}
    assert set(by_name) == {entry.name for entry in figures.REGISTRY}
    assert "evidence: no " in by_name["sim_state_rlpd_vs_mulligan_compact"]
    assert "evidence: yes" in by_name["headline_real_sim"]
    assert lines[-1].startswith(f"{len(figures.REGISTRY)} entries; ")
    assert (
        "sim_state_rlpd_vs_mulligan_compact" in lines[-1] and "headline_real_sim" not in lines[-1]
    )


def test_check_reports_the_number_of_checked_entries(monkeypatch, capsys, cold_build):
    import hashlib

    from paper import figures

    entries = (
        figures.PaperFigure("one", "main", "restored", ("a.pdf", "b.pdf"), lambda: None),
        figures.PaperFigure("two", "main", "restored", ("c.pdf",), lambda: None),
    )
    (cold_build / "figs").mkdir(parents=True)
    manifest = {}
    for entry in entries:
        for name in entry.outputs:
            (cold_build / "figs" / name).write_text(name)
            digest = hashlib.sha256(name.encode()).hexdigest()
            manifest[name] = {"entry": entry.name, "evidence": False, "sha256": digest}
    monkeypatch.setattr(figures, "REGISTRY", entries)
    monkeypatch.setattr(figures, "load_manifest", lambda: {"figures": manifest})
    assert figures.check(only=["one", "two"]) == 0
    assert capsys.readouterr().out == "Checked 2 entries (3 files): all match\n"
    del manifest["c.pdf"]["evidence"]
    assert figures.check(only=["one", "two"]) == 1
    out = capsys.readouterr().out
    assert "c.pdf: no evidence flag in the manifest" in out
    assert out.endswith("Checked 2 entries (3 files): 1 failures\n")
