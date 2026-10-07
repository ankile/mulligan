"""Tests for the Fig. 3 task-sequence strips and the Fig. 4 reset card / range overlays."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mulligan.plotting import paper
from mulligan.release.download import pinned_revision
from paper import fig_reset_ranges, fig_tasks
from tests.paper.evidence import reads_evidence

ROOT = Path(__file__).resolve().parents[2]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
# sha256 of the frozen manuscript's figs/real_world_square_d2_reset_card.pdf.
CARD_PDF_SHA256 = "05be2049639b1915a865ea4e163b1c6c5d45bd64217eb65531b46e08547358f5"


@pytest.fixture
def build_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "BUILD_DIR", tmp_path)
    monkeypatch.setattr(paper, "FIGS_DIR", tmp_path / "figs")
    monkeypatch.setattr(paper, "PROOF_DIR", tmp_path / "proofs")
    return tmp_path


def test_task_sequence_provenance_is_consistent():
    prov = fig_tasks.load_provenance()
    assert set(prov["tasks"]) == {"marker", "nut", "cable"}
    assert set(prov["crops"]) == set(prov["tasks"])
    for name, (left, top, right, bottom) in prov["crops"].items():
        assert 0 <= left < right <= 640 and 0 <= top < bottom <= 480, name
        assert (right - left) * 3 == (bottom - top) * 4, f"{name}: crop is not 4:3"
    for name, task in prov["tasks"].items():
        assert task["repo"].startswith("mulligan/"), name
        assert HEX40.match(pinned_revision(task["repo"])), name
        assert 0 < task["num_steps"] <= task["length"], name
        indices = fig_tasks._pick_indices(name, task)
        assert len(set(indices)) == len(indices), name
        for pick in task["picks"]:
            assert pick["time_s"] == round(pick["frame"] / prov["fps"], 2)
            assert pick["label"]
        assert task["output"] == f"figs/task_sequence_{name}.jpg"


def test_task_sequences_are_median_final_round_successes():
    """Each strip is the median-length counted success (sorted by num_steps, then episode
    index; element len // 2) of the HiL-IDQL+Mulligan arm on the task's final-round block."""
    lock = {
        d["id"]: d
        for d in json.loads((ROOT / "release/round-datasets.json").read_text())["datasets"]
    }
    for name, task in fig_tasks.load_provenance()["tasks"].items():
        dataset = lock[task["repo"]]
        methods = {(p["session_id"], p["name"]): p["method"] for p in dataset["policies"]}
        (episode,) = [
            e
            for e in dataset["episodes"]
            if (e["session_id"], e["source_episode_index"])
            == (task["session_id"], task["source_episode_index"])
        ]
        assert methods[(episode["session_id"], episode["policy"])] == "HiL-IDQL+Mulligan", name
        assert (episode["num_steps"], episode["frames"]) == (task["num_steps"], task["length"])
        successes = sorted(
            (e["num_steps"], e["source_episode_index"])
            for e in dataset["episodes"]
            if e["success"] and e["role"] == "counted" and e["policy"] == episode["policy"]
        )
        assert successes[len(successes) // 2][1] == task["source_episode_index"], name
        assert "caveat" not in task, name


def test_side1_projection_lands_in_frame():
    for task in fig_reset_ranges.OVERLAY_SPECS:
        frame = fig_reset_ranges.OpFrame(task)
        px = frame.to_px([(0.0, 0.0)])[0]
        assert 0 <= px[0] < 640 and 0 <= px[1] < 480, (task, px)
        for key in ("mover", "place"):
            ring = fig_reset_ranges._project_box(
                frame, fig_reset_ranges.OVERLAY_SPECS[task][key]["box"]
            )
            assert np.isfinite(ring).all()


def test_card_target_is_the_square_row():
    row = fig_reset_ranges._card_target()
    assert row["manifest_idx"] == 1
    spec = fig_reset_ranges.get_task_spec("square_d2")
    (fwd_lo, fwd_hi), (left_lo, left_hi) = spec.bounds_arr[:2]
    assert fwd_lo <= row["nut_x"] <= fwd_hi and left_lo <= row["nut_y"] <= left_hi


def test_card_matches_frozen_figure(build_dir):
    record = fig_reset_ranges.build_card_record()
    assert record.pdf_path == build_dir / "figs/real_world_square_d2_reset_card.pdf"
    assert record.width_pt == pytest.approx(0.30 * 5.5 * 72)
    assert record.height_pt == pytest.approx(fig_reset_ranges.PANEL_HEIGHT_IN * 72)
    assert record.sha256 == CARD_PDF_SHA256
    assert record.proof_png.is_file()


@reads_evidence
def test_range_overlays_render(build_dir):
    paths = fig_reset_ranges.build_range_overlays()
    assert [p.name for p in paths] == [
        "real_world_square_d2_init_ranges_side1.png",
        "real_world_marker_d2_init_ranges_side1.png",
    ]
    for path in paths:
        assert Image.open(path).size == (1080, 902)


@pytest.mark.network
def test_task_sequences_render(build_dir):
    paths = fig_tasks.build_task_sequences()
    assert [p.name for p in paths] == [f"task_sequence_{n}.jpg" for n in ("marker", "nut", "cable")]
    for path in paths:
        assert Image.open(path).size == (2748, 409)
