"""``python -m mulligan.real.operator_ui.preview``: cards from a manifest without the robot stack."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mulligan.real.operator_ui import preview

REPO = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO / "data/real/manifests/routing_d2/r08/routing_d2_r8_promote25_fill25_blind_dagger.json"
)

_needs_manifest = pytest.mark.skipif(not MANIFEST.exists(), reason="R8 manifest not present")


@_needs_manifest
def test_preview_renders_the_requested_rows(tmp_path, capsys):
    preview.main(
        ["--manifest", str(MANIFEST), "--idx", "0", "--idx", "3", "--out-dir", str(tmp_path)]
    )
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["target_0000.png", "target_0003.png"]
    out = capsys.readouterr().out
    assert "task=routing_d2" in out and "manifest_idx=3" in out


@_needs_manifest
def test_preview_collection_style_and_bad_index(tmp_path):
    import cv2

    preview.main(["--manifest", str(MANIFEST), "--collection-style", "--out-dir", str(tmp_path)])
    assert (tmp_path / "target_0000.png").exists()
    for phase in ("setup", "policy", "human", "saving", "choose"):
        panel = cv2.imread(str(tmp_path / f"target_0000_collect_{phase}.png"))
        assert panel.shape == (926, 1000, 3), phase
    with pytest.raises(SystemExit, match="no rows with manifest_idx"):
        preview.main(["--manifest", str(MANIFEST), "--idx", "999999", "--out-dir", str(tmp_path)])


@_needs_manifest
def test_dashboard_preview_covers_setup_running_and_reset(tmp_path):
    import cv2

    preview.main(["--manifest", str(MANIFEST), "--dashboard", "--out-dir", str(tmp_path)])
    for phase in ("setup", "running", "reset"):
        panel = cv2.imread(str(tmp_path / f"target_0000_{phase}.png"))
        assert panel.shape == (926, 1000, 3)


def test_preview_module_stays_light():
    code = (
        "import sys; import mulligan.real.operator_ui.preview; "
        "heavy = sorted(m for m in ('torch', 'lerobot') if m in sys.modules); "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)
