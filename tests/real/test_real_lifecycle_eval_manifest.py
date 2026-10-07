from pathlib import Path

import numpy as np
import pytest

from mulligan.real.eval.eval_manifest import (
    EvalManifestStateOverride,
    _apply_manual_state_corrections,
    _audit_manifest_label,
    _compress_ranges,
    _min_cross,
    _min_pairwise,
)


def test_compress_ranges_collapses_contiguous_and_splits_gaps() -> None:
    # A full contiguous claim collapses to one token (the start..end form).
    assert _compress_ranges(set(range(0, 50))) == "0..49"
    assert _compress_ranges({7}) == "7..7"
    # Rejection-sampling gaps produce multiple tokens, in sorted order.
    assert _compress_ranges({0, 1, 2, 5, 6, 9}) == "0..2,5..6,9..9"

    with pytest.raises(RuntimeError, match="empty stream-index set"):
        _compress_ranges(set())


def test_min_pairwise_and_min_cross_periodic() -> None:
    # dim 0 non-periodic (control), dim 1 periodic (wraparound matters).
    periodic_mask = np.array([False, True])

    # Rows 0 and 1 are far by direct distance in the periodic dim (0.85) but
    # close across the wrap (0.15), which must be the pairwise minimum.
    unit = np.array(
        [
            [0.0, 0.10],
            [0.0, 0.95],
            [0.5, 0.10],
        ]
    )
    d01_periodic = min(abs(0.10 - 0.95), 1.0 - abs(0.10 - 0.95))
    expected_pairwise = float(np.linalg.norm(np.array([0.0, d01_periodic])))
    assert _min_pairwise(unit, periodic_mask) == expected_pairwise

    # Cross set: the single a-row is closest to b-row 0 through the periodic
    # wrap (0.07), not to the b-row that is near in the non-periodic dim.
    a_unit = np.array([[0.2, 0.05]])
    b_unit = np.array(
        [
            [0.2, 0.98],
            [0.9, 0.05],
        ]
    )
    d_periodic = min(abs(0.05 - 0.98), 1.0 - abs(0.05 - 0.98))
    expected_cross = float(np.linalg.norm(np.array([0.0, d_periodic])))
    assert _min_cross(a_unit, b_unit, periodic_mask) == expected_cross


def test_manual_state_corrections_update_rows_and_record_audit() -> None:
    rows = [
        {"manifest_idx": 0, "peg_x": 1.0, "peg_y": 2.0},
        {"manifest_idx": 1, "peg_x": 3.0, "peg_y": 4.0},
    ]

    applied = _apply_manual_state_corrections(
        rows,
        (
            EvalManifestStateOverride(
                manifest_idx=1,
                values={"peg_x": 5.0},
                reason="operator-reviewed reset frame",
            ),
        ),
    )

    assert rows[1]["peg_x"] == 5.0
    assert applied == [
        {
            "manifest_idx": 1,
            "updates": {"peg_x": {"old": 3.0, "new": 5.0}},
            "reason": "operator-reviewed reset frame",
        }
    ]


def test_manual_state_corrections_fail_loudly_for_invalid_contract() -> None:
    rows = [{"manifest_idx": 0, "peg_y": 2.0}]

    with pytest.raises(RuntimeError, match="missing manifest_idx=7"):
        _apply_manual_state_corrections(
            rows,
            (
                EvalManifestStateOverride(
                    manifest_idx=7,
                    values={"peg_y": 3.0},
                    reason="bad row",
                ),
            ),
        )

    with pytest.raises(RuntimeError, match="references missing key 'peg_x'"):
        _apply_manual_state_corrections(
            rows,
            (
                EvalManifestStateOverride(
                    manifest_idx=0,
                    values={"peg_x": 3.0},
                    reason="bad key",
                ),
            ),
        )


def test_audit_manifest_label_uses_repo_relative_paths() -> None:
    root = Path("/repo")
    path = root / "runs/r03/blind_inputs/manifest.json"

    assert _audit_manifest_label(path, root) == "runs/r03/blind_inputs/manifest.json"

    with pytest.raises(RuntimeError, match="not under audit_path_root"):
        _audit_manifest_label(Path("/tmp/other/manifest.json"), root)
