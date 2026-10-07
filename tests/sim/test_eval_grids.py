"""The locked sim eval grids regenerate byte-for-byte (configs/sim/README.md, Evaluation grids).

``POINTS_SHA256`` is the sha256 of the canonical JSON of each grid's ``points``. It
identifies the evaluation starts independently of the task and protocol names recorded
in the manifest (and so of ``manifest_hash``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mulligan.sim.eval import grid_eval

GRIDS = {
    "square_narrow_valid_sobol8k_seed2026052402.json": (
        [
            "make-valid-sobol-manifest",
            "--task",
            "square_narrow",
            "--num-points",
            "8000",
            "--seed",
            "2026052402",
        ],
        "45a9963f5592f495477c44beabe8d3c3e3173e920c2b2ba009c1e69532e53b17",
        "068b2deb11e553cdedfb0f789081b6fd7c4112aa7e22ecc3b655f90492168476",
        "4264115a4fdb4008e5037a7d4995a90cdec221e9db8d9c1e1169b7d368bee27c",
    ),
    "square_broad_valid_sobol30k_seed2026052499.json": (
        [
            "make-valid-sobol-manifest",
            "--task",
            "square_broad",
            "--num-points",
            "30000",
            "--seed",
            "2026052499",
            "--candidate-power",
            "16",
        ],
        "e0b5056648639f54189386a92384212d4fb09949bda23b622c10f8427aaa085c",
        "304210db9e7b3d175970f846753e2cd26ab59686fef3693c6f24f29c3f55a5e5",
        "03d92cd94e579d3adb4a9be7ca8018923948c2be33695ab9d3ac673d2f3e8d8a",
    ),
}
EQUAL_TILE_HASH = "41166760da933ccca99e5ac01694fbcb1ab2042312abcdded7ac7a67a724dbea"
EQUAL_TILE_SHA256 = "d8ee57bff9e88fbe4ba2807d8d9ae672b2f62ccd6af3c9c02baeccf9f7117624"
EQUAL_TILE_POINTS_SHA256 = "e1d58add55ccc0f9cbc6f27023603eee44709ca251008a676a08c3ac0707ea3a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _points_sha256(manifest: dict) -> str:
    canonical = json.dumps(manifest["points"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_valid_sobol_grid_regenerates(tmp_path: Path, name: str):
    argv, expected_hash, expected_sha256, expected_points = GRIDS[name]
    output = tmp_path / name
    assert grid_eval.main([*argv, "--output", str(output)]) == 0
    manifest = json.loads(output.read_text())
    assert manifest["manifest_hash"] == expected_hash
    assert grid_eval.manifest_hash(manifest) == expected_hash
    assert _sha256(output) == expected_sha256
    assert _points_sha256(manifest) == expected_points


def test_equal_tile_grid_regenerates(tmp_path: Path):
    output = tmp_path / "square_narrow_equal_tile_sobol_100pcell_seed20260524.json"
    argv = ["make-equal-tile-manifest", "--points-per-cell", "100", "--seed", "20260524"]
    assert grid_eval.main([*argv, "--output", str(output)]) == 0
    manifest = json.loads(output.read_text())
    assert manifest["manifest_hash"] == EQUAL_TILE_HASH
    assert grid_eval.manifest_hash(manifest) == EQUAL_TILE_HASH
    assert _sha256(output) == EQUAL_TILE_SHA256
    assert _points_sha256(manifest) == EQUAL_TILE_POINTS_SHA256
