import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mulligan.real.lifecycle.tasks import get_task_spec
from mulligan.data import split_protocol_quota as splitter
from mulligan.data.split_protocol_quota import _load_credited_rows
from mulligan.data.split_protocol_quota import _validate_real_manifest_metadata
from mulligan.sim.collect.quota import ProtocolQuotaLedger


@pytest.mark.parametrize(
    "module", ["mulligan.data.split_protocol_quota", "mulligan.data.split_blind"]
)
def test_splitters_import_without_arena_modules(module: str) -> None:
    # The Arena client package and the dropped registration modules.
    code = (
        "import sys\n"
        f"import {module}\n"
        "bad = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in ('policy_arena', 'convex')\n"
        "    or m.rsplit('.', 1)[-1] in ('arena_freshness', 'policy_arena_datasets', 'arena_resubmit')\n"
        ")\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_split_protocol_parser_has_no_arena_options() -> None:
    dests = {action.dest for action in splitter.build_parser()._actions}
    assert not {d for d in dests if "arena" in d}


def test_split_protocol_saved_mode_accepts_credited_failure(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    row = {
        "episode_index": 0,
        "success": False,
        "credited_protocol_arms": {"no_cf": ["baseline_uniform"]},
    }
    ledger.write_text(json.dumps(row) + "\n")

    assert _load_credited_rows(ledger, credit_outcome_mode="saved") == [row]


def test_split_protocol_success_mode_rejects_credited_failure(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "success": False,
                "credited_protocol_arms": {"no_cf": ["baseline_uniform"]},
            }
        )
        + "\n"
    )

    with pytest.raises(SystemExit, match="non-success row has credits"):
        _load_credited_rows(ledger, credit_outcome_mode="success")


def _write_square_d2_quota_manifest(path: Path) -> None:
    row = {
        "manifest_idx": 0,
        "nut_x": 0.01,
        "nut_y": -0.02,
        "nut_yaw": 0.3,
        "peg_x": 0.2413,
        "peg_y": -0.0635,
        "source": "sobol",
        "sources": ["sobol"],
    }
    path.write_text(
        json.dumps(
            {
                "task": "square_d2",
                "keys": list(get_task_spec("square_d2").manifest_keys),
                "match_tolerance": 1e-3,
                "states": [row],
            }
        )
    )


def _quota_for_manifest(manifest: Path, ledger: Path) -> ProtocolQuotaLedger:
    return ProtocolQuotaLedger(
        manifest_path=manifest,
        targets_by_protocol={"no_cf": 1, "with_cf": 1},
        ledger_path=ledger,
        arms_by_protocol={"no_cf": ["sobol"], "with_cf": ["sobol"]},
    )


def test_validate_real_manifest_metadata_accepts_square_d2_task_specific_columns(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_square_d2_quota_manifest(manifest)
    quota = _quota_for_manifest(manifest, ledger)
    spec = get_task_spec("square_d2")
    row = {"episode_index": 0, "manifest_idx": 0}
    source = SimpleNamespace(
        meta=SimpleNamespace(episodes=[{"dataset_from_index": 0}]),
        hf_dataset=[
            {
                "manifest_idx": 0,
                "nut_x": 0.01,
                "nut_y": -0.02,
                "nut_yaw": 0.3,
                "peg_x": 0.2413,
                "peg_y": -0.0635,
            }
        ],
    )

    _validate_real_manifest_metadata(source, 0, row, quota, spec)


def test_validate_real_manifest_metadata_rejects_pen_alias_for_square_d2(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    ledger = tmp_path / "ledger.jsonl"
    _write_square_d2_quota_manifest(manifest)
    quota = _quota_for_manifest(manifest, ledger)
    spec = get_task_spec("square_d2")
    row = {"episode_index": 0, "manifest_idx": 0}
    source = SimpleNamespace(
        meta=SimpleNamespace(episodes=[{"dataset_from_index": 0}]),
        hf_dataset=[
            {
                "manifest_idx": 0,
                "pen_x": 0.01,
                "pen_y": -0.02,
                "pen_yaw": 0.3,
                "peg_x": 0.2413,
                "peg_y": -0.0635,
            }
        ],
    )

    with pytest.raises(SystemExit, match="manifest key 'nut_x'.*missing"):
        _validate_real_manifest_metadata(source, 0, row, quota, spec)
