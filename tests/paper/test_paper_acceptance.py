"""Acceptance: rebuild every figure and table from any working directory and check them.

Runs ``python -m paper.figures`` into a temporary build directory from a temporary working
directory (no AWS or W&B credentials involved), then ``--check`` against the bundled
manifest and ``python -m paper.appendix.build --check`` against the frozen manuscript.
The teaser (Chrome) and the task sequences (HF videos) are covered by the ``network``
variant. The paper evidence comes from ``$MULLIGAN_PAPER_EVIDENCE`` (a local mirror) or the
Hub, which makes the test a network test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.paper.evidence import reads_evidence

ROOT = Path(__file__).resolve().parents[2]


def _has_chrome() -> bool:
    from paper.teaser.build_teaser import ChromeNotFound, chrome_binary

    try:
        chrome_binary()
    except ChromeNotFound:
        return False
    return True


def _run(args: list[str], cwd: Path, build: Path) -> str:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("AWS_", "WANDB_"))
    }
    env.update(MULLIGAN_PAPER_BUILD=str(build), PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result.stdout


def _rebuild_and_check(tmp_path: Path, exclude: list[str]) -> None:
    build, cwd = tmp_path / "build", tmp_path / "elsewhere"
    cwd.mkdir()
    skip = ["--exclude", *exclude] if exclude else []
    _run(["-m", "paper.figures", *skip], cwd, build)
    assert ": all match" in _run(["-m", "paper.figures", "--check", *skip], cwd, build)
    out = _run(["-m", "paper.appendix.build", "--check"], cwd, build)
    assert "Generated tables equal the manuscript's" in out


@pytest.mark.slow
@reads_evidence
def test_rebuild_everything_but_chrome_and_hf_videos(tmp_path):
    _rebuild_and_check(tmp_path, ["teaser", "task_sequences"])


@pytest.mark.slow
@pytest.mark.network
@reads_evidence
@pytest.mark.skipif(not _has_chrome(), reason="the teaser needs Chrome or Chromium")
def test_rebuild_everything(tmp_path):
    _rebuild_and_check(tmp_path, [])
