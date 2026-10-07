"""The SAC/RLPD agent reproduces its recorded reference updates bitwise.

The reference ``rlpd_sac_parity.npz`` (69 arrays: every actor / critic / target-critic /
temperature parameter after five UTD-20 updates, plus the update infos) lives on the
``mulligan/paper-evidence`` dataset under ``fixtures/`` and is identified by the sha256 in
``fixtures.json`` next to this file; ``MULLIGAN_PAPER_EVIDENCE`` points at a local mirror instead.

The replay runs in a subprocess on the JAX CPU backend (the reference was made on CPU; GPU
kernels are not bitwise-stable across devices).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FIXTURE = "rlpd_sac_parity.npz"


def _fixture_record() -> dict:
    records = json.loads((Path(__file__).with_name("fixtures.json")).read_text())["fixtures"]
    (record,) = [r for r in records if r["name"] == FIXTURE]
    return record


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_fixture() -> Path:
    """The fixture from the local mirror, else from the pinned HF dataset. Skips only when
    neither is configured (no mirror and no published revision); any other error fails."""
    from paper.appendix.artifacts import EVIDENCE_ENV, HF_REPO, HF_REVISION

    mirror = os.environ.get(EVIDENCE_ENV)
    if mirror:
        path = Path(mirror) / "fixtures" / FIXTURE
        if not path.exists():
            raise FileNotFoundError(f"{EVIDENCE_ENV} is set but {path} does not exist")
        return path
    if HF_REVISION is None:
        pytest.skip(
            f"paper evidence not configured: {HF_REPO} has no pinned revision yet and "
            f"{EVIDENCE_ENV} is unset; set it to a local mirror holding fixtures/{FIXTURE}"
        )
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(HF_REPO, f"fixtures/{FIXTURE}", repo_type="dataset", revision=HF_REVISION)
    )


@pytest.mark.network
def test_agent_matches_the_recorded_reference():
    pytest.importorskip("jax")
    record = _fixture_record()
    path = _fetch_fixture()
    assert _sha256(path) == record["sha256"], f"{path} is not the recorded fixture"

    env = dict(os.environ, JAX_PLATFORMS="cpu")
    proc = subprocess.run(
        [sys.executable, "-m", "tests.baselines.rlpd.sac_reference", str(path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "REFERENCE OK" in proc.stdout and "0 mismatches" in proc.stdout, proc.stdout


def test_fixture_fetch_skips_only_when_evidence_is_not_configured(monkeypatch, tmp_path):
    import huggingface_hub

    from paper.appendix import artifacts

    monkeypatch.delenv(artifacts.EVIDENCE_ENV, raising=False)
    monkeypatch.setattr(artifacts, "HF_REVISION", None)
    with pytest.raises(pytest.skip.Exception, match="not configured"):
        _fetch_fixture()

    def hub_error(*args, **kwargs):
        raise huggingface_hub.errors.EntryNotFoundError("fixture missing from the revision")

    monkeypatch.setattr(artifacts, "HF_REVISION", "0" * 40)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub_error)
    with pytest.raises(huggingface_hub.errors.EntryNotFoundError):
        _fetch_fixture()

    monkeypatch.setenv(artifacts.EVIDENCE_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _fetch_fixture()
