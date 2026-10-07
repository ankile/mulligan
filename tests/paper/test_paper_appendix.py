"""Evidence integrity, scoped figure checks, and restoration of authored appendix assets."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from mulligan.plotting import paper
from paper import figures as driver
from paper.appendix import artifacts
from paper.appendix.reference_data import prepare


EVIDENCE_ROOTS = {"real", "sim", "assets"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def build_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(paper, "FIGS_DIR", tmp_path / "build/figs")
    return tmp_path / "build"


def _package(tmp_path, monkeypatch, data: bytes):
    """A one-file package pinned in a local evidence mirror."""
    mirror = tmp_path / "mirror"
    (mirror / "sim/study").mkdir(parents=True)
    (mirror / "sim/study/seeds.csv").write_bytes(data)
    package = tmp_path / "study"
    package.mkdir()
    lock = dict(
        schema=artifacts.SCHEMA,
        files=[dict(path="sim/study/seeds.csv", sha256=_sha(data), size=len(data))],
    )
    (package / "inputs.json").write_text(json.dumps(lock))
    monkeypatch.setenv(artifacts.EVIDENCE_ENV, str(mirror))
    monkeypatch.setattr(artifacts, "cache_dir", lambda name: tmp_path / "cache" / name)
    return package, mirror


def test_inputs_are_hash_locked_and_corrupt_cache_fails(tmp_path, monkeypatch):
    data = b"seed,success\n1,0.75\n"
    package, mirror = _package(tmp_path, monkeypatch, data)
    cached = artifacts.load_inputs(package)["sim/study/seeds.csv"]
    assert cached.read_bytes() == data
    assert cached.is_relative_to(tmp_path / "cache/study")
    # A warm cache is re-verified and never silently replaced.
    cached.write_bytes(b"seed,success\n1,0.50\n")
    with pytest.raises(RuntimeError, match="size mismatch|hash mismatch"):
        artifacts.load_inputs(package)
    # Changed evidence bytes are rejected before they reach the cache.
    cached.unlink()
    (mirror / "sim/study/seeds.csv").write_bytes(b"seed,success\n1,0.99\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        artifacts.load_inputs(package)
    assert not cached.exists()


def test_missing_mirror_file_and_failed_download_fail_loudly(tmp_path, monkeypatch):
    import huggingface_hub

    package, mirror = _package(tmp_path, monkeypatch, b"x,y\n")
    (mirror / "sim/study/seeds.csv").unlink()
    with pytest.raises(FileNotFoundError, match="MULLIGAN_PAPER_EVIDENCE"):
        artifacts.load_inputs(package)
    monkeypatch.delenv(artifacts.EVIDENCE_ENV)

    def offline(repo, filename, **kwargs):
        assert (repo, kwargs["revision"]) == (artifacts.HF_REPO, artifacts.HF_REVISION)
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(RuntimeError, match="MULLIGAN_PAPER_EVIDENCE"):
        artifacts.load_inputs(package)


def test_evidence_revision_is_a_pinned_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", artifacts.HF_REVISION)


def test_input_locks_name_role_paths_only():
    for path in sorted((artifacts.ROOT / "paper/appendix").glob("*/inputs.json")):
        lock = json.loads(path.read_text())
        assert lock["schema"] == artifacts.SCHEMA, path
        for row in lock["files"]:
            assert set(row) == {"path", "sha256", "size"}, path
            assert row["path"].split("/")[0] in EVIDENCE_ROOTS, (path, row["path"])
            assert not re.search(r"\d{4}-\d{2}-\d{2}", row["path"]), (path, row["path"])


def test_write_table_check_compares_with_the_manuscript(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "TABLES_DIR", tmp_path / "tables")
    reference = artifacts.REFERENCE_TABLES / "redq_table.tex"
    body = reference.read_text()
    assert artifacts.write_table("redq_table.tex", body, check=True).read_text() == body
    with pytest.raises(AssertionError, match="differs from the manuscript"):
        artifacts.write_table("redq_table.tex", body + "% edited\n", check=True)


def test_scoped_figure_check(build_dir, monkeypatch):
    data = b"figure bytes"
    (build_dir / "figs").mkdir(parents=True)
    (build_dir / "figs/appendix_plot.pdf").write_bytes(data)
    entries = (
        driver.PaperFigure(
            "appendix_plot", "appendix", "built", ("appendix_plot.pdf",), lambda: None, 1.0
        ),
        driver.PaperFigure("other", "main", "restored", ("other.pdf",), lambda: None),
    )
    manifest = {
        "appendix_plot.pdf": dict(sha256=_sha(data), width_pt=396.0, sources=[], evidence=True)
    }
    monkeypatch.setattr(driver, "REGISTRY", entries)
    monkeypatch.setattr(driver, "load_manifest", lambda: {"figures": manifest})
    assert driver.check(only=["appendix_plot"]) == 0
    record = manifest.pop("appendix_plot.pdf")
    assert driver.check(only=["appendix_plot"]) == 1
    manifest["appendix_plot.pdf"] = record
    (build_dir / "figs/appendix_plot.pdf").write_bytes(b"changed")
    assert driver.check(only=["appendix_plot"]) == 1
    with pytest.raises(SystemExit, match="unknown figure names"):
        driver.check(only=["typo"])


def test_restored_assets_copy_and_check(build_dir, tmp_path, monkeypatch):
    inputs = {}
    for name in prepare.ASSETS:
        path = tmp_path / "inputs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(name)
        inputs["assets/illustrations/" + name] = path
    monkeypatch.setattr(prepare, "load_inputs", lambda package: inputs)
    paths = prepare.materialize_assets()
    assert sorted(path.name for path in paths) == sorted(prepare.ASSETS)
    prepare.materialize_assets(check=True)
    paths[0].write_text("edited")
    with pytest.raises(AssertionError, match="differs from its archived export"):
        prepare.materialize_assets(check=True)


def test_data_ledger_constants_are_documented_and_used():
    """Every count in ``tracker_values.json`` is an integer with a description, and the
    ledger reads each one (the ledger table itself is checked by the appendix build)."""
    from paper.appendix.data_ledger import prepare as ledger

    values = json.loads((ledger.HERE / "tracker_values.json").read_text())
    used = set(re.findall(r'documented_count\("(\w+)"\)', (ledger.HERE / "prepare.py").read_text()))
    assert set(values) == used
    for key, row in values.items():
        assert set(row) == {"value", "description"}, key
        assert isinstance(row["value"], int) and row["value"] > 0, key
        assert row["description"].strip(), key
        assert ledger.documented_count(key) == row["value"]


def test_check_on_empty_build_names_the_build_command(build_dir, monkeypatch):
    from paper.appendix import build

    monkeypatch.setattr("sys.argv", ["build", "--check"])
    with pytest.raises(
        SystemExit, match=r"missing from .*initial_state_marker_hardest\.jpg.*paper\.figures"
    ):
        build.main()
