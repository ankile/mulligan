"""Collectors have no default Hub namespace: a bare dataset name needs --hf-namespace."""

from __future__ import annotations

import sys

import pytest

MODEL = "hf://mulligan/real-marker-d2-r05-mulligan-dp"


def _parse(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    return module.parse_args()


@pytest.mark.parametrize("module_name", ["rollout", "dagger"])
def test_push_needs_namespace_for_bare_dataset_name(monkeypatch, capsys, module_name):
    import importlib

    module = importlib.import_module(f"mulligan.real.collect.{module_name}")
    base = ["--model", MODEL, "--task-name", "marker_d2", "--dataset-name", "my-dataset"]
    base.append("--push-to-hub")
    with pytest.raises(SystemExit):
        _parse(monkeypatch, module, base)
    assert "--hf-namespace" in capsys.readouterr().err

    args = _parse(monkeypatch, module, [*base, "--hf-namespace", "my-org"])
    assert args.hf_namespace == "my-org"
    args = _parse(
        monkeypatch,
        module,
        [
            "--model",
            MODEL,
            "--task-name",
            "marker_d2",
            "--dataset-name",
            "org/name",
            "--push-to-hub",
        ],
    )
    assert args.hf_namespace is None


@pytest.mark.parametrize("module_name", ["rollout", "dagger"])
def test_model_is_the_only_policy_source(monkeypatch, module_name):
    import importlib

    module = importlib.import_module(f"mulligan.real.collect.{module_name}")
    args = _parse(
        monkeypatch, module, ["--model", MODEL, "--task-name", "marker_d2", "--dataset-name", "d"]
    )
    assert args.model == MODEL
    for removed in (
        "wandb_artifact",
        "checkpoint_path",
        "openpi_config",
        "arena_url",
        "register_in_arena",
        "exec_gate",
    ):
        assert not hasattr(args, removed), removed
    with pytest.raises(SystemExit):
        _parse(monkeypatch, module, ["--dataset-name", "d"])


def test_push_dataset_requires_namespace_for_bare_repo_id(monkeypatch, capsys, tmp_path):
    from mulligan.tools import push_dataset

    monkeypatch.setattr(sys, "argv", ["push_dataset", "--repo-id", "bare", "--root", str(tmp_path)])
    with pytest.raises(SystemExit):
        push_dataset.main()
    assert "--hf-namespace" in capsys.readouterr().err


def _record_push(monkeypatch, tmp_path, extra):
    """Run push_dataset.main on a local dataset with every Hub call recorded."""
    import lerobot.datasets.io_utils as io_utils
    import lerobot.datasets.utils as lerobot_utils

    from mulligan.tools import push_dataset

    for name in ("data", "meta"):
        (tmp_path / name).mkdir()
    calls = {}

    class Card:
        def push_to_hub(self, **kwargs):
            calls["card_push"] = kwargs

    def make_card(**kwargs):
        calls["card"] = kwargs
        return Card()

    class Api:
        def __getattr__(self, name):
            return lambda **kwargs: calls.setdefault(name, kwargs)

    class Dataset:
        def __init__(self, **kwargs):
            pass

        def push_to_hub(self, **kwargs):
            calls["dataset_push"] = kwargs

    info = {"total_episodes": 1, "total_frames": 2, "fps": 10, "features": {}}
    monkeypatch.setattr(io_utils, "load_info", lambda root: info)
    monkeypatch.setattr(lerobot_utils, "create_lerobot_dataset_card", make_card)
    monkeypatch.setattr(push_dataset, "HfApi", Api)
    monkeypatch.setattr(push_dataset, "LeRobotDataset", Dataset)
    monkeypatch.setattr(push_dataset, "advance_lerobot_version_tag", lambda *a, **k: None)
    argv = ["push_dataset", "--repo-id", "org/demos", "--root", str(tmp_path), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    push_dataset.main()
    return calls


def test_push_dataset_license_defaults_to_mit(monkeypatch, tmp_path):
    calls = _record_push(monkeypatch, tmp_path, [])
    assert calls["dataset_push"]["license"] == "mit"


def test_push_dataset_license_flag_reaches_the_card(monkeypatch, tmp_path):
    calls = _record_push(monkeypatch, tmp_path, ["--force", "--license", "cc-by-4.0"])
    assert calls["card"]["license"] == "cc-by-4.0"
    assert calls["card_push"]["repo_id"] == "org/demos"
