"""--dataset-revisions / --dataset-episodes in both real trainers (no network).

The loading tests run the real ``train()`` / ``main()`` for two steps on the tiny local
LeRobot fixtures of the trainer harnesses, with the Hub sync disabled, and check
from the checkpoint metadata that only the selected episodes were read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from mulligan.real.train import critic, iql_args, policy
from mulligan.real.train.dataset_selectors import EPISODE_SOURCES_PATH
from mulligan.release.hub import HubCheckpoint, parse_hf_uri
from mulligan.real.train.hub_data import (
    DEFAULT_DATASET_REVISION,
    parse_dataset_pins,
    resolve_encoder_source,
    resolve_selected_episodes,
)

PIN = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def harness_environ():
    """The trainer harnesses set CPU-only/threading env vars at import; keep them out of later tests."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_both_parsers_take_repeatable_pins():
    argv = [
        "--repo-ids",
        "mulligan/a,mulligan/b",
        "--dataset-revisions",
        f"mulligan/a={PIN}",
        "mulligan/b=v3.0",
        "--dataset-episodes",
        "mulligan/a=session_id:b03",
    ]
    dp = policy.build_parser().parse_args(argv)
    iql = iql_args.parse_args([*argv, "--encoder-artifact", "hf://mulligan/x-dp"])
    for args in (dp, iql):
        assert args.dataset_revisions == [f"mulligan/a={PIN}", "mulligan/b=v3.0"]
        assert args.dataset_episodes == ["mulligan/a=session_id:b03"]
    assert policy.build_parser().parse_args(["--repo-ids", "x/y"]).dataset_revisions is None


def test_selector_flags_are_repeatable_in_both_trainers():
    argv = [
        "--repo-ids",
        "mulligan/a,mulligan/b",
        "--dataset-revisions",
        f"mulligan/a={PIN}",
        "--dataset-episodes",
        "mulligan/a=episode_index:0-4",
        "--dataset-revisions",
        f"mulligan/b={PIN}",
        "--dataset-episodes",
        "mulligan/b=session_id:b03",
    ]
    dp = policy.build_parser().parse_args(argv)
    iql = iql_args.parse_args([*argv, "--encoder-artifact", "hf://mulligan/x-dp"])
    for args in (dp, iql):
        revisions, selectors = parse_dataset_pins(
            args.dataset_revisions, args.dataset_episodes, ["mulligan/a", "mulligan/b"]
        )
        assert revisions == {"mulligan/a": PIN, "mulligan/b": PIN}
        assert selectors["mulligan/a"].values == (0, 1, 2, 3, 4)
        assert selectors["mulligan/b"].values == ("b03",)


def test_launcher_overrides_selectors_per_repo():
    from mulligan.real.train.launch import config_to_argv

    config = {
        "args": {"repo-ids": ["mulligan/a", "mulligan/b"]},
        "dataset_revisions": {"mulligan/a": PIN, "mulligan/b": PIN},
        "dataset_episodes": {"mulligan/a": {"episode_index": [0, 1]}},
    }
    argv = config_to_argv(
        config,
        [
            "--output-dir",
            "out",
            "--dataset-episodes",
            "mulligan/a=episode_index:5",
            "--dataset-episodes=mulligan/b=session_id:b01",
            "--seed",
            "3",
        ],
    )
    args = policy.build_parser().parse_args(argv)
    _, selectors = parse_dataset_pins(
        args.dataset_revisions, args.dataset_episodes, ["mulligan/a", "mulligan/b"]
    )
    assert selectors["mulligan/a"].values == (5,)
    assert selectors["mulligan/b"].values == ("b01",)
    assert args.seed == 3 and args.output_dir == "out"
    with pytest.raises(ValueError, match="KEY=VALUE"):
        config_to_argv(config, ["--dataset-revisions", "nope"])


def test_iql_parse_args_applies_recipe_but_build_parser_does_not(monkeypatch):
    import dataclasses

    from mulligan.real.train import iql_recipes

    probe = dataclasses.replace(iql_recipes.DIVL, name="probe", tau_min=0.25)
    monkeypatch.setitem(iql_recipes.RECIPES, "probe", probe)
    argv = ["--repo-ids", "x/y", "--encoder-artifact", "d", "--iql-recipe", "probe"]
    assert iql_args.parse_args(argv).tau_min == 0.25
    assert iql_args.parse_args([*argv, "--tau-min", "0.4"]).tau_min == 0.4
    assert iql_args.build_parser().parse_args(argv).tau_min == 0.5


def test_pins_are_validated_against_the_run_repos():
    revisions, selectors = parse_dataset_pins(
        [f"a/b={PIN}"], ["a/b=episode_index:0-2"], ["a/b", "c/d"]
    )
    assert revisions == {"a/b": PIN}
    assert selectors["a/b"].values == (0, 1, 2)
    with pytest.raises(ValueError, match="not in the run's repo ids"):
        parse_dataset_pins(["x/y=1"], None, ["a/b"])
    with pytest.raises(ValueError, match="needs a pinned --dataset-revisions"):
        parse_dataset_pins(None, ["a/b=episode_index:0"], ["a/b"])


def _write_fake_dataset(root: Path, repo_id: str, total_episodes: int, sessions: list[str]):
    repo_dir = root / repo_id
    (repo_dir / "meta").mkdir(parents=True)
    pd.DataFrame({"episode_index": list(range(len(sessions))), "session_id": sessions}).to_parquet(
        repo_dir / EPISODE_SOURCES_PATH
    )
    return repo_dir


def test_session_selector_resolves_from_the_local_provenance(tmp_path, monkeypatch):
    _write_fake_dataset(tmp_path, "m/r", 5, ["b01", "b01", "b03", "b02", "b03"])

    class _Meta:
        def __init__(self, repo_id, root=None, **_):
            self.total_episodes = 5

    monkeypatch.setattr("mulligan.real.train.hub_data.LeRobotDatasetMetadata", _Meta)
    revisions, selectors = parse_dataset_pins(
        [f"m/r={PIN}"], ["m/r=session_id:b03"], ["m/r", "m/other"]
    )
    selected = resolve_selected_episodes(["m/r", "m/other"], tmp_path, revisions, selectors)
    assert selected == {"m/r": [2, 4], "m/other": None}

    _, bad = parse_dataset_pins([f"m/r={PIN}"], ["m/r=episode_index:3-7"], ["m/r"])
    with pytest.raises(ValueError, match="outside"):
        resolve_selected_episodes(["m/r"], tmp_path, revisions, bad)


def test_encoder_source_resolution(tmp_path):
    assert parse_hf_uri("hf://mulligan/x-dp") == HubCheckpoint("mulligan/x-dp", None, None)
    assert parse_hf_uri(f"hf://mulligan/x-dp@{PIN}") == HubCheckpoint("mulligan/x-dp", PIN, None)
    with pytest.raises(ValueError):
        parse_hf_uri("hf://x-dp")
    local, recorded = resolve_encoder_source(str(tmp_path))
    assert local == tmp_path.resolve() and recorded == str(tmp_path.resolve())


def test_missing_local_encoder_path_is_not_sent_to_wandb(tmp_path, monkeypatch):
    import sys
    import types

    def _no_wandb():
        raise AssertionError("a local path must not reach the W&B API")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=_no_wandb))
    typo = tmp_path / "outputs" / "marker_dp" / "diffusion_x" / "finel"
    for spec in (str(typo), "outputs/marker_dp/finel", "checkpoint_dir"):
        with pytest.raises(FileNotFoundError, match="no such local directory"):
            resolve_encoder_source(spec)


def test_wandb_encoder_source_is_recorded_as_a_model_id(tmp_path, monkeypatch):
    import sys
    import types

    requested = []

    class _Artifact:
        def download(self):
            return str(tmp_path)

    class _Api:
        def artifact(self, name):
            requested.append(name)
            return _Artifact()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=_Api))
    for spec in ("ent/proj/dp-final:v0", "wandb://ent/proj/dp-final:v0"):
        local, recorded = resolve_encoder_source(spec)
        assert local == tmp_path and recorded == "wandb://ent/proj/dp-final:v0"
    assert requested == ["ent/proj/dp-final:v0"] * 2


def test_cache_revision_pins_are_enforced():
    critic.validate_cache_revisions({"a/b": PIN}, {"a/b": PIN}, label="cache")
    critic.validate_cache_revisions({"a/b": PIN}, {}, label="cache")
    with pytest.raises(ValueError, match="was built from"):
        critic.validate_cache_revisions({"a/b": "f" * 40}, {"a/b": PIN}, label="cache")


def test_positional_episode_ids_for_episode_subsets():
    class _Sub:
        def __init__(self, rows, selected=None):
            self.hf_dataset = {"episode_index": rows}
            self.features = {"episode_index": None}
            self.episodes = selected

        def __len__(self):
            return len(self.hf_dataset["episode_index"])

    # A whole repo keeps episode_index + offset; a subset (episodes 1 and 3) stays dense.
    whole = _Sub([0, 0, 1, 1, 1])
    subset = _Sub([1, 1, 3, 3], selected=[1, 3])
    episode_ids, dataset_ids, _, from_idx, _ = critic.build_multidataset_frame_metadata(
        [whole, subset], ["a/b", "c/d"]
    )
    assert episode_ids.tolist() == [0, 0, 1, 1, 1, 2, 2, 3, 3]
    assert dataset_ids.tolist() == [0] * 5 + [1] * 4
    assert from_idx == [0, 2, 5, 7]

    # A whole repo whose episode_index skips a value would get positional ids that differ
    # from its episode_index: refused.
    with pytest.raises(ValueError, match="not 0..N-1 in row order"):
        critic.build_multidataset_frame_metadata([_Sub([0, 0, 1, 3, 3])], ["a/b"])


@pytest.mark.slow
def test_critic_frame_lookups_follow_the_absolute_index_of_an_episode_subset(
    tmp_path, harness_environ
):
    """--dataset-episodes loads a subset whose batch['index'] stays absolute; the intervention
    targets must still read the lookup row of that frame."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from mulligan.real.train.iql_eval import loaded_frame_index, local_frame_index
    from tests.real import trainer_harness_iql as harness

    with harness._patched_env():
        harness.ensure_fixture(tmp_path)
        root = tmp_path / harness.REPO_ID
        full = LeRobotDataset(harness.REPO_ID, root=root)
        subset = LeRobotDataset(harness.REPO_ID, root=root, episodes=[2, 3])
        rows = list(range(len(subset)))
        abs_idx = torch.tensor([int(subset[i]["index"]) for i in rows])
    assert len(subset) == 2 * harness.EP_LEN
    assert abs_idx.min() >= 2 * harness.EP_LEN

    def lookups(ds):
        frames = critic.iql_classify_frames(
            [harness.REPO_ID], [ds], n_global_frames=len(ds), require_intervention=True
        )
        mc = critic.discounted_mc_returns_by_dataset(
            [ds],
            [harness.REPO_ID],
            gamma=0.9,
            reward_shift=0.0,
            intervention_negative_reward=-1.0,
            intervention_values_by_dataset=frames.intervention_values_by_dataset,
        )
        return frames.intervention_values_by_dataset, mc

    full_interv, full_mc = lookups(full)
    sub_interv, _ = lookups(subset)
    anchored = torch.isfinite(full_mc[0][abs_idx])  # the invalid suffix has no MC target
    rows = [r for r, keep in zip(rows, anchored.tolist(), strict=True) if keep]
    abs_idx = abs_idx[anchored]
    assert full_interv[0][abs_idx].any() and not full_interv[0][abs_idx].all()

    batch = {
        "index": abs_idx,
        "dataset_index": torch.zeros(len(rows), dtype=torch.long),
        "reward": torch.zeros(len(rows), 1),
    }
    batch["index"] = local_frame_index(
        batch["index"], batch["dataset_index"], loaded_frame_index([subset])
    )
    assert batch["index"].tolist() == rows
    critic.apply_intervention_reward_shaping(
        batch,
        intervention_by_frame=torch.zeros(1),
        intervention_by_dataset=sub_interv,
        intervention_negative_reward=-1.0,
        reward_shift=0.0,
        horizon=1,
    )
    assert batch["reward"][:, 0].tolist() == (-full_interv[0][abs_idx].float()).tolist()
    with pytest.raises(IndexError, match="not loaded rows"):
        local_frame_index(torch.tensor([0]), torch.tensor([0]), loaded_frame_index([subset]))


@pytest.mark.slow
def test_dp_trainer_reads_only_selected_episodes(tmp_path, harness_environ):
    from tests.real import trainer_harness_dp as harness

    data_root = tmp_path / "data"
    harness.ensure_fixture(data_root)
    out_dir = tmp_path / "out"
    harness.run_dp_trace(
        tmp_path / "trace.json",
        data_root=data_root,
        steps=2,
        output_dir=out_dir,
        extra_argv=[
            "--dataset-revisions",
            f"{harness.REPO_ID}={PIN}",
            "--dataset-episodes",
            f"{harness.REPO_ID}=episode_index:0,2",
        ],
    )
    (final,) = out_dir.glob("diffusion_*/final")
    meta = json.loads((final / policy.TRAIN_METADATA_FILENAME).read_text())
    assert meta["dataset_revisions"] == {harness.REPO_ID: PIN}
    # The harness reads its local fixture with --no-dataset-sync: no Hub commit to record.
    assert meta["dataset_commits"] == {harness.REPO_ID: None}
    assert meta["dataset_episodes"] == {harness.REPO_ID: {"episode_index": [0, 2]}}
    dataset_cfg = meta["run_config"]["dataset"]
    assert dataset_cfg[harness.REPO_ID.replace("/", "_")]["used_episodes"] == 2
    assert dataset_cfg["train_episodes"] == [0, 1]
    assert dataset_cfg["training_frames"] < harness.N_EPISODES * harness.EP_LEN * 2 / 3


@pytest.mark.slow
def test_dp_selectors_on_several_repos_eval_repos_and_filter_dagger(
    tmp_path, harness_environ, monkeypatch
):
    """Selectors apply per repo (one flag per repo), to eval-only repos, and before
    --filter-dagger (which then filters inside the selection)."""
    from tests.real import trainer_harness_dp as harness
    from tests.real.tiny_real_dataset import build_tiny_real_dataset

    data_root = tmp_path / "data"
    harness.ensure_fixture(data_root)
    second, held_out = "tiny/real-tiny-b", "tiny/real-tiny-eval"
    for seed, repo in ((1, second), (2, held_out)):
        build_tiny_real_dataset(
            data_root,
            repo_id=repo,
            n_episodes=4,
            ep_len=harness.EP_LEN,
            cameras=harness.CAMERAS,
            seed=seed,
        )
    import sys

    import torch

    out_dir = tmp_path / "out"
    # The trace harness counts one forward per step; validation adds forwards, so call
    # the trainer directly on the harness argv.
    argv = harness.dp_argv(
        data_root=data_root,
        output_dir=out_dir,
        steps=2,
        seed=0,
        extra_argv=[
            "--repo-ids",
            f"{harness.REPO_ID},{second}",
            "--eval-repo-ids",
            held_out,
            "--eval-freq",
            "2",
            "--num-eval-batches",
            "1",
            "--filter-dagger",
            "--dataset-revisions",
            f"{harness.REPO_ID}={PIN}",
            f"{second}={PIN}",
            f"{held_out}={PIN}",
            "--dataset-episodes",
            f"{harness.REPO_ID}=episode_index:0,1",
            "--dataset-episodes",
            f"{second}=episode_index:1-3",
            "--dataset-episodes",
            f"{held_out}=episode_index:2,3",
        ],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["mulligan.real.train.policy", *argv])
    policy.train(policy.parse_args())
    (final,) = out_dir.glob("diffusion_*/final")
    meta = json.loads((final / policy.TRAIN_METADATA_FILENAME).read_text())
    assert meta["dataset_episodes"] == {
        harness.REPO_ID: {"episode_index": [0, 1]},
        second: {"episode_index": [1, 2, 3]},
        held_out: {"episode_index": [2, 3]},
    }
    dataset_cfg = meta["run_config"]["dataset"]
    # Successful episodes of the fixtures are the even ones; --filter-dagger keeps those inside
    # each selection: {0, 1} -> {0} and {1, 2, 3} -> {2} (2 each without the selectors).
    assert dataset_cfg["tiny_real-tiny"]["used_episodes"] == 1
    assert dataset_cfg["tiny_real-tiny-b"]["used_episodes"] == 1
    assert dataset_cfg["train_episodes"] == [0, 1]
    # Only the two selected eval episodes are read (all four would give more frames).
    assert 0 < dataset_cfg["val_frames"] <= 2 * harness.EP_LEN


@pytest.mark.slow
def test_critic_trainer_reads_only_selected_episodes(tmp_path, harness_environ):
    from tests.real import trainer_harness_iql as harness

    data_root = tmp_path / "data"
    harness.run_iql_trace(
        tmp_path / "trace.json",
        data_root=data_root,
        steps=4,  # the harness' critic LR warmup is 3 steps
        extra_argv=[
            "--dataset-revisions",
            f"{harness.REPO_ID}={PIN}",
            "--dataset-episodes",
            f"{harness.REPO_ID}=episode_index:2-3",
            # The intervention lookup is read at every replay refresh; the subset's
            # absolute frame indices (40..79) are all past its 40 loaded rows.
            "--intervention-negative-reward",
            "-1.0",
        ],
    )
    package = data_root / "_trainer_outputs" / "checkpoints" / "final"
    assert (package / "iql_checkpoint.pt").is_file()
    meta = json.loads((package / "metadata.json").read_text())
    data_cfg = meta["training_config"]["data"]
    assert data_cfg["n_episodes"] == 2
    assert data_cfg["dataset_revisions"] == {harness.REPO_ID: PIN}
    assert data_cfg["dataset_episodes"] == {harness.REPO_ID: {"episode_index": [2, 3]}}
    assert meta["dataset_episodes"] == data_cfg["dataset_episodes"]
    assert meta["dp_artifact"] == str(Path(meta["dp_artifact"]).resolve())


@pytest.mark.slow
def test_critic_auto_resumes_without_wandb(tmp_path, harness_environ, monkeypatch):
    """Auto-resume is keyed on the output dir and run name, not on W&B."""
    from tests.real import trainer_harness_iql as harness

    build_argv = harness.build_argv
    monkeypatch.setattr(
        harness,
        "build_argv",
        lambda **kw: [a for a in build_argv(**kw) if a != "--no-auto-resume"],
    )
    data_root = tmp_path / "data"
    out_dir = tmp_path / "run"  # the harness wipes its own output dir before each run
    extra = ["--resume-checkpoint-freq", "2", "--output-dir", str(out_dir)]
    harness.run_iql_trace(tmp_path / "trace.json", data_root=data_root, steps=4, extra_argv=extra)
    (meta_path,) = (out_dir / "_resume").glob("*/resume_meta.json")
    assert json.loads(meta_path.read_text())["completed"] is True
    with pytest.raises(RuntimeError, match="already marked completed"):
        harness.run_iql_trace(
            tmp_path / "trace2.json", data_root=data_root, steps=4, extra_argv=extra
        )


def test_no_dataset_sync_keeps_local_copies(tmp_path, monkeypatch):
    from mulligan.real.train import hub_data

    repo_dir = tmp_path / "m" / "local"
    (repo_dir / "meta").mkdir(parents=True)
    (repo_dir / "meta" / "info.json").write_text("{}")
    (repo_dir / "data").mkdir()
    shard = repo_dir / "data" / "chunk-000.parquet"
    shard.write_text("rows")

    def no_sync(*_args, **_kwargs):
        raise AssertionError("--no-dataset-sync must not sync")

    monkeypatch.setattr(hub_data, "sync_datasets", no_sync)
    hub_data.prepare_datasets(["m/local"], tmp_path, {}, sync=False)
    assert shard.read_text() == "rows"
    with pytest.raises(FileNotFoundError, match="--no-dataset-sync"):
        hub_data.prepare_datasets(["m/missing"], tmp_path, {}, sync=False)
    assert hub_data.dataset_commits(["m/local"], {}, synced=False) == {"m/local": None}
    monkeypatch.setattr(hub_data, "resolve_dataset_commit", lambda repo, rev: f"{repo}@{rev}")
    assert hub_data.dataset_commits(["m/local"], {"m/local": "v9"}, synced=True) == {
        "m/local": "m/local@v9"
    }
    argv = ["--repo-ids", "m/local", "--no-dataset-sync"]
    assert policy.build_parser().parse_args(argv).no_dataset_sync is True
    assert iql_args.parse_args([*argv, "--encoder-artifact", "x"]).no_dataset_sync is True


def test_multi_dataset_sub_datasets_load_at_pinned_revisions(monkeypatch):
    import lerobot.datasets.multi_dataset as lerobot_multi_dataset

    from mulligan.real.train import hub_data

    calls = []

    def fake(repo_id, *args, **kwargs):
        calls.append((repo_id, kwargs["revision"]))

    monkeypatch.setattr(lerobot_multi_dataset, "LeRobotDataset", fake)
    with hub_data._pinned_subdatasets({"a/b": PIN}):
        lerobot_multi_dataset.LeRobotDataset("a/b", root="x")
        lerobot_multi_dataset.LeRobotDataset("c/d", root="y")
    assert calls == [("a/b", PIN), ("c/d", DEFAULT_DATASET_REVISION)]
    assert lerobot_multi_dataset.LeRobotDataset is fake


def test_unpinned_repos_default_to_the_codebase_tag():
    assert DEFAULT_DATASET_REVISION == "v3.0"
