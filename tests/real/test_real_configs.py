"""The real trainer and evaluation configs: one per released checkpoint, parseable, pinned."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from mulligan.real.train.launch import config_to_argv, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs" / "real"
TRAINER_CONFIGS = sorted(
    p
    for p in CONFIG_ROOT.glob("*/**/*.yaml")
    if not p.name.endswith(("_eval.yaml", "_sampler.yaml"))
)
EVAL_CONFIGS = sorted(CONFIG_ROOT.glob("*/*_eval.yaml"))
VIEWS = {
    c["repo"]: c
    for c in json.loads((REPO_ROOT / "release" / "training-views.json").read_text())["checkpoints"]
}
MODELS = {
    m["repo"]: m for m in json.loads((REPO_ROOT / "release" / "models.json").read_text())["models"]
}
TRAINER_KEYS = {
    "kind",
    "trainer",
    "checkpoint",
    "retrain",
    "paper",
    "dataset_revisions",
    "dataset_episodes",
    "args",
}


def _id(path: Path) -> str:
    return str(path.relative_to(CONFIG_ROOT))


def _parse(kind: str, argv: list[str]):
    """The trainer's own parse (the critic's applies ``--iql-recipe`` on top of argparse)."""
    if kind == "dp-actor":
        from mulligan.real.train.policy import parse_args
    else:
        from mulligan.real.train.iql_args import parse_args
    return parse_args(argv)


def test_one_config_per_released_real_checkpoint():
    repos = Counter(load_config(p)["checkpoint"]["repo"] for p in TRAINER_CONFIGS)
    assert set(repos) == set(VIEWS)
    assert max(repos.values()) == 1


@pytest.mark.parametrize("path", TRAINER_CONFIGS, ids=_id)
def test_config_keys(path):
    config = load_config(path)
    extra = {"camera_crops"} if config["kind"] == "dp-actor" else set()
    assert set(config) == TRAINER_KEYS | extra


@pytest.mark.parametrize("path", TRAINER_CONFIGS, ids=_id)
def test_config_parses_with_its_trainer(path):
    config = load_config(path)
    args = _parse(config["kind"], config_to_argv(config, ["--output-dir", "outputs/config-check"]))
    assert args.seed == config["args"]["seed"]
    assert args.training_steps >= config["checkpoint"]["step"]


@pytest.mark.parametrize("path", TRAINER_CONFIGS, ids=_id)
def test_config_sets_each_boolean_flag_once(path):
    config = load_config(path)
    for key in config["args"]:
        if key.startswith("no-"):
            assert key[3:] not in config["args"], f"both --{key[3:]} and --{key}"


@pytest.mark.parametrize("path", TRAINER_CONFIGS, ids=_id)
def test_config_pins_match_training_views(path):
    config = load_config(path)
    view = VIEWS[config["checkpoint"]["repo"]]
    assert config["retrain"] == view["retrain"]
    assert config["dataset_revisions"] == {s["repo"]: s["revision"] for s in view["sources"]}
    assert config["dataset_episodes"] == {
        s["repo"]: s["selector"] for s in view["sources"] if s["selector"] != "all"
    }
    model = MODELS[config["checkpoint"]["repo"]]
    assert config["checkpoint"]["revision"] == model["revision"]
    assert [c["step"] for c in model["checkpoints"]] == [config["checkpoint"]["step"]]
    argv = config_to_argv(config)
    assert "--dataset-revisions" in argv


@pytest.mark.parametrize("path", TRAINER_CONFIGS, ids=_id)
def test_dp_configs_pin_their_crops(path):
    """A DP retrain applies the config's crops, whatever the local station config says."""
    from mulligan.real.policy.side_crop import parse_side_crop
    from mulligan.real.train.policy import build_parser

    config = load_config(path)
    if config["kind"] != "dp-actor":
        assert "camera_crops" not in config
        return
    args = build_parser().parse_args(config_to_argv(config, ["--output-dir", "x"]))
    assert {cam: list(box) for cam, box in parse_side_crop(args.side_crop).items()} == config[
        "camera_crops"
    ]
    override = config_to_argv(config, ["--side-crop", "wrist_left=0,0,10,10"])
    args = build_parser().parse_args(override)
    assert parse_side_crop(args.side_crop)["wrist_left"] == (0, 0, 10, 10)


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=_id)
def test_eval_configs_reference_flattened_configs(path):
    config = yaml.safe_load(path.read_text())
    for session in config["sessions"]:
        for arm in session["arms"]:
            for role in ("actor", "critic"):
                if role not in arm:
                    continue
                ref = load_config(REPO_ROOT / arm[role]["config"])
                assert ref["checkpoint"]["repo"] == arm[role]["repo"]
                assert ref["checkpoint"]["revision"] == arm[role]["revision"]
                assert ref["checkpoint"]["step"] == arm[role]["step"]
            if "critic" in arm:
                assert arm["num_action_samples"] in (16, 32)


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=_id)
def test_eval_session_manifests_are_located(path):
    """Every session names its start manifest: a shipped one (sha256 checked here) or a
    subset derived from one, stored in the eval dataset (checked by the network test)."""
    import hashlib

    config = yaml.safe_load(path.read_text())
    for session in config["sessions"]:
        if "manifest" in session:
            data = (REPO_ROOT / session["manifest"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == session["manifest_sha256"]
        else:
            rel = session["manifest_in_eval_dataset"]
            assert rel == f"meta/sessions/{session['session_id']}/meta/initial_states_manifest.json"


@pytest.mark.network
def test_derived_session_manifests_match_the_eval_dataset():
    import hashlib

    from huggingface_hub import hf_hub_download

    checked = 0
    for path in EVAL_CONFIGS:
        config = yaml.safe_load(path.read_text())
        repo, rev = config["eval_dataset"]["repo"], config["eval_dataset"]["revision"]
        for session in config["sessions"]:
            if "manifest_in_eval_dataset" not in session:
                continue
            local = hf_hub_download(
                repo, session["manifest_in_eval_dataset"], repo_type="dataset", revision=rev
            )
            digest = hashlib.sha256(Path(local).read_bytes()).hexdigest()
            assert digest == session["manifest_sha256"]
            checked += 1
    assert checked == 3


# Best-of-N sample count of the critic arm the paper deployed, per task and round.
EXPECTED_N = {
    ("marker_d2", 2): 16,
    ("marker_d2", 3): 16,
    ("marker_d2", 4): 16,
    ("marker_d2", 5): 32,
    ("square_d2", 3): 16,
    ("square_d2", 4): 16,
    ("square_d2", 5): 32,
    ("routing_d2", 3): 32,
    ("routing_d2", 4): 32,
    ("routing_d2", 5): 32,
}


def test_eval_n_per_round():
    found = {}
    for path in EVAL_CONFIGS:
        config = yaml.safe_load(path.read_text())
        ns = {
            arm["num_action_samples"]
            for s in config["sessions"]
            for arm in s["arms"]
            if "critic" in arm
        }
        if ns:
            assert len(ns) == 1, path
            found[config["task"], config["round"]] = ns.pop()
    assert found == EXPECTED_N


def test_launcher_appends_overrides_after_double_dash(capsys):
    from mulligan.real.train.launch import main

    path = CONFIG_ROOT / "marker_d2" / "r05_mulligan_dp.yaml"
    main([str(path), "--output-dir", "out", "--print-argv", "--", "--training-steps", "300"])
    argv = capsys.readouterr().out.split()
    assert argv[-4:] == ["--output-dir", "out", "--training-steps", "300"]
    assert argv[argv.index("--dataset-revisions") + 1].startswith("mulligan/")


PAPER_KEYS = ["task", "round", "arm", "seed", "results", "evaluations", "collections"]


@pytest.mark.parametrize("path", TRAINER_CONFIGS, ids=_id)
def test_config_says_where_the_paper_uses_it(path):
    """``paper`` names the task, round, arm and seed, and every evaluation or collection of the
    checkpoint; a checkpoint the paper neither evaluated nor collected with does not exist."""
    config = load_config(path)
    paper = config["paper"]
    assert list(paper) == PAPER_KEYS
    assert paper["seed"] == config["args"]["seed"]
    assert paper["evaluations"] or paper["collections"]
    for result in paper["results"]:
        assert result in ("Fig. 5", "Fig. 7", "App. B.1", "App. B.6")


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=_id)
def test_eval_arms_appear_in_the_trainer_configs(path):
    config = yaml.safe_load(path.read_text())
    repo = config["eval_dataset"]["repo"]
    for session in config["sessions"]:
        for arm in session["arms"]:
            actor = load_config(REPO_ROOT / arm["actor"]["config"])["paper"]["evaluations"]
            if "critic" in arm:
                n = arm["num_action_samples"]
                assert f"{repo} {session['session_id']} ({arm['method']} actor, N={n})" in actor
                critic = load_config(REPO_ROOT / arm["critic"]["config"])["paper"]["evaluations"]
                assert f"{repo} {session['session_id']} ({arm['method']}, N={n})" in critic
            else:
                assert f"{repo} {session['session_id']} ({arm['method']})" in actor
