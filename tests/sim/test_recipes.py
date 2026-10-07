"""Checks of configs/sim/recipes.json and the helpers that turn it into command lines."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from mulligan.sim import recipes as R

ROOT = Path(__file__).resolve().parents[2]

RECIPES = R.load(ROOT / "configs" / "sim" / "recipes.json")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HF_REF = re.compile(r"^hf://(mulligan/sim-[a-z0-9-]+)@([0-9a-f]{40})/seed-([1-5])$")
SCRIPTS = [
    "scripts/sim/collect_round.sh",
    "scripts/sim/split_round.sh",
    "scripts/sim/train_cell.sh",
    "scripts/sim/train_divl_heads.sh",
    "scripts/sim/eval_cell.sh",
    "scripts/examples/slurm_train.sbatch",
]

# 34 human-in-the-loop cells and 12 autonomous IQL cells carry an IDQL agent and a DIVL
# head (the 46 cells of the DIVL campaign); the 12 N=1 autonomous cells only an agent.
# Six RLPD and two HiL-SERL recipes train the baselines.
N_IDQL = 58
N_DIVL = 46
N_RLPD = 6
N_HILSERL = 2
IDQL_FAMILIES = {"hil", "autonomous_baseline"}


def _idql_recipes() -> list[dict]:
    return [r for r in RECIPES["recipes"] if r["family"] in IDQL_FAMILIES]


def _recipe(recipe_id: str) -> dict:
    return R.get_recipe(RECIPES, recipe_id)


def _walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, (*path, key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, (*path, i))
    else:
        yield path, node


def test_schema():
    assert RECIPES["schema_version"] == 1
    assert RECIPES["trainer"] == "mulligan.training.train"
    assert set(RECIPES["tasks"]) == {"square_narrow", "square_broad"}
    for grid in RECIPES["grids"].values():
        assert grid["task"] in RECIPES["tasks"]
        assert re.fullmatch(r"[0-9a-f]{64}", grid["manifest_hash"])
    stage_keys = {"kind", "args", "model", "checkpoints", "training_data"}
    for r in RECIPES["recipes"]:
        assert r["status"] == "ok", (r["id"], r["status"])
        assert r["task"] in RECIPES["tasks"]
        assert r["round"] in {0, 1, 2, 3}
        assert r["family"] in R.FAMILIES
        if r["family"] not in IDQL_FAMILIES:
            continue
        assert r["seeds"] == [1, 2, 3, 4, 5]
        assert r["eval"]["grid"] in RECIPES["grids"]
        assert RECIPES["grids"][r["eval"]["grid"]]["task"] == r["task"]
        assert r["eval"]["num_action_samples"] in {1, 32}
        want = "divl_head" if r["family"] == "hil" else "idql_agent"
        assert r["eval"]["stage"] == want and want in r
        assert r["id"].startswith(r["task"].replace("_", "-") + f"-r{r['round']:02d}-")
        assert r["datasets"], r["id"]
        for stage in R.STAGES:
            if stage not in r:
                continue
            spec = r[stage]
            assert stage_keys <= set(spec), (r["id"], stage)
            assert spec["kind"] == stage
            assert spec["training_data"] in {"exact", "differs"}
            assert [c["seed"] for c in spec["checkpoints"]] == r["seeds"]
            assert all(c["path"] == f"seed-{c['seed']}" for c in spec["checkpoints"])
            assert all(a.startswith("--") and "=" in a for a in spec["args"]), (r["id"], stage)
            flags = [a.split("=", 1)[0] for a in spec["args"]]
            assert len(flags) == len(set(flags)), (r["id"], stage)
            forbidden = {
                "--dataset.repo_ids",
                "--dataset.revisions",
                "--training.seed",
                "--pretrained_artifact",
                "--dataset.root",
                "--system.checkpoint_dir",
            }
            assert not forbidden & set(flags), (r["id"], stage)
            assert not any(f.startswith("--wandb.") for f in flags), (r["id"], stage)
            for seed, extra in spec.get("seed_args", {}).items():
                assert int(seed) in r["seeds"]
                assert not {a.split("=", 1)[0] for a in extra} & set(flags)


def test_counts_and_unique_ids():
    ids = [r["id"] for r in RECIPES["recipes"]]
    assert len(ids) == len(set(ids)) == N_IDQL + N_RLPD + N_HILSERL
    assert len(_idql_recipes()) == N_IDQL
    families = [r["family"] for r in RECIPES["recipes"]]
    assert (families.count("rlpd"), families.count("hilserl")) == (N_RLPD, N_HILSERL)
    assert sum("divl_head" in r for r in RECIPES["recipes"]) == N_DIVL
    keys = [r["source_cell_key"] for r in RECIPES["recipes"] if "source_cell_key" in r]
    assert len(keys) == len(set(keys)) == N_DIVL
    n1 = [r for r in _idql_recipes() if r["eval"]["num_action_samples"] == 1]
    assert {r["method"] for r in n1} == {"auto-filtered-bc-n1", "auto-plain-il-n1"}
    assert len(n1) == 12 and not any("divl_head" in r for r in n1)
    repos = [r[s]["model"]["repo"] for r in _idql_recipes() for s in R.STAGES if s in r]
    assert len(repos) == len(set(repos)) == 104


def test_every_ref_is_a_pinned_mulligan_repo():
    for r in RECIPES["recipes"]:
        for d in r["datasets"]:
            assert d["repo"].startswith(f"mulligan/sim-{r['task'].replace('_', '-')}-c"), d
            assert HEX40.match(d["revision"]), d
        for stage in R.STAGES:
            if stage in r and "model" in r[stage]:
                model = r[stage]["model"]
                assert model["repo"] == f"mulligan/sim-{r['id']}-{stage.split('_')[0]}"
                assert HEX40.match(model["revision"])
                for ck in r[stage]["checkpoints"]:
                    assert ck["checkpoint"] == f"{model['repo']}/{ck['path']}", ck
    for path, value in _walk(RECIPES):
        if isinstance(value, str) and value.startswith("hf://"):
            assert HF_REF.match(value), (path, value)
        if isinstance(value, str) and value.startswith("mulligan/") and " " not in value:
            assert re.fullmatch(r"mulligan/sim-[a-z0-9-]+(/seed-[1-5])?", value), (path, value)


def test_divl_parents_resolve_to_released_agents():
    released = {
        r["idql_agent"]["model"]["repo"]: r["idql_agent"]["model"]["revision"]
        for r in _idql_recipes()
    }
    for r in _idql_recipes():
        if "divl_head" not in r:
            continue
        agent = r["idql_agent"]["model"]
        for ck in r["divl_head"]["checkpoints"]:
            repo, revision, seed = HF_REF.match(ck["parent"]).groups()
            assert released.get(repo) == revision
            assert (repo, revision, int(seed)) == (agent["repo"], agent["revision"], ck["seed"])
            assert "parent_checkpoint" not in ck  # `parent` pins it


def test_divl_heads_follow_the_locked_protocol():
    locked = {
        "--policy.type=idql_divl",
        "--training.update_components=critic_value_only",
        "--policy.num_atoms=101",
        "--policy.hl_gauss_sigma_ratio=0.75",
        "--policy.tau_base=0.7",
        "--policy.tau_min=0.5",
        "--policy.tau_max=0.95",
        "--policy.tau_entropy_alpha=0.4",
        "--training.batch_size=1024",
    }
    for r in _idql_recipes():
        if "divl_head" in r:
            args = set(r["divl_head"]["args"])
            assert locked <= args, r["id"]
            steps = RECIPES["tasks"][r["task"]]["training_steps"]
            assert f"--training.training_steps={steps}" in args
        agent = set(r["idql_agent"]["args"])
        assert "--policy.type=idql" in agent
        assert f"--env.name={RECIPES['tasks'][r['task']]['env_name']}" in agent


def test_rounds_produce_every_training_dataset():
    used = {d["repo"] for r in RECIPES["recipes"] for d in r["datasets"]}
    produced = set()
    for entry in RECIPES["rounds"]:
        for split in entry["splits"]:
            produced |= set(split["targets"].values())
            assert HEX40.match(split["input_revision"])
        produced |= {ro["dataset"] for ro in entry.get("rollouts", [])}
    produced |= {
        r["autonomous_collection"]["dataset"]
        for r in RECIPES["recipes"]
        if "autonomous_collection" in r
    }
    assert used <= produced
    # Narrow R2 also rolled out the R1 Sobol with-CF agent (start design only; no R2 Sobol arm).
    assert produced - used == {"mulligan/sim-square-narrow-c02-sobol-policy-rollouts"}


def test_round_inputs_exist():
    for entry in RECIPES["rounds"]:
        paths = [s["manifest"] for s in entry["splits"]]
        paths += [ro["starts"] for ro in entry.get("rollouts", [])]
        if "dagger" in entry:
            paths += [entry["dagger"]["starts"], entry["dagger"]["manifest"]]
        for p in paths:
            assert (ROOT / p).is_file(), p
    for r in RECIPES["recipes"]:
        if "autonomous_collection" in r:
            assert (ROOT / r["autonomous_collection"]["starts"]).is_file()


def test_train_argv_idql_agent():
    r = _recipe("square-narrow-r00-baseline")
    argv = R.train_argv(r, "idql_agent", 4, checkpoint_dir="ckpt")
    assert argv[0] == "--dataset.repo_ids=mulligan/sim-square-narrow-c00-teleop-baseline"
    revisions = json.loads(argv[1].split("=", 1)[1])
    assert revisions == {d["repo"]: d["revision"] for d in r["datasets"]}
    assert "--training.amp_dtype=bfloat16" in argv  # seed 4 ran in bf16, seeds 1-3 and 5 in fp32
    assert "--training.amp_dtype=none" in R.train_argv(r, "idql_agent", 1)
    assert argv[-2:] == ["--training.seed=4", "--system.checkpoint_dir=ckpt"]
    assert not any(a.startswith("--pretrained_artifact") for a in argv)
    with pytest.raises(ValueError):
        R.train_argv(r, "idql_agent", 6)
    with pytest.raises(ValueError):
        R.train_argv(r, "idql_agent", 1, parent="somewhere")


def test_train_argv_divl_head():
    r = _recipe("square-narrow-r01-baseline")
    argv = R.train_argv(r, "divl_head", 2)
    parent = R.released_checkpoint(r, "idql_agent", 2)
    assert argv[-1] == f"--pretrained_artifact={parent}"
    assert "--training.seed=2" in argv
    local = R.train_argv(r, "divl_head", 2, parent="/tmp/run/final_model")
    assert local[-1] == "--pretrained_artifact=/tmp/run/final_model"
    with pytest.raises(ValueError):
        R.train_argv(_recipe("square-narrow-r01-auto-plain-il-n1"), "divl_head", 1)


def test_recipe_args_parse_with_the_trainer_config():
    draccus = pytest.importorskip("draccus")
    train_config = pytest.importorskip("mulligan.configs.train")
    # A parse takes ~2 s; recipes share most flags, so parse each distinct flag set once.
    cases = {}
    for r in _idql_recipes():
        for stage in ("idql_agent", "divl_head"):
            if stage not in r:
                continue
            for seed in r["seeds"]:
                spec = r[stage]
                flags = (*spec["args"], *spec.get("seed_args", {}).get(str(seed), []))
                cases.setdefault(flags, (r, stage, seed))
    assert len(cases) < 40
    for r, stage, seed in cases.values():
        argv = R.train_argv(r, stage, seed) + ["--system.device=cpu"]
        cfg = draccus.parse(train_config.TrainConfig, args=argv)
        assert cfg.training.seed == seed
        assert cfg.dataset.revisions == {d["repo"]: d["revision"] for d in r["datasets"]}
        if stage == "divl_head":
            assert cfg.training.update_components == "critic_value_only"
            assert cfg.pretrained_artifact == R.released_checkpoint(r, "idql_agent", seed)


def test_cli_prints_nul_separated_argv():
    out = io.StringIO()
    with redirect_stdout(out):
        R.main(["train-argv", "square-broad-r02-mulligan", "--stage", "divl_head", "--seed", "3"])
    items = out.getvalue().split("\0")
    assert items[-1] == ""
    assert items[:-1] == R.train_argv(_recipe("square-broad-r02-mulligan"), "divl_head", 3)


def test_collection_and_split_argv():
    dagger = R.dagger_argv(
        RECIPES, ROOT, "square_narrow", 1, dataset_root="data", ledger="ledger.jsonl"
    )
    assert "--dataset-name=sim-square-narrow-c01-dagger-mixed" in dagger
    assert "--adaptive-protocol-quota-ledger=ledger.jsonl" in dagger
    assert not any(a.startswith("--operator") or a == "--push-to-hub" for a in dagger)
    replay = R.dagger_argv(
        RECIPES,
        ROOT,
        "square_narrow",
        1,
        dataset_root="data",
        ledger="ledger.jsonl",
        operator="replay",
    )
    assert f"--operator={R.REPLAY_OPERATOR}" in replay and "--headless" in replay
    kwargs = json.loads(replay[-1].split("=", 1)[1])
    assert kwargs["repo_id"] == "mulligan/sim-square-narrow-c01-dagger-mixed"
    assert HEX40.match(kwargs["revision"])
    routed = [a for a in dagger if a.startswith("--routed-policy=")]
    assert len(routed) == 2 and all(HF_REF.match(a.split("=", 2)[2]) for a in routed)
    with pytest.raises(ValueError):
        R.dagger_argv(RECIPES, ROOT, "square_broad", 0, dataset_root="d", ledger="l")

    module, split = R.split_argv(
        RECIPES, ROOT, "square_broad", 2, 0, dataset_root="data", output_root="out", ledger="l"
    )
    assert module == "mulligan.data.split_protocol_quota"
    assert "--source-root=data/sim-square-broad-c02-dagger-mixed" in split
    assert "--target=with_cf.mulligan=mulligan/sim-square-broad-c02-dagger-mulligan" in split
    module, split = R.split_argv(
        RECIPES, ROOT, "square_broad", 0, 1, dataset_root="data", output_root="out"
    )
    assert module == "mulligan.data.split_blind"
    assert "--max-source-episodes=200" in split

    auto = R.autonomous_argv(
        RECIPES,
        ROOT,
        _recipe("square-broad-r03-auto-plain-il-n1"),
        dataset_root="data",
        audit_output="audit.json",
    )
    prev = _recipe("square-broad-r02-auto-plain-il-n1")
    assert f"--checkpoint={R.released_checkpoint(prev, 'idql_agent', 1)}" in auto
    assert "--num-action-samples=1" in auto
    rollouts = R.rollouts_argv(
        RECIPES, ROOT, "square_broad", 1, 0, dataset_root="data", audit_output="a.json"
    )
    assert "--num-episodes=200" in rollouts


def test_collector_and_splitter_argvs_parse():
    """Every collection and split command line parses with the real CLI parsers."""
    from mulligan.data import split_blind, split_protocol_quota
    from mulligan.sim.collect import dagger, rollouts

    rollouts_parser = rollouts.build_parser()
    parsers = {
        "mulligan.data.split_blind": split_blind.build_parser(),
        "mulligan.data.split_protocol_quota": split_protocol_quota.build_parser(),
    }
    n = 0
    for entry in RECIPES["rounds"]:
        task, rnd = entry["task"], entry["round"]
        if "dagger" in entry:
            for operator in ("human", "replay"):
                args = dagger.parse_args(
                    R.dagger_argv(
                        RECIPES, ROOT, task, rnd, dataset_root="d", ledger="l", operator=operator
                    )
                )
                assert args.auto_save_on_success and args.adaptive_protocol_quota_ledger == "l"
                n += 1
        for i in range(len(entry.get("rollouts", []))):
            argv = R.rollouts_argv(RECIPES, ROOT, task, rnd, i, dataset_root="d", audit_output="a")
            assert rollouts_parser.parse_args(argv).max_steps == 400
            n += 1
        for i in range(len(entry["splits"])):
            module, argv = R.split_argv(
                RECIPES, ROOT, task, rnd, i, dataset_root="d", output_root="o", ledger="l"
            )
            args = parsers[module].parse_args(argv)
            assert args.drop_visual_features and args.target
            n += 1
    for recipe in RECIPES["recipes"]:
        if "autonomous_collection" in recipe:
            argv = R.autonomous_argv(RECIPES, ROOT, recipe, dataset_root="d", audit_output="a")
            rollouts_parser.parse_args(argv)
            n += 1
    assert n > 20


def _replay_kwargs(argv: list[str]) -> dict:
    (kwargs,) = [a.split("=", 1)[1] for a in argv if a.startswith("--operator-kwargs=")]
    return json.loads(kwargs)


def test_replay_dataset_forms(tmp_path, monkeypatch):
    def replay(dataset):
        return _replay_kwargs(
            R.dagger_argv(
                RECIPES,
                ROOT,
                "square_narrow",
                1,
                dataset_root="d",
                ledger="l",
                operator="replay",
                replay_dataset=dataset,
                replay_max_episodes=3,
            )
        )

    rev = "0123456789abcdef0123456789abcdef01234567"
    assert replay(f"someone/recording@{rev}") == {
        "repo_id": "someone/recording",
        "revision": rev,
        "max_episodes": 3,
    }
    # A bare released repo is read at its release pin.
    repo = "mulligan/sim-square-broad-c02-dagger-mixed"
    pins = json.loads((ROOT / "release/revisions.json").read_text())["repos"]
    assert replay(repo) == {"repo_id": repo, "revision": pins[repo]["revision"], "max_episodes": 3}
    # A local directory (relative to the working directory) is read in place.
    (tmp_path / "rec").mkdir()
    monkeypatch.chdir(tmp_path)
    assert replay("rec") == {"root": str(tmp_path / "rec"), "max_episodes": 3}

    for bad in ("someone/recording", "mulligan/not-released", "mulligan/sim-square@", "@abc"):
        with pytest.raises((ValueError, KeyError)):
            replay(bad)
    with pytest.raises(ValueError, match="dataset repo"):
        replay("mulligan/sim-square-narrow-r00-baseline-idql")
    with pytest.raises(ValueError, match="operator='replay'"):
        R.dagger_argv(
            RECIPES, ROOT, "square_narrow", 1, dataset_root="d", ledger="l", replay_dataset="rec"
        )


def test_collect_round_script_passes_replay_dataset(tmp_path):
    """collect_round.sh hands --replay-dataset to the recipes, then runs the collector."""
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$2" == mulligan.sim.collect.dagger ]]; then\n'
        '  shift 2; printf "%s\\n" "$@"; exit 0\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    fake.chmod(0o755)
    (tmp_path / "rec").mkdir()
    rev = "0123456789abcdef0123456789abcdef01234567"
    for dataset, expected in (
        ("rec", {"root": str(tmp_path / "rec")}),
        (f"someone/recording@{rev}", {"repo_id": "someone/recording", "revision": rev}),
    ):
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/sim/collect_round.sh"),
                "dagger",
                "--task",
                "square_narrow",
                "--round",
                "1",
                "--operator",
                "replay",
                "--replay-dataset",
                dataset,
                "--output-dir",
                str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "PYTHON": str(fake), "PYTHONPATH": str(ROOT)},
        )
        assert result.returncode == 0, result.stderr
        argv = result.stdout.splitlines()[1:]
        assert f"--operator={R.REPLAY_OPERATOR}" in argv
        assert _replay_kwargs(argv) == expected


def test_eval_cell_grid_file_gets_its_own_output_dir(tmp_path):
    """eval_cell.sh --grid-file writes under a grid-specific directory and names the grid."""
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$2" == mulligan.sim.eval.grid_eval ]]; then\n'
        '  shift 2; printf "%s\\n" "$@"; exit 0\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    fake.chmod(0o755)
    grid = tmp_path / "smoke50.json"
    grid.write_text(json.dumps({"manifest_hash": "abcdef0123456789"}))
    common = [
        "bash",
        str(ROOT / "scripts/sim/eval_cell.sh"),
        "square-narrow-r01-baseline",
        "--seed",
        "1",
        "--stage",
        "idql_agent",
        "--checkpoint",
        "/ckpt",
        "--grid-file",
        str(grid),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    env = {**os.environ, "PYTHON": str(fake), "PYTHONPATH": str(ROOT)}
    out = tmp_path / "out/sim/eval/square-narrow-r01-baseline/idql_agent/seed-1/n32"
    grid_dir = out / "grid-smoke50-abcdef01"

    result = subprocess.run(common, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    banner, *argv = result.stdout.splitlines()
    assert f"grid file {grid}" in banner and "square_narrow_sobol8k" in banner
    assert f"--output-dir={grid_dir}" in argv and f"--point-manifest={grid}" in argv

    result = subprocess.run(common + ["--merge"], capture_output=True, text=True, env=env)
    assert result.returncode == 2 and "needs --n-shards" in result.stderr
    result = subprocess.run(
        common + ["--merge", "--n-shards", "2"], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[1:] == [
        "merge",
        "--point-manifest",
        str(grid),
        "--output-dir",
        str(grid_dir),
        "--n-shards",
        "2",
    ]


def test_eval_cell_forwards_seeds(tmp_path):
    """--eval-seed / --env-seed reach grid_eval and keep seeded results in their own directory."""
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$2" == mulligan.sim.eval.grid_eval ]]; then\n'
        '  shift 2; printf "%s\\n" "$@"; exit 0\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    fake.chmod(0o755)
    grid = tmp_path / "smoke50.json"
    grid.write_text(json.dumps({"manifest_hash": "abcdef0123456789"}))
    common = [
        "bash",
        str(ROOT / "scripts/sim/eval_cell.sh"),
        "square-narrow-r01-baseline",
        "--seed",
        "1",
        "--stage",
        "idql_agent",
        "--checkpoint",
        "/ckpt",
        "--grid-file",
        str(grid),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    env = {**os.environ, "PYTHON": str(fake), "PYTHONPATH": str(ROOT)}
    grid_dir = (
        tmp_path
        / "out/sim/eval/square-narrow-r01-baseline/idql_agent/seed-1/n32/grid-smoke50-abcdef01"
    )

    for seeds, subdir in (
        (["--eval-seed", "7"], "eval-seed-7"),
        (["--env-seed", "3"], "env-seed-3"),
        (["--eval-seed", "7", "--env-seed", "3"], "eval-seed-7-env-seed-3"),
    ):
        result = subprocess.run(common + seeds, capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr
        banner, *argv = result.stdout.splitlines()
        assert f"({subdir})" in banner
        assert f"--output-dir={grid_dir / subdir}" in argv
        assert argv[-len(seeds) :] == seeds
        result = subprocess.run(
            common + seeds + ["--merge", "--n-shards", "2"], capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, result.stderr
        assert str(grid_dir / subdir) in result.stdout.splitlines()[1:]

    result = subprocess.run(common, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()[1:]
    assert f"--output-dir={grid_dir}" in argv
    assert not any(a.startswith(("--eval-seed", "--env-seed")) for a in argv)

    result = subprocess.run(common + ["--eval-seed", "-1"], capture_output=True, text=True, env=env)
    assert result.returncode == 2 and "non-negative integers" in result.stderr


def test_grid_eval_argv():
    r = _recipe("square-broad-r01-auto-filtered-bc-n1")
    argv = R.grid_eval_argv(
        r, checkpoint="ck", grid_file="g.json", output_dir="o", shard=2, n_shards=4
    )
    assert argv[0] == "eval"
    assert "--num-action-samples=1" in argv
    assert argv[-2:] == ["--shard-idx=2", "--n-shards=4"]
    gen = R.grid_generate_argv(RECIPES, "square_broad_sobol30k", "g.json")
    assert gen[0] == "make-valid-sobol-manifest" and gen[-1] == "--output=g.json"


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse_and_print_usage(script):
    path = ROOT / script
    subprocess.run(["bash", "-n", str(path)], check=True)
    if script.endswith(".sh"):
        result = subprocess.run(["bash", str(path), "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "Usage" in result.stdout


PAPER_RESULTS = {
    "Fig. 2",
    "App. I.4",
    "Fig. 5",
    "Fig. 7",
    "App. B.2",
    "App. B.5",
    "App. G.7",
    "App. J.2",
    "App. J.4",
}


def test_every_recipe_names_its_paper_results():
    for r in RECIPES["recipes"]:
        assert r["paper"], r["id"]
        assert set(r["paper"]) <= PAPER_RESULTS, r["id"]
    fig7 = {r["id"] for r in RECIPES["recipes"] if "Fig. 7" in r["paper"]}
    assert fig7 == {
        "square-narrow-r01-baseline",
        "square-narrow-r01-baseline-straddled-auto-success",
        "square-narrow-r01-mulligan-no-cf",
        "square-narrow-r01-mulligan",
    }
    out = io.StringIO()
    with redirect_stdout(out):
        R.main(["list", "--paper", "Fig. 7"])
    assert set(out.getvalue().split()) == fig7


# ------------------------------------------------------------------ RLPD and HiL-SERL


def _baselines(family: str) -> list[dict]:
    return [r for r in RECIPES["recipes"] if r["family"] == family]


@pytest.mark.parametrize(
    ("family", "stage"), [("rlpd", "rlpd_agent"), ("hilserl", "hilserl_agent")]
)
def test_baseline_recipes(family, stage):
    from mulligan.baselines.hilserl.tasks import get_task

    for r in _baselines(family):
        assert r["id"].startswith(r["task"].replace("_", "-") + f"-{family}"), r["id"]
        assert r["eval"] == {"stage": stage, "log": r["eval"]["log"]}
        assert R.train_stage(r) == stage and R.default_stage(r) == stage
        assert set(r[stage]) == {"kind", "trainer", "args"} and r[stage]["kind"] == stage
        assert f"--task={r['task']}" in r[stage]["args"]
        assert not {"--seed", "--output_dir", "--session"} & {
            a.split("=", 1)[0] for a in r[stage]["args"]
        }
        # the teleop demos are the task's pinned dataset
        task = get_task(r["task"])
        for d in r["datasets"]:
            assert (d["repo"], d["revision"], d["episodes"]) == (
                task.demo_repo_id,
                task.demo_revision,
                task.num_episodes,
            )
        with pytest.raises(ValueError, match="evaluated during training"):
            R.grid_spec(RECIPES, r)
        with pytest.raises(ValueError, match="no released"):
            R.released_checkpoint(r, stage, r["seeds"][0])
    assert {r["seeds"][0] for r in _baselines(family)} == {1}


def test_rlpd_recipe_args_parse_with_the_trainer():
    from mulligan.baselines.rlpd.train import build_parser

    for r in _baselines("rlpd"):
        argv = R.train_argv(r, "rlpd_agent", 3, checkpoint_dir="run")
        assert argv[-2:] == ["--seed=3", "--output_dir=run"]
        args = build_parser().parse_args(argv)
        assert (args.task, args.seed, args.output_dir) == (r["task"], 3, "run")
        assert args.hidden_dims == (256, 256, 256) and not args.backup_entropy
        teleop = not r["datasets"]
        assert (args.offline_data != "teleop") == teleop, r["id"]
    early = R.get_recipe(RECIPES, "square-narrow-rlpd-early-kill")
    assert build_parser().parse_args(R.train_argv(early, "rlpd_agent", 1)).early_kill


def test_hilserl_recipe_argv_and_trainer():
    r = _recipe("square-narrow-hilserl")
    argv = R.train_argv(r, "hilserl_agent", 1, checkpoint_dir="session")
    assert argv[-2:] == ["--seed=1", "--session=session"]
    with pytest.raises(ValueError):
        R.train_argv(r, "hilserl_agent", 2)
    out = io.StringIO()
    with redirect_stdout(out):
        R.main(["trainer", "square-narrow-hilserl"])
    assert out.getvalue().split() == ["hilserl_agent", "mulligan.baselines.hilserl.nohuman"]
    out = io.StringIO()
    with redirect_stdout(out):
        R.main(["trainer", "square-narrow-r01-mulligan", "--stage", "divl_head"])
    assert out.getvalue().split() == ["divl_head", "mulligan.training.train"]


def test_train_cell_runs_each_family_with_its_trainer(tmp_path):
    """train_cell.sh picks the recipe's trainer and run directory (fake interpreter)."""
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ " $* " == *" mulligan.sim.recipes "* ]]; then exec {sys.executable} "$@"; fi\n'
        'printf "%s\\n" "$@"\n'
    )
    fake.chmod(0o755)
    env = {**os.environ, "PYTHON": str(fake), "MULLIGAN_OUTPUT_DIR": str(tmp_path / "out")}
    for recipe, module, flag in (
        ("square-narrow-rlpd", "mulligan.baselines.rlpd.train", "--output_dir"),
        ("square-broad-hilserl", "mulligan.baselines.hilserl.nohuman", "--session"),
    ):
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/sim/train_cell.sh"),
                recipe,
                "--seed",
                "1",
                "--max_steps=9",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout.splitlines()
        stage = R.train_stage(_recipe(recipe))
        run = tmp_path / "out" / "sim" / "train" / recipe / stage / "seed-1"
        assert out[0] == f"== {recipe} {stage} seed 1 -> {run}"
        assert out[1:3] == ["-m", module]
        assert out[-3:] == ["--seed=1", f"{flag}={run}", "--max_steps=9"]
