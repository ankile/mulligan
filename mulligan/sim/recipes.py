"""Read ``configs/sim/recipes.json`` and build the command lines of the sim round scripts.

The scripts under ``scripts/sim/`` call this module and run the argv it prints, so the
flags of every sim command live here in one place. Argv lists are written to stdout
NUL-separated (read them in bash with ``while IFS= read -r -d ''``); other subcommands
print plain text.

    python -m mulligan.sim.recipes list --task square_narrow
    python -m mulligan.sim.recipes show square-narrow-r01-baseline
    python -m mulligan.sim.recipes train-argv square-narrow-r01-baseline \
        --stage idql_agent --seed 1 --checkpoint-dir outputs/sim/train
    python -m mulligan.sim.recipes trainer square-narrow-rlpd    # training stage and module

See ``configs/sim/README.md`` for the recipe schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPES = REPO_ROOT / "configs" / "sim" / "recipes.json"
STAGES = ("idql_agent", "divl_head", "rlpd_agent", "hilserl_agent")
# The stage a recipe trains first (its other stage, a DIVL head, starts from it).
TRAIN_STAGES = ("idql_agent", "rlpd_agent", "hilserl_agent")
FAMILIES = ("hil", "autonomous_baseline", "rlpd", "hilserl")
# Scripted operator that replays the recorded human segments of a DAgger dataset.
REPLAY_OPERATOR = "mulligan.sim.collect.replay_operator:make_replay_operator"


def load(path: Path | str = DEFAULT_RECIPES) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def data_root(path: Path | str = DEFAULT_RECIPES) -> Path:
    """Repository root that the repo-relative paths in ``recipes.json`` refer to."""
    return Path(path).resolve().parents[2]


def get_recipe(recipes: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    for recipe in recipes["recipes"]:
        if recipe["id"] == recipe_id:
            if recipe["status"] != "ok":
                raise ValueError(f"recipe {recipe_id} is not runnable: {recipe['status']}")
            return recipe
    known = ", ".join(r["id"] for r in recipes["recipes"])
    raise KeyError(f"unknown recipe {recipe_id!r}; known: {known}")


def get_stage(recipe: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if stage not in recipe:
        raise ValueError(f"recipe {recipe['id']} has no {stage} stage")
    return recipe[stage]


def default_stage(recipe: dict[str, Any]) -> str:
    """The stage the paper reports: the DIVL head for the human-in-the-loop arms, the IDQL agent
    for the autonomous baselines, the agent for RLPD and HiL-SERL."""
    return recipe["eval"]["stage"]


def train_stage(recipe: dict[str, Any]) -> str:
    """The stage ``scripts/sim/train_cell.sh`` trains."""
    return next(stage for stage in TRAIN_STAGES if stage in recipe)


def trainer(recipes: dict[str, Any], recipe: dict[str, Any], stage: str) -> str:
    """The module that trains ``stage`` (``python -m <module> <train_argv>``)."""
    return get_stage(recipe, stage).get("trainer", recipes["trainer"])


def _check_seed(recipe: dict[str, Any], seed: int) -> None:
    if seed not in recipe["seeds"]:
        raise ValueError(f"recipe {recipe['id']} has seeds {recipe['seeds']}, got {seed}")


def released_checkpoint(recipe: dict[str, Any], stage: str, seed: int) -> str:
    """``hf://`` reference of the released checkpoint for one seed."""
    _check_seed(recipe, seed)
    spec = get_stage(recipe, stage)
    if "model" not in spec:
        raise ValueError(f"recipe {recipe['id']} has no released {stage} checkpoints")
    model = spec["model"]
    return f"hf://{model['repo']}@{model['revision']}/seed-{seed}"


def train_argv(
    recipe: dict[str, Any],
    stage: str,
    seed: int,
    *,
    checkpoint_dir: str | None = None,
    parent: str | None = None,
) -> list[str]:
    """Arguments for ``python -m <trainer(...)>`` (without the module itself).

    ``checkpoint_dir`` is the run directory (RLPD ``--output_dir``, HiL-SERL ``--session``).
    ``parent`` overrides the DIVL head's released parent (e.g. a locally retrained agent).
    """
    _check_seed(recipe, seed)
    spec = get_stage(recipe, stage)
    if stage in ("rlpd_agent", "hilserl_agent"):
        if parent is not None:
            raise ValueError("--parent only applies to the divl_head stage")
        argv = [*spec["args"], *spec.get("seed_args", {}).get(str(seed), []), f"--seed={seed}"]
        if checkpoint_dir is not None:
            flag = "--output_dir" if stage == "rlpd_agent" else "--session"
            argv.append(f"{flag}={checkpoint_dir}")
        return argv
    datasets = recipe["datasets"]
    revisions = {d["repo"]: d["revision"] for d in datasets}
    argv = [
        "--dataset.repo_ids=" + ",".join(d["repo"] for d in datasets),
        "--dataset.revisions=" + json.dumps(revisions, separators=(",", ":")),
        *spec["args"],
        *spec.get("seed_args", {}).get(str(seed), []),
        f"--training.seed={seed}",
    ]
    if stage == "divl_head":
        if parent is None:
            parent = released_checkpoint(recipe, spec["parent_stage"], seed)
        argv.append(f"--pretrained_artifact={parent}")
    elif parent is not None:
        raise ValueError("--parent only applies to the divl_head stage")
    if checkpoint_dir is not None:
        argv.append(f"--system.checkpoint_dir={checkpoint_dir}")
    return argv


def grid_spec(recipes: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    if "grid" not in recipe["eval"]:
        raise ValueError(
            f"recipe {recipe['id']} is evaluated during training: see {recipe['eval']['log']} "
            "in its run directory"
        )
    return recipes["grids"][recipe["eval"]["grid"]]


def grid_generate_argv(recipes: dict[str, Any], grid_id: str, output: str) -> list[str]:
    """Arguments for ``python -m mulligan.sim.eval.grid_eval`` that regenerate a locked grid."""
    return [*recipes["grids"][grid_id]["generate"], f"--output={output}"]


def grid_eval_argv(
    recipe: dict[str, Any],
    *,
    checkpoint: str,
    grid_file: str,
    output_dir: str,
    num_action_samples: int | None = None,
    shard: int | None = None,
    n_shards: int | None = None,
) -> list[str]:
    """Arguments for ``python -m mulligan.sim.eval.grid_eval eval``."""
    n = recipe["eval"]["num_action_samples"] if num_action_samples is None else num_action_samples
    argv = [
        "eval",
        f"--artifact-path={checkpoint}",
        f"--point-manifest={grid_file}",
        f"--output-dir={output_dir}",
        f"--num-action-samples={n}",
    ]
    if (shard is None) != (n_shards is None):
        raise ValueError("--shard and --n-shards go together")
    if shard is not None:
        argv += [f"--shard-idx={shard}", f"--n-shards={n_shards}"]
    return argv


def get_round(recipes: dict[str, Any], task: str, round_number: int) -> dict[str, Any]:
    for entry in recipes["rounds"]:
        if entry["task"] == task and entry["round"] == round_number:
            return entry
    known = ", ".join(f"{e['task']}/r{e['round']}" for e in recipes["rounds"])
    raise KeyError(f"no round {task}/r{round_number}; known: {known}")


def _repo_path(root: Path, rel: str) -> str:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f"{path} (listed in recipes.json) does not exist")
    return str(path)


def _dataset_name(repo: str) -> str:
    return repo.split("/", 1)[1]


def dagger_argv(
    recipes: dict[str, Any],
    root: Path,
    task: str,
    round_number: int,
    *,
    dataset_root: str,
    ledger: str,
    operator: str = "human",
    replay_dataset: str | None = None,
    replay_max_episodes: int | None = None,
    hub_namespace: str | None = None,
    push: bool = False,
) -> list[str]:
    """Arguments for ``python -m mulligan.sim.collect.dagger`` for one blinded DAgger round.

    ``operator`` is ``human`` (SpaceMouse + keyboard), ``replay`` (the recorded human segments
    of ``replay_dataset``, default the released mixed collection of the round, headless) or a
    ``module:factory`` passed through to the collector. ``replay_dataset`` is read by
    :func:`replay_source`. ``replay_max_episodes`` ends a replay session after that many
    episodes.
    """
    entry = get_round(recipes, task, round_number)
    if "dagger" not in entry:
        raise ValueError(f"{task}/r{round_number} has no DAgger collection (R0 is teleop)")
    spec = entry["dagger"]
    mulligan, baseline = spec["policies"]["mulligan"], spec["policies"]["baseline"]
    manifest = _repo_path(root, spec["manifest"])
    argv = [
        f"--env={recipes['tasks'][task]['env_name']}",
        "--robot=Panda",
        f"--routed-policy={mulligan['routing_label']}={mulligan['checkpoint']}",
        f"--routed-policy={baseline['routing_label']}={baseline['checkpoint']}",
        f"--policy-routing-manifest={manifest}",
        f"--cameras={spec['cameras']}",
        f"--dataset-name={_dataset_name(spec['dataset'])}",
        f"--dataset-path={dataset_root}",
        "--auto-save-on-success",
        "--sampler=list",
        f"--initial-states-file={_repo_path(root, spec['starts'])}",
        "--no-sampler-shuffle",
        f"--adaptive-protocol-quota-manifest={manifest}",
        f"--adaptive-protocol-quota-targets={spec['protocol_targets']}",
        f"--adaptive-protocol-quota-arms={spec['protocol_arms']}",
        f"--adaptive-protocol-quota-ledger={ledger}",
        f"--adaptive-protocol-quota-balance-slack={spec['balance_slack']}",
        f"--adaptive-protocol-quota-progress-window={spec['progress_window']}",
        f"--num-action-samples={spec['num_action_samples']}",
    ]
    if operator == "replay":
        if replay_dataset is None:
            repo = spec["dataset"]
            kwargs = {"repo_id": repo, "revision": _dataset_pin(recipes, entry, repo)}
        else:
            kwargs = replay_source(replay_dataset)
        if replay_max_episodes is not None:
            kwargs["max_episodes"] = replay_max_episodes
        argv += [
            "--headless",
            f"--operator={REPLAY_OPERATOR}",
            "--operator-kwargs=" + json.dumps(kwargs, separators=(",", ":")),
        ]
    elif replay_dataset is not None or replay_max_episodes is not None:
        raise ValueError("replay_dataset and replay_max_episodes need operator='replay'")
    elif operator != "human":
        if ":" not in operator:
            raise ValueError(f"operator must be human, replay or module:factory, got {operator!r}")
        argv.append(f"--operator={operator}")
    if hub_namespace is not None:
        argv.append(f"--hub-namespace={hub_namespace}")
    if push:
        argv.append("--push-to-hub")
    return argv


def replay_source(replay_dataset: str) -> dict[str, str]:
    """Replay-operator keyword arguments for a recorded DAgger dataset.

    ``REPO@REVISION`` reads an HF dataset at that revision; a bare released ``mulligan/*``
    repo is read at its pin in ``release/revisions.json``; an existing local directory (a
    LeRobot dataset root, e.g. ``outputs/sim/data/<name>``) is read in place.
    """
    repo, sep, revision = replay_dataset.partition("@")
    if sep:
        if not repo or not revision:
            raise ValueError(f"--replay-dataset {replay_dataset!r}: expected REPO@REVISION")
        return {"repo_id": repo, "revision": revision}
    path = Path(replay_dataset).expanduser()
    if path.is_dir():
        return {"root": str(path.resolve())}
    if replay_dataset.startswith("mulligan/") and replay_dataset.count("/") == 1:
        from mulligan.release.download import pinned_revision, repo_type

        if repo_type(replay_dataset) != "dataset":
            raise ValueError(f"--replay-dataset {replay_dataset} is not a dataset repo")
        return {"repo_id": replay_dataset, "revision": pinned_revision(replay_dataset)}
    raise ValueError(
        f"--replay-dataset {replay_dataset!r} is not a local directory or a released "
        "mulligan/* dataset; pin other HF datasets as REPO@REVISION"
    )


def _dataset_pin(recipes: dict[str, Any], entry: dict[str, Any], repo: str) -> str:
    for split in entry.get("splits", []):
        if split["input"] == repo:
            return split["input_revision"]
    raise KeyError(f"no pinned revision for {repo} in round {entry['task']}/r{entry['round']}")


def rollouts_argv(
    recipes: dict[str, Any],
    root: Path,
    task: str,
    round_number: int,
    index: int,
    *,
    dataset_root: str,
    audit_output: str,
    push: bool = False,
) -> list[str]:
    """Arguments for ``python -m mulligan.sim.collect.rollouts`` (a round's policy rollouts)."""
    entry = get_round(recipes, task, round_number)
    rollouts = entry.get("rollouts", [])
    if not 0 <= index < len(rollouts):
        raise IndexError(f"{task}/r{round_number} has {len(rollouts)} rollout jobs, got {index}")
    spec = rollouts[index]
    return _rollouts_common(
        recipes,
        task,
        checkpoint=spec["checkpoint"],
        starts=_repo_path(root, spec["starts"]),
        dataset=spec["dataset"],
        num_episodes=spec["num_episodes"],
        dataset_root=dataset_root,
        audit_output=audit_output,
        push=push,
    )


def autonomous_argv(
    recipes: dict[str, Any],
    root: Path,
    recipe: dict[str, Any],
    *,
    dataset_root: str,
    audit_output: str,
    predecessor: str | None = None,
    push: bool = False,
) -> list[str]:
    """Arguments for the autonomous-baseline collection that feeds ``recipe``."""
    if "autonomous_collection" not in recipe:
        raise ValueError(f"recipe {recipe['id']} has no autonomous collection")
    spec = recipe["autonomous_collection"]
    argv = _rollouts_common(
        recipes,
        recipe["task"],
        checkpoint=predecessor or spec["predecessor"],
        starts=_repo_path(root, spec["starts"]),
        dataset=spec["dataset"],
        num_episodes=spec["num_episodes"],
        dataset_root=dataset_root,
        audit_output=audit_output,
        push=push,
        num_envs=spec["num_envs"],
    )
    argv += [f"--num-action-samples={spec['num_action_samples']}", f"--seed={spec['seed']}"]
    return argv


def _rollouts_common(
    recipes: dict[str, Any],
    task: str,
    *,
    checkpoint: str,
    starts: str,
    dataset: str,
    num_episodes: int,
    dataset_root: str,
    audit_output: str,
    push: bool,
    num_envs: int = 10,
) -> list[str]:
    argv = [
        f"--checkpoint={checkpoint}",
        f"--dataset-name={_dataset_name(dataset)}",
        f"--initial-states-json-file={starts}",
        f"--audit-output={audit_output}",
        f"--env={recipes['tasks'][task]['env_name']}",
        "--robot=Panda",
        f"--num-episodes={num_episodes}",
        "--max-steps=400",
        f"--num-envs={num_envs}",
        f"--dataset-path={dataset_root}",
    ]
    if push:
        argv.append("--push-to-hub")
    return argv


def split_argv(
    recipes: dict[str, Any],
    root: Path,
    task: str,
    round_number: int,
    index: int,
    *,
    dataset_root: str,
    output_root: str,
    source_root: str | None = None,
    ledger: str | None = None,
    push: bool = False,
) -> tuple[str, list[str]]:
    """(module, arguments) for splitting a round's mixed collection into per-arm datasets.

    ``source_root`` defaults to ``<dataset_root>/<name>``, where the collector wrote it.
    """
    entry = get_round(recipes, task, round_number)
    splits = entry["splits"]
    if not 0 <= index < len(splits):
        raise IndexError(f"{task}/r{round_number} has {len(splits)} splits, got {index}")
    spec = splits[index]
    if source_root is None:
        source_root = str(Path(dataset_root) / _dataset_name(spec["input"]))
    argv = [
        f"--source-repo={spec['input']}",
        f"--source-root={source_root}",
        f"--manifest={_repo_path(root, spec['manifest'])}",
    ]
    if spec["kind"] == "protocol_quota":
        if ledger is None:
            raise ValueError("a protocol-quota split needs the collection ledger (--ledger)")
        module = "mulligan.data.split_protocol_quota"
        argv += [
            f"--ledger={ledger}",
            f"--expected-per-protocol={spec['expected_per_protocol']}",
            f"--expected-protocol-arms={spec['expected_protocol_arms']}",
        ]
    elif spec["kind"] == "blind":
        module = "mulligan.data.split_blind"
        argv.append(f"--expected-per-source={spec['expected_per_source']}")
        if "max_source_episodes" in spec:
            argv.append(f"--max-source-episodes={spec['max_source_episodes']}")
    else:
        raise ValueError(f"unknown split kind {spec['kind']!r}")
    argv += [f"--target={key}={repo}" for key, repo in spec["targets"].items()]
    argv += [f"--output-root={output_root}", "--drop-visual-features"]
    if push:
        argv.append("--push")
    return module, argv


def _print0(argv: list[str]) -> None:
    sys.stdout.write("".join(f"{item}\0" for item in argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="recipe ids, one per line")
    p.add_argument("--task", choices=("square_narrow", "square_broad"))
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--with-divl", action="store_true", help="only recipes with a DIVL head")
    p.add_argument(
        "--paper", metavar="RESULT", help='only recipes a paper result reads, e.g. "Fig. 7"'
    )

    p = sub.add_parser("show", help="one recipe as JSON")
    p.add_argument("recipe")

    p = sub.add_parser("seeds", help="the recipe's seeds, one per line")
    p.add_argument("recipe")

    p = sub.add_parser("default-stage", help="the stage the paper reports")
    p.add_argument("recipe")

    p = sub.add_parser("trainer", help="training stage and trainer module (default: train stage)")
    p.add_argument("recipe")
    p.add_argument("--stage", choices=STAGES)

    p = sub.add_parser("checkpoint", help="hf:// reference of a released checkpoint")
    p.add_argument("recipe")
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--seed", type=int, required=True)

    p = sub.add_parser("train-argv", help="argv for the stage's trainer (NUL-separated)")
    p.add_argument("recipe")
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint-dir")
    p.add_argument("--parent", help="DIVL parent checkpoint (default: the released agent)")

    p = sub.add_parser(
        "grid", help="grid id, manifest hash, shard count and deployment N of a recipe's eval"
    )
    p.add_argument("recipe")

    p = sub.add_parser("grid-generate-argv", help="argv for grid_eval that rebuilds a grid")
    p.add_argument("grid")
    p.add_argument("--output", required=True)

    p = sub.add_parser("eval-argv", help="argv for mulligan.sim.eval.grid_eval eval")
    p.add_argument("recipe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--grid-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-action-samples", type=int)
    p.add_argument("--shard", type=int)
    p.add_argument("--n-shards", type=int)

    p = sub.add_parser("dagger-argv", help="argv for mulligan.sim.collect.dagger")
    p.add_argument("--task", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--operator", default="human", help="human, replay or module:factory")
    p.add_argument(
        "--replay-dataset",
        help="recording to replay: REPO@REVISION, a released mulligan/* dataset (read at its "
        "release pin) or a local dataset directory (default: the round's released collection)",
    )
    p.add_argument("--replay-max-episodes", type=int)
    p.add_argument("--hub-namespace")
    p.add_argument("--push", action="store_true")

    p = sub.add_parser("rollouts-argv", help="argv for mulligan.sim.collect.rollouts")
    p.add_argument("--task", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--audit-output", required=True)
    p.add_argument("--push", action="store_true")

    p = sub.add_parser("autonomous-argv", help="argv for an autonomous-baseline collection")
    p.add_argument("recipe")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--audit-output", required=True)
    p.add_argument("--predecessor")
    p.add_argument("--push", action="store_true")

    p = sub.add_parser("split-argv", help="module then argv of a round split (NUL-separated)")
    p.add_argument("--task", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--source-root")
    p.add_argument("--output-root", required=True)
    p.add_argument("--ledger")
    p.add_argument("--push", action="store_true")

    p = sub.add_parser("round-info", help="counts of splits and rollout jobs of a round")
    p.add_argument("--task", required=True)
    p.add_argument("--round", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    recipes = load(args.recipes)
    root = data_root(args.recipes)
    cmd = args.command
    if cmd == "list":
        for r in recipes["recipes"]:
            if args.task and r["task"] != args.task:
                continue
            if args.family and r["family"] != args.family:
                continue
            if args.with_divl and "divl_head" not in r:
                continue
            if args.paper and args.paper not in r["paper"]:
                continue
            print(r["id"])
    elif cmd == "show":
        print(json.dumps(get_recipe(recipes, args.recipe), indent=2))
    elif cmd == "seeds":
        print("\n".join(str(s) for s in get_recipe(recipes, args.recipe)["seeds"]))
    elif cmd == "default-stage":
        print(default_stage(get_recipe(recipes, args.recipe)))
    elif cmd == "trainer":
        recipe = get_recipe(recipes, args.recipe)
        stage = args.stage or train_stage(recipe)
        print(stage, trainer(recipes, recipe, stage))
    elif cmd == "checkpoint":
        print(released_checkpoint(get_recipe(recipes, args.recipe), args.stage, args.seed))
    elif cmd == "train-argv":
        recipe = get_recipe(recipes, args.recipe)
        _print0(
            train_argv(
                recipe,
                args.stage,
                args.seed,
                checkpoint_dir=args.checkpoint_dir,
                parent=args.parent,
            )
        )
    elif cmd == "grid":
        recipe = get_recipe(recipes, args.recipe)
        grid = grid_spec(recipes, recipe)
        print(
            recipe["eval"]["grid"],
            grid["manifest_hash"],
            grid["shards"],
            recipe["eval"]["num_action_samples"],
        )
    elif cmd == "grid-generate-argv":
        _print0(grid_generate_argv(recipes, args.grid, args.output))
    elif cmd == "eval-argv":
        _print0(
            grid_eval_argv(
                get_recipe(recipes, args.recipe),
                checkpoint=args.checkpoint,
                grid_file=args.grid_file,
                output_dir=args.output_dir,
                num_action_samples=args.num_action_samples,
                shard=args.shard,
                n_shards=args.n_shards,
            )
        )
    elif cmd == "dagger-argv":
        _print0(
            dagger_argv(
                recipes,
                root,
                args.task,
                args.round,
                dataset_root=args.dataset_root,
                ledger=args.ledger,
                operator=args.operator,
                replay_dataset=args.replay_dataset,
                replay_max_episodes=args.replay_max_episodes,
                hub_namespace=args.hub_namespace,
                push=args.push,
            )
        )
    elif cmd == "rollouts-argv":
        _print0(
            rollouts_argv(
                recipes,
                root,
                args.task,
                args.round,
                args.index,
                dataset_root=args.dataset_root,
                audit_output=args.audit_output,
                push=args.push,
            )
        )
    elif cmd == "autonomous-argv":
        _print0(
            autonomous_argv(
                recipes,
                root,
                get_recipe(recipes, args.recipe),
                dataset_root=args.dataset_root,
                audit_output=args.audit_output,
                predecessor=args.predecessor,
                push=args.push,
            )
        )
    elif cmd == "split-argv":
        module, split = split_argv(
            recipes,
            root,
            args.task,
            args.round,
            args.index,
            dataset_root=args.dataset_root,
            source_root=args.source_root,
            output_root=args.output_root,
            ledger=args.ledger,
            push=args.push,
        )
        _print0([module, *split])
    elif cmd == "round-info":
        entry = get_round(recipes, args.task, args.round)
        print(
            f"dagger={int('dagger' in entry)} splits={len(entry['splits'])} "
            f"rollouts={len(entry.get('rollouts', []))}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc.args[0] if exc.args else exc}") from None
