"""[CI, network] smoke test of the sim pipeline, CPU only, tiny settings.

1. download the Square-Narrow R0 training view (parquet only);
2. train an IDQL agent for 200 steps with a small network (recipe
   ``square-narrow-r00-baseline`` plus overrides);
3. run a 2-start blinded DAgger round (Square-Narrow R1 protocol) with the replay
   operator, both routing arms served by the checkpoint from step 2;
4. grid-evaluate that checkpoint on the first 10 starts of the locked 8k grid;
5. plot the per-start outcomes.

Each stage runs as a subprocess, as a user would run it. Wall time on a laptop-class
CPU is a few minutes, far under the 30 min CI budget.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mulligan.sim import recipes as R

pytestmark = [pytest.mark.network, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = "square-narrow-r00-baseline"
SMALL_TRAINING = [
    "--policy.hidden_dims=[64, 64]",
    "--policy.down_dims=[32, 64]",
    "--policy.num_action_samples=4",
    "--training.batch_size=32",
    "--training.training_steps=200",
    "--eval.freq=1000000",
    "--eval.num_envs=2",
    "--eval.eval_num_action_samples=[1]",
    "--wandb.enabled=False",
    "--system.device=cpu",
]


def _run(module: str, argv: list[str], log: Path) -> None:
    env = {**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    with log.open("w") as fh:
        proc = subprocess.run(
            [sys.executable, "-m", module, *argv],
            cwd=REPO_ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=1200,
        )
    if proc.returncode != 0:
        tail = "".join(log.read_text().splitlines(keepends=True)[-40:])
        raise AssertionError(f"python -m {module} exited {proc.returncode}; log tail:\n{tail}")


def test_sim_pipeline_smoke(tmp_path: Path):
    recipes = R.load(REPO_ROOT / "configs/sim/recipes.json")

    # 1-2. Download + train.
    recipe = R.get_recipe(recipes, RECIPE)
    train_argv = R.train_argv(recipe, "idql_agent", 1, checkpoint_dir=str(tmp_path / "ckpt"))
    _run("mulligan.training.train", train_argv + SMALL_TRAINING, tmp_path / "train.log")
    (checkpoint,) = (tmp_path / "ckpt").glob("*/checkpoints/final_model")
    assert (checkpoint / "policy.pt").is_file()
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["env_name"] == "NutAssemblySquare"

    # 3. Two-start DAgger round driven by the replay operator.
    dagger_argv = R.dagger_argv(
        recipes,
        REPO_ROOT,
        "square_narrow",
        1,
        dataset_root=str(tmp_path / "data"),
        ledger=str(tmp_path / "ledger.jsonl"),
        operator="replay",
        replay_max_episodes=2,
    )
    labels = [a.split("=", 2)[1] for a in dagger_argv if a.startswith("--routed-policy=")]
    dagger_argv = [
        a
        for a in dagger_argv
        if not a.startswith(("--routed-policy=", "--cameras=", "--num-action-samples="))
    ]
    dagger_argv += [f"--routed-policy={label}={checkpoint}" for label in labels]
    dagger_argv += ["--num-action-samples=4", "--device=cpu"]
    _run("mulligan.sim.collect.dagger", dagger_argv, tmp_path / "dagger.log")
    ledger = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len({row["episode_index"] for row in ledger if "episode_index" in row}) == 2
    info = json.loads(
        (tmp_path / "data/sim-square-narrow-c01-dagger-mixed/meta/info.json").read_text()
    )
    assert info["total_episodes"] == 2

    # 4. Grid eval of the first 10 starts of the locked Square-Narrow grid.
    grid = tmp_path / "grid.json"
    _run(
        "mulligan.sim.eval.grid_eval",
        R.grid_generate_argv(recipes, "square_narrow_sobol8k", str(grid)),
        tmp_path / "grid.log",
    )
    eval_dir = tmp_path / "eval"
    eval_argv = [
        "eval",
        f"--artifact-path={checkpoint}",
        f"--point-manifest={grid}",
        f"--output-dir={eval_dir}",
        "--n-shards=800",
        "--shard-idx=1",
        "--num-envs=2",
        "--num-action-samples=4",
        "--device=cpu",
    ]
    _run("mulligan.sim.eval.grid_eval", eval_argv, tmp_path / "eval.log")
    points = json.loads((eval_dir / "point_results_shard_1_of_800.json").read_text())
    rows = points["points"] if isinstance(points, dict) else points
    assert len(rows) == 10

    # 5. One plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.scatter(
        [r["nut_y"] for r in rows],
        [r["nut_yaw"] for r in rows],
        c=["tab:green" if r["success"] else "tab:red" for r in rows],
    )
    ax.set_xlabel("nut y (m)")
    ax.set_ylabel("nut yaw (rad)")
    out = tmp_path / "grid_eval_smoke.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    assert out.stat().st_size > 0
