"""Welch-t contrasts over the frozen-actor DIVL campaign (N = 32 grid success, five seeds).

Shared by the simulation significance table and the sampling-ablation numbers of the paper:
the round-1 sampler/actor-data ablation on Square-Narrow (tab:appendix-sampler-ablation,
fig:component-ablations) and the round-0 Sobol-vs-uniform comparison. Run
``python -m paper.stats.sim_contrasts`` to print them.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from mulligan.sim import recipes as sim_recipes
from mulligan.sim.task_names import sim_task_display_name

DIVL = (
    Path(__file__).resolve().parents[1] / "data/sim/sim_significance/divl_frozen_actor_results.csv"
)
SEEDS = (1, 2, 3, 4, 5)
REFERENCE = "square_narrow_r1_baseline_uniform_no_cf_human_only"
# Square-Narrow round-1 cells of the sampler/actor-data ablation, in table order.
SAMPLER_ABLATION = (
    REFERENCE,
    "square_narrow_r1_baseline_uniform_no_cf_straddled_auto_success",
    "square_narrow_r1_sobol_no_cf_human_only",
    "square_narrow_r1_sobol_with_cf_human_only",
    "square_narrow_r1_mulligan_no_cf_human_only",
    "square_narrow_r1_mulligan_with_cf_human_only",
)


def welch(a, b) -> tuple[float, float, float, float]:
    """Mean difference a - b, its Welch-t 95% interval, and the two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    se = np.sqrt(va + vb)
    half = student_t.ppf(0.975, df) * se
    p = 2 * student_t.sf(abs(delta) / se, df)
    return float(delta), float(delta - half), float(delta + half), float(p)


def mean_se(values) -> tuple[float, float]:
    values = np.asarray(values, float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def divl_cells(path: Path = DIVL) -> dict[str, np.ndarray]:
    """Per-seed N = 32 grid success (%) of every human-in-the-loop cell, ordered by seed."""
    seeds = defaultdict(dict)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["family"] == "hil":
                seeds[row["cell_key"]][int(row["seed"])] = float(row["grid_n32_pct"])
    for key, by_seed in seeds.items():
        if tuple(sorted(by_seed)) != SEEDS:
            raise ValueError(f"{path}: {key} has seeds {sorted(by_seed)}, expected {SEEDS}")
    return {key: np.array([by_seed[s] for s in SEEDS]) for key, by_seed in seeds.items()}


def recipe_ids() -> dict[str, str]:
    """Released recipe id (``configs/sim/recipes.json``) of each DIVL campaign cell key."""
    ids = {}
    for recipe in sim_recipes.load()["recipes"]:
        key = recipe.get("source_cell_key")
        if key is None:
            continue
        if key in ids:
            raise ValueError(f"cell {key} maps to recipes {ids[key]} and {recipe['id']}")
        ids[key] = recipe["id"]
    return ids


def recipe_id(cell_key: str, ids: dict[str, str] | None = None) -> str:
    ids = recipe_ids() if ids is None else ids
    if cell_key not in ids:
        raise KeyError(f"no recipe in {sim_recipes.DEFAULT_RECIPES} has source_cell_key {cell_key}")
    return ids[cell_key]


def sampler_ablation(cells: dict[str, np.ndarray] | None = None) -> list[dict]:
    """Mean +- SE of each ablation cell and its Welch contrast against the uniform reference."""
    cells = divl_cells() if cells is None else cells
    rows = []
    for key in SAMPLER_ABLATION:
        mean, se = mean_se(cells[key])
        delta, lo, hi, p = welch(cells[key], cells[REFERENCE])
        rows.append(dict(cell_key=key, mean=mean, se=se, delta=delta, lo=lo, hi=hi, p=p))
    return rows


def sobol_round0(cells: dict[str, np.ndarray] | None = None) -> dict[str, dict]:
    """Round-0 Sobol-vs-uniform demonstrations per task (N = 32 critic reranking)."""
    cells = divl_cells() if cells is None else cells
    out = {}
    for task in ("square_narrow", "square_broad"):
        uniform, sobol = cells[f"{task}_r0_baseline_uniform"], cells[f"{task}_r0_sobol"]
        delta, lo, hi, p = welch(sobol, uniform)
        out[task] = dict(
            uniform=mean_se(uniform), sobol=mean_se(sobol), delta=delta, lo=lo, hi=hi, p=p
        )
    return out


def main() -> None:
    cells = divl_cells()
    ids = recipe_ids()
    print("Square-Narrow R1 sampler / actor-data ablation (grid SR %, delta vs. reference):")
    for row in sampler_ablation(cells):
        contrast = (
            "(reference)"
            if row["cell_key"] == REFERENCE
            else f"{row['delta']:+.2f} [{row['lo']:+.2f}, {row['hi']:+.2f}]"
        )
        recipe = recipe_id(row["cell_key"], ids)
        print(
            f"  {row['cell_key']:55s} {recipe:50s} {row['mean']:.2f} +- {row['se']:.2f}  {contrast}"
        )
    print("Round-0 Sobol vs. uniform demonstrations:")
    for task, row in sobol_round0(cells).items():
        (um, us), (sm, ss) = row["uniform"], row["sobol"]
        uniform = recipe_id(f"{task}_r0_baseline_uniform", ids)
        sobol = recipe_id(f"{task}_r0_sobol", ids)
        print(
            f"  {task} ({sim_task_display_name(task)}; {uniform} -> {sobol}): {um:.2f} +- {us:.2f} -> {sm:.2f} +- {ss:.2f}  "
            f"{row['delta']:+.2f} pp [{row['lo']:+.2f}, {row['hi']:+.2f}]"
        )


if __name__ == "__main__":
    main()
