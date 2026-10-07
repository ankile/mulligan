"""Extract the locked simulation appendix evidence; no training or experiment services needed."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from mulligan.plotting import paper
from paper.appendix.artifacts import cache_dir, load_inputs, sha256

HERE = Path(__file__).resolve().parent
CACHE = cache_dir("simulation")
# Critic-objective conditions as named in ``sim/critic/objective_final_sr.csv``, with labels.
CONDITIONS = [
    ("Baseline [0,1]", "Baseline [0,1], clipped targets"),
    ("Shifted [-1,0]", "Shifted [-1,0], no clipping"),
    ("Intervention-terminal v2", "Intervention-terminal"),
    ("Actor cadence 4:1:1", "Actor cadence 4:1:1"),
    ("Actor cadence 2:1:1", "Actor cadence 2:1:1"),
    ("Flat critic sampling", "Flat critic sampling"),
    ("QC b1024 s8", "QC (no V-network)"),
    ("QC + intv + cad2", "QC + intervention + cadence 2:1"),
    ("QC shifted [-1,0]", "QC + shifted [-1,0]"),
]


def extract() -> dict[str, Path]:
    sources = load_inputs(HERE)
    out = CACHE / "data"
    out.mkdir(parents=True, exist_ok=True)
    # Final-step (300k) success of every seed that reached it.
    objective = pd.read_csv(sources["sim/critic/objective_final_sr.csv"])
    labels = dict(CONDITIONS)
    assert set(objective.condition) == set(labels) and (objective.step == 300000).all()
    assert len(objective) == 41 and not objective.duplicated(["condition", "seed"]).any()
    critic = pd.concat(
        [objective[objective.condition == condition] for condition in labels], ignore_index=True
    )
    critic.insert(1, "label", critic.condition.map(labels))
    critic["success_rate"] = critic.success_rate.astype(float)
    critic.to_csv(out / "critic_objective_seeds.csv", index=False, float_format="%.12g")
    # Counts digitized from the archived Square-Narrow R2 bucket raster, not rollout logs.
    buckets = pd.read_csv(sources["sim/sampling/bucket_counts.csv"])
    assert buckets["rank"].tolist() == list(range(1, 81)) and set(buckets.rollouts) == {50}
    assert buckets.successes.is_monotonic_increasing and buckets.successes.sum() == 3438
    buckets.to_csv(out / "bucket_counts.csv", index=False)
    states = []
    for arm in ("uniform", "sobol"):
        payload = json.loads(sources[f"sim/sampling/{arm}_starts.json"].read_text())
        assert len(payload["states"]) == 100
        for i, state in enumerate(payload["states"]):
            states.append({"arm": arm, "index": i, **state})
    pd.DataFrame(states).to_csv(out / "initial_states.csv", index=False, float_format="%.17g")
    # The seed summary's eighth arm (autonomous IDQL + success BC) is not plotted.
    efficiency = pd.read_csv(sources["sim/efficiency/seed_summary.csv"])
    assert len(efficiency) == 320
    efficiency = efficiency[efficiency.series_key != "auto_iql_success_bc_n32"]
    assert len(efficiency) == 280
    for key, group in efficiency.groupby(["task_key", "series_key", "round"]):
        assert sorted(group.seed) == [1, 2, 3, 4, 5], key
        assert set(group.point_status) == {"complete"}, key
    efficiency.to_csv(out / "efficiency_seeds.csv", index=False, float_format="%.17g")
    return {p.name: p for p in sorted(out.iterdir()) if p.is_file()}


def write_tables(*, check=False):
    from .tables import write_tables as write

    return write(check=check)


def materialize_assets(*, check=False) -> list[Path]:
    """Restore the authored state-evolution illustration from its archived export."""
    source = load_inputs(HERE)["assets/illustrations/state_evolution.pdf"]
    target = paper.FIGS_DIR / "overview_state_evolution.pdf"
    if check:
        if sha256(target) != sha256(source):
            raise AssertionError(f"{target} differs from its archived export")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return [target]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for name, path in extract().items():
        print(name, path)


if __name__ == "__main__":
    main()
