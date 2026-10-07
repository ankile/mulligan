# Simulation and sampling appendix evidence

This package owns five generated figures, five empirical tables, and the archived
manual state-evolution schematic. Normal reproduction reads `inputs.json` and the
hash-checked paper-evidence files it pins (under `sim/` and
`assets/illustrations/`).

## Reproduce

After `uv sync`:

```sh
python -m paper.appendix.simulation.prepare
python -m paper.figures --only value_learning_bar_chart square_narrow_r2_bucket_success_sorted square_narrow_init_distribution_sobol_vs_uniform sim_success_speed sim_success_throughput
```

The first command extracts the tidy tables into
`paper/build/cache/simulation/data/`. `python -m paper.appendix.build` also calls
`prepare.write_tables()` to write the five `*_table.tex` bodies to
`paper/build/tables` and, through the `restored_state_evolution` registry entry,
`prepare.materialize_assets()` to copy the manual schematic to
`paper/build/figs`. Both accept `check=True`: tables must equal the frozen
manuscript's bodies and the restored PDF must equal its archived export. Figures
use `mulligan.plotting.paper`; the paper driver owns their output directory and
manifest.

The run summaries of the sampler/head study are archived one record per run
(`sim/critic/square_narrow_sampler_runs.json`), and the efficiency seed summary is
frozen as `sim/efficiency/seed_summary.csv`.

## Evidence and calculations

- **Initial coverage:** the two 100-state R0 JSON manifests. Extracted
  `initial_states.csv` retains coordinates and state identity. Centered
  discrepancy is SciPy's CD statistic, not star discrepancy. Nearest-neighbor
  distance is Euclidean on normalized y/yaw, without periodic wrap at the yaw
  boundary. Marginal spread is the mean population standard deviation of ten
  equal-width bin counts on each coordinate. These describe two fixed draws.
- **Bucket difficulty:** `sim/sampling/bucket_counts.csv`, the 80 rank-ordered
  per-bucket success counts (50 rollouts each, 3438/4000) digitized from the
  archived Square-Narrow R2 bucket raster, whose bars lie on a 2-point grid.
  Per-rollout records and state-to-rank identities are not available; the
  counts are a digitization, not rollout logs. The figure shows seed 1, not an
  average across seeds.
- **Speed/throughput:** `sim/efficiency/seed_summary.csv`: 280 plotted seed rows, seven series, two tasks, four
  rounds, five seeds. Its eighth series (autonomous IDQL + success BC) is not
  plotted. Mean duration uses successful episodes at 20 Hz; throughput divides
  successes by all episode time. Figures recompute equal-seed means and
  two-sided Student-t 95% intervals from recorded seed metrics. Individual
  episode records are not included. These metrics use the scalar-IQL heads, not
  the frozen-actor DIVL heads.
- **Critic objectives:** `sim/critic/objective_final_sr.csv`, nine conditions and
  41 completed-seed final-step (300k) scores. The figure and table share
  final-step scores and sample SE. QC's two incomplete conditions retain four
  completed seeds; flat sampling has three. No pairing across conditions is
  claimed. The QC shifted mean is 91.25%.
- **DIVL regimes/samplers:** 45 finished run summaries/configs
  (`square_narrow_sampler_runs.json`, one record per run with its sampler, head and seed)
  for the Square-Narrow R1 three-by-three sampler/head study. The n32 metric, 400 fixed starts,
  seed 20260524, and final-step scope are checked. Square-Broad R1 and R15
  replicates are in `sim/critic/divl_regime_replicates.csv`; Square-Broad R1's
  five values are in reported order (`seed_known` false), R15's are seeds 1-3.
  The R15 fixed/adaptive changes compute to -1.17/-0.83 points. All tables report
  numeric deltas.
- **REDQ:** capacity `final_sr.csv` rows preserve seeds, step, and n32 evaluation
  identity. Three conditions, five seeds, final 300k. Mean/SE and deltas are
  recomputed; a small observed difference is not an equivalence test and no
  statistical-null labels are generated.
- **Frozen actors and sampler table:** full campaign results are archived. Strictly filter `family=hil` (the HiL collection cells): 170 rows, 34 complete cells,
  seeds 1–5; autonomous cells are excluded. `frozen_actor_summary.json` verifies
  the 20/14 sign split, [-0.470,+0.5525] range and +0.0419002 point
  equal-cell mean. The six R1 sampler/actor-data rows, R0 contrasts, and Narrow
  R0–R3 sequence are computed from these same named cells. Arm-difference CIs
  use Welch's t method across five training seeds, not rollout-level tests.
- **State-evolution schematic:** the exported PDF is archived as an authored
  illustration (`assets/illustrations/state_evolution.pdf`). It is not
  experimental evidence. Restoring the exported PDF preserves author typography.

`tables.py` writes `critic_objective_table.tex`, `divl_regime_table.tex`,
`divl_sampler_table.tex`, `redq_table.tex`, and `sampler_ablation_table.tex`.
All derived CSV/JSON files go in `paper/build/cache/simulation/data/`. Only code,
locks, and instructions are in git.
