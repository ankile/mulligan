# Real-world appendix results

This package reproduces eight appendix figures and four statistical tables from hash-pinned paper evidence. Rendering reads only the pinned evidence and the frozen configuration.

## Reproduce

After `uv sync`:

```bash
python -m paper.appendix.real_results.prepare
python -m paper.appendix.real_results.plot
```

The first command writes the derived CSVs to `paper/build/cache/real_results/data/`; the second writes all eight figures to `paper/build/figs` and proof PNGs to `paper/build/proofs` (`--only <name>` for a subset). For the paper, use the shared registry:

```bash
python -m paper.figures --only real_world_headline_mulligan_vs_baseline_detailed real_world_marker_d2_substage real_world_square_d2_substage real_world_cf_ablation_square real_world_dagger_collection_success_detailed real_world_burden_progression real_world_success_speed real_world_success_throughput
```

`python -m paper.appendix.build` calls `statistics.write_tables()` for the four `real_results_*.tex` bodies in `paper/build/tables`; with `--check` they must equal the frozen manuscript's bytes. Inputs are fetched and SHA-256 checked by `paper.appendix.artifacts.load_inputs`; deleting `paper/build/cache` is safe.

## Evidence and study boundaries

`inputs.json` pins the evidence path, size and SHA-256 of every file the package reads. `config.json` freezes the shared renderer's study configuration (the headline suite, the stage study, and the collection lines) with paths relative to the paper evidence. The 112 input files include paired episode outcomes and durations, policy summaries, stage rung counts, collection ledgers and per-episode frame counts, the frozen headline tables, and the Mulligan round-dataset lock (`real/results/round_datasets.json`). Run and checkpoint identifiers are release ids.

- Route Cable uses the final reviewed evaluation (all 750 episodes human-reviewed; an episode that seats only the right-most clip has no first-clip seat; `real/results/routing/headline/` in the evidence, and committed as `paper/data/real/routing/`) and the 100-episode collection increments (R1--R5, each a frame-weighted regrouping of two 50-episode collection sessions, with collector success rates for the SR adjustment).
- Marker and Nut headline and execution metrics come from `real/results/{marker,square}/headline/`. Counts are checked against the locked headline tables before plotting. Routing uses full-episode success, all 50 starts, and the final reviewed labels. The R0–R5 axis maps collection rounds R0/R2/R4/R6/R8/R9 to 100–600 training episodes per arm. Collection panels use the 100-episode increments.
- Stage plots use the stage-labeled blocks listed in the `stage` section of `config.json`. The Nut R5 headline block (Sobol seed 2026081801) has no stage labels and is not substituted, so stage-labeled subsets need not match all headline observations. Nut R5 DP uses the R5 three-arm block on Sobol seed 2026070901 (36/50 stage successes), while its critic uses the R5 critic-screen block on the same starts (36/50); the R5 two-arm block on those starts has no stage labels. Conditional insertion divides by successful grasps, not all episodes.
- CF ablation is rebuilt from the per-policy records, capped at the final tested no-CF round (Nut R4). The two arms share the R0 source actor.
- Collection success excludes CF replay episodes and counts retained, uncredited fresh failures in each arm's denominator. Burden uses saved valid frames of fresh episodes, excluding CF episodes. The human-share ratio is independently recomputed from `split_episodes.csv` and checked against each frozen ingest summary; collector-SR normalization uses the collecting policy's held-out success rate and its source, both stored in that summary. These are saved trajectory frames, not an operator wall-time estimate.
- The campaign table preserves 300 Marker pairs and 350 Nut pairs. Nut includes the R5 headline block (Sobol seed 2026081801) once per start. The R5 blocks on Sobol seed 2026070901 (two-arm, three-arm and critic screen) are excluded from the headline and campaign comparisons.
- Final Nut contrasts use the R5 headline block (50 starts, all three arms blinded and interleaved in one session) and exact McNemar, as do Marker and routing. Bootstrap and permutation seeds are explicit in `statistics.py`; 20,000 draws throughout. Routing score contrasts use seed 20260907; endpoint contrasts use 20260909. Nut uses 2026090101. Campaign intervals use the shared 20260704 seed and within-block stratification.

The shared renderers preserve paper print sizes, colors, markers, fonts and uncertainty conventions: Wilson z=1 for these real-world proportions, ±1 SE for successful durations, and one episode-bootstrap SE for throughput. Sources recorded in the figure manifest include the input lock, package code, and reused rendering/statistics modules. The plotted intervals are not the 95% paired-comparison intervals in the statistical tables.

The full-system campaign tables select the critic where evaluated and the actor otherwise. Blocks enter only when both selected treatment and baseline share the same recorded starts; unmatched actor-only blocks are not substituted for missing critic comparisons.
