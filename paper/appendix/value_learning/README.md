# Real-world value-learning appendix figures

These figures reproduce the Marker D2 offline-selection studies. They read only
their pinned evidence, not the headline tables or the headline robot evaluations.

## Reproduce

After `uv sync`:

```bash
python -m paper.appendix.value_learning.prepare
python -m paper.figures --only real_world_value_fqe real_world_value_training
python -m paper.figures --check --only real_world_value_fqe real_world_value_training
```

The first command fetches the pinned producer files and rebuilds the tidy
extract in `paper/build/cache/value_learning/data/`. For plot-only
reproduction, the second command fetches the five pinned derived CSV files
directly from the paper evidence (`real/value_learning/tables/`). Both paths work with an empty build
cache. `sources.json` pins every raw producer file by SHA-256 and byte size;
`dataset.json` pins the five derived tables, which the extract must reproduce
byte for byte, and records the hash of the `sources.json` it was derived from.
Changed or missing inputs fail loudly.

## Files and evidence

- `eval/<policy>/summary.json`: 16 original corrected-FQE producer outputs,
  including all three evaluator seeds, training traces, robot successes and
  attempt counts, and the policy-bank and initial-value tensor hashes.
- `training/<step>/`: nine checkpoint batteries, each with its original
  `summary.json`, `candidate_probe.csv`, and `per_episode_features.csv`.
- `tables/fqe_policies.csv`: one recorded robot observation per policy, with
  successes, attempts, family, round, and original bank hashes.
- `tables/fqe_seeds.csv`: 48 evaluator-seed return estimates.
- `tables/training_probes.csv` and `tables/training_episodes.csv`: 450 rows each,
  nine checkpoints by the same 50 manifest states. Both diagnostics use the
  R4 baseline block excluded from critic training, containing 23 successes.
- `tables/training_checkpoints.csv`: critic checkpoint ids (release ids) and the
  producer AUROC used to cross-check the reanalysis.
- `analysis.json` (written by the plot builder): recomputed correlations, fit coefficients, checkpoint
  statistics, and the paired endpoint AUROC bootstrap interval.

Paths above are relative to `real/value_learning/` of the paper evidence; the build
cache `paper/build/cache/value_learning/` is disposable. Only code, these
instructions, and the small input locks are in git. The paper driver
registers the two PDFs, uses `mulligan.plotting.paper` at full text width, and records
their hashes in the shared paper-figure manifest.

## Statistical definitions

FQE uses gamma 0.995, timeout bootstrapping, 40,000 updates, and the final
checkpoint for each of three evaluator seeds. All policies are scored on 150
R5 reset images corresponding to 50 manifest starts. Robot rates come from
each policy's own round, not a common prospective test. The R5 IQL point is the
31/50 screening critic (`mulligan/real-marker-d2-r05-mulligan-idql-critic-screen-b02`),
not the selected 38/50 critic.

FQE markers show evaluator means, small dots show individual seeds, and vertical
whiskers show one standard error over the three evaluators. Horizontal whiskers
are Wilson 95% intervals from recorded robot counts, without adjustment for
repeated states in pooled sessions. These error bars measure different sources
of uncertainty. They do not cover state-distribution shift. Dashed ordinary
least-squares fits and Spearman correlations are descriptive; no regression
p-values or fit confidence bands treat the 16 related policies as independent
experiments. The nine-policy subset in the original report excludes all six
baseline policies and the Ours R0 actor; its rho is reported in `analysis.json`.

The training figure recomputes outcome AUROC from candidate-maximum-Q scores.
Its pointwise 95% intervals use 5,000 stratified bootstrap resamples of the 23
successful and 27 failed episodes, with the same resamples at every checkpoint.
The initial-value panel averages only those same 23 successful episodes and
shows one SE across episodes. It is conditional on success and is not a
calibration estimate for the whole reset distribution. All checkpoint intervals
are conditional on this critic recipe and seed, not training-seed uncertainty.

The paper uses the same 50-episode block for both panels; averaging the initial
values over all 250 episodes instead gives a 475k success-conditioned
mean of 0.069 rather than 0.082. The extractor retains all episodes so this
choice is auditable.
Alternative proposed actions have no observed counterfactual outcomes. Therefore
candidate-score AUROC is outcome discrimination across recorded states, not
measured within-state action quality or robot success.
