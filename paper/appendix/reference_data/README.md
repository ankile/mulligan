# Appendix reference data

This package owns the existing tables and authored assets in `baselines.tex`,
`data_collection.tex`, `prior_data.tex`, `prior_reset_ranges.tex`, `tasks.tex`,
`hyperparameters.tex`, `eval_protocol.tex`, `implementation.tex`, `results.tex`,
and `related_work.tex`. The complete file, table and asset inventory is in
`inventory.json`.

## Reproduce

After `uv sync`:

```bash
python -m paper.appendix.reference_data.prepare --restore-assets
python -m paper.appendix.reference_data.prepare --check
```

The command writes the generated table body `reference_recap.tex` to `paper/build/tables/` and the authored table
bodies to `paper/build/tables/authored/<label>.tex`; with `--check`, every generated
body must equal the frozen manuscript's copy in `paper/reference/tables/`.
`--restore-assets` also copies the four authored illustration exports to
`paper/build/figs/`. The paper includes the table bodies inside its authored
captions and table environments.

`inputs.json` pins every input by evidence path, SHA-256 and byte count. Inputs are fetched from the paper evidence
(`$MULLIGAN_PAPER_EVIDENCE` or the HF dataset) into
`paper/build/cache/reference_data/raw/`; a hash mismatch fails. The cache is
disposable.

## Empirical tables

`reference_recap.tex` recomputes three comparisons from the per-seed records in
`recap/runs.csv`: the locked held-out grid readout (8,000 starts on Square-Narrow,
30,000 on Square-Broad) of each seed's final checkpoint. Each cohort requires exactly
seeds 1–5 for both learners and completed grid readouts. Means and standard errors use training
seeds as the replication unit. The difference is RECAP minus DP+IQL within seed;
its 95% interval is the mean difference plus/minus Student-t(4) times its standard
error. Every mean and standard error is checked against the archived independent
summary. Rates are already in percent. This table uses the six-step RECAP
campaign and its scalar-IQL (DP+IQL) controls, not the DIVL heads. Run and
checkpoint identifiers (release ids) remain in the source records. Reproducing the table does not rerun the simulator.

The **stage-label census** of 36 per-repository records is checked and
summarized in `analysis.json`; the paper prints no coverage table. Episode counts are checked against the
independently preserved `episode_counts.json`. Each repository must occur once,
label intersections and disjoint counts must add up, and the five-line totals
must agree. Percentages are computed from summed counts, not averaged from repository
percentages. The final-round training-set ledger is built by the
[data_ledger](../data_ledger/README.md) package, not from this census.

The census is not the entire collection campaign. Many included repositories
are evaluation or diagnostic datasets, and coverage includes partially labeled
repositories; the paper's text and headings give its scope. The Route
Cable census covers the collection and evaluation datasets of the velocity-action
lineage and excludes the final 15-arm evaluation (`mulligan/real-routing-d2-r00-r05-eval`). `paper/build/cache/reference_data/data/analysis.json` records the
computed totals and paired results.

## Authored tables and assets

Fifteen tables are authored specifications or transcriptions: the two prior-work
comparisons (prior data, reset ranges), the task summary, initial-state ranges and
scoring tables, the two stage-ladder tables, six hyperparameter tables, and
transcriptions of the per-round counterfactual counts and the Cable collection
increments. Their bodies are the frozen manuscript's
(`paper/reference/tables/authored/`), each pinned by SHA-256 in `inventory.json`, and
`python -m paper.appendix.build --check` requires a pinned body for every authored
table of the catalog; the manuscript sources themselves are not released. Their captions, surrounding prose and
bibliography remain authored in the paper. The robot-time cells of the "Ours" rows of the prior-data table are checked
against the [data_ledger](../data_ledger/README.md) producer.

The trajectory funnel and three operator cards are authored illustration exports,
not result charts. `materialize_assets()` restores their approved PDF/PNG bytes
without conversion, replotting, or image resampling (also run by the
`restored_reference_data` entry of `paper.figures`). `python -m paper.appendix.build --check`
requires the restored files to equal the archived exports. Their appearance is
kept as an illustration of the operator interface, rather than restyled as a
data plot.
