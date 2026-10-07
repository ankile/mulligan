# Main-text component panels

These two main-text figures read the frozen appendix evidence; they are outside the
appendix-only catalog.

After `uv sync`:

```bash
python -m paper.figures --only ablations_sim_real collection_burden_real_sim
python -m paper.figures --check --only ablations_sim_real collection_burden_real_sim
```

The build reads only `inputs.json`,
fetches the pinned files from the paper evidence, and verifies their hashes. It
writes tidy source tables under `paper/build/cache/component_panels/data/`
(`python -m paper.appendix.component_panels.prepare` writes only these), proofs
under `paper/build/proofs/`, and PDFs to `paper/build/figs/` through the shared
paper renderer.

## Evidence and definitions

- Simulation success uses five seeds per cell from the frozen DIVL N32 actor
  campaign, exactly the cells in the appendix sampling table. These are later
  value-function evaluations on fixed actors trained from the HiL collection
  datasets, not newly collected datasets. Bars show means, seed dots, and 95%
  Student-t intervals. The explicitly broken axis retains the paper's zoom.
- The CF panels use the pinned, human-reviewed pooled arm tables from
  `real_results`. Thread Nut stops at R4; Insert Marker stops at R2, the last
  no-CF evaluations. R0 is the shared source actor, shown once. Whiskers are
  Wilson one-standard-error intervals, z=1. Nut R3 with-CF pools 150 episodes;
  Marker R2 with-CF pools 100; other displayed points have 50. These are
  descriptive pooled comparisons, not a paired significance test.
- Real burden uses saved valid-frame human-control shares in fresh collection
  episodes, excluding CF replay. Failure-adjusted shares divide by the collector
  failure rate recorded in each frozen summary.
- Simulation burden uses the fraction of the collected R1 episodes with at least
  one intervention. Its failure divisors come from the scalar-IDQL R0 collector
  grid, **not** the frozen-actor DIVL evaluations. Both metrics
  are indexed to their respective uniform baseline. Wilson 95% bounds propagate
  the intervention incidence only; failure divisors and baseline scaling factors
  are treated as fixed.
- Raw and adjusted burden are different descriptive quantities. The adjusted
  ratio is not an estimate of failure incidence on the sampled initial states,
  nor an accounting of all operator time.

## Evidence provenance

Most inputs are shared with the `simulation` and `real_results` packages (the
frozen-actor results, the line headline tables, the Nut collection summaries);
the simulation R1 dataset statistics and the scalar-IQL R0 predecessor grid are
simulation collection records.
