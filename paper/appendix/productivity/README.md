# Real-world productivity figures and teaser

The appendix collection-success plot and main-text throughput figure show
Insert Marker, Thread Nut, and Route Cable. The main text uses a Thread Nut-only
collection-success panel beside its burden figure. After `uv sync`:

```bash
python -m paper.figures --only real_world_dagger_collection_success real_world_dagger_collection_success_square real_world_success_throughput_headline
python -m paper.figures --check --only real_world_dagger_collection_success real_world_dagger_collection_success_square real_world_success_throughput_headline
python -m paper.teaser.build_teaser
```

The full three-task collection-success plot is in the appendix build catalog;
its Nut-only main-text companion and throughput plot are outside that catalog.
All plotters use the shared paper renderer and print-size conventions. The
`*_mixed` modules (`paper/plotting/collection_success_mixed.py`,
`throughput_mixed.py`) draw real-world tasks only.

## Collection success

`inputs.json` pins the three collection tables and three collector-evaluation
tables (`real/collection/{marker,square,routing}/` of the paper evidence). Marker and Nut tables are
rebuilt from the frozen per-policy evidence in
`paper/appendix/real_results`. Cable uses the 100-episode increments: R1--R4 each
pool two 50-episode sessions of the velocity-action collectors and R5 is the
100-episode session collected by the R4 checkpoint. Its collector
series is episode full success from the held-out evaluations of the collecting
checkpoints, matching collection's episode-completion metric; the clip-progress
headline table is not used.

Solid lines count successful fresh collection episodes, excluding CF replay.
Dashed lines use the actual preceding-round collector's held-out episode success,
not a concurrently evaluated replacement policy. Marker and Nut R4/R5 use the
R3/R4 reranked collectors; every Cable round was collected with DP. Pooled
counts and the source arm/evaluation round survive in
`paper/build/cache/productivity/data/collection_points.csv`
(`python -m paper.appendix.productivity.prepare` writes only this table).
Whiskers are Wilson one-standard-error intervals, z=1. This comparison is
observational: collection and held-out evaluation have different initial states.

## Throughput

The raw episode outcomes, durations, and serialized protocol come from
`paper/appendix/real_results/{inputs,config}.json`, whose exact bytes and shared
code are included in the figure's source hashes. Reproduction calls its
frozen-data preparer.
`paper/build/cache/productivity/data/throughput_points.csv` records every
plotted point, source arm, completed-task count, and error bar.

Throughput is completed full tasks divided by total robot time across successful
and failed episodes. Cable uses the separate long-horizon `routing_d2` evaluation,
with both clips required for a completed task. It is not clip-progress throughput.
Ours follows DP until the deployed BoN takeover, with Cable's takeover at R3.
Whiskers are bootstrap standard errors.

## Teaser

`teaser.py` regenerates the teaser's bucket counts from the frozen simulation
extractor: the 80 recovered per-state success counts drawn by
`paper/teaser/assets/bucket_counts.js` (`python -m paper.appendix.productivity.teaser`
prints the file body). Each state has 50 rollouts. There are 3438 successes in
4000 rollouts (85.95%); the hardest 20 states, 25% of 80, contain 533 of 562
failures (94.84%). These counts are digitized from an archived raster. The right-hand state-selection
illustration remains schematic.

The HTML (`paper/teaser/teaser.html`) derives its 80 bar heights,
hardest-quarter boundary, overall line, and rounded 86%/25%/95% labels from
those counts. `python -m paper.teaser.build_teaser` (also the `overview_teaser`
entry of `paper.figures`) regenerates the counts into a staged copy, requires
the committed `bucket_counts.js` to match, prints the page with headless Chrome
or Chromium (`$MULLIGAN_CHROME` or on `PATH`), and crops it with Ghostscript
(`gs`). The Chrome output is renderer-dependent, so `--check` verifies its
presence but not its hash.

## Evidence provenance

The Marker/Nut tables and the collection tables were written by the `real_results`
preparer from its frozen config, and the Cable collector evaluation comes from
the policy summaries of the Cable collection sessions. The
real-results and simulation locks pin the per-episode and raster provenance.
