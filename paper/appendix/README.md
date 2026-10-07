# Appendix data and figures

The appendix has one build entry point. Its packages read frozen, hash-pinned
paper evidence.

After `uv sync`, from any directory:

```bash
python -m paper.appendix.build
python -m paper.appendix.build --check
```

The build regenerates 19 native PDF figures (18 plots and the evaluation-protocol
schematic), renders the five initial-state photos from the released round-0 videos,
restores five authored illustrations into `paper/build/figs`, and writes the generated
table bodies to `paper/build/tables` (authored table bodies to `paper/build/tables/authored`).
The coverage check requires the catalog to account for all 29 included
figures/assets and all 31 table labels (16 generated, 15 authored) of the frozen manuscript
(`paper/reference/manuscript.json`), so adding an appendix item without
declaring its source fails.

`--check` additionally requires every generated table body to equal the frozen
manuscript's copy in `paper/reference/tables`, every authored table to have a
sha256-pinned body (`reference_data/inventory.json`), every restored illustration to
equal its archived export, and the appendix figures to match
`paper/figures_manifest.json`. Use the ordinary paper driver to rebuild or check
one plot:

```bash
python -m paper.figures --only real_world_value_fqe
python -m paper.figures --check --only real_world_value_fqe
```

The appendix's checks are scoped to its catalog. `python -m paper.figures`
(optionally `--exclude teaser` on a machine without Chrome) builds and checks
the full paper registry, including main-text figures.

## Package ownership

| Package | Existing appendix material | Evidence boundary |
|---|---|---|
| [value_learning](value_learning/README.md) | Real-world offline value diagnostics | Original FQE producer outputs, evaluator seeds, and per-episode training probes |
| [real_results](real_results/README.md) | Real-world results, paired tests, substages, collection and effort | Frozen per-state outcomes, stage batteries, collection ledgers, and explicitly selected evaluation sessions |
| [simulation](simulation/README.md) | Simulation efficiency, sampling, critic ablations and DIVL/REDQ | Archived seed summaries, run records, reset states, and a documented raster digitization |
| [reference_data](reference_data/README.md) | RECAP, authored tables and collection illustrations | Raw seed records; separately identified authored specifications and cited literature transcriptions |
| [productivity](productivity/README.md) | All-task collection success, with compact main-text companion | Frozen fresh-episode counts and actual collector held-out evaluations |
| [data_ledger](data_ledger/README.md) | Final-round training-set ledger (episodes and frames per task) and per-round real-world data composition | Pinned collection ledgers, actor provenance, rollout accounting, and documented constants |
| [hilserl](hilserl/README.md) | HiL-SERL in simulation: per-session eval curves vs. RLPD and operator-cost table | Per-milestone eval CSVs and session totals, one seed per session |
| [critic_data](critic_data/README.md) | Critic training-data composition: human-only vs all-data critic behind the same frozen actor | Per-seed pairs and the Student-t summary, five seeds per task |

Three generated tables come from builders outside these packages, reading
frozen data under `paper/data/sim`: the simulation significance table
(`paper/stats/sim_welch_table.py`), the DIVL frozen-actor table
(`paper/appendix/paper_side/divl_comparison.py`), and the training-compute
table (`paper/appendix/paper_side/training_compute.py`).

`catalog.json` is the complete figure/table inventory. Package inventories and
READMEs describe individual sources and limitations.

## Files and storage

Each package pins the evidence files it reads in `inputs.json`: path, SHA-256
and byte size. The path is relative to the paper evidence, read from a local
mirror directory (`$MULLIGAN_PAPER_EVIDENCE`) or from the HF dataset
`mulligan/paper-evidence` at the revision pinned in `artifacts.py` (`HF_REVISION`).
`artifacts.py` is the shared fetch-and-verify implementation: every file is
checked against its lock and copied into `paper/build/cache/<package>/`
(`raw/<path>` for inputs, `data/` for derived tables and analysis). A changed
or corrupt cached file fails rather than being overwritten. The value-learning
package keeps its two-stage locks: `sources.json` pins the raw producer
outputs and `dataset.json` pins the derived tables that `prepare.extract()`
must reproduce byte for byte.

The evidence is organized by role, not by package or capture date; a file read
by several packages is stored once:

```text
real/results/{marker,square,routing}/   eval outcomes, stage batteries, headline tables
real/results/round_datasets.json        the round-dataset lock (release/round-datasets.json)
real/collection/                        split ledgers, collection success
real/training/, real/value_learning/    training-run records, offline value studies
sim/{results,critic,sampling,efficiency,collection}/
assets/{illustrations,reset_ranges}/    restored exports, reset composites
fixtures/rlpd_sac_parity.npz            RLPD agent parity fixture (tests/baselines)
manifest.json                           SHA-256 and size of every file
```

Run and checkpoint identifiers in the evidence are release ids: a released
checkpoint or dataset by its `mulligan/<repo>` id, anything unreleased as
`unreleased/<domain>/<kind>-NNNN`. The manuscript's LaTeX sources are not in the evidence: the authored table
bodies the build checks are the frozen copies in `paper/reference/tables/authored`.

`paper/build/` is disposable. Code, locks, and these instructions are in git.
Generated figures and tables are not committed; the frozen manuscript's table
bodies (`paper/reference/tables`) and the figure manifest are the references
they are checked against. The evidence is frozen: a changed input needs a new
`mulligan/paper-evidence` revision and new locks.

## Reproduction and scientific checks

A fresh checkout needs the locked Python environment (`uv sync`); the paper
evidence is downloaded from the Hub on first use (or read from a local mirror via
`$MULLIGAN_PAPER_EVIDENCE`). Deleting `paper/build/cache`
exercises the same cold-cache path as a fresh checkout. Run the build and check
commands above.

The build uses `mulligan.plotting.paper` and the standard paper registry, with shared
colors, vocabulary, print dimensions, font embedding, and uncertainty styles.
The registry records hashes of input manifests and actual rendering code.
Authored SVG/PDF illustrations are restored from approved exports; they are not
passed through a new graphics exporter. Under `--check`, a restored asset that differs from its
archived export fails.

The main-text component and burden panels also consume frozen appendix evidence.
Their separate [component-panels package](component_panels/README.md) rebuilds
`ablations_sim_real` and `collection_burden_real_sim`, including the side-by-side
Thread Nut and Insert Marker CF comparison. They are outside the appendix-only
catalog and build command above.

The full appendix collection-success figure, compact main-text Nut companion,
all-real throughput figure, and 80-state teaser data export are documented in
[productivity](productivity/README.md).
