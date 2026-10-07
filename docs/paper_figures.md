# Paper figures and tables

Every figure and table of the paper and its appendix rebuilds from frozen inputs with one
command.

```bash
uv sync
uv run python -m paper.figures                 # all 42 figure files -> paper/build/figs
uv run python -m paper.appendix.build          # appendix tables -> paper/build/tables (+ appendix figures)
uv run python -m paper.figures --check         # built bytes vs paper/figures_manifest.json
uv run python -m paper.appendix.build --check  # table bodies vs the frozen manuscript
```

`mulligan-paper-figures` is the console-script form of `python -m paper.figures`. Both commands work from
any working directory; `MULLIGAN_PAPER_BUILD` moves `paper/build`.

## Inputs

| Source | What | Where |
|---|---|---|
| Frozen paper data | simulation headline CSV, RLPD/HiL-SERL learning curves, Route Cable snapshot, real headline tables, reranking, sim significance and training-compute inputs, F3/F4 pins | `paper/data/{real,sim,online_rl}` (git) |
| Paper evidence | the appendix packages' pinned inputs (`paper/appendix/<pkg>/inputs.json`, value_learning `sources.json`/`dataset.json`) and the reset-range photo composites, organized by role (`real/`, `sim/`, `assets/`; plus `fixtures/` for the RLPD parity test) | `$MULLIGAN_PAPER_EVIDENCE/<path>` or HF `mulligan/paper-evidence` |
| Released datasets | the three Fig. 3 episodes; the round-0 demonstration durations | `mulligan/real-*` and `mulligan/sim-*` HF datasets at their `release/revisions.json` pins |

Every evidence file is checked against its sha256 and size before use and copied into `paper/build/cache/<package>/raw/<path>`.
Run and checkpoint identifiers in the evidence and in `paper/data` are release ids (`mulligan/<repo>[/<subfolder>]` for
released checkpoints and datasets, `unreleased/<domain>/<kind>-NNNN` otherwise); the layout is described in
`paper/appendix/README.md`.
The evidence is read from `mulligan/paper-evidence` at `HF_REVISION` (`paper/appendix/artifacts.py`, the
`release-1` commit); `MULLIGAN_PAPER_EVIDENCE` points the builders at a local mirror with the same layout instead.

Offline and without a mirror, `python -m paper.figures` stops at the first entry that needs the evidence.
`python -m paper.figures --keep-going` builds every entry it can, prints a summary with the reason for each
entry it could not build, and exits non-zero. Five of the 32 registry entries (11 of the 42 files) build
without the evidence: `sim_state_rlpd_vs_mulligan_compact` (Fig. 2), `task_sequences`
(Fig. 3, reads released videos from Hugging Face), `real_world_square_d2_reset_card` (Fig. 4, left),
`initial_states` (the App. C photos, also from released videos) and `overview_eval_protocol`. Of the
appendix tables, `sim_significance_table.tex` (`python -m paper.stats.sim_welch_table`),
`divl_frozen_actor_table.tex` (`python -m paper.appendix.paper_side.divl_comparison`) and `training_compute.tex`
(`python -m paper.appendix.paper_side.training_compute`) build without it; `python -m paper.appendix.build` needs
it. `--list` shows `evidence: yes/no` per entry. The flag is recorded in `paper/figures_manifest.json` by
`--update-manifest` from the evidence files each build actually read, and every build fails if an entry's reads
disagree with its recorded flag.

System requirements beyond the Python environment: Chrome or Chromium for the teaser (`$MULLIGAN_CHROME` or
`google-chrome`/`chromium` on `PATH`), Ghostscript (`gs`) for its crop, and poppler (`pdftoppm`) for the visual
comparison. A missing Chrome is a hard error; `python -m paper.figures --exclude teaser` builds everything else.

## Registry

`python -m paper.figures --list` prints the entries, each with whether it needs the paper evidence. Main text:

| Figure | Files | Builder |
|---|---|---|
| 1 teaser | `overview_teaser.pdf` | `paper/teaser/build_teaser.py` (HTML, Chrome); bucket counts from `paper.appendix.productivity.teaser` |
| 2 RLPD vs Mulligan | `sim_state_rlpd_vs_mulligan_compact.pdf` | `paper.plotting.state_rlpd_learning_curves` |
| 3 task sequences | `task_sequence_{marker,nut,cable}.jpg` | `paper.fig_tasks` (HF videos) |
| 4 reset card and ranges | `real_world_square_d2_reset_card.pdf`, `real_world_{square,marker}_d2_init_ranges_side1.png` | `paper.fig_reset_ranges` |
| 5 headline | `headline_real_sim.pdf` | `paper.plotting.sim_paper_headline_combined` |
| 6 throughput | `real_world_success_throughput_headline.pdf` | `paper.appendix.productivity.plot` |
| 7 ablations | `ablations_sim_real.pdf` | `paper.appendix.component_panels.plot` |
| 8 collection success | `real_world_dagger_collection_success_square.pdf` | `paper.appendix.productivity.plot` |
| 9 burden | `collection_burden_real_sim.pdf` | `paper.appendix.component_panels.plot` |

The appendix figures are built by the packages under `paper/appendix/` (see `paper/appendix/README.md`); the
authored illustrations (funnel, operator cards, state evolution) are restored byte-for-byte from the evidence.
The appendix catalog is `paper/appendix/catalog.json`; `paper/reference/manuscript.json` is the inventory of the
frozen manuscript that both checks compare against.

Numbers quoted in the text are printed by `paper/stats/*` (`python -m paper.stats.<module>`) and asserted in
`tests/paper/test_paper_stats.py`.

## Reproducibility notes

- The matplotlib figures, the restored illustrations and the reset card rebuild byte-identically to the manuscript's.
- The two reset-range overlays differ in about 0.01% of pixels by at most 2/255 (image resampling on this platform).
- The task-sequence JPGs differ by about 1/255 on average: the released videos are the same bytes, but FFmpeg's SIMD
  colour conversion on x86 differs from the platform the manuscript figures were rendered on.
- Because of these two, `--check` does not hash the task-sequence JPGs and the reset-range PNGs (their bytes depend on
  the platform); it compares their pixels with `paper/reference/figs` instead and fails above a mean absolute
  difference of 2/255 or more than 0.1% of pixels moving by more than 16/255 (`RASTER_TOLERANCE` in
  `paper/figures.py`, the rasterizer of `paper/compare_reference.py`).
- The teaser is printed by the local Chrome; fonts and the JPEG encoding of the embedded photo differ between
  Chrome builds, so its bytes are never stable and `--check` verifies its presence only. Ghostscript's crop
  writes no file names, dates or document ids into the PDF.

`python -m paper.compare_reference --out DIR` rasterizes every rebuilt figure and its `paper/reference/figs` copy at
150 dpi, writes side-by-side PNGs and prints the pixel-difference table.

After any change to code or data that a figure depends on, `python -m paper.figures --update-manifest` rebuilds and
records the new hashes; review the figures before committing the manifest.
