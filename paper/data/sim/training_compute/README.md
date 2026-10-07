# Training-run compute

Source of the training-compute table (`tab:training-compute`, App. G.7; written as
`paper/build/tables/training_compute.tex`). The records cover 386 distinct W&B
runs: 170 actor/scalar-critic and 170 frozen-actor DIVL fits for the 34
human-in-the-loop simulation cells (five seeds each), and the 36 actors and 10
critics behind the released real-world curves, resolved from their model
artifacts with W&B `logged_by()`.

- `records.json`: the frozen W&B extract. Runs and checkpoints are named by release
  ids (`run`, `checkpoint`).
- `runs.csv` and `summary.json` (including the per-GPU-model breakdown) are frozen
  derived files; the builder re-derives both from `records.json` and requires
  byte identity, then writes the table.

Hours are W&B `_runtime` times the recorded GPU count (one for every run; run
`mulligan/sim-square-broad-r01-baseline-straddled-auto-success-idql/seed-3` has no recorded GPU model). This is logged run time including setup and
in-run evaluation, not scheduler-billed time. Separate evaluations, feature caches,
failed runs and autonomous baselines are excluded.

Rebuild offline (after `uv sync`, from any directory; `python -m paper.appendix.build`
runs the same builder):

```sh
python -m paper.appendix.paper_side.training_compute   # --out-dir DIR, default paper/build/tables
```
