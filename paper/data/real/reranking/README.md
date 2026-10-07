# Reranking-budget evidence

After `uv sync`, from any directory:

```sh
python -m paper.appendix.paper_side.reranking_check
```

The check reads the release headline CSVs
(`paper/data/real/{marker,square}/` for Marker and Nut,
`paper/data/real/routing/` for Cable), verifies the per-task,
per-round N of the ten reranked real-world settings (parsed from the source
policy names) against the frozen `release_settings.csv`, checks the two frozen
summaries below, and checks every input against the hashes in
`input_hashes.json` (keys relative to `paper/data/real`).

The two summaries are frozen analysis outputs:

- `square_r5_offline_nsweep.csv`: four Nut R5 candidate-screen critics rescore
  fixed 32-proposal banks, with 20 resamples without replacement for each N < 32.
  The metric is outcome AUROC of the maximum candidate score, not robot success.
- `square_r3_latency_summary.csv`: per-chunk reranking latency of the Nut R3
  HiL-IDQL policy. The paper uses the `all_chunks` rows (797 chunks, 20 episodes).
  `time_total_ms` runs from the start of candidate generation through the
  selection of the highest-scoring chunk (`q_min.argmax()`) in the vision IDQL
  policy (`mulligan/real/policy/vision_idql.py`).

The simulation N=1/32/128 numbers in the reranking-budget subsection are the
final-step (`is_final_step`) rows of the capacity study's `final_sr.csv`
(`sim/critic/capacity_final_sr.csv` of the paper evidence), label
`Baseline (chunk=5)`, tasks Square-Broad (round 15, 11 seeds) and Square-Narrow
(round 10, 5 seeds), as mean and standard error over seeds.
