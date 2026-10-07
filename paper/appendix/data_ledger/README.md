# data_ledger

Final-round training-set ledger (`tab:appendix-data-ledger`, `paper/build/tables/data_ledger.tex`), and the per-round training-data composition of the two real-world collection campaigns (`tab:appendix-data-composition`, `paper/build/tables/data_composition.tex`; figure `real_world_data_composition.pdf`, `plot.py`).

Inputs pinned by `inputs.json`: round-0 teleoperation statistics and the R1--R5 split ledgers of Insert Marker and Thread Nut (the HG-DAgger+Mulligan with-CF view), the line human-frame share tables as a cross-check, the Route Cable actor provenance (`real/training/routing/runs.json`, R0 teleop frames and R5 retained episodes), the 100-episode increment frame counts and the five increment split ledgers, the recorded lengths of the Mulligan arm's 100 round-0 demonstrations (`real/collection/routing/r0_episode_lengths.csv`), and the paper rollout accounting. The simulation rows read the frame-level parquet of the public training repos of the HG-DAgger+Mulligan actor (`mulligan/sim-square-{narrow,broad}-c00-teleop-sobol` and `-c0{1,2,3}-dagger-mulligan`) at their `release/revisions.json` pins, and the critic sets come from `release/models.json` (each deployed critic's training datasets) and `release/datasets.json` (their recorded episode and frame totals).

The few counts that no pinned table records (the Route Cable R5 parent collection size, the simulation round-0 demonstrations per arm, and the Insert Marker and Route Cable critic sets that the release totals are checked against) are documented constants in `tracker_values.json`: each entry holds the value and a description of what it counts.

`|D_h|` counts the actor's episodes (round-0 demonstrations plus every collected episode with human frames; Route Cable's count is the post-filter retained episodes). `D_h` frames are human-controlled valid frames and `D_all` frames the valid frames of the same episodes; the critic set is the recorded size of the deployed final-round critic's training repos (Thread Nut: the R5 headline critic `mulligan/real-square-d2-r05-mulligan-idql-critic`, whose set adds the two R5 rollout repos `mulligan/real-square-d2-r05-eval-b02-mulligan-{dp,idql}-policy-rollouts` to the set of the R5 screen critics). Collected episodes sum the round-0 session and the parent collections of both arms (Route Cable adds its R5 parent collection); held-out evaluation is not counted.

The composition splits each round's valid training frames (HG-DAgger no-CF view, Mulligan with-CF view) into demonstrations (round-0 demonstrations and the human frames of counterfactual replays), corrections (human frames of policy-start episodes), and autonomous frames (every policy-controlled frame, including those inside mixed-control replays). Route Cable rounds are the five 100-episode increments (`real/collection/routing/r{1..5}/split_episodes.csv`) and must reproduce `increment_frame_counts.csv`; the Mulligan totals must equal the ledger's `D_h` and `D_all` frames.

```sh
python -m paper.appendix.data_ledger.prepare           # write both tables
python -m paper.appendix.data_ledger.prepare --check   # require the manuscript's table bodies
python -m paper.figures --only real_world_data_composition
```
