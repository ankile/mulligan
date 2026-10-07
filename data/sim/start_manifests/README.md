# Locked simulation round inputs

The locked start lists and manifests of the simulation rounds R0-R3 (subdirectories `blind_inputs`,
`init_states`, `manifests`, `rollout_inputs`). Task, arm and file names are the
release names (tasks `square_narrow`, `square_broad`; arms `baseline_uniform`,
`sobol`, `mulligan`); the start states themselves are exactly the collected ones. They
are hash-locked: never edit them. `INDEX.csv` lists every file with its
path, sha256, size and, for the start lists, `states_sha256` (the sha256 of the
canonical JSON of the `states` array, which does not depend on the names around
it; for the `.npy` files it equals `sha256`). `tests/sim/test_start_manifests.py`
checks the tree against it.

Layout: `<task>/rNN/<source subdir>/<file name>`, with
`square_narrow` = Square-Narrow (`NutAssemblySquare`, source prefix `square_narrow_`)
and `square_broad` = Square-Broad (`Square_D1`, source prefix `square_broad_`).
The source subdirectory is kept because a few file names occur in two of them with
different contents (for example `square_narrow_baseline_uniform_r2_policy_rollout.json`
in both `init_states/` and `rollout_inputs/`). `rNN` is the collection round the
file feeds; evidence from round-0 policies that was used to design round 1
(`*_r0_eval_sobol_100_199*`) is filed under `r01`.

## Collections

| Task | Round | Collector input (`collection_starts`) | Released raw collection |
|---|---|---|---|
| Square-Narrow | R0 | `r00/blind_inputs/square_narrow_r0_mixed.json` | `mulligan/sim-square-narrow-c00-teleop-mixed` |
| Square-Narrow | R1 | `r01/blind_inputs/square_narrow_r1.json` | `mulligan/sim-square-narrow-c01-dagger-mixed` |
| Square-Narrow | R2 | `r02/blind_inputs/square_narrow_r2.json` | `mulligan/sim-square-narrow-c02-dagger-mixed` |
| Square-Narrow | R3 | `r03/blind_inputs/square_narrow_r3.json` | `mulligan/sim-square-narrow-c03-dagger-mixed` |
| Square-Broad | R0 | `r00/blind_inputs/square_broad_r0_mixed.json` | `mulligan/sim-square-broad-c00-teleop-mixed` |
| Square-Broad | R1 | `r01/blind_inputs/square_broad_r1.json` | `mulligan/sim-square-broad-c01-dagger-mixed` |
| Square-Broad | R2 | `r02/blind_inputs/square_broad_r2.json` | `mulligan/sim-square-broad-c02-dagger-mixed` |
| Square-Broad | R3 | `r03/blind_inputs/square_broad_r3.json` | `mulligan/sim-square-broad-c03-dagger-mixed` |

The collector replays the list with `SquareListSampler` / `SquareD1ListSampler`
(`mulligan/sampling/sobol.py`). The matching `*_manifest.json` maps each list entry
to its blinded source arm and is what the episode splitter reads.

## Roles (`INDEX.csv` column `role`)

| Role | Meaning |
|---|---|
| `collection_starts` | the interleaved start list given to the collector |
| `collection_starts_qpos` | the same starts as object qpos (`.npy`) |
| `blind_manifest` | per-start source arm and index for the blinded list |
| `arm_starts` | the start list of one arm (baseline uniform, Sobol, Mulligan) |
| `collection_manifest` | round recipe: arms, protocol quotas, source files, split targets |
| `points_manifest` | per-start provenance of a start-design arm (promoted failures, FPS fill) |
| `sampler_candidate` | the locked start-design candidate a `points_manifest` names as `design_source` |
| `rollout_input` / `rollout_input_qpos` | starts for the autonomous prep rollouts of the previous round's policies |
| `rollout_manifest` | which prep rollouts ran from which inputs |
| `rollout_audit` | per-start outcomes of those prep rollouts (the start-design evidence) |
| `eval_points` | diagnostic eval starts on the next slice of the Sobol stream |
| `coverage_reference` | Sobol slice used as the coverage reference in the Broad start design |

Round-0 arm lists are reproducible from library code; `tests/sim/test_sobol.py`
regenerates all four byte-for-byte (see the module docstring of
`mulligan/sampling/sobol.py` for the seeds).

## Hash pins

`states_sha256` is the stable identity of a start list. `pinned_by` names the other record that
cites the file: the autonomous-baseline protocol (`autonomous_baselines.yaml`), which starts the
autonomous arms from the baseline-uniform list of each round, or the Broad R3 list's
`_meta.locked_sampler`.

## Not included

- Eval grids (the 8k/30k valid-Sobol grids and the R0 equal-tile grid): see
  `configs/sim/README.md` (Evaluation grids).
- Start designs that were not collected.
- Collection ledgers (`ledgers/*.jsonl`). The released mixed collections do not carry them; each released
  split view records the episodes it took in `meta/protocol_split_manifest.jsonl`.

The JSON files keep the dataset and model ids they were written with (`source_artifact`); the released
dataset names are in the table above.
