# Configs

Every setting a paper run used, one place per kind:

| Kind | Where | Source | README |
|---|---|---|---|
| Simulation training recipes (IDQL agent + DIVL head per cell; the RLPD and HiL-SERL baselines), evaluation grids, round collection and split specs | `sim/recipes.json` | checked by `tests/sim/test_recipes.py` | [sim/README.md](sim/README.md) |
| Simulation start design (Alg. 2), R1-R3 | `sim/<task>/rNN_sampler.yaml` | by hand | [sim/README.md](sim/README.md#start-design-configs) |
| Real-robot training, one config per released checkpoint | `real/<task>/rNN_<arm>_dp.yaml`, `real/<task>/rNN_critic*.yaml` | the checkpoint's training run (do not edit) | [real/README.md](real/README.md) |
| Real-robot evaluation sessions, one config per round | `real/<task>/rNN_eval.yaml` | derived from the round datasets' `meta/round_dataset.json` (do not edit) | [real/README.md](real/README.md) |
| Real-robot start design (Alg. 2), Marker and Nut R1-R5 | `real/<task>/rNN_sampler.yaml` | by hand | [real/README.md](real/README.md#start-design) |
| Robot station templates (camera layout, station identity) | `real/station.example.yaml`, `real/station.env.example` | by hand | [docs/station.md](../docs/station.md) |

Shared conventions:

- Task folders use the code's task names: `square_narrow`, `square_broad` (simulation) and `marker_d2`,
  `square_d2`, `routing_d2` (real). The paper calls them Square-Narrow, Square-Broad, Insert Marker,
  Thread Nut and Route Cable.
- `rNN` is a round, two digits (`r00`-`r05`). `cNN` is a collection increment and `bNN` an evaluation
  session, as in the Hugging Face repo names ([docs/data.md](../docs/data.md#names)).
- Datasets and checkpoints are pinned to a commit of a `mulligan/*` Hugging Face repo, equal to
  `release/revisions.json` (`tests/release/test_revision_consistency.py`).
- Derived files start with `# Derived from <source> ...`; tests check them against that source (do not
  edit them by hand).
- Every config names the paper results it produces: `paper` in each sim recipe and each real trainer
  and eval config, a `Paper:` note in the header comment of the start-design and baseline configs. Figure and
  appendix numbers are the manuscript's; [docs/reproduce.md](../docs/reproduce.md) has the commands
  per result.
