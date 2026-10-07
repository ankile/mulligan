# Arena data

The data files of the public [Policy Arena](https://arena.mulligan.page) and its
[release catalog](https://arena.mulligan.page/data/catalog.html). The datasets they describe are public in the
[mulligan](https://huggingface.co/mulligan) organization on the Hugging Face Hub, tagged `release-1`.

| File | Contents |
|---|---|
| `release.json` | The Arena snapshot: per task, the released datasets, policies, evaluation sessions and results the Arena pages show |
| `ui.json` | Arena display data: datasets, evaluation blocks and review coverage; shares `selectionSha256` with `release.json` |
| `sim-statistics.json` | Per-point success comparisons behind the simulation results, with the `release.json` hash they were computed against |
| `manifest.json` | The catalog: every released dataset repository of the five campaigns, its revision, role, rounds, parent and cameras |
| `catalog.html`, `catalog.js` | The catalog page, rendered from `manifest.json` |
| `episode-lineage-audit.json` | Episode-level ancestry of the derived views, matched against their parents |
| `evaluation-bundles.json` | The cells, seeds and locked grids of the two simulation evaluation datasets |
| `training-recipes.json` | The dataset inputs of the 46 simulation DIVL cells and the final real-world critics, as release ids and revisions |
| `SHA256SUMS` | Checksums of the files above; `bun run package:release` checks them |

`arena/README.md` covers building and deploying the site from these files.

## Names

Dataset ids are lowercase kebab case: task first, then two-digit indexes, so lexicographic order matches numeric
order. Task tokens are `real-marker-d2`, `real-square-d2`, `real-routing-d2`, `sim-square-narrow` and
`sim-square-broad`.

| Kind | Pattern | Example |
|---|---|---|
| Teleop collection | `{task}-cNN-teleop-mixed` | `mulligan/real-marker-d2-c00-teleop-mixed` |
| DAgger collection | `{task}-cNN-dagger-mixed` | `mulligan/real-routing-d2-c09-dagger-mixed` |
| Teleop view | `{task}-c00-teleop-{variant}` | `mulligan/sim-square-broad-c00-teleop-sobol` |
| DAgger view | `{task}-cNN-dagger-{variant}` | `mulligan/real-square-d2-c03-dagger-mulligan` |
| Round evaluation | `{task}-rNN-eval` | `mulligan/real-marker-d2-r05-eval` |
| Screen evaluation | `{task}-rNN-screen` | `mulligan/real-marker-d2-r05-screen` |
| Multi-round evaluation | `{task}-rNN-rNN-eval` | `mulligan/real-routing-d2-r00-r05-eval` |
| Sim policy rollouts | `{task}-cNN-{arm}-policy-rollouts` | `mulligan/sim-square-narrow-c02-auto-iql-n32-policy-rollouts` |

`cNN` is a collection increment, `rNN` an evaluated model round, and `bNN` (the `session_id` of a round
dataset) one recorded evaluation session. DAgger variants are `baseline`, `mulligan`, `mulligan-no-cf`,
`sobol-with-cf` and `sobol-no-cf`; R0 treatment demos are `sobol`, since R0 has no counterfactual collection.
`validation` is the held-out teleop view. The two Square-Broad preview views keep the `first100` suffix.

## Routing rounds

Routing D2 trained a model every collection increment. The released rounds group them in pairs:

| Released round | Collection increments | Original model round |
|---|---|---|
| R0 | C00 | R0 |
| R1 | C01 + C02 | R2 |
| R2 | C03 + C04 | R4 |
| R3 | C05 + C06 | R6 |
| R4 | C07 + C08 | R8 |
| R5 | C09 | R9 |

C01-C08 were collected by velocity-action policies and C09 by relative-action policies. The final evaluation, 15 policies and
750 human-reviewed episodes, is one dataset with a policy/round table (`mulligan/real/lifecycle/routing_d2_lineage.py`).
The earlier Routing evaluations that the R5 critic trained on are released as `velocity-rNN` datasets under
their original model round.
