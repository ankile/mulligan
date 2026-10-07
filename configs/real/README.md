# Real-robot configs

## What is here

| File | Kind | Count | Source |
|---|---|---|---|
| `<task>/rNN_<arm>_dp.yaml`, `routing_d2/velocity/rNN_<arm>_dp.yaml` | Trainer config of one released diffusion-policy (DP) actor | 58 | the checkpoint's training run |
| `<task>/rNN_critic.yaml`, `rNN_critic_screen_bNN.yaml`, `routing_d2/velocity/rNN_critic.yaml`, `marker_d2/c05_collector_critic.yaml` | Trainer config of one released IDQL critic | 18 | the checkpoint's training run |
| `<task>/rNN_eval.yaml` | The blinded evaluation sessions of one round: arms (actor, critic, N), start manifest, seed | 18 | the round datasets' `meta/round_dataset.json` |
| `marker_d2/rNN_sampler.yaml`, `square_d2/rNN_sampler.yaml` | Start design (Alg. 2) of collection round N, R1-R5 | 10 | by hand |
| `station.example.yaml` | Camera layout of the paper's station, with placeholder ZED serials (`MULLIGAN_STATION_CONFIG`; [docs/station.md](../../docs/station.md)) | 1 | by hand |
| `station.env.example` | Keys of the per-machine station identity file `~/.config/droid/station.env` | 1 | by hand |

The trainer configs (`*_dp.yaml`, `*_critic*.yaml`) hold the trainer flags of each released checkpoint,
with every value whose current default differs written out. The eval configs are derived from the public
`meta/round_dataset.json` of each evaluation dataset (named in their first line).

## Naming

- `<task>` is the task name of the code and the model repos: `marker_d2` (Insert Marker), `square_d2`
  (Thread Nut), `routing_d2` (Route Cable).
- `rNN` is the public round, two digits (R0-R5). A trainer config is named after its model repo
  `mulligan/real-<task>-rNN-<arm>-dp` / `-mulligan-idql-critic[-screen-bNN]`, with `-` as `_` and
  `mulligan_idql_critic` shortened to `critic`.
- `<arm>`: `baseline` (HG-DAgger), `mulligan` (HG-DAgger+Mulligan), `sobol` (the R0 Sobol-demo actor of
  the Mulligan arm), `mulligan_no_cf` (the no-counterfactual ablation). Critics are the HiL-IDQL+Mulligan
  arm.
- `routing_d2/velocity/rNN_*`: the velocity-action Cable lineage (source rounds R0-R7) that collected
  Cable increments c01-c08. `marker_d2/c05_collector_critic.yaml`: the critic that collected Marker c05.
  `*_screen_bNN`: critics of the R5 screening sessions (`*-r05-screen`).

## Run one

```bash
# retrain a checkpoint (one GPU); a critic downloads the frozen DP encoder it names
python -m mulligan.real.train.launch configs/real/marker_d2/r05_mulligan_dp.yaml --output-dir outputs/real/marker_r05_dp
python -m mulligan.real.train.launch configs/real/marker_d2/r05_critic.yaml --output-dir outputs/real/marker_r05_critic
# print the trainer argv instead; flags after -- override the config
python -m mulligan.real.train.launch configs/real/marker_d2/r05_mulligan_dp.yaml --output-dir out --print-argv -- --training-steps 500
```

A start-design config runs with `mulligan.sampling.real_design` (see Start design below).

An eval config is a record, not a launcher input: to re-run a session, pass its `manifest`, `seed` and
arms to `mulligan.real.eval.manifest_eval` ([docs/real_robot_eval.md](../../docs/real_robot_eval.md)).

## Paper mapping

Every trainer config has a generated `paper` block:

```yaml
paper:
  task: Insert Marker          # paper task name
  round: R5                    # public round; "velocity R3" = velocity-action Cable round; c05 = collection increment
  arm: HG-DAgger+Mulligan      # paper arm
  seed: 1                      # training seed (--seed)
  results: [Fig. 5, App. B.1]  # figures and appendix sections that plot an evaluation of the checkpoint
  evaluations: [...]           # "<eval dataset> <session> (<arm>[ actor], N=<n>)", from release/models.json
  collections: [...]           # the *-dagger-mixed collections the checkpoint collected
```

`results` follows from the evaluations: a counted session of a round dataset is a point of Fig. 5 (real
headline) and App. B.1 (per-arm success rates); a no-CF session is Fig. 7 (panels 3-4) and, for Thread
Nut, App. B.6. Sessions of `*-r05-screen` are not headline points. Collections feed the next
round's training data and the collection figures (Fig. 8, Fig. 9, App. B.8-B.10). The eval configs carry
the same `paper` mapping per round. Per-result commands: [docs/reproduce.md](../../docs/reproduce.md).

## Trainer config fields

| Key | Meaning |
|---|---|
| `kind`, `trainer` | `dp-actor` (`mulligan.real.train.policy`) or `idql-critic` (`mulligan.real.train.critic`) |
| `checkpoint` | The released repo, its pinned revision and the training `step` of the released checkpoint |
| `retrain` | `exact` if the public repos hold exactly the run's training and validation episodes (75 of 76), else `approximate` |
| `paper` | Where the paper uses the checkpoint (above) |
| `dataset_revisions`, `dataset_episodes` | Pinned revision and episode selector of every training and validation repo (`release/training-views.json`) |
| `args` | Trainer flags (`--key value`; `true` is a bare flag) |
| `camera_crops` | DP actors: the per-camera crop boxes the run applied, in stored 640x480 pixels |

An eval config has `task` and `round` (public round), `paper`, the pinned `eval_dataset`, and per
session: `session_id` (`bNN`), `seed` (the Sobol seed of the session's held-out manifest),
`manifest_sha256` with `manifest` (a file under `data/real/manifests/`) or `manifest_in_eval_dataset`,
`n_starts`, and the `arms`: `policy_key` and `method` (the paper arm), `role` (`counted` or
`no-cf-ablation`), `episodes`, the `actor` and optional `critic` (repo, revision, step, trainer `config`)
and the Best-of-N `num_action_samples` of a critic arm.

[docs/real_robot.md](../../docs/real_robot.md) lists the checkpoint steps and the approximate checkpoint.

## Start design

`rNN_sampler.yaml` (Marker and Thread Nut, R1-R5) configures `mulligan.sampling.real_design`: failure
promotion from the previous round's held-out evaluation, coverage fill, a fresh baseline and the paired
order. `locked` names the locked collection manifest it produced (`data/real/manifests/`). Its `inputs`
name the earlier manifests by their repo-relative path in `data/real/manifests/` (resolved against
`--inputs-root`, the repository root); the evaluation outcome tables and stage labels are not released, so
the config leaves them `null` and the design cannot be rebuilt bit for bit from this repository. Give those
two inputs with `--input` (your own evaluation outcomes and stage labels), and repeat `--input` to replace a
list input such as `support_manifests`:

```bash
python -m mulligan.sampling.real_design --config configs/real/marker_d2/r02_sampler.yaml \
    --inputs-root . --input eval_outcomes=<csv> --input stage_labels=<csv> --out <manifest.json>
```

Route Cable has no sampler config: `real_design` covers only what the Marker and Thread Nut manifests need;
the Cable manifests ship locked in
`data/real/manifests/routing_d2/`.
