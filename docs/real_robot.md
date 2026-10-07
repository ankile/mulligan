# Real robot: train, deploy, collect

This page covers the real-robot half of Mulligan: retraining the released diffusion-policy actors (DP) and
IDQL critics from the public datasets, loading released checkpoints for deployment (Best-of-N reranking with
a critic), and running collection and blinded evaluation on a DROID-style Franka station. Station setup
(cameras, SpaceMouse, DROID fork, robot environment) is in `docs/install.md` and the station docs.

What can and cannot be reproduced:

- Training (tier 3): every released real checkpoint has a config under `configs/real/<task>/`
  that pins the dataset revisions and episode selectors it was trained on. One checkpoint is
  `approximate` (below).
- Deployment: the 58 DP and 18 critic checkpoints load from Hugging Face without W&B.
- Collection and evaluation (tier 4): the collectors and the blinded evaluation are runnable protocol code.
  A new campaign with a new operator gives new data and results; human interventions and outcome labels
  cannot be replayed.

Tasks: `marker_d2` (Marker), `square_d2` (Nut), `routing_d2` (Cable; its two round axes are in
[data.md](data.md#route-cable-two-round-axes)).
The objects, table layout, start-state ranges and success criteria of each task are in
[hardware/objects.md](hardware/objects.md).

## Released checkpoints and configs

| Kind | Repos | Config |
|---|---|---|
| DP actor | `mulligan/real-<task>-d2-rNN-{baseline,sobol,mulligan,mulligan-no-cf}-dp` (58) | `configs/real/<task>/rNN_<arm>_dp.yaml` |
| IDQL critic | `mulligan/real-<task>-d2-rNN-mulligan-idql-critic` (18, incl. `-screen-bNN` and the Marker `c05-collector` critic) | `configs/real/<task>/rNN_critic.yaml` |
| Round evaluation | `mulligan/real-<task>-d2-rNN-eval`, `mulligan/real-routing-d2-r00-r05-eval` | `configs/real/<task>/rNN_eval.yaml` |

Each repo holds one checkpoint. The step it was taken at is pinned in the config (`checkpoint.step`):
Marker R0 actors at 50k, Nut R0 and Cable velocity-action R0-R1 actors at 75k, all other actors at 100k. Critics:
Marker 125k (R2, R4, `c05-collector`), 100k (R3), 200k (R5) and 150k (R5 `screen-b02`); Nut 150k (R3,
R4, R5 `screen-b02`) and 200k (R5, R5 `screen-b03`); Cable 270k (R3, R4) and 314k (R5); Cable
velocity-action 150k (R4, R5), 190k (R6) and 230k (R7).

`rNN_eval.yaml` lists, per blinded session, the arms (actor, critic and the Best-of-N sample count `N`),
the start-manifest SHA256 and the Sobol seed. `N` per round: Marker R2-R4 16, Marker R5 32, Nut R3-R4 16,
Nut R5 32, Cable R3-R5 32.

### Approximate checkpoints

`mulligan/real-routing-d2-velocity-r05-mulligan-idql-critic` is the one `approximate` checkpoint
(`release/training-views.json`): its frozen encoder comes from an unreleased version of the DP run released as
`mulligan/real-routing-d2-velocity-r05-mulligan-dp`. Its config trains on the released DP's encoder, and the
critic is deployed with the released DP passed explicitly (below), as it was in every paper deployment.

## Train

```bash
uv sync --frozen
# DP actor (one GPU)
uv run python -m mulligan.real.train.launch configs/real/marker_d2/r05_mulligan_dp.yaml --output-dir outputs/marker_r05_dp
# IDQL critic on the frozen encoder of its DP (downloaded from HF at the pinned revision)
uv run python -m mulligan.real.train.launch configs/real/marker_d2/r05_critic.yaml --output-dir outputs/marker_r05_critic
# Show the trainer argv without running; append trainer flags after `--` to override
uv run python -m mulligan.real.train.launch configs/real/marker_d2/r05_mulligan_dp.yaml --output-dir out --print-argv -- --training-steps 500
```

The trainers can also be called directly (`python -m mulligan.real.train.policy`,
`python -m mulligan.real.train.critic`). Data selection flags:

- `--dataset-revisions <repo>=<revision> ...` pins each dataset (training and validation) to a Hub revision.
- `--dataset-episodes <repo>=session_id:<id>[,<id>] ...` or `<repo>=episode_index:<i>[,<j>-<k>] ...`
  restricts a repo to some episodes. A `session_id` selector is resolved against the repo's public
  `meta/episode_provenance.parquet` at the pinned revision; a selector needs a pinned revision.
  Both flags take several values and can be repeated; one selector per repo. After the launcher's `--`
  they replace the config's entry for the repos they name and keep the others.
- Unpinned repos resolve the `v3.0` tag (LeRobot dataset format).
- By default each trainer first syncs the local copy of every repo to its pinned revision (stale local
  parquet shards are deleted). `--no-dataset-sync` reads the local copies under the dataset root as they
  are, for datasets that are not on the Hub (for example ones you collected and have not pushed).

DP configs also pin the per-camera crop boxes of the checkpoint (`camera_crops`, passed as
`--camera-crop <role>=x0,y0,x1,y1` in stored 640x480 pixels), so a retrain does not depend on the crops of
your station config.

Both trainers record the revisions and selectors in their run config and checkpoint metadata; the DP
trainer also records the Hub commit each repo was read at (`dataset_commits`, null with
`--no-dataset-sync`). The critic
writes `checkpoints/<step_N|final>/{iql_checkpoint.pt,metadata.json}`, the layout of the released critic
repos, so a retrained critic loads with the same loader (pass its directory as the model id).

W&B is optional: pass `--use-wandb` (and `--wandb-project`) to log; without it nothing is sent anywhere.
Critic frozen-feature caches are rebuilt in-process; add `--embedding-cache-output <path>` to keep them.

## Deploy (load a policy)

Model ids are `hf://<org>/<name>[@<revision>][/<subfolder>]` or a local checkpoint directory; the release
has one `hf://` parser (`mulligan.release.hub.parse_hf_uri`), which also accepts
`hf://<org>/<name>[/<subfolder>][@<revision>]`. `<revision>` is a commit sha, a tag or `refs/pr/<N>`. Without
`@<revision>` a `mulligan/*` repo loads at its release pin (`release/revisions.json`;
`mulligan/real/policy/released_checkpoints.json` is kept equal to it by a test). Downloads are anonymous.

A critic repo holds the critic only. Its `metadata.json` names the DP actor it reranks (`dp_artifact`);
the loader resolves that to the released DP repo and loads both. The approximate critic
`mulligan/real-routing-d2-velocity-r05-mulligan-idql-critic` needs the DP given explicitly: `dp_artifact_override=` below,
`--fixed-policy-dp-override <arm>=hf://mulligan/real-routing-d2-velocity-r05-mulligan-dp` in
`mulligan.real.collect.blind_dagger`, or `--fixed-policy-dp-override <label>=...` in
`mulligan.real.eval.manifest_eval` (the other collectors take DP actors only).

```python
from mulligan.real.policy.loader import load_policy_by_model_id

entry = load_policy_by_model_id(
    "hf://mulligan/real-marker-d2-r05-mulligan-idql-critic", policy_id=0, device="cuda"
)
entry.policy.num_action_samples = 32  # Best-of-N as deployed in R5 (see r05_eval.yaml)
```

`load_policy_by_model_id(model_id, policy_id, device, noise_scheduler=None, num_inference_steps=None,
default_camera_height=480, default_camera_width=640, n_action_steps=6,
dp_artifact_override=None)` returns a `PolicyEntry`; for a critic, `entry.policy` is a
`VisionIDQLRealWorldPolicy` that samples `num_action_samples` DP action chunks and executes the one with
the highest critic value. `tests/real/test_bon_equivalence.py` checks this path against a recorded
reference: the candidate chunks, critic scores and chosen chunk of the R5 critics on recorded
observations.

## Collect and evaluate

Collection (`python -m mulligan.real.collect.blind_dagger`, `...collect.dagger`, `...collect.teleop`,
`...collect.rollout`) and the blinded evaluation (`python -m mulligan.real.eval.manifest_eval`) need the
station environment (`robot/`, `bash scripts/sync_robot_env.sh`). Pushing a dataset needs an explicit
namespace: `--push-to-hub --hf-namespace <user-or-org>` (or a `NAMESPACE/NAME` dataset name); there is no
default account.

```bash
python -m mulligan.real.collect.blind_dagger \
    --task-name marker_d2 \
    --arm baseline_uniform=hf://mulligan/real-marker-d2-r01-baseline-dp \
    --arm mulligan_sobol=hf://mulligan/real-marker-d2-r01-mulligan-dp \
    --initial-states-manifest <locked manifest>.json \
    --protocol-quota-targets no_cf=50,with_cf=50 \
    --protocol-quota-arms 'no_cf=baseline_uniform,mulligan_sobol;with_cf=mulligan_sobol' \
    --protocol-quota-selection-mode soft_weighted \
    --protocol-quota-ledger ./data/<dataset>/meta/protocol_quota_ledger.jsonl \
    --dataset-name <dataset> \
    --push-to-hub --hf-namespace <your-namespace>
```

A collection is split into per-arm training datasets after it is reviewed. The paper's DAgger views
(`*-cNN-dagger-{baseline,mulligan,mulligan-no-cf}`) come from the protocol-quota split, as in simulation:

```bash
python -m mulligan.data.split_protocol_quota --source-repo <dataset> --source-root ./data/<dataset> \
    --manifest <locked manifest>.json --ledger ./data/<dataset>/meta/protocol_quota_ledger.jsonl \
    --expected-per-protocol no_cf=50,with_cf=50 \
    --target no_cf.baseline_uniform=<repo> --target no_cf.mulligan_sobol=<repo> --target with_cf.mulligan_sobol=<repo> \
    --output-root outputs/real/splits
```

`python -m mulligan.real.data.split` splits a collection by one ledger key: the R0 teleop collection by
`manifest_source` (`--split-key manifest_source --ledger ./data/<dataset>/meta/teleop_manifest_ledger.jsonl`;
this produced the `*-c00-teleop-{baseline,sobol,validation}` views), or a DAgger collection by `arm_key`
into one dataset per arm. Each view records its parent, ledger hash and episodes in
`meta/dataset_lineage.json`.

Start sets for a new round come from `mulligan.sampling.real_design` (Alg. 2 on the real robot),
configured per round by `configs/real/<task>/rNN_sampler.yaml` (Marker and Nut, R1-R5). Each config's
`locked` entry names the manifest that round collected on and its SHA256; the locked manifests ship in
`data/real/manifests/` (with `INDEX.csv`). The config's `inputs:` are the earlier manifests the design
covers and avoids, which are in `data/real/manifests/`, and the previous round's evaluation outcome
table and stage labels (Nut R4: a failure table), which are not released. `tests/real/test_real_design.py` rebuilds the locked Marker R1
manifest from trimmed copies of its recorded inputs (`tests/real/fixtures/real_design/`) and checks that
all ten sampler configs load. For a new round, point `eval_outcomes` and `stage_labels` at your own
evaluation outcomes and stage labels.
