# Policy Arena

The Policy Arena is the web app the Mulligan project used to run its real-robot evaluations: robots
submit evaluation sessions, the Arena fits Bradley-Terry ratings from the pairwise outcomes, plays
back every rollout from Hugging Face, and lets signed-in editors review outcomes, label task stages
and curate datasets. `arena/` ships it in two forms built from the same source tree:

- [Deploy your own Arena](#deploy-your-own-arena): the full app, with your own Convex backend and
  your own static host (Vercel or Cloudflare Pages).
- [The Mulligan snapshot](#the-mulligan-snapshot): the read-only site at
  <https://arena.mulligan.page>, built from frozen data with no backend.

`arena/README.md` lists the layout and the build and check commands. The app is MIT-licensed
(`arena/LICENSE`).

## Deploy your own Arena

### What runs where

| Part | Where it runs | Source |
|---|---|---|
| Database, queries, mutations, scheduled actions | your Convex deployment | `arena/convex/` |
| Sign-in | Convex Auth with Hugging Face OAuth, on the same deployment | `arena/convex/auth.ts` |
| Machine API (robots, scripts) | Convex HTTP Actions at `https://<deployment>.convex.site/api/v1` | `arena/convex/http.ts` |
| Web app | static files on Vercel, Cloudflare Pages or any static host | `arena/app/`, `arena/src/` |
| Videos and dataset metadata | read by the browser from public Hugging Face datasets | `arena/src/lib/hf-api.ts` |
| Robot-side client | your robot or training machines | `arena/python/` (`policy-arena`) |

Reads are public: anyone who can open the site can see the leaderboard, sessions and datasets.
Every write requires either a signed-in Hugging Face account on the editor allowlist or a machine
key. Functions meant only for operators (`seed:clearAll`, `operators:seed`) are internal:
`npx convex run` with deploy credentials can call them, the public API cannot.

### Prerequisites

- [bun](https://bun.sh) 1.3.14 and Node 20.19+, 22.13+ or 24.
- A [Convex](https://convex.dev) account (the free tier is enough to start). A self-hosted Convex
  backend also works; see "Without a Convex account" below.
- A Hugging Face account, for the OAuth app that signs editors in.
- A Vercel or Cloudflare account for the frontend.
- Python 3.10+ on the machines that submit results.

All commands below run from `arena/` unless they say otherwise.

### 1. Backend

```bash
bun install --frozen-lockfile
npx convex dev
```

The first `npx convex dev` asks you to log in and to create a project. It creates a development
deployment, pushes `convex/` (schema, indexes, functions), keeps watching for changes, and writes
`CONVEX_DEPLOYMENT`, `VITE_CONVEX_URL` and `VITE_CONVEX_SITE_URL` to `.env.local` (git-ignored).
Use `npx convex dev --once` to push without watching.

For the site other people use, deploy the same code to the project's production deployment with
`npx convex deploy`, and set the variables below with `npx convex env set --prod ...` (or in the
Convex dashboard under Settings, Environment Variables). Development and production deployments
have separate data and separate settings.

### 2. Secrets and sign-in

```bash
bun scripts/setup_deployment.ts --site-url https://arena.example.org    # add --prod for production
```

This sets `SITE_URL` (where sign-in returns to: your frontend URL, `http://localhost:5173` while
developing), generates the Convex Auth key pair `JWT_PRIVATE_KEY` and `JWKS`, and sets a random
`ARENA_SERVICE_TOKEN`. It prints only the names it set and keeps values that already exist
(`--force` replaces them; a new JWT key signs everyone out). The interactive `npx @convex-dev/auth`
does the same key setup if you prefer it.

Create a Hugging Face OAuth app (huggingface.co, Settings, Connected Apps, "Create App"):

- Redirect URL: `https://<deployment>.convex.site/api/auth/callback/huggingface`, where
  `<deployment>` is the name in `VITE_CONVEX_URL` (for production, the production deployment's name).
- Scopes: `openid` and `profile`. The Arena does not ask for email.

```bash
npx convex env set AUTH_HUGGINGFACE_ID <client id>
npx convex env set AUTH_HUGGINGFACE_SECRET <client secret>
```

### 3. The first editor

Writes from the web app are allowed only for Hugging Face accounts listed in `ARENA_EDITOR_SUBS`,
by their stable account id (the OIDC `sub`), not by username, because usernames can change.

1. Open the app and choose "Sign in with Hugging Face". You are signed in, but the edit controls
   stay hidden and every save fails with "not an allowlisted editor".
2. Find your account id: the `providerAccountId` of your row in the `authAccounts` table
   (`npx convex data authAccounts`, or the Convex dashboard), or the `_id` field of
   `https://huggingface.co/api/users/<username>/overview`.
3. `npx convex env set ARENA_EDITOR_SUBS <id>`; add more editors comma-separated. The change takes
   effect immediately; the header then shows an "editor" badge.

With `ARENA_EDITOR_SUBS` empty or unset, no one can edit from the web app: the handlers fail closed.

### 4. Machine keys for robots and scripts

Robots, evaluation scripts and training jobs write through the machine API with a per-machine key.
The deployment stores only SHA-256 digests, in `POLICY_ARENA_MACHINE_KEYS_JSON`, each with scopes
`ingest`, `curate` or `admin`. `arena/docs/machine-api.md` shows how to create, install, check and
rotate keys. `ARENA_SERVICE_TOKEN` stays on the deployment; never copy it to a machine.

Optional settings:

- `HF_TOKEN`: dataset summary statistics read the datasets' `meta/` parquet files with it (read
  access is enough). Outcome-review apply writes corrected outcomes back to the dataset with it
  (needs write access). Without it, those actions fail with an explicit error; everything else works.

### 5. Frontend

While developing, run `bun run dev` next to `npx convex dev` and open <http://localhost:5173>.

`bun run build` type-checks the app and the backend, builds `dist/` for the deployment in
`VITE_CONVEX_URL` (from the environment or `.env.local`), and runs the leak check, which rejects
private keys and Hugging Face tokens in the built files. The Content-Security-Policy is
generated at build time and names exactly your deployment (`scripts/security_headers.ts`); the other
security headers come from `vercel.json` or `public/_headers`.

**Vercel.** Import the repository, set the project's root directory to `arena`, and add the
environment variable `VITE_CONVEX_URL=https://<deployment>.convex.cloud` (the production
deployment). `vercel.json` sets the install command, `bun run build` and the output directory
`dist`. To push the backend on every frontend deploy instead, set the build command to
`npx convex deploy --cmd 'bun run build'` and add a production deploy key from the Convex dashboard
as `CONVEX_DEPLOY_KEY`; `convex deploy` then sets `VITE_CONVEX_URL` for the build itself. From the
command line: `npx vercel` in `arena/` (`npx vercel --prod` for production).

**Cloudflare Pages.** Connect the repository with root directory `arena`, build command
`bun run build`, output directory `dist`, and the environment variables `VITE_CONVEX_URL` and
`BUN_VERSION=1.3.14` (the build image installs that bun).
`public/_headers` is copied into `dist/` and applies the security headers. Without Git integration,
build locally and upload:

```bash
VITE_CONVEX_URL=https://<deployment>.convex.cloud bun run build
npx wrangler pages deploy dist --project-name <your-project>
```

After the first deploy, set `SITE_URL` on the deployment to the site's final URL (with `--prod`
for production) so sign-in returns there.

### 6. Robots and scripts

The robot side talks to the Arena only through the Python client. Install it where evaluations
run, with the deployment URL and that machine's key:

```bash
pip install ./arena/python            # from the repository root; needs the `convex` package
export POLICY_ARENA_URL=https://<deployment>.convex.cloud
# key in $POLICY_ARENA_API_KEY or ~/.config/policy-arena/api_key (mode 0600)
```

A typical evaluation script registers the dataset it recorded and submits the session: which
policies ran, and for each round (one initial state) each policy's outcome and the episode index
of its rollout in that dataset:

```python
from policy_arena import (DatasetInput, PolicyArenaClient, PolicyInput,
                          RoundInput, RoundResultInput)

arena = PolicyArenaClient()          # $POLICY_ARENA_URL; key from the environment or file
arena.whoami()                       # fails early if the key is missing or rejected

arena.register_dataset(DatasetInput(
    repo_id="your-org/eval-pick-cube-2026-10-01", name="Pick cube eval, Oct 1",
    task="pick_cube", source_type="eval", environment="pick_cube"))
policies = [PolicyInput(name="dp-baseline", model_id="hf://your-org/dp-baseline", environment="pick_cube"),
            PolicyInput(name="dp-dagger-r1", model_id="hf://your-org/dp-dagger-r1", environment="pick_cube")]
rounds = [RoundInput(round_index=0, results=[
    RoundResultInput(model_id="hf://your-org/dp-baseline", success=False, episode_index=0, num_frames=412),
    RoundResultInput(model_id="hf://your-org/dp-dagger-r1", success=True, episode_index=1, num_frames=305)])]
session_id = arena.submit_eval_session("your-org/eval-pick-cube-2026-10-01", policies, rounds)
```

Policies are registered on first use, keyed by `model_id`. The leaderboard groups by
`environment`. Other calls: `get_recommended_opponents` (which policies to pair next),
`add_rounds` (extend a session), `set_policy_tags`, `set_task_status` / `set_policy_status`
(lifecycle: mainline, retired, ablation, testing), `add_operator` and `operator=` on submission
(who ran the robot), `upsert_task_spec` / `upsert_stage_task_spec` (task definitions for outcome
and stage review, below), and the stage-prediction upload (`upload_stage_prediction_run`). The
training and evaluation code in this repository does not call the Arena; you add these calls to
your own evaluation loop.

The outcome and stage review screens need the task definitions of the Mulligan tasks: camera
layout, display crops and subtask-mark counts (`taskSpecs`), and the stage-label vocabulary the
stage review form checks labels against (`stageTaskSpecs`). Export them from the Python task
registry and upload them with a key that has the `ingest` scope; run this from the repository
root in the environment where `mulligan` and the `policy-arena` client are installed:

```bash
python -m mulligan.tools.export_arena_task_specs --out arena_task_specs.json   # dry run: write the JSON only
python -m mulligan.tools.export_arena_task_specs --url https://<deployment>.convex.cloud
```

The JSON holds one object per mutation call, `{"task_specs": [...], "stage_task_specs": [...]}`,
with exactly the arguments of `taskSpecs:upsert` and `stageTaskSpecs:upsert` (integers as plain
JSON numbers; the Python client encodes the 64-bit ones). Rows are keyed by the task name the
released datasets carry, and each exported stage-label taxonomy becomes the task's live version.
Camera keys come from your station config (`MULLIGAN_STATION_CONFIG`, see `docs/station.md`);
the released datasets name cameras by role, which the Arena matches without them. Re-running the
export is safe: rows are replaced, and a taxonomy version whose content changed is rejected.
`tests/real/test_export_arena_task_specs.py` checks the rows against the mutation validators.

`scripts/rebuild_dataset_stats.ts` refreshes the dataset summary statistics after you change a
dataset on the Hub outside the Arena (`POLICY_ARENA_URL` and a key with the `ingest` scope).

### 7. Example data

To see the Arena with real data before your own evaluations exist, load the Mulligan real-robot
evaluations (21 sessions, 46 policies, three tasks) from the frozen snapshot file:

```bash
python arena/python/examples/import_mulligan_snapshot.py arena/data/release.json   # --dry-run to preview
```

It needs a key with the `ingest` scope. Videos stream from the public `mulligan/*` datasets. The
live app pools all sessions of a task when it fits ratings, so the numbers differ from the
snapshot's per-grid ratings. Reruns are idempotent. Some review aids (event timelines, subtask
marks) recognize the Mulligan task definitions and do nothing for other tasks.

### Without a Convex account

`CONVEX_AGENT_MODE=anonymous npx convex dev` runs a local backend at `http://127.0.0.1:3210`
(HTTP Actions at `:3211`) without an account and writes those URLs to `.env.local`. Convex
downloads a prebuilt backend binary, which needs glibc 2.35 or newer. For the Python client against
a local or self-hosted backend, also set `POLICY_ARENA_API_URL` (for example
`http://127.0.0.1:3211/api/v1`). Hugging Face sign-in needs a redirect URL the OAuth app accepts,
so a local backend is mainly useful for the machine API and read-only browsing.

## The Mulligan snapshot

The public Arena at https://arena.mulligan.page is a static, read-only site. Its pages are
`arena/index.html` and `arena/overview.html`, built by `vite.release.config.ts` from the shared
screens with the frozen-data adapter in `arena/src/release/`, and its data is `arena/data/`.

### Contents

- `arena/data/` holds the data files of the public site, in the schema that `catalog.html` and the Arena
  frontend read; `arena/data/SHA256SUMS` lists them with their hashes. Models, runs and datasets appear
  under their release ids (`mulligan/*`), and every
  dataset pin is the repo's revision in `release/revisions.json`. These files are not the `release/*.json`
  manifests.
- The self-deploy app ships `convex/`, `app/`, the Python client, the backend and UI tests, and the
  scripts `setup_deployment.ts`, `capture_api.ts`, `security_headers.ts`, `leak_check.ts --app`,
  `rebuild_dataset_stats.ts` and `apply_local.ts`. `bun.lock` pins the dependencies.

Building, checking and deploying the snapshot: [`arena/README.md`](../arena/README.md#the-mulligan-snapshot).
