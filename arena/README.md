# Policy Arena

A web app for evaluating robot policies on hardware. Robots submit evaluation sessions in which
several policies run from the same initial states. The Arena fits Bradley-Terry ratings from the
pairwise outcomes, shows a leaderboard per task, plays back every rollout from its Hugging Face
dataset, and lets signed-in editors review outcomes, label task stages and curate datasets.

The same source tree builds two sites:

- **Your own Arena** (`bun run dev`, `bun run build`): the full app, with a Convex database and
  functions you deploy to your own Convex account, and a static frontend on Vercel or Cloudflare
  Pages. This is how the Mulligan project ran its evaluations.
- **The Mulligan snapshot** (`bun run package:release`): the read-only site at
  <https://arena.mulligan.page>, built from the frozen files in `data/` with no backend.

[`docs/arena.md`](../docs/arena.md) in the repository root is the full deployment guide; this page
is the quick reference. The app code is MIT-licensed (`LICENSE`).

## Layout

| Path | What it is |
|---|---|
| `src/` | The screens both sites share (leaderboard, sessions, pairings, data explorer, coverage, outcome and stage review) and the snapshot's frozen-data adapter (`src/release/`). |
| `app/` | Entry of your own Arena: `index.html`, `main.tsx` (Convex client and sign-in), `client.ts` (live Convex hooks), `AuthControls.tsx`. |
| `convex/` | Backend: schema, queries and mutations, access control (`access.ts`), Hugging Face sign-in (`auth.ts`), machine HTTP API (`http.ts`, `machineAuth.ts`), dataset statistics, the outcome-review apply worker (`apply/`, `applyWorker.ts`). `_generated/` is Convex codegen, committed. |
| `python/` | `policy-arena` Python client for robots and scripts, its tests, and `examples/import_mulligan_snapshot.py`. |
| `scripts/` | Deployment setup (`setup_deployment.ts`), API-type capture (`capture_api.ts`), closure and leak checks (`release_closure.ts`, `leak_check.ts`), CSP generation (`security_headers.ts`), snapshot packaging and verification (`package_release.ts`, `verify_release_data.ts`, `verify_release_browser.py`), dataset-stats rebuild (`rebuild_dataset_stats.ts`), offline apply runner (`apply_local.ts`). |
| `tests/` | Backend and UI tests (`bun test`, with `convex-test` in memory) and their fixtures (`tests/fixtures/`). Shared-logic tests sit next to their modules in `src/`. |
| `docs/machine-api.md` | Machine keys, scopes and routes. |
| `data/`, `index.html`, `overview.html`, `vite.release.config.ts` | The Mulligan snapshot: frozen data and its two pages. |
| `vercel.json`, `public/_headers` | Security headers for Vercel and Cloudflare Pages. The CSP is computed at build time (`scripts/security_headers.ts`). |
| `.env.example` | Every setting, frontend and backend. |
| `release-closure.txt`, `tsconfig.release.json` | The exact file set of the snapshot build (see Checks). |
| `package.json`, `bun.lock`, `tsconfig*.json`, `eslint.config.js`, `vite.config.ts`, `convex.json` | Toolchain: pinned dependencies (bun 1.3.14), TypeScript, lint, the self-deploy Vite build and Convex bundling settings. |

## Your own Arena: quick start

Requires [bun](https://bun.sh) 1.3.14, Node 20.19+, 22.13+ or 24, and a free
[Convex](https://convex.dev) account. From `arena/`:

```bash
bun install --frozen-lockfile
npx convex dev                       # log in, create a project; writes .env.local, pushes convex/
bun scripts/setup_deployment.ts --site-url http://localhost:5173   # auth keys, service token
npx convex env set AUTH_HUGGINGFACE_ID <client id>                 # your Hugging Face OAuth app
npx convex env set AUTH_HUGGINGFACE_SECRET <client secret>
bun run dev                          # second terminal: http://localhost:5173
```

Sign in once, then allow yourself to edit: `npx convex env set ARENA_EDITOR_SUBS <your Hugging Face
account id>` (how to find it, production deployment, hosting and machine keys: `docs/arena.md`).

```bash
bun run build                        # type-check, build dist/ for VITE_CONVEX_URL, leak check
```

Deploy `dist/` with Vercel (`vercel.json` holds the build settings) or Cloudflare Pages (build
command `bun run build`, output `dist`); set `VITE_CONVEX_URL` in the host's build environment.

Robots and scripts write through the machine API:

```bash
pip install ./python                 # from arena/
export POLICY_ARENA_URL=https://<your-deployment>.convex.cloud POLICY_ARENA_API_KEY=<key-id>.<secret>
python python/examples/import_mulligan_snapshot.py data/release.json   # optional example data
```

## Checks

```bash
bun run typecheck      # app, Convex backend, release closure, build configs
bun run lint
bun test               # backend, UI, release adapter and shared logic
bun run check:api      # src/release/api.ts matches the backend types (see below)
bun run check:closure  # the snapshot build imports exactly release-closure.txt
```

The shared screens import `api`, `Id` and `Doc` from `src/release/api.ts`: Convex's `anyApi` proxy
typed with the argument and result types of the backend functions they call. This keeps the
snapshot build free of `convex/`. After changing one of those backend functions, run
`bun scripts/capture_api.ts --write`. The self-deploy build (`vite.config.ts`) swaps exactly two
modules: `src/lib/arenaClient.ts` for `app/client.ts` and `src/release/ReleaseLinks.tsx` for
`app/AuthControls.tsx`, and fails if a snapshot-only module reaches its bundle.

## The Mulligan snapshot

```bash
bun install --frozen-lockfile
bun run check:closure                         # the build imports exactly release-closure.txt
bun run package:release data/release.json     # type-check, build, stage data into dist-release/
bun run verify:data                           # recompute ratings and results from data/
bun run leak-check                            # no Convex URL, live client or sign-in text in dist-release/
```

`package:release` runs `tsc -p tsconfig.release.json`, then `vite build`, copies the data files
listed in `data/SHA256SUMS` after checking their hashes, writes `_headers` and `robots.txt`, and
runs the leak check. Add `--preview` to stage a non-indexed preview. Every screen reads the frozen files in
`data/`; videos stream from the pinned `mulligan/*` dataset revisions on Hugging Face.

The browser check drives the staged site in headless Chrome. It needs network access for the videos:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory dist-release &
uv tool run --from playwright==1.63.0 python scripts/verify_release_browser.py http://127.0.0.1:8000/ browser-check
kill %1
```

It uses `$MULLIGAN_CHROME` if set, otherwise the first of `google-chrome`, `google-chrome-stable`,
`chromium` or `chromium-browser` on `PATH`.

To deploy the staged site to Cloudflare Pages (an API token with Pages write access for the account that
owns the project):

```bash
npx wrangler@4.132.0 pages deploy dist-release --project-name <project>
```

On `arena.mulligan.page` only, `src/release/analytics.ts` posts each page view (path, referring host,
`utm_source` or `ref`) and each followed link to another host or download to the paper site's visit
counter (`connect-src https://mulligan.page`). It sets no cookies; a copy served from any other origin
sends nothing.

Run the browser check against the preview URL Wrangler prints before promoting it. The build is
deterministic: with the same bun, Node and `bun.lock`, two `package:release` runs give the same files
(`cd dist-release && sha256sum index.html overview.html assets/*`).

If you add or remove an import of a shared screen, regenerate the snapshot's file set and commit it:

```bash
bun scripts/release_closure.ts --write        # rewrites release-closure.txt and the tsconfig include list
```

`bun run check:closure` fails when the snapshot build reaches a file that is not listed, when a
listed file is no longer reached, or when `src/` holds a file outside the closure (code used only by
your own Arena goes in `app/` or `convex/`). The Vite build refuses to bundle a file outside
`release-closure.txt`. `data/` is the data set for the next snapshot deployment.
