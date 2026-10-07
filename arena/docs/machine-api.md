# Machine API

Reads (Convex queries) are public. People write through the web app after signing in with
Hugging Face (see ["The first editor"](../../docs/arena.md#3-the-first-editor) in
`docs/arena.md`). Robots, training jobs and scripts write through Convex HTTP Actions with
per-machine bearer keys. Convex never stores a raw key, only its SHA-256
digest.

## Keys and scopes

A key has the form `<key-id>.<secret>`. Keep the whole key on the machine that uses it. The
deployment stores the digest in the `POLICY_ARENA_MACHINE_KEYS_JSON` environment variable:

```json
{
  "robot-a-2026-01": { "sha256": "<64 lowercase hex characters>", "scopes": ["ingest"] },
  "curator-2026-01": { "sha256": "<...>", "scopes": ["curate"] },
  "admin-2026-01": { "sha256": "<...>", "scopes": ["admin"] }
}
```

Key ids are 3 to 100 characters from `A-Z a-z 0-9 _ -`. The scopes:

- `ingest`: submit evaluation sessions and rollout sessions, register datasets, publish task and
  stage specs, stage predictions, refresh dataset statistics.
- `curate`: edit lifecycle statuses, tags, operators and human review records.
- `admin`: delete records and run repair utilities.

Use one key per machine and a separate admin key. A leaked ingest key cannot edit reviews or
delete data, and revoking one machine does not interrupt another.

`ARENA_SERVICE_TOKEN` is a random secret that only the HTTP Actions use to call the mutation
handlers. Set it on the deployment and nowhere else: never on a robot, workstation or worker.

## Create a key

```bash
key_id=robot-a-$(date +%Y-%m)
key="$key_id.$(openssl rand -hex 32)"
digest=$(printf %s "$key" | sha256sum | cut -d' ' -f1)   # macOS: shasum -a 256

# On the machine that will use it (mode 0600):
mkdir -p ~/.config/policy-arena
printf %s "$key" > ~/.config/policy-arena/api_key && chmod 600 ~/.config/policy-arena/api_key

# On your workstation, with the deployment selected (.env.local or CONVEX_DEPLOYMENT):
npx convex env set POLICY_ARENA_MACHINE_KEYS_JSON \
  "{\"$key_id\": {\"sha256\": \"$digest\", \"scopes\": [\"ingest\"]}}"
```

The variable holds the whole registry, so include the existing entries when you add one. Generate
the key on the machine that uses it, or move it over a secure channel; do not paste it into shell
history on a shared host, logs, issues or source control.

Check a key without changing data:

```bash
curl --fail-with-body -H "Authorization: Bearer $(cat ~/.config/policy-arena/api_key)" \
  https://<your-deployment>.convex.site/api/v1/auth/whoami
```

To rotate, add the new entry, switch the machine to the new key, check `whoami`, then remove the
old entry.

## Python client

```bash
pip install ./arena/python        # or: uv pip install ./arena/python
export POLICY_ARENA_URL=https://<your-deployment>.convex.cloud
```

```python
from policy_arena import PolicyArenaClient

arena = PolicyArenaClient()        # url from $POLICY_ARENA_URL
arena.whoami()                     # {"ok": True, "actor": "robot-a-2026-01", "scopes": ["ingest"]}
```

The client reads the key from `POLICY_ARENA_API_KEY`, else from the file at
`POLICY_ARENA_API_KEY_PATH` (default `~/.config/policy-arena/api_key`). Queries need no key; a
write without a key fails before any request is sent. The machine API base defaults to
`https://<deployment>.convex.site/api/v1` for a `.convex.cloud` URL; for a self-hosted or local
backend pass `api_url=` or set `POLICY_ARENA_API_URL`.

Session submission sends an `Idempotency-Key`, so retrying the same body with the same key returns
the original session instead of creating a duplicate.

## Routes

Every write is `POST /api/v1/mutate/<module>/<function>` with a JSON body of the mutation
arguments (Convex JSON encoding, so 64-bit integers are `{"$integer": "<base64>"}`; the Python
client does this for you). `convex/http.ts` lists each route and the scope it needs. Errors return
`{"ok": false, "error": "<message>", "code": "<code>", "error_id": "<id>"}` without internal
details; the full error is in the deployment logs under that `error_id`.
