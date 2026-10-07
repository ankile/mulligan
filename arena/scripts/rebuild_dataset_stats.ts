/**
 * Queue authoritative Policy Arena dataset-summary refreshes in Convex and
 * wait for every selected dataset to finish.
 *
 * The Convex worker pins each computation to the current Hugging Face commit
 * SHA and reads meta/episodes parquet statistics. This script is the manual
 * reconciliation path for Hub changes made outside dataset registration and
 * the native outcome-review apply worker.
 *
 * Usage (POLICY_ARENA_URL is the Convex deployment URL; POLICY_ARENA_API_URL defaults to
 * the matching .convex.site/api/v1 for a .convex.cloud deployment):
 *   POLICY_ARENA_URL=https://<deployment>.convex.cloud bun run scripts/rebuild_dataset_stats.ts
 *   POLICY_ARENA_URL=... bun run scripts/rebuild_dataset_stats.ts org/repo-a org/repo-b
 */

function arenaUrls(): { url: string; apiUrl: string } {
  const url = process.env.POLICY_ARENA_URL?.trim();
  if (!url) throw new Error("Set POLICY_ARENA_URL to the Convex deployment URL");
  const apiUrl = process.env.POLICY_ARENA_API_URL?.trim()
    || (url.endsWith(".convex.cloud") ? `${url.slice(0, -".convex.cloud".length)}.convex.site/api/v1` : "");
  if (!apiUrl) throw new Error("Set POLICY_ARENA_API_URL: the deployment URL does not end in .convex.cloud");
  return { url, apiUrl };
}
const { url: ARENA_URL, apiUrl: ARENA_API_URL } = arenaUrls();
const BATCH_SIZE = 6;
const POLL_INTERVAL_MS = 2000;
const REFRESH_TIMEOUT_MS = 10 * 60 * 1000;

type RegisteredDataset = {
  repo_id: string;
  name: string;
  stats_status?: "pending" | "ready" | "error";
  stats_hf_sha?: string;
  stats_error?: string;
  stats_refresh_requested_at?: number;
};

type RefreshResult = {
  repoId: string;
  status: "ready" | "error" | "timeout";
  hfSha?: string;
  error?: string;
};

async function loadApiKey(): Promise<string> {
  const envKey = process.env.POLICY_ARENA_API_KEY?.trim();
  if (envKey) return envKey;
  const home = process.env.HOME;
  if (!home) {
    throw new Error("HOME is not set and POLICY_ARENA_API_KEY is absent");
  }
  const keyFile = Bun.file(process.env.POLICY_ARENA_API_KEY_PATH?.replace(/^~(?=\/)/, home)
    ?? `${home}/.config/policy-arena/api_key`);
  if (!(await keyFile.exists())) {
    throw new Error(
      "No arena API key found. Set POLICY_ARENA_API_KEY or create ~/.config/policy-arena/api_key",
    );
  }
  const apiKey = (await keyFile.text()).trim();
  if (!apiKey) throw new Error("Arena API key file is empty");
  return apiKey;
}

async function machineMutation<T>(
  apiKey: string,
  operation: string,
  args: Record<string, unknown>,
): Promise<T> {
  const response = await fetch(`${ARENA_API_URL}/mutate/${operation}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(args),
  });
  const payload = (await response.json()) as
    | { ok: true; value: T }
    | { ok: false; error: string };
  if (!response.ok || !payload.ok) {
    throw new Error(
      `Policy Arena ${operation} failed with HTTP ${response.status}: ${payload.error}`,
    );
  }
  return payload.value;
}

async function convexRequest<T>(
  kind: "query" | "mutation",
  path: string,
  args: Record<string, unknown>,
): Promise<T> {
  const response = await fetch(`${ARENA_URL}/api/${kind}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ path, args, format: "json" }),
  });
  if (!response.ok) {
    throw new Error(`Convex ${kind} ${path} returned HTTP ${response.status}`);
  }
  const payload = (await response.json()) as
    | { status: "success"; value: T }
    | { status: "error"; errorMessage: string };
  if (payload.status !== "success") {
    throw new Error(`Convex ${kind} ${path} failed: ${payload.errorMessage}`);
  }
  return payload.value;
}

async function waitForBatch(
  datasets: RegisteredDataset[],
  requestedAt: Map<string, number>,
): Promise<RefreshResult[]> {
  const pending = new Map(
    datasets.map((dataset) => [dataset.repo_id, dataset]),
  );
  const results: RefreshResult[] = [];
  const deadline = Date.now() + REFRESH_TIMEOUT_MS;
  while (pending.size > 0) {
    if (Date.now() >= deadline) {
      for (const repoId of pending.keys()) {
        console.error(`[timeout] ${repoId}`);
        results.push({ repoId, status: "timeout" });
      }
      break;
    }
    const repoIds = [...pending.keys()];
    const rows = await Promise.allSettled(
      repoIds.map((repoId) =>
        convexRequest<RegisteredDataset | null>("query", "datasets:getByRepo", {
          repo_id: repoId,
        }),
      ),
    );
    for (let index = 0; index < rows.length; index++) {
      const repoId = repoIds[index];
      const settled = rows[index];
      if (settled.status === "rejected") {
        const error =
          settled.reason instanceof Error
            ? settled.reason.message
            : String(settled.reason);
        console.error(`[error] ${repoId}: ${error}`);
        results.push({ repoId, status: "error", error });
        pending.delete(repoId);
        continue;
      }
      const row = settled.value;
      if (row === null) {
        const error = "Registered dataset disappeared during refresh";
        console.error(`[error] ${repoId}: ${error}`);
        results.push({ repoId, status: "error", error });
        pending.delete(repoId);
        continue;
      }
      if (row.stats_status === "error") {
        const error = row.stats_error ?? "Stats refresh failed";
        console.error(`[error] ${row.repo_id}: ${error}`);
        results.push({ repoId: row.repo_id, status: "error", error });
        pending.delete(row.repo_id);
        continue;
      }
      if (
        row.stats_status === "ready" &&
        row.stats_refresh_requested_at === undefined
      ) {
        console.log(
          `[ready] ${row.repo_id} @ ${row.stats_hf_sha?.slice(0, 8)}`,
        );
        results.push({
          repoId: row.repo_id,
          status: "ready",
          hfSha: row.stats_hf_sha,
        });
        pending.delete(row.repo_id);
      } else if (
        row.stats_refresh_requested_at !== undefined &&
        row.stats_refresh_requested_at !== requestedAt.get(row.repo_id)
      ) {
        const error = "Another stats refresh replaced this request";
        console.error(`[error] ${row.repo_id}: ${error}`);
        results.push({ repoId: row.repo_id, status: "error", error });
        pending.delete(row.repo_id);
      }
    }
    if (pending.size > 0) await Bun.sleep(POLL_INTERVAL_MS);
  }
  return results;
}

async function main(): Promise<void> {
  const requested = new Set(Bun.argv.slice(2));
  const apiKey = await loadApiKey();
  const registered = await convexRequest<RegisteredDataset[]>(
    "query",
    "datasets:list",
    {},
  );
  const selected =
    requested.size === 0
      ? registered
      : registered.filter((dataset) => requested.has(dataset.repo_id));
  const found = new Set(selected.map((dataset) => dataset.repo_id));
  const missing = [...requested].filter((repoId) => !found.has(repoId));
  if (missing.length > 0) {
    throw new Error(
      `Datasets are not registered in Policy Arena: ${missing.join(", ")}`,
    );
  }

  console.log(
    `Refreshing ${selected.length} dataset(s) in batches of ${BATCH_SIZE}`,
  );
  const results: RefreshResult[] = [];
  for (let start = 0; start < selected.length; start += BATCH_SIZE) {
    const batch = selected.slice(start, start + BATCH_SIZE);
    const requestedAt = new Map<string, number>();
    const queued = await Promise.allSettled(
      batch.map((dataset) =>
        machineMutation<number>(apiKey, "datasets/refreshStats", {
          repo_id: dataset.repo_id,
        }),
      ),
    );
    const waiting: RegisteredDataset[] = [];
    for (let index = 0; index < queued.length; index++) {
      const dataset = batch[index];
      const settled = queued[index];
      if (settled.status === "rejected") {
        const error =
          settled.reason instanceof Error
            ? settled.reason.message
            : String(settled.reason);
        console.error(`[error] ${dataset.repo_id}: ${error}`);
        results.push({ repoId: dataset.repo_id, status: "error", error });
        continue;
      }
      requestedAt.set(dataset.repo_id, settled.value);
      waiting.push(dataset);
      console.log(`[queued] ${dataset.repo_id}`);
    }
    results.push(...(await waitForBatch(waiting, requestedAt)));
    const ready = results.filter((result) => result.status === "ready").length;
    const errors = results.filter((result) => result.status === "error").length;
    const timeouts = results.filter(
      (result) => result.status === "timeout",
    ).length;
    console.log(
      `[progress] ${results.length}/${selected.length} complete, ${ready} ready, ${errors} error, ${timeouts} timeout`,
    );
  }

  const ready = results.filter((result) => result.status === "ready");
  const errors = results.filter((result) => result.status === "error");
  const timeouts = results.filter((result) => result.status === "timeout");
  console.log(
    `[summary] ${results.length} processed, ${ready.length} ready, ${errors.length} error, ${timeouts.length} timeout`,
  );
  for (const result of [...errors, ...timeouts]) {
    console.error(
      `[failed] ${result.repoId}: ${result.error ?? "Timed out waiting for refresh"}`,
    );
  }
  if (errors.length > 0 || timeouts.length > 0) {
    process.exitCode = 1;
  }
}

await main();
