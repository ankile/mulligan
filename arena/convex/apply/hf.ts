"use node";

/**
 * HuggingFace Hub access for the apply pipeline (node runtime only).
 *
 * Uses @huggingface/hub for listing/commits (LFS-aware) and raw fetch for the
 * endpoints the lib does not cover (repo sha, tags). The LeRobot codebase
 * version tag is moved onto every verified push (advanceLerobotVersionTag) —
 * training and metadata reads resolve v3.0, never raw main.
 */

import { listFiles, downloadFile, commitIter } from "@huggingface/hub";
import type { CommitOperation } from "@huggingface/hub";

/** lerobot.datasets.lerobot_dataset.CODEBASE_VERSION — bump with LeRobot. */
export const CODEBASE_VERSION = "v3.0";

const HUB = "https://huggingface.co";

export interface HfClient {
  repoId: string;
  token: string;
}

function authHeaders(client: HfClient): Record<string, string> {
  return { Authorization: `Bearer ${client.token}` };
}

async function checkOk(res: Response, what: string): Promise<void> {
  if (!res.ok) {
    const body = await res.text().catch(() => "<unreadable>");
    throw new Error(`${what} failed: HTTP ${res.status} ${body.slice(0, 500)}`);
  }
}

/** Current commit sha of a revision (HfApi().dataset_info(repo).sha). */
export async function revisionSha(client: HfClient, revision = "main"): Promise<string> {
  const res = await fetch(
    `${HUB}/api/datasets/${client.repoId}/revision/${encodeURIComponent(revision)}`,
    { headers: authHeaders(client) }
  );
  await checkOk(res, `resolve ${client.repoId}@${revision}`);
  const info = (await res.json()) as { sha?: string };
  if (!info.sha) throw new Error(`No sha in revision info for ${client.repoId}@${revision}`);
  return info.sha;
}

export async function listRepoFiles(
  client: HfClient,
  revision = "main",
  path?: string
): Promise<string[]> {
  const paths: string[] = [];
  for await (const entry of listFiles({
    repo: { type: "dataset", name: client.repoId },
    revision,
    recursive: true,
    path,
    accessToken: client.token,
  })) {
    if (entry.type === "file") paths.push(entry.path);
  }
  return paths;
}

export async function downloadRepoFile(
  client: HfClient,
  path: string,
  revision = "main"
): Promise<Uint8Array> {
  const res = await downloadFile({
    repo: { type: "dataset", name: client.repoId },
    path,
    revision,
    accessToken: client.token,
  });
  if (!res) throw new Error(`${client.repoId}@${revision}: file not found: ${path}`);
  return new Uint8Array(await res.arrayBuffer());
}

export async function downloadRepoText(
  client: HfClient,
  path: string,
  revision = "main"
): Promise<string> {
  return new TextDecoder().decode(await downloadRepoFile(client, path, revision));
}

/**
 * One atomic commit with every changed file. `parentCommit` pins the base so
 * a concurrent push to main fails this commit loudly instead of silently
 * overwriting it (the retried job then re-applies from the new state).
 * Returns the new commit sha.
 */
export async function commitFiles(args: {
  client: HfClient;
  files: Map<string, Uint8Array | string>;
  message: string;
  parentCommit: string;
  branch?: string;
  fetch?: typeof fetch;
  onProgress?: (message: string) => Promise<void>;
}): Promise<string> {
  if (args.files.size === 0) throw new Error("commitFiles called with no files");
  const operations: CommitOperation[] = [...args.files.entries()].map(([path, content]) => {
    const bytes = typeof content === "string" ? new TextEncoder().encode(content) : content;
    return {
      operation: "addOrUpdate" as const,
      path,
      // Blob copies the view's bytes itself; no intermediate exact-length copy.
      content: new Blob([bytes as unknown as BlobPart]),
    };
  });
  const iterator = commitIter({
    repo: { type: "dataset", name: args.client.repoId },
    accessToken: args.client.token,
    branch: args.branch ?? "main",
    parentCommit: args.parentCommit,
    title: args.message,
    operations,
    fetch: async (input, init) => {
      const response = await (args.fetch ?? fetch)(input, init);
      if (response.ok && String(input).endsWith("/info/lfs/objects/batch")) {
        const batch = await response.clone().json() as {
          transfer?: string;
          objects: Array<{ actions?: { upload?: unknown } }>;
        };
        const fresh = batch.objects.filter((object) => object.actions?.upload !== undefined).length;
        await args.onProgress?.(`Upload: ${fresh} new LFS objects via ${batch.transfer ?? "basic"}`);
      }
      return response;
    },
    // hub 2.16.1 starts five Xet shard builders, each allocating ~64 MiB
    // before chunking. Use basic/multipart LFS within the 512 MiB action.
    useXet: false,
  });
  let next = await iterator.next();
  while (!next.done) {
    if (next.value.event === "phase") {
      await args.onProgress?.(`Upload: ${next.value.phase}`);
    }
    next = await iterator.next();
  }
  const oid = next.value?.commit?.oid;
  if (!oid) throw new Error("commit returned no oid");
  return oid;
}

/** advance_lerobot_version_tag: move v3.0 onto the given main sha. */
export async function advanceLerobotVersionTag(
  client: HfClient,
  targetSha: string,
  tag = CODEBASE_VERSION
): Promise<void> {
  const del = await fetch(`${HUB}/api/datasets/${client.repoId}/tag/${encodeURIComponent(tag)}`, {
    method: "DELETE",
    headers: authHeaders(client),
  });
  // 404 = tag did not exist yet (RevisionNotFoundError pass in Python).
  if (!del.ok && del.status !== 404) {
    await checkOk(del, `delete tag ${tag} on ${client.repoId}`);
  }
  const create = await fetch(
    `${HUB}/api/datasets/${client.repoId}/tag/${encodeURIComponent(targetSha)}`,
    {
      method: "POST",
      headers: { ...authHeaders(client), "Content-Type": "application/json" },
      body: JSON.stringify({ tag }),
    }
  );
  await checkOk(create, `create tag ${tag}@${targetSha.slice(0, 8)} on ${client.repoId}`);
}
