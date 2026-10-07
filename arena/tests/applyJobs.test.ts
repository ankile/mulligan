import { describe, expect, test } from "bun:test";
import { convexTest } from "convex-test";
import schema from "../convex/schema";
import { internal } from "../convex/_generated/api";
import { resolveUploadBranch } from "../convex/applyWorker";

import {
  isStaleNativeApplyJob,
  NATIVE_STALE_REAPER_DELAY_MS,
  STALE_APPLYING_MS,
} from "../convex/applyJobs";

describe("native apply stale-job watchdog", () => {
  const now = 1_000_000_000;

  test("reaper runs after the stale threshold", () => {
    expect(NATIVE_STALE_REAPER_DELAY_MS).toBeGreaterThan(STALE_APPLYING_MS);
  });

  test("expires only stale native applying jobs", () => {
    expect(
      isStaleNativeApplyJob(
        {
          status: "applying",
          worker_id: "convex-action",
          started_at: now - STALE_APPLYING_MS - 1,
        },
        now
      )
    ).toBe(true);

    for (const job of [
      {
        status: "applying",
        worker_id: "convex-action",
        started_at: now - STALE_APPLYING_MS,
      },
      {
        status: "applied",
        worker_id: "convex-action",
        started_at: now - STALE_APPLYING_MS - 1,
      },
      {
        status: "applying",
        worker_id: "python-worker",
        started_at: now - STALE_APPLYING_MS - 1,
      },
      null,
    ]) {
      expect(isStaleNativeApplyJob(job, now)).toBe(false);
    }
  });
});

test("watchdog retains the last phase and commit anchors when the native process dies", async () => {
  const t = convexTest(schema, {
    "../convex/_generated/server.ts": () => import("../convex/_generated/server"),
    "../convex/applyJobs.ts": () => import("../convex/applyJobs"),
  });
  const id = await t.run((ctx) => ctx.db.insert("applyJobs", {
    dataset_repo: "test/recovery", requested_at: 0, requested_by: "test",
    status: "applying", worker_id: "convex-action", started_at: 0,
  }));
  await t.mutation(internal.applyJobs.recordProgressInternal, {
    id, message: "Reading snapshot (RSS 270 MiB)", pre_apply_sha: "before",
  });
  await t.mutation(internal.applyJobs.recordProgressInternal, {
    id, message: "Committed (RSS 350 MiB)", hf_commit_sha: "after",
  });
  await t.mutation(internal.applyJobs.recordProgressInternal, {
    id, message: "Updating v3.0",
  });
  await t.mutation(internal.applyJobs.failStaleNativeInternal, { id });
  const job = await t.run((ctx) => ctx.db.get(id));
  expect(job!.status).toBe("failed");
  expect(job!.pre_apply_sha).toBe("before");
  expect(job!.hf_commit_sha).toBe("after");
  expect(job!.log_tail).toContain("Updating v3.0");
  await expect(t.mutation(internal.applyJobs.recordProgressInternal, {
    id, message: "late progress",
  })).rejects.toThrow("Only an active native apply");
});

test("validation uploads cannot target main or be used for a production apply", () => {
  expect(resolveUploadBranch(false)).toBe("main");
  expect(resolveUploadBranch(true, "before")).toBeNull();
  expect(resolveUploadBranch(true, "before", "apply-validation/test")).toBe("apply-validation/test");
  expect(() => resolveUploadBranch(true, "before", "main")).toThrow("apply-validation/");
  expect(() => resolveUploadBranch(true, undefined, "apply-validation/test")).toThrow("pinned revision");
  expect(() => resolveUploadBranch(false, "before")).toThrow("dry-run");
  expect(() => resolveUploadBranch(false, undefined, "apply-validation/test")).toThrow("dry-run");
});

test("reporting a terminal failure preserves previously recorded diagnostics", async () => {
  const t = convexTest(schema, {
    "../convex/_generated/server.ts": () => import("../convex/_generated/server"),
    "../convex/applyJobs.ts": () => import("../convex/applyJobs"),
  });
  const id = await t.run((ctx) => ctx.db.insert("applyJobs", {
    dataset_repo: "test/recovery", requested_at: 0, requested_by: "test",
    status: "applying", worker_id: "convex-action", started_at: 0,
  }));
  await t.mutation(internal.applyJobs.recordProgressInternal, {
    id, message: "HF committed; tag update pending", pre_apply_sha: "before", hf_commit_sha: "after",
  });
  await t.mutation(internal.applyJobs.finishInternal, { id, ok: false, error: "terminated" });
  const job = await t.run((ctx) => ctx.db.get(id));
  expect(job!.status).toBe("failed");
  expect(job!.pre_apply_sha).toBe("before");
  expect(job!.hf_commit_sha).toBe("after");
  expect(job!.log_tail).toBe("HF committed; tag update pending");
});
