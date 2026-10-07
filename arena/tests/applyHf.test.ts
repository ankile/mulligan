import { describe, expect, test } from "bun:test";
import { commitFiles } from "../convex/apply/hf";

describe("apply upload transport", () => {
  test("uploads through basic LFS and commits all files atomically against the pinned parent", async () => {
    const uploaded: string[] = [];
    let committed = false;
    const phases: string[] = [];
    const fetchMock = async (input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/preupload/")) {
        const body = JSON.parse(String(init!.body));
        return Response.json({ files: body.files.map((f: { path: string }) => ({ ...f, uploadMode: "lfs" })) });
      }
      if (url.endsWith("/objects/batch")) {
        const body = JSON.parse(String(init!.body));
        // Xet allocates five 64 MiB shard buffers in hub 2.16.1, in addition
        // to the resident parquet data. It cannot be offered by this worker.
        expect(body.transfers).not.toContain("xet");
        return Response.json({ transfer: "basic", objects: body.objects.map((o: { oid: string }) => ({
          ...o, actions: { upload: { href: `https://upload.test/${o.oid}` } },
        })) });
      }
      if (url.startsWith("https://upload.test/")) {
        uploaded.push(await (init!.body as Blob).text());
        return new Response(null, { status: 200 });
      }
      if (url.endsWith("/commit/apply-validation%2Ftest")) {
        expect(uploaded.sort()).toEqual(["first", "second"]);
        const lines = String(init!.body).split("\n").map((line) => JSON.parse(line));
        expect(lines[0].value.parentCommit).toBe("before");
        expect(lines.slice(1).map((l) => l.value.path)).toEqual(["a.parquet", "b.parquet"]);
        committed = true;
        return Response.json({ commitOid: "after", commitUrl: "https://hf.test/after" });
      }
      throw new Error(`Unexpected request ${url}`);
    };
    const sha = await commitFiles({
      client: { repoId: "test/dataset", token: "hf_test_only_token" },
      files: new Map([["a.parquet", "first"], ["b.parquet", "second"]]),
      message: "test", parentCommit: "before", branch: "apply-validation/test",
      fetch: fetchMock as typeof fetch,
      onProgress: async (phase) => { phases.push(phase); },
    });
    expect(sha).toBe("after");
    expect(committed).toBe(true);
    expect(phases).toEqual(["Upload: preuploading", "Upload: uploadingLargeFiles", "Upload: 2 new LFS objects via basic", "Upload: committing"]);
  });
});
