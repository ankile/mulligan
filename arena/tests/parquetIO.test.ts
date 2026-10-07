import { describe, expect, test } from "bun:test";
import {
  Table,
  Schema,
  Field,
  Float32,
  Int64,
  FixedSizeList,
  tableFromIPC,
  tableToIPC,
  vectorFromArray,
  makeTable,
} from "apache-arrow";
import { readParquet } from "parquet-wasm";
import {
  writeArrowTable,
  readArrowTable,
  readFrameColumns,
  replaceEditColumns,
  rewriteEditColumnsStreaming,
} from "../convex/apply/parquetIO";

const N = 10_000; // > STREAM_BATCH_ROWS + STREAM_ROW_GROUP_ROWS so batching/row-group flushes are exercised

function makeDataFile(): Uint8Array {
  const ints = (f: (i: number) => number) =>
    vectorFromArray(BigInt64Array.from({ length: N }, (_, i) => BigInt(f(i))), new Int64());
  const floats = (f: (i: number) => number) => vectorFromArray(Float32Array.from({ length: N }, (_, i) => f(i)), new Float32());
  const obs = vectorFromArray(
    Array.from({ length: N }, (_, i) => Float32Array.from([i * 0.5, -i, 1.25])),
    new FixedSizeList(3, new Field("element", new Float32(), true))
  );
  const table = makeTable({
    "observation.state": obs,
    success: ints(() => 0),
    reward: floats(() => 0),
    done: ints(() => 0),
    is_valid: ints(() => 1),
    frame_index: ints((i) => i % 100),
    episode_index: ints((i) => Math.floor(i / 100)),
  });
  const withMeta = new Table(
    new Schema(table.schema.fields, new Map([["huggingface", '{"info": {"features": {}}}']])),
    table.batches
  );
  return writeArrowTable(withMeta);
}

describe("parquetIO data-file rewrite", () => {
  test("streaming rewrite equals whole-table rewrite (values + schema metadata)", async () => {
    const buf = makeDataFile();
    const cols = await readFrameColumns("data/chunk-000/file-000.parquet", buf);
    expect(cols.numRows).toBe(N);
    expect(cols.isValid).not.toBeNull();
    // Edit: episode 7 becomes a success at frame 40 with a soft-truncated tail.
    for (let r = 0; r < N; r++) {
      if (cols.episodeIndex[r] !== 7) continue;
      cols.success[r] = 1;
      const f = cols.frameIndex[r];
      cols.reward[r] = f >= 40 ? 1 : 0;
      cols.done[r] = f >= 40 ? 1 : 0;
      cols.isValid![r] = f > 40 ? 0 : 1;
    }
    cols.dirty = true;

    const whole = writeArrowTable(replaceEditColumns(readArrowTable(buf), cols));
    const streamed = await rewriteEditColumnsStreaming("data/chunk-000/file-000.parquet", buf, cols);

    const a = readArrowTable(whole);
    const b = readArrowTable(streamed);
    expect(b.numRows).toBe(N);
    expect(b.schema.fields.map((f) => `${f.name}:${f.type}`)).toEqual(
      a.schema.fields.map((f) => `${f.name}:${f.type}`)
    );
    expect([...b.schema.metadata.entries()]).toEqual([...a.schema.metadata.entries()]);
    for (const name of ["success", "reward", "done", "is_valid", "frame_index", "episode_index"]) {
      expect([...b.getChild(name)!.toArray()]).toEqual([...a.getChild(name)!.toArray()]);
    }
    const obsA = a.getChild("observation.state")!;
    const obsB = b.getChild("observation.state")!;
    for (const r of [0, 4095, 4096, 8191, 8192, N - 1]) {
      expect([...obsB.get(r)!]).toEqual([...obsA.get(r)!]);
    }
    // Edited values landed.
    const successB = b.getChild("success")!.toArray();
    const epB = b.getChild("episode_index")!.toArray();
    for (let r = 0; r < N; r++) {
      expect(Number(successB[r])).toBe(Number(epB[r]) === 7 ? 1 : 0);
    }
    // The projected read agrees with the parquet-wasm full read of the same file.
    const full = tableFromIPC(readParquet(streamed).intoIPCStream());
    expect(full.numRows).toBe(N);
  });

  test("streaming rewrite fails loud on a row-count mismatch", async () => {
    const buf = makeDataFile();
    const cols = await readFrameColumns("f.parquet", buf);
    cols.numRows = N + 1;
    await expect(rewriteEditColumnsStreaming("f.parquet", buf, cols)).rejects.toThrow(/streamed 10000 rows, expected 10001/);
  });

  test("validation footer forces fresh uploads without changing schemas or values", async () => {
    const buf = makeDataFile();
    const cols = await readFrameColumns("f.parquet", buf);
    const normal = await rewriteEditColumnsStreaming("f.parquet", buf, cols);
    const validation = await rewriteEditColumnsStreaming("f.parquet", buf, cols, "validation-job-1");
    const second = await rewriteEditColumnsStreaming("f.parquet", buf, cols, "validation-job-2");
    expect(Buffer.compare(Buffer.from(normal), Buffer.from(validation))).not.toBe(0);
    expect(Buffer.compare(Buffer.from(validation), Buffer.from(second))).not.toBe(0);
    const a = readArrowTable(normal);
    const b = readArrowTable(validation);
    expect([...b.schema.metadata.entries()]).toEqual([...a.schema.metadata.entries()]);
    expect(tableToIPC(b, "stream")).toEqual(tableToIPC(a, "stream"));
    const metaNormal = writeArrowTable(a);
    const metaValidation = writeArrowTable(a, "validation-job-1");
    expect(Buffer.compare(Buffer.from(metaNormal), Buffer.from(metaValidation))).not.toBe(0);
    expect(tableToIPC(readArrowTable(metaValidation), "stream")).toEqual(tableToIPC(readArrowTable(metaNormal), "stream"));
  });
});
