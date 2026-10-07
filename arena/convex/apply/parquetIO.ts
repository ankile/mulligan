"use node";

/**
 * Parquet read/write for LeRobot dataset files via parquet-wasm + apache-arrow.
 *
 * Node-runtime only ("use node" actions and tests) — never import from Convex
 * default-runtime functions. Round trips preserve the pyarrow schema
 * (including the huggingface metadata key) and values bit for bit on LeRobot
 * data and meta/episodes files.
 */

import {
  readParquet,
  readSchema,
  writeParquet,
  transformParquetStream,
  ParquetFile,
  WriterPropertiesBuilder,
  Compression,
  Table as WasmTable,
} from "parquet-wasm";
import type { RecordBatch as WasmRecordBatch } from "parquet-wasm";
import {
  Table,
  RecordBatch,
  Schema,
  tableFromIPC,
  tableToIPC,
  makeVector,
  makeBuilder,
  makeData,
  Data,
  Struct,
  Field,
  Type,
} from "apache-arrow";
import type { FileFrameColumns } from "./frames";

const STREAM_BATCH_ROWS = 4096;
const STREAM_ROW_GROUP_ROWS = 8192;

export function readArrowTable(buf: Uint8Array): Table {
  // Files may carry several record batches (LeRobot appends meta/episodes in
  // per-session row groups); all rebuilds below are batch-aligned.
  return tableFromIPC(readParquet(buf).intoIPCStream());
}

/** Whole-table write — meta/episodes files only (a few KB). Data files go through
 * rewriteEditColumnsStreaming: a whole-table read+write of one ~20 MB data file
 * peaks at ~180 MB of wasm linear memory (never returned to the OS), too much
 * for the 512 MiB Convex node action on large datasets. */
export function writeArrowTable(table: Table, createdBy?: string): Uint8Array {
  let builder = new WriterPropertiesBuilder().setCompression(Compression.SNAPPY);
  if (createdBy !== undefined) builder = builder.setCreatedBy(createdBy);
  const props = builder.build();
  return writeParquet(WasmTable.fromIPCStream(tableToIPC(table, "stream")), props);
}

/**
 * Rewrite one data file with the edited columns swapped in, streaming
 * batch-by-batch through wasm so memory is bounded by one batch plus the
 * writer's in-progress row group (~60 MB wasm peak on the R8 parent vs ~180 MB
 * whole-table). The streamed batches drop the file's schema metadata, so the
 * `huggingface` pandas metadata is re-attached from readSchema — verified
 * value- and metadata-equal to pyarrow reads of the whole-table path.
 */
export async function rewriteEditColumnsStreaming(
  path: string,
  buf: Uint8Array,
  cols: FileFrameColumns,
  createdBy?: string
): Promise<Uint8Array> {
  const fileSchema = fileArrowSchema(buf);
  const file = await ParquetFile.fromFile(new Blob([buf as unknown as BlobPart]));
  const input: ReadableStream<WasmRecordBatch> = await file.stream({ batchSize: STREAM_BATCH_ROWS });
  let offset = 0;
  const rebuilt = new ReadableStream<WasmRecordBatch>({
    async start(controller) {
      const reader = input.getReader();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          const batchTable = tableFromIPC(value.intoIPCStream());
          const withMeta = new Table(
            new Schema(batchTable.schema.fields, fileSchema.metadata),
            batchTable.batches
          );
          const edited = replaceEditColumns(withMeta, cols, offset);
          offset += batchTable.numRows;
          const wasmTable = WasmTable.fromIPCStream(tableToIPC(edited, "stream"));
          for (const rb of wasmTable.recordBatches()) controller.enqueue(rb);
          // wasm-bindgen objects are otherwise released only when JS GC runs
          // their finalizer; the batches hold their own refcounts.
          wasmTable.free();
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });
  let builder = new WriterPropertiesBuilder()
    .setCompression(Compression.SNAPPY)
    .setMaxRowGroupSize(STREAM_ROW_GROUP_ROWS);
  if (createdBy !== undefined) builder = builder.setCreatedBy(createdBy);
  const props = builder.build();
  const output: ReadableStream<Uint8Array> = await transformParquetStream(rebuilt, props);
  const chunks: Uint8Array[] = [];
  let total = 0;
  const outReader = output.getReader();
  for (;;) {
    const { done, value } = await outReader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  file.free();
  if (offset !== cols.numRows) {
    throw new Error(`${path}: streamed ${offset} rows, expected ${cols.numRows}`);
  }
  const out = new Uint8Array(total);
  let pos = 0;
  for (const chunk of chunks) {
    out.set(chunk, pos);
    pos += chunk.length;
  }
  return out;
}

function columnAsFloat64(table: Table, name: string, path: string): Float64Array {
  const col = table.getChild(name);
  if (!col) throw new Error(`${path}: missing required column ${name}`);
  const out = new Float64Array(table.numRows);
  const raw = col.toArray() as ArrayLike<number | bigint>;
  for (let i = 0; i < table.numRows; i++) out[i] = Number(raw[i]);
  return out;
}

const FRAME_COLUMNS = ["episode_index", "frame_index", "reward", "done", "success"] as const;

/** Arrow schema (fields + the `huggingface` pandas metadata) of a parquet file. */
function fileArrowSchema(buf: Uint8Array): Schema {
  return tableFromIPC(readSchema(buf).intoIPCStream()).schema;
}

/**
 * Column-projected, batch-streamed read of the scalar edit columns of one data
 * file (~7 MB wasm on the R8 parent's 37k-row file). The synchronous
 * `readParquet(buf, {columns})` IGNORES the projection (decodes all 41 columns,
 * ~91 MB wasm) and `ParquetFile.read({columns})` produced IPC that apache-arrow
 * could not load; only `ParquetFile.stream({columns})` projects correctly.
 */
export async function readFrameColumns(path: string, buf: Uint8Array): Promise<FileFrameColumns> {
  const hasIsValid = fileArrowSchema(buf).fields.some((f) => f.name === "is_valid");
  const columns = hasIsValid ? [...FRAME_COLUMNS, "is_valid"] : [...FRAME_COLUMNS];
  const file = await ParquetFile.fromFile(new Blob([buf as unknown as BlobPart]));
  const stream: ReadableStream<WasmRecordBatch> = await file.stream({ columns, batchSize: STREAM_BATCH_ROWS });
  const parts: FileFrameColumns[] = [];
  const reader = stream.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const table = tableFromIPC(value.intoIPCStream());
    if (table.schema.fields.length !== columns.length) {
      throw new Error(
        `${path}: projected read returned ${table.schema.fields.length} columns, expected ${columns.length}`
      );
    }
    parts.push(extractFrameColumns(path, table));
  }
  file.free();
  const numRows = parts.reduce((n, p) => n + p.numRows, 0);
  const concat = <T extends Float64Array | Float32Array>(make: (n: number) => T, pick: (p: FileFrameColumns) => T): T => {
    const out = make(numRows);
    let pos = 0;
    for (const p of parts) {
      out.set(pick(p), pos);
      pos += p.numRows;
    }
    return out;
  };
  return {
    path,
    numRows,
    episodeIndex: concat((n) => new Float64Array(n), (p) => p.episodeIndex),
    frameIndex: concat((n) => new Float64Array(n), (p) => p.frameIndex),
    reward: concat((n) => new Float32Array(n), (p) => p.reward),
    done: concat((n) => new Float64Array(n), (p) => p.done),
    success: concat((n) => new Float64Array(n), (p) => p.success),
    isValid: hasIsValid ? concat((n) => new Float64Array(n), (p) => p.isValid!) : null,
    dirty: false,
  };
}

/** Extract the scalar edit columns from one data-file table. */
export function extractFrameColumns(path: string, table: Table): FileFrameColumns {
  const rewardCol = table.getChild("reward");
  if (!rewardCol) throw new Error(`${path}: missing required column reward`);
  if (rewardCol.type.typeId !== Type.Float) {
    throw new Error(`${path}: reward column is not a float type`);
  }
  const hasIsValid = table.schema.fields.some((f) => f.name === "is_valid");
  return {
    path,
    numRows: table.numRows,
    episodeIndex: columnAsFloat64(table, "episode_index", path),
    frameIndex: columnAsFloat64(table, "frame_index", path),
    reward: Float32Array.from(rewardCol.toArray() as ArrayLike<number>),
    done: columnAsFloat64(table, "done", path),
    success: columnAsFloat64(table, "success", path),
    isValid: hasIsValid ? columnAsFloat64(table, "is_valid", path) : null,
    dirty: false,
  };
}

function int64Data(values: Float64Array, field: Field): Data {
  const big = new BigInt64Array(values.length);
  for (let i = 0; i < values.length; i++) {
    if (!Number.isInteger(values[i])) throw new Error(`Non-integer int64 value ${values[i]}`);
    big[i] = BigInt(values[i]);
  }
  return makeVector({ data: big, type: field.type }).data[0];
}

/** Rebuild a table batch-by-batch with per-field replacement Data providers.
 * Reuses the ORIGINAL schema object, so field types, nullability, and the
 * huggingface pandas metadata are preserved exactly. */
function rebuildTable(
  table: Table,
  makeReplacement: (fieldName: string, batchStart: number, batchLength: number, batchIndex: number) => Data | null,
  baseOffset = 0
): Table {
  const schema = table.schema;
  const batches: RecordBatch[] = [];
  let offset = baseOffset;
  for (let b = 0; b < table.batches.length; b++) {
    const batch = table.batches[b];
    const n = batch.numRows;
    const children: Data[] = schema.fields.map((field) => {
      const replacement = makeReplacement(field.name, offset, n, b);
      if (replacement !== null) {
        if (replacement.length !== n) {
          throw new Error(
            `${field.name}: replacement batch length ${replacement.length} != ${n}`
          );
        }
        return replacement;
      }
      const child = batch.getChild(field.name);
      if (!child) throw new Error(`missing column ${field.name} in batch ${b}`);
      if (child.data.length !== 1) {
        throw new Error(`${field.name}: batch child has ${child.data.length} data chunks`);
      }
      return child.data[0];
    });
    const structData = makeData({
      type: new Struct(schema.fields),
      length: n,
      nullCount: 0,
      children,
    });
    batches.push(new RecordBatch(schema, structData));
    offset += n;
  }
  return new Table(schema, batches);
}

/** Rebuild a data-file table (or a batch slice of one starting at file row
 * `baseOffset`) with the edited columns swapped in. */
export function replaceEditColumns(table: Table, cols: FileFrameColumns, baseOffset = 0): Table {
  const fieldByName = new Map(table.schema.fields.map((f) => [f.name, f]));
  return rebuildTable(table, (name, start, length) => {
    if (name === "reward") {
      return makeVector({ data: cols.reward.slice(start, start + length), type: fieldByName.get(name)!.type }).data[0];
    }
    if (name === "done") return int64Data(cols.done.slice(start, start + length), fieldByName.get(name)!);
    if (name === "success") return int64Data(cols.success.slice(start, start + length), fieldByName.get(name)!);
    if (name === "is_valid" && cols.isValid !== null) {
      return int64Data(cols.isValid.slice(start, start + length), fieldByName.get(name)!);
    }
    return null;
  }, baseOffset);
}

/**
 * Rebuild selected list-typed cells of a table (meta/episodes stats patch).
 * `patches` maps column name → (rowPos → new cell values). Unpatched rows keep
 * their original cells; the column's arrow type (list<double> / list<int64>)
 * is preserved from the original field.
 */
export function patchListColumns(
  table: Table,
  patches: Map<string, Map<number, number[]>>
): Table {
  for (const name of patches.keys()) {
    const field = table.schema.fields.find((f) => f.name === name);
    if (!field) throw new Error(`missing column ${name}`);
    if (field.type.typeId !== Type.List) {
      throw new Error(`${name}: expected a list column for stats patch, got ${field.type}`);
    }
  }
  return rebuildTable(table, (name, start, length, batchIndex) => {
    const patch = patches.get(name);
    if (!patch) return null;
    const field = table.schema.fields.find((f) => f.name === name)!;
    const childType = (field.type.children as Field[])[0].type;
    const isInt = childType.typeId === Type.Int;
    const builder = makeBuilder({ type: field.type, nullValues: [null] });
    const original = table.batches[batchIndex].getChild(name)!;
    for (let i = 0; i < length; i++) {
      const cell = patch.get(start + i);
      if (cell !== undefined) {
        builder.append(isInt ? (cell.map((v) => BigInt(v)) as unknown as number[]) : cell);
      } else {
        const orig = original.get(i);
        if (orig === null) throw new Error(`${name}: unexpected null cell at row ${start + i}`);
        builder.append(orig.toArray());
      }
    }
    return builder.finish().toVector().data[0];
  });
}

/** Column of numbers from a possibly-bigint arrow column (meta reads). */
export function numberColumn(table: Table, name: string): number[] {
  const col = table.getChild(name);
  if (!col) throw new Error(`missing column ${name}`);
  const out: number[] = [];
  for (let i = 0; i < table.numRows; i++) out.push(Number(col.get(i)));
  return out;
}

/** List-of-strings column (meta/episodes `tasks`, meta/tasks task names). */
export function stringListColumn(table: Table, name: string): string[][] {
  const col = table.getChild(name);
  if (!col) throw new Error(`missing column ${name}`);
  const out: string[][] = [];
  for (let i = 0; i < table.numRows; i++) {
    const cell = col.get(i);
    if (cell === null) throw new Error(`${name}: null cell at row ${i}`);
    out.push([...cell].map((v: unknown) => String(v)));
  }
  return out;
}

export function stringColumn(table: Table, name: string): string[] {
  const col = table.getChild(name);
  if (!col) throw new Error(`missing column ${name}`);
  const out: string[] = [];
  for (let i = 0; i < table.numRows; i++) out.push(String(col.get(i)));
  return out;
}

/** Read one list<float64-or-int64> cell as numbers. */
export function listCell(table: Table, name: string, row: number): number[] {
  const col = table.getChild(name);
  if (!col) throw new Error(`missing column ${name}`);
  const cell = col.get(row);
  if (cell === null) throw new Error(`${name}: null cell at row ${row}`);
  return [...cell].map((v: unknown) => Number(v));
}
