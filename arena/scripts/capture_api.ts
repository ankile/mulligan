/**
 * Keep src/release/api.ts in step with the Convex backend in convex/.
 *
 *   bun scripts/capture_api.ts           fail if src/release/api.ts differs from a fresh capture
 *   bun scripts/capture_api.ts --write   rewrite src/release/api.ts from convex/
 *
 * The shared screens import `api`, `Id` and `Doc` from src/release/api.ts, so the read-only
 * release build never type-checks the backend. This script captures, for every `api.<module>.<fn>`
 * the shared screens reference, the argument and result types of that function in convex/, and for
 * every `Doc<"table">` they use, the document type. Run it with --write after changing a backend
 * function the screens call. At runtime `api` stays Convex's `anyApi` proxy, so the capture changes
 * types only, never the built JavaScript.
 */
import ts from "typescript";
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
const TARGET = join(ROOT, "src/release/api.ts");
const PROBE = join(ROOT, "src/release/__api_probe__.ts");
const CLOSURE = join(ROOT, "release-closure.txt");

const HEADER = `/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-empty-object-type -- captured backend types, kept verbatim */
/**
 * Function references for the shared screens, typed from the Convex backend.
 *
 * At runtime \`api\` is Convex's \`anyApi\` proxy, the same object the backend's generated \`api\`
 * module exports: \`api.policies.leaderboard\` is a reference whose name is the string key
 * "policies:leaderboard". In the read-only release build \`client.ts\` hands that key to the
 * frozen-export adapter (\`adapter.ts\`), which serves the release queries and throws for everything
 * else. In the self-deploy build (vite.config.ts) the same references go to the live deployment.
 * No backend module is imported, so the release build never type-checks convex/.
 *
 * The argument and result types below are captured from convex/ by scripts/capture_api.ts
 * (\`FunctionArgs\` / \`FunctionReturnType\` of each function the shared screens reference). Do not
 * edit them by hand: run \`bun scripts/capture_api.ts --write\` after changing those functions.
 */
import { anyApi } from "convex/server";
import type { DefaultFunctionArgs, FunctionReference } from "convex/server";
import type { GenericId } from "convex/values";
import type { PairOutcome } from "../../convex/bradleyTerry";

export type Id<TableName extends string> = GenericId<TableName>;
export type Doc<TableName extends keyof Documents> = Documents[TableName];
type Query<Args extends DefaultFunctionArgs, Result> = FunctionReference<"query", "public", Args, Result>;
type Mutation<Args extends DefaultFunctionArgs, Result> = FunctionReference<"mutation", "public", Args, Result>;

export const api = anyApi as unknown as ReleaseApi;
`;

/** `api.<module>.<fn>` and `Doc<"table">` references in the TypeScript files of the release closure. */
function references(): { functions: [string, string][]; documents: string[] } {
  const files = readFileSync(CLOSURE, "utf8").split("\n").filter((f) => /\.tsx?$/.test(f) && f !== "src/release/api.ts");
  const functions = new Set<string>();
  const documents = new Set<string>();
  for (const file of files) {
    const text = readFileSync(join(ROOT, file), "utf8");
    for (const m of text.matchAll(/\bapi\.([A-Za-z]\w*)\.([A-Za-z]\w*)/g)) functions.add(`${m[1]}.${m[2]}`);
    for (const m of text.matchAll(/\bDoc<"([A-Za-z]\w*)">/g)) documents.add(m[1]);
  }
  return {
    functions: [...functions].sort().map((f) => f.split(".") as [string, string]),
    documents: [...documents].sort(),
  };
}

function capture(): string {
  const { functions, documents } = references();
  const probe = [
    `import type { FunctionArgs, FunctionReturnType } from "convex/server";`,
    `import type { api } from "../../convex/_generated/api";`,
    ...(documents.length ? [`import type { Doc } from "../../convex/_generated/dataModel";`] : []),
    ...functions.flatMap(([m, f]) => [
      `export type A__${m}__${f} = FunctionArgs<typeof api.${m}.${f}>;`,
      `export type R__${m}__${f} = FunctionReturnType<typeof api.${m}.${f}>;`,
      `export type K__${m}__${f} = (typeof api.${m}.${f})["_type"];`,
    ]),
    ...documents.map((t) => `export type D__${t} = Doc<"${t}">;`),
  ].join("\n");
  const { config } = ts.readConfigFile(join(ROOT, "tsconfig.app.json"), ts.sys.readFile);
  const options = ts.parseJsonConfigFileContent({ ...config, include: [], files: [] }, ts.sys, ROOT).options;
  const host = ts.createCompilerHost(options);
  const readFile = host.readFile.bind(host);
  host.readFile = (name) => (resolve(name) === PROBE ? probe : readFile(name));
  const fileExists = host.fileExists.bind(host);
  host.fileExists = (name) => resolve(name) === PROBE || fileExists(name);
  const program = ts.createProgram([PROBE], options, host);
  const errors = ts.getPreEmitDiagnostics(program).filter((d) => d.category === ts.DiagnosticCategory.Error);
  if (errors.length)
    throw new Error(ts.formatDiagnostics(errors, { getCanonicalFileName: (f) => f, getCurrentDirectory: () => ROOT, getNewLine: () => "\n" }));
  const checker = program.getTypeChecker();
  const source = program.getSourceFile(PROBE)!;
  const aliases = new Map<string, ts.Type>();
  source.forEachChild((node) => {
    if (ts.isTypeAliasDeclaration(node)) aliases.set(node.name.text, checker.getTypeAtLocation(node.name));
  });
  const printer = ts.createPrinter({ removeComments: true });
  const flags =
    ts.NodeBuilderFlags.NoTruncation | ts.NodeBuilderFlags.MultilineObjectLiterals | ts.NodeBuilderFlags.UseAliasDefinedOutsideCurrentScope;
  const print = (name: string, indent: string): string => {
    const node = checker.typeToTypeNode(aliases.get(name)!, source, flags);
    if (!node) throw new Error(`cannot print ${name}`);
    return printer
      .printNode(ts.EmitHint.Unspecified, node, source)
      // Two Convex aliases the release module does not import, spelled out.
      .replace(/\bEmptyObject\b/g, "Record<string, never>")
      .replace(/\bCursor\b/g, "string")
      .replace(/^( {4})+/gm, (spaces) => "  ".repeat(spaces.length / 4))
      .split("\n")
      .join("\n" + indent);
  };
  const kind = (m: string, f: string) => {
    const k = checker.typeToString(aliases.get(`K__${m}__${f}`)!);
    if (k !== '"query"' && k !== '"mutation"') throw new Error(`${m}.${f} is a ${k}, not a query or mutation`);
    return k === '"query"' ? "Query" : "Mutation";
  };
  const lines = [HEADER, "export type ReleaseApi = {"];
  for (const module of [...new Set(functions.map(([m]) => m))]) {
    lines.push(`  ${module}: {`);
    for (const [, f] of functions.filter(([m]) => m === module))
      lines.push(`    ${f}: ${kind(module, f)}<${print(`A__${module}__${f}`, "    ")}, ${print(`R__${module}__${f}`, "    ")}>;`);
    lines.push("  };");
  }
  lines.push("};", "type Documents = {");
  for (const t of documents) lines.push(`  ${t}: ${print(`D__${t}`, "  ")};`);
  lines.push("};", "");
  return lines.join("\n");
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  if (args.length > 1 || (args.length === 1 && args[0] !== "--write"))
    throw new Error("Usage: bun scripts/capture_api.ts [--write]");
  const fresh = capture();
  if (args[0] === "--write") {
    writeFileSync(TARGET, fresh);
    console.log(`Wrote ${TARGET}`);
  } else if (readFileSync(TARGET, "utf8") !== fresh) {
    console.error("src/release/api.ts differs from the backend in convex/: run `bun scripts/capture_api.ts --write`");
    process.exit(1);
  } else {
    console.log("src/release/api.ts matches convex/");
  }
}
