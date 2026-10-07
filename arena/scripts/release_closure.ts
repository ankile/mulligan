/**
 * Walk the import graph of the release build and compare it with the committed allowed set.
 *
 *   bun scripts/release_closure.ts           check release-closure.txt, tsconfig.release.json and the tree
 *   bun scripts/release_closure.ts --write   regenerate release-closure.txt and the tsconfig include list
 *
 * Entries are the HTML inputs of vite.release.config.ts. Every relative import (static, dynamic,
 * type-only, re-export, CSS side effect, CSS @import) is followed; bare package imports stop at
 * node_modules. Release tests (src/**\/*.test.ts[x]) may only import files inside the closure, and
 * src/ holds nothing else: files used only by the self-deploy build live in app/ and convex/.
 */
import ts from "typescript";
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";

export const ROOT = resolve(import.meta.dir, "..");
export const ENTRIES = ["index.html", "overview.html"];
export const CLOSURE_FILE = join(ROOT, "release-closure.txt");
const TSCONFIG = join(ROOT, "tsconfig.release.json");
const TS_EXT = new Set([".ts", ".tsx", ".mts", ".cts"]);

const rel = (p: string) => relative(ROOT, p).split("\\").join("/");

function compilerOptions(): ts.CompilerOptions {
  const { config, error } = ts.readConfigFile(join(ROOT, "tsconfig.release.json"), ts.sys.readFile);
  if (error) throw new Error(ts.flattenDiagnosticMessageText(error.messageText, "\n"));
  return ts.parseJsonConfigFileContent({ ...config, include: [], files: [] }, ts.sys, ROOT).options;
}

function htmlScripts(file: string): string[] {
  const html = readFileSync(file, "utf8");
  const found = [...html.matchAll(/<script[^>]*\bsrc="([^"]+)"[^>]*>/g)].map((m) => m[1]);
  if (!found.length) throw new Error(`${rel(file)} has no module script`);
  return found.map((src) => {
    if (!src.startsWith("/")) throw new Error(`${rel(file)}: expected a root-relative script, got ${src}`);
    return join(ROOT, src.slice(1));
  });
}

function cssImports(file: string): string[] {
  const css = readFileSync(file, "utf8");
  return [...css.matchAll(/@import\s+(?:url\()?["']([^"']+)["']/g)].map((m) => m[1]);
}

function resolveImport(spec: string, from: string, options: ts.CompilerOptions): string[] {
  if (!spec.startsWith(".") && !spec.startsWith("/")) return []; // package import
  const direct = resolve(dirname(from), spec);
  if (!TS_EXT.has(extname(direct)) && existsSync(direct) && statSync(direct).isFile()) return [direct];
  const resolved = ts.resolveModuleName(spec, from, options, ts.sys).resolvedModule;
  if (!resolved) throw new Error(`${rel(from)}: cannot resolve ${spec}`);
  const target = resolved.resolvedFileName;
  // tsc reads a declaration file; the bundler loads its .js sibling. Both belong to the closure.
  const js = target.endsWith(".d.ts") ? target.slice(0, -5) + ".js" : null;
  return js && existsSync(js) ? [target, js] : [target];
}

function importsOf(file: string, options: ts.CompilerOptions): string[] {
  const ext = extname(file);
  const specs = ext === ".css"
    ? cssImports(file)
    : TS_EXT.has(ext) || ext === ".js" || file.endsWith(".d.ts")
      ? ts.preProcessFile(readFileSync(file, "utf8"), true, true).importedFiles.map((f) => f.fileName)
      : [];
  return specs.flatMap((s) => resolveImport(s, file, options));
}

function walk(starts: string[], options: ts.CompilerOptions): Set<string> {
  const seen = new Set<string>();
  const stack = [...starts];
  while (stack.length) {
    const file = stack.pop()!;
    if (seen.has(file)) continue;
    if (!file.startsWith(ROOT + "/") || file.includes("/node_modules/"))
      throw new Error(`import leaves the Arena root: ${file}`);
    seen.add(file);
    for (const dep of importsOf(file, options)) {
      if (dep.includes("/node_modules/")) continue;
      stack.push(dep);
    }
  }
  return seen;
}

export function computeClosure(): string[] {
  const options = compilerOptions();
  const entries = ENTRIES.map((e) => join(ROOT, e));
  const files = walk(entries.flatMap(htmlScripts), options);
  for (const e of entries) files.add(e);
  return [...files].map(rel).sort();
}

function listTree(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((d) => {
    const path = join(dir, d.name);
    return d.isDirectory() ? listTree(path) : [rel(path)];
  });
}

const isTest = (p: string) => /^src\/.*\.test\.tsx?$/.test(p);

function releaseInclude(closure: string[]): string[] {
  return closure.filter((p) => TS_EXT.has(extname(p)));
}

function check(): string[] {
  const problems: string[] = [];
  const closure = computeClosure();
  const committed = readFileSync(CLOSURE_FILE, "utf8").split("\n").filter(Boolean);
  const added = closure.filter((p) => !committed.includes(p));
  const removed = committed.filter((p) => !closure.includes(p));
  if (added.length) problems.push(`release build reaches files outside release-closure.txt: ${added.join(", ")}`);
  if (removed.length) problems.push(`release-closure.txt lists files the build no longer reaches: ${removed.join(", ")}`);
  const generated = closure.filter((p) => p.startsWith("convex/_generated/"));
  if (generated.length) problems.push(`release build imports the Convex generated API: ${generated.join(", ")}`);
  const include: string[] = JSON.parse(readFileSync(TSCONFIG, "utf8")).include;
  if (JSON.stringify(include) !== JSON.stringify(releaseInclude(committed)))
    problems.push("tsconfig.release.json include differs from the TypeScript files in release-closure.txt");
  const allowed = new Set(committed);
  const src = listTree(join(ROOT, "src"));
  const options = compilerOptions();
  for (const t of src.filter(isTest))
    for (const dep of importsOf(join(ROOT, t), options).map(rel))
      if (!allowed.has(dep)) problems.push(`${t} imports ${dep}, which is outside the release closure`);
  const stray = src.filter((p) => !allowed.has(p) && !isTest(p));
  if (stray.length) problems.push(`src/ holds files outside the release closure (self-deploy-only code goes in app/): ${stray.join(", ")}`);
  return problems;
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  if (args.length > 1 || (args.length === 1 && args[0] !== "--write"))
    throw new Error("Usage: bun scripts/release_closure.ts [--write]");
  if (args[0] === "--write") {
    const closure = computeClosure();
    writeFileSync(CLOSURE_FILE, closure.join("\n") + "\n");
    const config = JSON.parse(readFileSync(TSCONFIG, "utf8"));
    config.include = releaseInclude(closure);
    writeFileSync(TSCONFIG, JSON.stringify(config, null, 2) + "\n");
    console.log(`Wrote ${closure.length} files to release-closure.txt`);
  } else {
    const problems = check();
    if (problems.length) {
      for (const p of problems) console.error(p);
      process.exit(1);
    }
    console.log(`Release closure OK: ${readFileSync(CLOSURE_FILE, "utf8").split("\n").filter(Boolean).length} files`);
  }
}
