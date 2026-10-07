/**
 * Build the static release entry, then stage it with the frozen data files in dist-release/.
 *
 *   bun scripts/package_release.ts data/release.json [--preview]      (from arena/)
 *
 * The data always comes from arena/data/, hash-checked against data/SHA256SUMS. Staged files are
 * copied (never linked) and nothing outside the Arena root is read or written.
 */
import { copyFileSync, lstatSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { validateRelease } from "../src/release/types";
import { createReleaseAdapter } from "../src/release/adapter";
import { leakCheck } from "./leak_check";

const ROOT = resolve(import.meta.dir, "..");
const DATA = join(ROOT, "data");
const TARGET = join(ROOT, "dist-release");

const [source, ...flags] = process.argv.slice(2);
if (!source || flags.some((f) => f !== "--preview"))
  throw new Error("Usage: bun scripts/package_release.ts data/release.json [--preview]");
if (realpathSync(resolve(source)) !== join(DATA, "release.json"))
  throw new Error(`The release is staged from ${join(DATA, "release.json")}, not ${source}`);

const sums = readFileSync(join(DATA, "SHA256SUMS"), "utf8").split("\n").filter(Boolean).map((line) => {
  const [digest, name] = line.split("  ");
  return { digest, name };
});
for (const { digest, name } of sums) {
  const path = join(DATA, name);
  if (!lstatSync(path).isFile()) throw new Error(`${path} must be a regular file`);
  if (createHash("sha256").update(readFileSync(path)).digest("hex") !== digest)
    throw new Error(`${path} does not match data/SHA256SUMS`);
}

const read = (name: string) => JSON.parse(readFileSync(join(DATA, name), "utf8"));
const data = validateRelease(read("release.json"));
createReleaseAdapter(data, read("ui.json"), read("sim-statistics.json"));
if (!flags.includes("--preview") && data.state !== "public_verified")
  throw new Error("Production release requires public verification receipts");

const result = spawnSync("bun", ["run", "build:release"], { cwd: ROOT, stdio: "inherit" });
if (result.status !== 0) throw new Error("Release build failed");

mkdirSync(join(TARGET, "data"), { recursive: true });
for (const { name } of sums) copyFileSync(join(DATA, name), join(TARGET, "data", name));
copyFileSync(join(ROOT, "public", "favicon.svg"), join(TARGET, "favicon.svg"));
const headers = `/*
  Content-Security-Policy: default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; form-action 'none'; script-src 'self' https://static.cloudflareinsights.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https://huggingface.co https://*.hf.co; media-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co https://*.xethub.hf.co; connect-src 'self' https://cloudflareinsights.com https://mulligan.page https://huggingface.co https://*.huggingface.co https://*.hf.co https://*.xethub.hf.co; upgrade-insecure-requests
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: camera=(), microphone=(), geolocation=(), browsing-topics=()
  Strict-Transport-Security: max-age=63072000; includeSubDomains
/data/*
  Cache-Control: public, max-age=0, must-revalidate
`;
writeFileSync(join(TARGET, "_headers"), headers);
writeFileSync(join(TARGET, "robots.txt"), flags.includes("--preview") ? "User-agent: *\nDisallow: /\n" : "User-agent: *\nAllow: /\n");

const leaks = leakCheck(TARGET);
if (leaks.length) throw new Error(`Internal backend leaked into the release bundle:\n${leaks.join("\n")}`);
console.log(`Validated ${data.tasks.length} tasks; release staged at ${TARGET}`);
