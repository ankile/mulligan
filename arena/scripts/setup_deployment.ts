/**
 * Set the server-side secrets a fresh Policy Arena deployment needs, without printing them.
 *
 *   bun scripts/setup_deployment.ts --site-url https://arena.example.org [--prod] [--force]
 *
 * Runs `npx convex env set` against the deployment the Convex CLI selects (CONVEX_DEPLOYMENT in
 * .env.local, or the production deployment of that project with --prod) and sets:
 *   SITE_URL             where Convex Auth sends users back after signing in
 *   JWT_PRIVATE_KEY, JWKS  the Convex Auth signing key pair (RS256), freshly generated
 *   ARENA_SERVICE_TOKEN  random bridge secret between the machine HTTP API and the mutations
 * Variables that already have a value are left alone unless --force is given (rotating the JWT
 * key signs every user out; rotating the service token needs no other change).
 *
 * The Hugging Face OAuth app (AUTH_HUGGINGFACE_ID / AUTH_HUGGINGFACE_SECRET), the editor
 * allowlist (ARENA_EDITOR_SUBS) and the machine keys (POLICY_ARENA_MACHINE_KEYS_JSON) are set by
 * hand; see README.md.
 */
import { spawnSync } from "node:child_process";
import { generateKeyPairSync, randomBytes } from "node:crypto";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");
const args = process.argv.slice(2);
const flag = (name: string) => args.includes(name);
const siteIndex = args.indexOf("--site-url");
const siteUrl = siteIndex >= 0 ? args[siteIndex + 1] : undefined;
const known = new Set(["--prod", "--force", "--site-url", siteUrl]);
if (!siteUrl || !/^https?:\/\/[^/]+$/.test(siteUrl) || args.some((a) => !known.has(a)))
  throw new Error("Usage: bun scripts/setup_deployment.ts --site-url https://host[:port] [--prod] [--force]");

const target = flag("--prod") ? ["--prod"] : [];

/** `npx convex env <args>`; a value goes in on stdin, so it never appears in a process listing. */
function convex(args: string[], input?: string): string {
  const result = spawnSync("npx", ["convex", "env", ...args, ...target], { cwd: ROOT, encoding: "utf8", input });
  if (result.status !== 0) throw new Error(`npx convex env ${args.join(" ")} failed:\n${result.stderr}`);
  return result.stdout;
}

const existing = new Set(
  convex(["list"]).split("\n").map((line) => line.split("=")[0]).filter(Boolean),
);

function values(): [string, string][] {
  const keys = generateKeyPairSync("rsa", { modulusLength: 2048 });
  const privateKey = keys.privateKey.export({ type: "pkcs8", format: "pem" }).toString().trimEnd().replace(/\n/g, " ");
  const jwks = JSON.stringify({ keys: [{ use: "sig", ...keys.publicKey.export({ format: "jwk" }) }] });
  return [
    ["SITE_URL", siteUrl!],
    ["JWT_PRIVATE_KEY", privateKey],
    ["JWKS", jwks],
    ["ARENA_SERVICE_TOKEN", randomBytes(32).toString("hex")],
  ];
}

const pending = values();
const jwtPair = new Set(["JWT_PRIVATE_KEY", "JWKS"]);
for (const [name, value] of pending) {
  // The key pair is set together or not at all.
  const present = jwtPair.has(name) ? [...jwtPair].some((n) => existing.has(n)) : existing.has(name);
  if (present && !flag("--force")) {
    console.log(`${name}: already set, kept`);
    continue;
  }
  convex(["set", name], value);
  console.log(`${name}: set`);
}
