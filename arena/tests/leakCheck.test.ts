import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { leakCheck } from "../scripts/leak_check";

const dirs: string[] = [];
function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "arena-leak-"));
  dirs.push(dir);
  return dir;
}
function site(js: string): string {
  const dir = tempDir();
  mkdirSync(join(dir, "assets"));
  writeFileSync(join(dir, "assets", "index.js"), js);
  writeFileSync(join(dir, "index.html"), "<!doctype html><title>Policy Arena</title>");
  return dir;
}
afterAll(() => dirs.forEach((d) => rmSync(d, { recursive: true, force: true })));

describe("leak check modes", () => {
  const userBuild = 'new ConvexReactClient("https://example.convex.cloud");signIn("huggingface")';

  test("the self-deploy build may hold the user's Convex client and sign-in", () => {
    expect(leakCheck(site(userBuild), "app")).toEqual([]);
  });

  test("the read-only snapshot may not", () => {
    expect(leakCheck(site(userBuild), "release").length).toBeGreaterThan(0);
  });

  test("both modes reject secrets without echoing them", () => {
    const token = `hf_${"a1B2".repeat(9)}`;
    const leaked = site(`const t = "${token}"; const k = "-----BEGIN PRIVATE KEY-----";`);
    const expected = ["assets/index.js: private key", "assets/index.js: Hugging Face token"];
    expect(leakCheck(leaked, "app")).toEqual(expected);
    const release = leakCheck(leaked, "release");
    expect(release).toEqual(expected);
    expect(release.join("\n")).not.toContain(token);
  });

  test("a build without JavaScript is an error", () => {
    expect(() => leakCheck(tempDir())).toThrow("No built JavaScript");
  });
});
