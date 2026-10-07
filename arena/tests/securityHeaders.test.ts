import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import config from "../vercel.json";
import { STATIC_HEADERS, contentSecurityPolicy, convexOrigins } from "../scripts/security_headers";

const ROOT = join(import.meta.dir, "..");

describe("static host headers", () => {
  test("vercel.json applies exactly the shared headers to every route", () => {
    const allRoutes = config.headers.find((entry) => entry.source === "/(.*)");
    expect(allRoutes).toBeDefined();
    expect(allRoutes!.headers.map(({ key, value }) => [key, value])).toEqual(STATIC_HEADERS);
  });

  test("public/_headers (Cloudflare Pages) matches vercel.json", () => {
    const lines = readFileSync(join(ROOT, "public/_headers"), "utf8").trimEnd().split("\n");
    expect(lines[0]).toBe("/*");
    expect(lines.slice(1).map((l) => l.trim().split(/: (.*)/s).slice(0, 2))).toEqual(STATIC_HEADERS);
  });

  test("sets browser hardening headers", () => {
    const headers = new Map(STATIC_HEADERS);
    expect(headers.get("Strict-Transport-Security")).toContain("max-age=63072000");
    expect(headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(headers.get("X-Frame-Options")).toBe("DENY");
    expect(headers.get("Referrer-Policy")).toBe("strict-origin-when-cross-origin");
    expect(headers.get("Permissions-Policy")).toContain("camera=()");
    expect(headers.get("Cross-Origin-Opener-Policy")).toBe("same-origin");
  });
});

describe("content security policy", () => {
  const csp = contentSecurityPolicy("https://example.convex.cloud");

  test("is deny-by-default", () => {
    expect(csp).toContain("default-src 'self'");
    expect(csp).toContain("object-src 'none'");
    expect(csp).toContain("script-src 'self'");
    expect(csp).toContain("upgrade-insecure-requests");
    expect(csp).not.toContain("'unsafe-eval'");
    expect(csp).not.toMatch(/script-src[^;]*'unsafe-inline'/);
  });

  test("names the configured Convex deployment exactly, never a wildcard", () => {
    expect(csp).toContain(
      "connect-src 'self' https://example.convex.cloud wss://example.convex.cloud https://example.convex.site ",
    );
    expect(csp).not.toContain("*.convex");
    expect(csp).toContain("https://fonts.googleapis.com");
    expect(csp).toContain("https://fonts.gstatic.com");
    expect(csp).toContain("https://huggingface.co");
    expect(csp).not.toContain("amazonaws.com");
  });

  test("supports a self-hosted or local backend", () => {
    expect(convexOrigins("http://127.0.0.1:3210", "http://127.0.0.1:3211")).toEqual([
      "http://127.0.0.1:3210",
      "ws://127.0.0.1:3210",
      "http://127.0.0.1:3211",
    ]);
    expect(contentSecurityPolicy("http://127.0.0.1:3210")).not.toContain("upgrade-insecure-requests");
    expect(() => convexOrigins("ftp://example.org")).toThrow();
  });
});
