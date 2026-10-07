/**
 * Security headers of the self-deploy build.
 *
 * The Content-Security-Policy names the configured Convex deployment exactly (no wildcard), so it
 * is computed at build time from VITE_CONVEX_URL and injected into index.html as a <meta> tag by
 * vite.config.ts. The other headers do not depend on the deployment; they live in `vercel.json`
 * (Vercel) and `public/_headers` (Cloudflare Pages), and tests/securityHeaders.test.ts checks
 * both files against STATIC_HEADERS.
 */

export const STATIC_HEADERS: [string, string][] = [
  ["Strict-Transport-Security", "max-age=63072000; includeSubDomains"],
  ["X-Content-Type-Options", "nosniff"],
  ["X-Frame-Options", "DENY"],
  ["Referrer-Policy", "strict-origin-when-cross-origin"],
  ["Permissions-Policy", "camera=(), microphone=(), geolocation=(), browsing-topics=()"],
  ["Cross-Origin-Opener-Policy", "same-origin"],
];

const HF = "https://huggingface.co https://*.huggingface.co https://*.hf.co";

/** Origins the browser must reach for a Convex deployment URL (client API and websocket). */
export function convexOrigins(convexUrl: string, siteUrl?: string): string[] {
  const url = new URL(convexUrl);
  if (url.protocol !== "https:" && url.protocol !== "http:")
    throw new Error(`VITE_CONVEX_URL must be an http(s) URL, got ${convexUrl}`);
  const ws = `${url.protocol === "https:" ? "wss:" : "ws:"}//${url.host}`;
  const origins = [url.origin, ws];
  const site = siteUrl
    ? new URL(siteUrl).origin
    : url.hostname.endsWith(".convex.cloud")
      ? url.origin.replace(/\.convex\.cloud$/, ".convex.site")
      : null;
  if (site) origins.push(site);
  return origins;
}

/** The meta-tag policy. frame-ancestors is not honored in <meta>; X-Frame-Options covers it. */
export function contentSecurityPolicy(convexUrl: string, siteUrl?: string): string {
  return [
    "default-src 'self'",
    "base-uri 'self'",
    "object-src 'none'",
    "form-action 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    `img-src 'self' data: blob: ${HF}`,
    `media-src 'self' blob: ${HF} https://*.xethub.hf.co`,
    `connect-src 'self' ${convexOrigins(convexUrl, siteUrl).join(" ")} ${HF} https://*.xethub.hf.co`,
    "worker-src 'self' blob:",
    "manifest-src 'self'",
    // A local backend (http://127.0.0.1) must not be upgraded to https.
    ...(new URL(convexUrl).protocol === "https:" ? ["upgrade-insecure-requests"] : []),
  ].join("; ");
}
