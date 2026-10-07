/**
 * Self-deploy build: the full Policy Arena (editing, labeling, sign-in) against your own Convex
 * deployment. `bun run dev` serves it; `bun run build` writes dist/ for Vercel or Cloudflare Pages.
 *
 * The shared screens in src/ are the same files the read-only Mulligan snapshot builds from
 * (vite.release.config.ts). Two modules are swapped here and only here:
 *   src/lib/arenaClient.ts       -> app/client.ts         (live Convex hooks instead of frozen data)
 *   src/release/ReleaseLinks.tsx -> app/AuthControls.tsx  (sign-in instead of release links)
 * The build fails if a release-only module still reaches the bundle.
 */
import { defineConfig, loadEnv } from 'vite';
import type { Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { contentSecurityPolicy } from './scripts/security_headers';

const root = dirname(fileURLToPath(import.meta.url));

const SWAPS = new Map([
  [resolve(root, 'src/lib/arenaClient.ts'), resolve(root, 'app/client.ts')],
  [resolve(root, 'src/release/ReleaseLinks.tsx'), resolve(root, 'app/AuthControls.tsx')],
]);
const RELEASE_ONLY = ['src/release/client.ts', 'src/release/adapter.ts', 'src/release/ReleaseLinks.tsx', 'src/release/main.tsx'];

function liveModules(): Plugin {
  return {
    name: 'policy-arena-live-modules',
    enforce: 'pre',
    async resolveId(source, importer, options) {
      if (!importer) return null;
      const resolved = await this.resolve(source, importer, { ...options, skipSelf: true });
      const swap = resolved && SWAPS.get(resolved.id.split('?')[0]);
      return swap ?? null;
    },
    generateBundle() {
      const leaked = [...this.getModuleIds()]
        .map((id) => relative(root, id.split('?')[0]))
        .filter((file) => RELEASE_ONLY.includes(file));
      if (leaked.length) this.error(`Release-only modules reached the self-deploy build: ${leaked.join(', ')}`);
    },
  };
}

function contentSecurityPolicyMeta(convexUrl: string, siteUrl: string | undefined): Plugin {
  return {
    name: 'policy-arena-csp',
    apply: 'build',
    transformIndexHtml: () => [
      { tag: 'meta', attrs: { 'http-equiv': 'Content-Security-Policy', content: contentSecurityPolicy(convexUrl, siteUrl) }, injectTo: 'head-prepend' },
    ],
  };
}

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, root, 'VITE_');
  const convexUrl = env.VITE_CONVEX_URL;
  if (command === 'build' && !convexUrl)
    throw new Error('VITE_CONVEX_URL is not set. Set it in .env.local (npx convex dev writes it) or in the host build settings; see .env.example.');
  return {
    root: resolve(root, 'app'),
    envDir: root,
    publicDir: resolve(root, 'public'),
    plugins: [
      liveModules(),
      react(),
      tailwindcss(),
      ...(convexUrl ? [contentSecurityPolicyMeta(convexUrl, env.VITE_CONVEX_SITE_URL || undefined)] : []),
    ],
    server: { fs: { allow: [root] } },
    build: { outDir: resolve(root, 'dist'), emptyOutDir: true },
  };
});
