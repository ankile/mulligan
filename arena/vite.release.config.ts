import { defineConfig } from 'vite';
import type { Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { dirname, relative, resolve } from 'node:path';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));

/** Fail the build if it loads a source file that is not in the committed release closure. */
function releaseClosureGuard(): Plugin {
  const allowed = new Set(readFileSync(resolve(root, 'release-closure.txt'), 'utf8').split('\n').filter(Boolean));
  return {
    name: 'release-closure-guard',
    apply: 'build',
    generateBundle() {
      const outside = [...this.getModuleIds()]
        .filter((id) => !id.startsWith('\0'))
        .map((id) => id.split('?')[0])
        .filter((file) => file.startsWith(root + '/') && !file.includes('/node_modules/'))
        .map((file) => relative(root, file))
        .filter((file) => !allowed.has(file));
      if (outside.length) this.error(`Release build reached files outside release-closure.txt: ${outside.join(', ')}`);
    },
  };
}

export default defineConfig({
  root,
  plugins: [react(), tailwindcss(), releaseClosureGuard()],
  publicDir: false,
  build: {
    rollupOptions: {input: {index: resolve(root, 'index.html'), overview: resolve(root, 'overview.html')}},
    outDir: resolve(root, 'dist-release'),
    emptyOutDir: true,
  },
});
