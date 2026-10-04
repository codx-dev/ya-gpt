import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  worker: { format: 'es' },
  base: './',
  // Keep WASM and model files as assets, including under a relative base URL.
  build: { assetsInlineLimit: 0 },
  server: { fs: { allow: [fileURLToPath(new URL('..', import.meta.url))] } },
});
