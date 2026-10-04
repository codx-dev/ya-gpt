import { defineConfig } from '@playwright/test';

// Shader tests are independent of the Rust artifact. Full WASM/UI coverage is
// in the normal browser suite, which builds the real Rust module first.
export default defineConfig({
  testDir: './tests', testMatch: 'gpu.spec.ts', timeout: 60_000,
  use: {
    baseURL: 'http://127.0.0.1:4175',
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
      args: process.env.YA_GPT_SOFTWARE_GPU === '1' ? ['--enable-unsafe-webgpu', '--use-angle=swiftshader'] : [],
    },
  },
  webServer: { command: 'node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 4175 --strictPort', url: 'http://127.0.0.1:4175/tests/gpu.html', reuseExistingServer: !process.env.CI },
});
