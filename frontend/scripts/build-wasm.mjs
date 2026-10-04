import { spawnSync } from 'node:child_process';
import { copyFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const build = spawnSync('cargo', ['build', '-p', 'ya-gpt-wasm', '--target', 'wasm32-unknown-unknown', '--release'], { cwd: root, stdio: 'inherit' });
if (build.error) throw build.error;
if (build.status !== 0) {
  console.error('WASM build failed. Ensure Cargo dependencies are available and install the target with: rustup target add wasm32-unknown-unknown');
  process.exit(build.status ?? 1);
}
const destination = new URL('../src/runtime/generated/', import.meta.url);
await mkdir(destination, { recursive: true });
const target = resolve(root, process.env.CARGO_TARGET_DIR || 'target');
await copyFile(resolve(target, 'wasm32-unknown-unknown/release/ya_gpt_wasm.wasm'), new URL('ya_gpt_wasm.wasm', destination));
