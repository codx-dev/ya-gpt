import { expect, test } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import fixture from './fixtures/gpu.json' with { type: 'json' };
import type { WasmExports } from '../src/runtime/bridge';

test('complete WASM inference logits match Naive', async ({ page }) => {
  await page.goto('/tests/gpu.html');
  const support = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
  if (process.env.YA_GPT_REQUIRE_GPU === '1') expect(support).toBe(true);
  test.skip(!support, 'No WebGPU adapter available.');
  const result = await page.evaluate(async () => {
    const hostPath = '/src/runtime/webgpu.ts', wasmPath = '/src/runtime/generated/ya_gpt_wasm.wasm?url';
    const { createGpuHost } = await import(hostPath);
    const { default: url } = await import(wasmPath);
    const host = await createGpuHost();
    const { instance } = await WebAssembly.instantiate(await (await fetch(url)).arrayBuffer(), { ya_gpu: host.imports });
    const wasm = instance.exports as WasmExports;
    host.memory = wasm.memory;
    const pointers: number[] = [];
    const write = (bytes: Uint8Array) => {
      const p = wasm.alloc(bytes.length); pointers.push(p);
      new Uint8Array(wasm.memory.buffer, p, bytes.length).set(bytes); return p;
    };
    host.begin();
    try {
      const bytes = new Uint8Array(await (await fetch('/tests/fixtures/micro.bin')).arrayBuffer());
      if (wasm.generation_begin(write(bytes), bytes.length, write(new TextEncoder().encode('abc')), 3, 1, 42, 0)) throw new Error('begin failed');
      const descriptor = write(new Uint8Array(12));
      if (wasm.generation_forward(descriptor)) throw new Error('forward failed');
      const [handle, offset, length] = new Uint32Array(wasm.memory.buffer, descriptor, 3);
      const logits: number[] = Array.from(await host.readback(handle, offset, length));
      wasm.generation_end();
      for (const p of pointers) wasm.dealloc(p);
      await host.finish();
      return { logits, buffers: host.bufferCount };
    } finally { host.dispose(); }
  });
  expect(result.logits).toHaveLength(6);
  result.logits.forEach((value, i) => expect(Math.abs(value - fixture.model.logits.slice(-6)[i])).toBeLessThan(3e-4));
  expect(result.buffers).toBe(0);
});

test('the real WASM generates seeded text and releases model resources', async ({ page }) => {
  await page.goto('/');
  const support = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
  if (process.env.YA_GPT_REQUIRE_GPU === '1') expect(support).toBe(true);
  test.skip(!support, 'No WebGPU adapter available.');
  await expect(page.getByText('WebGPU runtime ready')).toBeVisible();
  await page.getByLabel('Model file', { exact: true }).setInputFiles(fileURLToPath(new URL('./fixtures/micro.bin', import.meta.url)));
  await page.getByRole('tab', { name: /Generate/ }).click();
  await page.getByLabel('Prompt', { exact: true }).fill('abc');
  await page.getByLabel('New tokens', { exact: true }).fill('8');
  await page.getByLabel('Random seed', { exact: true }).fill('42');
  for (let run = 0; run < 2; run++) {
    await page.getByRole('button', { name: /Generate text/ }).click();
    await expect(page.getByLabel('Generated text', { exact: true })).toHaveText(fixture.model.generated);
    await expect(page.getByRole('button', { name: /Generate text/ })).toBeEnabled();
  }
  const path = '/src/runtime/index.ts';
  const edgeCases = await page.evaluate(async (path) => {
    const { runtime } = await import(path);
    const file = new File([await (await fetch('/tests/fixtures/micro.bin')).arrayBuffer()], 'micro.bin');
    const options = { backend: 'webgpu', prompt: 'abcé🦀abcé🦀', num_tokens: 3, seed: '18446744073709551615' };
    const first: string = await runtime.generate(file, options);
    const second: string = await runtime.generate(file, options);
    const empty: string = await runtime.generate(file, { ...options, prompt: '' });
    let invalid = '';
    try { await runtime.generate(new File(['invalid'], 'bad.bin'), options); } catch (e) { invalid = String(e); }
    const recovered: string = await runtime.generate(file, options);
    return { first, second, empty, invalid, recovered };
  }, path);
  expect(edgeCases.first).toBe(edgeCases.second);
  expect(edgeCases.recovered).toBe(edgeCases.first);
  expect(Array.from(edgeCases.first)).toHaveLength(13);
  expect(Array.from(edgeCases.empty)).toHaveLength(4);
  expect(edgeCases.invalid).not.toBe('');
});
