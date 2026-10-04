import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { decodeMulti } from '@msgpack/msgpack';

test('the real serialization worker updates authors without WebGPU and recovers after errors', async ({ page }) => {
  await page.addInitScript(() => { Object.defineProperty(navigator, 'gpu', { value: undefined }); });
  await page.goto('/tests/gpu.html');
  const output = await page.evaluate(async () => {
    const path = '/src/runtime/index.ts';
    const { runtime } = await import(path);
    const original = new File([await (await fetch('/tests/fixtures/training-dropout.bin')).arrayBuffer()], 'trained.bin');
    const first = runtime.serializeModel(original, '  Zoë 🦀  ');
    let concurrent = '';
    try { await runtime.serializeModel(original, 'second'); } catch (error) { concurrent = String(error); }
    const custom = await first;
    const metadata = await runtime.inspectModel(new File([custom], original.name));
    let invalid = '';
    try { await runtime.serializeModel(new File(['bad'], 'bad.bin'), 'Ada'); } catch (error) { invalid = String(error); }
    const restored = await runtime.serializeModel(new File([custom], original.name), '');
    return { custom: Array.from(custom) as number[], restored: Array.from(restored) as number[], metadata, concurrent, invalid, busy: runtime.busy };
  });
  const expected = await readFile(new URL('./fixtures/training-dropout.bin', import.meta.url));
  expect(output.metadata.header.authors).toBe('  Zoë 🦀  ');
  const custom = [...decodeMulti(new Uint8Array(output.custom))];
  const original = [...decodeMulti(expected)];
  custom[1] = original[1]; expect(custom).toEqual(original);
  expect(new Uint8Array(output.restored)).toEqual(new Uint8Array(expected));
  expect(output.concurrent).toContain('already running');
  expect(output.invalid).not.toBe('');
  expect(output.busy).toBe(false);
});
