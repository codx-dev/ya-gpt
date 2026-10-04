import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { decodeMulti } from '@msgpack/msgpack';
import fixtures from './fixtures/training.json' with { type: 'json' };

function close(actual: unknown, expected: unknown) {
  if (typeof expected === 'number') {
    expect(typeof actual).toBe('number');
    expect(Number.isFinite(actual)).toBe(true);
    expect(Math.abs((actual as number) - expected)).toBeLessThanOrEqual(3e-3 * (1 + Math.abs(expected)));
  } else if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true);
    expect(actual).toHaveLength(expected.length);
    expected.forEach((v, i) => close((actual as unknown[])[i], v));
  } else expect(actual).toEqual(expected);
}

for (const fixture of fixtures) {
  test(`${fixture.name}: real worker/WASM training matches Rust and exports a reusable model`, async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto('/');
    const support = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
    if (process.env.YA_GPT_REQUIRE_GPU === '1') expect(support).toBe(true);
    test.skip(!support, 'No WebGPU adapter available.');
    const result = await page.evaluate(async (fixture) => {
      const path = '/src/runtime/index.ts';
      const { runtime } = await import(path);
      const updates: number[] = [];
      const job = runtime.train(fixture.text, fixture.options, (p: { completed: number }) => updates.push(p.completed));
      let concurrent = '';
      try { await runtime.generate(new File(['x'], 'x.bin'), { backend: 'webgpu', prompt: '', num_tokens: 1, seed: '0' }); } catch (error) { concurrent = String(error); }
      const trained = await job.result;
      const file = new File([trained.bytes], 'trained.bin');
      const metadata = await runtime.inspectModel(file);
      const generated: string = await runtime.generate(file, { backend: 'webgpu', prompt: 'abc', num_tokens: 5, seed: '42' });
      return { bytes: Array.from(trained.bytes) as number[], completed: trained.completed, stopped: trained.stopped, updates, concurrent, generated, metadata, busy: runtime.busy };
    }, fixture);
    const expected = await readFile(new URL(`./fixtures/${fixture.name}.bin`, import.meta.url));
    close([...decodeMulti(new Uint8Array(result.bytes))], [...decodeMulti(expected)]);
    expect(result.completed).toBe(fixture.options.iterations);
    expect(result.stopped).toBe(false); expect(result.busy).toBe(false);
    expect(result.updates).toContain(1);
    expect(result.concurrent).toContain('already running');
    expect(Array.from(result.generated)).toHaveLength(8);
    expect(result.metadata.header.train_loss).toBeGreaterThan(0);
    expect(fixture.final_loss).toBeLessThan(fixture.initial_loss);
  });
}

test('Train UI stops at a completed step, downloads and explicitly selects the result', async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto('/');
  const support = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
  if (process.env.YA_GPT_REQUIRE_GPU === '1') expect(support).toBe(true);
  test.skip(!support, 'No WebGPU adapter available.');
  await expect(page.getByText('WebGPU runtime ready')).toBeVisible();
  await page.getByRole('tab', { name: /Train/ }).click();
  await page.getByLabel('Training text', { exact: true }).fill(fixtures[0].text);
  for (const [label, value] of [['Context length', '5'], ['Embedding width', '6'], ['Attention heads', '2'], ['Transformer layers', '2'], ['Batch size', '2'], ['Iterations', '1000'], ['Random seed', '42']]) {
    await page.getByLabel(label, { exact: true }).fill(value);
  }
  await page.getByRole('button', { name: /Start training/ }).click();
  await expect(page.getByLabel('Context length', { exact: true })).toBeDisabled();
  await expect.poll(async () => Number(await page.getByLabel('Training progress', { exact: true }).getAttribute('value'))).toBeGreaterThan(0);
  await page.getByRole('button', { name: 'Stop training', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Download trained model', exact: true })).toBeEnabled({ timeout: 120_000 });
  await expect(page.locator('.selected-model')).toHaveCount(0);
  const author = page.getByLabel('Author', { exact: true });
  const defaultAuthor = await author.inputValue();
  expect(defaultAuthor.length).toBeGreaterThan(0);
  await author.fill('Zoë 🦀 <zoe@example.test>');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download trained model', exact: true }).click();
  const saved = await download;
  expect(saved.suggestedFilename()).toMatch(/^ya-gpt-\d+-steps\.bin$/);
  const data = await readFile((await saved.path())!);
  const decoded = [...decodeMulti(data)]; expect(decoded[0]).toBe('ya-gpt');
  expect(decoded[1]).toBe('Zoë 🦀 <zoe@example.test>');
  await page.getByRole('button', { name: 'Use model', exact: true }).click();
  await expect(page.getByRole('tab', { name: /Generate/ })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('.selected-model')).toContainText('Trained in browser');
  await page.getByRole('tab', { name: /Inspect/ }).click();
  await expect(page.locator('dl > div').filter({ has: page.getByText('Authors', { exact: true }) }).locator('dd')).toHaveText('Zoë 🦀 <zoe@example.test>');
  await page.getByRole('tab', { name: /Train/ }).click();
  await author.fill('');
  const fallbackDownload = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download trained model', exact: true }).click();
  const fallback = [...decodeMulti(await readFile((await (await fallbackDownload).path())!))];
  expect(fallback[1]).toBe(defaultAuthor);
  fallback[1] = decoded[1]; expect(fallback).toEqual(decoded);
  await page.getByRole('tab', { name: /Generate/ }).click();
  await page.getByLabel('Prompt', { exact: true }).fill('abc');
  await page.getByLabel('New tokens', { exact: true }).fill('3');
  await page.getByRole('button', { name: /Generate text/ }).click();
  await expect(page.getByLabel('Generated text', { exact: true })).toBeVisible();
});
