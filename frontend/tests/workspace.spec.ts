import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

const model = fileURLToPath(new URL('../../assets/model-shakespeare-small-30000-bs32-nembd32-head8-layer4.bin', import.meta.url));

async function diskFiles(page: Page) {
  return page.evaluate(async () => {
    const root = await (await navigator.storage.getDirectory()).getDirectoryHandle('ya-gpt-temporary-models');
    const files: string[] = [];
    for await (const [session, dir] of root.entries()) {
      if (dir.kind === 'directory') for await (const [name] of (dir as FileSystemDirectoryHandle).entries()) files.push(`${session}/${name}`);
    }
    return files.sort();
  });
}

test('local model inspection and editable forms preserve their values', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.runtime-status')).not.toContainText('Loading');
  await page.getByLabel('Prompt', { exact: true }).fill('JULIET:');
  await page.getByRole('tab', { name: /Train/ }).click();
  await page.getByLabel('Preset', { exact: true }).selectOption('small');
  await expect(page.getByLabel('Batch size', { exact: true })).toHaveValue('16');
  await page.getByLabel('Embedding width', { exact: true }).fill('65');
  await expect(page.getByText('Embedding width must be divisible by attention heads.')).toBeVisible();
  await page.getByRole('tab', { name: /Generate/ }).click();
  await expect(page.getByLabel('Prompt', { exact: true })).toHaveValue('JULIET:');
  await page.getByRole('button', { name: /Generate text/ }).click();
  await expect(page.locator('#panel-Generate').getByRole('alert')).toHaveText('No model selected. Click “Load Shakespeare example” in the sidebar to get started.');
  await page.getByLabel('Model file', { exact: true }).setInputFiles(model);
  await expect(page.getByRole('tab', { name: /Inspect/ })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('#panel-Generate [role="alert"]')).toHaveCount(0);
  await expect(page.getByText('55,810', { exact: true })).toBeVisible();
  await expect(page.getByText('66 characters', { exact: true })).toBeVisible();
  await expect(page.getByText('1.7360', { exact: true })).toBeVisible();
  await expect(page.getByText('1.9066', { exact: true })).toBeVisible();
  await expect.poll(() => diskFiles(page)).toEqual([]);
});

test('bundled example uses temporary disk storage and unload removes it', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Load Shakespeare example/ }).click();
  await expect(page.getByText('55,810', { exact: true })).toBeVisible();
  await expect.poll(async () => (await diskFiles(page)).length).toBe(1);
  await page.getByRole('button', { name: 'Unload model' }).click();
  await expect(page.getByRole('heading', { name: 'No model selected' })).toBeVisible();
  await expect.poll(() => diskFiles(page)).toEqual([]);
});

test('URL download works and invalid replacements preserve the selected model', async ({ page }) => {
  await page.route('https://models.test/good.bin', async (route) => route.fulfill({ body: await readFile(model), headers: { 'access-control-allow-origin': '*' } }));
  await page.route('https://models.test/bad.bin', (route) => route.fulfill({ body: 'not a model', headers: { 'access-control-allow-origin': '*' } }));
  await page.goto('/');
  await page.getByRole('button', { name: 'From URL' }).click();
  await page.getByLabel('Model URL', { exact: true }).fill('https://models.test/good.bin');
  await page.getByRole('button', { name: 'Download model', exact: true }).click();
  await expect(page.getByText('55,810', { exact: true })).toBeVisible();
  const files = await diskFiles(page);
  await page.getByLabel('Model URL', { exact: true }).fill('https://models.test/bad.bin');
  await page.getByRole('button', { name: 'Download model', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Invalid');
  await expect(page.getByText('55,810', { exact: true })).toBeVisible();
  await expect.poll(() => diskFiles(page)).toEqual(files);
});

test('cleanup protects another tab and reclaims a closed tab on startup', async ({ page, context }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Load Shakespeare example/ }).click();
  await expect(page.getByText('55,810', { exact: true })).toBeVisible();
  const firstFiles = await diskFiles(page);
  const second = await context.newPage();
  await second.goto('/');
  await second.getByRole('button', { name: /Load Shakespeare example/ }).click();
  await expect(second.getByText('55,810', { exact: true })).toBeVisible();
  await expect.poll(async () => (await diskFiles(second)).length).toBe(2);
  await page.close();
  const third = await context.newPage();
  await third.goto('/');
  await expect.poll(async () => (await diskFiles(second)).length).toBe(1);
  expect(await diskFiles(second)).not.toEqual(firstFiles);
  await expect(second.getByText('55,810', { exact: true })).toBeVisible();
});

test('local inspection remains usable without browser disk storage or compute support', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'storage', { value: undefined });
    Object.defineProperty(navigator, 'gpu', { value: undefined });
    WebAssembly.validate = () => false;
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: /Load Shakespeare example/ })).toBeDisabled();
  await page.getByLabel('Model file', { exact: true }).setInputFiles(model);
  await expect(page.getByText('55,810', { exact: true })).toBeVisible();
});

test('mobile layout has no horizontal overflow and tabs work with a keyboard', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('/');
  const generate = page.getByRole('tab', { name: /Generate/ });
  await generate.focus(); await generate.press('ArrowRight');
  await expect(page.getByRole('tab', { name: /Train/ })).toBeFocused();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test('production WASM and example assets load under a subdirectory', async ({ page }) => {
  const failures: string[] = [];
  const wasmResponse = page.waitForResponse((response) => response.url().includes('/workspace/') && response.url().endsWith('.wasm'));
  page.on('pageerror', (error) => failures.push(error.message));
  await page.goto('http://127.0.0.1:4174/workspace/');
  const wasm = await wasmResponse;
  expect(wasm.ok()).toBe(true);
  expect(WebAssembly.validate(new Uint8Array(await wasm.body()))).toBe(true);
  await expect(page.locator('.runtime-status')).not.toContainText('Loading');
  await page.getByRole('button', { name: /Load Shakespeare example/ }).click();
  await expect(page.getByText('55,810', { exact: true })).toBeVisible();
  expect(failures).toEqual([]);
});
