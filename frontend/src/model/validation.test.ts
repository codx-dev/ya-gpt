import assert from 'node:assert/strict';
import { test } from 'node:test';
import { configErrors, generateErrors, presets, seedError, trainErrors } from './validation.ts';

test('preserves the full Rust u64 seed range without rounding', () => {
  for (const seed of ['', '0', '9007199254740993', '18446744073709551615']) assert.equal(seedError(seed), undefined);
  for (const seed of ['-1', '1.2', '1e5', ' 1', '18446744073709551616', 'abc']) assert.ok(seedError(seed));
});

test('validates relationships and finite positive integer parameters', () => {
  assert.deepEqual(configErrors(presets.tiny), []);
  assert.match(configErrors({ ...presets.tiny, n_embd: 33 }).join(), /divisible/);
  for (const n of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    assert.ok(configErrors({ ...presets.tiny, n_layer: n }).length);
    assert.ok(generateErrors({ backend: 'simd', prompt: '', num_tokens: n, seed: '' }).length);
  }
  assert.ok(configErrors({ ...presets.tiny, dropout: 1 }).length);
});

test('training requires usable train and validation splits for Unicode text', () => {
  const options = { backend: 'simd' as const, model: presets.tiny, batch_size: 8,
    iterations: 1000, learning_rate: 0.0003, weight_decay: 0.01, seed: '', report_loss_per_step: false };
  assert.deepEqual(trainErrors(options, '🙂'.repeat(330)), []);
  assert.match(trainErrors(options, '🙂'.repeat(320)).join(), /validation splits/);
  assert.match(trainErrors({ ...options, learning_rate: Infinity }, 'a'.repeat(400)).join(), /Learning rate/);
  assert.match(trainErrors({ ...options, weight_decay: -1 }, 'a'.repeat(400)).join(), /Weight decay/);
});

test('CLI preset defaults stay consistent with the Rust source', async () => {
  const { readFile } = await import('node:fs/promises');
  const config = await readFile(new URL('../../../crates/core/src/config.rs', import.meta.url), 'utf8');
  const cli = await readFile(new URL('../../../crates/cli/src/cli.rs', import.meta.url), 'utf8');
  for (const [name, { batch_size, iterations, ...model }] of Object.entries(presets)) {
    const body = config.match(new RegExp(`pub fn ${name}\\(\\)[\\s\\S]*?Self \\{([\\s\\S]*?)\\}`))?.[1];
    assert.ok(body, `Rust preset ${name} exists`);
    for (const [field, value] of Object.entries(model)) {
      assert.equal(Number(body.match(new RegExp(`${field}: ([\\d.]+)`))?.[1]), value);
    }
    const variant = name[0].toUpperCase() + name.slice(1);
    const defaults = cli.match(new RegExp(`Self::${variant} => \\((\\d+), ([\\d_]+)\\)`));
    assert.equal(Number(defaults?.[1]), batch_size);
    assert.equal(Number(defaults?.[2].replaceAll('_', '')), iterations);
  }
});
