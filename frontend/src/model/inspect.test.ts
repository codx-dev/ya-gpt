import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { encode } from '@msgpack/msgpack';
import { inspectModel as inspect, METADATA_PREFIX_LIMIT } from './inspect.ts';
import { metadataWasm } from '../../tests/wasm.ts';

const example = new URL('../../../assets/model-shakespeare-small-30000-bs32-nembd32-head8-layer4.bin', import.meta.url);
const prefix: unknown[] = ['ya-gpt', 'Author', '0', 'f32', 55810, 1.736, 1.9066, 32, 32, 8, 4, 0.2, 66];
// Rust stores losses and dropout as f32, including when their values are integers.
const binary = (values: unknown[]) => new Blob(values.map((value, index) => new Uint8Array(encode(value, {
  forceFloat32: true, forceIntegerToFloat: [5, 6, 11].includes(index),
}))));
const wasm = await metadataWasm();
const inspectModel = (file: Blob) => inspect(file, wasm);

test('reads real Rust metadata without decoding model tensors', async () => {
  const bytes = await readFile(example);
  const metadata = await inspectModel(new Blob([bytes]));
  assert.equal(metadata.params, 55810);
  assert.equal(metadata.name, 'ya-gpt');
  assert.equal(metadata.version, '0');
  assert.equal(metadata.train_tokens, 0);
  assert.ok(Math.abs(metadata.train_loss - 1.7360060214996338) < 1e-7);
  assert.ok(Math.abs(metadata.validation_loss - 1.9066011905670166) < 1e-7);
  assert.deepEqual({ ...metadata.config, dropout: Number(metadata.config.dropout.toFixed(1)) }, {
    block_size: 32, n_embd: 32, n_head: 8, n_layer: 4, dropout: 0.2,
  });
  assert.equal(metadata.vocab_size, 66);
  // The prefix ends at byte 71. Neither a tensor payload nor valid trailing data is needed.
  assert.deepEqual(await inspectModel(new Blob([bytes.subarray(0, 71), new Uint8Array([0xc1])])), metadata);
});

test('rejects empty files and every truncated position in the real prefix', async () => {
  const bytes = await readFile(example);
  for (let length = 0; length < 71; length++) {
    await assert.rejects(inspectModel(new Blob([bytes.subarray(0, length)])), /empty|truncated/);
  }
});

test('rejects incompatible formats, wrong types, and invalid dimensions', async () => {
  for (const [index, value, pattern] of [
    [0, 'other', /Incompatible/], [2, '2', /unknown version/], [3, 'f16', /Incompatible/],
    [1, 7, /Invalid/], [4, 0, /dimensions/], [5, NaN, /finite/], [6, Infinity, /finite/],
    [7, 0, /dimensions/], [8, 31, /dimensions/], [9, 0, /dimensions/],
    [10, 1.5, /Invalid/], [11, 1, /dimensions/], [12, 0, /dimensions/],
    [0, [], /Invalid/], [1, 'é'.repeat(32_769), /64 KiB/],
  ] as [number, unknown, RegExp][]) {
    const values = [...prefix]; values[index] = value;
    await assert.rejects(inspectModel(binary(values)), pattern);
  }
});

test('reads a bounded prefix even for a large model', async () => {
  let sliced = false;
  const file = {
    size: 100_000_000,
    slice(start: number, end: number) {
      assert.equal(start, 0); assert.equal(end, METADATA_PREFIX_LIMIT);
      sliced = true;
      return binary(prefix);
    },
    arrayBuffer() { throw new Error('Must not read the entire model.'); },
  } as unknown as Blob;
  await inspectModel(file);
  assert.equal(sliced, true);
});

test('version 1 exposes the training split size without requiring tensors', async () => {
  const values = [...prefix, 1_234_567]; values[2] = '1';
  const bytes = new Uint8Array(await binary(values).arrayBuffer());
  const metadata = await inspectModel(new Blob([bytes, new Uint8Array([0xc1])]));
  assert.equal(metadata.version, '1');
  assert.equal(metadata.train_tokens, 1_234_567);
  assert.equal('header' in metadata, false);
  assert.equal('major' in metadata, false);
  for (let length = 1; length < bytes.length; length++) {
    await assert.rejects(inspectModel(new Blob([bytes.subarray(0, length)])), /Invalid|truncated/);
  }
  for (const count of [-1, 1.5, NaN, '10']) {
    await assert.rejects(inspectModel(binary([...values.slice(0, 13), count])), /Invalid/);
  }
  assert.equal((await inspectModel(binary([...values.slice(0, 13), 0]))).train_tokens, 0);
  assert.equal(wasm.model_inspect_result_ptr(), 0);
  assert.equal(wasm.model_inspect_result_len(), 0);
});

test('Rust-trained version 1 fixture records its actual training data split', async () => {
  const bytes = await readFile(new URL('../../tests/fixtures/training-dropout.bin', import.meta.url));
  const fixtures = JSON.parse(await readFile(new URL('../../tests/fixtures/training.json', import.meta.url), 'utf8'));
  const metadata = await inspectModel(new Blob([bytes]));
  assert.equal(metadata.version, '1');
  assert.equal(metadata.train_tokens, Math.floor(Array.from(fixtures[0].text).length * 0.9));
  assert.equal(metadata.vocab_size, 6);
});
