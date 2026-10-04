import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { encode } from '@msgpack/msgpack';
import { inspectModel } from './inspect.ts';

const example = new URL('../../../assets/model-shakespeare-small-30000-bs32-nembd32-head8-layer4.bin', import.meta.url);
const prefix: unknown[] = ['ya-gpt', 'Author', '0', 'f32', 55810, 1.736, 1.9066, 32, 32, 8, 4, 0.2, 66];
const binary = (values: unknown[]) => new Blob(values.map((value) => new Uint8Array(encode(value))));

test('reads real Rust metadata without decoding model tensors', async () => {
  const bytes = await readFile(example);
  const metadata = await inspectModel(new Blob([bytes]));
  assert.equal(metadata.header.params, 55810);
  assert.equal(metadata.header.name, 'ya-gpt');
  assert.equal(metadata.header.train_loss, 1.7360060214996338);
  assert.equal(metadata.header.validation_loss, 1.9066011905670166);
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
    [0, 'other', /Incompatible/], [2, '1', /Incompatible/], [3, 'f16', /Incompatible/],
    [1, 7, /types/], [4, 0, /dimensions/], [5, NaN, /types/],
    [7, 0, /dimensions/], [8, 31, /dimensions/], [9, 0, /dimensions/],
    [10, 1.5, /dimensions/], [11, 1, /dimensions/], [12, 0, /dimensions/],
    [0, [], /Invalid/],
  ] as [number, unknown, RegExp][]) {
    const values = [...prefix]; values[index] = value;
    await assert.rejects(inspectModel(binary(values)), pattern);
  }
});

test('does not read the rest of a large model stream', async () => {
  const bytes = new Uint8Array(await binary(prefix).arrayBuffer());
  let cancelled = false;
  const file = {
    size: 100_000_000,
    stream: () => new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(bytes); },
      cancel() { cancelled = true; },
    }),
  } as Blob;
  await inspectModel(file);
  assert.equal(cancelled, true);
});
