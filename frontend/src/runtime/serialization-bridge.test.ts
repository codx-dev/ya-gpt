import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { decodeMulti } from '@msgpack/msgpack';
import { inspectModel } from '../model/inspect.ts';
import { authorError, serializeModel } from './serialization-bridge.ts';
import type { SerializationExports } from './serialization-bridge';
import { metadataWasm as actualWasm } from '../../tests/wasm.ts';

test('real WASM preserves all model data while updating or restoring its author', async () => {
  const wasm = await actualWasm();
  const original = await readFile(new URL('../../tests/fixtures/training-dropout.bin', import.meta.url));
  const expected = [...decodeMulti(original)];
  let latest: Uint8Array = original;
  for (const author of ['Ada Lovelace', 'Zoë 🦀 <zoe@example.test>', '  ', '']) {
    const bytes = serializeModel(wasm, latest, author);
    const fields = [...decodeMulti(bytes)];
    assert.equal(fields[1], author || expected[1]);
    // Every header field, tensor, gradient, tokenizer and loss must survive unchanged.
    fields[1] = expected[1]; assert.deepEqual(fields, expected);
    const metadata = await inspectModel(new Blob([bytes]), wasm);
    assert.equal(metadata.authors, author || expected[1]);
    assert.equal(metadata.version, '1');
    assert.equal(metadata.train_tokens, expected[13]);
    assert.equal(wasm.model_serialize_result_len(), 0);
    assert.equal(wasm.model_serialize_result_ptr(), 0);
    latest = bytes;
  }
  assert.deepEqual(latest, new Uint8Array(original));
  assert.throws(() => serializeModel(wasm, new Uint8Array([0]), 'Ada'), /invalid model file/);
  assert.equal(wasm.model_serialize_result_len(), 0);
  assert.deepEqual(serializeModel(wasm, original, ''), new Uint8Array(original));
});

test('author limit counts UTF-8 bytes and accepts exactly the inspector limit', async () => {
  assert.equal(authorError(''), undefined);
  assert.equal(authorError('é'.repeat(32_768)), undefined);
  assert.match(authorError('é'.repeat(32_769))!, /64 KiB/);
  const wasm = await actualWasm();
  const original = await readFile(new URL('../../tests/fixtures/micro.bin', import.meta.url));
  const author = '🦀'.repeat(16_384);
  const bytes = serializeModel(wasm, original, author);
  assert.equal((await inspectModel(new Blob([bytes]), wasm)).authors, author);
  assert.throws(() => serializeModel(wasm, original, author + 'a'), /64 KiB/);
});

test('serialization frees both inputs and its result across memory growth and failures', () => {
  for (const fail of [false, true]) {
    const memory = new WebAssembly.Memory({ initial: 1 });
    const allocations = new Set<number>();
    let next = 1024, ended = 0;
    const wasm: SerializationExports = {
      memory, version: () => 1,
      alloc(n) { memory.grow(1); const ptr = next; next += Math.max(4, Math.ceil(n / 4) * 4); allocations.add(ptr); return ptr; },
      dealloc(ptr) { return allocations.delete(ptr) ? 0 : -1; },
      model_serialize(ptr, len, author, authorLen) {
        assert.deepEqual([...new Uint8Array(memory.buffer, ptr, len)], [1, 2, 3]);
        assert.equal(new TextDecoder().decode(new Uint8Array(memory.buffer, author, authorLen)), 'é🦀');
        memory.grow(1); new Uint8Array(memory.buffer, 16, 3).set([4, 5, 6]);
        return fail ? -1 : 0;
      },
      model_serialize_result_ptr: () => 16, model_serialize_result_len: () => 3,
      model_serialize_end() { ended++; new Uint8Array(memory.buffer, 16, 3).fill(0); return 0; },
      error_ptr: () => 0, error_len: () => 0,
    };
    if (fail) assert.throws(() => serializeModel(wasm, new Uint8Array([1, 2, 3]), 'é🦀'), /serialization failed/);
    else assert.deepEqual([...serializeModel(wasm, new Uint8Array([1, 2, 3]), 'é🦀')], [4, 5, 6]);
    assert.equal(ended, 1); assert.equal(allocations.size, 0);
  }
});
