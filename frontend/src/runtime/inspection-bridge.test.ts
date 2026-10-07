import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkInspector, inspectMetadata } from './inspection-bridge.ts';
import type { InspectionExports } from './inspection-bridge';

test('inspection copies JSON and releases input/results across memory growth and failures', () => {
  for (const failure of ['none', 'allocation', 'parse', 'json', 'cleanup'] as const) {
    const memory = new WebAssembly.Memory({ initial: 1 });
    const allocations = new Set<number>();
    const bytes = new TextEncoder().encode(failure === 'json' ? '{' : '{"version":"1","train_tokens":1234}');
    let ended = 0;
    const wasm: InspectionExports = {
      memory, version: () => 1,
      alloc() { memory.grow(1); if (failure === 'allocation') return 0; allocations.add(1024); return 1024; },
      dealloc(ptr) { return allocations.delete(ptr) ? 0 : -1; },
      model_inspect(ptr, len) {
        assert.deepEqual([...new Uint8Array(memory.buffer, ptr, len)], [1, 2, 3]);
        memory.grow(1);
        new Uint8Array(memory.buffer, 16, bytes.length).set(bytes);
        return failure === 'parse' ? -1 : 0;
      },
      model_inspect_result_ptr: () => 16, model_inspect_result_len: () => bytes.length,
      model_inspect_end() {
        ended++;
        new Uint8Array(memory.buffer, 16, bytes.length).fill(0);
        return failure === 'cleanup' ? -1 : 0;
      },
      error_ptr: () => 0, error_len: () => 0,
    };
    if (failure === 'none') {
      assert.deepEqual(inspectMetadata(wasm, new Uint8Array([1, 2, 3])), { version: '1', train_tokens: 1234 });
    } else {
      assert.throws(() => inspectMetadata(wasm, new Uint8Array([1, 2, 3])), failure === 'json' ? SyntaxError : Error);
    }
    assert.equal(ended, 1);
    assert.equal(allocations.size, 0);
  }
});

test('rejects older WASM builds without inspection exports', () => {
  assert.throws(() => checkInspector({ version: () => 1 } as InspectionExports), /Rebuild WASM/);
  assert.throws(() => checkInspector({ version: () => 2 } as InspectionExports), /Unsupported/);
});
