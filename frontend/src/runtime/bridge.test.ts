import assert from 'node:assert/strict';
import { test } from 'node:test';
import { WasmBridge } from './bridge.ts';
import type { WasmExports } from './bridge';
import type { GpuHost } from './gpu-host';
import type { GenerateOptions } from '../model/types';

const options: GenerateOptions = { backend: 'webgpu', prompt: 'é🦀', num_tokens: 2, seed: '18446744073709551615' };

function mock(fail = false) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const allocations = new Set<number>();
  const seeds: number[] = [];
  let next = 1024, tokens = 0, ended = 0, finished = 0;
  const wasm = {
    memory, version: () => 1,
    alloc(n: number) { const ptr = next; next += Math.max(4, Math.ceil(n / 4) * 4); allocations.add(ptr); return ptr; },
    dealloc(p: number) { return allocations.delete(p) ? 0 : -1; },
    generation_begin(_m: number, _n: number, prompt: number, len: number, _count: number, lo: number, hi: number) {
      assert.equal(new TextDecoder().decode(new Uint8Array(memory.buffer, prompt, len)), 'é🦀');
      seeds.push(lo, hi); return 0;
    },
    generation_forward(ptr: number) { memory.grow(1); new Uint32Array(memory.buffer, ptr, 3).set([1, 12, 3]); return 0; },
    generation_accept_logits(ptr: number, count: number) {
      assert.deepEqual([...new Float32Array(memory.buffer, ptr, count)], [1, 2, 3]);
      return ++tokens === 2 ? 1 : 0;
    },
    generation_result_ptr() { new Uint8Array(memory.buffer, 16, 2).set([111, 107]); return 16; },
    generation_result_len: () => 2, generation_end() { ended++; return 0; }, error_ptr: () => 0, error_len: () => 0,
  } as WasmExports;
  const gpu = {
    begin() {},
    async readback(_h: number, offset: number, count: number) {
      assert.equal(offset, 12); assert.equal(count, 3);
      memory.grow(1);
      if (fail) throw new Error('readback failed');
      return new Float32Array([1, 2, 3]);
    },
    async finish() { finished++; },
  } as unknown as GpuHost;
  return { bridge: new WasmBridge(wasm, gpu), allocations, seeds, ended: () => ended, finished: () => finished };
}

test('generation survives memory growth and preserves all 64 seed bits', async () => {
  const m = mock();
  assert.equal(await m.bridge.generate(new Uint8Array([1]), options), 'ok');
  assert.deepEqual(m.seeds, [0xffffffff, 0xffffffff]);
  assert.equal(m.allocations.size, 0);
  assert.equal(m.ended(), 1); assert.equal(m.finished(), 1);
});

test('failures free allocations and close GPU/session state; overlapping runs reject', async () => {
  const m = mock(true);
  const first = m.bridge.generate(new Uint8Array([1]), options);
  await assert.rejects(m.bridge.generate(new Uint8Array([1]), options), /already running/);
  await assert.rejects(first, /readback failed/);
  assert.equal(m.allocations.size, 0);
  assert.equal(m.ended(), 1); assert.equal(m.finished(), 1);
  await assert.rejects(m.bridge.generate(new Uint8Array([1]), options), /readback failed/);
});
