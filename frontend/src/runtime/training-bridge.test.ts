import assert from 'node:assert/strict';
import { test } from 'node:test';
import { runTraining, trainingErrors } from './training-bridge.ts';
import type { TrainingExports } from './training-bridge';
import type { GpuHost } from './gpu-host';
import type { TrainOptions, TrainingProgress } from '../model/types';

const text = 'abcé🦀'.repeat(40);
const options: TrainOptions = { backend: 'webgpu', model: { block_size: 3, n_embd: 6, n_head: 2, n_layer: 1, dropout: 0.2 }, batch_size: 2, iterations: 4, learning_rate: 0.001, weight_decay: 0.01, report_loss_per_step: false, seed: '18446744073709551615' };

function mock(failure?: 'readback' | 'checkpoint' | 'early-stop') {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const allocations = new Set<number>();
  const phases = [1, 2, 3, 4, 5, 6];
  let next = 1024, stage = 0, ended = 0, finished = 0, stopped = 0, checkpoints = 0;
  const bits = (v: number) => new Uint32Array(new Float32Array([v]).buffer)[0];
  const wasm = {
    memory, version: () => 1, training_version: () => 1,
    alloc(n: number) { const ptr = next; next += Math.max(4, Math.ceil(n / 4) * 4); allocations.add(ptr); return ptr; },
    dealloc(ptr: number) { return allocations.delete(ptr) ? 0 : -1; },
    training_begin(ptr: number, len: number, settings: number) {
      assert.equal(new TextDecoder().decode(new Uint8Array(memory.buffer, ptr, len)), text);
      assert.deepEqual([...new Uint32Array(memory.buffer, settings + 40, 2)], [0xffffffff, 0xffffffff]);
      return 0;
    },
    training_advance(ptr: number) {
      memory.grow(1);
      const kind = failure === 'early-stop' ? 7 : phases[stage];
      const count = kind === 2 ? 1 : kind === 5 ? 4 : 2;
      new Uint32Array(memory.buffer, ptr, 8).set([kind, 1, 8, count, stage >= 2 ? 1 : 0, 4, bits(1.5), bits(1.6)]);
      return 0;
    },
    training_accept(ptr: number, count: number) {
      assert.equal(new Float32Array(memory.buffer, ptr, count)[0], 0);
      stage++; return 0;
    },
    training_stop() { stopped++; return 0; },
    training_result_ptr() { new Uint8Array(memory.buffer, 16, 3).set([1, 2, 3]); return 16; },
    training_result_len: () => 3,
    training_end() { ended++; return 0; }, error_ptr: () => 0, error_len: () => 0,
  } as unknown as TrainingExports;
  const gpu = {
    device: { limits: { maxBufferSize: 100_000, maxStorageBufferBindingSize: 100_000 } },
    begin() {},
    async readback(_handle: number, offset: number, count: number) {
      memory.grow(1); assert.equal(offset, 8);
      if (failure === 'readback') throw new Error('device lost');
      return new Float32Array(count);
    },
    async checkpoint() { checkpoints++; if (failure === 'checkpoint') throw new Error('GPU validation failed'); },
    async finish() { finished++; },
  } as unknown as GpuHost;
  return { wasm, gpu, allocations, stats: () => ({ ended, finished, stopped, checkpoints }) };
}

test('training preserves u64 seeds, refreshes grown memory, exports and frees ownership', async () => {
  const m = mock(), progress: TrainingProgress[] = [];
  const result = await runTraining(m.wasm, m.gpu, text, options, () => true, (p) => progress.push(p));
  assert.deepEqual([...result.bytes], [1, 2, 3]);
  assert.equal(result.stopped, true); assert.equal(result.completed, 1);
  assert.equal(result.seed, options.seed);
  assert.equal(result.trainLoss, 1.5);
  assert.equal(progress.at(-1)?.phase, 'complete');
  assert.equal(m.allocations.size, 0);
  assert.deepEqual(m.stats(), { ended: 1, finished: 1, stopped: 1, checkpoints: 5 });
});

test('GPU errors and stopping before the first step close the session without an export', async () => {
  for (const failure of ['readback', 'checkpoint', 'early-stop'] as const) {
    const m = mock(failure);
    await assert.rejects(runTraining(m.wasm, m.gpu, text, options, () => true, () => {}));
    assert.equal(m.allocations.size, 0);
    assert.equal(m.stats().ended, 1); assert.equal(m.stats().finished, 1);
  }
});

test('invalid settings and GPU tensor limits reject before model allocation', async () => {
  assert.ok(trainingErrors({ ...options, iterations: 0xffffffff }, text).length);
  assert.ok(trainingErrors({ ...options, seed: '18446744073709551616' }, text).length);
  const m = mock();
  await assert.rejects(runTraining(m.wasm, m.gpu, text, { ...options, batch_size: 1_000_000 }, () => false, () => {}), /limits/);
  assert.equal(m.allocations.size, 0);
});
