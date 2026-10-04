import assert from 'node:assert/strict';
import { test } from 'node:test';
import { GpuHost } from './gpu-host.ts';
import type { GpuDevice, GpuPipeline } from './webgpu-types';

function mock() {
  const created: { size: number; destroyed: boolean; data: ArrayBuffer }[] = [];
  const dispatches: number[][] = [];
  const writes: Uint8Array[] = [];
  const bindings: unknown[][] = [];
  let scopes = 0;
  let submits = 0;
  let resolveMap: (() => void) | undefined;
  let resolveLost: (info: { message: string }) => void = () => {};
  let wait = false;
  const device = {
    limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 2048, maxComputeWorkgroupsPerDimension: 1024 },
    lost: new Promise((resolve) => { resolveLost = resolve; }), addEventListener() {}, destroy() {}, pushErrorScope() { scopes++; }, async popErrorScope() { scopes--; return null; },
    createBuffer({ size }: { size: number }) {
      const b = { size, destroyed: false, data: new ArrayBuffer(size), destroy() { this.destroyed = true; },
        mapAsync() { return wait ? new Promise<void>((r) => { resolveMap = r; }) : Promise.resolve(); },
        unmap() {}, getMappedRange() { return this.data; } };
      created.push(b); return b;
    },
    queue: { writeBuffer(_b: unknown, _offset: number, view: ArrayBufferView) { writes.push(new Uint8Array(view.buffer, view.byteOffset, view.byteLength).slice()); }, submit() { submits++; }, async onSubmittedWorkDone() {} },
    createBindGroup({ entries }: { entries: { resource: { buffer: unknown } }[] }) { bindings.push(entries.map((e) => e.resource.buffer)); return {}; },
    createCommandEncoder() { return {
      beginComputePass() { return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups(x: number, y: number) { dispatches.push([x, y]); }, end() {} }; },
      copyBufferToBuffer() {}, finish() { return {}; },
    }; },
  } as unknown as GpuDevice;
  const pipelines = new Map(Array.from({ length: 10 }, (_, i) => [i + 1, { getBindGroupLayout() { return {}; } } as unknown as GpuPipeline]));
  const host = new GpuHost(device, pipelines);
  return { host, created, dispatches, writes, bindings, scopes: () => scopes, submissions: () => submits, delayMap: () => { wait = true; }, completeMap: () => resolveMap?.(), lose: () => resolveLost({ message: 'test device loss' }) };
}

test('commands batch until readback and retired resources survive pending work', async () => {
  const m = mock(); m.host.begin();
  const a = m.host.create(12), b = m.host.create(12);
  m.host.dispatch(1, [a, 3, 0x3f800000]);
  m.host.dispatch(8, [a, 3]);
  m.host.release(b);
  assert.equal(m.submissions(), 0);
  assert.ok(m.created.every((b) => !b.destroyed));
  m.delayMap();
  const pending = m.host.readback(a, 0, 3);
  assert.equal(m.submissions(), 1);
  assert.ok(m.created.every((b) => !b.destroyed));
  m.completeMap(); await pending;
  assert.ok(m.created[2].destroyed); // retired b
  assert.ok(!m.created[1].destroyed); // live a
  assert.equal(m.host.bufferCount, 1);
  m.host.release(a); await m.host.finish();
  assert.equal(m.host.bufferCount, 0);
  assert.ok(m.created[1].destroyed);
});

test('imports read fresh WASM views after growth and preserve per-dispatch uniforms', async () => {
  const m = mock(); m.host.begin();
  const memory = new WebAssembly.Memory({ initial: 1 }); m.host.memory = memory;
  const handle = m.host.imports.buffer_create(4);
  memory.grow(1);
  new Uint32Array(memory.buffer, 65536, 3).set([handle, 1, 0x3f800000]);
  assert.equal(m.host.imports.dispatch(1, 65536, 3), 0);
  new Uint32Array(memory.buffer, 65536, 3).set([handle, 1, 0x40000000]);
  assert.equal(m.host.imports.dispatch(1, 65536, 3), 0);
  assert.equal(new Uint32Array(m.writes[0].buffer)[1], 0x3f800000);
  assert.equal(new Uint32Array(m.writes[1].buffer)[1], 0x40000000);
  await m.host.readback(handle, 0, 1);
  m.host.release(handle); await m.host.finish();
});

test('bad imports become statuses and allocation/shape limits are checked', () => {
  const { host } = mock();
  host.memory = new WebAssembly.Memory({ initial: 1 });
  assert.equal(host.imports.buffer_create(2049), 0);
  assert.match(host.lastError!.message, /limits/);
  assert.equal(host.imports.buffer_upload(999, 0, 4), -1);
  assert.equal(host.imports.dispatch(1, 65532, 3), -1);
  const a = host.create(4);
  assert.throws(() => host.dispatch(8, [a, 3]), /shape/);
  host.release(a);
  assert.throws(() => host.upload(a, new Float32Array(1)), /handle/);
});

test('device loss interrupts pending readback and releases all request buffers', async () => {
  const m = mock(); m.host.begin();
  const h = m.host.create(4); m.delayMap();
  const readback = m.host.readback(h, 0, 1);
  m.lose();
  await assert.rejects(readback, /device lost/);
  await assert.rejects(m.host.finish(), /device lost/);
  assert.equal(m.host.bufferCount, 0);
  assert.ok(m.created[1].destroyed);
  assert.throws(() => m.host.create(4), /device lost/);
});

test('training descriptors retain all parameters and never alias writable bindings', async () => {
  const m = mock(); m.host.begin();
  const a = m.host.create(12), b = m.host.create(12);
  const params = Array.from({ length: 12 }, (_, i) => i + 100);
  m.host.dispatch(10, [1, 3, a, b, 0, 0, 0, 0, 0, 0, ...params]);
  assert.deepEqual([...new Uint32Array(m.writes[0].buffer)], [1, 3, ...params, 0, 0]);
  assert.equal(m.bindings[0].length, 9);
  assert.equal(new Set(m.bindings[0]).size, 9);
  assert.throws(() => m.host.dispatch(10, [1, 3, a, a, 0, 0, 0, 0, 0, 0, ...params]), /alias/);
  await m.host.readback(a, 0, 3);
  await m.host.checkpoint();
  assert.equal(m.scopes(), 3);
  assert.equal(m.host.bufferCount, 2);
  assert.ok(!m.created[1].destroyed);
  await m.host.finish(); m.host.dispose();
  assert.equal(m.scopes(), 0);
  assert.ok(m.created.every((buffer) => buffer.destroyed));
});

test('large tensors dispatch in two dimensions and reject excess workgroups', async () => {
  const m = mock(); m.host.begin();
  Object.assign(m.host.device.limits, { maxComputeWorkgroupsPerDimension: 2 });
  const a = m.host.create(129 * 4);
  m.host.dispatch(8, [a, 129]);
  assert.deepEqual(m.dispatches, [[2, 2]]);
  const b = m.host.create(257 * 4);
  assert.throws(() => m.host.dispatch(8, [b, 257]), /dispatch limits/);
  await m.host.finish();
});
