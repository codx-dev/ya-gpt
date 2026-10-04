import assert from 'node:assert/strict';
import { test } from 'node:test';
import { serializeInWorker } from './serialization-client.ts';

test('export workers receive immutable files and terminate on success or failure', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  const workers: FakeWorker[] = [];
  let failPost = false;
  class FakeWorker {
    message?: unknown; terminated = false;
    onmessage?: (event: { data: unknown }) => void;
    onerror?: (event: { message: string }) => void;
    onmessageerror?: () => void;
    postMessage(message: unknown) { if (failPost) throw new Error('post failed'); this.message = message; }
    terminate() { this.terminated = true; }
    constructor() { workers.push(this); }
  }
  Object.defineProperty(globalThis, 'Worker', { value: FakeWorker, configurable: true });
  try {
    const file = new File(['original'], 'trained.bin');
    const result = serializeInWorker(file, '  Ada 🦀  ');
    assert.deepEqual(workers[0].message, { file, author: '  Ada 🦀  ' });
    workers[0].onmessage!({ data: { type: 'result', bytes: new Uint8Array([1, 2, 3]) } });
    assert.deepEqual([...(await result)], [1, 2, 3]);
    assert.equal(await file.text(), 'original');
    for (const failure of ['result', 'crash', 'message', 'post']) {
      failPost = failure === 'post';
      const pending = serializeInWorker(file, '');
      const worker = workers.at(-1)!;
      if (failure === 'result') worker.onmessage!({ data: { type: 'error', message: 'serialization failed' } });
      if (failure === 'crash') worker.onerror!({ message: 'worker failed' });
      if (failure === 'message') worker.onmessageerror!();
      await assert.rejects(pending);
    }
    assert.ok(workers.every((worker) => worker.terminated));
    failPost = false;
    const retry = serializeInWorker(file, '');
    workers.at(-1)!.onmessage!({ data: { type: 'result', bytes: new Uint8Array([4]) } });
    assert.deepEqual([...(await retry)], [4]);
  } finally {
    if (original) Object.defineProperty(globalThis, 'Worker', original);
    else Reflect.deleteProperty(globalThis, 'Worker');
  }
});
