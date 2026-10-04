import assert from 'node:assert/strict';
import { test } from 'node:test';
import { startTrainingWorker } from './training-client.ts';
import type { TrainOptions } from '../model/types';

test('worker completion, failure, stopping, and callback errors release the worker', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  const workers: FakeWorker[] = [];
  class FakeWorker {
    messages: unknown[] = []; terminated = false;
    onmessage?: (event: { data: unknown }) => void;
    onerror?: (event: { message: string }) => void;
    postMessage(message: unknown) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    constructor() { workers.push(this); }
  }
  Object.defineProperty(globalThis, 'Worker', { value: FakeWorker, configurable: true });
  try {
    const job = startTrainingWorker('abc', {} as TrainOptions);
    job.stop();
    assert.deepEqual(workers[0].messages[1], { type: 'stop' });
    workers[0].onmessage!({ data: { type: 'result', result: { completed: 2 } } });
    assert.equal((await job.result).completed, 2); assert.ok(workers[0].terminated);
    job.stop(); assert.equal(workers[0].messages.length, 2);
    const failed = startTrainingWorker('abc', {} as TrainOptions);
    workers[1].onerror!({ message: 'worker lost' });
    await assert.rejects(failed.result, /worker lost/); assert.ok(workers[1].terminated);
    const callback = startTrainingWorker('abc', {} as TrainOptions, () => { throw new Error('callback failed'); });
    workers[2].onmessage!({ data: { type: 'progress', progress: {} } });
    await assert.rejects(callback.result, /callback failed/); assert.ok(workers[2].terminated);
  } finally {
    if (original) Object.defineProperty(globalThis, 'Worker', original);
    else Reflect.deleteProperty(globalThis, 'Worker');
  }
});
