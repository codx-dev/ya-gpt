import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ModelLoader } from './loader.ts';
import type { ModelMetadata } from './types';

const metadata = {} as ModelMetadata;
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
function candidate(name: string) {
  let removed = false;
  return { file: new File(['model'], name), remove: async () => { removed = true; }, get removed() { return removed; } };
}

test('failed replacement keeps the current model and removes the invalid download', async () => {
  const loader = new ModelLoader(async (file) => { if (file.name === 'bad') throw new Error('Invalid'); return metadata; });
  const old = candidate('old'), bad = candidate('bad');
  await loader.load(async () => old, 'old', 'test');
  await assert.rejects(loader.load(async () => bad, 'bad', 'test'), /Invalid/);
  assert.equal(loader.current?.name, 'old');
  assert.equal(old.removed, false); assert.equal(bad.removed, true);
});

test('a stale inspection cannot replace a newer model', async () => {
  const pending = deferred<ModelMetadata>();
  const started = deferred<void>();
  const loader = new ModelLoader(async (file) => {
    if (file.name === 'slow') { started.resolve(); return pending.promise; }
    return metadata;
  });
  const slow = candidate('slow'), fresh = candidate('fresh');
  const first = loader.load(async () => slow, 'slow', 'test');
  await started.promise;
  await loader.load(async () => fresh, 'fresh', 'test');
  pending.resolve(metadata);
  assert.equal(await first, undefined);
  assert.equal(loader.current?.name, 'fresh'); assert.equal(slow.removed, true); assert.equal(fresh.removed, false);
});

test('cancelling acquisition cleans the late result and preserves current', async () => {
  const loader = new ModelLoader(async () => metadata);
  const old = candidate('old'), late = candidate('late');
  await loader.load(async () => old, 'old', 'test');
  const pending = deferred<typeof late>();
  const loading = loader.load(() => pending.promise, 'late', 'test');
  loader.cancel(); pending.resolve(late);
  assert.equal(await loading, undefined);
  assert.equal(loader.current?.name, 'old'); assert.equal(late.removed, true); assert.equal(old.removed, false);
});

test('successful replacement and unloading remove owned temporary files', async () => {
  const loader = new ModelLoader(async () => metadata);
  const first = candidate('first'), second = candidate('second');
  await loader.load(async () => first, 'first', 'test');
  await loader.load(async () => second, 'second', 'test');
  assert.equal(first.removed, true); assert.equal(second.removed, false);
  await loader.clear();
  assert.equal(loader.current, undefined); assert.equal(second.removed, true);
});
