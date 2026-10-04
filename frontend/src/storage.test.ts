import assert from 'node:assert/strict';
import { test } from 'node:test';
import { downloadError, downloadModel } from './storage.ts';
import type { DownloadProgress, ModelStorage } from './storage';

function memoryStorage() {
  const chunks: Uint8Array[] = [];
  let removed = false, closed = false;
  const storage: ModelStorage = { async create() { return {
    writable: new WritableStream<Uint8Array>({ write(chunk) { chunks.push(chunk); }, close() { closed = true; } }),
    async finish() { assert.ok(closed); return { file: new File(chunks.map((c) => new Uint8Array(c)), 'model.bin'), remove: async () => { removed = true; } }; },
    async remove() { removed = true; },
  }; } };
  return { storage, get removed() { return removed; } };
}

test('streams to storage with byte progress even without content length', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(new Uint8Array([1, 2, 3])));
  const target = memoryStorage();
  const progress: DownloadProgress[] = [];
  const result = await downloadModel('https://example.com/model.bin', target.storage, new AbortController().signal, (p) => progress.push(p));
  assert.deepEqual(new Uint8Array(await result.file.arrayBuffer()), new Uint8Array([1, 2, 3]));
  assert.deepEqual(progress.at(-1), { received: 3, total: undefined });
  assert.equal(target.removed, false);
});

test('reports known totals and ignores compressed content lengths', async (t) => {
  for (const headers of [{ 'content-length': '3' }, { 'content-length': '1', 'content-encoding': 'gzip' }] as Record<string, string>[]) {
    t.mock.method(globalThis, 'fetch', async () => new Response(new Uint8Array([1, 2, 3]), { headers }));
    let progress: DownloadProgress | undefined;
    await downloadModel('https://example.com/model.bin', memoryStorage().storage, new AbortController().signal, (p) => { progress = p; });
    assert.equal(progress?.total, 'content-encoding' in headers ? undefined : 3);
    t.mock.restoreAll();
  }
});

test('cancellation aborts the writer and removes a partial file', async (t) => {
  const controller = new AbortController();
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    start(stream) { stream.enqueue(new Uint8Array([1, 2, 3])); },
  })));
  const target = memoryStorage();
  await assert.rejects(downloadModel('https://example.com/model.bin', target.storage, controller.signal,
    ({ received }) => { if (received) controller.abort(); }), { name: 'AbortError' });
  assert.equal(target.removed, true);
});

test('HTTP failures and unsupported URL schemes never create a file', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 404 }));
  const storage: ModelStorage = { create() { throw new Error('must not create'); } };
  await assert.rejects(downloadModel('https://example.com/missing', storage, new AbortController().signal, () => {}), /HTTP 404/);
  await assert.rejects(downloadModel('file:///model.bin', storage, new AbortController().signal, () => {}), /HTTP or HTTPS/);
});

test('storage failure cancels the response stream', async (t) => {
  let cancelled = false;
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({ cancel() { cancelled = true; } })));
  const storage: ModelStorage = { async create() { throw new DOMException('full', 'QuotaExceededError'); } };
  await assert.rejects(downloadModel('https://example.com/model.bin', storage, new AbortController().signal, () => {}), { name: 'QuotaExceededError' });
  assert.equal(cancelled, true);
  assert.match(downloadError(new DOMException('full', 'QuotaExceededError')), /not enough browser storage/);
  assert.match(downloadError(new TypeError('fetch failed')), /CORS/);
});

test('empty downloads and write errors remove their temporary files', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(new Uint8Array()));
  const target = memoryStorage();
  await assert.rejects(downloadModel('https://example.com/model.bin', target.storage, new AbortController().signal, () => {}), /empty/);
  assert.equal(target.removed, true);
  t.mock.restoreAll();
  t.mock.method(globalThis, 'fetch', async () => new Response(new Uint8Array([1])));
  let removed = false;
  const storage: ModelStorage = { async create() { return {
    writable: new WritableStream({ write() { throw new DOMException('full', 'QuotaExceededError'); } }),
    async finish() { throw new Error('must not finish'); },
    async remove() { removed = true; },
  }; } };
  await assert.rejects(downloadModel('https://example.com/model.bin', storage, new AbortController().signal, () => {}), { name: 'QuotaExceededError' });
  assert.equal(removed, true);
});

test('an already cancelled request makes no fetch', async (t) => {
  t.mock.method(globalThis, 'fetch', () => { throw new Error('must not fetch'); });
  await assert.rejects(downloadModel('https://example.com/model.bin', memoryStorage().storage, AbortSignal.abort(), () => {}), { name: 'AbortError' });
});
