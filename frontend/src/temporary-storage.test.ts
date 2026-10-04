import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { TestContext } from 'node:test';
import { TemporaryStorage, sessionLock, storageRoot, supportsTemporaryStorage } from './storage.ts';

class Directory {
  kind = 'directory';
  children = new Map<string, Directory>();
  async getDirectoryHandle(name: string, options?: { create?: boolean }) {
    if (!this.children.has(name)) {
      if (!options?.create) throw new DOMException('missing', 'NotFoundError');
      this.children.set(name, new Directory());
    }
    return this.children.get(name)!;
  }
  async *entries() { yield* [...this.children.entries()]; }
  async removeEntry(name: string) {
    if (!this.children.delete(name)) throw new DOMException('missing', 'NotFoundError');
  }
}

function environment(t: TestContext) {
  const root = new Directory();
  const held = new Set<string>();
  const locks = {
    async request(name: string, optionsOrCallback: { ifAvailable?: boolean } | ((lock: object | null) => Promise<void>), callback?: (lock: object | null) => Promise<void>) {
      const fn = typeof optionsOrCallback === 'function' ? optionsOrCallback : callback!;
      if (held.has(name)) {
        assert.ok(typeof optionsOrCallback === 'object' && optionsOrCallback.ifAvailable);
        return fn(null);
      }
      held.add(name);
      try { return await fn({ name }); } finally { held.delete(name); }
    },
  };
  for (const [key, value] of Object.entries({ storage: { getDirectory: async () => root }, locks })) {
    const original = Object.getOwnPropertyDescriptor(navigator, key);
    Object.defineProperty(navigator, key, { value, configurable: true });
    t.after(() => { if (original) Object.defineProperty(navigator, key, original); else Reflect.deleteProperty(navigator, key); });
  }
  return { root, held };
}

test('startup reclaims abandoned directories while protecting active sessions', async (t) => {
  const { root, held } = environment(t);
  const models = await root.getDirectoryHandle(storageRoot, { create: true });
  await models.getDirectoryHandle('abandoned', { create: true });
  const first = new TemporaryStorage();
  await first.prepare();
  assert.equal(models.children.has('abandoned'), false);
  const [firstId] = models.children.keys();
  assert.ok(held.has(sessionLock(firstId)));
  const second = new TemporaryStorage();
  await second.prepare();
  assert.equal(models.children.size, 2);
  assert.ok(models.children.has(firstId));
  await first.dispose();
  assert.equal(models.children.has(firstId), false);
  assert.equal(held.has(sessionLock(firstId)), false);
  await second.dispose();
  assert.equal(models.children.size, 0);
});

test('concurrent startups can both finish cleanup without deleting either session', async (t) => {
  const { root } = environment(t);
  const models = await root.getDirectoryHandle(storageRoot, { create: true });
  await models.getDirectoryHandle('orphan', { create: true });
  const first = new TemporaryStorage(), second = new TemporaryStorage();
  await Promise.all([first.prepare(), second.prepare()]);
  assert.equal(models.children.has('orphan'), false);
  assert.equal(models.children.size, 2);
  await Promise.all([first.dispose(), second.dispose()]);
  assert.equal(models.children.size, 0);
});

test('an insecure context can construct the app storage object and report unsupported downloads', async (t) => {
  environment(t);
  const original = Object.getOwnPropertyDescriptor(crypto, 'randomUUID');
  Object.defineProperty(crypto, 'randomUUID', { value: undefined, configurable: true });
  t.after(() => { if (original) Object.defineProperty(crypto, 'randomUUID', original); else Reflect.deleteProperty(crypto, 'randomUUID'); });
  const storage = new TemporaryStorage();
  assert.equal(supportsTemporaryStorage(), false);
  await assert.rejects(storage.prepare(), /unavailable/);
});
