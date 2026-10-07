import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

test('the Rust WASM exposes ABI v1 with only handwritten GPU imports', async () => {
  const bytes = await readFile(new URL('./runtime/generated/ya_gpt_wasm.wasm', import.meta.url));
  assert.ok(WebAssembly.validate(bytes));
  const module = await WebAssembly.compile(bytes);
  const imports = WebAssembly.Module.imports(module);
  assert.deepEqual(imports.map((i) => `${i.module}.${i.name}`).sort(), [
    'ya_gpu.buffer_create', 'ya_gpu.buffer_release', 'ya_gpu.buffer_upload', 'ya_gpu.dispatch',
  ]);
  const instance = await WebAssembly.instantiate(module, { ya_gpu: {
    buffer_create: () => 0, buffer_upload: () => -1, buffer_release: () => {}, dispatch: () => -1,
  } });
  assert.equal((instance.exports.version as () => number)(), 1);
  for (const name of ['model_inspect', 'model_inspect_result_ptr', 'model_inspect_result_len', 'model_inspect_end']) {
    assert.equal(typeof instance.exports[name], 'function', name);
  }
  assert.equal((instance.exports.training_version as () => number)(), 1);
  assert.equal((instance.exports.training_end as () => number)(), 0);
  assert.equal((instance.exports.training_stop as () => number)(), -1);
  const alloc = instance.exports.alloc as (n: number) => number;
  const dealloc = instance.exports.dealloc as (p: number) => number;
  const ptr = alloc(13);
  assert.equal(ptr % 4, 0);
  assert.equal(dealloc(ptr), 0);
  assert.equal(dealloc(ptr), -1);
});
