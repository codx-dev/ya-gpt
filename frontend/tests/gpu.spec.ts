import { expect, test } from '@playwright/test';
import fixture from './fixtures/gpu.json' with { type: 'json' };
import trainingFixture from './fixtures/gpu-training.json' with { type: 'json' };

test('WebGPU primitives and a transformer block match Rust Naive fixtures', async ({ page }) => {
  await page.goto('/tests/gpu.html');
  const support = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
  if (process.env.YA_GPT_REQUIRE_GPU === '1') expect(support, 'A WebGPU adapter is required').toBe(true);
  test.skip(!support, 'No WebGPU adapter; set YA_GPT_REQUIRE_GPU=1 to require GPU execution.');
  const results = await page.evaluate(async (f) => {
    const path = '/src/runtime/webgpu.ts';
    const { createGpuHost } = await import(path);
    const host = await createGpuHost();
    host.begin();
    const handles: number[] = [];
    const alloc = (n: number) => { const h: number = host.create(n * 4); handles.push(h); return h; };
    const upload = (values: number[], indices = false) => {
      const h = alloc(values.length); host.upload(h, indices ? new Uint32Array(values) : new Float32Array(values)); return h;
    };
    const read = async (h: number, n: number): Promise<number[]> => Array.from(await host.readback(h, 0, n));
    const bits = (n: number) => new Uint32Array(new Float32Array([n]).buffer)[0];
    try {
      const l = f.linear, nx = f.norm, e = f.embedding;
      const x = upload(l.x), weight = upload(l.w), bias = upload(l.b), out = alloc(15);
      host.dispatch(1, [out, 15, bits(99)]);
      host.dispatch(2, [x, weight, bias, out, 3, 7, 5]);
      const biased = await read(out, 15);
      host.dispatch(2, [x, weight, 0, out, 3, 7, 5]);
      const unbiased = await read(out, 15);
      const normalized = alloc(18);
      host.dispatch(3, [upload(nx.x), upload(nx.gamma), upload(nx.beta), normalized, 3, 6, bits(1e-5)]);
      const norm = await read(normalized, 18);
      const embedded = alloc(36);
      host.dispatch(4, [upload(e.tokens, true), upload(e.table), upload(e.positions), embedded, 36, 3, 6]);
      const embedding = await read(embedded, 36);

      const b = f.block, input = upload(b.input), n1 = alloc(36), qkv = alloc(108), scores = alloc(36), context = alloc(36), residual = alloc(36), n2 = alloc(36), hidden = alloc(144), output = alloc(36);
      host.dispatch(3, [input, upload(b.norm1[0]), upload(b.norm1[1]), n1, 6, 6, bits(1e-5)]);
      host.dispatch(2, [n1, upload(b.qkv), 0, qkv, 6, 6, 18]);
      host.dispatch(5, [qkv, scores, 2, 3, 6, 2]);
      host.dispatch(6, [scores, 2, 3, 2]);
      host.dispatch(7, [qkv, scores, context, 2, 3, 6, 2]);
      host.dispatch(2, [context, upload(b.attention[0]), upload(b.attention[1]), residual, 6, 6, 6]);
      host.dispatch(9, [residual, input, 36]);
      host.dispatch(3, [residual, upload(b.norm2[0]), upload(b.norm2[1]), n2, 6, 6, bits(1e-5)]);
      host.dispatch(2, [n2, upload(b.expand[0]), upload(b.expand[1]), hidden, 6, 6, 24]);
      host.dispatch(8, [hidden, 144]);
      host.dispatch(2, [hidden, upload(b.project[0]), upload(b.project[1]), output, 6, 24, 6]);
      host.dispatch(9, [output, residual, 36]);
      const block = await read(output, 36);
      const probabilities = await read(scores, 36);
      for (const h of handles) host.release(h);
      await host.finish();
      return { biased, unbiased, norm, embedding, block, probabilities, buffers: host.bufferCount };
    } finally { host.dispose(); }
  }, fixture);
  for (const [actual, expected] of [
    [results.biased, fixture.linear.biased], [results.unbiased, fixture.linear.unbiased],
    [results.norm, fixture.norm.expected], [results.embedding, fixture.embedding.expected], [results.block, fixture.block.expected],
  ]) {
    expect(actual).toHaveLength(expected.length);
    actual.forEach((value, i) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value - expected[i])).toBeLessThanOrEqual(3e-4 * (1 + Math.abs(expected[i])));
    });
  }
  for (let row = 0; row < 12; row++) {
    const probs = results.probabilities.slice(row * 3, row * 3 + 3);
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    for (let k = row % 3 + 1; k < 3; k++) expect(probs[k]).toBe(0);
  }
  expect(results.buffers).toBe(0);
});

test('training gradients, stable loss, AdamW and finite checks match Rust Naive', async ({ page }) => {
  await page.goto('/tests/gpu.html');
  const support = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
  if (process.env.YA_GPT_REQUIRE_GPU === '1') expect(support).toBe(true);
  test.skip(!support, 'No WebGPU adapter available.');
  const results = await page.evaluate(async (f) => {
    const path = '/src/runtime/webgpu.ts';
    const { createGpuHost } = await import(path);
    const host = await createGpuHost(true);
    host.begin();
    const handles: number[] = [];
    const alloc = (n: number) => { const h: number = host.create(n * 4); handles.push(h); return h; };
    const upload = (values: number[], indices = false) => {
      const h = alloc(values.length); host.upload(h, indices ? new Uint32Array(values) : new Float32Array(values)); return h;
    };
    const read = async (h: number, n: number): Promise<number[]> => Array.from(await host.readback(h, 0, n));
    const bits = (n: number) => new Uint32Array(new Float32Array([n]).buffer)[0];
    const run = (kind: number, count: number, buffers: number[], params: number[] = []) =>
      host.dispatch(10, [kind, count, ...buffers, ...Array(8 - buffers.length).fill(0), ...params, ...Array(12 - params.length).fill(0)]);
    try {
      const l = f.linear, nx = f.norm, e = f.embedding, ce = f.entropy, opt = f.adamw;
      const x = upload(l.x), w = upload(l.w), dy = upload(l.dy);
      const dx = alloc(21), dw = alloc(35), db = alloc(5);
      // Repeat the pass to verify gradients overwrite rather than accumulate.
      for (let i = 0; i < 2; i++) {
        run(8, 21, [dx, dy, w], [3, 7, 5]);
        run(9, 35, [dw, x, dy], [3, 7, 5]);
        run(10, 5, [db, dy], [3]);
      }
      const linear = [await read(dx, 21), await read(dw, 35), await read(db, 5)];
      const out = alloc(18), normalized = alloc(18), inverse = alloc(3);
      const gamma = upload(nx.gamma), ndy = upload(nx.dy), ndx = alloc(18), dg = alloc(6), nb = alloc(6);
      run(5, 3, [out, upload(nx.x), gamma, upload(nx.beta), normalized, inverse], [6, bits(1e-5)]);
      run(6, 3, [ndx, ndy, gamma, normalized, inverse], [6]);
      run(7, 6, [dg, nb, ndy, normalized], [3]);
      const norm = [await read(out, 18), await read(ndx, 18), await read(dg, 6), await read(nb, 6)];
      const edy = upload(e.dy), dt = alloc(30), dp = alloc(30);
      run(11, 30, [dt, upload(e.tokens, true), edy], [6, 3, 6]);
      run(12, 30, [dp, edy], [6, 3, 6]);
      const embedding = [await read(dt, 30), await read(dp, 30)];
      const gradient = alloc(6), loss = alloc(2), invalid = alloc(2), summary = alloc(2), zero = alloc(1);
      run(16, 2, [gradient, upload(ce.logits), upload(ce.targets, true), loss, invalid], [3, 1]);
      run(20, 1, [summary, loss, invalid, zero], [2, 2, 0]);
      const entropy = [await read(gradient, 6), await read(summary, 2)];
      const values = upload(opt.initial), gradients = upload(opt.gradients), first = alloc(259), second = alloc(259), flags = alloc(2);
      for (let step = 1; step <= 3; step++) {
        run(19, 2, [values, gradients, first, second, flags], [259, bits(0.01), bits(0.03), bits(0.9), bits(0.999), bits(1e-8), bits(1 - Math.fround(Math.fround(0.9) ** step)), bits(1 - Math.fround(Math.fround(0.999) ** step)), 0]);
      }
      const adamw = [await read(values, 259), await read(first, 259), await read(second, 259), await read(flags, 2)];
      const special = Array(259).fill(1); special[258] = Infinity;
      run(18, 2, [flags, upload(special)], [259, 0]);
      const nonFinite = await read(flags, 2);
      // Invalid labels and non-finite logits must be reported instead of exported.
      run(16, 2, [gradient, upload([NaN, 0, 0, 0, 0, 0]), upload([0, 3], true), loss, invalid], [3, 1]);
      const invalidLoss = await read(invalid, 2);
      // Overflow during an update is also an error, even with finite inputs.
      const overflow = upload(Array(259).fill(3e38));
      run(19, 2, [values, overflow, first, second, flags], [259, bits(0.01), bits(0.03), bits(0.9), bits(0.999), bits(1e-8), bits(0.1), bits(0.001), 0]);
      const invalidUpdate = await read(flags, 2);
      await host.checkpoint();
      for (const h of handles) host.release(h);
      await host.finish();
      return { linear, norm, embedding, entropy, adamw, nonFinite, invalidLoss, invalidUpdate, buffers: host.bufferCount };
    } finally { host.dispose(); }
  }, trainingFixture);
  const f = trainingFixture;
  for (const [actual, expected] of [
    [results.linear, [f.linear.dx, f.linear.dw, f.linear.db]],
    [results.norm, [f.norm.out, f.norm.dx, f.norm.dg, f.norm.db]],
    [results.embedding, [f.embedding.dt, f.embedding.dp]],
    [results.entropy, [f.entropy.gradient, [f.entropy.loss, 0]]],
    [results.adamw, [f.adamw.values, f.adamw.first, f.adamw.second, [0, 0]]],
  ]) {
    actual.forEach((buffer, b) => {
      expect(buffer).toHaveLength(expected[b].length);
      buffer.forEach((value, i) => {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - expected[b][i])).toBeLessThanOrEqual(3e-4 * (1 + Math.abs(expected[b][i])));
      });
    });
  }
  expect(results.nonFinite).toEqual([0, 1]);
  expect(results.invalidLoss).toEqual([1, 1]);
  expect(results.invalidUpdate).toEqual([1, 1]);
  expect(results.buffers).toBe(0);
});
