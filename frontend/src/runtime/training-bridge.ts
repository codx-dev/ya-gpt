import type { TrainOptions, TrainingProgress, TrainingResult } from '../model/types';
import { trainErrors } from '../model/validation.ts';
import type { GpuHost } from './gpu-host';
import type { WasmExports } from './bridge';

export interface TrainingExports extends WasmExports {
  training_version(): number;
  training_begin(text: number, length: number, options: number): number;
  training_advance(descriptor: number): number;
  training_accept(ptr: number, count: number): number;
  training_stop(): number;
  training_result_ptr(): number;
  training_result_len(): number;
  training_end(): number;
}

export function trainingErrors(options: TrainOptions, text: string): string[] {
  const errors = trainErrors(options, text);
  if (options.backend !== 'webgpu') errors.push('Training requires WebGPU.');
  if (options.iterations > 0x7fff_ffff) errors.push('Iteration count exceeds the WASM limit.');
  for (const value of [options.batch_size, options.model.block_size, options.model.n_embd, options.model.n_head, options.model.n_layer]) {
    if (value > 0xffff_ffff) { errors.push('Model dimensions exceed the WASM limit.'); break; }
  }
  return errors;
}

export async function runTraining(
  wasm: TrainingExports, gpu: GpuHost, text: string, options: TrainOptions,
  stopping: () => boolean, report: (value: TrainingProgress) => void,
): Promise<TrainingResult> {
  const errors = trainingErrors(options, text);
  if (errors.length) throw new Error(errors.join(' '));
  if (wasm.version() !== 1 || wasm.training_version?.() !== 1) throw new Error('Unsupported WASM training bridge. Rebuild WASM and reload.');
  gpu.memory = wasm.memory;
  const started = performance.now();
  const owned = new Set<number>();
  let begun = false;
  let result: TrainingResult | undefined;
  let failure: unknown;
  const check = (status: number) => {
    if (status < 0) throw gpu.lastError ?? new Error(new TextDecoder().decode(new Uint8Array(wasm.memory.buffer, wasm.error_ptr() >>> 0, wasm.error_len())) || 'Training failed.');
  };
  const alloc = (size: number) => {
    const ptr = wasm.alloc(size) >>> 0;
    if (!ptr) throw new Error('WASM allocation failed. Reduce the model or dataset size.');
    owned.add(ptr); return ptr;
  };
  const free = (ptr: number) => { check(wasm.dealloc(ptr)); owned.delete(ptr); };
  try {
    gpu.begin(); begun = true;
    let seed = options.seed;
    if (!seed) {
      const words = crypto.getRandomValues(new Uint32Array(2));
      seed = (BigInt(words[0]) | BigInt(words[1]) << 32n).toString();
    }
    const value = BigInt(seed);
    const bits = (n: number) => new Uint32Array(new Float32Array([n]).buffer)[0];
    const c = options.model;
    // Reject oversized tensors before model initialization and GPU allocations.
    const vocab = new Set(['\0', ...text]).size;
    const limit = Math.min(gpu.device.limits.maxStorageBufferBindingSize, gpu.device.limits.maxBufferSize, 0x7fff_fffc) / 4;
    for (const n of [c.n_embd * c.n_embd * 4, vocab * c.n_embd, c.block_size * c.n_embd,
      options.batch_size * c.block_size * c.n_embd * 4, options.batch_size * c.n_head * c.block_size ** 2,
      options.batch_size * c.block_size * vocab]) {
      if (!Number.isSafeInteger(n) || n > limit) throw new Error('Training tensors exceed WebGPU limits. Reduce batch size, context length, or model size.');
    }
    const input = new TextEncoder().encode(text);
    if (input.length > 0x7fff_ffff) throw new Error('Dataset exceeds the WASM memory limit.');
    const textPtr = alloc(input.length), optionsPtr = alloc(48);
    new Uint8Array(wasm.memory.buffer, textPtr, input.length).set(input);
    new Uint32Array(wasm.memory.buffer, optionsPtr, 12).set([
      c.block_size, c.n_embd, c.n_head, c.n_layer, bits(c.dropout), options.batch_size,
      options.iterations, bits(options.learning_rate), bits(options.weight_decay), Number(options.report_loss_per_step),
      Number(value & 0xffff_ffffn), Number(value >> 32n),
    ]);
    check(wasm.training_begin(textPtr, input.length, optionsPtr));
    free(textPtr); free(optionsPtr);
    const descriptor = alloc(32);
    let readbackPtr = 0, capacity = 0;
    let stopped = false;
    for (;;) {
      if (stopping() && !stopped) { check(wasm.training_stop()); stopped = true; }
      check(wasm.training_advance(descriptor));
      const fields = Array.from(new Uint32Array(wasm.memory.buffer, descriptor, 8));
      const [kind, handle, offset, count, completed, total] = fields;
      const losses = new Float32Array(new Uint32Array(fields.slice(6)).buffer);
      if (kind === 7) throw new DOMException('Training stopped before the first completed step.', 'AbortError');
      const phase = kind === 6 ? 'complete' : kind === 5 ? 'exporting' : kind >= 3 ? 'evaluating' : 'training';
      report({ phase, completed, total, elapsedMs: performance.now() - started,
        trainLoss: Number.isFinite(losses[0]) ? losses[0] : undefined, validationLoss: Number.isFinite(losses[1]) ? losses[1] : undefined });
      if (kind === 6) {
        const length = wasm.training_result_len() >>> 0;
        if (!length) throw new Error('Training returned an empty model.');
        const bytes = new Uint8Array(wasm.memory.buffer, wasm.training_result_ptr() >>> 0, length).slice();
        result = { bytes, completed, total, seed, trainLoss: losses[0], validationLoss: losses[1], stopped: completed < total };
        break;
      }
      if (kind < 1 || kind > 5 || !count || count > 2 * 1024 * 1024) throw new Error('Invalid training readback descriptor.');
      const data = await gpu.readback(handle, offset, count);
      await gpu.checkpoint();
      if (count > capacity) {
        if (readbackPtr) free(readbackPtr);
        capacity = count; readbackPtr = alloc(count * 4);
      }
      new Float32Array(wasm.memory.buffer, readbackPtr, count).set(data);
      check(wasm.training_accept(readbackPtr, count));
      // Let worker stop messages run even when a readback resolves immediately.
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
  } catch (error) { failure = error; }
  finally {
    try { check(wasm.training_end()); } catch (error) { failure ??= error; }
    for (const ptr of owned) { try { check(wasm.dealloc(ptr)); } catch (error) { failure ??= error; } }
    try { if (begun) await gpu.finish(); } catch (error) { failure ??= error; }
  }
  if (failure) throw failure;
  if (!result) throw new Error('No trained model returned.');
  return result;
}
