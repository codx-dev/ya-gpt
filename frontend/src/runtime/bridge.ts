import type { GenerateOptions } from '../model/types';
import { generateErrors } from '../model/validation.ts';
import type { GpuHost } from './gpu-host';

export interface WasmExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  version(): number;
  alloc(bytes: number): number;
  dealloc(ptr: number): number;
  generation_begin(model: number, modelLen: number, prompt: number, promptLen: number, count: number, seedLo: number, seedHi: number): number;
  generation_forward(descriptor: number): number;
  generation_accept_logits(ptr: number, count: number): number;
  generation_result_ptr(): number;
  generation_result_len(): number;
  generation_end(): number;
  error_ptr(): number;
  error_len(): number;
}

export class WasmBridge {
  readonly wasm: WasmExports;
  readonly gpu: GpuHost;
  private busy = false;
  private failure?: Error;
  constructor(wasm: WasmExports, gpu: GpuHost) {
    this.wasm = wasm; this.gpu = gpu;
    if (wasm.version() !== 1) throw new Error('Unsupported WASM bridge version.');
    gpu.memory = wasm.memory;
  }

  private text(ptr: number, len: number): string {
    return new TextDecoder('utf-8', { fatal: true }).decode(new Uint8Array(this.wasm.memory.buffer, ptr >>> 0, len));
  }

  private check(status: number): number {
    if (status < 0) throw this.gpu.lastError ?? new Error(this.text(this.wasm.error_ptr(), this.wasm.error_len()) || 'WASM operation failed.');
    return status;
  }

  async generate(model: Uint8Array, options: GenerateOptions): Promise<string> {
    if (this.failure) throw this.failure;
    if (this.busy) throw new Error('A generation is already running.');
    if (options.backend !== 'webgpu') throw new Error('CPU SIMD generation is not implemented.');
    const errors = generateErrors(options);
    if (options.num_tokens > 0xffff_ffff) errors.push('New token count exceeds the WASM limit.');
    if (!model.length || model.length > 0x7fff_ffff) errors.push('Invalid model file size.');
    if (errors.length) throw new Error(errors.join(' '));
    this.busy = true;
    const owned = new Set<number>();
    let begun = false;
    let failure: unknown;
    let result = '';
    const alloc = (size: number): number => {
      const ptr = this.wasm.alloc(size) >>> 0;
      if (!ptr) throw new Error('WASM allocation failed.');
      owned.add(ptr); return ptr;
    };
    const free = (ptr: number) => { this.check(this.wasm.dealloc(ptr)); owned.delete(ptr); };
    const write = (bytes: Uint8Array) => {
      const ptr = alloc(bytes.length);
      new Uint8Array(this.wasm.memory.buffer, ptr, bytes.length).set(bytes);
      return ptr;
    };
    try {
      this.gpu.begin(); begun = true;
      const prompt = new TextEncoder().encode(options.prompt);
      if (prompt.length > 0x7fff_ffff) throw new Error('Prompt exceeds the WASM memory limit.');
      const modelPtr = write(model), promptPtr = write(prompt);
      let seed: bigint;
      if (options.seed) seed = BigInt(options.seed);
      else {
        const words = crypto.getRandomValues(new Uint32Array(2));
        seed = BigInt(words[0]) | (BigInt(words[1]) << 32n);
      }
      this.check(this.wasm.generation_begin(modelPtr, model.length, promptPtr, prompt.length, options.num_tokens, Number(seed & 0xffff_ffffn), Number(seed >> 32n)));
      free(modelPtr); free(promptPtr);
      const descriptor = alloc(12);
      let logitsPtr = 0, vocab = 0;
      for (let i = 0; i < options.num_tokens; i++) {
        this.check(this.wasm.generation_forward(descriptor));
        // Copy scalar fields; never hold a WASM view across the await below.
        const view = new DataView(this.wasm.memory.buffer, descriptor, 12);
        const handle = view.getUint32(0, true), offset = view.getUint32(4, true), count = view.getUint32(8, true);
        if (!logitsPtr) { vocab = count; logitsPtr = alloc(count * 4); }
        if (count !== vocab) throw new Error('Generation vocabulary changed.');
        const logits = await this.gpu.readback(handle, offset, count);
        new Float32Array(this.wasm.memory.buffer, logitsPtr, count).set(logits);
        const done = this.check(this.wasm.generation_accept_logits(logitsPtr, count));
        if ((done === 1) !== (i === options.num_tokens - 1)) throw new Error('Invalid generation completion state.');
      }
      result = this.text(this.wasm.generation_result_ptr(), this.wasm.generation_result_len());
    } catch (error) {
      failure = error;
      if (error instanceof WebAssembly.RuntimeError) {
        failure = this.failure = new Error('The WASM runtime trapped. Reload to start a new runtime.', { cause: error });
      }
    }
    finally {
      try {
        this.check(this.wasm.generation_end());
        for (const ptr of owned) this.check(this.wasm.dealloc(ptr));
      } catch (error) { failure ??= error; }
      try { if (begun) await this.gpu.finish(); }
      catch (error) { failure ??= error; }
      if (this.failure) this.gpu.dispose();
      this.busy = false;
    }
    if (failure) throw failure;
    return result;
  }
}
