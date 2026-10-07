import type { ModelMetadata } from '../model/types';
import type { WasmExports } from './bridge';

export interface InspectionExports extends Pick<WasmExports, 'memory' | 'version' | 'alloc' | 'dealloc' | 'error_ptr' | 'error_len'> {
  model_inspect(model: number, modelLen: number): number;
  model_inspect_result_ptr(): number;
  model_inspect_result_len(): number;
  model_inspect_end(): number;
}

export function checkInspector(wasm: InspectionExports): void {
  if (wasm.version() !== 1 || ['model_inspect', 'model_inspect_result_ptr', 'model_inspect_result_len', 'model_inspect_end']
    .some((name) => typeof wasm[name as keyof InspectionExports] !== 'function')) {
    throw new Error('Unsupported WASM model inspector. Rebuild WASM and reload.');
  }
}

/** Synchronous calls keep the reusable inspection instance isolated between requests. */
export function inspectMetadata(wasm: InspectionExports, prefix: Uint8Array): ModelMetadata {
  checkInspector(wasm);
  let ptr = 0;
  let result: ModelMetadata | undefined;
  let failure: unknown;
  const text = (ptr: number, len: number) => new TextDecoder('utf-8', { fatal: true })
    .decode(new Uint8Array(wasm.memory.buffer, ptr >>> 0, len >>> 0));
  const check = (status: number) => {
    if (status < 0) throw new Error(text(wasm.error_ptr(), wasm.error_len()) || 'Model inspection failed.');
  };
  try {
    ptr = wasm.alloc(prefix.length) >>> 0;
    if (!ptr) throw new Error('WASM allocation failed while inspecting the model.');
    new Uint8Array(wasm.memory.buffer, ptr, prefix.length).set(prefix);
    check(wasm.model_inspect(ptr, prefix.length));
    // Decode before cleanup, and recreate memory views after calls that may grow memory.
    result = JSON.parse(text(wasm.model_inspect_result_ptr(), wasm.model_inspect_result_len())) as ModelMetadata;
  } catch (error) { failure = error; }
  finally {
    try { check(wasm.model_inspect_end()); } catch (error) { failure ??= error; }
    if (ptr) {
      try { check(wasm.dealloc(ptr)); } catch (error) { failure ??= error; }
    }
  }
  if (failure) throw failure;
  if (!result) throw new Error('No model metadata returned.');
  return result;
}
