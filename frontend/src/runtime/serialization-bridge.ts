import type { WasmExports } from './bridge';

export interface SerializationExports extends Pick<WasmExports, 'memory' | 'version' | 'alloc' | 'dealloc' | 'error_ptr' | 'error_len'> {
  model_serialize(model: number, modelLen: number, author: number, authorLen: number): number;
  model_serialize_result_ptr(): number;
  model_serialize_result_len(): number;
  model_serialize_end(): number;
}

export function authorError(author: string): string | undefined {
  if (new TextEncoder().encode(author).length > 64 * 1024) return 'Author must not exceed 64 KiB of UTF-8 text.';
}

/** Uses the core serializer on CPU buffers; no GPU host or training session is needed. */
export function serializeModel(wasm: SerializationExports, model: Uint8Array, author: string): Uint8Array<ArrayBuffer> {
  if (wasm.version() !== 1 || ['model_serialize', 'model_serialize_result_ptr', 'model_serialize_result_len', 'model_serialize_end']
    .some((name) => typeof wasm[name as keyof SerializationExports] !== 'function')) {
    throw new Error('Unsupported WASM model serializer. Rebuild WASM and reload.');
  }
  if (!model.length || model.length > 0x7fff_ffff) throw new Error('Invalid model file size.');
  const error = authorError(author);
  if (error) throw new Error(error);
  const owned = new Set<number>();
  let result: Uint8Array<ArrayBuffer> | undefined;
  let failure: unknown;
  const check = (status: number) => {
    if (status < 0) throw new Error(new TextDecoder().decode(new Uint8Array(wasm.memory.buffer,
      wasm.error_ptr() >>> 0, wasm.error_len() >>> 0)) || 'Model serialization failed.');
  };
  const write = (bytes: Uint8Array) => {
    const ptr = wasm.alloc(bytes.length) >>> 0;
    if (!ptr) throw new Error('WASM allocation failed while preparing the model.');
    owned.add(ptr);
    new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
    return ptr;
  };
  try {
    const encoded = new TextEncoder().encode(author);
    const modelPtr = write(model), authorPtr = write(encoded);
    check(wasm.model_serialize(modelPtr, model.length, authorPtr, encoded.length));
    const length = wasm.model_serialize_result_len() >>> 0;
    if (!length) throw new Error('Model serialization returned an empty file.');
    result = new Uint8Array(wasm.memory.buffer, wasm.model_serialize_result_ptr() >>> 0, length).slice();
  } catch (cause) { failure = cause; }
  finally {
    try { check(wasm.model_serialize_end()); } catch (cause) { failure ??= cause; }
    for (const ptr of owned) {
      try { check(wasm.dealloc(ptr)); } catch (cause) { failure ??= cause; }
    }
  }
  if (failure) throw failure;
  if (!result) throw new Error('No serialized model returned.');
  return result;
}
