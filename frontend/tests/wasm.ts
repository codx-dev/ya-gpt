import { readFile } from 'node:fs/promises';
import type { InspectionExports } from '../src/runtime/inspection-bridge';
import type { SerializationExports } from '../src/runtime/serialization-bridge';

/** Exercise the real Rust parser/serializer and fail on any attempted GPU access. */
export async function metadataWasm(): Promise<InspectionExports & SerializationExports> {
  const bytes = await readFile(new URL('../src/runtime/generated/ya_gpt_wasm.wasm', import.meta.url));
  const noGpu = () => { throw new Error('Metadata operations must not call GPU imports.'); };
  const { instance } = await WebAssembly.instantiate(bytes, { ya_gpu: {
    buffer_create: noGpu, buffer_upload: noGpu, buffer_release: noGpu, dispatch: noGpu,
  } });
  return instance.exports as unknown as InspectionExports & SerializationExports;
}
