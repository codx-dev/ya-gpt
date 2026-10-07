import type { ModelMetadata } from './types';
import { configErrors, positiveInteger } from './validation.ts';
import { inspectMetadata } from '../runtime/inspection-bridge.ts';
import type { InspectionExports } from '../runtime/inspection-bridge.ts';

/** Four 64 KiB strings plus MessagePack framing and scalar fields; matches the Rust ABI. */
export const METADATA_PREFIX_LIMIT = 4 * 64 * 1024 + 256;

export async function inspectModel(file: Blob, wasm: InspectionExports): Promise<ModelMetadata> {
  if (!file.size) throw new Error('The model file is empty.');
  const prefix = new Uint8Array(await file.slice(0, METADATA_PREFIX_LIMIT).arrayBuffer());
  const metadata = inspectMetadata(wasm, prefix);
  if (!positiveInteger(metadata.params) || !positiveInteger(metadata.vocab_size) ||
      !Number.isSafeInteger(metadata.train_tokens) || metadata.train_tokens < 0 ||
      configErrors(metadata.config).length) {
    throw new Error('Invalid model dimensions, parameter count, or training token count.');
  }
  return metadata;
}
