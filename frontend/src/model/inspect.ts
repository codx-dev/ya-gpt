import { decodeMultiStream } from '@msgpack/msgpack';
import type { ModelMetadata } from './types';
import { configErrors, positiveInteger } from './validation.ts';

/** Rust msgpacker writes flattened, consecutive values, not a map or array. */
export async function inspectModel(file: Blob): Promise<ModelMetadata> {
  if (!file.size) throw new Error('The model file is empty.');
  const values: unknown[] = [];
  try {
    // Stop after the metadata; never decode tokenizer maps or tensor arrays.
    for await (const value of decodeMultiStream<undefined>(file.stream(), {
      maxStrLength: 64 * 1024,
      maxArrayLength: 0,
      maxMapLength: 0,
      maxBinLength: 0,
      maxExtLength: 0,
    })) {
      values.push(value);
      if (values.length === 13) break;
    }
  } catch {
    throw new Error('Invalid or truncated YA-GPT model metadata.');
  }
  if (values.length !== 13) throw new Error('Invalid or truncated YA-GPT model metadata.');
  const [name, authors, major, fp, params, train_loss, validation_loss,
    block_size, n_embd, n_head, n_layer, dropout, vocab_size] = values;
  if (![name, authors, major, fp].every((v) => typeof v === 'string') ||
      ![params, train_loss, validation_loss, block_size, n_embd, n_head, n_layer, dropout, vocab_size]
        .every((v) => typeof v === 'number' && Number.isFinite(v))) {
    throw new Error('Invalid model metadata types.');
  }
  if (name !== 'ya-gpt' || major !== '0' || fp !== 'f32') {
    throw new Error('Incompatible model: expected ya-gpt version 0 with f32 weights.');
  }
  const metadata = {
    header: { name, authors, major, fp, params, train_loss, validation_loss },
    config: { block_size, n_embd, n_head, n_layer, dropout },
    vocab_size,
  } as ModelMetadata;
  if (!positiveInteger(metadata.header.params) || !positiveInteger(metadata.vocab_size) ||
      configErrors(metadata.config).length) {
    throw new Error('Invalid model dimensions or parameter count.');
  }
  return metadata;
}
