import type { GenerateOptions, ModelConfig, TrainOptions } from './types';

export const presets = {
  tiny: { block_size: 32, n_embd: 32, n_head: 4, n_layer: 2, dropout: 0.2, batch_size: 8, iterations: 1_000 },
  small: { block_size: 64, n_embd: 64, n_head: 4, n_layer: 2, dropout: 0.2, batch_size: 16, iterations: 2_000 },
  medium: { block_size: 128, n_embd: 128, n_head: 4, n_layer: 4, dropout: 0.2, batch_size: 32, iterations: 5_000 },
  large: { block_size: 256, n_embd: 256, n_head: 8, n_layer: 6, dropout: 0.2, batch_size: 64, iterations: 10_000 },
} satisfies Record<string, ModelConfig & { batch_size: number; iterations: number }>;
export type Preset = keyof typeof presets;

export function positiveInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value > 0;
}

export function seedError(seed: string): string | undefined {
  if (seed === '') return;
  if (!/^\d{1,20}$/.test(seed) || BigInt(seed) > 18_446_744_073_709_551_615n) {
    return 'Seed must be an unsigned 64-bit integer, or empty for a random seed.';
  }
}

export function configErrors(config: ModelConfig): string[] {
  const errors: string[] = [];
  for (const [key, label] of [
    ['block_size', 'Context length'], ['n_embd', 'Embedding width'],
    ['n_head', 'Attention heads'], ['n_layer', 'Transformer layers'],
  ] as const) {
    if (!positiveInteger(config[key])) errors.push(`${label} must be a positive integer.`);
  }
  if (positiveInteger(config.n_head) && config.n_embd % config.n_head !== 0) {
    errors.push('Embedding width must be divisible by attention heads.');
  }
  if (!Number.isFinite(config.dropout) || config.dropout < 0 || config.dropout >= 1) {
    errors.push('Dropout must be between 0 (inclusive) and 1 (exclusive).');
  }
  return errors;
}

export function generateErrors(options: GenerateOptions): string[] {
  return [
    !positiveInteger(options.num_tokens) ? 'New tokens must be a positive integer.' : undefined,
    seedError(options.seed),
  ].filter((error): error is string => !!error);
}

export function trainErrors(options: TrainOptions, text: string): string[] {
  const errors = configErrors(options.model);
  for (const [key, label] of [['batch_size', 'Batch size'], ['iterations', 'Iterations']] as const) {
    if (!positiveInteger(options[key])) errors.push(`${label} must be a positive integer.`);
  }
  for (const [key, label] of [['learning_rate', 'Learning rate'], ['weight_decay', 'Weight decay']] as const) {
    if (!Number.isFinite(options[key]) || options[key] < 0 || options[key] > 1) {
      errors.push(`${label} must be between 0 and 1, inclusive.`);
    }
  }
  const seed = seedError(options.seed);
  if (seed) errors.push(seed);
  // The char tokenizer splits 90/10; both splits need context + one target.
  const characters = Array.from(text).length;
  const split = Math.floor(characters * 0.9);
  if (split <= options.model.block_size || characters - split <= options.model.block_size) {
    errors.push('Provide enough text for both the 90% training and 10% validation splits to exceed the context length.');
  }
  return errors;
}
