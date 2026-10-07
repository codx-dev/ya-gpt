export type Backend = 'webgpu' | 'simd';

export interface ModelConfig {
  block_size: number;
  n_embd: number;
  n_head: number;
  n_layer: number;
  dropout: number;
}

export interface RichModelHeader {
  name: string;
  authors: string;
  version: string;
  fp: string;
  params: number;
  train_loss: number;
  validation_loss: number;
  config: ModelConfig;
  vocab_size: number;
  /** Number of tokens in the training data split; version 0 defaults to zero. */
  train_tokens: number;
}

export type ModelMetadata = RichModelHeader;

export interface GenerateOptions {
  backend: Backend;
  prompt: string;
  num_tokens: number;
  /** Decimal u64 string, never a JavaScript number. Empty means random. */
  seed: string;
}

export interface TrainOptions {
  backend: Backend;
  model: ModelConfig;
  batch_size: number;
  iterations: number;
  learning_rate: number;
  weight_decay: number;
  report_loss_per_step: boolean;
  seed: string;
}

export interface TrainingProgress {
  phase: 'initializing' | 'training' | 'evaluating' | 'exporting' | 'complete';
  completed: number;
  total: number;
  elapsedMs: number;
  trainLoss?: number;
  validationLoss?: number;
}

export interface TrainingResult {
  bytes: Uint8Array<ArrayBuffer>;
  completed: number;
  total: number;
  seed: string;
  trainLoss: number;
  validationLoss: number;
  stopped: boolean;
}

export interface TrainingJob {
  stop(): void;
  result: Promise<TrainingResult>;
}
