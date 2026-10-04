import type { ModelMetadata } from './types';
import type { TemporaryModel } from '../storage';

export interface LoadedModel {
  file: File;
  name: string;
  source: string;
  metadata: ModelMetadata;
  remove?: () => Promise<void>;
}

/** Owns replacement/cancellation separately from React so stale reads cannot win. */
export class ModelLoader {
  current?: LoadedModel;
  private controller?: AbortController;
  private readonly inspect: (file: File) => Promise<ModelMetadata>;
  private readonly cleanupError: (error: unknown) => void;

  constructor(
    inspect: (file: File) => Promise<ModelMetadata>,
    cleanupError: (error: unknown) => void = () => {},
  ) { this.inspect = inspect; this.cleanupError = cleanupError; }

  cancel() { this.controller?.abort(); }

  async load(
    acquire: (signal: AbortSignal) => Promise<TemporaryModel | { file: File }>,
    name: string,
    source: string,
  ): Promise<LoadedModel | undefined> {
    this.cancel();
    const controller = new AbortController();
    this.controller = controller;
    let candidate: Awaited<ReturnType<typeof acquire>> | undefined;
    try {
      candidate = await acquire(controller.signal);
      controller.signal.throwIfAborted();
      const metadata = await this.inspect(candidate.file);
      controller.signal.throwIfAborted();
      const next = { ...candidate, name, source, metadata };
      const previous = this.current;
      // Commit synchronously: cancellation cannot leave current pointing to a deleted file.
      this.current = next;
      candidate = undefined; // Ownership passes to current.
      void previous?.remove?.().catch(this.cleanupError);
      return next;
    } catch (error) {
      if (candidate && 'remove' in candidate) await candidate.remove().catch(this.cleanupError);
      if (controller.signal.aborted) return;
      throw error;
    }
  }

  async clear() {
    this.cancel();
    const previous = this.current;
    this.current = undefined;
    await previous?.remove?.();
  }
}
