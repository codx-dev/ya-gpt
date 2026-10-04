export interface DownloadProgress { received: number; total?: number }
export interface TemporaryModel { file: File; remove(): Promise<void> }
export interface ModelStorage {
  create(): Promise<{ writable: WritableStream<Uint8Array>; finish(): Promise<TemporaryModel>; remove(): Promise<void> }>;
}

export const storageRoot = 'ya-gpt-temporary-models';
export const sessionLock = (id: string) => `${storageRoot}:${id}`;

export function supportsTemporaryStorage(): boolean {
  return !!navigator.storage?.getDirectory && !!navigator.locks && typeof globalThis.crypto?.randomUUID === 'function';
}

export class TemporaryStorage implements ModelStorage {
  private id = '';
  private directory?: FileSystemDirectoryHandle;
  private root?: FileSystemDirectoryHandle;
  private ready?: Promise<void>;
  private lockTask?: Promise<void>;
  private release?: () => void;

  prepare(): Promise<void> {
    this.ready ??= new Promise<void>((resolve, reject) => {
      if (!supportsTemporaryStorage()) {
        reject(new Error('Temporary disk storage is unavailable. Use HTTPS or localhost in a browser supporting OPFS and Web Locks, or select a local file.'));
        return;
      }
      this.id = crypto.randomUUID();
      const lifetime = new Promise<void>((done) => { this.release = done; });
      this.lockTask = navigator.locks.request(sessionLock(this.id), async () => {
        try {
          const opfs = await navigator.storage.getDirectory();
          this.root = await opfs.getDirectoryHandle(storageRoot, { create: true });
          // Acquire the session lock before publishing its directory to other tabs.
          this.directory = await this.root.getDirectoryHandle(this.id, { create: true });
          for await (const [id, entry] of this.root.entries()) {
            if (id === this.id || entry.kind !== 'directory') continue;
            await navigator.locks.request(sessionLock(id), { ifAvailable: true }, async (lock) => {
              if (lock) {
                try { await this.root!.removeEntry(id, { recursive: true }); }
                catch (error) { if (!(error instanceof DOMException && error.name === 'NotFoundError')) throw error; }
              }
            });
          }
          resolve();
          await lifetime;
        } catch (error) { reject(error); }
      }).catch(reject);
    });
    return this.ready;
  }

  async create() {
    await this.prepare();
    const directory = this.directory!;
    const id = `${crypto.randomUUID()}.bin`;
    const remove = async () => {
      try { await directory.removeEntry(id); }
      catch (error) { if (!(error instanceof DOMException && error.name === 'NotFoundError')) throw error; }
    };
    const handle = await directory.getFileHandle(id, { create: true });
    try {
      const writable = await handle.createWritable();
      return {
        writable,
        finish: async () => ({ file: await handle.getFile(), remove }),
        remove,
      };
    } catch (error) {
      await remove();
      throw error;
    }
  }

  async dispose(): Promise<void> {
    if (!this.ready) return;
    try {
      await this.ready;
      await this.root!.removeEntry(this.id, { recursive: true });
    } finally { this.release?.(); await this.lockTask; }
  }
}

export function downloadError(error: unknown): string {
  if (error instanceof DOMException && error.name === 'QuotaExceededError') {
    return 'There is not enough browser storage for this model. Clear temporary files or select a local file.';
  }
  if (error instanceof DOMException && ['SecurityError', 'NotAllowedError'].includes(error.name)) {
    return 'Browser storage access was denied. Allow site storage or select a local file.';
  }
  if (error instanceof TypeError) {
    return 'The download failed. Check the URL, connection, and whether the server allows cross-origin requests (CORS).';
  }
  return error instanceof Error ? error.message : 'Unable to load the model.';
}

export async function downloadModel(
  url: string,
  storage: ModelStorage,
  signal: AbortSignal,
  onProgress: (progress: DownloadProgress) => void,
): Promise<TemporaryModel> {
  const parsed = new URL(url);
  if (!['https:', 'http:'].includes(parsed.protocol) || parsed.username || parsed.password) {
    throw new Error('Use an HTTP or HTTPS model URL without embedded credentials.');
  }
  signal.throwIfAborted();
  const response = await fetch(parsed, { signal, credentials: 'omit', cache: 'no-store' });
  if (!response.ok) {
    await response.body?.cancel();
    throw new Error(`Download failed: HTTP ${response.status}.`);
  }
  if (!response.body) throw new Error('The download returned no file data.');
  // Content-Length can describe compressed bytes; do not show misleading percentages.
  const length = Number(response.headers.get('content-length'));
  const total = !response.headers.has('content-encoding') && Number.isFinite(length) && length > 0 ? length : undefined;
  let target: Awaited<ReturnType<ModelStorage['create']>>;
  try { target = await storage.create(); }
  catch (error) { await response.body.cancel(); throw error; }
  let received = 0;
  try {
    onProgress({ received, total });
    const progress = new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        received += chunk.byteLength;
        onProgress({ received, total });
        controller.enqueue(chunk);
      },
    });
    await response.body.pipeThrough(progress).pipeTo(target.writable, { signal });
    signal.throwIfAborted();
    if (!received) throw new Error('The downloaded model is empty.');
    return await target.finish();
  } catch (error) {
    // pipeTo aborts the writable before settling, so no partial file survives.
    await target.remove();
    throw error;
  }
}
