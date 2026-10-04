import type { SerializationRequest, SerializationReply } from './serialization-worker';

export function serializeInWorker(file: File, author: string): Promise<Uint8Array<ArrayBuffer>> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./serialization-worker.ts', import.meta.url), { type: 'module' });
    let settled = false;
    const finish = (error?: Error, bytes?: Uint8Array<ArrayBuffer>) => {
      if (settled) return;
      settled = true; worker.terminate();
      if (error) reject(error);
      else if (bytes) resolve(bytes);
      else reject(new Error('No serialized model returned.'));
    };
    worker.onmessage = ({ data }: MessageEvent<SerializationReply>) => {
      if (data.type === 'result') finish(undefined, data.bytes);
      else finish(new Error(data.message));
    };
    worker.onerror = (event) => finish(new Error(event.message || 'Model serialization worker failed.'));
    worker.onmessageerror = () => finish(new Error('Unable to read serialized model response.'));
    try { worker.postMessage({ file, author } satisfies SerializationRequest); }
    catch (error) { finish(error instanceof Error ? error : new Error(String(error))); }
  });
}
