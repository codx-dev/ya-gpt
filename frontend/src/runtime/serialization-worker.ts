import wasmUrl from './generated/ya_gpt_wasm.wasm?url';
import { serializeModel } from './serialization-bridge';
import type { SerializationExports } from './serialization-bridge';

export type SerializationRequest = { file: File; author: string };
export type SerializationReply = { type: 'result'; bytes: Uint8Array<ArrayBuffer> } | { type: 'error'; message: string };

const worker = self as unknown as {
  onmessage: ((event: MessageEvent<SerializationRequest>) => void) | null;
  postMessage(value: SerializationReply, transfer?: Transferable[]): void;
};
worker.onmessage = ({ data: { file, author } }) => {
  worker.onmessage = null;
  void (async () => {
    try {
      const response = await fetch(wasmUrl);
      if (!response.ok) throw new Error(`WASM download failed: HTTP ${response.status}`);
      const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), { ya_gpu: {
        // This instance only calls the Naive serializer. GPU imports cannot allocate resources.
        buffer_create: () => 0, buffer_upload: () => -1, buffer_release: () => {}, dispatch: () => -1,
      } });
      const bytes = serializeModel(instance.exports as unknown as SerializationExports, new Uint8Array(await file.arrayBuffer()), author);
      worker.postMessage({ type: 'result', bytes }, [bytes.buffer]);
    } catch (error) {
      worker.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    }
  })();
};
