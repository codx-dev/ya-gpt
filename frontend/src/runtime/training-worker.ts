import wasmUrl from './generated/ya_gpt_wasm.wasm?url';
import type { TrainOptions, TrainingProgress, TrainingResult } from '../model/types';
import { createGpuHost } from './webgpu';
import { runTraining } from './training-bridge';
import type { TrainingExports } from './training-bridge';

export type TrainingRequest = { type: 'start'; text: string; options: TrainOptions } | { type: 'stop' };
export type TrainingReply = { type: 'progress'; progress: TrainingProgress } | { type: 'result'; result: TrainingResult } | { type: 'error'; name: string; message: string };

// This module is only instantiated as a dedicated worker.
const worker = self as unknown as {
  onmessage: ((event: MessageEvent<TrainingRequest>) => void) | null;
  postMessage(value: TrainingReply, transfer?: Transferable[]): void;
};
let active = false, stopping = false;
worker.onmessage = (event) => {
  if (event.data.type === 'stop') { stopping = true; return; }
  if (active) return;
  active = true;
  const { text, options } = event.data;
  void (async () => {
    let gpu: Awaited<ReturnType<typeof createGpuHost>> | undefined;
    try {
      worker.postMessage({ type: 'progress', progress: { phase: 'initializing', completed: 0, total: options.iterations, elapsedMs: 0 } });
      const response = await fetch(wasmUrl);
      if (!response.ok) throw new Error(`WASM download failed: HTTP ${response.status}`);
      const module = await WebAssembly.compile(await response.arrayBuffer());
      gpu = await createGpuHost(true);
      const instance = await WebAssembly.instantiate(module, { ya_gpu: gpu.imports });
      const result = await runTraining(instance.exports as TrainingExports, gpu, text, options, () => stopping,
        (progress) => worker.postMessage({ type: 'progress', progress }));
      gpu.dispose(); gpu = undefined;
      worker.postMessage({ type: 'result', result }, [result.bytes.buffer]);
    } catch (error) {
      gpu?.dispose(); gpu = undefined;
      worker.postMessage({ type: 'error', name: error instanceof Error ? error.name : 'Error', message: error instanceof Error ? error.message : String(error) });
    } finally { gpu?.dispose(); }
  })();
};
