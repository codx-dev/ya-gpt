import wasmUrl from './generated/ya_gpt_wasm.wasm?url';
import { inspectModel } from '../model/inspect';
import type { Backend, GenerateOptions, ModelMetadata, TrainOptions, TrainingJob, TrainingProgress } from '../model/types';
import { WasmBridge } from './bridge';
import type { WasmExports } from './bridge';
import { createGpuHost } from './webgpu';
import { startTrainingWorker } from './training-client';
import { trainingErrors } from './training-bridge';
import { serializeInWorker } from './serialization-client';
import { authorError } from './serialization-bridge';
import { checkInspector } from './inspection-bridge';
import type { InspectionExports } from './inspection-bridge';

export interface RuntimeCapabilities {
  version: number;
  inspect: boolean;
  generate: boolean;
  train: boolean;
  error?: string;
}

export interface ModelRuntime {
  initialize(): Promise<RuntimeCapabilities>;
  inspectModel(file: File): Promise<ModelMetadata>;
  generate(model: File, options: GenerateOptions): Promise<string>;
  train(text: string, options: TrainOptions, report?: (value: TrainingProgress) => void): TrainingJob;
  serializeModel(file: File, author: string): Promise<Uint8Array<ArrayBuffer>>;
  readonly busy: boolean;
  subscribe(listener: () => void): () => void;
}

let initialization: Promise<RuntimeCapabilities> | undefined;
let compiledModule: Promise<WebAssembly.Module> | undefined;
let inspection: Promise<InspectionExports> | undefined;

function loadModule(): Promise<WebAssembly.Module> {
  compiledModule ??= (async () => {
    const response = await fetch(wasmUrl);
    if (!response.ok) throw new Error(`WASM download failed: HTTP ${response.status}`);
    return WebAssembly.compile(await response.arrayBuffer());
  })();
  return compiledModule;
}

function loadInspector(): Promise<InspectionExports> {
  inspection ??= (async () => {
    const noGpu = () => { throw new Error('Model inspection must not access the GPU.'); };
    const instance = await WebAssembly.instantiate(await loadModule(), { ya_gpu: {
      buffer_create: noGpu, buffer_upload: noGpu, buffer_release: noGpu, dispatch: noGpu,
    } });
    const exports = instance.exports as unknown as InspectionExports;
    checkInspector(exports);
    return exports;
  })();
  return inspection;
}

let bridge: WasmBridge | undefined;
let busy = false;
const listeners = new Set<() => void>();
function setBusy(value: boolean) { busy = value; for (const listener of listeners) listener(); }
export const runtime: ModelRuntime = {
  get busy() { return busy; },
  subscribe(listener) { listeners.add(listener); return () => { listeners.delete(listener); }; },
  initialize() {
    initialization ??= (async () => {
      let inspect = false;
      try {
        await loadInspector();
        inspect = true;
        const module = await loadModule();
        const gpu = await createGpuHost();
        let train = false;
        try {
          const instance = await WebAssembly.instantiate(module, { ya_gpu: gpu.imports });
          bridge = new WasmBridge(instance.exports as WasmExports, gpu);
          train = typeof Worker !== 'undefined' && typeof instance.exports.training_version === 'function' && instance.exports.training_version() === 1;
        } catch (error) { gpu.dispose(); throw error; }
        return { version: 1, inspect: true, generate: true, train };
      } catch (error) {
        return { version: 1, inspect, generate: false, train: false, error: error instanceof Error ? error.message : String(error) };
      }
    })();
    return initialization;
  },
  async inspectModel(file) {
    return inspectModel(file, await loadInspector());
  },
  async serializeModel(file, author) {
    if (busy) throw new Error('Another compute operation is already running.');
    if (!file.size || file.size > 0x7fff_ffff) throw new Error('Invalid model file size.');
    const error = authorError(author);
    if (error) throw new Error(error);
    if (typeof Worker === 'undefined') throw new Error('Web workers are unavailable in this browser.');
    setBusy(true);
    try { return await serializeInWorker(file, author); }
    finally { setBusy(false); }
  },
  async generate(model, options) {
    if (busy) throw new Error('Another compute operation is already running.');
    setBusy(true);
    try {
      const capabilities = await this.initialize();
      if (!bridge || !capabilities.generate) throw new Error(capabilities.error || 'WebGPU generation is unavailable.');
      if (!model.size || model.size > 0x7fff_ffff) throw new Error('Invalid model file size.');
      return await bridge.generate(new Uint8Array(await model.arrayBuffer()), options);
    } finally { setBusy(false); }
  },
  train(text, options, report) {
    if (busy) throw new Error('Another compute operation is already running.');
    const errors = trainingErrors(options, text);
    if (errors.length) throw new Error(errors.join(' '));
    if (typeof Worker === 'undefined') throw new Error('Web workers are unavailable in this browser.');
    setBusy(true);
    try {
      const job = startTrainingWorker(text, options, report);
      return { stop: job.stop, result: job.result.finally(() => setBusy(false)) };
    } catch (error) { setBusy(false); throw error; }
  },
};

export type BackendSupport = Record<Backend, boolean>;

export async function detectBackends(): Promise<BackendSupport> {
  const capabilities = await runtime.initialize();
  return { webgpu: capabilities.generate, simd: false };
}
