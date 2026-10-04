import fill from './shaders/fill.wgsl?raw';
import linear from './shaders/linear.wgsl?raw';
import norm from './shaders/norm.wgsl?raw';
import embedding from './shaders/embedding.wgsl?raw';
import scores from './shaders/scores.wgsl?raw';
import softmax from './shaders/softmax.wgsl?raw';
import context from './shaders/context.wgsl?raw';
import relu from './shaders/relu.wgsl?raw';
import add from './shaders/add.wgsl?raw';
import { GpuHost } from './gpu-host';
import type { Gpu, GpuPipeline } from './webgpu-types';

export async function createGpuHost(training = false): Promise<GpuHost> {
  const gpu = (navigator as Navigator & { gpu?: Gpu }).gpu;
  if (!gpu) throw new Error('WebGPU is unavailable. Use a supported browser on HTTPS or localhost.');
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter is available.');
  const device = await adapter.requestDevice();
  try {
    const pipelines = new Map<number, GpuPipeline>();
    const sources = [fill, linear, norm, embedding, scores, softmax, context, relu, add];
    if (training) sources.push((await import('./shaders/training.wgsl?raw')).default);
    for (const [index, code] of sources.entries()) {
      const label = `ya-gpt kernel ${index + 1}`;
      const module = device.createShaderModule({ label, code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`${label}: ${errors.map((m) => `line ${m.lineNum}: ${m.message}`).join('; ')}`);
      pipelines.set(index + 1, await device.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint: 'main' } }));
    }
    return new GpuHost(device, pipelines);
  } catch (error) { device.destroy(); throw error; }
}
