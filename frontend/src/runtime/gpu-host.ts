import type { GpuBuffer, GpuDevice, GpuEncoder, GpuPipeline } from './webgpu-types';

const READ = 1, COPY_SRC = 4, COPY_DST = 8, UNIFORM = 64, STORAGE = 128;
const argumentCounts = [0, 3, 7, 7, 7, 6, 4, 7, 2, 3, 22];

function elements(...dimensions: number[]): number {
  let count = 1;
  for (const n of dimensions) {
    if (!Number.isInteger(n) || n <= 0) throw new Error('GPU dimensions must be positive integers.');
    count *= n;
    if (count > 0x1fff_ffff) throw new Error('GPU tensor shape overflow.');
  }
  return count;
}

export class GpuHost {
  readonly device: GpuDevice;
  private readonly pipelines: Map<number, GpuPipeline>;
  memory?: WebAssembly.Memory;
  lastError?: Error;
  private failure?: Error;
  private readonly buffers = new Map<number, GpuBuffer>();
  private readonly retired = new Set<GpuBuffer>();
  private nextHandle = 1;
  private encoder?: GpuEncoder;
  private staging?: GpuBuffer;
  private readonly lost: Promise<never>;
  private scopeOpen = false;
  private readonly dummy: GpuBuffer;
  private readonly trainingDummies: GpuBuffer[] = [];

  constructor(device: GpuDevice, pipelines: Map<number, GpuPipeline>) {
    this.device = device; this.pipelines = pipelines;
    this.dummy = device.createBuffer({ size: 4, usage: STORAGE });
    this.lost = device.lost.then((info) => {
      this.failure = new Error(`WebGPU device lost: ${info.message || 'device unavailable'}`);
      throw this.failure;
    });
    void this.lost.catch(() => {});
    device.addEventListener('uncapturederror', (event) => {
      this.failure = new Error(`WebGPU: ${event.error.message}`);
    });
  }

  get bufferCount(): number { return this.buffers.size; }

  assertReady() {
    if (this.failure) throw this.failure;
  }

  private safe(fallback: number, action: () => number): number {
    try { this.assertReady(); return action(); }
    catch (error) { this.lastError = error instanceof Error ? error : new Error(String(error)); return fallback; }
  }

  private bytes(ptr: number, length: number): Uint8Array<ArrayBuffer> {
    const memory = this.memory?.buffer;
    // i32 arguments arrive as signed JS numbers, addresses are unsigned.
    ptr >>>= 0; length >>>= 0;
    if (!memory || ptr + length > memory.byteLength) throw new Error('GPU import references invalid WASM memory.');
    return new Uint8Array(memory as ArrayBuffer, ptr, length);
  }

  readonly imports = {
    buffer_create: (bytes: number): number => this.safe(0, () => this.create(bytes >>> 0)),
    buffer_upload: (handle: number, ptr: number, bytes: number): number => this.safe(-1, () => {
      this.upload(handle >>> 0, this.bytes(ptr, bytes)); return 0;
    }),
    buffer_release: (handle: number): void => { this.safe(-1, () => { this.release(handle >>> 0); return 0; }); },
    dispatch: (op: number, ptr: number, length: number): number => this.safe(-1, () => {
      if (length !== argumentCounts[op]) throw new Error('Invalid GPU descriptor length.');
      const bytes = this.bytes(ptr, length * 4);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const args = Array.from({ length }, (_, i) => view.getUint32(i * 4, true));
      this.dispatch(op, args); return 0;
    }),
  };

  create(bytes: number): number {
    this.assertReady();
    if (!Number.isInteger(bytes) || bytes < 4 || bytes % 4 || bytes > Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize)) {
      throw new Error('Tensor exceeds WebGPU buffer limits.');
    }
    if (this.nextHandle > 0xffff_ffff) throw new Error('GPU handles exhausted. Reload the runtime.');
    const buffer = this.device.createBuffer({ size: bytes, usage: STORAGE | COPY_SRC | COPY_DST });
    const handle = this.nextHandle++;
    this.buffers.set(handle, buffer);
    return handle;
  }

  private buffer(handle: number, words?: number): GpuBuffer {
    const buffer = this.buffers.get(handle);
    if (!buffer) throw new Error(`Invalid GPU buffer handle ${handle}.`);
    if (words !== undefined && buffer.size !== words * 4) throw new Error('GPU buffer shape mismatch.');
    return buffer;
  }

  upload(handle: number, data: ArrayBufferView) {
    this.assertReady();
    const buffer = this.buffer(handle);
    if (data.byteLength > buffer.size || data.byteLength % 4) throw new Error('Invalid GPU upload size.');
    this.device.queue.writeBuffer(buffer, 0, data);
  }

  release(handle: number) {
    const buffer = this.buffer(handle);
    this.buffers.delete(handle);
    this.retired.add(buffer);
  }

  dispatch(op: number, a: number[]) {
    this.assertReady();
    if (a.length !== argumentCounts[op]) throw new Error('Invalid GPU dispatch descriptor.');
    const pipeline = this.pipelines.get(op);
    if (!pipeline) throw new Error(`Unknown GPU kernel ${op}.`);
    let buffers: GpuBuffer[], params: number[], count: number;
    switch (op) {
      case 1: // fill
        count = elements(a[1]); buffers = [this.buffer(a[0], count)]; params = [count, a[2]]; break;
      case 2: { // linear
        const [input, weight, bias, output, rows, inputs, outputs] = a;
        count = elements(rows, outputs);
        buffers = [this.buffer(input, elements(rows, inputs)), this.buffer(weight, elements(inputs, outputs)), bias ? this.buffer(bias, outputs) : this.dummy, this.buffer(output, count)];
        params = [rows, inputs, outputs, bias ? 1 : 0]; break;
      }
      case 3: { // normalization
        const [input, gamma, beta, output, rows, channels, epsilon] = a;
        const n = elements(rows, channels);
        const e = new Float32Array(new Uint32Array([epsilon]).buffer)[0];
        if (!Number.isFinite(e) || e <= 0) throw new Error('Invalid normalization epsilon.');
        buffers = [this.buffer(input, n), this.buffer(gamma, channels), this.buffer(beta, channels), this.buffer(output, n)];
        count = rows; params = [rows, channels, epsilon]; break;
      }
      case 4: { // embeddings (Rust checks token bounds and table dimensions)
        const [tokens, table, positions, output, n, time, channels] = a;
        elements(n); elements(time, channels);
        if (n % channels || (n / channels) % time) throw new Error('Invalid embedding shape.');
        buffers = [this.buffer(tokens, n / channels), this.buffer(table), this.buffer(positions), this.buffer(output, n)];
        count = n; params = [n, time, channels]; break;
      }
      case 5: case 7: { // attention scores / context
        const context = op === 7;
        const base = context ? 3 : 2;
        const [batch, time, channels, heads] = a.slice(base);
        const n = elements(batch, time, channels), scores = elements(batch, heads, time, time);
        if (channels % heads) throw new Error('Invalid attention heads.');
        buffers = [this.buffer(a[0], 3 * n), this.buffer(a[1], scores)];
        if (context) buffers.push(this.buffer(a[2], n));
        count = context ? n : scores; params = [batch, time, channels, heads]; break;
      }
      case 6: { // causal softmax, one invocation per query
        const [scores, batch, time, heads] = a;
        count = elements(batch, time, heads);
        buffers = [this.buffer(scores, elements(batch, heads, time, time))]; params = [batch, time, heads]; break;
      }
      case 8: // ReLU
        count = elements(a[1]); buffers = [this.buffer(a[0], count)]; params = [count]; break;
      case 9: // residual addition
        count = elements(a[2]); buffers = [this.buffer(a[0], count), this.buffer(a[1], count)]; params = [count]; break;
      case 10: { // private training ABI: kind, count, eight handles, twelve parameters
        if (!Number.isInteger(a[0]) || a[0] < 1 || a[0] > 20) throw new Error('Unknown training operation.');
        count = elements(a[1]);
        const handles = a.slice(2, 10);
        const live = handles.filter(Boolean);
        if (new Set(live).size !== live.length) throw new Error('Training storage bindings must not alias.');
        while (this.trainingDummies.length < 8) this.trainingDummies.push(this.device.createBuffer({ size: 4, usage: STORAGE }));
        buffers = handles.map((handle, i) => handle ? this.buffer(handle) : this.trainingDummies[i]);
        params = [a[0], count, ...a.slice(10)]; break;
      }
      default: throw new Error('Unknown GPU operation.');
    }
    const groups = Math.ceil(count / 64);
    const groupsX = Math.min(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
    const groupsY = Math.ceil(groups / groupsX);
    if (groupsY > this.device.limits.maxComputeWorkgroupsPerDimension) throw new Error('Tensor exceeds WebGPU dispatch limits. Reduce batch size or context length.');
    // Every dispatch owns its uniform values until submission completes.
    const uniformSize = op === 10 ? 64 : 16;
    const uniform = this.device.createBuffer({ size: uniformSize, usage: UNIFORM | COPY_DST });
    this.retired.add(uniform);
    const words = new Uint32Array(uniformSize / 4); words.set(params);
    this.device.queue.writeBuffer(uniform, 0, words);
    const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...buffers, uniform].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.encoder ??= this.device.createCommandEncoder();
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(groupsX, groupsY); pass.end();
  }

  begin() {
    this.assertReady();
    if (this.scopeOpen) throw new Error('GPU operation already active.');
    this.lastError = undefined;
    for (const filter of ['internal', 'out-of-memory', 'validation'] as const) this.device.pushErrorScope(filter);
    this.scopeOpen = true;
  }

  async readback(handle: number, offset: number, count: number): Promise<Float32Array<ArrayBuffer>> {
    this.assertReady();
    const buffer = this.buffer(handle), bytes = elements(count) * 4;
    if (!Number.isInteger(offset) || offset < 0 || offset % 4 || offset + bytes > buffer.size) throw new Error('Invalid GPU readback range.');
    if (!this.staging || this.staging.size !== bytes) {
      this.staging?.destroy();
      this.staging = this.device.createBuffer({ size: bytes, usage: READ | COPY_DST });
    }
    this.encoder ??= this.device.createCommandEncoder();
    this.encoder.copyBufferToBuffer(buffer, offset, this.staging, 0, bytes);
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = undefined;
    const staging = this.staging;
    let completed = false;
    try {
      await Promise.race([staging.mapAsync(READ, 0, bytes), this.lost]);
      completed = true;
      this.assertReady();
      return new Float32Array(staging.getMappedRange(0, bytes).slice(0));
    } finally {
      staging.unmap();
      // All prior commands precede the staging copy on this queue.
      if (completed) this.collect();
    }
  }

  private collect() { for (const buffer of this.retired) buffer.destroy(); this.retired.clear(); }

  async checkpoint() {
    this.assertReady();
    if (!this.scopeOpen) throw new Error('No active GPU operation.');
    const errors = await Promise.all([this.device.popErrorScope(), this.device.popErrorScope(), this.device.popErrorScope()]);
    this.scopeOpen = false;
    const error = errors.find((e) => e);
    if (error) throw new Error(`WebGPU: ${error.message}`);
    this.begin();
  }

  async finish() {
    // Discard an incomplete encoder on failure. Rust has already released its
    // buffers; wait for previous submissions before destroying those resources.
    this.encoder = undefined;
    let error: unknown;
    try { await Promise.race([this.device.queue.onSubmittedWorkDone(), this.lost]); }
    catch (cause) { error = cause; }
    // Normally Rust has released every handle. Also reclaim allocations left
    // behind if WASM trapped before its destructors could execute.
    for (const buffer of this.buffers.values()) this.retired.add(buffer);
    this.buffers.clear();
    this.collect();
    if (this.scopeOpen) {
      this.scopeOpen = false;
      const errors = await Promise.all([this.device.popErrorScope(), this.device.popErrorScope(), this.device.popErrorScope()]);
      error ??= errors.find((e) => e)?.message;
    }
    if (error) throw error instanceof Error ? error : new Error(String(error));
    this.assertReady();
  }

  dispose() {
    this.encoder = undefined;
    for (const buffer of this.buffers.values()) buffer.destroy();
    this.buffers.clear(); this.collect(); this.staging?.destroy(); this.dummy.destroy();
    for (const buffer of this.trainingDummies) buffer.destroy();
    this.device.destroy();
  }
}
