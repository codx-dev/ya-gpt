# Browser generation and training bridge

`WebGpuEngine` implements the existing `ya_gpt::engine::Engine` using synchronous
imports supplied by the TypeScript WebGPU host. Rust owns model parsing,
tokenization, block orchestration, and sampling. TypeScript owns GPU resources,
WGSL pipelines, command submission, and asynchronous readback. There are no
`wgpu`, `wasm-bindgen`, `web-sys`, or JavaScript binding-generation dependencies.

Inference and training use stepped browser APIs. The synchronous loss, finite
check, synchronization, and sampling methods still return errors because their
results require asynchronous GPU readback. Training calls the existing core
forward/backward operations and GPU AdamW kernels between readback phases, rather
than calling the core synchronous training loop. Core, CPU, and CUDA interfaces
are unchanged. WGSL uses FP32 with the core model's layouts, causal attention,
ReLU, and dropout behavior; results may differ slightly between devices.

## Build and tests

Install the `wasm32-unknown-unknown` target (`rustup target add
wasm32-unknown-unknown`) and Node 24 or later. From `frontend`, run `npm install`
and `npm run dev` or `npm run build`. Both build the real Rust WASM first through
`npm run wasm:build`, which executes:

```sh
cargo build -p ya-gpt-wasm --target wasm32-unknown-unknown --release
```

The script copies the artifact into the ignored `frontend/src/runtime/generated`
directory. Vite emits the WASM as an asset and resolves it relative to the hosting
base, including subdirectories. There is no checked-in placeholder or fallback
binary. Use HTTPS or localhost in a browser with WebGPU. Model inspection remains
available when compute initialization fails.

The Inspect tab calls the core `rich_header_from_bytes` parser through a separate
WASM instance, sharing the compiled module with generation. It requires WASM but
does not require WebGPU or a worker, and remains available during training. The
browser reads at most `4 * 64 * 1024 + 256` bytes from the model file; the parser
only decodes metadata, without loading tokenizer data or tensors. Each metadata
string is limited to 64 KiB of UTF-8 text. Inspection validates metadata, not the
complete model.

Inspection returns the same flat `RichModelHeader` JSON shape as the CLI, including
`version`, `config`, `vocab_size`, and `train_tokens`. The last field counts tokens
in the training data split, not tokens processed over all optimizer steps. Version
0 files display “Not recorded” for that count, while their JSON retains the core's
`train_tokens: 0` fallback. Version 1 displays its stored count, including zero.

```sh
cargo test -p ya-gpt-wasm
cargo run -p ya-gpt-wasm --example fixtures # refresh deterministic Naive fixtures
cargo run -p ya-gpt-wasm --example training_fixtures # refresh training references
cd frontend
npm test                 # builds WASM; includes its ABI/import checks
npm run typecheck
npm run test:browser     # requires Playwright Chromium; generation and training
npm run test:gpu         # shader comparisons, independent of the WASM build
```

GPU tests explicitly skip when no adapter exists. Set `YA_GPT_REQUIRE_GPU=1` to
make that an error. For the dedicated shader suite, `YA_GPT_SOFTWARE_GPU=1` enables
Chromium's SwiftShader WebGPU flags; `PLAYWRIGHT_CHROMIUM_EXECUTABLE` selects an
installed Chromium. Numerical fixtures compare irregular shapes, normalization,
embeddings, biased/unbiased linear layers, and a complete transformer block with
the Rust `Naive` engine. Training fixtures also cover backward kernels, repeated
tokens, stable cross entropy, AdamW, and non-finite detection. Normal browser tests
compare complete multi-layer training runs (including dropout) against Rust model
exports, and cover stopping, downloading, selecting, and generating with a result.

## Training

The Train tab uses the existing settings to initialize a new model from UTF-8
text, with the core character tokenizer and 90/10 train/validation split. A
dedicated module worker owns its WASM instance and WebGPU device. Only one compute
operation runs at a time; inspection remains available while training. Training
pipelines are compiled separately, so their initialization does not affect
generation. Loss evaluation matches the core optimizer's reporting cadence and
averages ten batches per split; the checkbox enables evaluation every step.

Stopping finishes an in-flight optimizer step, evaluates, and exports the model.
Stopping before any step starts produces no model. Final or stopped
models have separate Download and Use model actions; they remain in memory until
replaced or the page is closed/reloaded. Training failures preserve the previous
successful result and the selected model. There is no checkpoint/resume or
fine-tuning API.

After training, the Author field starts with the crate authors from the model's
header. Download trained model and Use model both serialize with that field's
current value. Nonempty text calls `to_bytes_with_author`; an empty string calls
`to_bytes` and restores the crate authors. Unicode and whitespace are preserved.
Author metadata is limited to 64 KiB of UTF-8, matching the browser inspector.
The original trained result stays intact; preparing an export uses a separate
WASM worker with `Naive` CPU buffers, without GPU allocations or retraining.
The latest prepared file is reused by both actions until its author or model
changes. Failed exports leave the trained result available for retry.

`runtime.serializeModel(file, author)` returns the new bytes. It uses the runtime's
existing busy guard and releases its worker on success and failure.

The runtime's `train(text, options, onProgress?)` returns a job with `stop()` and
`result: Promise<TrainingResult>`. Progress carries the phase, completed/total
iterations, elapsed milliseconds and available losses. The result contains model
bytes, completed/total iterations, effective seed, final losses and `stopped`.
The exported `.bin` uses the core serializer, including parameter gradients and
loss metadata; optimizer moments are not serialized. Weights and gradients are
read back in chunks of at most 8 MiB into a frozen export snapshot, which is the
only source accepted by `buffer_to_vec`.

## ABI version 1

The exported `memory` is unshared wasm32 linear memory. All pointers, lengths,
handles, and descriptor words are unsigned 32-bit integers, little-endian.
JavaScript normalizes imported i32 addresses with `>>> 0`. Floating-point values
in descriptors are carried as their f32 bit patterns. Host imports must consume
borrowed slices before returning and must never call WASM exports reentrantly.

Exports:

| Export | Contract |
| --- | --- |
| `version()` | Returns `1`. |
| `alloc(bytes)` | Returns a nonzero, four-byte-aligned pointer to initialized memory, or `0` on failure. |
| `dealloc(ptr)` | Frees an allocation by its base pointer. Unknown/double frees return `-1`. |
| `generation_begin(model_ptr, model_len, prompt_ptr, prompt_len, count, seed_lo, seed_hi)` | Borrows model bytes and UTF-8 prompt during the call. Creates one session with a positive token count. |
| `generation_forward(descriptor_ptr)` | Records one inference pass and writes three u32 words: logits handle, byte offset of final row, vocabulary length in elements. Keeps logits alive until accepted. |
| `generation_accept_logits(ptr, count)` | Borrows exactly one vocabulary row of f32 values, samples and appends a token. Returns `0` to continue or `1` when complete. |
| `generation_result_ptr()`, `generation_result_len()` | Borrowed UTF-8 result, valid until session end. Empty until generation completes. |
| `generation_end()` | Idempotently drops the session and GPU handles. |
| `error_ptr()`, `error_len()` | Borrowed UTF-8 error, valid until the next state-changing call. |

Status-returning exports return `0` on success unless stated otherwise and `-1`
on failure. Input pointers must be live allocation base addresses; lengths are
checked against allocation capacity. The caller frees its input allocations;
result/error pointers must not be freed. Recreate typed-array views after WASM
calls that can grow memory, and never preserve views across asynchronous waits.

Imports, all under `ya_gpu`:

| Import | Contract |
| --- | --- |
| `buffer_create(bytes)` | Allocates zero-initialized storage/copy memory; returns a nonzero handle or `0`. Empty Rust buffers use a four-byte physical allocation. |
| `buffer_upload(handle, ptr, bytes)` | Uploads a borrowed WASM byte slice; returns `0` or `-1`. |
| `buffer_release(handle)` | Removes the handle. Defers destruction while recorded/submitted commands reference it. |
| `dispatch(op, args_ptr, word_count)` | Records a kernel from the descriptor below; returns `0` or `-1`. |

The host captures synchronous import exceptions in `lastError` rather than
throwing through Rust. Async validation, memory, and device errors reject the
frontend operation. WASM traps invalidate the bridge and require a reload.

| Op | Descriptor words, in order |
| --- | --- |
| 1: fill | output, length, value bits |
| 2: linear | input, weight, bias (0 = absent), output, rows, inputs, outputs |
| 3: normalization | input, gamma, beta, output, rows, channels, epsilon bits |
| 4: embedding | tokens, token table, position table, output, element count, time, channels |
| 5: attention scores | qkv, scores, batch, time, channels, heads |
| 6: causal softmax | scores, batch, time, heads |
| 7: attention context | qkv, probabilities, output, batch, time, channels, heads |
| 8: ReLU | buffer, length |
| 9: residual addition | mutable branch, residual, length |
| 10: training | kind, invocation count, eight buffer handles (0 = unused), twelve parameter words |

Training adds exports without changing ABI v1 generation calls:

| Export | Contract |
| --- | --- |
| `training_version()` | Returns `1`. |
| `training_begin(text_ptr, text_len, options_ptr)` | Creates a session from UTF-8 text and 12 option words. |
| `training_advance(descriptor_ptr)` | Records the next phase and writes eight descriptor words. Rejects an unaccepted previous readback. |
| `training_accept(ptr, count)` | Accepts the requested FP32 readback and advances the session. |
| `training_stop()` | Requests stopping at a completed optimizer step. |
| `training_result_ptr()`, `training_result_len()` | Borrow the serialized model until session end. |
| `training_end()` | Idempotently releases the session. |

Option words: context length, embedding width, heads, layers, dropout bits, batch
size, iterations, learning-rate bits, weight-decay bits, per-step-loss flag,
seed low, seed high. Descriptor words: phase, GPU handle, byte offset, float count,
completed steps, total steps, train-loss bits, validation-loss bits. Phases are
1 gradients, 2 optimizer update, 3 train evaluation, 4 validation evaluation,
5 export readback, 6 complete, 7 stopped before any step. Phases 6 and 7 need no
readback. Unknown losses are represented by NaN until evaluation completes.
Training and generation sessions are mutually exclusive within each instance.

Model inspection adds four exports while retaining ABI v1:

| Export | Contract |
| --- | --- |
| `model_inspect(model_ptr, model_len)` | Borrows a metadata prefix from a live allocation, limited to `4 * 64 * 1024 + 256` bytes. Calls the core rich-header parser for versions 0 and 1 and stores UTF-8 JSON. Clears the previous result, including on failure. Does not access GPU imports or compute sessions. |
| `model_inspect_result_ptr()`, `model_inspect_result_len()` | Borrow the JSON until the next inspection or inspection cleanup. Return zero when there is no result. |
| `model_inspect_end()` | Idempotently releases the JSON result. The caller separately frees its input allocation. |

Model serialization adds four exports while retaining ABI v1:

| Export | Contract |
| --- | --- |
| `model_serialize(model_ptr, model_len, author_ptr, author_len)` | Borrows a model and UTF-8 author from live allocations. Requires no active compute session. Empty author uses the crate default; nonempty author overrides it. Clears any previous result, including on failure. |
| `model_serialize_result_ptr()`, `model_serialize_result_len()` | Borrow the output until the next serialization or cleanup. Both are zero when no result is available. |
| `model_serialize_end()` | Idempotently releases the output bytes. |

The caller copies the result, calls `model_serialize_end`, and frees both inputs.
These exports use the core parser and serializer, preserve tensor data and loss
metadata, and never call GPU imports. Older WASM builds without these exports
produce an explicit rebuild-and-reload error when preparing an author export.

Training kernel kinds are defined in `training_engine.rs` and `training.wgsl`:
1 copy, 2 multiply, 3 addition, 4 ReLU gradient, 5 cached normalization,
6 normalization input gradient, 7 normalization parameter gradients,
8 linear input gradient, 9 linear weight gradient, 10 bias gradient,
11 token embedding gradient, 12 position embedding gradient,
13 attention probability gradient, 14 softmax gradient, 15 QKV gradient,
16 cross entropy, 17 status sum, 18 finite scan, 19 AdamW, 20 loss/status summary.
The host supplies eight distinct storage buffers and a 64-byte uniform block
(kind, invocation count, twelve parameters, two padding words). Bindings B and C
use integer storage to preserve token-index bits, with float reinterpretation
for arithmetic. Reductions use one writer per result; no float atomics are needed.

Each dispatch gets immutable uniform storage until submission completes. One
encoder batches a token's forward pass and final-row copy into one submission.
TypeScript awaits staging-buffer mapping, copies only the vocabulary row into
WASM, then resumes sampling. Weights stay on the GPU for the request; workspace,
pipelines, and staging storage are reused. Per-request resources are reclaimed
on both success and failure, including resources left by a WASM trap.

Training uses separate gradient and optimizer submissions, each followed by
status readback and GPU error-scope checks. Non-finite values stop the run before
it can be exported. Dropout masks use the reference Rust RNG streams and causal
draw order. Kernels flatten a two-dimensional dispatch when one dimension is
insufficient; tensor allocations are checked against device and wasm32 limits.
The worker and its device are disposed after every run, including failures.
