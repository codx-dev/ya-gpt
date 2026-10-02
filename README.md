# Yet Another GPT implementation

Yet another GPT implementation in pure Rust, with minimal dependencies and no ML frameworks.

The main goal is to experiment with and compare the performance of different engine implementations. The model uses only self-attention; cross-attention is intentionally omitted to keep the implementation as simple as possible.

Supports training and text generation through a naive CPU backend. CUDA and SIMD backends are planned.

This is *NOT* intended for production use.

```sh
cargo run --release -- train --preset tiny --input data/input.txt --output out/model-tiny
cargo run --release -- generate --input out/model-tiny --num-tokens 128 "Hello"
```

Below is a flamegraph for preset small with 100 interactions, using the Naive backen:

![Flamegraph](./images/flamegraph-small-100.svg)
