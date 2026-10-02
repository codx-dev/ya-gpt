# Yet Another GPT implementation

Yet another GPT implementation in pure Rust, with minimal dependencies and no ML frameworks.

The main goal is to experiment with and compare the performance of different engine implementations. The model uses only self-attention; cross-attention is intentionally omitted to keep the implementation as simple as possible.

Supports training and text generation through a naive CPU or SIMD backend. CUDA backend is planned.

This is *NOT* intended for production use.

The model on `./assets/model-simd-wide-medium-5000.bin` was trained with the following command:

```
# AMD EPYC 7402P 24-Core Processor
$ time just train-simd-wide medium 5000
RUSTFLAGS="-C target-cpu=native" cargo run --release -p ya-gpt-cli --features simd-wide -- train --engine simd-wide --preset medium --iterations 5000 --input data/input.txt --output ./out/model-simd-wide.bin
..
step 5000: train loss 1.4411, val loss 1.6158
just train-simd-wide medium 5000  7345.51s user 0.48s system 99% cpu 2:03:02.69 total
```

A model will start to perform reasonably with ~1.88 loss (cross-entropy)

```sh
$ cargo run -- inspect --input ./assets/model-simd-wide-medium-5000.bin | jq
{
  "name": "ya-gpt",
  "authors": "Victor Lopez <vhrlopes@gmail.com>",
  "major": "0",
  "fp": "f32",
  "params": 825154
}
```

Here is the output of this medium performance model:

```sh
$ just generate-simd-wide 200 ./assets/model-simd-wide-medium-5000.bin 
ROMEO:
So you dispity have the soul
worthlenous to of Richard: I say be Richard:
O, would by him say it not fortune hath long.
Thoral'd this being himself are day to quarrels;
To hear a fair harm that depae
```

Below is a flamegraph for preset small with 100 interactions, using the Naive backend:

![Flamegraph](./images/flamegraph-small-100.svg)
