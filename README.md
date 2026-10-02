# Yet Another GPT implementation

Yet another GPT implementation in pure Rust, with minimal dependencies and no ML frameworks.

The main goal is to experiment with and compare the performance of different engine implementations. The model uses only self-attention; cross-attention is intentionally omitted to keep the implementation as simple as possible.

Supports training and text generation through a naive CPU backend. CUDA and SIMD backends are planned.

This is *NOT* intended for production use.

```sh
# AMD EPYC 7402P 24-Core Processor
$ time just train-simd-wide medium 500
..
step 500: train loss 2.3079, val loss 2.3360
just train-simd-wide medium 500  703.11s user 0.30s system 99% cpu 11:46.02 total

$ cargo run -- inspect --input ./out/model-simd-wide.bin
{"name":"ya-gpt","authors":"Victor Lopez <vhrlopes@gmail.com>","major":"0","fp":"f32","params":825154}

# a ~1.88 loss would yield some decent output, but requires more training
$ time just generate-simd-wide
ROMEO:
Arf my; yoou youlls e thige I nco to mpno the,
The sy n he asande nd is hes, by arist tondss,
Me kerer naded, d
..
just generate-simd-wide  16.26s user 0.17s system 99% cpu 16.493 total
```

Below is a flamegraph for preset small with 100 interactions, using the Naive backend:

![Flamegraph](./images/flamegraph-small-100.svg)
