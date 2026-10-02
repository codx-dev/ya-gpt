use rand::{SeedableRng, rngs::StdRng};
use ya_gpt::{
    config::ModelConfig,
    engine::{naive::Naive, *},
    model::{Block, gpt::Gpt, optimizer::AdamW},
    tokenizer::Tokenizer,
};

use crate::*;

fn gpu() -> CudaEngine {
    CudaEngine::new(0).expect("CUDA tests require an Ada-or-newer GPU, driver, and cuBLAS")
}

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 % 19) as f32 - 9.0) / 7.0).collect()
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: {a} != {e}, tolerance {tolerance}"
        );
    }
}

fn download(en: &CudaEngine, buffer: &CudaBuffer) -> Vec<f32> {
    en.buffer_to_vec(buffer).unwrap()
}

fn assert_same_model<A: Engine, B: Engine>(a_en: &A, a: &Gpt<A>, b_en: &B, b: &Gpt<B>) {
    assert_eq!(a.config, b.config);
    assert_eq!(a.vocab_size, b.vocab_size);
    assert_eq!(a.tokenizer, b.tokenizer);
    assert_eq!(a.blocks.len(), b.blocks.len());
    let a_parameters = a.parameters();
    let b_parameters = b.parameters();
    assert_eq!(a_parameters.len(), b_parameters.len());
    for (index, (a, b)) in a_parameters.iter().zip(&b_parameters).enumerate() {
        for (name, a, b) in [
            ("values", &a.values, &b.values),
            ("gradients", &a.gradients, &b.gradients),
        ] {
            let a: Vec<_> = a_en
                .buffer_to_vec(a)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let b: Vec<_> = b_en
                .buffer_to_vec(b)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(
                a, b,
                "parameter {index} {name} changed during serialization"
            );
        }
    }
}

#[test]
fn model_serialization_round_trip_preserves_contents() {
    let mut model = Gpt::new(
        &Naive,
        ModelConfig::micro(),
        Tokenizer::new("abc"),
        &mut rng(),
    )
    .unwrap();
    // Include nonzero gradients so the check covers both parts of each parameter.
    for p in model.parameters_mut() {
        p.gradients = values(p.gradients.len());
    }
    let mut bytes = model.to_bytes(&Naive).unwrap();
    for _ in 0..8 {
        let restored = Gpt::try_from_bytes(&Naive, &bytes).unwrap();
        assert_same_model(&Naive, &model, &Naive, &restored);
        // HashMap iteration order can change on each decode, so serialization
        // promises identical contents rather than canonical byte ordering.
        bytes = restored.to_bytes(&Naive).unwrap();
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn buffers_round_trip_empty_storage_and_finiteness() {
    let en = gpu();
    for n in [0, 1, 17, 255, 256, 257, 65537] {
        assert_eq!(download(&en, &en.zeroes(n).unwrap()), vec![0.0; n]);
        assert_eq!(download(&en, &en.filled(-2.5, n).unwrap()), vec![-2.5; n]);
        let mut input = values(n);
        let buffer = en.buffer_from_slice(&input).unwrap();
        input.fill(10.0);
        assert_eq!(en.buffer_len(&buffer), n);
        assert_eq!(download(&en, &buffer), values(n));
        assert!(en.buffers_all_finite(&[&buffer]).unwrap());
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = vec![0.0; 257];
        input[256] = invalid;
        assert!(
            !en.buffers_all_finite(&[&en.buffer_from_slice(&input).unwrap()])
                .unwrap()
        );
    }
    assert!(en.buffers_all_finite(&[]).unwrap());
    assert_eq!(en.indices_from_slice(&[]).unwrap().len, 0);
    assert!(en.indices_from_slice(&[usize::MAX]).is_err());
    en.synchronize().unwrap();
}

#[test]
#[ignore = "requires CUDA hardware"]
fn linear_matches_naive_with_bias_and_overwrites_gradients() {
    let en = gpu();
    for (rows, inputs, outputs) in [(1, 1, 1), (3, 7, 9), (2, 17, 5), (5, 32, 16), (2, 257, 3)] {
        let spec = LinearSpec {
            rows,
            inputs,
            outputs,
        };
        let (x, w, b, dy) = (
            values(rows * inputs),
            values(inputs * outputs),
            values(outputs),
            values(rows * outputs),
        );
        let (gx, gw, gb, gdy) = (
            en.buffer_from_slice(&x).unwrap(),
            en.buffer_from_slice(&w).unwrap(),
            en.buffer_from_slice(&b).unwrap(),
            en.buffer_from_slice(&dy).unwrap(),
        );
        for bias in [false, true] {
            let mut expected = vec![f32::NAN; dy.len()];
            let mut actual = en.filled(f32::NAN, dy.len()).unwrap();
            Naive
                .linear_forward(
                    spec,
                    LinearWeights {
                        weight: &w,
                        bias: bias.then_some(&b),
                    },
                    &x,
                    &mut expected,
                )
                .unwrap();
            en.linear_forward(
                spec,
                LinearWeights {
                    weight: &gw,
                    bias: bias.then_some(&gb),
                },
                &gx,
                &mut actual,
            )
            .unwrap();
            close(&download(&en, &actual), &expected, 3e-5);
            let (mut dx, mut dw, mut db) = (
                vec![f32::NAN; x.len()],
                vec![f32::NAN; w.len()],
                vec![f32::NAN; b.len()],
            );
            Naive
                .linear_backward(
                    spec,
                    LinearWeights {
                        weight: &w,
                        bias: bias.then_some(&b),
                    },
                    LinearGradients {
                        weight: &mut dw,
                        bias: bias.then_some(&mut db),
                    },
                    &x,
                    &dy,
                    &mut dx,
                )
                .unwrap();
            let (mut gdx, mut gdw, mut gdb) = (
                en.filled(f32::NAN, dx.len()).unwrap(),
                en.filled(f32::NAN, dw.len()).unwrap(),
                en.filled(f32::NAN, db.len()).unwrap(),
            );
            for _ in 0..2 {
                en.linear_backward(
                    spec,
                    LinearWeights {
                        weight: &gw,
                        bias: bias.then_some(&gb),
                    },
                    LinearGradients {
                        weight: &mut gdw,
                        bias: bias.then_some(&mut gdb),
                    },
                    &gx,
                    &gdy,
                    &mut gdx,
                )
                .unwrap();
                close(&download(&en, &gdx), &dx, 3e-5);
                close(&download(&en, &gdw), &dw, 3e-5);
                if bias {
                    close(&download(&en, &gdb), &db, 3e-5);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_matches_naive_and_reuses_cache() {
    let en = gpu();
    let mut cache = CudaLayerNormCache::default();
    for (rows, channels) in [(3, 17), (1, 1), (2, 7), (3, 257)] {
        let spec = NormSpec {
            rows,
            channels,
            epsilon: 1e-5,
        };
        let (x, gamma, beta, dy) = (
            values(rows * channels),
            values(channels),
            vec![0.3; channels],
            values(rows * channels),
        );
        let (gx, gg, gb, gdy) = (
            en.buffer_from_slice(&x).unwrap(),
            en.buffer_from_slice(&gamma).unwrap(),
            en.buffer_from_slice(&beta).unwrap(),
            en.buffer_from_slice(&dy).unwrap(),
        );
        let mut nc = <Naive as Engine>::LayerNormCache::default();
        let mut expected = vec![f32::NAN; x.len()];
        let mut actual = en.filled(f32::NAN, x.len()).unwrap();
        Naive
            .layer_norm_forward(
                spec,
                NormWeights {
                    gamma: &gamma,
                    beta: &beta,
                },
                &x,
                &mut expected,
                Some(&mut nc),
            )
            .unwrap();
        en.layer_norm_forward(
            spec,
            NormWeights {
                gamma: &gg,
                beta: &gb,
            },
            &gx,
            &mut actual,
            Some(&mut cache),
        )
        .unwrap();
        close(&download(&en, &actual), &expected, 3e-5);
        let mut uncached = en.zeroes(x.len()).unwrap();
        en.layer_norm_forward(
            spec,
            NormWeights {
                gamma: &gg,
                beta: &gb,
            },
            &gx,
            &mut uncached,
            None,
        )
        .unwrap();
        assert_eq!(download(&en, &actual), download(&en, &uncached));
        let (mut dx, mut dg, mut db) = (
            vec![f32::NAN; x.len()],
            vec![f32::NAN; channels],
            vec![f32::NAN; channels],
        );
        Naive
            .layer_norm_backward(
                spec,
                NormWeights {
                    gamma: &gamma,
                    beta: &beta,
                },
                NormGradients {
                    gamma: &mut dg,
                    beta: &mut db,
                },
                &nc,
                &dy,
                &mut dx,
            )
            .unwrap();
        let (mut gdx, mut gdg, mut gdb) = (
            en.filled(f32::NAN, dx.len()).unwrap(),
            en.filled(f32::NAN, dg.len()).unwrap(),
            en.filled(f32::NAN, db.len()).unwrap(),
        );
        for _ in 0..2 {
            en.layer_norm_backward(
                spec,
                NormWeights {
                    gamma: &gg,
                    beta: &gb,
                },
                NormGradients {
                    gamma: &mut gdg,
                    beta: &mut gdb,
                },
                &cache,
                &gdy,
                &mut gdx,
            )
            .unwrap();
            close(&download(&en, &gdx), &dx, 5e-5);
            close(&download(&en, &gdg), &dg, 5e-5);
            close(&download(&en, &gdb), &db, 5e-5);
        }
        let wrong = NormSpec {
            epsilon: 1e-3,
            ..spec
        };
        assert!(
            en.layer_norm_backward(
                wrong,
                NormWeights {
                    gamma: &gg,
                    beta: &gb
                },
                NormGradients {
                    gamma: &mut gdg,
                    beta: &mut gdb
                },
                &cache,
                &gdy,
                &mut gdx
            )
            .is_err()
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn embeddings_accumulate_repeated_tokens_and_clear_unused_rows() {
    let en = gpu();
    let spec = EmbeddingSpec {
        batch: 2,
        time: 3,
        channels: 7,
        vocab: 5,
        positions: 4,
    };
    let tokens = vec![1, 1, 3, 1, 0, 3];
    let gt = en.indices_from_slice(&tokens).unwrap();
    let table = values(spec.vocab * spec.channels);
    let positions = values(spec.positions * spec.channels);
    let dy = values(tokens.len() * spec.channels);
    let mut expected = vec![f32::NAN; dy.len()];
    let mut actual = en.filled(f32::NAN, dy.len()).unwrap();
    Naive
        .token_position_embedding_forward(spec, &tokens, &table, &positions, &mut expected)
        .unwrap();
    en.token_position_embedding_forward(
        spec,
        &gt,
        &en.buffer_from_slice(&table).unwrap(),
        &en.buffer_from_slice(&positions).unwrap(),
        &mut actual,
    )
    .unwrap();
    close(&download(&en, &actual), &expected, 1e-6);
    let (mut dt, mut dp) = (vec![f32::NAN; table.len()], vec![f32::NAN; positions.len()]);
    Naive
        .token_position_embedding_backward(spec, &tokens, &dy, &mut dt, &mut dp)
        .unwrap();
    let (mut gdt, mut gdp) = (
        en.filled(f32::NAN, dt.len()).unwrap(),
        en.filled(f32::NAN, dp.len()).unwrap(),
    );
    for _ in 0..2 {
        en.token_position_embedding_backward(
            spec,
            &gt,
            &en.buffer_from_slice(&dy).unwrap(),
            &mut gdt,
            &mut gdp,
        )
        .unwrap();
        close(&download(&en, &gdt), &dt, 2e-6);
        close(&download(&en, &gdp), &dp, 2e-6);
    }
    let invalid = en.indices_from_slice(&[5, 1, 3, 1, 0, 3]).unwrap();
    assert!(
        en.token_position_embedding_backward(spec, &invalid, &actual, &mut gdt, &mut gdp)
            .is_err()
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn loss_sampling_and_non_finite_errors() {
    let en = gpu();
    for (rows, vocab) in [(1, 1), (3, 7), (2, 257)] {
        let logits: Vec<_> = values(rows * vocab)
            .into_iter()
            .map(|v| v * 30.0 + 1000.0)
            .collect();
        let targets: Vec<_> = (0..rows).map(|i| i % vocab).collect();
        let mut gradient = vec![f32::NAN; logits.len()];
        let expected = Naive
            .cross_entropy_with_grad(&logits, &targets, rows, vocab, &mut gradient)
            .unwrap();
        let gl = en.buffer_from_slice(&logits).unwrap();
        let gt = en.indices_from_slice(&targets).unwrap();
        let mut gg = en.filled(f32::NAN, logits.len()).unwrap();
        let loss = en
            .cross_entropy_with_grad(&gl, &gt, rows, vocab, &mut gg)
            .unwrap();
        close(&[loss], &[expected], 3e-5);
        close(&download(&en, &gg), &gradient, 3e-5);
        close(
            &[en.cross_entropy(&gl, &gt, rows, vocab).unwrap()],
            &[loss],
            1e-6,
        );
    }
    let logits = vec![f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0];
    let gl = en.buffer_from_slice(&logits).unwrap();
    for uniform in [0.0, 0.1, 0.5, 0.9, 1.0] {
        assert_eq!(
            en.sample_last_token(&gl, 2, 3, uniform).unwrap(),
            Naive.sample_last_token(&logits, 2, 3, uniform).unwrap()
        );
    }
    let underflow = en.buffer_from_slice(&[-1000.0, 0.0, -1000.0]).unwrap();
    assert_eq!(en.sample_last_token(&underflow, 1, 3, 1.0).unwrap(), 1);
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bad = en.buffer_from_slice(&[0.0, invalid, 1.0]).unwrap();
        assert!(
            en.cross_entropy(&bad, &en.indices_from_slice(&[0]).unwrap(), 1, 3)
                .is_err()
        );
        assert!(en.sample_last_token(&bad, 1, 3, 0.5).is_err());
    }
    for uniform in [f32::NAN, -0.1, 1.1] {
        assert!(en.sample_last_token(&underflow, 1, 3, uniform).is_err());
    }
    assert!(
        en.cross_entropy(&underflow, &en.indices_from_slice(&[3]).unwrap(), 1, 3)
            .is_err()
    );
    assert!(en.sample_last_token(&underflow, 0, 3, 0.5).is_err());
    assert!(en.sample_last_token(&underflow, 1, 2, 0.5).is_err());
}

fn block<E: Engine>(en: &E, channels: usize, heads: usize) -> Block<E> {
    let config = ModelConfig {
        n_embd: channels,
        n_head: heads,
        ..ModelConfig::micro()
    };
    let mut block = Block::new(en, &config, &mut rng()).unwrap();
    for p in block.parameters_mut() {
        p.values = en
            .buffer_from_slice(
                &values(en.buffer_len(&p.values))
                    .iter()
                    .map(|v| v * 0.08)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    for norm in [&mut block.norm1, &mut block.norm2] {
        norm.gamma.values = en.filled(0.9, channels).unwrap();
    }
    block.expand.bias.as_mut().unwrap().values = en
        .buffer_from_slice(
            &(0..4 * channels)
                .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    block
}

#[test]
#[ignore = "requires CUDA hardware"]
fn blocks_match_naive_and_saved_cache_survives_workspace_reuse() {
    let en = gpu();
    let mut ws = CudaWorkspace::default();
    let mut cache = CudaBlockCache::default();
    let mut nws = <Naive as Engine>::Workspace::default();
    for (batch, time, channels, heads) in [(2, 9, 18, 2), (1, 3, 8, 2), (2, 1, 3, 3), (1, 2, 1, 1)]
    {
        let mut gb = block(&en, channels, heads);
        let mut nb = block(&Naive, channels, heads);
        let s = BlockSpec {
            batch,
            time,
            channels,
            heads,
            dropout: 0.0,
        };
        let x = values(s.elements());
        let dy = values(x.len());
        let mut nc = <Naive as Engine>::BlockCache::default();
        let mut expected = vec![f32::NAN; x.len()];
        let mut actual = en.filled(f32::NAN, x.len()).unwrap();
        let gx = en.buffer_from_slice(&x).unwrap();
        Naive
            .block_forward(
                s,
                nb.weights(),
                &x,
                ForwardMode::default(),
                &mut nws,
                &mut expected,
                Some(&mut nc),
            )
            .unwrap();
        en.block_forward(
            s,
            gb.weights(),
            &gx,
            ForwardMode::default(),
            &mut ws,
            &mut actual,
            Some(&mut cache),
        )
        .unwrap();
        close(&download(&en, &actual), &expected, 2e-4);
        let other = BlockSpec {
            time: time + 1,
            ..s
        };
        en.block_forward(
            other,
            gb.weights(),
            &en.filled(3.0, other.elements()).unwrap(),
            ForwardMode::default(),
            &mut ws,
            &mut en.zeroes(other.elements()).unwrap(),
            None,
        )
        .unwrap();
        let mut dx = vec![f32::NAN; x.len()];
        let (w, g) = nb.parts();
        Naive
            .block_backward(s, w, g, &nc, &dy, &mut nws, &mut dx)
            .unwrap();
        let gdy = en.buffer_from_slice(&dy).unwrap();
        let mut gdx = en.filled(f32::NAN, x.len()).unwrap();
        for _ in 0..2 {
            for p in gb.parameters_mut() {
                p.gradients = en.filled(f32::NAN, en.buffer_len(&p.gradients)).unwrap();
            }
            let (w, g) = gb.parts();
            en.block_backward(s, w, g, &cache, &gdy, &mut ws, &mut gdx)
                .unwrap();
            close(&download(&en, &gdx), &dx, 3e-4);
            for (a, e) in gb.parameters().iter().zip(nb.parameters()) {
                close(&download(&en, &a.gradients), &e.gradients, 3e-4);
            }
        }
        let (w, g) = gb.parts();
        assert!(
            en.block_backward(other, w, g, &cache, &gdy, &mut ws, &mut gdx)
                .is_err()
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn attention_is_causal_and_separates_batches_and_heads() {
    let en = gpu();
    let s = BlockSpec {
        batch: 2,
        time: 5,
        channels: 6,
        heads: 2,
        dropout: 0.0,
    };
    let count = s.batch * s.heads * s.time * s.time;
    let mut qkv = values(3 * s.elements());
    let mut probabilities = en.zeroes(count).unwrap();
    let mut mask = en.zeroes(count).unwrap();
    let mut out = en.zeroes(s.elements()).unwrap();
    en.attention_forward(
        s,
        &en.buffer_from_slice(&qkv).unwrap(),
        &mut probabilities,
        &mut mask,
        &mut out,
        ForwardMode::default(),
    )
    .unwrap();
    let original = download(&en, &out);
    let p = download(&en, &probabilities);
    for row in 0..s.batch * s.heads * s.time {
        let query = row % s.time;
        close(
            &[p[row * s.time..(row + 1) * s.time].iter().sum()],
            &[1.0],
            1e-6,
        );
        assert!(
            p[row * s.time + query + 1..(row + 1) * s.time]
                .iter()
                .all(|&v| v == 0.0)
        );
    }
    // Change only the final token of the second head of the second batch.
    let width = s.channels / s.heads;
    for part in 0..3 {
        let start = ((s.batch * s.time - 1) * 3 + part) * s.channels + width;
        qkv[start..start + width].fill(10.0);
    }
    en.attention_forward(
        s,
        &en.buffer_from_slice(&qkv).unwrap(),
        &mut probabilities,
        &mut mask,
        &mut out,
        ForwardMode::default(),
    )
    .unwrap();
    let changed = download(&en, &out);
    let untouched = s.elements() - width;
    assert_eq!(&original[..untouched], &changed[..untouched]);
    assert_ne!(&original[untouched..], &changed[untouched..]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn dropout_repeats_masks_and_backward_matches_finite_differences() {
    let en = gpu();
    let mut b = block(&en, 6, 2);
    let s = BlockSpec {
        batch: 2,
        time: 3,
        channels: 6,
        heads: 2,
        dropout: 0.3,
    };
    let mode = ForwardMode { seed: Some(42) };
    let mut x = values(s.elements());
    let dy = values(x.len());
    let gx = en.buffer_from_slice(&x).unwrap();
    let gdy = en.buffer_from_slice(&dy).unwrap();
    let mut out = en.zeroes(x.len()).unwrap();
    let mut ws = CudaWorkspace::default();
    let mut cache = CudaBlockCache::default();
    en.block_forward(
        s,
        b.weights(),
        &gx,
        mode,
        &mut ws,
        &mut out,
        Some(&mut cache),
    )
    .unwrap();
    let first = download(&en, &out);
    let saved = cache.saved.as_ref().unwrap();
    let first_masks = [
        &saved.attention_mask,
        &saved.projection_mask,
        &saved.mlp_mask,
    ]
    .map(|m| download(&en, m));
    for mask in &first_masks {
        assert!(
            mask.iter()
                .all(|&v| v == 0.0 || (v - 1.0 / 0.7).abs() < 1e-6)
        );
        assert!(mask.contains(&0.0));
        assert!(mask.iter().any(|&v| v > 1.0));
    }
    assert_ne!(first_masks[1], first_masks[2]);
    en.block_forward(s, b.weights(), &gx, mode, &mut ws, &mut out, None)
        .unwrap();
    assert_eq!(download(&en, &out), first);
    en.block_forward(
        s,
        b.weights(),
        &gx,
        mode,
        &mut ws,
        &mut out,
        Some(&mut cache),
    )
    .unwrap();
    let saved = cache.saved.as_ref().unwrap();
    for (m, expected) in [
        &saved.attention_mask,
        &saved.projection_mask,
        &saved.mlp_mask,
    ]
    .iter()
    .zip(&first_masks)
    {
        assert_eq!(download(&en, m), *expected);
    }
    let mut dx = en.zeroes(x.len()).unwrap();
    let (w, g) = b.parts();
    en.block_backward(s, w, g, &cache, &gdy, &mut ws, &mut dx)
        .unwrap();
    let analytic = download(&en, &dx);
    let objective = |input: &[f32], ws: &mut CudaWorkspace, out: &mut CudaBuffer| -> f32 {
        en.block_forward(
            s,
            b.weights(),
            &en.buffer_from_slice(input).unwrap(),
            mode,
            ws,
            out,
            None,
        )
        .unwrap();
        download(&en, out).iter().zip(&dy).map(|(a, b)| a * b).sum()
    };
    for index in [0, 7, x.len() / 2, x.len() - 1] {
        let original = x[index];
        let h = 2e-3 * (1.0 + original.abs());
        x[index] = original + h;
        let plus = objective(&x, &mut ws, &mut out);
        x[index] = original - h;
        let minus = objective(&x, &mut ws, &mut out);
        x[index] = original;
        close(&[analytic[index]], &[(plus - minus) / (2.0 * h)], 5e-3);
    }
    for parameter in 0..b.parameters().len() {
        let p = b.parameters()[parameter];
        let mut values = download(&en, &p.values);
        let index = values.len() / 2;
        let expected = download(&en, &p.gradients)[index];
        let original = values[index];
        let h = 2e-3 * (1.0 + original.abs());
        let mut objectives = [0.0; 2];
        for (value, objective) in [original + h, original - h]
            .into_iter()
            .zip(&mut objectives)
        {
            values[index] = value;
            b.parameters_mut()[parameter].values = en.buffer_from_slice(&values).unwrap();
            en.block_forward(s, b.weights(), &gx, mode, &mut ws, &mut out, None)
                .unwrap();
            *objective = download(&en, &out)
                .iter()
                .zip(&dy)
                .map(|(a, b)| a * b)
                .sum();
        }
        values[index] = original;
        b.parameters_mut()[parameter].values = en.buffer_from_slice(&values).unwrap();
        close(
            &[expected],
            &[(objectives[0] - objectives[1]) / (2.0 * h)],
            5e-3,
        );
    }
    en.block_forward(
        s,
        b.weights(),
        &gx,
        ForwardMode { seed: Some(43) },
        &mut ws,
        &mut out,
        Some(&mut cache),
    )
    .unwrap();
    assert_ne!(
        download(&en, &cache.saved.as_ref().unwrap().projection_mask),
        first_masks[1]
    );
    en.block_forward(
        s,
        b.weights(),
        &gx,
        ForwardMode::default(),
        &mut ws,
        &mut out,
        None,
    )
    .unwrap();
    let no_seed = download(&en, &out);
    en.block_forward(
        BlockSpec { dropout: 0.0, ..s },
        b.weights(),
        &gx,
        mode,
        &mut ws,
        &mut out,
        None,
    )
    .unwrap();
    assert_eq!(download(&en, &out), no_seed);
    en.block_forward(
        s,
        b.weights(),
        &gx,
        ForwardMode { seed: Some(0) },
        &mut ws,
        &mut out,
        Some(&mut cache),
    )
    .unwrap();
    let saved = cache.saved.as_ref().unwrap();
    let projection = download(&en, &saved.projection_mask);
    let mlp = download(&en, &saved.mlp_mask);
    assert_ne!(&projection[1..], &mlp[..mlp.len() - 1]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn adamw_matches_cpu_and_reports_invalid_updates() {
    let en = gpu();
    let n = 257;
    let mut p = values(n);
    let g = values(n);
    let (mut m, mut v) = (vec![0.0; n], vec![0.0; n]);
    let (mut gp, gg, mut gm, mut gv) = (
        en.buffer_from_slice(&p).unwrap(),
        en.buffer_from_slice(&g).unwrap(),
        en.zeroes(n).unwrap(),
        en.zeroes(n).unwrap(),
    );
    let base = AdamWConfig {
        learning_rate: 0.03,
        weight_decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        step: 1,
    };
    for step in 1..=4 {
        let config = AdamWConfig { step, ..base };
        Naive
            .adamw_step(
                &mut [AdamWGroup {
                    values: &mut p,
                    gradients: &g,
                    first_moment: &mut m,
                    second_moment: &mut v,
                }],
                config,
            )
            .unwrap();
        en.adamw_step(
            &mut [AdamWGroup {
                values: &mut gp,
                gradients: &gg,
                first_moment: &mut gm,
                second_moment: &mut gv,
            }],
            config,
        )
        .unwrap();
        close(&download(&en, &gp), &p, 1e-5);
        close(&download(&en, &gm), &m, 1e-5);
        close(&download(&en, &gv), &v, 1e-5);
    }
    assert!(
        en.adamw_step(&mut [], AdamWConfig { step: 0, ..base })
            .is_err()
    );
    assert!(
        en.adamw_step(
            &mut [],
            AdamWConfig {
                learning_rate: f32::NAN,
                ..base
            }
        )
        .is_err()
    );
    assert!(
        en.adamw_step(&mut [], AdamWConfig { beta1: 1.0, ..base })
            .is_err()
    );
    let before = download(&en, &gp);
    let wrong = en.zeroes(n - 1).unwrap();
    assert!(
        en.adamw_step(
            &mut [AdamWGroup {
                values: &mut gp,
                gradients: &wrong,
                first_moment: &mut gm,
                second_moment: &mut gv
            }],
            base
        )
        .is_err()
    );
    assert_eq!(download(&en, &gp), before);
    for invalid in [f32::NAN, f32::INFINITY, f32::MAX] {
        let bad = en.filled(invalid, n).unwrap();
        assert!(
            en.adamw_step(
                &mut [AdamWGroup {
                    values: &mut gp,
                    gradients: &bad,
                    first_moment: &mut gm,
                    second_moment: &mut gv
                }],
                base
            )
            .is_err()
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn complete_model_parity_training_generation_and_serialization() {
    let en = gpu();
    let tokenizer = Tokenizer::new("abc");
    let config = ModelConfig::micro();
    let mut cpu = Gpt::new(&Naive, config.clone(), tokenizer.clone(), &mut rng()).unwrap();
    let bytes = cpu.to_bytes(&Naive).unwrap();
    let mut model = Gpt::try_from_bytes(&en, &bytes).unwrap();
    assert_same_model(&Naive, &cpu, &en, &model);
    // Tokenizer::char_to_idx is a HashMap: compare decoded state, not map order.
    let round_trip = Gpt::try_from_bytes(&Naive, &model.to_bytes(&en).unwrap()).unwrap();
    assert_same_model(&Naive, &cpu, &Naive, &round_trip);
    let tokens = [0, 1, 2, 0];
    let targets = [1, 2, 0, 1];
    close(
        &download(&en, &model.forward(&en, &tokens, 2, 2).unwrap()),
        &cpu.forward(&Naive, &tokens, 2, 2).unwrap(),
        2e-4,
    );
    let cpu_loss = cpu
        .loss_and_backward(&Naive, &tokens, &targets, 2, 2, &mut rng())
        .unwrap();
    let gpu_loss = model
        .loss_and_backward(&en, &tokens, &targets, 2, 2, &mut rng())
        .unwrap();
    close(&[gpu_loss], &[cpu_loss], 2e-4);
    for (gp, cp) in model.parameters().iter().zip(cpu.parameters()) {
        close(&download(&en, &gp.gradients), &cp.gradients, 5e-4);
    }
    let mut optimizer = AdamW::new(
        &en,
        &tokenizer,
        "abcabcabcabcabcabc",
        &mut model,
        config,
        false,
    )
    .unwrap()
    .with_learning_rate(0.03);
    let mut workspace = CudaWorkspace::default();
    for _ in 0..30 {
        model
            .loss_and_backward_with_workspace(
                &en,
                &tokens,
                &targets,
                2,
                2,
                &mut rng(),
                &mut workspace,
            )
            .unwrap();
        optimizer.step(&en, &mut model.parameters_mut()).unwrap();
    }
    let final_loss = en
        .cross_entropy(
            &model.forward(&en, &tokens, 2, 2).unwrap(),
            &en.indices_from_slice(&targets).unwrap(),
            4,
            model.vocab_size,
        )
        .unwrap();
    assert!(
        final_loss < gpu_loss * 0.8,
        "loss did not improve: {gpu_loss} -> {final_loss}"
    );
    let generated = model.generate(&en, &[0], 8, &mut rng()).unwrap();
    assert_eq!(generated.len(), 9);
    assert!(generated.iter().all(|&t| t < model.vocab_size));
    let restored = Gpt::try_from_bytes(&Naive, &model.to_bytes(&en).unwrap()).unwrap();
    assert_same_model(&en, &model, &Naive, &restored);
    close(
        &restored.forward(&Naive, &tokens, 2, 2).unwrap(),
        &download(&en, &model.forward(&en, &tokens, 2, 2).unwrap()),
        3e-4,
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn invalid_shapes_caches_and_foreign_buffers_return_errors() {
    let en = gpu();
    let other = gpu();
    let foreign = other.zeroes(4).unwrap();
    assert!(en.buffer_to_vec(&foreign).is_err());
    let mut b = block(&en, 4, 2);
    let spec = BlockSpec {
        batch: 1,
        time: 1,
        channels: 4,
        heads: 2,
        dropout: 0.0,
    };
    let x = en.zeroes(4).unwrap();
    let mut out = en.zeroes(4).unwrap();
    let mut ws = CudaWorkspace::default();
    let mut cache = CudaBlockCache::default();
    let (w, g) = b.parts();
    assert!(
        en.block_backward(spec, w, g, &cache, &x, &mut ws, &mut out)
            .is_err()
    );
    en.block_forward(
        spec,
        b.weights(),
        &x,
        ForwardMode::default(),
        &mut ws,
        &mut out,
        Some(&mut cache),
    )
    .unwrap();
    assert!(
        en.block_forward(
            spec,
            b.weights(),
            &en.zeroes(3).unwrap(),
            ForwardMode::default(),
            &mut ws,
            &mut out,
            Some(&mut cache)
        )
        .is_err()
    );
    let (w, g) = b.parts();
    assert!(
        en.block_backward(spec, w, g, &cache, &x, &mut ws, &mut out)
            .is_err()
    );
    assert!(
        en.block_forward(
            BlockSpec { heads: 0, ..spec },
            b.weights(),
            &x,
            ForwardMode::default(),
            &mut ws,
            &mut out,
            None
        )
        .is_err()
    );
    assert!(
        en.block_forward(
            BlockSpec {
                dropout: 1.0,
                ..spec
            },
            b.weights(),
            &x,
            ForwardMode::default(),
            &mut ws,
            &mut out,
            None
        )
        .is_err()
    );
    assert!(
        en.linear_forward(
            LinearSpec {
                rows: 1,
                inputs: 4,
                outputs: 4
            },
            LinearWeights {
                weight: &en.zeroes(15).unwrap(),
                bias: None
            },
            &x,
            &mut out
        )
        .is_err()
    );
}
