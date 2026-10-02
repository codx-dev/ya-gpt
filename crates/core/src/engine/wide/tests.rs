use rand::{SeedableRng as _, rngs::StdRng};

use super::{SimdWide, SimdWideBlockCache, SimdWideLayerNormCache, SimdWideWorkspace};
use crate::{
    config::ModelConfig,
    engine::{naive::Naive, *},
    model::{Block, gpt::Gpt, optimizer::AdamW},
    tokenizer::Tokenizer,
};

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: {a} != {e} (tolerance {tolerance})"
        );
    }
}

fn values(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7 % 19) as f32 - 9.0) / 7.0)
        .collect()
}

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&a, &b)| a as f64 * b as f64).sum()
}

#[test]
fn buffers_round_trip_and_finiteness_covers_every_lane_and_tail() {
    for len in [0, 1, 7, 8, 9, 16, 17, 33] {
        assert_eq!(SimdWide.zeroes(len).unwrap(), vec![0.0; len]);
        assert_eq!(SimdWide.filled(-2.5, len).unwrap(), vec![-2.5; len]);
        let mut input = values(len);
        let buffer = SimdWide.buffer_from_slice(&input).unwrap();
        assert_eq!(SimdWide.buffer_len(&buffer), len);
        input.fill(100.0);
        let mut downloaded = SimdWide.buffer_to_vec(&buffer).unwrap();
        assert_eq!(downloaded, values(len));
        downloaded.fill(0.0);
        assert_eq!(buffer, values(len));
        assert!(SimdWide.buffers_all_finite(&[&buffer, &vec![]]).unwrap());
        for index in 0..len {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut bad = buffer.clone();
                bad[index] = invalid;
                assert!(!SimdWide.buffers_all_finite(&[&buffer, &bad]).unwrap());
            }
        }
    }
    let mut indices = vec![2, 0, 2];
    let uploaded = SimdWide.indices_from_slice(&indices).unwrap();
    indices.fill(0);
    assert_eq!(uploaded, [2, 0, 2]);
    assert!(SimdWide.indices_from_slice(&[]).unwrap().is_empty());
    assert!(SimdWide.buffers_all_finite(&[]).unwrap());
    SimdWide.synchronize().unwrap();
}

#[test]
fn linear_matches_naive_and_overwrites_gradients() {
    for inputs in [1, 7, 8, 9, 17] {
        for outputs in [1, 7, 8, 9, 17, 32] {
            let spec = LinearSpec {
                rows: 3,
                inputs,
                outputs,
            };
            let x = values(spec.rows * inputs);
            let w = values(inputs * outputs);
            let bias = values(outputs);
            let dy = values(spec.rows * outputs);
            for with_bias in [false, true] {
                let weights = || LinearWeights {
                    weight: &w,
                    bias: with_bias.then_some(&bias),
                };
                let mut actual = vec![f32::NAN; dy.len()];
                let mut expected = actual.clone();
                SimdWide
                    .linear_forward(spec, weights(), &x, &mut actual)
                    .unwrap();
                Naive
                    .linear_forward(spec, weights(), &x, &mut expected)
                    .unwrap();
                close(&actual, &expected, 1e-5);
                let [mut dx, mut dw, mut db] =
                    [x.len(), w.len(), bias.len()].map(|len| vec![f32::NAN; len]);
                let [mut nx, mut nw, mut nb] =
                    [x.len(), w.len(), bias.len()].map(|len| vec![f32::NAN; len]);
                Naive
                    .linear_backward(
                        spec,
                        weights(),
                        LinearGradients {
                            weight: &mut nw,
                            bias: with_bias.then_some(&mut nb),
                        },
                        &x,
                        &dy,
                        &mut nx,
                    )
                    .unwrap();
                for _ in 0..2 {
                    SimdWide
                        .linear_backward(
                            spec,
                            weights(),
                            LinearGradients {
                                weight: &mut dw,
                                bias: with_bias.then_some(&mut db),
                            },
                            &x,
                            &dy,
                            &mut dx,
                        )
                        .unwrap();
                    close(&dx, &nx, 1e-5);
                    close(&dw, &nw, 1e-5);
                    if with_bias {
                        close(&db, &nb, 1e-5);
                    }
                }
            }
        }
    }
}

#[test]
fn normalization_matches_naive_and_refreshes_cache() {
    let mut cache = SimdWideLayerNormCache::default();
    for channels in [17, 1, 7, 8, 9, 32] {
        let s = NormSpec {
            rows: 3,
            channels,
            epsilon: 1e-5,
        };
        for constant in [false, true] {
            let x = if constant {
                vec![2.0; s.rows * channels]
            } else {
                values(s.rows * channels)
            };
            let gamma = values(channels);
            let beta = values(channels);
            let weights = || NormWeights {
                gamma: &gamma,
                beta: &beta,
            };
            let mut output = vec![f32::NAN; x.len()];
            let mut expected = output.clone();
            let mut naive_cache = <Naive as Engine>::LayerNormCache::default();
            SimdWide
                .layer_norm_forward(s, weights(), &x, &mut output, Some(&mut cache))
                .unwrap();
            Naive
                .layer_norm_forward(s, weights(), &x, &mut expected, Some(&mut naive_cache))
                .unwrap();
            close(&output, &expected, 1e-5);
            close(&cache.normalized, &naive_cache.normalized, 1e-5);
            close(&cache.inverse_std, &naive_cache.inverse_std, 1e-5);
            assert_eq!(cache.spec, Some(s));
            let cached = output.clone();
            SimdWide
                .layer_norm_forward(s, weights(), &x, &mut output, None)
                .unwrap();
            assert_eq!(output, cached);
            let dy: Vec<_> = values(x.len()).iter().map(|v| v * 0.01).collect();
            let [mut dx, mut dg, mut db] =
                [x.len(), channels, channels].map(|len| vec![f32::NAN; len]);
            let [mut nx, mut ng, mut nb] =
                [x.len(), channels, channels].map(|len| vec![f32::NAN; len]);
            Naive
                .layer_norm_backward(
                    s,
                    weights(),
                    NormGradients {
                        gamma: &mut ng,
                        beta: &mut nb,
                    },
                    &naive_cache,
                    &dy,
                    &mut nx,
                )
                .unwrap();
            for _ in 0..2 {
                SimdWide
                    .layer_norm_backward(
                        s,
                        weights(),
                        NormGradients {
                            gamma: &mut dg,
                            beta: &mut db,
                        },
                        &cache,
                        &dy,
                        &mut dx,
                    )
                    .unwrap();
                close(&dx, &nx, 1e-5);
                close(&dg, &ng, 1e-5);
                close(&db, &nb, 1e-5);
            }
            if channels == 1 {
                assert_eq!(output, vec![beta[0]; s.rows]);
                assert_eq!(dx, vec![0.0; s.rows]);
                assert_eq!(dg, [0.0]);
            }
        }
    }
}

#[test]
fn embeddings_match_naive_for_repeated_tokens_and_batch_positions() {
    for channels in [1, 7, 8, 9, 17] {
        let s = EmbeddingSpec {
            batch: 2,
            time: 3,
            channels,
            vocab: 4,
            positions: 5,
        };
        let tokens = vec![1, 1, 0, 1, 2, 1];
        let token_table = values(s.vocab * channels);
        let position_table = values(s.positions * channels);
        let mut output = vec![f32::NAN; tokens.len() * channels];
        let mut expected = output.clone();
        SimdWide
            .token_position_embedding_forward(
                s,
                &tokens,
                &token_table,
                &position_table,
                &mut output,
            )
            .unwrap();
        Naive
            .token_position_embedding_forward(
                s,
                &tokens,
                &token_table,
                &position_table,
                &mut expected,
            )
            .unwrap();
        assert_eq!(output, expected);
        let dy = values(output.len());
        let [mut dt, mut dp] =
            [token_table.len(), position_table.len()].map(|len| vec![f32::NAN; len]);
        let [mut nt, mut np] =
            [token_table.len(), position_table.len()].map(|len| vec![f32::NAN; len]);
        Naive
            .token_position_embedding_backward(s, &tokens, &dy, &mut nt, &mut np)
            .unwrap();
        for _ in 0..2 {
            SimdWide
                .token_position_embedding_backward(s, &tokens, &dy, &mut dt, &mut dp)
                .unwrap();
            assert_eq!(dt, nt);
            assert_eq!(dp, np);
        }
    }
}

#[test]
fn loss_matches_naive_for_lane_boundaries_and_extreme_logits() {
    for vocab in [1, 7, 8, 9, 17, 32] {
        for magnitude in [1.0, 1000.0] {
            let logits: Vec<_> = values(3 * vocab).iter().map(|v| v * magnitude).collect();
            let targets = vec![0, vocab / 2, vocab - 1];
            let mut gradient = vec![f32::NAN; logits.len()];
            let mut expected = gradient.clone();
            let naive_loss = Naive
                .cross_entropy_with_grad(&logits, &targets, 3, vocab, &mut expected)
                .unwrap();
            for _ in 0..2 {
                let loss = SimdWide
                    .cross_entropy_with_grad(&logits, &targets, 3, vocab, &mut gradient)
                    .unwrap();
                close(&[loss], &[naive_loss], 1e-5);
                close(&gradient, &expected, 1e-5);
                assert_eq!(
                    loss,
                    SimdWide.cross_entropy(&logits, &targets, 3, vocab).unwrap()
                );
                for row in gradient.chunks_exact(vocab) {
                    close(&[row.iter().sum()], &[0.0], 1e-6);
                }
                if vocab == 1 {
                    assert_eq!(loss, 0.0);
                    assert_eq!(gradient, vec![0.0; 3]);
                }
            }
        }
    }
    // Cover vector exponentials, subnormals, scalar tails, and -infinity after subtraction.
    let logits = vec![
        f32::MAX,
        -f32::MAX,
        0.0,
        -85.0,
        -90.0,
        -100.0,
        -1000.0,
        0.0,
        1.0,
    ];
    assert_eq!(
        SimdWide.cross_entropy(&logits, &vec![0], 1, 9).unwrap(),
        0.0
    );
    let masses = SimdWide::exp_chunk(
        &[
            0.0,
            -80.0,
            -85.0,
            -90.0,
            -100.0,
            -104.0,
            -1000.0,
            f32::NEG_INFINITY,
        ],
        0.0,
    );
    for (actual, exponent) in masses.iter().zip([
        0.0_f32,
        -80.0,
        -85.0,
        -90.0,
        -100.0,
        -104.0,
        -1000.0,
        f32::NEG_INFINITY,
    ]) {
        let expected = exponent.exp();
        assert!((*actual - expected).abs() <= expected * 1e-5 || *actual == expected);
    }
    for index in [0, 7, 8, 16] {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut logits = vec![0.0; 17];
            logits[index] = invalid;
            assert!(SimdWide.cross_entropy(&logits, &vec![0], 1, 17).is_err());
            assert!(
                SimdWide
                    .cross_entropy_with_grad(&logits, &vec![0], 1, 17, &mut vec![0.0; 17])
                    .is_err()
            );
        }
    }
    assert!(
        SimdWide
            .cross_entropy(&vec![f32::MAX, -f32::MAX], &vec![1], 1, 2)
            .is_err()
    );
}

#[test]
fn sampling_obeys_cdf_boundaries_and_uses_only_last_row() {
    let near_one = f32::from_bits(1.0_f32.to_bits() - 1);
    for vocab in [1, 7, 8, 9, 17, 32] {
        let mut logits = vec![f32::NAN; vocab];
        logits.extend(vec![0.0; vocab]);
        for uniform in [0.0, 0.1, 0.51, near_one] {
            assert_eq!(
                SimdWide
                    .sample_last_token(&logits, 2, vocab, uniform)
                    .unwrap(),
                Naive.sample_last_token(&logits, 2, vocab, uniform).unwrap()
            );
        }
        logits[vocab..].fill(-1000.0);
        logits[2 * vocab - 1] = 1000.0;
        for uniform in [0.0, 0.5, near_one] {
            assert_eq!(
                SimdWide
                    .sample_last_token(&logits, 2, vocab, uniform)
                    .unwrap(),
                vocab - 1
            );
        }
    }
    assert_eq!(
        SimdWide
            .sample_last_token(&vec![0.0; 8], 1, 8, 0.125)
            .unwrap(),
        1
    );
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in [0, 7, 8] {
            let mut logits = vec![0.0; 9];
            logits[index] = invalid;
            assert!(SimdWide.sample_last_token(&logits, 1, 9, 0.5).is_err());
        }
    }
}

fn optimizer_config() -> AdamWConfig {
    AdamWConfig {
        learning_rate: 0.01,
        weight_decay: 0.2,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        step: 1,
    }
}

fn update<E: Engine<Buffer = Vec<f32>>>(
    engine: &E,
    states: &mut [[Vec<f32>; 4]],
    config: AdamWConfig,
) -> anyhow::Result<()> {
    let mut groups: Vec<_> = states
        .iter_mut()
        .map(
            |[values, gradients, first_moment, second_moment]| AdamWGroup {
                values,
                gradients,
                first_moment,
                second_moment,
            },
        )
        .collect();
    engine.adamw_step(&mut groups, config)
}

#[test]
fn adamw_matches_naive_over_multiple_steps_and_groups() {
    let mut actual: Vec<_> = [0, 1, 7, 8, 9, 17, 32]
        .map(|len| [values(len), values(len), vec![0.0; len], vec![0.0; len]])
        .into();
    let mut expected = actual.clone();
    for step in 1..=5 {
        let config = AdamWConfig {
            step,
            ..optimizer_config()
        };
        update(&SimdWide, &mut actual, config).unwrap();
        update(&Naive, &mut expected, config).unwrap();
        for (a, e) in actual.iter().zip(&expected) {
            for (a, e) in a.iter().zip(e) {
                close(a, e, 1e-5);
            }
        }
        for states in [&mut actual, &mut expected] {
            for state in states {
                for gradient in &mut state[1] {
                    *gradient = -*gradient * 0.5;
                }
            }
        }
    }
}

#[test]
fn adamw_rejects_invalid_configuration_and_nonfinite_updates() {
    let config = optimizer_config();
    for invalid in [
        AdamWConfig {
            learning_rate: -1.0,
            ..config
        },
        AdamWConfig {
            learning_rate: f32::NAN,
            ..config
        },
        AdamWConfig {
            weight_decay: -1.0,
            ..config
        },
        AdamWConfig {
            weight_decay: f32::INFINITY,
            ..config
        },
        AdamWConfig {
            beta1: 1.0,
            ..config
        },
        AdamWConfig {
            beta2: -0.1,
            ..config
        },
        AdamWConfig {
            epsilon: 0.0,
            ..config
        },
        AdamWConfig { step: 0, ..config },
    ] {
        assert!(SimdWide.adamw_step(&mut [], invalid).is_err());
    }
    for index in [0, 7, 8, 16] {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let mut states = [[vec![1.0; 17], vec![1.0; 17], vec![0.0; 17], vec![0.0; 17]]];
            states[0][1][index] = invalid;
            assert!(update(&SimdWide, &mut states, config).is_err());
        }
    }
}

fn block<E: Engine<Buffer = Vec<f32>>>(engine: &E, channels: usize, heads: usize) -> Block<E> {
    let config = ModelConfig {
        n_embd: channels,
        n_head: heads,
        ..ModelConfig::micro()
    };
    let mut block = Block::new(engine, &config, &mut rng()).unwrap();
    for p in block.parameters_mut() {
        p.values = values(p.values.len())
            .into_iter()
            .map(|x| x * 0.08)
            .collect();
    }
    for norm in [&mut block.norm1, &mut block.norm2] {
        for gamma in &mut norm.gamma.values {
            *gamma += 0.9;
        }
    }
    for (i, bias) in block
        .expand
        .bias
        .as_mut()
        .unwrap()
        .values
        .iter_mut()
        .enumerate()
    {
        *bias = if i % 2 == 0 { 0.5 } else { -0.5 };
    }
    block
}

#[test]
fn blocks_match_naive_with_dropout_and_reused_caches_and_workspaces() {
    let mut ws = SimdWideWorkspace::default();
    let mut cache = SimdWideBlockCache::default();
    let mut nws = <Naive as Engine>::Workspace::default();
    let mut nc = <Naive as Engine>::BlockCache::default();
    for (batch, time, channels, heads) in [
        (2, 9, 18, 2),
        (1, 3, 16, 2),
        (2, 1, 3, 3),
        (1, 8, 8, 1),
        (1, 2, 1, 1),
    ] {
        let mut wide = block(&SimdWide, channels, heads);
        let mut naive = block(&Naive, channels, heads);
        for (dropout, seed) in [(0.0, None), (0.0, Some(42)), (0.3, Some(42)), (0.3, None)] {
            let spec = BlockSpec {
                batch,
                time,
                channels,
                heads,
                dropout,
            };
            let x = values(spec.elements());
            let dy: Vec<_> = values(x.len()).iter().map(|v| v * 0.7 + 0.2).collect();
            let mut out = vec![f32::NAN; x.len()];
            let mut expected = out.clone();
            let mode = ForwardMode { seed };
            SimdWide
                .block_forward(
                    spec,
                    wide.weights(),
                    &x,
                    mode,
                    &mut ws,
                    &mut out,
                    Some(&mut cache),
                )
                .unwrap();
            Naive
                .block_forward(
                    spec,
                    naive.weights(),
                    &x,
                    mode,
                    &mut nws,
                    &mut expected,
                    Some(&mut nc),
                )
                .unwrap();
            close(&out, &expected, 1e-4);
            assert_eq!(cache.spec, Some(spec));
            assert_eq!(cache.attention_mask, nc.attention_mask);
            assert_eq!(cache.projection_mask, nc.projection_mask);
            assert_eq!(cache.mlp_mask, nc.mlp_mask);
            close(&cache.probabilities, &nc.probabilities, 1e-5);
            let saved = out.clone();
            SimdWide
                .block_forward(spec, wide.weights(), &x, mode, &mut ws, &mut out, None)
                .unwrap();
            assert_eq!(out, saved);
            // Backward must read its cache even after unrelated inference resized the workspace.
            let other = BlockSpec {
                time: time + 1,
                ..spec
            };
            SimdWide
                .block_forward(
                    other,
                    wide.weights(),
                    &vec![3.0; other.elements()],
                    mode,
                    &mut ws,
                    &mut vec![0.0; other.elements()],
                    None,
                )
                .unwrap();
            let mut dx = vec![f32::NAN; x.len()];
            let mut nx = dx.clone();
            let (w, g) = naive.parts();
            Naive
                .block_backward(spec, w, g, &nc, &dy, &mut nws, &mut nx)
                .unwrap();
            for _ in 0..2 {
                for p in wide.parameters_mut() {
                    p.gradients.fill(f32::NAN);
                }
                let (w, g) = wide.parts();
                SimdWide
                    .block_backward(spec, w, g, &cache, &dy, &mut ws, &mut dx)
                    .unwrap();
                close(&dx, &nx, 1e-4);
                for (a, e) in wide.parameters().iter().zip(naive.parameters()) {
                    close(&a.gradients, &e.gradients, 1e-4);
                }
            }
        }
    }
}

#[test]
fn attention_is_causal_and_keeps_batches_and_heads_separate() {
    let spec = BlockSpec {
        batch: 2,
        time: 9,
        channels: 18,
        heads: 2,
        dropout: 0.0,
    };
    let mut qkv = values(3 * spec.elements());
    let mut out = vec![0.0; spec.elements()];
    let mut scores = vec![0.0; spec.time];
    SimdWide::attention_forward(
        spec,
        &qkv,
        ForwardMode::default(),
        &mut scores,
        &mut out,
        None,
    );
    let expected = out.clone();
    let row = (spec.time - 1) * 3 * spec.channels;
    // Modify the last token's first head in only the first batch.
    for component in 0..3 {
        qkv[row + component * spec.channels..row + component * spec.channels + 9].fill(10.0);
    }
    SimdWide::attention_forward(
        spec,
        &qkv,
        ForwardMode::default(),
        &mut scores,
        &mut out,
        None,
    );
    assert_eq!(
        &out[..(spec.time - 1) * spec.channels],
        &expected[..(spec.time - 1) * spec.channels]
    );
    assert_eq!(
        &out[spec.time * spec.channels..],
        &expected[spec.time * spec.channels..]
    );
    for (a, e) in out
        .chunks_exact(spec.channels)
        .zip(expected.chunks_exact(spec.channels))
    {
        assert_eq!(&a[9..], &e[9..]);
    }
    assert_ne!(
        &out[(spec.time - 1) * spec.channels..spec.time * spec.channels],
        &expected[(spec.time - 1) * spec.channels..spec.time * spec.channels]
    );
}

#[test]
fn block_gradients_match_selected_finite_differences() {
    for dropout in [0.0, 0.3] {
        let spec = BlockSpec {
            batch: 1,
            time: 2,
            channels: 9,
            heads: 1,
            dropout,
        };
        let mode = ForwardMode { seed: Some(42) };
        let mut block = block(&SimdWide, 9, 1);
        let mut x = values(spec.elements());
        let dy = values(x.len());
        let mut ws = SimdWideWorkspace::default();
        let mut cache = SimdWideBlockCache::default();
        let mut out = vec![0.0; x.len()];
        let mut dx = out.clone();
        SimdWide
            .block_forward(
                spec,
                block.weights(),
                &x,
                mode,
                &mut ws,
                &mut out,
                Some(&mut cache),
            )
            .unwrap();
        let (w, g) = block.parts();
        SimdWide
            .block_backward(spec, w, g, &cache, &dy, &mut ws, &mut dx)
            .unwrap();
        let gradients: Vec<_> = block
            .parameters()
            .iter()
            .map(|p| p.gradients.clone())
            .collect();
        for index in [0, 7, 8, 9, x.len() - 1] {
            let original = x[index];
            let h = 1e-3;
            x[index] = original + h;
            SimdWide
                .block_forward(spec, block.weights(), &x, mode, &mut ws, &mut out, None)
                .unwrap();
            let plus = dot(&out, &dy);
            x[index] = original - h;
            SimdWide
                .block_forward(spec, block.weights(), &x, mode, &mut ws, &mut out, None)
                .unwrap();
            let minus = dot(&out, &dy);
            x[index] = original;
            close(
                &[dx[index]],
                &[((plus - minus) / (2.0 * h as f64)) as f32],
                5e-3,
            );
        }
        for (parameter, expected) in gradients.iter().enumerate() {
            for index in [0, 7, 8, expected.len() - 1] {
                let original = block.parameters()[parameter].values[index];
                let h = 1e-3;
                block.parameters_mut()[parameter].values[index] = original + h;
                SimdWide
                    .block_forward(spec, block.weights(), &x, mode, &mut ws, &mut out, None)
                    .unwrap();
                let plus = dot(&out, &dy);
                block.parameters_mut()[parameter].values[index] = original - h;
                SimdWide
                    .block_forward(spec, block.weights(), &x, mode, &mut ws, &mut out, None)
                    .unwrap();
                let minus = dot(&out, &dy);
                block.parameters_mut()[parameter].values[index] = original;
                close(
                    &[expected[index]],
                    &[((plus - minus) / (2.0 * h as f64)) as f32],
                    5e-3,
                );
            }
        }
    }
}

#[test]
fn models_match_and_checkpoints_work_across_backends() {
    let config = ModelConfig {
        n_embd: 18,
        n_head: 2,
        block_size: 9,
        dropout: 0.3,
        ..ModelConfig::micro()
    };
    let tokenizer = Tokenizer::new("abcdefghijklmnopq");
    let mut naive = Gpt::new(&Naive, config.clone(), tokenizer.clone(), &mut rng()).unwrap();
    let mut wide = Gpt::new(&SimdWide, config, tokenizer, &mut rng()).unwrap();
    let tokens: Vec<_> = (0..18).map(|i| i % 17).collect();
    let targets: Vec<_> = (0..18).map(|i| (i + 1) % 17).collect();
    let n_loss = naive
        .loss_and_backward(&Naive, &tokens, &targets, 2, 9, &mut rng())
        .unwrap();
    let w_loss = wide
        .loss_and_backward(&SimdWide, &tokens, &targets, 2, 9, &mut rng())
        .unwrap();
    close(&[w_loss], &[n_loss], 1e-4);
    for (a, e) in wide.parameters().iter().zip(naive.parameters()) {
        close(&a.gradients, &e.gradients, 1e-4);
    }
    let bytes = naive.to_bytes(&Naive).unwrap();
    let loaded = Gpt::try_from_bytes(&SimdWide, &bytes).unwrap();
    assert_eq!(loaded.to_bytes(&SimdWide).unwrap(), bytes);
    close(
        &loaded.forward(&SimdWide, &tokens, 2, 9).unwrap(),
        &naive.forward(&Naive, &tokens, 2, 9).unwrap(),
        1e-4,
    );
    let bytes = wide.to_bytes(&SimdWide).unwrap();
    let loaded = Gpt::try_from_bytes(&Naive, &bytes).unwrap();
    assert_eq!(loaded.to_bytes(&Naive).unwrap(), bytes);
    close(
        &loaded.forward(&Naive, &tokens, 2, 9).unwrap(),
        &wide.forward(&SimdWide, &tokens, 2, 9).unwrap(),
        1e-4,
    );
}

#[test]
fn training_reduces_loss_and_reuses_parameter_allocations() {
    let config = ModelConfig {
        n_embd: 9,
        n_head: 1,
        ..ModelConfig::micro()
    };
    let tokenizer = Tokenizer::new("abc");
    let mut model = Gpt::new(&SimdWide, config.clone(), tokenizer.clone(), &mut rng()).unwrap();
    let pointers: Vec<_> = model
        .parameters()
        .iter()
        .map(|p| (p.values.as_ptr(), p.gradients.as_ptr()))
        .collect();
    let mut optimizer = AdamW::new(
        &SimdWide,
        &tokenizer,
        &"abc".repeat(100),
        &mut model,
        config,
        false,
    )
    .unwrap()
    .with_learning_rate(0.02);
    let tokens = [0, 1, 2, 0, 1, 2, 0, 1];
    let targets = [1, 2, 0, 1, 2, 0, 1, 2];
    let mut ws = SimdWideWorkspace::default();
    let initial = model
        .loss_and_backward_with_workspace(&SimdWide, &tokens, &targets, 2, 4, &mut rng(), &mut ws)
        .unwrap();
    for _ in 0..40 {
        model
            .loss_and_backward_with_workspace(
                &SimdWide,
                &tokens,
                &targets,
                2,
                4,
                &mut rng(),
                &mut ws,
            )
            .unwrap();
        optimizer
            .step(&SimdWide, &mut model.parameters_mut())
            .unwrap();
    }
    let loss = model
        .loss_and_backward_with_workspace(&SimdWide, &tokens, &targets, 2, 4, &mut rng(), &mut ws)
        .unwrap();
    assert!(loss < initial * 0.5, "loss {initial} -> {loss}");
    for (parameter, &(values, gradients)) in model.parameters().iter().zip(&pointers) {
        assert_eq!(parameter.values.as_ptr(), values);
        assert_eq!(parameter.gradients.as_ptr(), gradients);
    }
}

#[test]
#[ignore = "manual release-mode timing; run with --release --ignored --nocapture"]
fn time_linear_kernels() {
    use std::{hint::black_box, time::Instant};
    fn measure<E: Engine<Buffer = Vec<f32>>>(
        engine: &E,
        spec: LinearSpec,
    ) -> (std::time::Duration, std::time::Duration) {
        let input = values(spec.rows * spec.inputs);
        let weight = values(spec.inputs * spec.outputs);
        let dy = values(spec.rows * spec.outputs);
        let mut out = vec![0.0; dy.len()];
        let mut dx = vec![0.0; input.len()];
        let mut dw = vec![0.0; weight.len()];
        // Warm both paths before measuring.
        engine
            .linear_forward(
                spec,
                LinearWeights {
                    weight: &weight,
                    bias: None,
                },
                &input,
                &mut out,
            )
            .unwrap();
        engine
            .linear_backward(
                spec,
                LinearWeights {
                    weight: &weight,
                    bias: None,
                },
                LinearGradients {
                    weight: &mut dw,
                    bias: None,
                },
                &input,
                &dy,
                &mut dx,
            )
            .unwrap();
        let start = Instant::now();
        for _ in 0..100 {
            engine
                .linear_forward(
                    spec,
                    LinearWeights {
                        weight: black_box(&weight),
                        bias: None,
                    },
                    black_box(&input),
                    black_box(&mut out),
                )
                .unwrap();
            black_box(&out);
        }
        let forward = start.elapsed();
        let start = Instant::now();
        for _ in 0..100 {
            engine
                .linear_backward(
                    spec,
                    LinearWeights {
                        weight: black_box(&weight),
                        bias: None,
                    },
                    LinearGradients {
                        weight: black_box(&mut dw),
                        bias: None,
                    },
                    black_box(&input),
                    black_box(&dy),
                    black_box(&mut dx),
                )
                .unwrap();
            black_box((&dx, &dw));
        }
        (forward, start.elapsed())
    }
    for (rows, inputs, outputs) in [(32, 32, 128), (64, 64, 256), (33, 17, 65)] {
        let spec = LinearSpec {
            rows,
            inputs,
            outputs,
        };
        let naive = measure(&Naive, spec);
        let wide = measure(&SimdWide, spec);
        eprintln!("{spec:?}: naive forward/backward {naive:?}, wide {wide:?}");
    }
}
