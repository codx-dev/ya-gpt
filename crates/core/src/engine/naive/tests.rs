use super::{Naive, NaiveBlockCache, NaiveLayerNormCache};
use crate::engine::{BlockSpec, ForwardMode, LinearSpec, NormSpec, NormWeights};

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

fn dot(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&a, &b)| a as f64 * b as f64).sum()
}

fn numerical_gradient(input: &[f32], mut loss: impl FnMut(&[f32]) -> f64) -> Vec<f32> {
    let mut perturbed = input.to_vec();
    (0..input.len())
        .map(|i| {
            perturbed[i] = input[i] + 1e-3;
            let upper = perturbed[i];
            let plus = loss(&perturbed);
            perturbed[i] = input[i] - 1e-3;
            let lower = perturbed[i];
            let minus = loss(&perturbed);
            perturbed[i] = input[i];
            ((plus - minus) / (upper as f64 - lower as f64)) as f32
        })
        .collect()
}

#[test]
fn linear_forward_with_and_without_bias_overwrites_output() {
    let spec = LinearSpec {
        rows: 2,
        inputs: 3,
        outputs: 2,
    };
    let x = [1., 2., 3., 4., 5., 6.];
    let w = [1., -1., 2., 0., -1., 3.];
    let mut out = [f32::NAN; 4];

    Naive::linear_forward(spec, &x, &w, Some(&[0.5, -0.5]), &mut out);
    close(&out, &[2.5, 7.5, 8.5, 13.5], 1e-6);
    Naive::linear_forward(spec, &x, &w, None, &mut out);
    close(&out, &[2., 8., 8., 14.], 1e-6);
}

#[test]
fn linear_backward_overwrites_and_sums_gradients_across_rows() {
    let spec = LinearSpec {
        rows: 2,
        inputs: 3,
        outputs: 2,
    };
    let x = [1., 2., 3., 4., 5., 6.];
    let w = [1., -1., 2., 0., -1., 3.];
    let dy = [1., 2., 3., 4.];
    let mut dx = [f32::NAN; 6];
    let mut dw = [f32::NAN; 6];
    let mut db = [f32::NAN; 2];

    for _ in 0..2 {
        Naive::linear_backward(spec, &x, &w, &dy, &mut dx, &mut dw, Some(&mut db));
        close(&dx, &[-1., 2., 5., -1., 6., 9.], 1e-6);
        close(&dw, &[13., 18., 17., 24., 21., 30.], 1e-6);
        close(&db, &[4., 6.], 1e-6);
    }
    Naive::linear_backward(spec, &x, &w, &dy, &mut dx, &mut dw, None);
    close(&dx, &[-1., 2., 5., -1., 6., 9.], 1e-6);
    close(&dw, &[13., 18., 17., 24., 21., 30.], 1e-6);
}

#[test]
fn norm_forward_normalizes_each_row_and_refreshes_cache() {
    let spec = NormSpec {
        rows: 2,
        channels: 3,
        epsilon: 1e-5,
    };
    let x = [1., 2., 3., 5., 5., 5.];
    let gamma = vec![1., 2., -1.];
    let beta = vec![0., 0.5, 1.];
    let mut out = [f32::NAN; 6];
    let mut cache = NaiveLayerNormCache::default();
    let inv = 1.0 / (2.0_f32 / 3.0 + spec.epsilon).sqrt();

    Naive::norm_forward(
        spec,
        NormWeights {
            gamma: &gamma,
            beta: &beta,
        },
        &x,
        &mut out,
        Some(&mut cache),
    );
    close(&out, &[-inv, 0.5, 1. - inv, 0., 0.5, 1.], 1e-6);
    assert_eq!(cache.spec, Some(spec));
    close(&cache.normalized, &[-inv, 0., inv, 0., 0., 0.], 1e-6);
    close(&cache.inverse_std, &[inv, 1.0 / spec.epsilon.sqrt()], 1e-6);

    let mut uncached = [f32::NAN; 6];
    Naive::norm_forward(
        spec,
        NormWeights {
            gamma: &gamma,
            beta: &beta,
        },
        &x,
        &mut uncached,
        None,
    );
    assert_eq!(out, uncached);

    let smaller = NormSpec { rows: 1, ..spec };
    Naive::norm_forward(
        smaller,
        NormWeights {
            gamma: &gamma,
            beta: &beta,
        },
        &x[3..],
        &mut out[..3],
        Some(&mut cache),
    );
    assert_eq!(cache.spec, Some(smaller));
    close(&out[..3], &beta, 1e-6);
    close(&cache.normalized, &[0.; 3], 1e-6);
    close(&cache.inverse_std, &[1.0 / spec.epsilon.sqrt()], 1e-6);
}

#[test]
fn norm_backward_matches_finite_differences_for_input_gamma_and_beta() {
    let spec = NormSpec {
        rows: 2,
        channels: 3,
        epsilon: 1e-5,
    };
    let x = values(6);
    let gamma = vec![1., 2., -1.];
    let beta = vec![0.2, -0.5, 1.];
    let dy = [0.7, -0.2, 0.4, -0.9, 0.5, 0.3];
    let mut out = vec![0.; 6];
    let mut cache = NaiveLayerNormCache::default();
    Naive::norm_forward(
        spec,
        NormWeights {
            gamma: &gamma,
            beta: &beta,
        },
        &x,
        &mut out,
        Some(&mut cache),
    );

    let mut objective = |x: &[f32], gamma: &Vec<f32>, beta: &Vec<f32>| {
        Naive::norm_forward(spec, NormWeights { gamma, beta }, x, &mut out, None);
        dot(&out, &dy)
    };
    let expected_dx = numerical_gradient(&x, |x| objective(x, &gamma, &beta));
    let expected_dg = numerical_gradient(&gamma, |g| objective(&x, &g.to_vec(), &beta));
    let expected_db = numerical_gradient(&beta, |b| objective(&x, &gamma, &b.to_vec()));
    let mut dx = [f32::NAN; 6];
    let mut dg = [f32::NAN; 3];
    let mut db = [f32::NAN; 3];
    for _ in 0..2 {
        Naive::norm_backward(spec, &gamma, &cache, &dy, &mut dx, &mut dg, &mut db);
        close(&dx, &expected_dx, 2e-3);
        close(&dg, &expected_dg, 2e-3);
        close(&db, &expected_db, 2e-3);
    }
}

#[test]
fn norm_single_channel_has_only_beta_gradient() {
    let spec = NormSpec {
        rows: 2,
        channels: 1,
        epsilon: 1e-5,
    };
    let gamma = vec![2.];
    let beta = vec![0.5];
    let mut out = [f32::NAN; 2];
    let mut cache = NaiveLayerNormCache::default();
    Naive::norm_forward(
        spec,
        NormWeights {
            gamma: &gamma,
            beta: &beta,
        },
        &[3., -2.],
        &mut out,
        Some(&mut cache),
    );
    close(&out, &[0.5, 0.5], 1e-6);
    let mut dx = [f32::NAN; 2];
    let mut dg = [f32::NAN];
    let mut db = [f32::NAN];
    Naive::norm_backward(
        spec,
        &gamma,
        &cache,
        &[0.7, -0.2],
        &mut dx,
        &mut dg,
        &mut db,
    );
    close(&dx, &[0., 0.], 1e-6);
    close(&dg, &[0.], 1e-6);
    close(&db, &[0.5], 1e-6);
}

#[test]
fn dropout_rng_and_values_are_reproducible_and_stream_specific() {
    let probability = 0.3;
    let mode = ForwardMode { seed: Some(42) };
    assert!(Naive::dropout_rng(ForwardMode::default(), probability, 0).is_none());
    assert!(Naive::dropout_rng(mode, 0.0, 0).is_none());
    assert_eq!(Naive::dropout_value(&mut None, probability), 1.0);

    let samples = |seed, stream| {
        let mut rng = Naive::dropout_rng(ForwardMode { seed: Some(seed) }, probability, stream);
        assert!(rng.is_some());
        (0..256)
            .map(|_| Naive::dropout_value(&mut rng, probability))
            .collect::<Vec<_>>()
    };
    let mask = samples(42, 0);
    let scale = 1.0 / (1.0 - probability);
    assert_eq!(mask, samples(42, 0));
    assert_ne!(mask, samples(43, 0));
    assert_ne!(mask, samples(42, 1));
    assert!(mask.iter().all(|&m| m == 0.0 || m == scale));
    assert!(mask.contains(&0.0));
    assert!(mask.contains(&scale));
}

#[test]
fn dropout_add_saves_branch_mask_and_preserves_residual() {
    let branch = values(128);
    let residual = vec![0.7; branch.len()];
    let mode = ForwardMode { seed: Some(42) };
    let mut out = branch.clone();
    let mut mask = vec![f32::NAN; 1];
    Naive::dropout_add(&mut out, &residual, 0.5, mode, 1, Some(&mut mask));
    assert_eq!(mask.len(), branch.len());
    assert!(mask.contains(&0.0));
    assert!(mask.contains(&2.0));
    for i in 0..branch.len() {
        assert!(mask[i] == 0.0 || mask[i] == 2.0);
        close(&out[i..=i], &[branch[i] * mask[i] + residual[i]], 1e-6);
    }
    let expected = out.clone();
    let expected_mask = mask.clone();
    out.copy_from_slice(&branch);
    Naive::dropout_add(&mut out, &residual, 0.5, mode, 1, Some(&mut mask));
    assert_eq!(out, expected);
    assert_eq!(mask, expected_mask);
    out.copy_from_slice(&branch);
    Naive::dropout_add(&mut out, &residual, 0.5, mode, 1, None);
    assert_eq!(out, expected);
}

#[test]
fn dropout_add_without_dropout_clears_saved_mask() {
    for (mode, probability) in [
        (ForwardMode::default(), 0.5),
        (ForwardMode { seed: Some(42) }, 0.0),
    ] {
        let mut branch = [1., -2., 3.];
        let mut mask = vec![0.; 3];
        Naive::dropout_add(
            &mut branch,
            &[0.5, 1., -1.],
            probability,
            mode,
            2,
            Some(&mut mask),
        );
        close(&branch, &[1.5, -1., 2.], 1e-6);
        assert!(mask.is_empty());
    }
}

#[test]
fn masked_applies_saved_multipliers_or_passes_input_through() {
    let input = [1., -2., 3., 4.];
    let mut out = [f32::NAN; 4];
    Naive::masked(&input, &[0., 2., 2., 0.], &mut out);
    close(&out, &[0., -4., 6., 0.], 1e-6);
    Naive::masked(&input, &[], &mut out);
    assert_eq!(out, input);
}

#[test]
fn attention_forward_uniform_scores_average_causal_values_per_batch_and_head() {
    let spec = BlockSpec {
        batch: 2,
        time: 3,
        channels: 4,
        heads: 2,
        dropout: 0.5,
    };
    // Zero queries and keys give uniform probabilities over each causal prefix.
    let mut qkv = vec![0.; spec.elements() * 3];
    for (row, token) in qkv.chunks_exact_mut(3 * spec.channels).enumerate() {
        for ch in 0..spec.channels {
            token[2 * spec.channels + ch] = (10 * row + ch + 1) as f32;
        }
    }
    let mut cache = NaiveBlockCache::default();
    let mut scores = vec![f32::NAN; spec.time];
    let mut out = vec![f32::NAN; spec.elements()];
    // Reuse the training cache for inference to check that masks are cleared.
    for mode in [ForwardMode { seed: Some(42) }, ForwardMode::default()] {
        Naive::attention_forward(spec, &qkv, mode, &mut scores, &mut out, Some(&mut cache));
        assert_eq!(
            cache.probabilities.len(),
            spec.batch * spec.heads * spec.time * spec.time
        );
        if mode.seed.is_some() {
            assert_eq!(cache.attention_mask.len(), cache.probabilities.len());
            assert!(cache.attention_mask.contains(&2.0));
        } else {
            assert!(cache.attention_mask.is_empty());
        }
        for b in 0..spec.batch {
            for t in 0..spec.time {
                for h in 0..spec.heads {
                    let offset = ((b * spec.heads + h) * spec.time + t) * spec.time;
                    let row = &cache.probabilities[offset..offset + spec.time];
                    close(&row[..=t], &vec![1.0 / (t + 1) as f32; t + 1], 1e-6);
                    close(&row[t + 1..], &vec![0.; spec.time - t - 1], 1e-6);
                    for ch in h * 2..(h + 1) * 2 {
                        let expected = (0..=t)
                            .map(|k| {
                                let value = (10 * (b * spec.time + k) + ch + 1) as f32;
                                let mask = if mode.seed.is_some() {
                                    let m = cache.attention_mask[offset + k];
                                    assert!(m == 0.0 || m == 2.0);
                                    m
                                } else {
                                    1.0
                                };
                                value * mask / (t + 1) as f32
                            })
                            .sum::<f32>();
                        close(
                            &[out[(b * spec.time + t) * spec.channels + ch]],
                            &[expected],
                            1e-6,
                        );
                    }
                }
            }
        }
        let mut uncached = vec![f32::NAN; out.len()];
        Naive::attention_forward(spec, &qkv, mode, &mut scores, &mut uncached, None);
        assert_eq!(out, uncached);
    }
}

#[test]
fn attention_forward_scales_scores_and_uses_stable_softmax() {
    let spec = BlockSpec {
        batch: 1,
        time: 2,
        channels: 2,
        heads: 1,
        dropout: 0.0,
    };
    let q = 2.0_f32.sqrt();
    // Scaled logits are 100 and 100 + ln(3), so the second row has weights 1/4, 3/4.
    let qkv = [
        q,
        0.,
        100.,
        0.,
        2.,
        4.,
        q,
        0.,
        100. + 3.0_f32.ln(),
        0.,
        6.,
        8.,
    ];
    let mut out = [f32::NAN; 4];
    let mut cache = NaiveBlockCache::default();
    Naive::attention_forward(
        spec,
        &qkv,
        ForwardMode::default(),
        &mut [f32::NAN; 2],
        &mut out,
        Some(&mut cache),
    );
    close(&out, &[2., 4., 5., 7.], 1e-5);
    close(&cache.probabilities, &[1., 0., 0.25, 0.75], 1e-5);
}

#[test]
fn attention_forward_is_causal_and_batches_are_isolated() {
    let spec = BlockSpec {
        batch: 2,
        time: 3,
        channels: 4,
        heads: 2,
        dropout: 0.0,
    };
    let mut qkv = values(3 * spec.elements());
    let mut scores = vec![0.; spec.time];
    let mut original = vec![0.; spec.elements()];
    Naive::attention_forward(
        spec,
        &qkv,
        ForwardMode::default(),
        &mut scores,
        &mut original,
        None,
    );
    // Change the last token of batch zero and every token of batch one.
    qkv[2 * 3 * spec.channels..].fill(7.0);
    let mut changed = vec![0.; spec.elements()];
    Naive::attention_forward(
        spec,
        &qkv,
        ForwardMode::default(),
        &mut scores,
        &mut changed,
        None,
    );
    assert_eq!(
        &changed[..2 * spec.channels],
        &original[..2 * spec.channels]
    );
    assert_ne!(
        &changed[2 * spec.channels..],
        &original[2 * spec.channels..]
    );
}

#[test]
fn attention_backward_matches_finite_differences_with_and_without_dropout() {
    for (batch, time, channels, heads) in [(2, 3, 6, 2), (1, 3, 4, 1), (1, 2, 3, 3), (1, 1, 1, 1)] {
        for dropout in [0.0, 0.3] {
            let spec = BlockSpec {
                batch,
                time,
                channels,
                heads,
                dropout,
            };
            let qkv = values(3 * spec.elements());
            let dy = values(spec.elements());
            let mode = ForwardMode { seed: Some(42) };
            let mut scores = vec![0.; time];
            let mut out = vec![0.; spec.elements()];
            // The block caller saves QKV; the attention kernel saves probabilities and masks.
            let mut cache = NaiveBlockCache {
                qkv: qkv.clone(),
                ..Default::default()
            };
            Naive::attention_forward(spec, &qkv, mode, &mut scores, &mut out, Some(&mut cache));
            let expected = numerical_gradient(&qkv, |input| {
                // Reset the seed for every perturbation to hold the dropout mask fixed.
                Naive::attention_forward(spec, input, mode, &mut scores, &mut out, None);
                dot(&out, &dy)
            });
            let mut dqkv = vec![f32::NAN; qkv.len()];
            for _ in 0..2 {
                Naive::attention_backward(spec, &cache, &dy, &mut scores, &mut dqkv);
                close(&dqkv, &expected, 2e-3);
            }
        }
    }
}

#[test]
fn loss_averages_rows_and_overwrites_softmax_gradient() {
    let logits = [0., 2.0_f32.ln(), 3.0_f32.ln()].repeat(2);
    let targets = [2, 0];
    let expected_loss = (2.0_f32.ln() + 6.0_f32.ln()) / 2.0;
    let mut gradient = vec![f32::NAN; logits.len()];
    for _ in 0..2 {
        let loss = Naive::loss(&logits, &targets, 2, 3, Some(&mut gradient)).unwrap();
        close(&[loss], &[expected_loss], 1e-6);
        close(
            &gradient,
            &[1. / 12., 1. / 6., -0.25, -5. / 12., 1. / 6., 0.25],
            1e-6,
        );
        assert_eq!(loss, Naive::loss(&logits, &targets, 2, 3, None).unwrap());
    }
    let shifted: Vec<_> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| v + if i < 3 { 1000. } else { -1000. })
        .collect();
    let loss = Naive::loss(&shifted, &targets, 2, 3, Some(&mut gradient)).unwrap();
    close(&[loss], &[expected_loss], 5e-5);
    close(
        &gradient,
        &[1. / 12., 1. / 6., -0.25, -5. / 12., 1. / 6., 0.25],
        5e-5,
    );
}

#[test]
fn loss_gradient_matches_finite_differences() {
    let logits = values(12);
    let targets = [3, 0, 2];
    let mut gradient = vec![f32::NAN; logits.len()];
    Naive::loss(&logits, &targets, 3, 4, Some(&mut gradient)).unwrap();
    let expected = numerical_gradient(&logits, |input| {
        Naive::loss(input, &targets, 3, 4, None).unwrap() as f64
    });
    close(&gradient, &expected, 2e-3);
    for row in gradient.as_chunks::<4>().0 {
        close(&[row.iter().sum()], &[0.], 1e-6);
    }
}

#[test]
fn loss_single_class_is_zero_with_zero_gradient() {
    let mut gradient = vec![f32::NAN; 2];
    let loss = Naive::loss(&[1000., -1000.], &[0, 0], 2, 1, Some(&mut gradient)).unwrap();
    assert_eq!(loss, 0.0);
    assert_eq!(gradient, [0., 0.]);
}

// Exercise the Engine entry points separately from the kernels above.
mod engine_impl {
    use super::*;
    use crate::engine::{
        AdamWConfig, AdamWGroup, BlockGradients, BlockWeights, EmbeddingSpec, Engine,
        LinearGradients, LinearWeights, NormGradients, naive::NaiveWorkspace,
    };

    #[test]
    fn buffers_round_trip_without_aliasing_and_report_non_finite_values() {
        for len in [0, 4] {
            let zeroes = Naive.zeroes(len).unwrap();
            assert_eq!(Naive.buffer_len(&zeroes), len);
            assert_eq!(zeroes, vec![0.; len]);
            assert_eq!(Naive.filled(-2.5, len).unwrap(), vec![-2.5; len]);
        }
        let mut input = vec![1., -2., 3.];
        let buffer = Naive.buffer_from_slice(&input).unwrap();
        input[0] = 99.;
        let mut exported = Naive.buffer_to_vec(&buffer).unwrap();
        assert_eq!(exported, [1., -2., 3.]);
        exported[1] = 99.;
        assert_eq!(buffer, [1., -2., 3.]);

        let mut indices = vec![2, 0, 2];
        let tokens = Naive.indices_from_slice(&indices).unwrap();
        indices[0] = 99;
        assert_eq!(tokens, [2, 0, 2]);
        assert!(Naive.indices_from_slice(&[]).unwrap().is_empty());
        assert!(Naive.buffers_all_finite(&[]).unwrap());
        assert!(Naive.buffers_all_finite(&[&buffer, &vec![]]).unwrap());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                !Naive
                    .buffers_all_finite(&[&buffer, &vec![invalid]])
                    .unwrap()
            );
        }
        Naive.synchronize().unwrap();
    }

    #[test]
    fn linear_forward_and_backward_route_weights_biases_and_gradients() {
        let spec = LinearSpec {
            rows: 2,
            inputs: 3,
            outputs: 2,
        };
        let input = vec![1., 2., 3., 4., 5., 6.];
        let weight = vec![1., -1., 2., 0., -1., 3.];
        let bias = vec![0.5, -0.5];
        let dy = vec![1., 2., 3., 4.];
        for with_bias in [true, false] {
            let mut output = vec![f32::NAN; 4];
            let mut dx = vec![f32::NAN; 6];
            let mut dw = vec![f32::NAN; 6];
            let mut db = vec![f32::NAN; 2];
            for _ in 0..2 {
                Naive
                    .linear_forward(
                        spec,
                        LinearWeights {
                            weight: &weight,
                            bias: with_bias.then_some(&bias),
                        },
                        &input,
                        &mut output,
                    )
                    .unwrap();
                let expected = if with_bias {
                    [2.5, 7.5, 8.5, 13.5]
                } else {
                    [2., 8., 8., 14.]
                };
                close(&output, &expected, 1e-6);
                Naive
                    .linear_backward(
                        spec,
                        LinearWeights {
                            weight: &weight,
                            bias: with_bias.then_some(&bias),
                        },
                        LinearGradients {
                            weight: &mut dw,
                            bias: with_bias.then_some(&mut db),
                        },
                        &input,
                        &dy,
                        &mut dx,
                    )
                    .unwrap();
                close(&dx, &[-1., 2., 5., -1., 6., 9.], 1e-6);
                close(&dw, &[13., 18., 17., 24., 21., 30.], 1e-6);
                if with_bias {
                    close(&db, &[4., 6.], 1e-6);
                }
            }
        }
    }

    #[test]
    fn layer_norm_forward_and_backward_match_finite_differences() {
        let spec = NormSpec {
            rows: 2,
            channels: 3,
            epsilon: 1e-3,
        };
        let input = vec![1., 2., 3., -2., 1., 0.];
        let gamma = vec![1., 2., -1.];
        let beta = vec![0.2, -0.5, 1.];
        let dy = vec![0.7, -0.2, 0.4, -0.9, 0.5, 0.3];
        let mut output = vec![f32::NAN; 6];
        let mut cache = NaiveLayerNormCache::default();
        Naive
            .layer_norm_forward(
                spec,
                NormWeights {
                    gamma: &gamma,
                    beta: &beta,
                },
                &input,
                &mut output,
                Some(&mut cache),
            )
            .unwrap();
        let inv = 1. / (2. / 3. + spec.epsilon).sqrt();
        close(&output[..3], &[0.2 - inv, -0.5, 1. - inv], 1e-6);
        let saved = output.clone();
        let mut objective = |input: &[f32], gamma: &[f32], beta: &[f32]| {
            Naive
                .layer_norm_forward(
                    spec,
                    NormWeights {
                        gamma: &gamma.to_vec(),
                        beta: &beta.to_vec(),
                    },
                    &input.to_vec(),
                    &mut output,
                    None,
                )
                .unwrap();
            dot(&output, &dy)
        };
        let expected_dx = numerical_gradient(&input, |x| objective(x, &gamma, &beta));
        let expected_dg = numerical_gradient(&gamma, |g| objective(&input, g, &beta));
        let expected_db = numerical_gradient(&beta, |b| objective(&input, &gamma, b));
        objective(&input, &gamma, &beta);
        assert_eq!(output, saved);
        let mut dx = vec![f32::NAN; 6];
        let mut dg = vec![f32::NAN; 3];
        let mut db = vec![f32::NAN; 3];
        for _ in 0..2 {
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
                    &cache,
                    &dy,
                    &mut dx,
                )
                .unwrap();
            close(&dx, &expected_dx, 2e-3);
            close(&dg, &expected_dg, 2e-3);
            close(&db, &expected_db, 2e-3);
        }
    }

    #[test]
    fn embeddings_reset_positions_per_batch_and_sum_repeated_token_gradients() {
        let spec = EmbeddingSpec {
            batch: 2,
            time: 2,
            channels: 2,
            vocab: 3,
            positions: 3,
        };
        let tokens = Naive.indices_from_slice(&[1, 1, 0, 1]).unwrap();
        let table = vec![1., 2., 3., 4., 5., 6.];
        let positions = vec![10., 20., 30., 40., 50., 60.];
        let mut output = vec![f32::NAN; 8];
        Naive
            .token_position_embedding_forward(spec, &tokens, &table, &positions, &mut output)
            .unwrap();
        close(&output, &[13., 24., 33., 44., 11., 22., 33., 44.], 0.);
        let dy = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let mut dt = vec![f32::NAN; 6];
        let mut dp = vec![f32::NAN; 6];
        for _ in 0..2 {
            Naive
                .token_position_embedding_backward(spec, &tokens, &dy, &mut dt, &mut dp)
                .unwrap();
            close(&dt, &[5., 6., 11., 14., 0., 0.], 0.);
            close(&dp, &[6., 8., 10., 12., 0., 0.], 0.);
        }
    }

    #[test]
    fn cross_entropy_entry_points_agree_and_overwrite_gradients() {
        let logits = vec![1000., 1000., -1000., 1000.];
        let targets = Naive.indices_from_slice(&[1, 0]).unwrap();
        let mut gradient = vec![f32::NAN; 4];
        for _ in 0..2 {
            let loss = Naive
                .cross_entropy_with_grad(&logits, &targets, 2, 2, &mut gradient)
                .unwrap();
            close(&[loss], &[(2.0_f32.ln() + 2000.) / 2.], 1e-6);
            assert_eq!(loss, Naive.cross_entropy(&logits, &targets, 2, 2).unwrap());
            close(&gradient, &[0.25, -0.25, -0.5, 0.5], 1e-6);
        }
        let logits = values(12);
        let targets = vec![3, 0, 2];
        let mut gradient = vec![f32::NAN; 12];
        Naive
            .cross_entropy_with_grad(&logits, &targets, 3, 4, &mut gradient)
            .unwrap();
        let expected = numerical_gradient(&logits, |x| {
            Naive.cross_entropy(&x.to_vec(), &targets, 3, 4).unwrap() as f64
        });
        close(&gradient, &expected, 2e-3);
    }

    #[test]
    fn adamw_updates_multiple_groups_and_carries_moments_between_steps() {
        let mut w = vec![1., -2.];
        let mut m = vec![0.; 2];
        let mut v = vec![0.; 2];
        let mut decayed = vec![3.];
        let mut dm = vec![0.];
        let mut dv = vec![0.];
        let zero = vec![0.];
        for step in 1..=2 {
            let gradients = if step == 1 {
                vec![2., -4.]
            } else {
                vec![0., 0.]
            };
            Naive
                .adamw_step(
                    &mut [
                        AdamWGroup {
                            values: &mut w,
                            gradients: &gradients,
                            first_moment: &mut m,
                            second_moment: &mut v,
                        },
                        AdamWGroup {
                            values: &mut decayed,
                            gradients: &zero,
                            first_moment: &mut dm,
                            second_moment: &mut dv,
                        },
                    ],
                    AdamWConfig {
                        learning_rate: 0.1,
                        weight_decay: 0.2,
                        beta1: 0.5,
                        beta2: 0.5,
                        epsilon: 1.,
                        step,
                    },
                )
                .unwrap();
            if step == 1 {
                close(&w, &[0.98 - 0.2 / 3., -1.96 + 0.4 / 5.], 1e-6);
                close(&m, &[1., -2.], 1e-6);
                close(&v, &[2., 8.], 1e-6);
                close(&decayed, &[2.94], 1e-6);
            } else {
                close(
                    &w,
                    &[
                        (0.98 - 0.2 / 3.) * 0.98 - 0.1 * (2. / 3.) / ((4.0_f32 / 3.).sqrt() + 1.),
                        (-1.96 + 0.4 / 5.) * 0.98 + 0.1 * (4. / 3.) / ((16.0_f32 / 3.).sqrt() + 1.),
                    ],
                    1e-6,
                );
                close(&m, &[0.5, -1.], 1e-6);
                close(&v, &[1., 4.], 1e-6);
                close(&decayed, &[2.8812], 1e-6);
            }
            close(&dm, &[0.], 0.);
            close(&dv, &[0.], 0.);
        }
    }

    #[test]
    fn sampling_uses_last_row_and_skips_underflowed_probabilities() {
        let logits = vec![1000., -1000., 0., 3.0_f32.ln()];
        assert_eq!(Naive.sample_last_token(&logits, 2, 2, 0.).unwrap(), 0);
        assert_eq!(Naive.sample_last_token(&logits, 2, 2, 0.24).unwrap(), 0);
        assert_eq!(Naive.sample_last_token(&logits, 2, 2, 0.26).unwrap(), 1);
        let near_one = f32::from_bits(1.0_f32.to_bits() - 1);
        assert_eq!(Naive.sample_last_token(&logits, 2, 2, near_one).unwrap(), 1);
        let logits = vec![1000., 1000., 1000., -1000., 1000., -1000.];
        for uniform in [0., 0.5, near_one] {
            assert_eq!(Naive.sample_last_token(&logits, 2, 3, uniform).unwrap(), 1);
        }
        assert_eq!(
            Naive
                .sample_last_token(&vec![1000., -1000.], 2, 1, near_one)
                .unwrap(),
            0
        );
    }

    // Keep the fixture independent of the model layer so these tests exercise only Engine.
    type BlockBuffers = [Vec<f32>; 11];

    fn block_parameters(c: usize) -> BlockBuffers {
        let mut parameters = [
            c,
            c,
            3 * c * c,
            c * c,
            c,
            c,
            c,
            4 * c * c,
            4 * c,
            4 * c * c,
            c,
        ]
        .map(|len| {
            values(len)
                .into_iter()
                .map(|x| x * 0.08)
                .collect::<Vec<_>>()
        });
        for index in [0, 5] {
            for value in &mut parameters[index] {
                *value += 0.9;
            }
        }
        // Exercise both ReLU branches, away from its nondifferentiable zero.
        for (index, bias) in parameters[8].iter_mut().enumerate() {
            *bias = if index % 2 == 0 { 0.5 } else { -0.5 };
        }
        parameters
    }

    fn block_weights(parameters: &BlockBuffers) -> BlockWeights<'_, Vec<f32>> {
        let [
            g1,
            b1,
            qkv,
            attention,
            attention_bias,
            g2,
            b2,
            expand,
            expand_bias,
            project,
            project_bias,
        ] = parameters;
        BlockWeights {
            norm1: NormWeights {
                gamma: g1,
                beta: b1,
            },
            qkv,
            attention: LinearWeights {
                weight: attention,
                bias: Some(attention_bias),
            },
            norm2: NormWeights {
                gamma: g2,
                beta: b2,
            },
            expand: LinearWeights {
                weight: expand,
                bias: Some(expand_bias),
            },
            project: LinearWeights {
                weight: project,
                bias: Some(project_bias),
            },
        }
    }

    fn block_gradients(gradients: &mut BlockBuffers) -> BlockGradients<'_, Vec<f32>> {
        let [
            g1,
            b1,
            qkv,
            attention,
            attention_bias,
            g2,
            b2,
            expand,
            expand_bias,
            project,
            project_bias,
        ] = gradients;
        BlockGradients {
            norm1: NormGradients {
                gamma: g1,
                beta: b1,
            },
            qkv,
            attention: LinearGradients {
                weight: attention,
                bias: Some(attention_bias),
            },
            norm2: NormGradients {
                gamma: g2,
                beta: b2,
            },
            expand: LinearGradients {
                weight: expand,
                bias: Some(expand_bias),
            },
            project: LinearGradients {
                weight: project,
                bias: Some(project_bias),
            },
        }
    }

    #[test]
    fn block_single_channel_has_expected_residuals_and_relu() {
        let spec = BlockSpec {
            batch: 2,
            time: 2,
            channels: 1,
            heads: 1,
            dropout: 0.,
        };
        let parameters = [
            vec![2.],
            vec![3.],
            vec![1., 2., 4.],
            vec![2.],
            vec![1.],
            vec![5.],
            vec![2.],
            vec![1., -1., 2., -2.],
            vec![1., 1., -1., -1.],
            vec![1., 2., 3., 4.],
            vec![0.5],
        ];
        let input = vec![1., -2., 3., 4.];
        let mut out = vec![f32::NAN; 4];
        let mut ws = NaiveWorkspace::default();
        let mut cache = NaiveBlockCache::default();
        Naive
            .block_forward(
                spec,
                block_weights(&parameters),
                &input,
                ForwardMode::default(),
                &mut ws,
                &mut out,
                Some(&mut cache),
            )
            .unwrap();
        // Norm1 emits 3, hence V=12 and the attention branch contributes 25.
        // Norm2 emits 2; ReLU([3,-1,3,-5]) projects to 12.5.
        close(&out, &[38.5, 35.5, 40.5, 41.5], 1e-6);
        let dy = vec![1., -2., 3., -4.];
        let mut dx = vec![f32::NAN; 4];
        let mut gradients = parameters.each_ref().map(|p| vec![f32::NAN; p.len()]);
        Naive
            .block_backward(
                spec,
                block_weights(&parameters),
                block_gradients(&mut gradients),
                &cache,
                &dy,
                &mut ws,
                &mut dx,
            )
            .unwrap();
        // Single-channel layer norms have no input derivative: only the residual remains.
        close(&dx, &dy, 1e-6);
    }

    #[test]
    fn block_gradients_match_finite_differences_with_and_without_dropout() {
        for (batch, time, channels, heads) in [(2, 2, 4, 2), (1, 3, 3, 3), (1, 1, 3, 1)] {
            for dropout in [0., 0.3] {
                let spec = BlockSpec {
                    batch,
                    time,
                    channels,
                    heads,
                    dropout,
                };
                let mode = ForwardMode { seed: Some(42) };
                let mut parameters = block_parameters(channels);
                let input = values(spec.elements());
                let dy: Vec<_> = values(input.len())
                    .into_iter()
                    .map(|x| 0.7 * x + 0.2)
                    .collect();
                let mut out = vec![f32::NAN; input.len()];
                let mut dx = vec![f32::NAN; input.len()];
                let mut ws = NaiveWorkspace::default();
                let mut cache = NaiveBlockCache::default();
                Naive
                    .block_forward(
                        spec,
                        block_weights(&parameters),
                        &input,
                        mode,
                        &mut ws,
                        &mut out,
                        Some(&mut cache),
                    )
                    .unwrap();
                let cached_output = out.clone();
                assert!(cache.activated.contains(&0.));
                assert!(cache.activated.iter().any(|&x| x > 0.));

                // Resize and overwrite the workspace with unrelated work before backward.
                let other = BlockSpec {
                    batch: 1,
                    time: time + 1,
                    ..spec
                };
                Naive
                    .block_forward(
                        other,
                        block_weights(&parameters),
                        &vec![3.; other.elements()],
                        mode,
                        &mut ws,
                        &mut vec![0.; other.elements()],
                        None,
                    )
                    .unwrap();
                let mut gradients = parameters.each_ref().map(|p| vec![f32::NAN; p.len()]);
                Naive
                    .block_backward(
                        spec,
                        block_weights(&parameters),
                        block_gradients(&mut gradients),
                        &cache,
                        &dy,
                        &mut ws,
                        &mut dx,
                    )
                    .unwrap();
                let saved_dx = dx.clone();
                let saved_gradients = gradients.clone();
                Naive
                    .block_backward(
                        spec,
                        block_weights(&parameters),
                        block_gradients(&mut gradients),
                        &cache,
                        &dy,
                        &mut ws,
                        &mut dx,
                    )
                    .unwrap();
                assert_eq!(dx, saved_dx);
                assert_eq!(gradients, saved_gradients);

                Naive
                    .block_forward(
                        spec,
                        block_weights(&parameters),
                        &input,
                        mode,
                        &mut ws,
                        &mut out,
                        None,
                    )
                    .unwrap();
                assert_eq!(out, cached_output);
                // Reusing the seed holds every dropout mask fixed during differentiation.
                let expected_dx = numerical_gradient(&input, |x| {
                    Naive
                        .block_forward(
                            spec,
                            block_weights(&parameters),
                            &x.to_vec(),
                            mode,
                            &mut ws,
                            &mut out,
                            None,
                        )
                        .unwrap();
                    dot(&out, &dy)
                });
                close(&dx, &expected_dx, 3e-3);
                for index in 0..parameters.len() {
                    let original = parameters[index].clone();
                    let expected = numerical_gradient(&original, |p| {
                        parameters[index].copy_from_slice(p);
                        Naive
                            .block_forward(
                                spec,
                                block_weights(&parameters),
                                &input,
                                mode,
                                &mut ws,
                                &mut out,
                                None,
                            )
                            .unwrap();
                        dot(&out, &dy)
                    });
                    parameters[index] = original;
                    close(&gradients[index], &expected, 3e-3);
                }
            }
        }
    }

    #[test]
    fn block_cache_and_workspace_can_change_shape_and_switch_to_inference() {
        let mut ws = NaiveWorkspace::default();
        let mut cache = NaiveBlockCache::default();
        for (batch, time, channels, heads) in [(2, 3, 6, 2), (1, 1, 2, 1), (2, 2, 4, 4)] {
            let spec = BlockSpec {
                batch,
                time,
                channels,
                heads,
                dropout: 0.3,
            };
            let parameters = block_parameters(channels);
            let input = values(spec.elements());
            for mode in [ForwardMode { seed: Some(42) }, ForwardMode::default()] {
                let mut out = vec![f32::NAN; input.len()];
                Naive
                    .block_forward(
                        spec,
                        block_weights(&parameters),
                        &input,
                        mode,
                        &mut ws,
                        &mut out,
                        Some(&mut cache),
                    )
                    .unwrap();
                assert_eq!(cache.spec, Some(spec));
                for mask in [
                    &cache.attention_mask,
                    &cache.projection_mask,
                    &cache.mlp_mask,
                ] {
                    assert_eq!(mask.is_empty(), mode.seed.is_none());
                }
                let mut fresh_ws = NaiveWorkspace::default();
                let mut fresh_cache = NaiveBlockCache::default();
                let mut expected = vec![f32::NAN; input.len()];
                Naive
                    .block_forward(
                        spec,
                        block_weights(&parameters),
                        &input,
                        mode,
                        &mut fresh_ws,
                        &mut expected,
                        Some(&mut fresh_cache),
                    )
                    .unwrap();
                assert_eq!(out, expected);
                let dy = values(input.len());
                let mut dx = vec![f32::NAN; input.len()];
                let mut expected_dx = vec![f32::NAN; input.len()];
                let mut gradients = parameters.each_ref().map(|p| vec![f32::NAN; p.len()]);
                let mut expected_gradients = gradients.clone();
                Naive
                    .block_backward(
                        spec,
                        block_weights(&parameters),
                        block_gradients(&mut gradients),
                        &cache,
                        &dy,
                        &mut ws,
                        &mut dx,
                    )
                    .unwrap();
                Naive
                    .block_backward(
                        spec,
                        block_weights(&parameters),
                        block_gradients(&mut expected_gradients),
                        &fresh_cache,
                        &dy,
                        &mut fresh_ws,
                        &mut expected_dx,
                    )
                    .unwrap();
                close(&dx, &expected_dx, 0.);
                for (actual, expected) in gradients.iter().zip(&expected_gradients) {
                    close(actual, expected, 0.);
                }
            }
        }
    }
}
