use rand::{RngExt as _, SeedableRng, rngs::StdRng};

use super::types::{Attention, Block, FeedForward, Head, LayerNorm, Linear, Parameter};
use crate::{config::ModelConfig, engine::naive::Naive};

#[test]
fn filled_parameters_initialize_values_and_gradients() {
    let parameter = Parameter::filled(&Naive, 4, -2.5).unwrap();

    close(&parameter.values, &[-2.5; 4], 0.0);
    close(&parameter.gradients, &[0.0; 4], 0.0);

    let empty = Parameter::filled(&Naive, 0, 1.0).unwrap();

    assert!(empty.values.is_empty());
    assert!(empty.gradients.is_empty());
}

#[test]
fn normal_parameters_are_reproducible_with_zero_gradients() {
    let first = Parameter::normal(&Naive, 16, &mut seeded_rng()).unwrap();
    let second = Parameter::normal(&Naive, 16, &mut seeded_rng()).unwrap();

    close(&first.values, &second.values, 0.0);
    close(&first.gradients, &[0.0; 16], 0.0);
    assert!(first.values.iter().any(|value| *value != 0.0));

    let mut rng = seeded_rng();
    let mut untouched_rng = seeded_rng();
    let empty = Parameter::normal(&Naive, 0, &mut rng).unwrap();

    assert!(empty.values.is_empty());
    assert!(empty.gradients.is_empty());
    assert_eq!(rng.random::<u64>(), untouched_rng.random::<u64>());
}

#[test]
fn linear_forward_and_backward_match_hand_calculations() {
    let en = Naive;
    let mut linear = Linear::new(&en, 3, 2, true, &mut seeded_rng()).unwrap();
    set_parameters(
        linear.parameters_mut(),
        &[&[1.0, -1.0, 2.0, 0.5, -2.0, 3.0], &[0.5, -1.0]],
    );
    let input = vec![1.0, 2.0, -1.0, -2.0, 0.0, 3.0];
    let upstream = vec![2.0, -1.0, -3.0, 4.0];

    let output = linear.forward(&en, &input, 2).unwrap();
    let gradient = linear.backward(&en, &input, &upstream, 2).unwrap();

    close(&output, &[7.5, -4.0, -7.5, 10.0], 0.0);
    close(&gradient, &[3.0, 3.5, -7.0, -7.0, -4.0, 18.0], 0.0);
    close(
        &linear.parameters()[0].gradients,
        &[8.0, -9.0, 4.0, -2.0, -11.0, 13.0],
        0.0,
    );
    close(&linear.parameters()[1].gradients, &[-1.0, 3.0], 0.0);

    let gradient = linear.backward(&en, &input, &vec![0.0; 4], 2).unwrap();

    close(&gradient, &[0.0; 6], 0.0);
    for parameter in linear.parameters() {
        close(
            &parameter.gradients,
            &vec![0.0; parameter.values.len()],
            0.0,
        );
    }
}

#[test]
fn linear_without_bias_matches_finite_differences_with_more_rows_than_channels() {
    let en = Naive;
    let mut linear = Linear::new(&en, 2, 3, false, &mut seeded_rng()).unwrap();
    initialize_parameters(linear.parameters_mut());
    assert_eq!(linear.parameters().len(), 1);
    let input = vec![1.0, -0.5, 0.25, 2.0, -1.0, 0.75];
    let upstream = sample_values(9);

    let gradient = linear.backward(&en, &input, &upstream, 3).unwrap();
    let numerical = numerical_gradient(&input, |values| {
        dot(
            &linear.forward(&en, &values.to_vec(), 3).unwrap(),
            &upstream,
        )
    });

    close(&gradient, &numerical, 1e-6);
    assert_eq!(linear.parameters()[0].gradients.len(), 6);
    check_parameter_gradients(
        &mut linear,
        Linear::parameters,
        Linear::parameters_mut,
        |model| dot(&model.forward(&en, &input, 3).unwrap(), &upstream),
    );
}

#[test]
fn layer_norm_normalizes_rows_and_applies_affine_parameters() {
    let en = Naive;
    let mut norm = LayerNorm::new(&en, 3).unwrap();
    close(&norm.parameters()[0].values, &[1.0; 3], 0.0);
    close(&norm.parameters()[1].values, &[0.0; 3], 0.0);
    set_parameters(
        norm.parameters_mut(),
        &[&[1.0, 2.0, -1.0], &[0.5, -0.25, 1.0]],
    );
    let input = vec![1.0, 2.0, 3.0, 5.0, 5.0, 5.0];

    let output = norm.forward(&en, &input, 2).unwrap();
    let inverse_std = 1.0 / (2.0_f64 / 3.0 + 1e-5).sqrt();

    close(
        &output,
        &[0.5 - inverse_std, -0.25, 1.0 - inverse_std, 0.5, -0.25, 1.0],
        1e-12,
    );
}

#[test]
fn layer_norm_backward_matches_finite_differences() {
    let en = Naive;
    let mut norm = LayerNorm::new(&en, 3).unwrap();
    set_parameters(
        norm.parameters_mut(),
        &[&[0.7, -0.4, 1.2], &[0.2, 0.3, -0.1]],
    );
    let input = vec![1.0, -2.0, 0.5, 0.25, 1.5, -0.75];
    let upstream = vec![0.3, -0.4, 0.8, -0.6, 0.2, 0.5];

    let gradient = norm.backward(&en, &input, &upstream, 2).unwrap();
    let numerical = numerical_gradient(&input, |values| {
        dot(&norm.forward(&en, &values.to_vec(), 2).unwrap(), &upstream)
    });

    close(&gradient, &numerical, 1e-6);
    check_parameter_gradients(
        &mut norm,
        LayerNorm::parameters,
        LayerNorm::parameters_mut,
        |model| dot(&model.forward(&en, &input, 2).unwrap(), &upstream),
    );
}

#[test]
fn single_channel_layer_norm_has_only_bias_gradients() {
    let en = Naive;
    let mut norm = LayerNorm::new(&en, 1).unwrap();
    set_parameters(norm.parameters_mut(), &[&[2.0], &[0.5]]);
    let input = vec![2.0, -1.0, 4.0];

    let output = norm.forward(&en, &input, 3).unwrap();
    let gradient = norm
        .backward(&en, &input, &vec![0.5, -2.0, 3.0], 3)
        .unwrap();

    close(&output, &[0.5; 3], 0.0);
    close(&gradient, &[0.0; 3], 0.0);
    close(&norm.parameters()[0].gradients, &[0.0], 0.0);
    close(&norm.parameters()[1].gradients, &[1.5], 0.0);
}

#[test]
fn head_forward_is_causal_and_keeps_batches_separate() {
    let en = Naive;
    let mut head = Head::new(&en, 2, 2, 0.0, &mut seeded_rng()).unwrap();
    set_parameters(
        head.parameters_mut(),
        &[&[0.0; 4], &[0.0; 4], &[1.0, 0.0, 0.0, 1.0]],
    );
    let input = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
    ];

    let (output, cache) = head
        .forward(&en, input.clone(), 2, 3, false, &mut seeded_rng())
        .unwrap();
    let probabilities = [
        1.0,
        0.0,
        0.0,
        0.5,
        0.5,
        0.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ]
    .repeat(2);

    close(
        &output,
        &[
            1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0,
        ],
        1e-12,
    );
    close(&cache.input, &input, 0.0);
    close(&cache.query, &[0.0; 12], 0.0);
    close(&cache.key, &[0.0; 12], 0.0);
    close(&cache.value, &input, 0.0);
    close(&cache.probabilities, &probabilities, 1e-12);
    close(&cache.weights, &probabilities, 1e-12);
    close(&cache.mask, &[1.0; 18], 0.0);
}

#[test]
fn head_backward_matches_finite_differences_with_dropout() {
    let en = Naive;
    let mut head = Head::new(&en, 3, 2, 0.25, &mut seeded_rng()).unwrap();
    set_parameters(
        head.parameters_mut(),
        &[
            &[0.2, -0.3, 0.4, 0.1, -0.2, 0.5],
            &[-0.4, 0.1, 0.3, 0.6, 0.2, -0.5],
            &[0.5, -0.2, -0.3, 0.4, 0.6, 0.1],
        ],
    );
    let input = sample_values(18);
    let upstream = sample_values(12);
    let (_, cache) = head
        .forward(&en, input.clone(), 2, 3, true, &mut seeded_rng())
        .unwrap();

    let gradient = head.backward(&en, &cache, &upstream, 2, 3).unwrap();
    let numerical = numerical_gradient(&input, |values| {
        let (output, _) = head
            .forward(&en, values.to_vec(), 2, 3, true, &mut seeded_rng())
            .unwrap();
        dot(&output, &upstream)
    });

    close(&gradient, &numerical, 1e-6);
    check_parameter_gradients(&mut head, Head::parameters, Head::parameters_mut, |model| {
        let (output, _) = model
            .forward(&en, input.clone(), 2, 3, true, &mut seeded_rng())
            .unwrap();
        dot(&output, &upstream)
    });
}

#[test]
fn single_token_head_has_zero_query_and_key_gradients() {
    let en = Naive;
    let mut head = Head::new(&en, 3, 2, 0.0, &mut seeded_rng()).unwrap();
    set_parameters(
        head.parameters_mut(),
        &[&[0.1; 6], &[-0.2; 6], &[1.0, -1.0, 2.0, 0.5, -2.0, 3.0]],
    );
    let input = vec![1.0, 2.0, -1.0, -2.0, 0.0, 3.0];
    let (output, cache) = head
        .forward(&en, input, 2, 1, false, &mut seeded_rng())
        .unwrap();

    let gradient = head
        .backward(&en, &cache, &vec![2.0, -1.0, -3.0, 4.0], 2, 1)
        .unwrap();

    close(&output, &[7.0, -3.0, -8.0, 11.0], 0.0);
    close(&cache.probabilities, &[1.0; 2], 0.0);
    close(&gradient, &[3.0, 3.5, -7.0, -7.0, -4.0, 18.0], 0.0);
    close(&head.query.parameters()[0].gradients, &[0.0; 6], 0.0);
    close(&head.key.parameters()[0].gradients, &[0.0; 6], 0.0);
    close(
        &head.value.parameters()[0].gradients,
        &[8.0, -9.0, 4.0, -2.0, -11.0, 13.0],
        0.0,
    );
}

#[test]
fn attention_concatenates_one_two_three_and_four_heads_in_column_order() {
    let en = Naive;
    for head_count in [1, 2, 3, 4] {
        let channels = 2 * head_count;
        let config = model_config(channels, head_count, 0.0);
        let mut attention = Attention::new(&en, &config, &mut seeded_rng()).unwrap();
        for parameter in attention.parameters_mut() {
            parameter.values.fill(0.0);
        }
        for (index, head) in attention.heads.iter_mut().enumerate() {
            let mut parameters = head.value.parameters_mut();
            parameters[0].values[(2 * index) * 2] = 1.0;
            parameters[0].values[(2 * index + 1) * 2 + 1] = 1.0;
        }
        for channel in 0..channels {
            attention.projection.parameters_mut()[0].values[channel * channels + channel] = 1.0;
        }
        let input: Vec<f64> = (0..4 * channels).map(|index| index as f64 + 1.0).collect();
        let mut expected = input.clone();
        for batch in 0..2 {
            for channel in 0..channels {
                let first = batch * 2 * channels + channel;
                expected[first + channels] = (input[first] + input[first + channels]) / 2.0;
            }
        }

        let (output, cache) = attention
            .forward(&en, &input, 2, 2, false, &mut seeded_rng())
            .unwrap();

        close(&output, &expected, 1e-12);
        close(&cache.concatenated, &expected, 1e-12);
        close(&cache.mask, &vec![1.0; input.len()], 0.0);
        assert_eq!(cache.heads.len(), head_count);
        for head_cache in &cache.heads {
            close(&head_cache.input, &input, 0.0);
        }
    }
}

#[test]
fn attention_backward_matches_finite_differences_with_dropout() {
    let en = Naive;
    for head_count in [1, 3] {
        let config = model_config(3, head_count, 0.25);
        let mut attention = Attention::new(&en, &config, &mut seeded_rng()).unwrap();
        initialize_parameters(attention.parameters_mut());
        let input = sample_values(12);
        let upstream = sample_values(12);
        let (_, cache) = attention
            .forward(&en, &input, 2, 2, true, &mut seeded_rng())
            .unwrap();

        let gradient = attention.backward(&en, &cache, &upstream, 2, 2).unwrap();
        let numerical = numerical_gradient(&input, |values| {
            let (output, _) = attention
                .forward(&en, &values.to_vec(), 2, 2, true, &mut seeded_rng())
                .unwrap();
            dot(&output, &upstream)
        });

        close(&gradient, &numerical, 1e-6);
        check_parameter_gradients(
            &mut attention,
            Attention::parameters,
            Attention::parameters_mut,
            |model| {
                let (output, _) = model
                    .forward(&en, &input, 2, 2, true, &mut seeded_rng())
                    .unwrap();
                dot(&output, &upstream)
            },
        );
    }
}

#[test]
fn feed_forward_preserves_pre_relu_values_and_matches_hand_calculations() {
    let en = Naive;
    let model = feed_forward_fixture(0.5);
    let input = vec![2.0, -3.0, -1.0, 4.0];

    let (output, cache) = model
        .forward(&en, input.clone(), 2, false, &mut seeded_rng())
        .unwrap();

    close(&output, &[3.0, 5.0, 2.5, 2.5], 0.0);
    close(&cache.input, &input, 0.0);
    close(
        &cache.pre_relu,
        &[
            2.5, -3.5, -2.0, 3.0, 2.0, -3.0, -2.0, 3.0, -0.5, 3.5, 1.0, -4.0, -1.0, 4.0, 1.0, -4.0,
        ],
        0.0,
    );
    close(
        &cache.activated,
        &[
            2.5, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 3.0, 0.0, 3.5, 1.0, 0.0, 0.0, 4.0, 1.0, 0.0,
        ],
        0.0,
    );
    close(&cache.mask, &[1.0; 4], 0.0);
}

#[test]
fn feed_forward_backward_matches_finite_differences_with_dropout() {
    let en = Naive;
    let mut model = feed_forward_fixture(0.25);
    initialize_parameters(model.project.parameters_mut());
    let input = vec![2.0, -3.0, -1.0, 4.0, 0.75, 1.25, -2.0, -0.75];
    let upstream = sample_values(8);
    let (_, cache) = model
        .forward(&en, input.clone(), 4, true, &mut seeded_rng())
        .unwrap();
    assert!(cache.pre_relu.iter().all(|value| value.abs() > 1e-3));

    let gradient = model.backward(&en, &cache, &upstream, 4).unwrap();
    let numerical = numerical_gradient(&input, |values| {
        let (output, _) = model
            .forward(&en, values.to_vec(), 4, true, &mut seeded_rng())
            .unwrap();
        dot(&output, &upstream)
    });

    close(&gradient, &numerical, 1e-6);
    check_parameter_gradients(
        &mut model,
        FeedForward::parameters,
        FeedForward::parameters_mut,
        |model| {
            let (output, _) = model
                .forward(&en, input.clone(), 4, true, &mut seeded_rng())
                .unwrap();
            dot(&output, &upstream)
        },
    );
}

#[test]
fn block_zero_branches_preserve_residual_values_and_gradients() {
    let en = Naive;
    let config = model_config(3, 1, 0.0);
    let mut block = Block::new(&en, &config, &mut seeded_rng()).unwrap();
    zero_block_branches(&mut block);
    let input = sample_values(12);
    let upstream = sample_values(12);
    let (output, cache) = block
        .forward(&en, input.clone(), 2, 2, false, &mut seeded_rng())
        .unwrap();

    let gradient = block.backward(&en, &cache, &upstream, 2, 2).unwrap();

    close(&output, &input, 0.0);
    close(&cache.input, &input, 0.0);
    close(&cache.after_attention, &input, 0.0);
    close(&gradient, &upstream, 0.0);
    close(&upstream, &sample_values(12), 0.0);

    let gradient = block.backward(&en, &cache, &vec![0.0; 12], 2, 2).unwrap();

    close(&gradient, &[0.0; 12], 0.0);
    for parameter in block.parameters() {
        close(
            &parameter.gradients,
            &vec![0.0; parameter.values.len()],
            0.0,
        );
    }
}

#[test]
fn block_preserves_cache_before_the_second_residual_addition() {
    let en = Naive;
    let config = model_config(3, 1, 0.0);
    let mut block = Block::new(&en, &config, &mut seeded_rng()).unwrap();
    zero_block_branches(&mut block);
    block.attention.projection.parameters_mut()[1]
        .values
        .copy_from_slice(&[0.5, -0.25, 1.0]);
    block.feed_forward.project.parameters_mut()[1]
        .values
        .copy_from_slice(&[-1.0, 0.75, 0.25]);
    let input = vec![1.0, 2.0, 3.0, -2.0, 0.0, 4.0];

    let (mut output, cache) = block
        .forward(&en, input.clone(), 1, 2, false, &mut seeded_rng())
        .unwrap();

    close(&output, &[0.5, 2.5, 4.25, -2.5, 0.5, 5.25], 0.0);
    close(
        &cache.after_attention,
        &[1.5, 1.75, 4.0, -1.5, -0.25, 5.0],
        0.0,
    );
    output.fill(99.0);
    close(&cache.input, &input, 0.0);
    close(
        &cache.after_attention,
        &[1.5, 1.75, 4.0, -1.5, -0.25, 5.0],
        0.0,
    );
}

#[test]
fn block_backward_matches_finite_differences_in_training_and_evaluation() {
    let en = Naive;
    for training in [false, true] {
        let config = model_config(3, 3, 0.25);
        let mut block = Block::new(&en, &config, &mut seeded_rng()).unwrap();
        initialize_parameters(block.parameters_mut());
        set_parameters(
            block.norm1.parameters_mut(),
            &[&[0.8, 1.1, 0.9], &[0.1, -0.2, 0.05]],
        );
        set_parameters(
            block.norm2.parameters_mut(),
            &[&[0.9, 1.2, 0.7], &[0.2, 0.0, -0.1]],
        );
        for (index, value) in block.feed_forward.expand.parameters_mut()[1]
            .values
            .iter_mut()
            .enumerate()
        {
            *value = if index % 2 == 0 { 2.0 } else { -2.0 };
        }
        let input = sample_values(12);
        let upstream = sample_values(12);
        let (_, cache) = block
            .forward(&en, input.clone(), 2, 2, training, &mut seeded_rng())
            .unwrap();
        assert!(
            cache
                .feed_forward
                .pre_relu
                .iter()
                .all(|value| value.abs() > 1e-3)
        );

        let gradient = block.backward(&en, &cache, &upstream, 2, 2).unwrap();
        let numerical = numerical_gradient(&input, |values| {
            let (output, _) = block
                .forward(&en, values.to_vec(), 2, 2, training, &mut seeded_rng())
                .unwrap();
            dot(&output, &upstream)
        });

        close(&gradient, &numerical, 1e-6);
        check_parameter_gradients(
            &mut block,
            Block::parameters,
            Block::parameters_mut,
            |model| {
                let (output, _) = model
                    .forward(&en, input.clone(), 2, 2, training, &mut seeded_rng())
                    .unwrap();
                dot(&output, &upstream)
            },
        );
    }
}

#[test]
fn disabled_dropout_preserves_rng_and_matches_evaluation() {
    let en = Naive;
    let input = sample_values(12);
    let mut outputs = Vec::new();
    for (training, dropout) in [(false, 0.5), (true, 0.0)] {
        let config = model_config(3, 3, dropout);
        let block = Block::new(&en, &config, &mut seeded_rng()).unwrap();
        let mut rng = seeded_rng();
        let mut untouched_rng = seeded_rng();

        let (output, cache) = block
            .forward(&en, input.clone(), 2, 2, training, &mut rng)
            .unwrap();

        close(&cache.attention.mask, &[1.0; 12], 0.0);
        close(&cache.feed_forward.mask, &[1.0; 12], 0.0);
        for head_cache in &cache.attention.heads {
            close(&head_cache.mask, &[1.0; 8], 0.0);
        }
        assert_eq!(rng.random::<u64>(), untouched_rng.random::<u64>());
        outputs.push(output);
    }
    close(&outputs[0], &outputs[1], 0.0);
}

#[test]
fn training_dropout_is_reproducible_and_uses_inverted_masks() {
    let en = Naive;
    let config = model_config(3, 3, 0.5);
    let block = Block::new(&en, &config, &mut seeded_rng()).unwrap();
    let input = sample_values(12);

    let (first, first_cache) = block
        .forward(&en, input.clone(), 2, 2, true, &mut seeded_rng())
        .unwrap();
    let (second, second_cache) = block
        .forward(&en, input, 2, 2, true, &mut seeded_rng())
        .unwrap();

    close(&first, &second, 0.0);
    let mut first_masks = vec![&first_cache.attention.mask, &first_cache.feed_forward.mask];
    let mut second_masks = vec![
        &second_cache.attention.mask,
        &second_cache.feed_forward.mask,
    ];
    first_masks.extend(first_cache.attention.heads.iter().map(|head| &head.mask));
    second_masks.extend(second_cache.attention.heads.iter().map(|head| &head.mask));
    for (first_mask, second_mask) in first_masks.iter().zip(&second_masks) {
        close(first_mask, second_mask, 0.0);
        assert!(
            first_mask
                .iter()
                .all(|value| *value == 0.0 || *value == 2.0)
        );
    }
}

fn seeded_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn model_config(channels: usize, heads: usize, dropout: f64) -> ModelConfig {
    ModelConfig {
        block_size: 3,
        n_embd: channels,
        n_head: heads,
        n_layer: 1,
        dropout,
    }
}

fn sample_values(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| ((index * 7 % 19) as f64 - 9.0) / 7.0)
        .collect()
}

fn set_parameters(parameters: Vec<&mut Parameter<Naive>>, values: &[&[f64]]) {
    assert_eq!(parameters.len(), values.len());
    for (parameter, values) in parameters.into_iter().zip(values) {
        parameter.values.copy_from_slice(values);
    }
}

fn initialize_parameters(parameters: Vec<&mut Parameter<Naive>>) {
    for (parameter_index, parameter) in parameters.into_iter().enumerate() {
        for (index, value) in parameter.values.iter_mut().enumerate() {
            *value = (((index + 3 * parameter_index) % 11) as f64 - 5.0) * 0.07;
        }
        // Backward must overwrite existing gradients, not accumulate into them.
        parameter.gradients.fill(9.0);
    }
}

fn feed_forward_fixture(dropout: f64) -> FeedForward<Naive> {
    let mut model = FeedForward::new(&Naive, 2, dropout, &mut seeded_rng()).unwrap();
    set_parameters(
        model.expand.parameters_mut(),
        &[
            &[
                1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
            ],
            &[0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    );
    set_parameters(
        model.project.parameters_mut(),
        &[
            &[
                1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            &[0.5, -1.0],
        ],
    );
    model
}

fn zero_block_branches(block: &mut Block<Naive>) {
    for parameter in block.attention.parameters_mut() {
        parameter.values.fill(0.0);
    }
    for parameter in block.feed_forward.parameters_mut() {
        parameter.values.fill(0.0);
    }
}

fn check_parameter_gradients<M>(
    model: &mut M,
    parameters: for<'a> fn(&'a M) -> Vec<&'a Parameter<Naive>>,
    parameters_mut: for<'a> fn(&'a mut M) -> Vec<&'a mut Parameter<Naive>>,
    objective: impl Fn(&M) -> f64,
) {
    let analytic: Vec<_> = parameters(model)
        .iter()
        .map(|parameter| {
            assert_eq!(parameter.values.len(), parameter.gradients.len());
            parameter.gradients.clone()
        })
        .collect();

    for (index, gradient) in analytic.iter().enumerate() {
        let values = parameters(model)[index].values.clone();
        let numerical = numerical_gradient(&values, |perturbed| {
            parameters_mut(model)[index]
                .values
                .copy_from_slice(perturbed);
            objective(model)
        });
        parameters_mut(model)[index].values.copy_from_slice(&values);
        close(gradient, &numerical, 1e-6);
    }
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs()
                    <= tolerance * (1.0 + actual.abs().max(expected.abs())),
            "index {index}: actual {actual}, expected {expected}"
        );
    }
}

fn numerical_gradient(input: &[f64], mut objective: impl FnMut(&[f64]) -> f64) -> Vec<f64> {
    let epsilon = 1e-6;
    (0..input.len())
        .map(|index| {
            let mut plus = input.to_vec();
            let mut minus = input.to_vec();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            (objective(&plus) - objective(&minus)) / (2.0 * epsilon)
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
