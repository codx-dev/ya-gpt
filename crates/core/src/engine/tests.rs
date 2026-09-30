use crate::engine::Engine;

fn rectangular_matrix_multiplication_and_transpose<E: Engine>(en: &E) {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    close(
        &en.matmul(&a, &b, 2, 3, 2),
        &[58.0, 64.0, 139.0, 154.0],
        0.0,
    );
    close(
        &en.transpose(&a, 2, 3),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        0.0,
    );
}

fn softmax_is_stable_and_masks_future_positions<E: Engine>(en: &E) {
    close(&en.softmax(&[1000.0, 1000.0]), &[0.5, 0.5], 1e-12);
    close(&en.softmax(&[-1000.0, -1000.0]), &[0.5, 0.5], 1e-12);

    let masked = en.causal_mask(&[0.0; 9], 3);

    close(
        &en.softmax_rows(&masked, 3, 3),
        &[
            1.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
        1e-12,
    );

    close(
        &en.causal_mask_backward(&[1.0; 9], 3),
        &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        0.0,
    );
}

fn normalization_and_loss_match_hand_calculations<E: Engine>(en: &E) {
    close(
        &en.layer_norm(&[1.0, 3.0, 2.0, 2.0], &[2.0, 3.0], &[0.5, -0.5], 2, 2, 1.0),
        &[
            0.5 - 2.0 / 2.0_f64.sqrt(),
            -0.5 + 3.0 / 2.0_f64.sqrt(),
            0.5,
            -0.5,
        ],
        1e-12,
    );
    close(
        &[en.cross_entropy(&[0.0; 6], &[0, 2], 2, 3)],
        &[3.0_f64.ln()],
        1e-12,
    );
    close(
        &[en.cross_entropy(&[1000.0, -1000.0], &[1], 1, 2)],
        &[2000.0],
        1e-12,
    );
    close(
        &[en.cross_entropy(&[1000.0, 1000.0], &[0], 1, 2)],
        &[2.0_f64.ln()],
        1e-12,
    );
}

fn basic_operations_and_head_layout<E: Engine>(en: &E) {
    close(&en.add(&[1.0, 2.0], &[3.0, 4.0]), &[4.0, 6.0], 0.0);
    close(&en.scale(&[1.0, 2.0], 3.0), &[3.0, 6.0], 0.0);
    close(
        &en.add_bias(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0], 2, 2),
        &[6.0, 8.0, 8.0, 10.0],
        0.0,
    );
    close(&en.sum_rows(&[1.0, 2.0, 3.0, 4.0], 2, 2), &[4.0, 6.0], 0.0);
    let joined = en.concat_columns(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
    close(&joined, &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0], 0.0);
    close(
        &en.slice_columns(&joined, 2, 4, 2, 2),
        &[5.0, 6.0, 7.0, 8.0],
        0.0,
    );
}

fn repeated_embeddings_accumulate_gradients<E: Engine>(en: &E) {
    let tokens = [1, 0, 1];
    close(
        &en.embedding(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &tokens, 3, 2),
        &[3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        0.0,
    );
    close(
        &en.embedding_backward(&tokens, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2),
        &[3.0, 4.0, 6.0, 8.0, 0.0, 0.0],
        0.0,
    );
}

fn matmul_gradients_match_finite_differences<E: Engine>(en: &E) {
    let a = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1];
    let b = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2];
    let dy = [0.4, -0.7, 0.2, 0.8];
    let (da, db) = en.matmul_backward(&a, &b, &dy, 2, 3, 2);
    close(
        &da,
        &numerical_gradient(&a, |x| dot(&en.matmul(x, &b, 2, 3, 2), &dy)),
        1e-8,
    );
    close(
        &db,
        &numerical_gradient(&b, |x| dot(&en.matmul(&a, x, 2, 3, 2), &dy)),
        1e-8,
    );
}

fn softmax_and_cross_entropy_gradients_match_finite_differences<E: Engine>(en: &E) {
    let logits = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1];
    let dy = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2];
    let probabilities = en.softmax_rows(&logits, 2, 3);
    let gradient = en.softmax_rows_backward(&probabilities, &dy, 2, 3);
    close(
        &gradient,
        &numerical_gradient(&logits, |x| dot(&en.softmax_rows(x, 2, 3), &dy)),
        1e-8,
    );
    close(
        &en.cross_entropy_backward(&logits, &[1, 2], 2, 3),
        &numerical_gradient(&logits, |x| en.cross_entropy(x, &[1, 2], 2, 3)),
        1e-8,
    );
}

fn all_layer_norm_gradients_match_finite_differences<E: Engine>(en: &E) {
    let x = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1];
    let gamma = [0.8, 1.2, -0.5];
    let beta = [0.2, -0.1, 0.4];
    let dy = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2];
    let (dx, dg, db) = en.layer_norm_backward(&x, &gamma, &dy, 2, 3, 1e-5);
    close(
        &dx,
        &numerical_gradient(&x, |v| {
            dot(&en.layer_norm(v, &gamma, &beta, 2, 3, 1e-5), &dy)
        }),
        1e-8,
    );
    close(
        &dg,
        &numerical_gradient(&gamma, |v| {
            dot(&en.layer_norm(&x, v, &beta, 2, 3, 1e-5), &dy)
        }),
        1e-8,
    );
    close(
        &db,
        &numerical_gradient(&beta, |v| {
            dot(&en.layer_norm(&x, &gamma, v, 2, 3, 1e-5), &dy)
        }),
        1e-8,
    );
}

fn relu_dropout_bias_and_embedding_gradients_match_finite_differences<E: Engine>(en: &E) {
    let x = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1];
    let dy = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2];
    let mask = [2.0, 0.0, 2.0, 0.0, 0.0, 2.0];
    close(
        &en.relu_backward(&x, &dy),
        &numerical_gradient(&x, |v| dot(&en.relu(v), &dy)),
        1e-8,
    );
    close(&en.relu_backward(&[0.0], &[1.0]), &[0.0], 0.0);
    close(
        &en.dropout(&dy, &mask),
        &numerical_gradient(&x, |v| dot(&en.dropout(v, &mask), &dy)),
        1e-8,
    );
    close(
        &en.sum_rows(&dy, 2, 3),
        &numerical_gradient(&[0.1, 0.2, 0.3], |v| dot(&en.add_bias(&x, v, 2, 3), &dy)),
        1e-8,
    );
    close(
        &en.embedding_backward(&[1, 0, 1], &dy, 3, 2),
        &numerical_gradient(&x, |v| dot(&en.embedding(v, &[1, 0, 1], 3, 2), &dy)),
        1e-8,
    );
}

pub fn full_suite<E: Engine>(engine: &E) {
    rectangular_matrix_multiplication_and_transpose(engine);
    softmax_is_stable_and_masks_future_positions(engine);
    normalization_and_loss_match_hand_calculations(engine);
    basic_operations_and_head_layout(engine);
    repeated_embeddings_accumulate_gradients(engine);
    matmul_gradients_match_finite_differences(engine);
    softmax_and_cross_entropy_gradients_match_finite_differences(engine);
    all_layer_norm_gradients_match_finite_differences(engine);
    relu_dropout_bias_and_embedding_gradients_match_finite_differences(engine);
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());

    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite()
                && e.is_finite()
                && (a - e).abs() <= tolerance * (1.0 + a.abs().max(e.abs())),
            "index {index}: actual {a}, expected {e}"
        );
    }
}

fn numerical_gradient(input: &[f64], objective: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let epsilon = 1e-6;
    (0..input.len())
        .map(|i| {
            let mut plus = input.to_vec();
            let mut minus = input.to_vec();
            plus[i] += epsilon;
            minus[i] -= epsilon;
            (objective(&plus) - objective(&minus)) / (2.0 * epsilon)
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
