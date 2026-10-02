use crate::engine::Engine;

fn buffer_all_finite_detects_nan_and_infinities<E: Engine>(en: &E) {
    let finite = en
        .buffer_from_slice([0.0, -1.0, f64::MIN, f64::MAX])
        .unwrap();
    assert!(en.buffer_all_finite(&finite).unwrap());
    assert!(en.buffer_all_finite(&en.zeroes(0).unwrap()).unwrap());

    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let buffer = en.buffer_from_slice([1.0, invalid, -2.0]).unwrap();
        assert!(!en.buffer_all_finite(&buffer).unwrap());
    }
}

fn adamw_in_place_matches_hand_calculations_across_steps<E: Engine>(en: &E) {
    let mut values = en.buffer_from_slice([1.0, -2.0, 3.0]).unwrap();
    let gradients = en.buffer_from_slice([2.0, -4.0, 0.0]).unwrap();
    let mut first = en.zeroes(3).unwrap();
    let mut second = en.zeroes(3).unwrap();

    en.adamw_in_place(
        &mut values,
        &gradients,
        &mut first,
        &mut second,
        0.1,
        0.2,
        0.5,
        0.5,
        1.0,
        1,
    )
    .unwrap();

    close(
        &en.buffer_to_vec(&values).unwrap(),
        &[0.98 - 1.0 / 15.0, -1.88, 2.94],
        1e-12,
    );
    close(&en.buffer_to_vec(&first).unwrap(), &[1.0, -2.0, 0.0], 1e-12);
    close(&en.buffer_to_vec(&second).unwrap(), &[2.0, 8.0, 0.0], 1e-12);
    close(
        &en.buffer_to_vec(&gradients).unwrap(),
        &[2.0, -4.0, 0.0],
        0.0,
    );

    let gradients = en.buffer_from_slice([0.0, 2.0, -1.0]).unwrap();
    en.adamw_in_place(
        &mut values,
        &gradients,
        &mut first,
        &mut second,
        0.1,
        0.2,
        0.5,
        0.5,
        1.0,
        2,
    )
    .unwrap();

    close(
        &en.buffer_to_vec(&values).unwrap(),
        &[
            (0.98 - 1.0 / 15.0) * 0.98 - (1.0 / 15.0) / ((4.0_f64 / 3.0).sqrt() + 1.0),
            -1.88 * 0.98,
            2.94 * 0.98 + (1.0 / 15.0) / ((2.0_f64 / 3.0).sqrt() + 1.0),
        ],
        1e-12,
    );
    close(&en.buffer_to_vec(&first).unwrap(), &[0.5, 0.0, -0.5], 1e-12);
    close(&en.buffer_to_vec(&second).unwrap(), &[1.0, 6.0, 0.5], 1e-12);
    close(
        &en.buffer_to_vec(&gradients).unwrap(),
        &[0.0, 2.0, -1.0],
        0.0,
    );
}

fn adamw_in_place_applies_weight_decay_with_zero_gradients<E: Engine>(en: &E) {
    let mut values = en.buffer_from_slice([2.0, -4.0, 0.5]).unwrap();
    let gradients = en.zeroes(3).unwrap();
    let mut first = en.zeroes(3).unwrap();
    let mut second = en.zeroes(3).unwrap();

    en.adamw_in_place(
        &mut values,
        &gradients,
        &mut first,
        &mut second,
        0.25,
        0.1,
        0.9,
        0.999,
        1e-8,
        1,
    )
    .unwrap();

    close(
        &en.buffer_to_vec(&values).unwrap(),
        &[1.95, -3.9, 0.4875],
        1e-12,
    );
    close(&en.buffer_to_vec(&first).unwrap(), &[0.0; 3], 0.0);
    close(&en.buffer_to_vec(&second).unwrap(), &[0.0; 3], 0.0);
}

fn adamw_in_place_rejects_non_finite_updates<E: Engine>(en: &E) {
    let mut values = en.buffer_from_slice([1.0]).unwrap();
    let gradients = en.buffer_from_slice([f64::MAX]).unwrap();
    let mut first = en.zeroes(1).unwrap();
    let mut second = en.zeroes(1).unwrap();

    assert!(
        en.adamw_in_place(
            &mut values,
            &gradients,
            &mut first,
            &mut second,
            0.1,
            0.01,
            0.9,
            0.999,
            1e-8,
            1,
        )
        .is_err()
    );
}

fn rectangular_matrix_multiplication_and_transpose<E: Engine>(en: &E) {
    let a = en
        .buffer_from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let b = en
        .buffer_from_slice([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        .unwrap();

    let mut out = en.zeroes(4).unwrap();
    en.matmul(&a, &b, 2, 3, 2, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[58.0, 64.0, 139.0, 154.0], 0.0);

    let mut out = en.zeroes(6).unwrap();
    en.transpose(&a, 2, 3, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 0.0);
}

fn transpose_in_place_matches_transpose_on_rectangular_inputs<E: Engine>(en: &E) {
    let values = [1.0, -2.0, 0.5, 3.0, -4.0, 6.0];
    let input = en.buffer_from_slice(values).unwrap();

    let mut expected = en.zeroes(values.len()).unwrap();
    en.transpose(&input, 2, 3, &mut expected).unwrap();
    let expected = en.buffer_to_vec(&expected).unwrap();

    let mut out = en.buffer_from_slice(values).unwrap();
    en.transpose_in_place(&mut out, 2, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0, 3.0, -2.0, -4.0, 0.5, 6.0], 0.0);
    close(&out, &expected, 0.0);
}

fn transpose_in_place_handles_square_matrices<E: Engine>(en: &E) {
    let mut out = en
        .buffer_from_slice([-2.0, 0.5, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 0.25])
        .unwrap();

    en.transpose_in_place(&mut out, 3, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &[-2.0, 4.0, -7.0, 0.5, -5.0, 8.0, 3.0, 6.0, 0.25],
        0.0,
    );
}

fn transpose_in_place_round_trip_restores_rectangular_input<E: Engine>(en: &E) {
    let values = [1.0, -2.0, 0.5, 3.0, -4.0, 6.0];
    let mut out = en.buffer_from_slice(values).unwrap();

    en.transpose_in_place(&mut out, 3, 2).unwrap();
    let transposed = en.buffer_to_vec(&out).unwrap();

    close(&transposed, &[1.0, 0.5, -4.0, -2.0, 3.0, 6.0], 0.0);

    en.transpose_in_place(&mut out, 2, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &values, 0.0);
}

fn transpose_in_place_handles_single_row_column_and_element<E: Engine>(en: &E) {
    let values = [-2.0, 0.5, 0.0, 3.0];
    let mut out = en.buffer_from_slice(values).unwrap();

    en.transpose_in_place(&mut out, 1, 4).unwrap();
    let transposed = en.buffer_to_vec(&out).unwrap();

    close(&transposed, &values, 0.0);

    en.transpose_in_place(&mut out, 4, 1).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &values, 0.0);

    let mut out = en.buffer_from_slice([-3.5]).unwrap();

    en.transpose_in_place(&mut out, 1, 1).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-3.5], 0.0);
}

fn softmax_is_stable_and_masks_future_positions<E: Engine>(en: &E) {
    let a = en.buffer_from_slice([1000.0, 1000.0]).unwrap();
    let b = en.buffer_from_slice([-1000.0, -1000.0]).unwrap();

    let mut out = en.zeroes(2).unwrap();
    en.softmax(&a, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.5, 0.5], 1e-12);

    let mut out = en.zeroes(2).unwrap();
    en.softmax(&b, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.5, 0.5], 1e-12);

    let c = en.buffer_from_slice([0.0; 9]).unwrap();
    let mut masked = en.zeroes(9).unwrap();
    en.causal_mask(&c, 3, &mut masked).unwrap();

    let mut out = en.zeroes(9).unwrap();
    en.softmax_rows(&masked, 3, 3, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
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

    let mut out = en.zeroes(9).unwrap();
    let e = en.buffer_from_slice([1.0; 9]).unwrap();
    en.causal_mask_backward(&e, 3, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], 0.0);
}

fn softmax_in_place_matches_softmax<E: Engine>(en: &E) {
    let values = [0.0, 3.0_f64.ln(), 2.0_f64.ln()];
    let input = en.buffer_from_slice(values).unwrap();

    let mut expected = en.zeroes(values.len()).unwrap();
    en.softmax(&input, &mut expected).unwrap();
    let expected = en.buffer_to_vec(&expected).unwrap();

    let mut out = en.buffer_from_slice(values).unwrap();
    en.softmax_in_place(&mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0 / 6.0, 0.5, 1.0 / 3.0], 1e-12);
    close(&out, &expected, 1e-12);
    close(&[out.iter().sum()], &[1.0], 1e-12);
}

fn softmax_in_place_is_stable_for_extreme_logits<E: Engine>(en: &E) {
    for offset in [-1000.0, 0.0, 1000.0] {
        let mut out = en
            .buffer_from_slice([offset, offset + 3.0_f64.ln()])
            .unwrap();

        en.softmax_in_place(&mut out).unwrap();
        let out = en.buffer_to_vec(&out).unwrap();

        close(&out, &[0.25, 0.75], 1e-12);
    }

    let mut out = en.buffer_from_slice([1000.0, -1000.0, 1000.0]).unwrap();

    en.softmax_in_place(&mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.5, 0.0, 0.5], 1e-12);
}

fn softmax_in_place_handles_masked_entries<E: Engine>(en: &E) {
    let mut out = en
        .buffer_from_slice([f64::NEG_INFINITY, 2.0_f64.ln(), 0.0, f64::NEG_INFINITY])
        .unwrap();

    en.softmax_in_place(&mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0], 1e-12);
    close(&[out[0], out[3]], &[0.0; 2], 0.0);

    let mut out = en
        .buffer_from_slice([f64::NEG_INFINITY, -1000.0, f64::NEG_INFINITY])
        .unwrap();

    en.softmax_in_place(&mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0, 1.0, 0.0], 0.0);
}

fn softmax_in_place_handles_uniform_and_single_element_inputs<E: Engine>(en: &E) {
    for value in [-1000.0, 0.0, 1000.0] {
        let mut out = en.buffer_from_slice([value; 4]).unwrap();

        en.softmax_in_place(&mut out).unwrap();
        let out = en.buffer_to_vec(&out).unwrap();

        close(&out, &[0.25; 4], 1e-12);
    }

    let mut out = en.buffer_from_slice([-3.5]).unwrap();

    en.softmax_in_place(&mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0], 0.0);
}

fn softmax_rows_in_place_matches_softmax_rows<E: Engine>(en: &E) {
    let values = [
        0.0,
        2.0_f64.ln(),
        3.0_f64.ln(),
        3.0_f64.ln(),
        0.0,
        2.0_f64.ln(),
    ];
    let input = en.buffer_from_slice(values).unwrap();

    let mut expected = en.zeroes(values.len()).unwrap();
    en.softmax_rows(&input, 2, 3, &mut expected).unwrap();
    let expected = en.buffer_to_vec(&expected).unwrap();

    let mut out = en.buffer_from_slice(values).unwrap();
    en.softmax_rows_in_place(&mut out, 2, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &[1.0 / 6.0, 1.0 / 3.0, 0.5, 0.5, 1.0 / 6.0, 1.0 / 3.0],
        1e-12,
    );
    close(&out, &expected, 1e-12);
}

fn softmax_rows_in_place_is_stable_for_extreme_logits<E: Engine>(en: &E) {
    let mut out = en
        .buffer_from_slice([
            1000.0,
            1000.0 + 3.0_f64.ln(),
            -1000.0,
            -1000.0 + 3.0_f64.ln(),
            1000.0,
            -1000.0,
        ])
        .unwrap();

    en.softmax_rows_in_place(&mut out, 3, 2).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.25, 0.75, 0.25, 0.75, 1.0, 0.0], 1e-12);
}

fn softmax_rows_in_place_handles_masked_entries<E: Engine>(en: &E) {
    let mut out = en
        .buffer_from_slice([
            f64::NEG_INFINITY,
            2.0_f64.ln(),
            0.0,
            3.0_f64.ln(),
            f64::NEG_INFINITY,
            0.0,
            f64::NEG_INFINITY,
            -1000.0,
            f64::NEG_INFINITY,
        ])
        .unwrap();

    en.softmax_rows_in_place(&mut out, 3, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &[0.0, 2.0 / 3.0, 1.0 / 3.0, 0.75, 0.0, 0.25, 0.0, 1.0, 0.0],
        1e-12,
    );
    close(&[out[0], out[4], out[6], out[8]], &[0.0; 4], 0.0);
}

fn softmax_rows_in_place_handles_single_row_and_column<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([1000.0; 4]).unwrap();

    en.softmax_rows_in_place(&mut out, 1, 4).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.25; 4], 1e-12);

    let mut out = en.buffer_from_slice([-1000.0, 0.0, 1000.0]).unwrap();

    en.softmax_rows_in_place(&mut out, 3, 1).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0; 3], 0.0);

    let mut out = en.buffer_from_slice([-3.5]).unwrap();

    en.softmax_rows_in_place(&mut out, 1, 1).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0], 0.0);
}

fn causal_mask_in_place_matches_causal_mask<E: Engine>(en: &E) {
    let values = [-2.0, 0.5, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 0.0];
    let input = en.buffer_from_slice(values).unwrap();

    let mut expected = en.zeroes(values.len()).unwrap();
    en.causal_mask(&input, 3, &mut expected).unwrap();
    let expected = en.buffer_to_vec(&expected).unwrap();

    let mut out = en.buffer_from_slice(values).unwrap();
    en.causal_mask_in_place(&mut out, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    assert_eq!(
        out.as_slice(),
        &[
            -2.0,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            4.0,
            -5.0,
            f64::NEG_INFINITY,
            -7.0,
            8.0,
            0.0,
        ],
    );
    assert_eq!(out, expected);
}

fn causal_mask_in_place_is_idempotent<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([2.0, -4.0, 0.5, -1.0]).unwrap();

    en.causal_mask_in_place(&mut out, 2).unwrap();
    let first = en.buffer_to_vec(&out).unwrap();

    assert_eq!(first.as_slice(), &[2.0, f64::NEG_INFINITY, 0.5, -1.0]);

    en.causal_mask_in_place(&mut out, 2).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    assert_eq!(out, first);
}

fn causal_mask_in_place_preserves_single_position<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([-3.5]).unwrap();

    en.causal_mask_in_place(&mut out, 1).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-3.5], 0.0);
}

fn causal_mask_in_place_masks_future_softmax_probabilities<E: Engine>(en: &E) {
    let mut masked = en
        .buffer_from_slice([
            2.0,
            1000.0,
            2000.0,
            0.0,
            2.0_f64.ln(),
            1000.0,
            2.0_f64.ln(),
            0.0,
            0.0,
        ])
        .unwrap();

    en.causal_mask_in_place(&mut masked, 3).unwrap();

    let mut out = en.zeroes(9).unwrap();
    en.softmax_rows(&masked, 3, 3, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &[1.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 0.5, 0.25, 0.25],
        1e-12,
    );
}

fn causal_mask_backward_in_place_matches_causal_mask_backward<E: Engine>(en: &E) {
    let values = [-2.0, 0.5, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 0.25];
    let input = en.buffer_from_slice(values).unwrap();

    let mut expected = en.zeroes(values.len()).unwrap();
    en.causal_mask_backward(&input, 3, &mut expected).unwrap();
    let expected = en.buffer_to_vec(&expected).unwrap();

    let mut out = en.buffer_from_slice(values).unwrap();
    en.causal_mask_backward_in_place(&mut out, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &[-2.0, 0.0, 0.0, 4.0, -5.0, 0.0, -7.0, 8.0, 0.25],
        0.0,
    );
    close(&out, &expected, 0.0);
}

fn causal_mask_backward_in_place_is_idempotent<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([2.0, -4.0, 0.5, -1.0]).unwrap();

    en.causal_mask_backward_in_place(&mut out, 2).unwrap();
    let first = en.buffer_to_vec(&out).unwrap();

    close(&first, &[2.0, 0.0, 0.5, -1.0], 0.0);

    en.causal_mask_backward_in_place(&mut out, 2).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &first, 0.0);
}

fn causal_mask_backward_in_place_handles_single_position_and_zero_gradients<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([-3.5]).unwrap();

    en.causal_mask_backward_in_place(&mut out, 1).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-3.5], 0.0);

    let mut out = en.zeroes(9).unwrap();

    en.causal_mask_backward_in_place(&mut out, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0; 9], 0.0);
}

fn causal_mask_backward_in_place_matches_finite_differences<E: Engine>(en: &E) {
    let x = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1, -0.4, 0.6, 0.8];
    let dy = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2, 0.5, -0.7, 0.4];

    let input = en.buffer_from_slice(x).unwrap();
    let d_output = en.buffer_from_slice(dy).unwrap();

    let mut masked = en.zeroes(9).unwrap();
    en.causal_mask(&input, 3, &mut masked).unwrap();

    let mut probabilities = en.zeroes(9).unwrap();
    en.softmax_rows(&masked, 3, 3, &mut probabilities).unwrap();

    let mut out = en.zeroes(9).unwrap();
    en.softmax_rows_backward(&probabilities, &d_output, 3, 3, &mut out)
        .unwrap();
    en.causal_mask_backward_in_place(&mut out, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &numerical_gradient(&x, |v| {
            let v = en.buffer_from_slice(v).unwrap();
            let mut masked = en.zeroes(9).unwrap();
            en.causal_mask(&v, 3, &mut masked).unwrap();
            let mut out = en.zeroes(9).unwrap();
            en.softmax_rows(&masked, 3, 3, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );
    close(&[out[1], out[2], out[5]], &[0.0; 3], 0.0);
}

fn normalization_and_loss_match_hand_calculations<E: Engine>(en: &E) {
    let aa = en.buffer_from_slice([1.0, 3.0, 2.0, 2.0]).unwrap();
    let ab = en.buffer_from_slice([2.0, 3.0]).unwrap();
    let ac = en.buffer_from_slice([0.5, -0.5]).unwrap();

    let mut out = en.zeroes(4).unwrap();
    en.layer_norm(&aa, &ab, &ac, 2, 2, 1.0, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &[
            0.5 - 2.0 / 2.0_f64.sqrt(),
            -0.5 + 3.0 / 2.0_f64.sqrt(),
            0.5,
            -0.5,
        ],
        1e-12,
    );

    let ba = en.buffer_from_slice([0.0; 6]).unwrap();
    let out = en.cross_entropy(&ba, &[0, 2], 2, 3).unwrap();

    close(&[out], &[3.0_f64.ln()], 1e-12);

    let ca = en.buffer_from_slice([1000.0, -1000.0]).unwrap();
    let out = en.cross_entropy(&ca, &[1], 1, 2).unwrap();

    close(&[out], &[2000.0], 1e-12);

    let da = en.buffer_from_slice([1000.0, 1000.0]).unwrap();
    let out = en.cross_entropy(&da, &[0], 1, 2).unwrap();

    close(&[out], &[2.0_f64.ln()], 1e-12);
}

fn basic_operations_and_head_layout<E: Engine>(en: &E) {
    let aa = en.buffer_from_slice([1.0, 2.0]).unwrap();
    let ab = en.buffer_from_slice([3.0, 4.0]).unwrap();

    let mut out = en.zeroes(2).unwrap();
    en.add(&aa, &ab, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[4.0, 6.0], 0.0);

    let ba = en.buffer_from_slice([1.0, 2.0]).unwrap();

    let mut out = en.zeroes(2).unwrap();
    en.scale(&ba, 3.0, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[3.0, 6.0], 0.0);

    let ca = en.buffer_from_slice([1.0, 2.0, 3.0, 4.0]).unwrap();
    let cb = en.buffer_from_slice([5.0, 6.0]).unwrap();

    let mut out = en.zeroes(4).unwrap();
    en.add_bias(&ca, &cb, 2, 2, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[6.0, 8.0, 8.0, 10.0], 0.0);

    let da = en.buffer_from_slice([1.0, 2.0, 3.0, 4.0]).unwrap();

    let mut out = en.zeroes(2).unwrap();
    en.sum_rows(&da, 2, 2, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[4.0, 6.0], 0.0);

    let ea = en.buffer_from_slice([1.0, 2.0, 3.0, 4.0]).unwrap();
    let eb = en.buffer_from_slice([5.0, 6.0, 7.0, 8.0]).unwrap();

    let mut out = en.zeroes(8).unwrap();
    en.concat_columns(&ea, &eb, 2, 2, 2, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0], 0.0);

    let fa = en.buffer_from_slice(&out).unwrap();

    let mut out = en.zeroes(4).unwrap();
    en.slice_columns(&fa, 2, 4, 2, 2, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[5.0, 6.0, 7.0, 8.0], 0.0);
}

fn add_in_place_matches_add<E: Engine>(en: &E) {
    let values = [-2.0, 0.0, 0.5, 3.0, -4.0];
    let other = [1.0, -3.0, -0.5, 0.25, 4.0];
    let input = en.buffer_from_slice(values).unwrap();
    let b = en.buffer_from_slice(other).unwrap();

    let mut expected = en.zeroes(values.len()).unwrap();
    en.add(&input, &b, &mut expected).unwrap();
    let expected = en.buffer_to_vec(&expected).unwrap();

    let mut out = en.buffer_from_slice(values).unwrap();
    en.add_in_place(&mut out, &b).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-1.0, -3.0, 0.0, 3.25, 0.0], 0.0);
    close(&out, &expected, 0.0);

    let b = en.buffer_to_vec(&b).unwrap();

    close(&b, &other, 0.0);
}

fn add_in_place_handles_zero_operands<E: Engine>(en: &E) {
    let values = [-1.5, 0.0, 2.0, 0.25];
    let mut out = en.buffer_from_slice(values).unwrap();
    let b = en.zeroes(values.len()).unwrap();

    en.add_in_place(&mut out, &b).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &values, 0.0);

    let mut out = en.zeroes(values.len()).unwrap();
    let b = en.buffer_from_slice(values).unwrap();

    en.add_in_place(&mut out, &b).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &values, 0.0);
}

fn add_in_place_accumulates_repeated_calls<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([1.0, -2.0, 0.5]).unwrap();
    let b = en.buffer_from_slice([0.25, 2.0, -0.5]).unwrap();

    en.add_in_place(&mut out, &b).unwrap();
    let values = en.buffer_to_vec(&out).unwrap();

    close(&values, &[1.25, 0.0, 0.0], 0.0);

    en.add_in_place(&mut out, &b).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.5, 2.0, -0.5], 0.0);
}

fn add_in_place_handles_single_element_and_empty_buffers<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([-3.0]).unwrap();
    let b = en.buffer_from_slice([1.5]).unwrap();

    en.add_in_place(&mut out, &b).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-1.5], 0.0);

    let mut out = en.buffer_from_slice([]).unwrap();
    let b = en.buffer_from_slice([]).unwrap();

    en.add_in_place(&mut out, &b).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[], 0.0);
}

fn dropout_in_place_matches_dropout<E: Engine>(en: &E) {
    let values = [-2.0, 0.0, 0.5, 3.0, -4.0];
    let mask_values = [2.0, 0.0, 2.0, 0.0, 2.0];
    let input = en.buffer_from_slice(values).unwrap();
    let mask = en.buffer_from_slice(mask_values).unwrap();

    let mut expected = en.zeroes(values.len()).unwrap();
    en.dropout(&input, &mask, &mut expected).unwrap();
    let expected = en.buffer_to_vec(&expected).unwrap();

    let mut out = en.buffer_from_slice(values).unwrap();
    en.dropout_in_place(&mut out, &mask).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-4.0, 0.0, 1.0, 0.0, -8.0], 0.0);
    close(&out, &expected, 0.0);

    let mask = en.buffer_to_vec(&mask).unwrap();

    close(&mask, &mask_values, 0.0);
}

fn dropout_in_place_handles_zero_and_identity_masks<E: Engine>(en: &E) {
    let values = [-1.5, 0.0, 2.0, 0.25];
    let mut out = en.buffer_from_slice(values).unwrap();
    let mask = en.buffer_from_slice([1.0; 4]).unwrap();

    en.dropout_in_place(&mut out, &mask).unwrap();
    let unchanged = en.buffer_to_vec(&out).unwrap();

    close(&unchanged, &values, 0.0);

    let mask = en.zeroes(values.len()).unwrap();

    en.dropout_in_place(&mut out, &mask).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0; 4], 0.0);
}

fn dropout_in_place_composes_repeated_calls<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([-2.0, 0.5, 3.0, -4.0]).unwrap();
    let mask = en.buffer_from_slice([2.0, 0.0, 2.0, 0.0]).unwrap();

    en.dropout_in_place(&mut out, &mask).unwrap();
    let values = en.buffer_to_vec(&out).unwrap();

    close(&values, &[-4.0, 0.0, 6.0, 0.0], 0.0);

    en.dropout_in_place(&mut out, &mask).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-8.0, 0.0, 12.0, 0.0], 0.0);
}

fn dropout_in_place_handles_single_element_and_empty_buffers<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([-3.0]).unwrap();
    let mask = en.buffer_from_slice([2.0]).unwrap();

    en.dropout_in_place(&mut out, &mask).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-6.0], 0.0);

    let mut out = en.buffer_from_slice([]).unwrap();
    let mask = en.buffer_from_slice([]).unwrap();

    en.dropout_in_place(&mut out, &mask).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[], 0.0);
}

fn scale_in_place_matches_scale<E: Engine>(en: &E) {
    let values = [-2.0, 0.0, 0.5, 3.0, -4.0];
    let input = en.buffer_from_slice(values).unwrap();

    let mut expected = en.zeroes(values.len()).unwrap();
    en.scale(&input, -0.5, &mut expected).unwrap();
    let expected = en.buffer_to_vec(&expected).unwrap();

    let mut out = en.buffer_from_slice(values).unwrap();
    en.scale_in_place(&mut out, -0.5).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0, 0.0, -0.25, -1.5, 2.0], 0.0);
    close(&out, &expected, 0.0);
}

fn scale_in_place_handles_zero_and_identity_factors<E: Engine>(en: &E) {
    let values = [1.5, -2.0, 0.0, 4.0];
    let mut out = en.buffer_from_slice(values).unwrap();

    en.scale_in_place(&mut out, 1.0).unwrap();
    let unchanged = en.buffer_to_vec(&out).unwrap();

    close(&unchanged, &values, 0.0);

    en.scale_in_place(&mut out, 0.0).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0; 4], 0.0);
}

fn scale_in_place_composes_repeated_calls<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([-2.0, 0.0, 0.5, 3.0]).unwrap();

    en.scale_in_place(&mut out, 2.0).unwrap();
    let values = en.buffer_to_vec(&out).unwrap();

    close(&values, &[-4.0, 0.0, 1.0, 6.0], 0.0);

    en.scale_in_place(&mut out, -0.5).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[2.0, 0.0, -0.5, -3.0], 0.0);
}

fn scale_in_place_handles_single_element_and_empty_buffers<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([-3.0]).unwrap();

    en.scale_in_place(&mut out, -0.5).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.5], 0.0);

    let mut out = en.buffer_from_slice([]).unwrap();

    en.scale_in_place(&mut out, 2.0).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[], 0.0);
}

fn add_bias_in_place_matches_add_bias_on_rectangular_inputs<E: Engine>(en: &E) {
    let values = [1.0, -2.0, 0.5, 3.0, 4.0, -0.5];
    let input = en.buffer_from_slice(values).unwrap();
    let bias = en.buffer_from_slice([-0.5, 2.0, 0.0]).unwrap();

    let mut expected = en.zeroes(6).unwrap();
    en.add_bias(&input, &bias, 2, 3, &mut expected).unwrap();
    let expected = en.buffer_to_vec(&expected).unwrap();

    let mut out = en.buffer_from_slice(values).unwrap();
    en.add_bias_in_place(&mut out, &bias, 2, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.5, 0.0, 0.5, 2.5, 6.0, -0.5], 0.0);
    close(&out, &expected, 0.0);
}

fn add_bias_in_place_handles_single_row_and_column<E: Engine>(en: &E) {
    let mut out = en.buffer_from_slice([-1.0, 0.0, 2.0]).unwrap();
    let bias = en.buffer_from_slice([3.0, -4.0, 0.5]).unwrap();

    en.add_bias_in_place(&mut out, &bias, 1, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[2.0, -4.0, 2.5], 0.0);

    let mut out = en.buffer_from_slice([1.0, -2.0, 0.5]).unwrap();
    let bias = en.buffer_from_slice([-0.5]).unwrap();

    en.add_bias_in_place(&mut out, &bias, 3, 1).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.5, -2.5, 0.0], 0.0);

    let mut out = en.buffer_from_slice([-2.0]).unwrap();
    let bias = en.buffer_from_slice([0.5]).unwrap();

    en.add_bias_in_place(&mut out, &bias, 1, 1).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-1.5], 0.0);
}

fn add_bias_in_place_handles_zero_bias_and_zero_input<E: Engine>(en: &E) {
    let values = [1.5, -2.0, 0.0, 3.0, -4.5, 6.0];
    let mut out = en.buffer_from_slice(values).unwrap();
    let bias = en.zeroes(3).unwrap();

    en.add_bias_in_place(&mut out, &bias, 2, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &values, 0.0);

    let mut out = en.zeroes(6).unwrap();
    let bias = en.buffer_from_slice([1.0, -2.0, 0.5]).unwrap();

    en.add_bias_in_place(&mut out, &bias, 2, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0, -2.0, 0.5, 1.0, -2.0, 0.5], 0.0);
}

fn add_bias_in_place_accumulates_repeated_calls<E: Engine>(en: &E) {
    let mut out = en
        .buffer_from_slice([1.0, -2.0, 3.0, 4.0, -5.0, 6.0])
        .unwrap();
    let bias = en.buffer_from_slice([-0.5, 2.0, -3.0]).unwrap();

    en.add_bias_in_place(&mut out, &bias, 2, 3).unwrap();
    let values = en.buffer_to_vec(&out).unwrap();

    close(&values, &[0.5, 0.0, 0.0, 3.5, -3.0, 3.0], 0.0);

    en.add_bias_in_place(&mut out, &bias, 2, 3).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0, 2.0, -3.0, 3.0, -1.0, 0.0], 0.0);

    let bias = en.buffer_to_vec(&bias).unwrap();

    close(&bias, &[-0.5, 2.0, -3.0], 0.0);
}

fn repeated_embeddings_accumulate_gradients<E: Engine>(en: &E) {
    let tokens = [1, 0, 1];

    let aa = en
        .buffer_from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let mut out = en.zeroes(6).unwrap();
    en.embedding(&aa, &tokens, 3, 2, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[3.0, 4.0, 1.0, 2.0, 3.0, 4.0], 0.0);

    let ba = en
        .buffer_from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let mut out = en.zeroes(6).unwrap();
    en.embedding_backward(&tokens, &ba, 3, 2, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[3.0, 4.0, 6.0, 8.0, 0.0, 0.0], 0.0);
}

fn matmul_gradients_match_finite_differences<E: Engine>(en: &E) {
    let a = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1];
    let b = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2];
    let dy = [0.4, -0.7, 0.2, 0.8];

    let aa = en.buffer_from_slice(a).unwrap();
    let ab = en.buffer_from_slice(b).unwrap();
    let ac = en.buffer_from_slice(dy).unwrap();

    let mut out_a = en.zeroes(6).unwrap();
    let mut out_b = en.zeroes(6).unwrap();
    en.matmul_backward(&aa, &ab, &ac, 2, 3, 2, &mut out_a, &mut out_b)
        .unwrap();
    let da = en.buffer_to_vec(&out_a).unwrap();
    let db = en.buffer_to_vec(&out_b).unwrap();

    close(
        &da,
        &numerical_gradient(&a, |x| {
            let x = en.buffer_from_slice(x).unwrap();
            let mut out = en.zeroes(4).unwrap();
            en.matmul(&x, &ab, 2, 3, 2, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );

    close(
        &db,
        &numerical_gradient(&b, |x| {
            let x = en.buffer_from_slice(x).unwrap();
            let mut out = en.zeroes(4).unwrap();
            en.matmul(&aa, &x, 2, 3, 2, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );
}

fn softmax_and_cross_entropy_gradients_match_finite_differences<E: Engine>(en: &E) {
    let logits = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1];
    let dy = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2];

    let aa = en.buffer_from_slice(logits).unwrap();
    let ab = en.buffer_from_slice(dy).unwrap();

    let mut probabilities = en.zeroes(6).unwrap();
    en.softmax_rows(&aa, 2, 3, &mut probabilities).unwrap();

    let mut gradient = en.zeroes(6).unwrap();
    en.softmax_rows_backward(&probabilities, &ab, 2, 3, &mut gradient)
        .unwrap();

    let gradient = en.buffer_to_vec(&gradient).unwrap();

    close(
        &gradient,
        &numerical_gradient(&logits, |x| {
            let x = en.buffer_from_slice(x).unwrap();
            let mut out = en.zeroes(6).unwrap();
            en.softmax_rows(&x, 2, 3, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );

    let mut out = en.zeroes(6).unwrap();
    en.cross_entropy_backward(&aa, &[1, 2], 2, 3, &mut out)
        .unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &numerical_gradient(&logits, |x| {
            let x = en.buffer_from_slice(x).unwrap();

            en.cross_entropy(&x, &[1, 2], 2, 3).unwrap()
        }),
        1e-8,
    );
}

fn all_layer_norm_gradients_match_finite_differences<E: Engine>(en: &E) {
    let x = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1];
    let gamma = [0.8, 1.2, -0.5];
    let beta = [0.2, -0.1, 0.4];
    let dy = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2];

    let aa = en.buffer_from_slice(x).unwrap();
    let ab = en.buffer_from_slice(gamma).unwrap();
    let ac = en.buffer_from_slice(dy).unwrap();
    let ad = en.buffer_from_slice(beta).unwrap();

    let mut dx = en.zeroes(6).unwrap();
    let mut dg = en.zeroes(3).unwrap();
    let mut db = en.zeroes(3).unwrap();

    en.layer_norm_backward(&aa, &ab, &ac, 2, 3, 1e-5, &mut dx, &mut dg, &mut db)
        .unwrap();

    let dx = en.buffer_to_vec(&dx).unwrap();
    let dg = en.buffer_to_vec(&dg).unwrap();
    let db = en.buffer_to_vec(&db).unwrap();

    close(
        &dx,
        &numerical_gradient(&x, |v| {
            let v = en.buffer_from_slice(v).unwrap();
            let mut out = en.zeroes(6).unwrap();
            en.layer_norm(&v, &ab, &ad, 2, 3, 1e-5, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );

    close(
        &dg,
        &numerical_gradient(&gamma, |v| {
            let v = en.buffer_from_slice(v).unwrap();
            let mut out = en.zeroes(6).unwrap();
            en.layer_norm(&aa, &v, &ad, 2, 3, 1e-5, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );

    close(
        &db,
        &numerical_gradient(&beta, |v| {
            let v = en.buffer_from_slice(v).unwrap();
            let mut out = en.zeroes(6).unwrap();
            en.layer_norm(&aa, &ab, &v, 2, 3, 1e-5, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );
}

fn relu_dropout_bias_and_embedding_gradients_match_finite_differences<E: Engine>(en: &E) {
    let x = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1];
    let dy = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2];
    let mask = [2.0, 0.0, 2.0, 0.0, 0.0, 2.0];

    let aa = en.buffer_from_slice(x).unwrap();
    let ab = en.buffer_from_slice(dy).unwrap();
    let ac = en.buffer_from_slice(mask).unwrap();

    let mut out = en.zeroes(6).unwrap();
    en.relu_backward(&aa, &ab, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &numerical_gradient(&x, |v| {
            let v = en.buffer_from_slice(v).unwrap();
            let mut out = en.zeroes(6).unwrap();
            en.relu(&v, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );

    let ba = en.buffer_from_slice([0.0]).unwrap();
    let bb = en.buffer_from_slice([1.0]).unwrap();

    let mut out = en.zeroes(1).unwrap();
    en.relu_backward(&ba, &bb, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0], 0.0);

    let mut out = en.zeroes(6).unwrap();
    en.dropout(&ab, &ac, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &numerical_gradient(&x, |v| {
            let v = en.buffer_from_slice(v).unwrap();
            let mut out = en.zeroes(6).unwrap();
            en.dropout(&v, &ac, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );

    let mut out = en.zeroes(3).unwrap();
    en.sum_rows(&ab, 2, 3, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &numerical_gradient(&[0.1, 0.2, 0.3], |v| {
            let v = en.buffer_from_slice(v).unwrap();
            let mut out = en.zeroes(6).unwrap();
            en.add_bias(&aa, &v, 2, 3, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );

    let mut out = en.zeroes(6).unwrap();
    en.embedding_backward(&[1, 0, 1], &ab, 3, 2, &mut out)
        .unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &numerical_gradient(&x, |v| {
            let v = en.buffer_from_slice(v).unwrap();
            let mut out = en.zeroes(6).unwrap();
            en.embedding(&v, &[1, 0, 1], 3, 2, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );
}

fn buffers_preserve_values_and_zero_initialization<E: Engine>(en: &E) {
    let a = en.zeroes(4).unwrap();
    let out = en.buffer_to_vec(&a).unwrap();

    close(&out, &[0.0; 4], 0.0);

    let a = en.zeroes(0).unwrap();
    let out = en.buffer_to_vec(&a).unwrap();

    close(&out, &[], 0.0);

    let a = en.buffer_from_slice([-2.5, 0.0, 0.125, 1000.0]).unwrap();
    let out = en.buffer_to_vec(&a).unwrap();

    close(&out, &[-2.5, 0.0, 0.125, 1000.0], 0.0);
}

fn vector_matrix_products_overwrite_reused_outputs<E: Engine>(en: &E) {
    let a = en.buffer_from_slice([1.0, -2.0, 3.0]).unwrap();
    let b = en.buffer_from_slice([4.0, 5.0, -6.0]).unwrap();

    let mut out = en.buffer_from_slice([99.0; 1]).unwrap();
    en.matmul(&a, &b, 1, 3, 1, &mut out).unwrap();
    let values = en.buffer_to_vec(&out).unwrap();

    close(&values, &[-24.0], 0.0);

    let mut outer = en.buffer_from_slice([99.0; 9]).unwrap();
    en.matmul(&a, &b, 3, 1, 3, &mut outer).unwrap();
    let values = en.buffer_to_vec(&outer).unwrap();

    close(
        &values,
        &[4.0, 5.0, -6.0, -8.0, -10.0, 12.0, 12.0, 15.0, -18.0],
        0.0,
    );

    en.matmul(&a, &b, 1, 3, 1, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-24.0], 0.0);

    let dy = en.buffer_from_slice([2.0]).unwrap();
    let mut da = en.buffer_from_slice([99.0; 3]).unwrap();
    let mut db = en.buffer_from_slice([99.0; 3]).unwrap();
    en.matmul_backward(&a, &b, &dy, 1, 3, 1, &mut da, &mut db)
        .unwrap();
    let da = en.buffer_to_vec(&da).unwrap();
    let db = en.buffer_to_vec(&db).unwrap();

    close(&da, &[8.0, 10.0, -12.0], 0.0);
    close(&db, &[2.0, -4.0, 6.0], 0.0);
}

fn uneven_column_layouts_preserve_row_order<E: Engine>(en: &E) {
    let a = en.buffer_from_slice([-1.0, 2.0, -3.0]).unwrap();
    let b = en
        .buffer_from_slice([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .unwrap();

    let mut joined = en.buffer_from_slice([99.0; 9]).unwrap();
    en.concat_columns(&a, &b, 3, 1, 2, &mut joined).unwrap();
    let out = en.buffer_to_vec(&joined).unwrap();

    close(&out, &[-1.0, 4.0, 5.0, 2.0, 6.0, 7.0, -3.0, 8.0, 9.0], 0.0);

    let mut out = en.buffer_from_slice([99.0; 3]).unwrap();
    en.slice_columns(&joined, 3, 3, 0, 1, &mut out).unwrap();
    let values = en.buffer_to_vec(&out).unwrap();

    close(&values, &[-1.0, 2.0, -3.0], 0.0);

    let mut right = en.buffer_from_slice([99.0; 6]).unwrap();
    en.slice_columns(&joined, 3, 3, 1, 2, &mut right).unwrap();
    let values = en.buffer_to_vec(&right).unwrap();

    close(&values, &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 0.0);

    en.slice_columns(&joined, 3, 3, 1, 1, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[4.0, 6.0, 8.0], 0.0);
}

fn softmax_handles_masked_entries_and_independent_row_offsets<E: Engine>(en: &E) {
    let a = en
        .buffer_from_slice([2.0_f64.ln(), f64::NEG_INFINITY, 0.0])
        .unwrap();

    let mut out = en.zeroes(3).unwrap();
    en.softmax(&a, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[2.0 / 3.0, 0.0, 1.0 / 3.0], 1e-12);

    let b = en
        .buffer_from_slice([
            1000.0,
            1000.0 + 2.0_f64.ln(),
            f64::NEG_INFINITY,
            -1000.0 + 3.0_f64.ln(),
            -1000.0,
            f64::NEG_INFINITY,
        ])
        .unwrap();

    let mut out = en.zeroes(6).unwrap();
    en.softmax_rows(&b, 2, 3, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[1.0 / 3.0, 2.0 / 3.0, 0.0, 0.75, 0.25, 0.0], 1e-12);
}

fn masked_softmax_gradients_match_finite_differences<E: Engine>(en: &E) {
    let x = [0.2, -0.5, 0.7, 1.0, -0.3, 0.1, -0.4, 0.6, 0.8];
    let dy = [0.1, 0.6, -0.8, 0.9, 0.3, -0.2, 0.5, -0.7, 0.4];

    let a = en.buffer_from_slice(x).unwrap();
    let b = en.buffer_from_slice(dy).unwrap();

    let mut masked = en.zeroes(9).unwrap();
    en.causal_mask(&a, 3, &mut masked).unwrap();

    let mut probabilities = en.zeroes(9).unwrap();
    en.softmax_rows(&masked, 3, 3, &mut probabilities).unwrap();

    let mut gradient = en.zeroes(9).unwrap();
    en.softmax_rows_backward(&probabilities, &b, 3, 3, &mut gradient)
        .unwrap();

    let mut out = en.zeroes(9).unwrap();
    en.causal_mask_backward(&gradient, 3, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &numerical_gradient(&x, |v| {
            let v = en.buffer_from_slice(v).unwrap();
            let mut masked = en.zeroes(9).unwrap();
            en.causal_mask(&v, 3, &mut masked).unwrap();
            let mut out = en.zeroes(9).unwrap();
            en.softmax_rows(&masked, 3, 3, &mut out).unwrap();
            let out = en.buffer_to_vec(&out).unwrap();

            dot(&out, &dy)
        }),
        1e-8,
    );

    close(&[out[0], out[1], out[2], out[5]], &[0.0; 4], 0.0);
}

fn cross_entropy_gradients_average_rows_and_handle_extreme_logits<E: Engine>(en: &E) {
    let a = en.buffer_from_slice([0.0; 6]).unwrap();

    let mut out = en.zeroes(6).unwrap();
    en.cross_entropy_backward(&a, &[0, 2], 2, 3, &mut out)
        .unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(
        &out,
        &[
            -1.0 / 3.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
            -1.0 / 3.0,
        ],
        1e-12,
    );

    let b = en
        .buffer_from_slice([1000.0, 0.0, -1000.0, -1000.0, 0.0, 1000.0])
        .unwrap();
    let loss = en.cross_entropy(&b, &[0, 0], 2, 3).unwrap();

    close(&[loss], &[1000.0], 1e-12);

    let mut out = en.zeroes(6).unwrap();
    en.cross_entropy_backward(&b, &[0, 0], 2, 3, &mut out)
        .unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0, 0.0, 0.0, -0.5, 0.0, 0.5], 1e-12);
}

fn single_class_softmax_and_loss_have_zero_gradients<E: Engine>(en: &E) {
    let a = en.buffer_from_slice([-1000.0, 0.0, 1000.0]).unwrap();
    let b = en.buffer_from_slice([2.0, -3.0, 5.0]).unwrap();

    let mut probabilities = en.zeroes(3).unwrap();
    en.softmax_rows(&a, 3, 1, &mut probabilities).unwrap();
    let out = en.buffer_to_vec(&probabilities).unwrap();

    close(&out, &[1.0; 3], 0.0);

    let mut out = en.zeroes(3).unwrap();
    en.softmax_rows_backward(&probabilities, &b, 3, 1, &mut out)
        .unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0; 3], 0.0);

    let loss = en.cross_entropy(&a, &[0, 0, 0], 3, 1).unwrap();

    close(&[loss], &[0.0], 0.0);

    let mut out = en.zeroes(3).unwrap();
    en.cross_entropy_backward(&a, &[0, 0, 0], 3, 1, &mut out)
        .unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0; 3], 0.0);
}

fn constant_layer_norm_rows_have_finite_gradients<E: Engine>(en: &E) {
    let x = en
        .buffer_from_slice([4.0, 4.0, 4.0, -2.0, -2.0, -2.0])
        .unwrap();
    let gamma = en.buffer_from_slice([2.0, -1.0, 0.5]).unwrap();
    let dy = en
        .buffer_from_slice([1.0, 2.0, -2.0, -1.0, 1.0, 4.0])
        .unwrap();

    let mut dx = en.buffer_from_slice([99.0; 6]).unwrap();
    let mut dg = en.buffer_from_slice([99.0; 3]).unwrap();
    let mut db = en.buffer_from_slice([99.0; 3]).unwrap();
    en.layer_norm_backward(&x, &gamma, &dy, 2, 3, 0.25, &mut dx, &mut dg, &mut db)
        .unwrap();
    let dx = en.buffer_to_vec(&dx).unwrap();
    let dg = en.buffer_to_vec(&dg).unwrap();
    let db = en.buffer_to_vec(&db).unwrap();

    close(
        &dx,
        &[
            14.0 / 3.0,
            -10.0 / 3.0,
            -4.0 / 3.0,
            -10.0 / 3.0,
            -4.0 / 3.0,
            14.0 / 3.0,
        ],
        1e-12,
    );
    close(&dg, &[0.0; 3], 0.0);
    close(&db, &[0.0, 3.0, 2.0], 0.0);
}

fn single_column_layer_norm_has_only_bias_gradients<E: Engine>(en: &E) {
    let x = en.buffer_from_slice([2.0, -4.0, 10.0]).unwrap();
    let gamma = en.buffer_from_slice([-3.0]).unwrap();
    let beta = en.buffer_from_slice([0.5]).unwrap();
    let dy = en.buffer_from_slice([2.0, -3.0, 5.0]).unwrap();

    let mut out = en.zeroes(3).unwrap();
    en.layer_norm(&x, &gamma, &beta, 3, 1, 1e-5, &mut out)
        .unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.5; 3], 0.0);

    let mut dx = en.zeroes(3).unwrap();
    let mut dg = en.zeroes(1).unwrap();
    let mut db = en.zeroes(1).unwrap();
    en.layer_norm_backward(&x, &gamma, &dy, 3, 1, 1e-5, &mut dx, &mut dg, &mut db)
        .unwrap();
    let dx = en.buffer_to_vec(&dx).unwrap();
    let dg = en.buffer_to_vec(&dg).unwrap();
    let db = en.buffer_to_vec(&db).unwrap();

    close(&dx, &[0.0; 3], 0.0);
    close(&dg, &[0.0], 0.0);
    close(&db, &[4.0], 0.0);
}

fn relu_and_dropout_match_hand_calculations<E: Engine>(en: &E) {
    let x = en.buffer_from_slice([-2.0, 0.0, 3.0, -4.0]).unwrap();
    let dy = en.buffer_from_slice([1.0, -2.0, -3.0, 4.0]).unwrap();
    let mask = en.buffer_from_slice([2.0, 0.0, 0.0, 2.0]).unwrap();

    let mut out = en.zeroes(4).unwrap();
    en.relu(&x, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0, 0.0, 3.0, 0.0], 0.0);

    let mut out = en.zeroes(4).unwrap();
    en.relu_backward(&x, &dy, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0, 0.0, -3.0, 0.0], 0.0);

    let mut out = en.zeroes(4).unwrap();
    en.dropout(&x, &mask, &mut out).unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[-4.0, 0.0, 0.0, -8.0], 0.0);
}

fn embedding_gradients_clear_previously_used_rows<E: Engine>(en: &E) {
    let dy = en
        .buffer_from_slice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let mut out = en.buffer_from_slice([99.0; 6]).unwrap();
    en.embedding_backward(&[2, 0, 2], &dy, 3, 2, &mut out)
        .unwrap();
    let values = en.buffer_to_vec(&out).unwrap();

    close(&values, &[3.0, 4.0, 0.0, 0.0, 6.0, 8.0], 0.0);

    en.embedding_backward(&[1, 1, 1], &dy, 3, 2, &mut out)
        .unwrap();
    let out = en.buffer_to_vec(&out).unwrap();

    close(&out, &[0.0, 0.0, 9.0, 12.0, 0.0, 0.0], 0.0);
}

pub fn full_suite<E: Engine>(engine: &E) {
    buffer_all_finite_detects_nan_and_infinities(engine);
    adamw_in_place_matches_hand_calculations_across_steps(engine);
    adamw_in_place_applies_weight_decay_with_zero_gradients(engine);
    adamw_in_place_rejects_non_finite_updates(engine);
    rectangular_matrix_multiplication_and_transpose(engine);
    transpose_in_place_matches_transpose_on_rectangular_inputs(engine);
    transpose_in_place_handles_square_matrices(engine);
    transpose_in_place_round_trip_restores_rectangular_input(engine);
    transpose_in_place_handles_single_row_column_and_element(engine);
    softmax_is_stable_and_masks_future_positions(engine);
    softmax_in_place_matches_softmax(engine);
    softmax_in_place_is_stable_for_extreme_logits(engine);
    softmax_in_place_handles_masked_entries(engine);
    softmax_in_place_handles_uniform_and_single_element_inputs(engine);
    softmax_rows_in_place_matches_softmax_rows(engine);
    softmax_rows_in_place_is_stable_for_extreme_logits(engine);
    softmax_rows_in_place_handles_masked_entries(engine);
    softmax_rows_in_place_handles_single_row_and_column(engine);
    causal_mask_in_place_matches_causal_mask(engine);
    causal_mask_in_place_is_idempotent(engine);
    causal_mask_in_place_preserves_single_position(engine);
    causal_mask_in_place_masks_future_softmax_probabilities(engine);
    causal_mask_backward_in_place_matches_causal_mask_backward(engine);
    causal_mask_backward_in_place_is_idempotent(engine);
    causal_mask_backward_in_place_handles_single_position_and_zero_gradients(engine);
    causal_mask_backward_in_place_matches_finite_differences(engine);
    normalization_and_loss_match_hand_calculations(engine);
    basic_operations_and_head_layout(engine);
    add_in_place_matches_add(engine);
    add_in_place_handles_zero_operands(engine);
    add_in_place_accumulates_repeated_calls(engine);
    add_in_place_handles_single_element_and_empty_buffers(engine);
    dropout_in_place_matches_dropout(engine);
    dropout_in_place_handles_zero_and_identity_masks(engine);
    dropout_in_place_composes_repeated_calls(engine);
    dropout_in_place_handles_single_element_and_empty_buffers(engine);
    scale_in_place_matches_scale(engine);
    scale_in_place_handles_zero_and_identity_factors(engine);
    scale_in_place_composes_repeated_calls(engine);
    scale_in_place_handles_single_element_and_empty_buffers(engine);
    add_bias_in_place_matches_add_bias_on_rectangular_inputs(engine);
    add_bias_in_place_handles_single_row_and_column(engine);
    add_bias_in_place_handles_zero_bias_and_zero_input(engine);
    add_bias_in_place_accumulates_repeated_calls(engine);
    repeated_embeddings_accumulate_gradients(engine);
    matmul_gradients_match_finite_differences(engine);
    softmax_and_cross_entropy_gradients_match_finite_differences(engine);
    all_layer_norm_gradients_match_finite_differences(engine);
    relu_dropout_bias_and_embedding_gradients_match_finite_differences(engine);
    buffers_preserve_values_and_zero_initialization(engine);
    vector_matrix_products_overwrite_reused_outputs(engine);
    uneven_column_layouts_preserve_row_order(engine);
    softmax_handles_masked_entries_and_independent_row_offsets(engine);
    masked_softmax_gradients_match_finite_differences(engine);
    cross_entropy_gradients_average_rows_and_handle_extreme_logits(engine);
    single_class_softmax_and_loss_have_zero_gradients(engine);
    constant_layer_norm_rows_have_finite_gradients(engine);
    single_column_layer_norm_has_only_bias_gradients(engine);
    relu_and_dropout_match_hand_calculations(engine);
    embedding_gradients_clear_previously_used_rows(engine);
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
