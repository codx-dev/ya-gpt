use crate::engine::Engine;

#[derive(Debug, Default, Clone, Copy)]
pub struct Naive;

impl Engine for Naive {
    fn add(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    fn add_bias(&self, input: &[f64], bias: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        debug_assert!(check_matrix(input, rows, cols));
        debug_assert_eq!(bias.len(), cols);
        input
            .iter()
            .enumerate()
            .map(|(i, x)| x + bias[i % cols])
            .collect()
    }

    fn causal_mask(&self, input: &[f64], time: usize) -> Vec<f64> {
        debug_assert!(check_matrix(input, time, time));
        let mut output = input.to_vec();
        for row in 0..time {
            for col in row + 1..time {
                output[row * time + col] = f64::NEG_INFINITY;
            }
        }
        output
    }

    fn causal_mask_backward(&self, d_output: &[f64], time: usize) -> Vec<f64> {
        debug_assert!(check_matrix(d_output, time, time));
        let mut output = d_output.to_vec();
        for row in 0..time {
            for col in row + 1..time {
                output[row * time + col] = 0.0;
            }
        }
        output
    }

    fn concat_columns(
        &self,
        a: &[f64],
        b: &[f64],
        rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Vec<f64> {
        debug_assert!(check_matrix(a, rows, a_cols));
        debug_assert!(check_matrix(b, rows, b_cols));
        let mut output = Vec::new();
        for row in 0..rows {
            output.extend_from_slice(&a[row * a_cols..(row + 1) * a_cols]);
            output.extend_from_slice(&b[row * b_cols..(row + 1) * b_cols]);
        }
        output
    }

    fn cross_entropy(&self, logits: &[f64], targets: &[usize], rows: usize, vocab: usize) -> f64 {
        debug_assert!(check_matrix(logits, rows, vocab));
        debug_assert_eq!(targets.len(), rows);
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "logits must be finite"
        );
        let mut loss = 0.0;
        for (row, &target) in logits.chunks_exact(vocab).zip(targets) {
            assert!(target < vocab);
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum = row.iter().map(|x| (x - max).exp()).sum::<f64>();
            loss += (max - row[target]) + sum.ln();
        }
        loss / rows as f64
    }

    fn cross_entropy_backward(
        &self,
        logits: &[f64],
        targets: &[usize],
        rows: usize,
        vocab: usize,
    ) -> Vec<f64> {
        debug_assert_eq!(targets.len(), rows);
        let mut gradient = self.softmax_rows(logits, rows, vocab);
        for (row, &target) in targets.iter().enumerate() {
            assert!(target < vocab);
            gradient[row * vocab + target] -= 1.0;
        }
        self.scale(&gradient, 1.0 / rows as f64)
    }

    fn dropout(&self, input: &[f64], mask: &[f64]) -> Vec<f64> {
        multiply(input, mask)
    }

    fn embedding(
        &self,
        table: &[f64],
        tokens: &[usize],
        vocab: usize,
        channels: usize,
    ) -> Vec<f64> {
        debug_assert!(check_matrix(table, vocab, channels));
        let mut output =
            Vec::with_capacity(tokens.len().checked_mul(channels).expect("shape overflow"));
        for &token in tokens {
            assert!(token < vocab, "token outside vocabulary");
            output.extend_from_slice(&table[token * channels..(token + 1) * channels]);
        }
        output
    }

    fn embedding_backward(
        &self,
        tokens: &[usize],
        d_output: &[f64],
        vocab: usize,
        channels: usize,
    ) -> Vec<f64> {
        debug_assert!(check_matrix(d_output, tokens.len(), channels));
        let mut gradient = vec![0.0; vocab.checked_mul(channels).expect("shape overflow")];
        for (row, &token) in tokens.iter().enumerate() {
            assert!(token < vocab, "token outside vocabulary");
            for col in 0..channels {
                gradient[token * channels + col] += d_output[row * channels + col];
            }
        }
        gradient
    }

    fn layer_norm(
        &self,
        input: &[f64],
        gamma: &[f64],
        beta: &[f64],
        rows: usize,
        cols: usize,
        epsilon: f64,
    ) -> Vec<f64> {
        debug_assert!(check_matrix(input, rows, cols));
        debug_assert_eq!(gamma.len(), cols);
        debug_assert_eq!(beta.len(), cols);
        assert!(epsilon.is_finite() && epsilon > 0.0);
        let mut output = vec![0.0; input.len()];
        for row in 0..rows {
            let x = &input[row * cols..(row + 1) * cols];
            let mean = x.iter().sum::<f64>() / cols as f64;
            let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / cols as f64;
            let inverse_std = 1.0 / (variance + epsilon).sqrt();
            for col in 0..cols {
                output[row * cols + col] = (x[col] - mean) * inverse_std * gamma[col] + beta[col];
            }
        }
        output
    }

    fn layer_norm_backward(
        &self,
        input: &[f64],
        gamma: &[f64],
        d_output: &[f64],
        rows: usize,
        cols: usize,
        epsilon: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        debug_assert!(check_matrix(input, rows, cols));
        debug_assert!(check_matrix(d_output, rows, cols));
        debug_assert_eq!(gamma.len(), cols);
        assert!(epsilon.is_finite() && epsilon > 0.0);
        let mut d_input = vec![0.0; input.len()];
        let mut d_gamma = vec![0.0; cols];
        let mut d_beta = vec![0.0; cols];
        for row in 0..rows {
            let x = &input[row * cols..(row + 1) * cols];
            let dy = &d_output[row * cols..(row + 1) * cols];
            let mean = x.iter().sum::<f64>() / cols as f64;
            let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / cols as f64;
            let inverse_std = 1.0 / (variance + epsilon).sqrt();
            let normalized: Vec<f64> = x.iter().map(|v| (v - mean) * inverse_std).collect();
            let dx_hat = multiply(dy, gamma);
            let sum_dx_hat: f64 = dx_hat.iter().sum();
            let sum_product: f64 = dx_hat.iter().zip(&normalized).map(|(a, b)| a * b).sum();
            for col in 0..cols {
                d_input[row * cols + col] = inverse_std / cols as f64
                    * (cols as f64 * dx_hat[col] - sum_dx_hat - normalized[col] * sum_product);
                d_gamma[col] += dy[col] * normalized[col];
                d_beta[col] += dy[col];
            }
        }
        (d_input, d_gamma, d_beta)
    }

    fn matmul(&self, a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
        debug_assert!(check_matrix(a, rows, inner));
        debug_assert!(check_matrix(b, inner, cols));
        let mut output = vec![0.0; rows.checked_mul(cols).expect("shape overflow")];
        for row in 0..rows {
            for col in 0..cols {
                for k in 0..inner {
                    output[row * cols + col] += a[row * inner + k] * b[k * cols + col];
                }
            }
        }
        output
    }

    fn matmul_backward(
        &self,
        a: &[f64],
        b: &[f64],
        d_output: &[f64],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        debug_assert!(check_matrix(d_output, rows, cols));
        let d_a = self.matmul(d_output, &self.transpose(b, inner, cols), rows, cols, inner);
        let d_b = self.matmul(&self.transpose(a, rows, inner), d_output, inner, rows, cols);
        (d_a, d_b)
    }

    fn relu(&self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|x| x.max(0.0)).collect()
    }

    fn relu_backward(&self, input: &[f64], d_output: &[f64]) -> Vec<f64> {
        debug_assert_eq!(input.len(), d_output.len());
        input
            .iter()
            .zip(d_output)
            .map(|(x, d)| if *x > 0.0 { *d } else { 0.0 })
            .collect()
    }

    fn scale(&self, input: &[f64], factor: f64) -> Vec<f64> {
        input.iter().map(|x| x * factor).collect()
    }

    fn slice_columns(
        &self,
        input: &[f64],
        rows: usize,
        cols: usize,
        start: usize,
        width: usize,
    ) -> Vec<f64> {
        debug_assert!(check_matrix(input, rows, cols));
        assert!(width > 0 && start.checked_add(width).is_some_and(|end| end <= cols));
        let mut output = Vec::new();
        for row in 0..rows {
            output.extend_from_slice(&input[row * cols + start..row * cols + start + width]);
        }
        output
    }

    fn softmax(&self, input: &[f64]) -> Vec<f64> {
        assert!(!input.is_empty(), "softmax requires a nonempty row");
        assert!(
            input
                .iter()
                .all(|x| x.is_finite() || *x == f64::NEG_INFINITY)
        );
        let max = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(max.is_finite(), "softmax requires an unmasked entry");
        let exp: Vec<f64> = input.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        self.scale(&exp, 1.0 / sum)
    }

    fn softmax_rows(&self, input: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        debug_assert!(check_matrix(input, rows, cols));
        input
            .chunks_exact(cols)
            .flat_map(|i| self.softmax(i))
            .collect()
    }

    fn softmax_rows_backward(
        &self,
        probabilities: &[f64],
        d_output: &[f64],
        rows: usize,
        cols: usize,
    ) -> Vec<f64> {
        debug_assert!(check_matrix(probabilities, rows, cols));
        debug_assert!(check_matrix(d_output, rows, cols));
        probabilities
            .chunks_exact(cols)
            .zip(d_output.chunks_exact(cols))
            .flat_map(|(p, d)| softmax_backward(p, d))
            .collect()
    }

    fn sum_rows(&self, input: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        debug_assert!(check_matrix(input, rows, cols));
        let mut output = vec![0.0; cols];
        for row in input.chunks_exact(cols) {
            for (sum, value) in output.iter_mut().zip(row) {
                *sum += value;
            }
        }
        output
    }

    fn transpose(&self, input: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        debug_assert!(check_matrix(input, rows, cols));
        let mut output = vec![0.0; input.len()];
        for row in 0..rows {
            for col in 0..cols {
                output[col * rows + row] = input[row * cols + col];
            }
        }
        output
    }
}

fn check_matrix(input: &[f64], rows: usize, cols: usize) -> bool {
    assert!(rows > 0 && cols > 0, "matrix dimensions must be positive");
    assert_eq!(input.len(), rows.checked_mul(cols).expect("shape overflow"));
    true
}

fn multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

fn softmax_backward(probabilities: &[f64], d_output: &[f64]) -> Vec<f64> {
    debug_assert_eq!(probabilities.len(), d_output.len());
    let dot: f64 = probabilities.iter().zip(d_output).map(|(p, d)| p * d).sum();
    probabilities
        .iter()
        .zip(d_output)
        .map(|(p, d)| p * (d - dot))
        .collect()
}

#[test]
fn naive_full_suite() {
    super::tests::full_suite(&Naive)
}
