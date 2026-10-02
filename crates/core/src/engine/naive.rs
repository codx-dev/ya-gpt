use crate::engine::Engine;

#[derive(Debug, Default, Clone, Copy)]
pub struct Naive;

// NOTE: we don't resize the Vec on the implementation to emulate the SIMD/CUDA behavior.
pub type NaiveBuffer = Vec<f64>;

impl Engine for Naive {
    type Buffer = NaiveBuffer;

    fn slice(&self, buffer: &Self::Buffer, offset: usize, len: usize) -> Self::Buffer {
        // TODO buffer should probably leverage some unsafe pointer so we dont clone here
        buffer[offset..offset + len].to_vec()
    }

    fn copy(&self, dst: &mut Self::Buffer, src: &Self::Buffer, ofs: usize) {
        let len = src.len();
        let buf = &mut dst[ofs..ofs + len];

        buf.copy_from_slice(src);
    }

    fn clone(&self, buffer: &Self::Buffer) -> anyhow::Result<Self::Buffer> {
        Ok(buffer.clone())
    }

    fn zeroes(&self, len: usize) -> anyhow::Result<Self::Buffer> {
        Ok(vec![0.0f64; len])
    }

    fn filled(&self, value: f64, len: usize) -> anyhow::Result<Self::Buffer> {
        Ok(vec![value; len])
    }

    fn buffer_from_slice<S: AsRef<[f64]>>(&self, slice: S) -> anyhow::Result<Self::Buffer> {
        Ok(Vec::from(slice.as_ref()))
    }

    fn buffer_len(&self, buffer: &Self::Buffer) -> usize {
        buffer.len()
    }

    fn buffer_to_vec(&self, buffer: &Self::Buffer) -> anyhow::Result<Vec<f64>> {
        Ok(buffer.clone())
    }

    fn buffer_all_finite(&self, buffer: &Self::Buffer) -> anyhow::Result<bool> {
        Ok(buffer.iter().all(|value| value.is_finite()))
    }

    fn adamw_in_place(
        &self,
        values: &mut Self::Buffer,
        gradients: &Self::Buffer,
        first_moment: &mut Self::Buffer,
        second_moment: &mut Self::Buffer,
        learning_rate: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        step: i32,
    ) -> anyhow::Result<()> {
        debug_assert_eq!(values.len(), gradients.len());
        debug_assert_eq!(values.len(), first_moment.len());
        debug_assert_eq!(values.len(), second_moment.len());
        debug_assert!(step > 0);

        let correction1 = 1.0 - beta1.powi(step);
        let correction2 = 1.0 - beta2.powi(step);

        for i in 0..values.len() {
            let gradient = gradients[i];
            let m = beta1 * first_moment[i] + (1.0 - beta1) * gradient;
            let v = beta2 * second_moment[i] + (1.0 - beta2) * gradient * gradient;
            let mut value = values[i] * (1.0 - learning_rate * weight_decay);
            value -= learning_rate * (m / correction1) / ((v / correction2).sqrt() + epsilon);

            anyhow::ensure!(
                value.is_finite() && m.is_finite() && v.is_finite(),
                "non-finite optimizer update; reduce the learning rate"
            );

            values[i] = value;
            first_moment[i] = m;
            second_moment[i] = v;
        }

        Ok(())
    }

    fn add(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= a.len());
        debug_assert_eq!(a.len(), b.len());
        out.fill(0.0);
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
        Ok(())
    }

    fn add_in_place(&self, out: &mut Self::Buffer, b: &Self::Buffer) -> anyhow::Result<()> {
        debug_assert_eq!(out.len(), b.len());
        for i in 0..out.len() {
            out[i] += b[i];
        }
        Ok(())
    }

    fn add_bias_in_place(
        &self,
        out: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<()> {
        debug_assert!(rows > 0 && cols > 0, "matrix dimensions must be positive");
        debug_assert!(out.len() >= rows.checked_mul(cols).expect("shape overflow"));
        debug_assert_eq!(bias.len(), cols);
        for i in 0..out.len() {
            out[i] += bias[i % cols];
        }
        Ok(())
    }

    fn causal_mask_in_place(&self, out: &mut Self::Buffer, time: usize) -> anyhow::Result<()> {
        debug_assert!(check_matrix(out, time, time));
        for row in 0..time {
            for col in row + 1..time {
                out[row * time + col] = f64::NEG_INFINITY;
            }
        }
        Ok(())
    }

    fn causal_mask_backward_in_place(
        &self,
        out: &mut Self::Buffer,
        time: usize,
    ) -> anyhow::Result<()> {
        debug_assert!(check_matrix(out, time, time));
        for row in 0..time {
            for col in row + 1..time {
                out[row * time + col] = 0.0;
            }
        }
        Ok(())
    }

    fn concat_columns(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        rows: usize,
        a_cols: usize,
        b_cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(check_matrix(a, rows, a_cols));
        debug_assert!(check_matrix(b, rows, b_cols));
        let cols = a_cols + b_cols;
        debug_assert!(out.len() >= rows.checked_mul(cols).expect("shape overflow"));
        for row in 0..rows {
            let output = &mut out[row * cols..(row + 1) * cols];
            output[..a_cols].copy_from_slice(&a[row * a_cols..(row + 1) * a_cols]);
            output[a_cols..].copy_from_slice(&b[row * b_cols..(row + 1) * b_cols]);
        }
        Ok(())
    }

    fn cross_entropy(
        &self,
        logits: &Self::Buffer,
        targets: &[usize],
        rows: usize,
        vocab: usize,
    ) -> anyhow::Result<f64> {
        debug_assert!(check_matrix(logits, rows, vocab));
        debug_assert_eq!(targets.len(), rows);
        debug_assert!(
            logits.iter().all(|x| x.is_finite()),
            "logits must be finite"
        );
        let mut loss = 0.0;
        for (row, &target) in logits.chunks_exact(vocab).zip(targets) {
            debug_assert!(target < vocab);
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum = row.iter().map(|x| (x - max).exp()).sum::<f64>();
            loss += (max - row[target]) + sum.ln();
        }
        Ok(loss / rows as f64)
    }

    fn cross_entropy_backward(
        &self,
        logits: &Self::Buffer,
        targets: &[usize],
        rows: usize,
        vocab: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= rows.checked_mul(vocab).expect("shape overflow"));
        debug_assert_eq!(targets.len(), rows);
        self.softmax_rows(logits, rows, vocab, out)?;
        let mut gradient = out.clone();
        for (row, &target) in targets.iter().enumerate() {
            debug_assert!(target < vocab);
            gradient[row * vocab + target] -= 1.0;
        }
        self.scale(&gradient, 1.0 / rows as f64, out)
    }

    fn dropout(
        &self,
        input: &Self::Buffer,
        mask: &Self::Buffer,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= input.len());
        multiply(input, mask, out);
        Ok(())
    }

    fn dropout_in_place(&self, out: &mut Self::Buffer, mask: &Self::Buffer) -> anyhow::Result<()> {
        debug_assert!(out.len() >= mask.len());
        multiply_in_place(out, mask);
        Ok(())
    }

    fn embedding(
        &self,
        table: &Self::Buffer,
        tokens: &[usize],
        vocab: usize,
        channels: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(check_matrix(table, vocab, channels));
        debug_assert!(out.len() >= tokens.len().checked_mul(channels).expect("shape overflow"));
        for (row, &token) in tokens.iter().enumerate() {
            debug_assert!(token < vocab, "token outside vocabulary");
            out[row * channels..(row + 1) * channels]
                .copy_from_slice(&table[token * channels..(token + 1) * channels]);
        }
        Ok(())
    }

    fn embedding_backward(
        &self,
        tokens: &[usize],
        d_output: &Self::Buffer,
        vocab: usize,
        channels: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= vocab.checked_mul(channels).expect("shape overflow"));
        debug_assert!(check_matrix(d_output, tokens.len(), channels));
        out.fill(0.0);
        for (row, &token) in tokens.iter().enumerate() {
            debug_assert!(token < vocab, "token outside vocabulary");
            for col in 0..channels {
                out[token * channels + col] += d_output[row * channels + col];
            }
        }
        Ok(())
    }

    fn layer_norm(
        &self,
        input: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        rows: usize,
        cols: usize,
        epsilon: f64,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= input.len());
        debug_assert!(check_matrix(input, rows, cols));
        debug_assert_eq!(gamma.len(), cols);
        debug_assert_eq!(beta.len(), cols);
        debug_assert!(epsilon.is_finite() && epsilon > 0.0);
        out.fill(0.0);
        for row in 0..rows {
            let x = &input[row * cols..(row + 1) * cols];
            let mean = x.iter().sum::<f64>() / cols as f64;
            let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / cols as f64;
            let inverse_std = 1.0 / (variance + epsilon).sqrt();
            for col in 0..cols {
                out[row * cols + col] = (x[col] - mean) * inverse_std * gamma[col] + beta[col];
            }
        }
        Ok(())
    }

    fn layer_norm_backward(
        &self,
        input: &Self::Buffer,
        gamma: &Self::Buffer,
        d_output: &Self::Buffer,
        rows: usize,
        cols: usize,
        epsilon: f64,
        d_input: &mut Self::Buffer,
        d_gamma: &mut Self::Buffer,
        d_beta: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(d_input.len() >= input.len());
        debug_assert!(d_gamma.len() >= cols);
        debug_assert!(d_beta.len() >= cols);
        debug_assert!(check_matrix(input, rows, cols));
        debug_assert!(check_matrix(d_output, rows, cols));
        debug_assert_eq!(gamma.len(), cols);
        debug_assert!(epsilon.is_finite() && epsilon > 0.0);
        d_input.fill(0.0);
        d_gamma.fill(0.0);
        d_beta.fill(0.0);
        for row in 0..rows {
            let x = &input[row * cols..(row + 1) * cols];
            let dy = &d_output[row * cols..(row + 1) * cols];
            let mean = x.iter().sum::<f64>() / cols as f64;
            let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / cols as f64;
            let inverse_std = 1.0 / (variance + epsilon).sqrt();
            let normalized: Vec<f64> = x.iter().map(|v| (v - mean) * inverse_std).collect();
            let mut dx_hat = self.zeroes(dy.len())?;
            multiply(dy, gamma, &mut dx_hat);
            let sum_dx_hat: f64 = dx_hat.iter().sum();
            let sum_product: f64 = dx_hat.iter().zip(&normalized).map(|(a, b)| a * b).sum();
            for col in 0..cols {
                d_input[row * cols + col] = inverse_std / cols as f64
                    * (cols as f64 * dx_hat[col] - sum_dx_hat - normalized[col] * sum_product);
                d_gamma[col] += dy[col] * normalized[col];
                d_beta[col] += dy[col];
            }
        }
        Ok(())
    }

    fn matmul(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        rows: usize,
        inner: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= rows.checked_mul(cols).expect("shape overflow"));
        debug_assert!(check_matrix(a, rows, inner));
        debug_assert!(check_matrix(b, inner, cols));
        out.fill(0.0);
        for row in 0..rows {
            for col in 0..cols {
                for k in 0..inner {
                    out[row * cols + col] += a[row * inner + k] * b[k * cols + col];
                }
            }
        }
        Ok(())
    }

    fn matmul_backward(
        &self,
        a: &Self::Buffer,
        b: &Self::Buffer,
        d_output: &Self::Buffer,
        rows: usize,
        inner: usize,
        cols: usize,
        d_a: &mut Self::Buffer,
        d_b: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(d_a.len() >= rows.checked_mul(inner).expect("shape overflow"));
        debug_assert!(d_b.len() >= inner.checked_mul(cols).expect("shape overflow"));
        debug_assert!(check_matrix(d_output, rows, cols));
        // TODO optimize
        let mut transp_a = self.zeroes(b.len())?;
        let mut transp_b = self.zeroes(a.len())?;
        self.transpose(b, inner, cols, &mut transp_a)?;
        self.transpose(a, rows, inner, &mut transp_b)?;
        self.matmul(d_output, &transp_a, rows, cols, inner, d_a)?;
        self.matmul(&transp_b, d_output, inner, rows, cols, d_b)?;
        Ok(())
    }

    fn relu(&self, input: &Self::Buffer, out: &mut Self::Buffer) -> anyhow::Result<()> {
        debug_assert!(out.len() >= input.len());
        out.fill(0.0);
        for i in 0..input.len() {
            out[i] = input[i].max(0.0);
        }
        Ok(())
    }

    fn relu_backward(
        &self,
        input: &Self::Buffer,
        d_output: &Self::Buffer,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= input.len());
        debug_assert_eq!(input.len(), d_output.len());
        out.fill(0.0);
        for i in 0..input.len() {
            out[i] = if input[i] > 0.0 { d_output[i] } else { 0.0 };
        }
        Ok(())
    }

    fn scale(
        &self,
        input: &Self::Buffer,
        factor: f64,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= input.len());
        out.fill(0.0);
        for i in 0..input.len() {
            out[i] = input[i] * factor;
        }
        Ok(())
    }

    fn scale_in_place(&self, out: &mut Self::Buffer, factor: f64) -> anyhow::Result<()> {
        for o in out {
            *o *= factor;
        }
        Ok(())
    }

    fn slice_columns(
        &self,
        input: &Self::Buffer,
        rows: usize,
        cols: usize,
        start: usize,
        width: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(check_matrix(input, rows, cols));
        debug_assert!(width > 0 && start.checked_add(width).is_some_and(|end| end <= cols));
        debug_assert!(out.len() >= rows.checked_mul(width).expect("shape overflow"));
        for row in 0..rows {
            out[row * width..(row + 1) * width]
                .copy_from_slice(&input[row * cols + start..row * cols + start + width]);
        }
        Ok(())
    }

    fn softmax(&self, input: &Self::Buffer, out: &mut Self::Buffer) -> anyhow::Result<()> {
        debug_assert!(out.len() >= input.len());
        debug_assert!(!input.is_empty(), "softmax requires a nonempty row");
        debug_assert!(
            input
                .iter()
                .all(|x| x.is_finite() || *x == f64::NEG_INFINITY)
        );
        let max = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        debug_assert!(max.is_finite(), "softmax requires an unmasked entry");
        let mut exp = self.zeroes(input.len())?;
        let mut sum = 0.0;
        for i in 0..input.len() {
            exp[i] = (input[i] - max).exp();
            sum += exp[i];
        }
        self.scale(&exp, 1.0 / sum, out)
    }

    fn softmax_in_place(&self, out: &mut Self::Buffer) -> anyhow::Result<()> {
        debug_assert!(!out.is_empty(), "softmax requires a nonempty row");
        debug_assert!(out.iter().all(|x| x.is_finite() || *x == f64::NEG_INFINITY));
        let max = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        debug_assert!(max.is_finite(), "softmax requires an unmasked entry");
        let mut exp = self.zeroes(out.len())?;
        let mut sum = 0.0;
        for i in 0..out.len() {
            exp[i] = (out[i] - max).exp();
            sum += exp[i];
        }
        self.scale(&exp, 1.0 / sum, out)
    }

    fn softmax_rows(
        &self,
        input: &Self::Buffer,
        rows: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= input.len());
        debug_assert!(check_matrix(input, rows, cols));
        out.fill(0.0);
        // TODO maybe unsafe? or make Self::Buffer a DerefMut of [f64]
        let mut buf_inp = self.zeroes(cols)?;
        let mut buf_out = self.zeroes(cols)?;
        let mut i = 0;
        for chunk in input.chunks_exact(cols) {
            buf_inp[..cols].copy_from_slice(chunk);
            self.softmax(&buf_inp, &mut buf_out)?;
            out[i..i + cols].copy_from_slice(buf_out.as_slice());
            i += cols;
        }
        Ok(())
    }

    fn softmax_rows_in_place(
        &self,
        out: &mut Self::Buffer,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<()> {
        debug_assert!(check_matrix(out, rows, cols));
        // TODO maybe unsafe? or make Self::Buffer a DerefMut of [f64]
        let input = out.clone();
        let mut buf_inp = self.zeroes(cols)?;
        let mut buf_out = self.zeroes(cols)?;
        let mut i = 0;
        for chunk in input.chunks_exact(cols) {
            buf_inp[..cols].copy_from_slice(chunk);
            self.softmax(&buf_inp, &mut buf_out)?;
            out[i..i + cols].copy_from_slice(buf_out.as_slice());
            i += cols;
        }
        Ok(())
    }

    fn softmax_rows_backward(
        &self,
        probabilities: &Self::Buffer,
        d_output: &Self::Buffer,
        rows: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= probabilities.len());
        debug_assert!(check_matrix(probabilities, rows, cols));
        debug_assert!(check_matrix(d_output, rows, cols));
        out.fill(0.0);
        // TODO maybe unsafe? or make Self::Buffer a DerefMut of [f64]
        let mut buf_pro = self.zeroes(cols)?;
        let mut buf_dou = self.zeroes(cols)?;
        let mut buf_out = self.zeroes(cols)?;
        let mut i = 0;
        for (chunk_pro, chunk_dou) in probabilities
            .chunks_exact(cols)
            .zip(d_output.chunks_exact(cols))
        {
            buf_pro[..cols].copy_from_slice(chunk_pro);
            buf_dou[..cols].copy_from_slice(chunk_dou);
            softmax_backward(&buf_pro, &buf_dou, &mut buf_out);
            out[i..i + cols].copy_from_slice(buf_out.as_slice());
            i += cols;
        }
        Ok(())
    }

    fn sum_rows(
        &self,
        input: &Self::Buffer,
        rows: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= cols);
        debug_assert!(check_matrix(input, rows, cols));
        out.fill(0.0);
        for row in input.chunks_exact(cols) {
            for (sum, value) in out.iter_mut().zip(row) {
                *sum += value;
            }
        }
        Ok(())
    }

    fn transpose(
        &self,
        input: &Self::Buffer,
        rows: usize,
        cols: usize,
        out: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        debug_assert!(out.len() >= input.len());
        debug_assert!(check_matrix(input, rows, cols));
        out.fill(0.0);
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = input[row * cols + col];
            }
        }
        Ok(())
    }

    fn transpose_in_place(
        &self,
        out: &mut Self::Buffer,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<()> {
        debug_assert!(check_matrix(out, rows, cols));
        let input = out.clone();
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = input[row * cols + col];
            }
        }
        Ok(())
    }
}

fn check_matrix(input: &[f64], rows: usize, cols: usize) -> bool {
    debug_assert!(rows > 0 && cols > 0, "matrix dimensions must be positive");
    debug_assert_eq!(input.len(), rows.checked_mul(cols).expect("shape overflow"));
    true
}

fn multiply(a: &[f64], b: &[f64], out: &mut [f64]) {
    debug_assert!(out.len() >= a.len());
    debug_assert_eq!(a.len(), b.len());
    out.fill(0.0);
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

fn multiply_in_place(out: &mut [f64], b: &[f64]) {
    debug_assert_eq!(out.len(), b.len());
    for i in 0..out.len() {
        out[i] *= b[i];
    }
}

fn softmax_backward(probabilities: &[f64], d_output: &[f64], out: &mut [f64]) {
    debug_assert!(out.len() >= probabilities.len());
    debug_assert_eq!(probabilities.len(), d_output.len());
    let dot: f64 = probabilities.iter().zip(d_output).map(|(p, d)| p * d).sum();
    out.fill(0.0);
    for i in 0..probabilities.len() {
        out[i] = probabilities[i] * (d_output[i] - dot);
    }
}

#[test]
fn naive_full_suite() {
    super::tests::full_suite(&Naive)
}
