use rand::{RngExt as _, SeedableRng as _, rngs::StdRng};

use crate::engine::{
    BlockSpec, ForwardMode, LinearSpec, NormSpec, NormWeights,
    naive::{Naive, NaiveBlockCache, NaiveLayerNormCache},
};

impl Naive {
    pub fn linear_forward(
        s: LinearSpec,
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        for row in 0..s.rows {
            let y = &mut out[row * s.outputs..(row + 1) * s.outputs];

            match bias {
                Some(b) => y.copy_from_slice(b),
                None => y.fill(0.0),
            }

            for i in 0..s.inputs {
                let value = x[row * s.inputs + i];

                for j in 0..s.outputs {
                    y[j] += value * w[i * s.outputs + j];
                }
            }
        }
    }

    pub fn linear_backward(
        s: LinearSpec,
        x: &[f32],
        w: &[f32],
        dy: &[f32],
        dx: &mut [f32],
        dw: &mut [f32],
        db: Option<&mut [f32]>,
    ) {
        dx.fill(0.0);
        dw.fill(0.0);

        for row in 0..s.rows {
            for i in 0..s.inputs {
                for j in 0..s.outputs {
                    let gradient = dy[row * s.outputs + j];

                    dx[row * s.inputs + i] += gradient * w[i * s.outputs + j];
                    dw[i * s.outputs + j] += x[row * s.inputs + i] * gradient;
                }
            }
        }

        if let Some(db) = db {
            db.fill(0.0);

            for row in dy.chunks_exact(s.outputs) {
                for (sum, &value) in db.iter_mut().zip(row) {
                    *sum += value;
                }
            }
        }
    }

    pub fn norm_forward(
        s: NormSpec,
        w: NormWeights<'_, Vec<f32>>,
        x: &[f32],
        out: &mut [f32],
        mut cache: Option<&mut NaiveLayerNormCache>,
    ) {
        if let Some(cache) = cache.as_deref_mut() {
            cache.spec = Some(s);
            cache.normalized.resize(x.len(), 0.0);
            cache.inverse_std.resize(s.rows, 0.0);
        }

        for row in 0..s.rows {
            let start = row.saturating_mul(s.channels);
            let values = &x[start..start.saturating_add(s.channels)];
            let mean = values.iter().sum::<f32>() / s.channels as f32;
            let variance =
                values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / s.channels as f32;
            let inv = 1.0 / (variance + s.epsilon).sqrt();

            for ch in 0..s.channels {
                let normalized = (values[ch] - mean) * inv;

                out[start + ch] = normalized * w.gamma[ch] + w.beta[ch];

                if let Some(cache) = cache.as_deref_mut() {
                    cache.normalized[start + ch] = normalized;
                }
            }

            if let Some(cache) = cache.as_deref_mut() {
                cache.inverse_std[row] = inv;
            }
        }
    }

    pub fn norm_backward(
        s: NormSpec,
        gamma: &[f32],
        cache: &NaiveLayerNormCache,
        dy: &[f32],
        dx: &mut [f32],
        dg: &mut [f32],
        db: &mut [f32],
    ) {
        dg.fill(0.0);
        db.fill(0.0);

        for row in 0..s.rows {
            let start = row.saturating_mul(s.channels);
            let mut sum = 0.0;
            let mut product = 0.0;

            for ch in 0..s.channels {
                let gradient = dy[start + ch] * gamma[ch];

                sum += gradient;
                product += gradient * cache.normalized[start + ch];
            }

            for ch in 0..s.channels {
                let normalized = cache.normalized[start + ch];
                let dx_arg = normalized * product;
                let dx_arg = s.channels as f32 * dy[start + ch] * gamma[ch] - sum - dx_arg;

                dx[start + ch] = cache.inverse_std[row] / s.channels as f32 * dx_arg;
                dg[ch] += dy[start + ch] * normalized;
                db[ch] += dy[start + ch];
            }
        }
    }

    pub fn dropout_rng(mode: ForwardMode, probability: f32, stream: u64) -> Option<StdRng> {
        mode.seed
            .filter(|_| probability > 0.0)
            .map(|seed| StdRng::seed_from_u64(seed ^ stream.wrapping_mul(0x9e3779b97f4a7c15)))
    }

    pub fn dropout_value(rng: &mut Option<StdRng>, probability: f32) -> f32 {
        match rng {
            Some(rng) => {
                if rng.random::<f32>() < probability {
                    0.0
                } else {
                    1.0 / (1.0 - probability)
                }
            }
            None => 1.0,
        }
    }

    pub fn dropout_add(
        branch: &mut [f32],
        residual: &[f32],
        probability: f32,
        mode: ForwardMode,
        stream: u64,
        mut mask: Option<&mut Vec<f32>>,
    ) {
        let mut rng = Self::dropout_rng(mode, probability, stream);

        if let Some(mask) = mask.as_deref_mut() {
            match rng {
                Some(_) => mask.resize(branch.len(), 0.0),
                None => mask.clear(),
            }
        }
        for i in 0..branch.len() {
            let m = Self::dropout_value(&mut rng, probability);

            branch[i] = branch[i] * m + residual[i];

            if let Some(mask) = mask.as_deref_mut().filter(|m| !m.is_empty()) {
                mask[i] = m;
            }
        }
    }

    pub fn masked(input: &[f32], mask: &[f32], out: &mut [f32]) {
        for i in 0..input.len() {
            out[i] = input[i] * mask.get(i).copied().unwrap_or(1.0);
        }
    }

    pub fn attention_forward(
        s: BlockSpec,
        qkv: &[f32],
        mode: ForwardMode,
        scores: &mut [f32],
        out: &mut [f32],
        mut cache: Option<&mut NaiveBlockCache>,
    ) {
        let c = s.channels;
        let d = c / s.heads;
        let scale = 1.0 / (d as f32).sqrt();

        let mut rng = Self::dropout_rng(mode, s.dropout, 0);

        if let Some(cache) = cache.as_deref_mut() {
            let count = s
                .batch
                .saturating_mul(s.heads)
                .saturating_mul(s.time)
                .saturating_mul(s.time);

            cache.probabilities.resize(count, 0.0);
            cache.probabilities.fill(0.0);

            if rng.is_some() {
                cache.attention_mask.resize(count, 0.0);
                cache.attention_mask.fill(0.0);
            } else {
                cache.attention_mask.clear();
            }
        }

        out.fill(0.0);

        for b in 0..s.batch {
            for h in 0..s.heads {
                for t in 0..s.time {
                    let q = (b * s.time + t) * 3 * c + h * d;
                    let mut max = f32::NEG_INFINITY;
                    let mut total = 0.0;

                    for (k, score) in scores.iter_mut().enumerate().take(t + 1) {
                        let key = (b * s.time + k) * 3 * c + c + h * d;

                        *score = (0..d).map(|ch| qkv[q + ch] * qkv[key + ch]).sum::<f32>() * scale;
                        max = max.max(*score);
                    }

                    for score in &mut scores[..=t] {
                        *score = (*score - max).exp();
                        total += *score;
                    }

                    for (k, &score) in scores.iter().enumerate().take(t + 1) {
                        let p = score / total;
                        let mask = Self::dropout_value(&mut rng, s.dropout);

                        if let Some(cache) = cache.as_deref_mut() {
                            let offset = ((b * s.heads + h) * s.time + t) * s.time + k;

                            cache.probabilities[offset] = p;

                            if !cache.attention_mask.is_empty() {
                                cache.attention_mask[offset] = mask;
                            }
                        }

                        let value = (b * s.time + k) * 3 * c + 2 * c + h * d;
                        let output = (b * s.time + t) * c + h * d;

                        for ch in 0..d {
                            out[output + ch] += p * mask * qkv[value + ch];
                        }
                    }
                }
            }
        }
    }

    pub fn attention_backward(
        s: BlockSpec,
        cache: &NaiveBlockCache,
        dy: &[f32],
        scores: &mut [f32],
        dqkv: &mut [f32],
    ) {
        let c = s.channels;
        let d = c / s.heads;
        let scale = 1.0 / (d as f32).sqrt();

        dqkv.fill(0.0);

        for b in 0..s.batch {
            for h in 0..s.heads {
                for t in 0..s.time {
                    let q = (b * s.time + t) * 3 * c + h * d;
                    let output = (b * s.time + t) * c + h * d;
                    let offset = ((b * s.heads + h) * s.time + t) * s.time;
                    let mut dot = 0.0;

                    for (k, score) in scores.iter_mut().enumerate().take(t + 1) {
                        let value = (b * s.time + k) * 3 * c + 2 * c + h * d;
                        let p = cache.probabilities[offset + k];
                        let mask = cache.attention_mask.get(offset + k).copied().unwrap_or(1.0);

                        *score = (0..d)
                            .map(|ch| dy[output + ch] * cache.qkv[value + ch])
                            .sum::<f32>()
                            * mask;

                        dot += p * *score;

                        for ch in 0..d {
                            dqkv[value + ch] += p * mask * dy[output + ch];
                        }
                    }
                    for (k, &score) in scores.iter().enumerate().take(t + 1) {
                        let key = (b * s.time + k) * 3 * c + c + h * d;
                        let ds = cache.probabilities[offset + k] * (score - dot) * scale;

                        for ch in 0..d {
                            dqkv[q + ch] += ds * cache.qkv[key + ch];
                            dqkv[key + ch] += ds * cache.qkv[q + ch];
                        }
                    }
                }
            }
        }
    }

    pub fn loss(
        logits: &[f32],
        targets: &[usize],
        rows: usize,
        vocab: usize,
        mut gradient: Option<&mut Vec<f32>>,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logits; reduce the learning rate"
        );

        let mut loss = 0.0;
        let scale = 1.0 / rows as f32;

        for (r, row) in logits.chunks_exact(vocab).enumerate() {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f32 = row.iter().map(|v| (v - max).exp()).sum();

            loss += ((max - row[targets[r]]) + total.ln()) * scale;

            if let Some(g) = gradient.as_deref_mut() {
                for j in 0..vocab {
                    let arg = (row[j] - max).exp() / total;

                    g[r * vocab + j] = (arg - f32::from(j == targets[r])) * scale;
                }
            }
        }
        anyhow::ensure!(
            loss.is_finite(),
            "non-finite loss; reduce the learning rate"
        );

        Ok(loss)
    }
}
