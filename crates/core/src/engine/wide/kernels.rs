use ::wide::f32x8;
use alloc::vec::Vec;
use rand::{RngExt as _, SeedableRng as _, rngs::StdRng};

use crate::{
    engine::{
        AdamWConfig, AdamWGroup, BlockSpec, ForwardMode, LinearSpec, NormSpec, NormWeights,
        wide::{SimdWide, SimdWideBlockCache, SimdWideLayerNormCache},
    },
    utils,
};

const LANES: usize = 8;

// Array conversion supports unaligned row starts without casts or unsafe code.
#[inline]
fn load(values: &[f32]) -> f32x8 {
    f32x8::new(values[..LANES].try_into().unwrap())
}

#[inline]
fn store(out: &mut [f32], value: f32x8) {
    out[..LANES].copy_from_slice(&value.to_array());
}

#[inline]
fn map_in_place(values: &mut [f32], vector: impl Fn(f32x8) -> f32x8, scalar: impl Fn(f32) -> f32) {
    let mut chunks = values.chunks_exact_mut(LANES);
    for chunk in &mut chunks {
        let result = vector(load(chunk));
        store(chunk, result);
    }
    for value in chunks.into_remainder() {
        *value = scalar(*value);
    }
}

#[inline]
fn zip_assign(
    out: &mut [f32],
    values: &[f32],
    vector: impl Fn(f32x8, f32x8) -> f32x8,
    scalar: impl Fn(f32, f32) -> f32,
) {
    debug_assert_eq!(out.len(), values.len());
    let end = out.len() / LANES * LANES;
    for i in (0..end).step_by(LANES) {
        let result = vector(load(&out[i..]), load(&values[i..]));
        store(&mut out[i..], result);
    }
    for i in end..out.len() {
        out[i] = scalar(out[i], values[i]);
    }
}

impl SimdWide {
    pub(super) fn add_assign(out: &mut [f32], values: &[f32]) {
        zip_assign(out, values, |a, b| a + b, |a, b| a + b);
    }

    fn scaled_add(out: &mut [f32], values: &[f32], scale: f32) {
        let scale_v = f32x8::splat(scale);
        zip_assign(out, values, |a, b| a + scale_v * b, |a, b| a + scale * b);
    }

    pub(super) fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
        out.copy_from_slice(a);
        Self::add_assign(out, b);
    }

    pub(super) fn relu(values: &mut [f32]) {
        map_in_place(values, |v| v.max(f32x8::ZERO), |v| v.max(0.0));
    }

    pub(super) fn relu_backward(values: &[f32], gradient: &mut [f32]) {
        zip_assign(
            gradient,
            values,
            |g, x| x.simd_le(f32x8::ZERO).select(f32x8::ZERO, g),
            |g, x| if x <= 0.0 { 0.0 } else { g },
        );
    }

    fn sum(values: &[f32]) -> f32 {
        let mut total = f32x8::ZERO;
        let mut chunks = values.chunks_exact(LANES);
        for chunk in &mut chunks {
            total += load(chunk);
        }
        chunks
            .remainder()
            .iter()
            .fold(total.reduce_add(), |a, b| a + b)
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let end = a.len() / LANES * LANES;
        let mut total = f32x8::ZERO;
        for i in (0..end).step_by(LANES) {
            total += load(&a[i..]) * load(&b[i..]);
        }
        let mut sum = total.reduce_add();
        for i in end..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    pub(super) fn maximum(values: &[f32]) -> f32 {
        let mut maximum = f32x8::splat(f32::NEG_INFINITY);
        let mut chunks = values.chunks_exact(LANES);
        for chunk in &mut chunks {
            maximum = maximum.max(load(chunk));
        }
        maximum
            .to_array()
            .into_iter()
            .chain(chunks.remainder().iter().copied())
            .fold(f32::NEG_INFINITY, f32::max)
    }

    pub(super) fn all_finite(values: &[f32]) -> bool {
        let mut chunks = values.chunks_exact(LANES);
        chunks.by_ref().all(|chunk| load(chunk).is_finite().all())
            && chunks.remainder().iter().all(|v| v.is_finite())
    }

    pub(super) fn exp_chunk(values: &[f32], max: f32) -> [f32; LANES] {
        let mut result = [0.0; LANES];
        if values.len() == LANES {
            let shifted = load(values) - f32x8::splat(max);
            result = shifted.exp().to_array();
            // Preserve scalar subnormal/underflow behavior outside the usual
            // softmax range, including subtraction overflowing to -infinity.
            if shifted.simd_lt(f32x8::splat(-80.0)).any() {
                for (out, x) in result.iter_mut().zip(shifted.to_array()) {
                    if x < -80.0 {
                        *out = utils::expf(x);
                    }
                }
            }
        } else {
            for (out, &x) in result.iter_mut().zip(values) {
                *out = utils::expf(x - max);
            }
        }
        result
    }

    pub(super) fn exp_sum(values: &[f32], max: f32) -> f32 {
        let mut total = f32x8::ZERO;
        let mut chunks = values.chunks_exact(LANES);
        for chunk in &mut chunks {
            total += f32x8::new(Self::exp_chunk(chunk, max));
        }
        chunks
            .remainder()
            .iter()
            .fold(total.reduce_add(), |sum, &x| sum + utils::expf(x - max))
    }

    pub(super) fn linear_forward(
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
                Self::scaled_add(
                    y,
                    &w[i * s.outputs..(i + 1) * s.outputs],
                    x[row * s.inputs + i],
                );
            }
        }
    }

    pub(super) fn linear_backward(
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
            let gradient = &dy[row * s.outputs..(row + 1) * s.outputs];
            for i in 0..s.inputs {
                let start = i * s.outputs;
                dx[row * s.inputs + i] = Self::dot(gradient, &w[start..start + s.outputs]);
                Self::scaled_add(
                    &mut dw[start..start + s.outputs],
                    gradient,
                    x[row * s.inputs + i],
                );
            }
        }
        if let Some(db) = db {
            db.fill(0.0);
            for row in dy.chunks_exact(s.outputs) {
                Self::add_assign(db, row);
            }
        }
    }

    pub(super) fn norm_forward(
        s: NormSpec,
        w: NormWeights<'_, Vec<f32>>,
        x: &[f32],
        out: &mut [f32],
        mut cache: Option<&mut SimdWideLayerNormCache>,
    ) {
        if let Some(saved) = cache.as_deref_mut() {
            saved.spec = Some(s);
            saved.normalized.resize(x.len(), 0.0);
            saved.inverse_std.resize(s.rows, 0.0);
        }
        let end = s.channels / LANES * LANES;
        for row in 0..s.rows {
            let start = row * s.channels;
            let values = &x[start..start + s.channels];
            let mean = Self::sum(values) / s.channels as f32;
            let mean_v = f32x8::splat(mean);
            let mut variance_v = f32x8::ZERO;
            for i in (0..end).step_by(LANES) {
                let centered = load(&values[i..]) - mean_v;
                variance_v += centered * centered;
            }
            let variance = values[end..]
                .iter()
                .fold(variance_v.reduce_add(), |sum, &v| {
                    sum + (v - mean) * (v - mean)
                });
            let inv = 1.0 / utils::sqrtf(variance / s.channels as f32 + s.epsilon);
            for i in (0..end).step_by(LANES) {
                let normalized = (load(&values[i..]) - mean_v) * f32x8::splat(inv);
                store(
                    &mut out[start + i..],
                    normalized * load(&w.gamma[i..]) + load(&w.beta[i..]),
                );
                if let Some(saved) = cache.as_deref_mut() {
                    store(&mut saved.normalized[start + i..], normalized);
                }
            }
            for i in end..s.channels {
                let normalized = (values[i] - mean) * inv;
                out[start + i] = normalized * w.gamma[i] + w.beta[i];
                if let Some(saved) = cache.as_deref_mut() {
                    saved.normalized[start + i] = normalized;
                }
            }
            if let Some(saved) = cache.as_deref_mut() {
                saved.inverse_std[row] = inv;
            }
        }
    }

    pub(super) fn norm_backward(
        s: NormSpec,
        gamma: &[f32],
        cache: &SimdWideLayerNormCache,
        dy: &[f32],
        dx: &mut [f32],
        dg: &mut [f32],
        db: &mut [f32],
    ) {
        debug_assert_eq!(cache.spec, Some(s));
        dg.fill(0.0);
        db.fill(0.0);
        let end = s.channels / LANES * LANES;
        let channels = f32x8::splat(s.channels as f32);
        for row in 0..s.rows {
            let start = row * s.channels;
            let normalized = &cache.normalized[start..start + s.channels];
            let gradient = &dy[start..start + s.channels];
            let mut sum_v = f32x8::ZERO;
            let mut product_v = f32x8::ZERO;
            for i in (0..end).step_by(LANES) {
                let g = load(&gradient[i..]) * load(&gamma[i..]);
                sum_v += g;
                product_v += g * load(&normalized[i..]);
            }
            let mut sum = sum_v.reduce_add();
            let mut product = product_v.reduce_add();
            for i in end..s.channels {
                let g = gradient[i] * gamma[i];
                sum += g;
                product += g * normalized[i];
            }
            let scale = cache.inverse_std[row] / s.channels as f32;
            for i in (0..end).step_by(LANES) {
                let g = load(&gradient[i..]);
                let n = load(&normalized[i..]);
                let value = channels * g * load(&gamma[i..])
                    - f32x8::splat(sum)
                    - n * f32x8::splat(product);
                store(&mut dx[start + i..], f32x8::splat(scale) * value);
                let dg_v = load(&dg[i..]) + g * n;
                let db_v = load(&db[i..]) + g;
                store(&mut dg[i..], dg_v);
                store(&mut db[i..], db_v);
            }
            for i in end..s.channels {
                dx[start + i] = scale
                    * (s.channels as f32 * gradient[i] * gamma[i] - sum - normalized[i] * product);
                dg[i] += gradient[i] * normalized[i];
                db[i] += gradient[i];
            }
        }
    }

    fn dropout_rng(mode: ForwardMode, probability: f32, stream: u64) -> Option<StdRng> {
        mode.seed
            .filter(|_| probability > 0.0)
            .map(|seed| StdRng::seed_from_u64(seed ^ stream.wrapping_mul(0x9e3779b97f4a7c15)))
    }

    fn dropout_value(rng: &mut Option<StdRng>, probability: f32) -> f32 {
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

    pub(super) fn dropout_add(
        branch: &mut [f32],
        residual: &[f32],
        probability: f32,
        mode: ForwardMode,
        stream: u64,
        mut mask: Option<&mut Vec<f32>>,
    ) {
        let mut rng = Self::dropout_rng(mode, probability, stream);
        if let Some(mask) = mask.as_deref_mut() {
            if rng.is_some() {
                mask.resize(branch.len(), 0.0);
            } else {
                mask.clear();
            }
        }
        if rng.is_none() {
            Self::add_assign(branch, residual);
            return;
        }
        for (chunk_index, chunk) in branch.chunks_mut(LANES).enumerate() {
            let start = chunk_index * LANES;
            let mut multipliers = [0.0; LANES];
            for multiplier in &mut multipliers[..chunk.len()] {
                *multiplier = Self::dropout_value(&mut rng, probability);
            }
            if let Some(mask) = mask.as_deref_mut() {
                mask[start..start + chunk.len()].copy_from_slice(&multipliers[..chunk.len()]);
            }
            if chunk.len() == LANES {
                let result = load(chunk) * f32x8::new(multipliers) + load(&residual[start..]);
                store(chunk, result);
            } else {
                for i in 0..chunk.len() {
                    chunk[i] = chunk[i] * multipliers[i] + residual[start + i];
                }
            }
        }
    }

    pub(super) fn masked(input: &[f32], mask: &[f32], out: &mut [f32]) {
        out.copy_from_slice(input);
        if !mask.is_empty() {
            zip_assign(out, mask, |x, m| x * m, |x, m| x * m);
        }
    }

    pub(super) fn attention_forward(
        s: BlockSpec,
        qkv: &[f32],
        mode: ForwardMode,
        scores: &mut [f32],
        out: &mut [f32],
        mut cache: Option<&mut SimdWideBlockCache>,
    ) {
        let c = s.channels;
        let d = c / s.heads;
        let scale = 1.0 / utils::sqrtf(d as f32);
        let mut rng = Self::dropout_rng(mode, s.dropout, 0);
        if let Some(saved) = cache.as_deref_mut() {
            let count = s.batch * s.heads * s.time * s.time;
            saved.probabilities.resize(count, 0.0);
            saved.probabilities.fill(0.0);
            if rng.is_some() {
                saved.attention_mask.resize(count, 0.0);
                saved.attention_mask.fill(0.0);
            } else {
                saved.attention_mask.clear();
            }
        }
        out.fill(0.0);
        for b in 0..s.batch {
            for h in 0..s.heads {
                for t in 0..s.time {
                    let q = (b * s.time + t) * 3 * c + h * d;
                    let scores = &mut scores[..=t];
                    for (k, score) in scores.iter_mut().enumerate() {
                        let key = (b * s.time + k) * 3 * c + c + h * d;
                        *score = Self::dot(&qkv[q..q + d], &qkv[key..key + d]) * scale;
                    }
                    let max = Self::maximum(scores);
                    for chunk in scores.chunks_mut(LANES) {
                        let masses = Self::exp_chunk(chunk, max);
                        chunk.copy_from_slice(&masses[..chunk.len()]);
                    }
                    let total = Self::sum(scores);
                    for (k, &score) in scores.iter().enumerate() {
                        let p = score / total;
                        let mask = Self::dropout_value(&mut rng, s.dropout);
                        if let Some(saved) = cache.as_deref_mut() {
                            let offset = ((b * s.heads + h) * s.time + t) * s.time + k;
                            saved.probabilities[offset] = p;
                            if !saved.attention_mask.is_empty() {
                                saved.attention_mask[offset] = mask;
                            }
                        }
                        let value = (b * s.time + k) * 3 * c + 2 * c + h * d;
                        let output = (b * s.time + t) * c + h * d;
                        Self::scaled_add(
                            &mut out[output..output + d],
                            &qkv[value..value + d],
                            p * mask,
                        );
                    }
                }
            }
        }
    }

    pub(super) fn attention_backward(
        s: BlockSpec,
        cache: &SimdWideBlockCache,
        dy: &[f32],
        scores: &mut [f32],
        dqkv: &mut [f32],
    ) {
        let c = s.channels;
        let d = c / s.heads;
        let scale = 1.0 / utils::sqrtf(d as f32);
        dqkv.fill(0.0);
        for b in 0..s.batch {
            for h in 0..s.heads {
                for t in 0..s.time {
                    let q = (b * s.time + t) * 3 * c + h * d;
                    let output = (b * s.time + t) * c + h * d;
                    let offset = ((b * s.heads + h) * s.time + t) * s.time;
                    let gradient = &dy[output..output + d];
                    let scores = &mut scores[..=t];
                    for (k, score) in scores.iter_mut().enumerate() {
                        let value = (b * s.time + k) * 3 * c + 2 * c + h * d;
                        let p = cache.probabilities[offset + k];
                        let mask = cache.attention_mask.get(offset + k).copied().unwrap_or(1.0);
                        *score = Self::dot(gradient, &cache.qkv[value..value + d]) * mask;
                        Self::scaled_add(&mut dqkv[value..value + d], gradient, p * mask);
                    }
                    let dot = Self::dot(&cache.probabilities[offset..offset + t + 1], scores);
                    for (k, &score) in scores.iter().enumerate() {
                        let key = (b * s.time + k) * 3 * c + c + h * d;
                        let ds = cache.probabilities[offset + k] * (score - dot) * scale;
                        Self::scaled_add(&mut dqkv[q..q + d], &cache.qkv[key..key + d], ds);
                        Self::scaled_add(&mut dqkv[key..key + d], &cache.qkv[q..q + d], ds);
                    }
                }
            }
        }
    }

    pub(super) fn loss(
        logits: &[f32],
        targets: &[usize],
        rows: usize,
        vocab: usize,
        mut gradient: Option<&mut Vec<f32>>,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(
            Self::all_finite(logits),
            "non-finite logits; reduce the learning rate"
        );
        let mut loss = 0.0;
        let scale = 1.0 / rows as f32;
        for (r, row) in logits.chunks_exact(vocab).enumerate() {
            let max = Self::maximum(row);
            let total = Self::exp_sum(row, max);
            loss += ((max - row[targets[r]]) + utils::logf(total)) * scale;
            if let Some(g) = gradient.as_deref_mut() {
                let g = &mut g[r * vocab..(r + 1) * vocab];
                for (chunk_index, chunk) in row.chunks(LANES).enumerate() {
                    let start = chunk_index * LANES;
                    let mass = Self::exp_chunk(chunk, max);
                    let mut target = [0.0; LANES];
                    if (start..start + chunk.len()).contains(&targets[r]) {
                        target[targets[r] - start] = 1.0;
                    }
                    if chunk.len() == LANES {
                        store(
                            &mut g[start..],
                            (f32x8::new(mass) / f32x8::splat(total) - f32x8::new(target))
                                * f32x8::splat(scale),
                        );
                    } else {
                        for i in 0..chunk.len() {
                            g[start + i] = (mass[i] / total - target[i]) * scale;
                        }
                    }
                }
            }
        }
        anyhow::ensure!(
            loss.is_finite(),
            "non-finite loss; reduce the learning rate"
        );
        Ok(loss)
    }

    pub(super) fn adamw(
        groups: &mut [AdamWGroup<'_, Vec<f32>>],
        config: AdamWConfig,
    ) -> anyhow::Result<()> {
        let correction1 = 1.0 - utils::powi(config.beta1, config.step);
        let correction2 = 1.0 - utils::powi(config.beta2, config.step);
        let b1 = f32x8::splat(config.beta1);
        let b2 = f32x8::splat(config.beta2);
        let decay = 1.0 - config.learning_rate * config.weight_decay;
        for g in groups {
            let end = g.values.len() / LANES * LANES;
            for i in (0..end).step_by(LANES) {
                let gradient = load(&g.gradients[i..]);
                let m =
                    b1 * load(&g.first_moment[i..]) + f32x8::splat(1.0 - config.beta1) * gradient;
                let v = b2 * load(&g.second_moment[i..])
                    + f32x8::splat(1.0 - config.beta2) * gradient * gradient;
                let denominator =
                    (v / f32x8::splat(correction2)).sqrt() + f32x8::splat(config.epsilon);
                let value = load(&g.values[i..]) * f32x8::splat(decay)
                    - f32x8::splat(config.learning_rate) * (m / f32x8::splat(correction1))
                        / denominator;
                anyhow::ensure!(
                    (value.is_finite() & m.is_finite() & v.is_finite()).all(),
                    "non-finite optimizer update; reduce the learning rate"
                );
                store(&mut g.values[i..], value);
                store(&mut g.first_moment[i..], m);
                store(&mut g.second_moment[i..], v);
            }
            for i in end..g.values.len() {
                let gradient = g.gradients[i];
                let m = config.beta1 * g.first_moment[i] + (1.0 - config.beta1) * gradient;
                let v =
                    config.beta2 * g.second_moment[i] + (1.0 - config.beta2) * gradient * gradient;
                let denominator = utils::sqrtf(v / correction2) + config.epsilon;
                let value =
                    g.values[i] * decay - config.learning_rate * (m / correction1) / denominator;
                anyhow::ensure!(
                    value.is_finite() && m.is_finite() && v.is_finite(),
                    "non-finite optimizer update; reduce the learning rate"
                );
                g.values[i] = value;
                g.first_moment[i] = m;
                g.second_moment[i] = v;
            }
        }
        Ok(())
    }
}
