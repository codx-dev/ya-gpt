mod kernels;
mod types;

pub use types::*;

use crate::engine::{
    AdamWConfig, AdamWGroup, BlockGradients, BlockSpec, BlockWeights, EmbeddingSpec, Engine,
    ForwardMode, LinearGradients, LinearSpec, LinearWeights, NormGradients, NormSpec, NormWeights,
};

pub type NaiveBuffer = Vec<f32>;

#[derive(Debug, Default, Clone, Copy)]
pub struct Naive;

#[cfg(test)]
mod tests;

impl Engine for Naive {
    type Buffer = NaiveBuffer;
    type IndexBuffer = Vec<usize>;
    type BlockCache = NaiveBlockCache;
    type LayerNormCache = NaiveLayerNormCache;
    type Workspace = NaiveWorkspace;

    fn zeroes(&self, len: usize) -> anyhow::Result<Self::Buffer> {
        Ok(vec![0.0; len])
    }

    fn filled(&self, value: f32, len: usize) -> anyhow::Result<Self::Buffer> {
        Ok(vec![value; len])
    }

    fn buffer_len(&self, buffer: &Self::Buffer) -> usize {
        buffer.len()
    }

    fn buffer_from_slice(&self, values: &[f32]) -> anyhow::Result<Self::Buffer> {
        Ok(values.to_vec())
    }

    fn buffer_to_vec(&self, buffer: &Self::Buffer) -> anyhow::Result<Vec<f32>> {
        Ok(buffer.clone())
    }

    fn indices_from_slice(&self, values: &[usize]) -> anyhow::Result<Self::IndexBuffer> {
        Ok(values.to_vec())
    }

    fn synchronize(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn block_forward(
        &self,
        spec: BlockSpec,
        weights: BlockWeights<'_, Self::Buffer>,
        input: &Self::Buffer,
        mode: ForwardMode,
        ws: &mut Self::Workspace,
        output: &mut Self::Buffer,
        mut cache: Option<&mut Self::BlockCache>,
    ) -> anyhow::Result<()> {
        let n = spec.elements();
        let c = spec.channels;
        let rows = spec.batch.saturating_mul(spec.time);

        let norm = NormSpec {
            rows,
            channels: c,
            epsilon: 1e-5,
        };

        let linear = |inputs, outputs| LinearSpec {
            rows,
            inputs,
            outputs,
        };

        ws.normalized1.resize(n, 0.0);
        ws.qkv.resize(3 * n, 0.0);
        ws.context.resize(n, 0.0);
        ws.after_attention.resize(n, 0.0);
        ws.normalized2.resize(n, 0.0);
        ws.activated.resize(4 * n, 0.0);
        ws.projected.resize(n, 0.0);
        ws.scores.resize(spec.time, 0.0);

        if let Some(saved) = cache.as_deref_mut() {
            saved.spec = None;
        }

        self.layer_norm_forward(
            norm,
            weights.norm1,
            input,
            &mut ws.normalized1,
            cache.as_deref_mut().map(|saved| &mut saved.norm1),
        )?;

        let w_forward = LinearWeights {
            weight: weights.qkv,
            bias: None,
        };

        self.linear_forward(linear(c, 3 * c), w_forward, &ws.normalized1, &mut ws.qkv)?;

        Self::attention_forward(
            spec,
            &ws.qkv,
            mode,
            &mut ws.scores,
            &mut ws.context,
            cache.as_deref_mut(),
        );

        self.linear_forward(
            linear(c, c),
            weights.attention,
            &ws.context,
            &mut ws.after_attention,
        )?;

        Self::dropout_add(
            &mut ws.after_attention,
            input,
            spec.dropout,
            mode,
            1,
            cache.as_deref_mut().map(|saved| &mut saved.projection_mask),
        );

        self.layer_norm_forward(
            norm,
            weights.norm2,
            &ws.after_attention,
            &mut ws.normalized2,
            cache.as_deref_mut().map(|saved| &mut saved.norm2),
        )?;

        self.linear_forward(
            linear(c, 4 * c),
            weights.expand,
            &ws.normalized2,
            &mut ws.activated,
        )?;

        for x in &mut ws.activated {
            *x = x.max(0.0);
        }

        self.linear_forward(
            linear(4 * c, c),
            weights.project,
            &ws.activated,
            &mut ws.projected,
        )?;

        Self::dropout_add(
            &mut ws.projected,
            &ws.after_attention,
            spec.dropout,
            mode,
            2,
            cache.as_deref_mut().map(|saved| &mut saved.mlp_mask),
        );

        output.copy_from_slice(&ws.projected);

        if let Some(saved) = cache {
            saved.normalized1.clone_from(&ws.normalized1);
            saved.qkv.clone_from(&ws.qkv);
            saved.context.clone_from(&ws.context);
            saved.normalized2.clone_from(&ws.normalized2);
            saved.activated.clone_from(&ws.activated);
            saved.spec = Some(spec);
        }

        Ok(())
    }

    fn block_backward(
        &self,
        spec: BlockSpec,
        weights: BlockWeights<'_, Self::Buffer>,
        gradients: BlockGradients<'_, Self::Buffer>,
        cache: &Self::BlockCache,
        d_output: &Self::Buffer,
        ws: &mut Self::Workspace,
        d_input: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        let n = spec.elements();
        let c = spec.channels;
        let rows = spec.batch.saturating_mul(spec.time);

        let norm = NormSpec {
            rows,
            channels: c,
            epsilon: 1e-5,
        };

        let linear = |inputs, outputs| LinearSpec {
            rows,
            inputs,
            outputs,
        };

        ws.d0.resize(n, 0.0);
        ws.d1.resize(n, 0.0);
        ws.d2.resize(n, 0.0);
        ws.d_hidden.resize(4 * n, 0.0);
        ws.d_qkv.resize(3 * n, 0.0);
        ws.scores.resize(spec.time, 0.0);

        Self::masked(d_output, &cache.mlp_mask, &mut ws.d0);

        self.linear_backward(
            linear(4 * c, c),
            weights.project,
            gradients.project,
            &cache.activated,
            &ws.d0,
            &mut ws.d_hidden,
        )?;

        for (dx, &x) in ws.d_hidden.iter_mut().zip(&cache.activated) {
            if x <= 0.0 {
                *dx = 0.0;
            }
        }

        self.linear_backward(
            linear(c, 4 * c),
            weights.expand,
            gradients.expand,
            &cache.normalized2,
            &ws.d_hidden,
            &mut ws.d0,
        )?;

        self.layer_norm_backward(
            norm,
            weights.norm2,
            gradients.norm2,
            &cache.norm2,
            &ws.d0,
            &mut ws.d1,
        )?;

        for (dx, &dy) in ws.d1.iter_mut().zip(d_output) {
            *dx += dy;
        }

        Self::masked(&ws.d1, &cache.projection_mask, &mut ws.d0);

        self.linear_backward(
            linear(c, c),
            weights.attention,
            gradients.attention,
            &cache.context,
            &ws.d0,
            &mut ws.d2,
        )?;

        Self::attention_backward(spec, cache, &ws.d2, &mut ws.scores, &mut ws.d_qkv);

        let w_backward = LinearWeights {
            weight: weights.qkv,
            bias: None,
        };
        let g_backward = LinearGradients {
            weight: gradients.qkv,
            bias: None,
        };

        self.linear_backward(
            linear(c, 3 * c),
            w_backward,
            g_backward,
            &cache.normalized1,
            &ws.d_qkv,
            &mut ws.d0,
        )?;

        self.layer_norm_backward(
            norm,
            weights.norm1,
            gradients.norm1,
            &cache.norm1,
            &ws.d0,
            d_input,
        )?;

        for (dx, &residual) in d_input.iter_mut().zip(&ws.d1) {
            *dx += residual;
        }

        Ok(())
    }

    fn linear_forward(
        &self,
        spec: LinearSpec,
        weights: LinearWeights<'_, Self::Buffer>,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        Self::linear_forward(
            spec,
            input,
            weights.weight,
            weights.bias.map(Vec::as_slice),
            output,
        );

        Ok(())
    }

    fn linear_backward(
        &self,
        spec: LinearSpec,
        weights: LinearWeights<'_, Self::Buffer>,
        gradients: LinearGradients<'_, Self::Buffer>,
        input: &Self::Buffer,
        d_output: &Self::Buffer,
        d_input: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        Self::linear_backward(
            spec,
            input,
            weights.weight,
            d_output,
            d_input,
            gradients.weight,
            gradients.bias.map(Vec::as_mut_slice),
        );

        Ok(())
    }

    fn layer_norm_forward(
        &self,
        spec: NormSpec,
        weights: NormWeights<'_, Self::Buffer>,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        cache: Option<&mut Self::LayerNormCache>,
    ) -> anyhow::Result<()> {
        Self::norm_forward(spec, weights, input, output, cache);

        Ok(())
    }

    fn layer_norm_backward(
        &self,
        spec: NormSpec,
        weights: NormWeights<'_, Self::Buffer>,
        gradients: NormGradients<'_, Self::Buffer>,
        cache: &Self::LayerNormCache,
        d_output: &Self::Buffer,
        d_input: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        Self::norm_backward(
            spec,
            weights.gamma,
            cache,
            d_output,
            d_input,
            gradients.gamma,
            gradients.beta,
        );

        Ok(())
    }

    fn token_position_embedding_forward(
        &self,
        spec: EmbeddingSpec,
        tokens: &Self::IndexBuffer,
        token_table: &Self::Buffer,
        position_table: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        for (row, &token) in tokens.iter().enumerate() {
            for ch in 0..spec.channels {
                let idx_out = row * spec.channels + ch;
                let idx_tok = token * spec.channels + ch;
                let idx_pos = (row % spec.time) * spec.channels + ch;

                output[idx_out] = token_table[idx_tok] + position_table[idx_pos];
            }
        }

        Ok(())
    }

    fn token_position_embedding_backward(
        &self,
        spec: EmbeddingSpec,
        tokens: &Self::IndexBuffer,
        d_output: &Self::Buffer,
        d_token_table: &mut Self::Buffer,
        d_position_table: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        d_token_table.fill(0.0);
        d_position_table.fill(0.0);

        for (row, &token) in tokens.iter().enumerate() {
            for ch in 0..spec.channels {
                let dy = d_output[row * spec.channels + ch];

                d_token_table[token * spec.channels + ch] += dy;
                d_position_table[(row % spec.time) * spec.channels + ch] += dy;
            }
        }

        Ok(())
    }

    fn cross_entropy(
        &self,
        logits: &Self::Buffer,
        targets: &Self::IndexBuffer,
        rows: usize,
        vocab: usize,
    ) -> anyhow::Result<f32> {
        Self::loss(logits, targets, rows, vocab, None)
    }

    fn cross_entropy_with_grad(
        &self,
        logits: &Self::Buffer,
        targets: &Self::IndexBuffer,
        rows: usize,
        vocab: usize,
        d_logits: &mut Self::Buffer,
    ) -> anyhow::Result<f32> {
        Self::loss(logits, targets, rows, vocab, Some(d_logits))
    }

    fn buffers_all_finite(&self, buffers: &[&Self::Buffer]) -> anyhow::Result<bool> {
        Ok(buffers.iter().all(|b| b.iter().all(|v| v.is_finite())))
    }

    fn adamw_step(
        &self,
        groups: &mut [AdamWGroup<'_, Self::Buffer>],
        config: AdamWConfig,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.learning_rate.is_finite()
                && config.learning_rate >= 0.0
                && config.weight_decay.is_finite()
                && config.weight_decay >= 0.0
                && (0.0..1.0).contains(&config.beta1)
                && (0.0..1.0).contains(&config.beta2)
                && config.epsilon.is_finite()
                && config.epsilon > 0.0
                && config.step > 0,
            "invalid AdamW configuration"
        );

        let correction1 = 1.0 - config.beta1.powi(config.step);
        let correction2 = 1.0 - config.beta2.powi(config.step);

        for g in groups {
            for i in 0..g.values.len() {
                let gradient = g.gradients[i];

                let m = config.beta1 * g.first_moment[i];
                let m = m + (1.0 - config.beta1) * gradient;

                let v = config.beta2 * g.second_moment[i];
                let v = v + (1.0 - config.beta2) * gradient * gradient;

                let vx = (v / correction2).sqrt() + config.epsilon;
                let vx = config.learning_rate * (m / correction1) / vx;
                let vx = g.values[i] * (1.0 - config.learning_rate * config.weight_decay) - vx;

                anyhow::ensure!(
                    vx.is_finite() && m.is_finite() && v.is_finite(),
                    "non-finite optimizer update; reduce the learning rate"
                );

                g.values[i] = vx;
                g.first_moment[i] = m;
                g.second_moment[i] = v;
            }
        }

        Ok(())
    }

    fn sample_last_token(
        &self,
        logits: &Self::Buffer,
        rows: usize,
        vocab: usize,
        uniform: f32,
    ) -> anyhow::Result<usize> {
        let last = &logits[(rows - 1) * vocab..];

        anyhow::ensure!(
            last.iter().all(|v| v.is_finite()),
            "non-finite sampling logits"
        );

        let max = last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let total: f32 = last.iter().map(|v| (v - max).exp()).sum();
        let threshold = uniform * total;
        let mut cumulative = 0.0;
        let mut fallback = 0;

        for (index, &value) in last.iter().enumerate() {
            let mass = (value - max).exp();
            if mass > 0.0 {
                fallback = index;
            }

            cumulative += mass;

            if threshold < cumulative {
                return Ok(index);
            }
        }

        Ok(fallback)
    }
}
