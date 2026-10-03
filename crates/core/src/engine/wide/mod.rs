use alloc::{vec, vec::Vec};

mod kernels;
mod types;

pub use types::*;

use crate::engine::{
    AdamWConfig, AdamWGroup, BlockGradients, BlockSpec, BlockWeights, EmbeddingSpec, Engine,
    ForwardMode, LinearGradients, LinearSpec, LinearWeights, NormGradients, NormSpec, NormWeights,
};

/// Contiguous, unpadded CPU storage; SIMD loads do not require special alignment.
pub type SimdWideBuffer = Vec<f32>;

/// A synchronous CPU engine with portable SIMD kernels and scalar remainders.
///
/// Reductions and vector exponentials can differ slightly from scalar arithmetic.
/// Seeded dropout retains the scalar engine's random draw order.
#[derive(Debug, Default, Clone, Copy)]
pub struct SimdWide;

#[cfg(test)]
mod tests;

impl Engine for SimdWide {
    type Buffer = SimdWideBuffer;
    type IndexBuffer = Vec<usize>;
    type BlockCache = SimdWideBlockCache;
    type LayerNormCache = SimdWideLayerNormCache;
    type Workspace = SimdWideWorkspace;

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

        Self::relu(&mut ws.activated);

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
        debug_assert_eq!(cache.spec, Some(spec));
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

        Self::relu_backward(&cache.activated, &mut ws.d_hidden);

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

        Self::add_assign(&mut ws.d1, d_output);

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

        Self::add_assign(d_input, &ws.d1);

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
            let c = spec.channels;
            let position = row % spec.time;
            Self::add(
                &token_table[token * c..(token + 1) * c],
                &position_table[position * c..(position + 1) * c],
                &mut output[row * c..(row + 1) * c],
            );
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
            let c = spec.channels;
            let position = row % spec.time;
            let gradient = &d_output[row * c..(row + 1) * c];
            Self::add_assign(&mut d_token_table[token * c..(token + 1) * c], gradient);
            Self::add_assign(
                &mut d_position_table[position * c..(position + 1) * c],
                gradient,
            );
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
        Ok(buffers.iter().all(|b| Self::all_finite(b)))
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

        Self::adamw(groups, config)
    }

    fn sample_last_token(
        &self,
        logits: &Self::Buffer,
        rows: usize,
        vocab: usize,
        uniform: f32,
    ) -> anyhow::Result<usize> {
        let last = &logits[(rows - 1) * vocab..rows * vocab];
        anyhow::ensure!(Self::all_finite(last), "non-finite sampling logits");

        let max = Self::maximum(last);
        let threshold = uniform * Self::exp_sum(last, max);
        let mut cumulative = 0.0;
        let mut fallback = 0;

        // Use the same exponentiation for both passes, including scalar tails.
        for (chunk_index, chunk) in last.chunks(8).enumerate() {
            let masses = Self::exp_chunk(chunk, max);
            for (lane, &mass) in masses[..chunk.len()].iter().enumerate() {
                let index = chunk_index * 8 + lane;
                if mass > 0.0 {
                    fallback = index;
                }
                cumulative += mass;
                if threshold < cumulative {
                    return Ok(index);
                }
            }
        }
        Ok(fallback)
    }
}
