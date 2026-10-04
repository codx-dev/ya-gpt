use anyhow::{Result, ensure};
use ya_gpt::engine::{naive::Naive, *};

use crate::{
    WebGpuBuffer as B, WebGpuEngine as E, WebGpuIndices,
    engine::check,
    host::{self, Kernel, elements, positive},
};

#[derive(Default)]
pub struct NormCache {
    spec: Option<NormSpec>,
    normalized: Option<B>,
    inverse: Option<B>,
}

#[derive(Default)]
pub struct BlockCache {
    spec: Option<BlockSpec>,
    norm1: NormCache,
    norm2: NormCache,
    // n1, qkv, probabilities, attention mask, context, n2, activation,
    // projection mask, MLP mask. These belong to this layer, not the workspace.
    buffers: Vec<B>,
}

impl E {
    pub(crate) fn kernel(
        &self,
        kind: u32,
        count: usize,
        buffers: &[&B],
        params: &[u32],
    ) -> Result<()> {
        positive(&[count])?;
        ensure!(
            buffers.len() <= 8 && params.len() <= 12,
            "invalid training descriptor"
        );
        let mut args = [0u32; 22];
        args[0] = kind;
        args[1] = count as u32;
        for (slot, b) in args[2..10].iter_mut().zip(buffers) {
            *slot = b.handle;
        }
        args[10..10 + params.len()].copy_from_slice(params);
        host::run(Kernel::Training, &args)
    }

    fn multiply(&self, x: &B, mask: &B) -> Result<B> {
        check(mask, x.len)?;
        let out = self.zeroes(x.len)?;
        self.kernel(2, x.len, &[&out, x, mask], &[])?;
        Ok(out)
    }

    fn sum(&self, x: &B, y: &B) -> Result<B> {
        check(y, x.len)?;
        let out = self.zeroes(x.len)?;
        self.kernel(3, x.len, &[&out, x, y], &[])?;
        Ok(out)
    }

    pub(crate) fn norm_cached(
        &self,
        s: NormSpec,
        w: NormWeights<'_, B>,
        input: &B,
        output: &mut B,
        cache: &mut NormCache,
    ) -> Result<()> {
        let n = positive(&[s.rows, s.channels])?;
        ensure!(
            s.epsilon.is_finite() && s.epsilon > 0.0,
            "invalid normalization epsilon"
        );
        check(input, n)?;
        check(output, n)?;
        check(w.gamma, s.channels)?;
        check(w.beta, s.channels)?;
        let normalized = self.zeroes(n)?;
        let inverse = self.zeroes(s.rows)?;
        self.kernel(
            5,
            s.rows,
            &[output, input, w.gamma, w.beta, &normalized, &inverse],
            &[s.channels as u32, s.epsilon.to_bits()],
        )?;
        *cache = NormCache {
            spec: Some(s),
            normalized: Some(normalized),
            inverse: Some(inverse),
        };
        Ok(())
    }

    pub(crate) fn norm_backward(
        &self,
        s: NormSpec,
        w: NormWeights<'_, B>,
        g: NormGradients<'_, B>,
        cache: &NormCache,
        dy: &B,
        dx: &mut B,
    ) -> Result<()> {
        ensure!(cache.spec == Some(s), "normalization cache mismatch");
        let n = positive(&[s.rows, s.channels])?;
        check(dy, n)?;
        check(dx, n)?;
        check(w.gamma, s.channels)?;
        check(g.gamma, s.channels)?;
        check(g.beta, s.channels)?;
        let normalized = cache.normalized.as_ref().unwrap();
        self.kernel(
            6,
            s.rows,
            &[dx, dy, w.gamma, normalized, cache.inverse.as_ref().unwrap()],
            &[s.channels as u32],
        )?;
        self.kernel(
            7,
            s.channels,
            &[g.gamma, g.beta, dy, normalized],
            &[s.rows as u32],
        )
    }

    pub(crate) fn linear_backward_gpu(
        &self,
        s: LinearSpec,
        w: LinearWeights<'_, B>,
        g: LinearGradients<'_, B>,
        x: &B,
        dy: &B,
        dx: &mut B,
    ) -> Result<()> {
        let ni = positive(&[s.rows, s.inputs])?;
        let no = positive(&[s.rows, s.outputs])?;
        let nw = positive(&[s.inputs, s.outputs])?;
        check(x, ni)?;
        check(dx, ni)?;
        check(dy, no)?;
        check(w.weight, nw)?;
        check(g.weight, nw)?;
        ensure!(w.bias.is_some() == g.bias.is_some(), "linear bias mismatch");
        let params = [s.rows as u32, s.inputs as u32, s.outputs as u32];
        self.kernel(8, ni, &[dx, dy, w.weight], &params)?;
        self.kernel(9, nw, &[g.weight, x, dy], &params)?;
        if let Some(db) = g.bias {
            check(db, s.outputs)?;
            self.kernel(10, s.outputs, &[db, dy], &[s.rows as u32])?;
        }
        Ok(())
    }

    pub(crate) fn embedding_backward(
        &self,
        s: EmbeddingSpec,
        tokens: &WebGpuIndices,
        dy: &B,
        dt: &mut B,
        dp: &mut B,
    ) -> Result<()> {
        let n = positive(&[s.batch, s.time, s.channels])?;
        ensure!(s.time <= s.positions, "invalid embedding shape");
        check(dy, n)?;
        check(&tokens.data, s.batch * s.time)?;
        check(dt, positive(&[s.vocab, s.channels])?)?;
        check(dp, positive(&[s.positions, s.channels])?)?;
        let p = [(s.batch * s.time) as u32, s.time as u32, s.channels as u32];
        self.kernel(11, dt.len, &[dt, &tokens.data, dy], &p)?;
        self.kernel(12, dp.len, &[dp, dy], &p)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn block_cached(
        &self,
        s: BlockSpec,
        w: BlockWeights<'_, B>,
        input: &B,
        mode: ForwardMode,
        output: &mut B,
        cache: &mut BlockCache,
    ) -> Result<()> {
        let n = positive(&[s.batch, s.time, s.channels])?;
        ensure!(
            s.heads > 0 && s.channels.is_multiple_of(s.heads) && (0.0..1.0).contains(&s.dropout),
            "invalid block shape or dropout"
        );
        check(input, n)?;
        check(output, n)?;
        let count = positive(&[s.batch, s.heads, s.time, s.time])?;
        let c = s.channels;
        let rows = s.batch * s.time;
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
        let mut n1 = self.zeroes(n)?;
        let mut qkv = self.zeroes(elements(&[3, n])?)?;
        let probs = self.zeroes(count)?;
        let context = self.zeroes(n)?;
        let mut residual = self.zeroes(n)?;
        let mut n2 = self.zeroes(n)?;
        let mut activated = self.zeroes(elements(&[4, n])?)?;
        let mut saved = BlockCache::default();
        self.norm_cached(norm, w.norm1, input, &mut n1, &mut saved.norm1)?;
        self.linear_forward(
            linear(c, 3 * c),
            LinearWeights {
                weight: w.qkv,
                bias: None,
            },
            &n1,
            &mut qkv,
        )?;
        host::run(
            Kernel::Scores,
            &[
                qkv.handle,
                probs.handle,
                s.batch as u32,
                s.time as u32,
                c as u32,
                s.heads as u32,
            ],
        )?;
        host::run(
            Kernel::Softmax,
            &[probs.handle, s.batch as u32, s.time as u32, s.heads as u32],
        )?;
        let mut rng = Naive::dropout_rng(mode, s.dropout, 0);
        let mut masks = vec![0.0; count];
        for b in 0..s.batch {
            for h in 0..s.heads {
                for t in 0..s.time {
                    for k in 0..=t {
                        masks[((b * s.heads + h) * s.time + t) * s.time + k] =
                            Naive::dropout_value(&mut rng, s.dropout);
                    }
                }
            }
        }
        let amask = self.buffer_from_slice(&masks)?;
        let dropped = self.multiply(&probs, &amask)?;
        host::run(
            Kernel::Context,
            &[
                qkv.handle,
                dropped.handle,
                context.handle,
                s.batch as u32,
                s.time as u32,
                c as u32,
                s.heads as u32,
            ],
        )?;
        self.linear_forward(linear(c, c), w.attention, &context, &mut residual)?;
        let mask = |stream| -> Result<B> {
            let mut rng = Naive::dropout_rng(mode, s.dropout, stream);
            self.buffer_from_slice(
                &(0..n)
                    .map(|_| Naive::dropout_value(&mut rng, s.dropout))
                    .collect::<Vec<_>>(),
            )
        };
        let pmask = mask(1)?;
        let mmask = mask(2)?;
        let projected = self.multiply(&residual, &pmask)?;
        let after = self.sum(&projected, input)?;
        self.norm_cached(norm, w.norm2, &after, &mut n2, &mut saved.norm2)?;
        self.linear_forward(linear(c, 4 * c), w.expand, &n2, &mut activated)?;
        host::run(Kernel::Relu, &[activated.handle, activated.len as u32])?;
        self.linear_forward(linear(4 * c, c), w.project, &activated, &mut residual)?;
        let branch = self.multiply(&residual, &mmask)?;
        self.kernel(3, n, &[output, &branch, &after], &[])?;
        saved.spec = Some(s);
        saved.buffers = vec![n1, qkv, probs, amask, context, n2, activated, pmask, mmask];
        *cache = saved;
        Ok(())
    }

    pub(crate) fn block_backward_gpu(
        &self,
        s: BlockSpec,
        w: BlockWeights<'_, B>,
        g: BlockGradients<'_, B>,
        cache: &BlockCache,
        dy: &B,
        dx: &mut B,
    ) -> Result<()> {
        ensure!(cache.spec == Some(s), "block cache mismatch");
        let n = positive(&[s.batch, s.time, s.channels])?;
        check(dy, n)?;
        check(dx, n)?;
        let [n1, qkv, probs, amask, context, n2, activated, pmask, mmask] =
            cache.buffers.as_slice()
        else {
            anyhow::bail!("invalid block cache")
        };
        let c = s.channels;
        let rows = s.batch * s.time;
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
        let masked = self.multiply(dy, mmask)?;
        let mut dh = self.zeroes(4 * n)?;
        self.linear_backward_gpu(
            linear(4 * c, c),
            w.project,
            g.project,
            activated,
            &masked,
            &mut dh,
        )?;
        let dh_relu = self.zeroes(4 * n)?;
        self.kernel(4, 4 * n, &[&dh_relu, &dh, activated], &[])?;
        let mut d0 = self.zeroes(n)?;
        let mut d1 = self.zeroes(n)?;
        self.linear_backward_gpu(linear(c, 4 * c), w.expand, g.expand, n2, &dh_relu, &mut d0)?;
        self.norm_backward(norm, w.norm2, g.norm2, &cache.norm2, &d0, &mut d1)?;
        let skip = self.sum(&d1, dy)?;
        let masked = self.multiply(&skip, pmask)?;
        self.linear_backward_gpu(
            linear(c, c),
            w.attention,
            g.attention,
            context,
            &masked,
            &mut d0,
        )?;
        let dp = self.zeroes(probs.len)?;
        let ds = self.zeroes(probs.len)?;
        let dqkv = self.zeroes(qkv.len)?;
        let shape = [s.batch as u32, s.time as u32, c as u32, s.heads as u32];
        self.kernel(13, probs.len, &[&dp, &d0, qkv, amask], &shape)?;
        self.kernel(
            14,
            s.batch * s.heads * s.time,
            &[&ds, probs, &dp],
            &[
                s.time as u32,
                (1.0 / ((c / s.heads) as f32).sqrt()).to_bits(),
            ],
        )?;
        self.kernel(15, qkv.len, &[&dqkv, qkv, &ds, probs, amask, &d0], &shape)?;
        // dqkv is produced on the GPU and only borrowed by the following kernel.
        self.linear_backward_gpu(
            linear(c, 3 * c),
            LinearWeights {
                weight: w.qkv,
                bias: None,
            },
            LinearGradients {
                weight: g.qkv,
                bias: None,
            },
            n1,
            &dqkv,
            &mut d0,
        )?;
        self.norm_backward(norm, w.norm1, g.norm1, &cache.norm1, &d0, &mut d1)?;
        self.kernel(3, n, &[dx, &d1, &skip], &[])
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn loss_gpu(
        &self,
        logits: &B,
        targets: &WebGpuIndices,
        rows: usize,
        vocab: usize,
        gradient: bool,
    ) -> Result<(B, B, B)> {
        check(logits, positive(&[rows, vocab])?)?;
        check(&targets.data, rows)?;
        let grad = self.zeroes(if gradient { logits.len } else { 1 })?;
        let losses = self.zeroes(rows)?;
        let invalid = self.zeroes(rows)?;
        self.kernel(
            16,
            rows,
            &[&grad, logits, &targets.data, &losses, &invalid],
            &[vocab as u32, u32::from(gradient)],
        )?;
        Ok((grad, losses, invalid))
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn finite_gpu(&self, buffers: &[&B]) -> Result<B> {
        let n: usize = buffers.iter().map(|b| b.len.div_ceil(256)).sum();
        let flags = self.zeroes(n)?;
        let mut offset = 0;
        for b in buffers {
            let chunks = b.len.div_ceil(256);
            if chunks > 0 {
                self.kernel(18, chunks, &[&flags, b], &[b.len as u32, offset as u32])?;
            }
            offset += chunks;
        }
        Ok(flags)
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn summarize(&self, losses: &B, invalid: &B, flags: &B) -> Result<B> {
        let out = self.zeroes(2)?;
        self.kernel(
            20,
            1,
            &[&out, losses, invalid, flags],
            &[losses.len as u32, invalid.len as u32, flags.len as u32],
        )?;
        Ok(out)
    }

    pub(crate) fn adamw_gpu(&self, groups: &mut [AdamWGroup<'_, B>], c: AdamWConfig) -> Result<B> {
        ensure!(
            c.learning_rate.is_finite()
                && c.learning_rate >= 0.0
                && c.weight_decay.is_finite()
                && c.weight_decay >= 0.0
                && (0.0..1.0).contains(&c.beta1)
                && (0.0..1.0).contains(&c.beta2)
                && c.epsilon.is_finite()
                && c.epsilon > 0.0
                && c.step > 0,
            "invalid AdamW configuration"
        );
        let n = groups.iter().map(|g| g.values.len.div_ceil(256)).sum();
        let flags = self.zeroes(n)?;
        let mut offset = 0;
        for g in groups {
            let len = g.values.len;
            check(g.gradients, len)?;
            check(g.first_moment, len)?;
            check(g.second_moment, len)?;
            let chunks = len.div_ceil(256);
            if chunks > 0 {
                self.kernel(
                    19,
                    chunks,
                    &[
                        g.values,
                        g.gradients,
                        g.first_moment,
                        g.second_moment,
                        &flags,
                    ],
                    &[
                        len as u32,
                        c.learning_rate.to_bits(),
                        c.weight_decay.to_bits(),
                        c.beta1.to_bits(),
                        c.beta2.to_bits(),
                        c.epsilon.to_bits(),
                        (1.0 - ya_gpt::utils::powi(c.beta1, c.step)).to_bits(),
                        (1.0 - ya_gpt::utils::powi(c.beta2, c.step)).to_bits(),
                        offset as u32,
                    ],
                )?;
            }
            offset += chunks;
        }
        let out = self.zeroes(1)?;
        self.kernel(17, 1, &[&out, &flags], &[n as u32])?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::mock;
    use rand::{SeedableRng, rngs::StdRng};
    use ya_gpt::{config::ModelConfig, model::gpt::Gpt, tokenizer::Tokenizer};

    #[test]
    fn training_dropout_masks_match_the_reference_draw_order() {
        mock::reset();
        let config = ModelConfig {
            block_size: 5,
            n_embd: 6,
            n_head: 2,
            n_layer: 1,
            dropout: 0.2,
        };
        let model = Gpt::new(
            &Naive,
            config,
            Tokenizer::new("abc"),
            &mut StdRng::seed_from_u64(42),
        )
        .unwrap();
        let gpu = Gpt::try_from_bytes(&E, &model.to_bytes(&Naive).unwrap()).unwrap();
        let s = BlockSpec {
            batch: 2,
            time: 3,
            channels: 6,
            heads: 2,
            dropout: 0.2,
        };
        let mode = ForwardMode {
            seed: Some(u64::MAX),
        };
        let mut reference = ya_gpt::engine::naive::NaiveBlockCache::default();
        Naive
            .block_forward(
                s,
                model.blocks[0].weights(),
                &vec![0.25; 36],
                mode,
                &mut Default::default(),
                &mut vec![0.0; 36],
                Some(&mut reference),
            )
            .unwrap();
        let input = E.buffer_from_slice(&[0.25; 36]).unwrap();
        let mut output = E.zeroes(36).unwrap();
        let mut actual = BlockCache::default();
        E.block_cached(
            s,
            gpu.blocks[0].weights(),
            &input,
            mode,
            &mut output,
            &mut actual,
        )
        .unwrap();
        for (index, expected) in [
            (3, reference.attention_mask),
            (7, reference.projection_mask),
            (8, reference.mlp_mask),
        ] {
            assert_eq!(
                mock::read(actual.buffers[index].handle, 0, expected.len()),
                expected
            );
        }
        drop((gpu, input, output, actual));
        assert_eq!(mock::count(), 0);
    }
}
