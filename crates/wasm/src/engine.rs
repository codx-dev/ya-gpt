use anyhow::{Result, bail, ensure};
use ya_gpt::engine::*;

use crate::host::{self, Kernel, elements, positive};

/// An owned FP32 GPU allocation, represented by a browser handle.
pub struct WebGpuBuffer {
    pub(crate) handle: u32,
    pub(crate) len: usize,
    pub(crate) snapshot: Option<Vec<f32>>,
}

impl Drop for WebGpuBuffer {
    fn drop(&mut self) {
        host::release(self.handle);
    }
}

/// Immutable u32 indices with metadata for bounds checking without readback.
pub struct WebGpuIndices {
    pub(crate) data: WebGpuBuffer,
    max: Option<usize>,
}

/// Reused activation buffers. Default performs no GPU allocation.
#[derive(Default)]
pub struct WebGpuWorkspace {
    buffers: Vec<WebGpuBuffer>,
}

/// Browser GPU engine. Operations requiring readback use the stepped browser API.
#[derive(Default)]
pub struct WebGpuEngine;

pub(crate) fn check(buffer: &WebGpuBuffer, len: usize) -> Result<()> {
    ensure!(
        buffer.len == len,
        "GPU tensor length mismatch: {} != {len}",
        buffer.len
    );
    Ok(())
}

fn unsupported<T>() -> Result<T> {
    bail!("this WebGPU operation requires asynchronous readback; use the stepped browser API")
}

impl WebGpuEngine {
    fn upload(&self, words: &[u32]) -> Result<WebGpuBuffer> {
        let buffer = self.zeroes(words.len())?;
        if !words.is_empty() {
            host::upload(buffer.handle, words)?;
        }
        Ok(buffer)
    }

    fn linear_weights(&self, s: LinearSpec, w: &LinearWeights<'_, WebGpuBuffer>) -> Result<()> {
        positive(&[s.rows])?;
        positive(&[s.inputs, s.outputs])?;
        check(w.weight, elements(&[s.inputs, s.outputs])?)?;
        if let Some(bias) = w.bias {
            check(bias, s.outputs)?;
        }
        Ok(())
    }

    fn norm_weights(&self, s: NormSpec, w: &NormWeights<'_, WebGpuBuffer>) -> Result<()> {
        positive(&[s.rows, s.channels])?;
        ensure!(
            s.epsilon.is_finite() && s.epsilon > 0.0,
            "invalid normalization epsilon"
        );
        check(w.gamma, s.channels)?;
        check(w.beta, s.channels)
    }
}

#[allow(clippy::too_many_arguments)]
impl Engine for WebGpuEngine {
    type Buffer = WebGpuBuffer;
    type IndexBuffer = WebGpuIndices;
    type BlockCache = crate::training_engine::BlockCache;
    type LayerNormCache = crate::training_engine::NormCache;
    type Workspace = WebGpuWorkspace;

    fn zeroes(&self, len: usize) -> Result<WebGpuBuffer> {
        elements(&[len])?;
        Ok(WebGpuBuffer {
            handle: host::create((len.max(1) * 4) as u32)?,
            len,
            snapshot: None,
        })
    }

    fn filled(&self, value: f32, len: usize) -> Result<WebGpuBuffer> {
        let out = self.zeroes(len)?;
        if len > 0 {
            host::run(Kernel::Fill, &[out.handle, len as u32, value.to_bits()])?;
        }
        Ok(out)
    }

    fn buffer_len(&self, buffer: &WebGpuBuffer) -> usize {
        buffer.len
    }
    fn buffer_from_slice(&self, values: &[f32]) -> Result<WebGpuBuffer> {
        elements(&[values.len()])?;
        self.upload(&values.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
    }
    fn buffer_to_vec(&self, buffer: &WebGpuBuffer) -> Result<Vec<f32>> {
        buffer
            .snapshot
            .clone()
            .ok_or_else(|| anyhow::anyhow!("buffer requires asynchronous readback before export"))
    }
    fn indices_from_slice(&self, values: &[usize]) -> Result<WebGpuIndices> {
        elements(&[values.len()])?;
        let words = values
            .iter()
            .map(|&v| u32::try_from(v))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(WebGpuIndices {
            data: self.upload(&words)?,
            max: values.iter().copied().max(),
        })
    }
    fn synchronize(&self) -> Result<()> {
        unsupported()
    }

    fn linear_forward(
        &self,
        s: LinearSpec,
        w: LinearWeights<'_, WebGpuBuffer>,
        input: &WebGpuBuffer,
        output: &mut WebGpuBuffer,
    ) -> Result<()> {
        self.linear_weights(s, &w)?;
        check(input, elements(&[s.rows, s.inputs])?)?;
        check(output, elements(&[s.rows, s.outputs])?)?;
        host::run(
            Kernel::Linear,
            &[
                input.handle,
                w.weight.handle,
                w.bias.map_or(0, |b| b.handle),
                output.handle,
                s.rows as u32,
                s.inputs as u32,
                s.outputs as u32,
            ],
        )
    }

    fn layer_norm_forward(
        &self,
        s: NormSpec,
        w: NormWeights<'_, WebGpuBuffer>,
        input: &WebGpuBuffer,
        output: &mut WebGpuBuffer,
        cache: Option<&mut Self::LayerNormCache>,
    ) -> Result<()> {
        if let Some(cache) = cache {
            return self.norm_cached(s, w, input, output, cache);
        }
        self.norm_weights(s, &w)?;
        let n = elements(&[s.rows, s.channels])?;
        check(input, n)?;
        check(output, n)?;
        host::run(
            Kernel::Norm,
            &[
                input.handle,
                w.gamma.handle,
                w.beta.handle,
                output.handle,
                s.rows as u32,
                s.channels as u32,
                s.epsilon.to_bits(),
            ],
        )
    }

    fn token_position_embedding_forward(
        &self,
        s: EmbeddingSpec,
        tokens: &WebGpuIndices,
        token_table: &WebGpuBuffer,
        position_table: &WebGpuBuffer,
        output: &mut WebGpuBuffer,
    ) -> Result<()> {
        let n = positive(&[s.batch, s.time, s.channels])?;
        ensure!(
            s.vocab > 0 && s.time <= s.positions,
            "invalid embedding shape"
        );
        ensure!(
            tokens.max.is_none_or(|i| i < s.vocab),
            "token outside vocabulary"
        );
        check(&tokens.data, elements(&[s.batch, s.time])?)?;
        check(token_table, elements(&[s.vocab, s.channels])?)?;
        check(position_table, elements(&[s.positions, s.channels])?)?;
        check(output, n)?;
        host::run(
            Kernel::Embedding,
            &[
                tokens.data.handle,
                token_table.handle,
                position_table.handle,
                output.handle,
                n as u32,
                s.time as u32,
                s.channels as u32,
            ],
        )
    }

    fn block_forward(
        &self,
        s: BlockSpec,
        w: BlockWeights<'_, WebGpuBuffer>,
        input: &WebGpuBuffer,
        mode: ForwardMode,
        ws: &mut WebGpuWorkspace,
        output: &mut WebGpuBuffer,
        cache: Option<&mut Self::BlockCache>,
    ) -> Result<()> {
        if let Some(cache) = cache {
            return self.block_cached(s, w, input, mode, output, cache);
        }
        ensure!(
            mode.seed.is_none(),
            "WebGPU supports inference without backward caches only"
        );
        let n = positive(&[s.batch, s.time, s.channels])?;
        ensure!(
            s.heads > 0 && s.channels.is_multiple_of(s.heads),
            "invalid attention heads"
        );
        ensure!((0.0..1.0).contains(&s.dropout), "invalid dropout");
        let scores = positive(&[s.batch, s.heads, s.time, s.time])?;
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
        self.norm_weights(norm, &w.norm1)?;
        self.norm_weights(norm, &w.norm2)?;
        check(w.qkv, elements(&[3, c, c])?)?;
        self.linear_weights(linear(c, c), &w.attention)?;
        self.linear_weights(linear(c, 4 * c), &w.expand)?;
        self.linear_weights(linear(4 * c, c), &w.project)?;
        check(input, n)?;
        check(output, n)?;
        let lengths = [n, elements(&[3, n])?, n, n, n, elements(&[4, n])?, scores];
        if ws.buffers.iter().map(|b| b.len).ne(lengths) {
            ws.buffers = lengths
                .into_iter()
                .map(|len| self.zeroes(len))
                .collect::<Result<_>>()?;
        }
        let [
            normalized1,
            qkv,
            context,
            residual,
            normalized2,
            activated,
            probabilities,
        ] = ws.buffers.as_mut_slice()
        else {
            unreachable!()
        };
        self.layer_norm_forward(norm, w.norm1, input, normalized1, None)?;
        self.linear_forward(
            linear(c, 3 * c),
            LinearWeights {
                weight: w.qkv,
                bias: None,
            },
            normalized1,
            qkv,
        )?;
        host::run(
            Kernel::Scores,
            &[
                qkv.handle,
                probabilities.handle,
                s.batch as u32,
                s.time as u32,
                c as u32,
                s.heads as u32,
            ],
        )?;
        host::run(
            Kernel::Softmax,
            &[
                probabilities.handle,
                s.batch as u32,
                s.time as u32,
                s.heads as u32,
            ],
        )?;
        host::run(
            Kernel::Context,
            &[
                qkv.handle,
                probabilities.handle,
                context.handle,
                s.batch as u32,
                s.time as u32,
                c as u32,
                s.heads as u32,
            ],
        )?;
        self.linear_forward(linear(c, c), w.attention, context, residual)?;
        host::run(Kernel::Add, &[residual.handle, input.handle, n as u32])?;
        self.layer_norm_forward(norm, w.norm2, residual, normalized2, None)?;
        self.linear_forward(linear(c, 4 * c), w.expand, normalized2, activated)?;
        host::run(Kernel::Relu, &[activated.handle, (4 * n) as u32])?;
        self.linear_forward(linear(4 * c, c), w.project, activated, output)?;
        host::run(Kernel::Add, &[output.handle, residual.handle, n as u32])
    }

    fn block_backward(
        &self,
        s: BlockSpec,
        w: BlockWeights<'_, WebGpuBuffer>,
        g: BlockGradients<'_, WebGpuBuffer>,
        cache: &Self::BlockCache,
        dy: &WebGpuBuffer,
        _: &mut WebGpuWorkspace,
        dx: &mut WebGpuBuffer,
    ) -> Result<()> {
        self.block_backward_gpu(s, w, g, cache, dy, dx)
    }
    fn linear_backward(
        &self,
        s: LinearSpec,
        w: LinearWeights<'_, WebGpuBuffer>,
        g: LinearGradients<'_, WebGpuBuffer>,
        x: &WebGpuBuffer,
        dy: &WebGpuBuffer,
        dx: &mut WebGpuBuffer,
    ) -> Result<()> {
        self.linear_backward_gpu(s, w, g, x, dy, dx)
    }
    fn layer_norm_backward(
        &self,
        s: NormSpec,
        w: NormWeights<'_, WebGpuBuffer>,
        g: NormGradients<'_, WebGpuBuffer>,
        cache: &Self::LayerNormCache,
        dy: &WebGpuBuffer,
        dx: &mut WebGpuBuffer,
    ) -> Result<()> {
        self.norm_backward(s, w, g, cache, dy, dx)
    }
    fn token_position_embedding_backward(
        &self,
        s: EmbeddingSpec,
        tokens: &WebGpuIndices,
        dy: &WebGpuBuffer,
        dt: &mut WebGpuBuffer,
        dp: &mut WebGpuBuffer,
    ) -> Result<()> {
        self.embedding_backward(s, tokens, dy, dt, dp)
    }
    fn cross_entropy(
        &self,
        _: &WebGpuBuffer,
        _: &WebGpuIndices,
        _: usize,
        _: usize,
    ) -> Result<f32> {
        unsupported()
    }
    fn cross_entropy_with_grad(
        &self,
        _: &WebGpuBuffer,
        _: &WebGpuIndices,
        _: usize,
        _: usize,
        _: &mut WebGpuBuffer,
    ) -> Result<f32> {
        unsupported()
    }
    fn buffers_all_finite(&self, _: &[&WebGpuBuffer]) -> Result<bool> {
        unsupported()
    }
    fn adamw_step(
        &self,
        groups: &mut [AdamWGroup<'_, WebGpuBuffer>],
        config: AdamWConfig,
    ) -> Result<()> {
        self.adamw_gpu(groups, config).map(drop)
    }
    fn sample_last_token(&self, _: &WebGpuBuffer, _: usize, _: usize, _: f32) -> Result<usize> {
        unsupported()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_overflow_and_sync_readback() {
        assert!(elements(&[usize::MAX, 2]).is_err());
        assert!(positive(&[1, 0]).is_err());
        assert!(WebGpuEngine.synchronize().is_err());
        assert!(WebGpuEngine.buffers_all_finite(&[]).is_err());
    }

    #[test]
    fn empty_buffers_indices_and_invalid_shapes() {
        let en = WebGpuEngine;
        let empty = en.zeroes(0).unwrap();
        assert_eq!(en.buffer_len(&empty), 0);
        assert!(en.indices_from_slice(&[usize::MAX]).is_err());
        let w = en.filled(1.0, 6).unwrap();
        let x = en.filled(1.0, 3).unwrap();
        let mut y = en.zeroes(2).unwrap();
        assert!(
            en.linear_forward(
                LinearSpec {
                    rows: 1,
                    inputs: 2,
                    outputs: 3
                },
                LinearWeights {
                    weight: &w,
                    bias: None
                },
                &x,
                &mut y
            )
            .is_err()
        );
    }
}
