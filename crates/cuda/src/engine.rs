use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig, sys},
    driver::{CudaContext, CudaStream},
};
use ya_gpt::engine::*;

use crate::{
    kernels::{Kernels, launch},
    types::*,
    validation,
};

/// An FP32 CUDA backend with one ordered stream and an immutable cuBLAS handle.
///
/// Buffers, caches, and workspaces belong to the engine that created them.
/// Tensor operations enqueue work; downloads and scalar results synchronize.
pub struct CudaEngine {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) kernels: Kernels,
    blas: CudaBlas,
}

impl CudaEngine {
    /// Open a CUDA device with compute capability 8.9 or newer.
    ///
    /// Requires the NVIDIA driver and cuBLAS shared libraries at runtime.
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .with_context(|| format!("initializing CUDA device {device_ordinal}"))?;
        use cudarc::driver::sys::CUdevice_attribute::*;
        let major = context.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = context.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        ensure!(
            (major, minor) >= (8, 9),
            "CUDA backend requires compute capability 8.9 or newer; found {major}.{minor}"
        );
        let stream = context.new_stream().context("creating CUDA stream")?;
        let kernels = Kernels::load(&context)?;
        let blas = CudaBlas::new(stream.clone()).context("creating cuBLAS handle")?;
        // SAFETY: the live handle is configured once in its current context,
        // before it is exposed to tensor operations.
        unsafe {
            sys::cublasSetMathMode(*blas.handle(), sys::cublasMath_t::CUBLAS_PEDANTIC_MATH).result()
        }
        .context("configuring FP32 cuBLAS arithmetic")?;
        Ok(Self {
            context,
            stream,
            kernels,
            blas,
        })
    }

    pub(crate) fn check(&self, buffer: &CudaBuffer, len: usize) -> Result<()> {
        ensure!(
            Arc::ptr_eq(buffer.data.context(), &self.context),
            "buffer belongs to another CUDA engine"
        );
        ensure!(
            buffer.len == len,
            "buffer length mismatch: expected {len}, got {}",
            buffer.len
        );
        Ok(())
    }

    fn check_indices(&self, buffer: &CudaIndexBuffer, len: usize, bound: usize) -> Result<()> {
        ensure!(
            Arc::ptr_eq(buffer.data.context(), &self.context),
            "indices belong to another CUDA engine"
        );
        ensure!(buffer.len == len, "index buffer length mismatch");
        ensure!(
            buffer.max.is_none_or(|index| index < bound),
            "index outside vocabulary"
        );
        Ok(())
    }

    pub(crate) fn resize<'a>(
        &self,
        slot: &'a mut Option<CudaBuffer>,
        len: usize,
    ) -> Result<&'a mut CudaBuffer> {
        validation::elements(&[len])?;
        if let Some(buffer) = slot.as_ref() {
            self.check(buffer, buffer.len)?;
        }
        if slot
            .as_ref()
            .is_none_or(|buffer| buffer.data.len() < len.max(1))
        {
            *slot = Some(self.zeroes(len)?);
        }
        let buffer = slot.as_mut().expect("allocated buffer");
        buffer.len = len;
        Ok(buffer)
    }

    pub(crate) fn copy_buffer(&self, source: &CudaBuffer, target: &mut CudaBuffer) -> Result<()> {
        self.check(source, target.len)?;
        self.check(target, source.len)?;
        launch!(
            self,
            copy,
            source.len,
            &source.data,
            &mut target.data,
            &source.len
        );
        Ok(())
    }

    pub(crate) fn clone_buffer(&self, source: &CudaBuffer) -> Result<CudaBuffer> {
        let mut target = self.zeroes(source.len)?;
        self.copy_buffer(source, &mut target)?;
        Ok(target)
    }

    pub(crate) fn clear(&self, buffer: &mut CudaBuffer) -> Result<()> {
        launch!(
            self,
            fill,
            buffer.len,
            &mut buffer.data,
            &buffer.len,
            &0.0f32
        );
        Ok(())
    }

    pub(crate) fn add_buffer(&self, target: &mut CudaBuffer, source: &CudaBuffer) -> Result<()> {
        self.check(source, target.len)?;
        self.check(target, source.len)?;
        launch!(
            self,
            add,
            target.len,
            &mut target.data,
            &source.data,
            &target.len
        );
        Ok(())
    }

    pub(crate) fn check_linear_weights(
        &self,
        spec: LinearSpec,
        weights: &LinearWeights<'_, CudaBuffer>,
    ) -> Result<()> {
        validation::linear(spec)?;
        self.check(weights.weight, spec.inputs * spec.outputs)?;
        if let Some(bias) = weights.bias {
            self.check(bias, spec.outputs)?;
        }
        Ok(())
    }

    pub(crate) fn check_norm_weights(
        &self,
        spec: NormSpec,
        weights: &NormWeights<'_, CudaBuffer>,
    ) -> Result<()> {
        validation::norm(spec)?;
        self.check(weights.gamma, spec.channels)?;
        self.check(weights.beta, spec.channels)
    }

    pub(crate) fn check_norm_cache(
        &self,
        spec: NormSpec,
        cache: &CudaLayerNormCache,
    ) -> Result<()> {
        ensure!(
            cache.spec == Some(spec),
            "missing or incompatible layer normalization cache"
        );
        self.check(
            cache
                .normalized
                .as_ref()
                .context("missing normalized cache")?,
            spec.rows * spec.channels,
        )?;
        self.check(
            cache
                .inverse_std
                .as_ref()
                .context("missing inverse standard deviation cache")?,
            spec.rows,
        )
    }

    fn loss(
        &self,
        logits: &CudaBuffer,
        targets: &CudaIndexBuffer,
        rows: usize,
        vocab: usize,
        gradient: Option<&mut CudaBuffer>,
    ) -> Result<f32> {
        validation::dimension(rows)?;
        validation::dimension(vocab)?;
        let n = validation::elements(&[rows, vocab])?;
        self.check(logits, n)?;
        self.check_indices(targets, rows, vocab)?;
        if let Some(g) = gradient.as_ref() {
            self.check(g, n)?;
        }
        let with_grad = u32::from(gradient.is_some());
        let mut discard = self.zeroes(0)?;
        let gradient = gradient.unwrap_or(&mut discard);
        let mut losses = self.zeroes(rows)?;
        let mut total = self.zeroes(1)?;
        launch!(
            self,
            cross_entropy,
            rows,
            &logits.data,
            &targets.data,
            &mut losses.data,
            &mut gradient.data,
            &rows,
            &vocab,
            &with_grad
        );
        launch!(
            self,
            reduce_loss,
            1usize,
            &losses.data,
            &mut total.data,
            &rows
        );
        let loss = self.buffer_to_vec(&total)?[0];
        ensure!(
            loss.is_finite(),
            "non-finite logits or loss; reduce the learning rate"
        );
        Ok(loss)
    }
}

#[allow(clippy::too_many_arguments)]
impl Engine for CudaEngine {
    type Buffer = CudaBuffer;
    type IndexBuffer = CudaIndexBuffer;
    type BlockCache = CudaBlockCache;
    type LayerNormCache = CudaLayerNormCache;
    type Workspace = CudaWorkspace;

    fn zeroes(&self, len: usize) -> Result<CudaBuffer> {
        validation::elements(&[len])?;
        Ok(CudaBuffer {
            data: self
                .stream
                .alloc_zeros::<f32>(len.max(1))
                .context("allocating CUDA buffer")?,
            len,
        })
    }

    fn filled(&self, value: f32, len: usize) -> Result<CudaBuffer> {
        let mut buffer = self.zeroes(len)?;
        launch!(self, fill, len, &mut buffer.data, &len, &value);
        Ok(buffer)
    }

    fn buffer_len(&self, buffer: &CudaBuffer) -> usize {
        buffer.len
    }

    fn buffer_from_slice(&self, values: &[f32]) -> Result<CudaBuffer> {
        validation::elements(&[values.len()])?;
        if values.is_empty() {
            return self.zeroes(0);
        }
        Ok(CudaBuffer {
            data: self.stream.clone_htod(values)?,
            len: values.len(),
        })
    }

    fn buffer_to_vec(&self, buffer: &CudaBuffer) -> Result<Vec<f32>> {
        self.check(buffer, buffer.len)?;
        if buffer.len == 0 {
            self.synchronize()?;
            return Ok(Vec::new());
        }
        let values = self.stream.clone_dtoh(&buffer.data.slice(..buffer.len))?;
        self.synchronize()?;
        Ok(values)
    }

    fn indices_from_slice(&self, values: &[usize]) -> Result<CudaIndexBuffer> {
        validation::elements(&[values.len()])?;
        let indices = values
            .iter()
            .map(|&v| u32::try_from(v).context("index exceeds u32"))
            .collect::<Result<Vec<_>>>()?;
        let data = if values.is_empty() {
            self.stream.alloc_zeros::<u32>(1)?
        } else {
            self.stream.clone_htod(&indices)?
        };
        Ok(CudaIndexBuffer {
            data,
            len: values.len(),
            max: values.iter().copied().max(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .context("synchronizing CUDA operations")?;
        self.context.check_err().context("deferred CUDA error")
    }

    fn block_forward(
        &self,
        spec: BlockSpec,
        weights: BlockWeights<'_, CudaBuffer>,
        input: &CudaBuffer,
        mode: ForwardMode,
        ws: &mut CudaWorkspace,
        output: &mut CudaBuffer,
        cache: Option<&mut CudaBlockCache>,
    ) -> Result<()> {
        self.forward_block(spec, weights, input, mode, ws, output, cache)
    }

    fn block_backward(
        &self,
        spec: BlockSpec,
        weights: BlockWeights<'_, CudaBuffer>,
        gradients: BlockGradients<'_, CudaBuffer>,
        cache: &CudaBlockCache,
        d_output: &CudaBuffer,
        ws: &mut CudaWorkspace,
        d_input: &mut CudaBuffer,
    ) -> Result<()> {
        self.backward_block(spec, weights, gradients, cache, d_output, ws, d_input)
    }

    fn linear_forward(
        &self,
        spec: LinearSpec,
        weights: LinearWeights<'_, CudaBuffer>,
        input: &CudaBuffer,
        output: &mut CudaBuffer,
    ) -> Result<()> {
        self.check_linear_weights(spec, &weights)?;
        self.check(input, spec.rows * spec.inputs)?;
        self.check(output, spec.rows * spec.outputs)?;
        let (r, i, o) = (spec.rows as i32, spec.inputs as i32, spec.outputs as i32);
        let config = GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_N,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: o,
            n: r,
            k: i,
            alpha: 1.0f32,
            lda: o,
            ldb: i,
            beta: 0.0,
            ldc: o,
        };
        self.context.bind_to_thread()?;
        // SAFETY: checked lengths and i32 dimensions. cuBLAS sees row-major
        // Y = XW as column-major Y^T = W^T X^T.
        unsafe {
            self.blas
                .gemm(config, &weights.weight.data, &input.data, &mut output.data)
        }?;
        if let Some(bias) = weights.bias {
            launch!(
                self,
                bias_add,
                output.len,
                &mut output.data,
                &bias.data,
                &output.len,
                &spec.outputs
            );
        }
        Ok(())
    }

    fn linear_backward(
        &self,
        spec: LinearSpec,
        weights: LinearWeights<'_, CudaBuffer>,
        gradients: LinearGradients<'_, CudaBuffer>,
        input: &CudaBuffer,
        d_output: &CudaBuffer,
        d_input: &mut CudaBuffer,
    ) -> Result<()> {
        self.check_linear_weights(spec, &weights)?;
        self.check(input, spec.rows * spec.inputs)?;
        self.check(d_output, spec.rows * spec.outputs)?;
        self.check(d_input, spec.rows * spec.inputs)?;
        self.check(gradients.weight, spec.inputs * spec.outputs)?;
        if let Some(bias) = gradients.bias.as_ref() {
            self.check(bias, spec.outputs)?;
        }
        let (r, i, o) = (spec.rows as i32, spec.inputs as i32, spec.outputs as i32);
        let dx = GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_T,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: i,
            n: r,
            k: o,
            alpha: 1.0f32,
            lda: o,
            ldb: o,
            beta: 0.0,
            ldc: i,
        };
        let dw = GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_N,
            transb: sys::cublasOperation_t::CUBLAS_OP_T,
            m: o,
            n: i,
            k: r,
            alpha: 1.0f32,
            lda: o,
            ldb: i,
            beta: 0.0,
            ldc: o,
        };
        self.context.bind_to_thread()?;
        // SAFETY: validated buffers; beta=0 overwrites prior gradients.
        // dX^T = W dY^T, dW^T = dY^T X.
        unsafe {
            self.blas
                .gemm(dx, &weights.weight.data, &d_output.data, &mut d_input.data)?;
            self.blas
                .gemm(dw, &d_output.data, &input.data, &mut gradients.weight.data)?;
        }
        if let Some(bias) = gradients.bias {
            launch!(
                self,
                bias_backward,
                spec.outputs,
                &d_output.data,
                &mut bias.data,
                &spec.rows,
                &spec.outputs
            );
        }
        Ok(())
    }

    fn layer_norm_forward(
        &self,
        spec: NormSpec,
        weights: NormWeights<'_, CudaBuffer>,
        input: &CudaBuffer,
        output: &mut CudaBuffer,
        cache: Option<&mut CudaLayerNormCache>,
    ) -> Result<()> {
        let save = u32::from(cache.is_some());
        let mut discard = CudaLayerNormCache::default();
        let saved = cache.unwrap_or(&mut discard);
        saved.spec = None;
        self.check_norm_weights(spec, &weights)?;
        let n = spec.rows * spec.channels;
        self.check(input, n)?;
        self.check(output, n)?;
        let normalized = self.resize(&mut saved.normalized, if save != 0 { n } else { 0 })?;
        let inverse_std = self.resize(
            &mut saved.inverse_std,
            if save != 0 { spec.rows } else { 0 },
        )?;
        launch!(
            self,
            norm_forward,
            spec.rows,
            &input.data,
            &weights.gamma.data,
            &weights.beta.data,
            &mut output.data,
            &mut normalized.data,
            &mut inverse_std.data,
            &spec.rows,
            &spec.channels,
            &spec.epsilon,
            &save
        );
        saved.spec = Some(spec);
        Ok(())
    }

    fn layer_norm_backward(
        &self,
        spec: NormSpec,
        weights: NormWeights<'_, CudaBuffer>,
        gradients: NormGradients<'_, CudaBuffer>,
        cache: &CudaLayerNormCache,
        d_output: &CudaBuffer,
        d_input: &mut CudaBuffer,
    ) -> Result<()> {
        self.check_norm_weights(spec, &weights)?;
        self.check_norm_cache(spec, cache)?;
        self.check(d_output, spec.rows * spec.channels)?;
        self.check(d_input, d_output.len)?;
        self.check(gradients.gamma, spec.channels)?;
        self.check(gradients.beta, spec.channels)?;
        let normalized = cache.normalized.as_ref().expect("validated cache");
        let inverse_std = cache.inverse_std.as_ref().expect("validated cache");
        launch!(
            self,
            norm_backward_input,
            spec.rows,
            &d_output.data,
            &weights.gamma.data,
            &normalized.data,
            &inverse_std.data,
            &mut d_input.data,
            &spec.rows,
            &spec.channels
        );
        launch!(
            self,
            norm_backward_weights,
            spec.channels,
            &d_output.data,
            &normalized.data,
            &mut gradients.gamma.data,
            &mut gradients.beta.data,
            &spec.rows,
            &spec.channels
        );
        Ok(())
    }

    fn token_position_embedding_forward(
        &self,
        spec: EmbeddingSpec,
        tokens: &CudaIndexBuffer,
        token_table: &CudaBuffer,
        position_table: &CudaBuffer,
        output: &mut CudaBuffer,
    ) -> Result<()> {
        let n = validation::embedding(spec)?;
        self.check_indices(tokens, spec.batch * spec.time, spec.vocab)?;
        self.check(token_table, spec.vocab * spec.channels)?;
        self.check(position_table, spec.positions * spec.channels)?;
        self.check(output, n)?;
        launch!(
            self,
            embedding_forward,
            n,
            &tokens.data,
            &token_table.data,
            &position_table.data,
            &mut output.data,
            &n,
            &spec.time,
            &spec.channels
        );
        Ok(())
    }

    fn token_position_embedding_backward(
        &self,
        spec: EmbeddingSpec,
        tokens: &CudaIndexBuffer,
        d_output: &CudaBuffer,
        d_token_table: &mut CudaBuffer,
        d_position_table: &mut CudaBuffer,
    ) -> Result<()> {
        let n = validation::embedding(spec)?;
        self.check_indices(tokens, spec.batch * spec.time, spec.vocab)?;
        self.check(d_output, n)?;
        self.check(d_token_table, spec.vocab * spec.channels)?;
        self.check(d_position_table, spec.positions * spec.channels)?;
        self.clear(d_token_table)?;
        self.clear(d_position_table)?;
        launch!(
            self,
            embedding_backward,
            n,
            &tokens.data,
            &d_output.data,
            &mut d_token_table.data,
            &mut d_position_table.data,
            &n,
            &spec.time,
            &spec.channels
        );
        Ok(())
    }

    fn cross_entropy(
        &self,
        logits: &CudaBuffer,
        targets: &CudaIndexBuffer,
        rows: usize,
        vocab: usize,
    ) -> Result<f32> {
        self.loss(logits, targets, rows, vocab, None)
    }

    fn cross_entropy_with_grad(
        &self,
        logits: &CudaBuffer,
        targets: &CudaIndexBuffer,
        rows: usize,
        vocab: usize,
        d_logits: &mut CudaBuffer,
    ) -> Result<f32> {
        self.loss(logits, targets, rows, vocab, Some(d_logits))
    }

    fn buffers_all_finite(&self, buffers: &[&CudaBuffer]) -> Result<bool> {
        for buffer in buffers {
            self.check(buffer, buffer.len)?;
        }
        let mut invalid = self.stream.alloc_zeros::<u32>(1)?;
        for buffer in buffers {
            launch!(
                self,
                all_finite,
                buffer.len,
                &buffer.data,
                &mut invalid,
                &buffer.len
            );
        }
        let result = self.stream.clone_dtoh(&invalid)?[0] == 0;
        self.synchronize()?;
        Ok(result)
    }

    fn adamw_step(
        &self,
        groups: &mut [AdamWGroup<'_, CudaBuffer>],
        config: AdamWConfig,
    ) -> Result<()> {
        validation::adamw(config)?;
        for group in groups.iter() {
            let n = group.values.len;
            for buffer in [
                &*group.values,
                group.gradients,
                &*group.first_moment,
                &*group.second_moment,
            ] {
                self.check(buffer, n)?;
            }
        }
        let correction1 = 1.0 - config.beta1.powi(config.step);
        let correction2 = 1.0 - config.beta2.powi(config.step);
        let mut invalid = self.stream.alloc_zeros::<u32>(1)?;
        for group in groups {
            let n = group.values.len;
            launch!(
                self,
                adamw,
                n,
                &mut group.values.data,
                &group.gradients.data,
                &mut group.first_moment.data,
                &mut group.second_moment.data,
                &mut invalid,
                &n,
                &config.learning_rate,
                &config.weight_decay,
                &config.beta1,
                &config.beta2,
                &config.epsilon,
                &correction1,
                &correction2
            );
        }
        let bad = self.stream.clone_dtoh(&invalid)?[0];
        self.synchronize()?;
        ensure!(
            bad == 0,
            "non-finite optimizer update; reduce the learning rate"
        );
        Ok(())
    }

    fn sample_last_token(
        &self,
        logits: &CudaBuffer,
        rows: usize,
        vocab: usize,
        uniform: f32,
    ) -> Result<usize> {
        validation::dimension(rows)?;
        validation::dimension(vocab)?;
        self.check(logits, validation::elements(&[rows, vocab])?)?;
        ensure!(
            (0.0..=1.0).contains(&uniform),
            "sampling uniform must be finite and in [0, 1]"
        );
        let mut sampled = self.stream.alloc_zeros::<u32>(1)?;
        launch!(
            self,
            sample,
            1usize,
            &logits.data,
            &mut sampled,
            &rows,
            &vocab,
            &uniform
        );
        let index = self.stream.clone_dtoh(&sampled)?[0];
        self.synchronize()?;
        ensure!(index != u32::MAX, "non-finite sampling logits");
        Ok(index as usize)
    }
}
