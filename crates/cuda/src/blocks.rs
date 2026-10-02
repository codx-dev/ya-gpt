use anyhow::{Context, Result, ensure};
use ya_gpt::engine::*;

use crate::{CudaEngine, kernels::launch, types::*, validation};

fn norm_spec(s: BlockSpec) -> NormSpec {
    NormSpec {
        rows: s.batch * s.time,
        channels: s.channels,
        epsilon: 1e-5,
    }
}

fn linear_spec(s: BlockSpec, inputs: usize, outputs: usize) -> LinearSpec {
    LinearSpec {
        rows: s.batch * s.time,
        inputs,
        outputs,
    }
}

#[allow(clippy::too_many_arguments)]
impl CudaEngine {
    fn check_block_weights(&self, s: BlockSpec, w: &BlockWeights<'_, CudaBuffer>) -> Result<()> {
        let c = s.channels;
        self.check_norm_weights(norm_spec(s), &w.norm1)?;
        self.check_norm_weights(norm_spec(s), &w.norm2)?;
        self.check(w.qkv, validation::elements(&[3, c, c])?)?;
        self.check_linear_weights(linear_spec(s, c, c), &w.attention)?;
        self.check_linear_weights(linear_spec(s, c, 4 * c), &w.expand)?;
        self.check_linear_weights(linear_spec(s, 4 * c, c), &w.project)
    }

    fn check_block_gradients(
        &self,
        s: BlockSpec,
        g: &BlockGradients<'_, CudaBuffer>,
    ) -> Result<()> {
        let c = s.channels;
        for norm in [&g.norm1, &g.norm2] {
            self.check(norm.gamma, c)?;
            self.check(norm.beta, c)?;
        }
        self.check(g.qkv, 3 * c * c)?;
        for (linear, inputs, outputs) in [
            (&g.attention, c, c),
            (&g.expand, c, 4 * c),
            (&g.project, 4 * c, c),
        ] {
            self.check(linear.weight, inputs * outputs)?;
            if let Some(bias) = linear.bias.as_ref() {
                self.check(bias, outputs)?;
            }
        }
        Ok(())
    }

    pub(crate) fn attention_forward(
        &self,
        s: BlockSpec,
        qkv: &CudaBuffer,
        probabilities: &mut CudaBuffer,
        mask: &mut CudaBuffer,
        out: &mut CudaBuffer,
        mode: ForwardMode,
    ) -> Result<()> {
        let (n, count) = validation::block(s)?;
        self.check(qkv, 3 * n)?;
        self.check(probabilities, count)?;
        self.check(mask, count)?;
        self.check(out, n)?;
        let rows = count / s.time;
        let p = if mode.seed.is_some() { s.dropout } else { 0.0 };
        let seed = mode.seed.unwrap_or(0);
        launch!(
            self,
            attention_scores,
            count,
            &qkv.data,
            &mut probabilities.data,
            &count,
            &s.time,
            &s.channels,
            &s.heads
        );
        launch!(
            self,
            attention_softmax,
            rows,
            &mut probabilities.data,
            &mut mask.data,
            &rows,
            &s.time,
            &p,
            &seed
        );
        launch!(
            self,
            attention_context,
            n,
            &qkv.data,
            &probabilities.data,
            &mask.data,
            &mut out.data,
            &n,
            &s.time,
            &s.channels,
            &s.heads
        );
        Ok(())
    }

    fn attention_backward(
        &self,
        s: BlockSpec,
        saved: &SavedBlock,
        dy: &CudaBuffer,
        ds: &mut CudaBuffer,
        dqkv: &mut CudaBuffer,
    ) -> Result<()> {
        let count = ds.len;
        let rows = count / s.time;
        let scale = 1.0 / ((s.channels / s.heads) as f32).sqrt();
        launch!(
            self,
            attention_dprob,
            count,
            &saved.qkv.data,
            &dy.data,
            &saved.attention_mask.data,
            &mut ds.data,
            &count,
            &s.time,
            &s.channels,
            &s.heads
        );
        launch!(
            self,
            attention_dsoftmax,
            rows,
            &saved.probabilities.data,
            &mut ds.data,
            &rows,
            &s.time,
            &scale
        );
        launch!(
            self,
            attention_dqkv,
            dqkv.len,
            &saved.qkv.data,
            &saved.probabilities.data,
            &saved.attention_mask.data,
            &dy.data,
            &ds.data,
            &mut dqkv.data,
            &dqkv.len,
            &s.time,
            &s.channels,
            &s.heads
        );
        Ok(())
    }

    fn dropout_residual(
        &self,
        branch: &mut CudaBuffer,
        residual: &CudaBuffer,
        mask: &mut CudaBuffer,
        probability: f32,
        mode: ForwardMode,
        stream: u64,
    ) -> Result<()> {
        let p = if mode.seed.is_some() {
            probability
        } else {
            0.0
        };
        let seed = mode.seed.unwrap_or(0);
        launch!(
            self,
            dropout_add,
            branch.len,
            &mut branch.data,
            &residual.data,
            &mut mask.data,
            &branch.len,
            &p,
            &seed,
            &stream
        );
        Ok(())
    }

    fn masked(&self, input: &CudaBuffer, mask: &CudaBuffer, output: &mut CudaBuffer) -> Result<()> {
        launch!(
            self,
            masked,
            input.len,
            &input.data,
            &mask.data,
            &mut output.data,
            &input.len
        );
        Ok(())
    }

    pub(crate) fn forward_block(
        &self,
        s: BlockSpec,
        weights: BlockWeights<'_, CudaBuffer>,
        input: &CudaBuffer,
        mode: ForwardMode,
        ws: &mut CudaWorkspace,
        output: &mut CudaBuffer,
        mut cache: Option<&mut CudaBlockCache>,
    ) -> Result<()> {
        if let Some(saved) = cache.as_deref_mut() {
            saved.spec = None;
        }
        let (n, scores) = validation::block(s)?;
        self.check_block_weights(s, &weights)?;
        self.check(input, n)?;
        self.check(output, n)?;
        let c = s.channels;
        let norm = norm_spec(s);
        let linear = |i, o| linear_spec(s, i, o);
        let normalized1 = self.resize(&mut ws.normalized1, n)?;
        let qkv = self.resize(&mut ws.qkv, 3 * n)?;
        let context = self.resize(&mut ws.context, n)?;
        let after_attention = self.resize(&mut ws.after_attention, n)?;
        let normalized2 = self.resize(&mut ws.normalized2, n)?;
        let activated = self.resize(&mut ws.activated, 4 * n)?;
        let projected = self.resize(&mut ws.projected, n)?;
        let probabilities = self.resize(&mut ws.probabilities, scores)?;
        let attention_mask = self.resize(&mut ws.attention_mask, scores)?;
        let projection_mask = self.resize(&mut ws.projection_mask, n)?;
        let mlp_mask = self.resize(&mut ws.mlp_mask, n)?;

        self.layer_norm_forward(
            norm,
            weights.norm1,
            input,
            normalized1,
            cache.as_deref_mut().map(|saved| &mut saved.norm1),
        )?;
        self.linear_forward(
            linear(c, 3 * c),
            LinearWeights {
                weight: weights.qkv,
                bias: None,
            },
            normalized1,
            qkv,
        )?;
        self.attention_forward(s, qkv, probabilities, attention_mask, context, mode)?;
        self.linear_forward(linear(c, c), weights.attention, context, after_attention)?;
        self.dropout_residual(after_attention, input, projection_mask, s.dropout, mode, 1)?;
        self.layer_norm_forward(
            norm,
            weights.norm2,
            after_attention,
            normalized2,
            cache.as_deref_mut().map(|saved| &mut saved.norm2),
        )?;
        self.linear_forward(linear(c, 4 * c), weights.expand, normalized2, activated)?;
        launch!(
            self,
            relu,
            activated.len,
            &mut activated.data,
            &activated.len
        );
        self.linear_forward(linear(4 * c, c), weights.project, activated, projected)?;
        self.dropout_residual(projected, after_attention, mlp_mask, s.dropout, mode, 2)?;
        self.copy_buffer(projected, output)?;

        if let Some(cache) = cache {
            // Deep device copies: subsequent inference may overwrite or resize
            // every workspace allocation before this cache is used by backward.
            cache.saved = Some(SavedBlock {
                normalized1: self.clone_buffer(normalized1)?,
                qkv: self.clone_buffer(qkv)?,
                context: self.clone_buffer(context)?,
                normalized2: self.clone_buffer(normalized2)?,
                activated: self.clone_buffer(activated)?,
                probabilities: self.clone_buffer(probabilities)?,
                attention_mask: self.clone_buffer(attention_mask)?,
                projection_mask: self.clone_buffer(projection_mask)?,
                mlp_mask: self.clone_buffer(mlp_mask)?,
            });
            cache.spec = Some(s);
        }
        Ok(())
    }

    pub(crate) fn backward_block(
        &self,
        s: BlockSpec,
        weights: BlockWeights<'_, CudaBuffer>,
        gradients: BlockGradients<'_, CudaBuffer>,
        cache: &CudaBlockCache,
        d_output: &CudaBuffer,
        ws: &mut CudaWorkspace,
        d_input: &mut CudaBuffer,
    ) -> Result<()> {
        let (n, scores) = validation::block(s)?;
        self.check_block_weights(s, &weights)?;
        self.check_block_gradients(s, &gradients)?;
        ensure!(cache.spec == Some(s), "missing or incompatible block cache");
        let saved = cache.saved.as_ref().context("missing block activations")?;
        self.check_norm_cache(norm_spec(s), &cache.norm1)?;
        self.check_norm_cache(norm_spec(s), &cache.norm2)?;
        for (buffer, len) in [
            (&saved.normalized1, n),
            (&saved.qkv, 3 * n),
            (&saved.context, n),
            (&saved.normalized2, n),
            (&saved.activated, 4 * n),
            (&saved.probabilities, scores),
            (&saved.attention_mask, scores),
            (&saved.projection_mask, n),
            (&saved.mlp_mask, n),
        ] {
            self.check(buffer, len)?;
        }
        self.check(d_output, n)?;
        self.check(d_input, n)?;
        let c = s.channels;
        let norm = norm_spec(s);
        let linear = |i, o| linear_spec(s, i, o);
        let d0 = self.resize(&mut ws.d0, n)?;
        let d1 = self.resize(&mut ws.d1, n)?;
        let d2 = self.resize(&mut ws.d2, n)?;
        let d_hidden = self.resize(&mut ws.d_hidden, 4 * n)?;
        let d_qkv = self.resize(&mut ws.d_qkv, 3 * n)?;
        let d_scores = self.resize(&mut ws.d_scores, scores)?;

        self.masked(d_output, &saved.mlp_mask, d0)?;
        self.linear_backward(
            linear(4 * c, c),
            weights.project,
            gradients.project,
            &saved.activated,
            d0,
            d_hidden,
        )?;
        launch!(
            self,
            relu_backward,
            d_hidden.len,
            &saved.activated.data,
            &mut d_hidden.data,
            &d_hidden.len
        );
        self.linear_backward(
            linear(c, 4 * c),
            weights.expand,
            gradients.expand,
            &saved.normalized2,
            d_hidden,
            d0,
        )?;
        self.layer_norm_backward(norm, weights.norm2, gradients.norm2, &cache.norm2, d0, d1)?;
        self.add_buffer(d1, d_output)?;
        self.masked(d1, &saved.projection_mask, d0)?;
        self.linear_backward(
            linear(c, c),
            weights.attention,
            gradients.attention,
            &saved.context,
            d0,
            d2,
        )?;
        self.attention_backward(s, saved, d2, d_scores, d_qkv)?;
        self.linear_backward(
            linear(c, 3 * c),
            LinearWeights {
                weight: weights.qkv,
                bias: None,
            },
            LinearGradients {
                weight: gradients.qkv,
                bias: None,
            },
            &saved.normalized1,
            d_qkv,
            d0,
        )?;
        self.layer_norm_backward(
            norm,
            weights.norm1,
            gradients.norm1,
            &cache.norm1,
            d0,
            d_input,
        )?;
        self.add_buffer(d_input, d1)?;
        Ok(())
    }
}
