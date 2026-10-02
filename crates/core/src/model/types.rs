use rand::{Rng, RngExt as _};
use rand_distr::{Distribution as _, Normal};

use crate::{config::ModelConfig, engine::Engine};

// TODO this should probably have a Vec<f64> (disk/memory) and EN::Buffer (runtime) implementations
pub struct Parameter<EN: Engine> {
    pub values: EN::Buffer,
    pub gradients: EN::Buffer,
}

impl<EN: Engine> Parameter<EN> {
    pub fn filled(en: &EN, len: usize, value: f64) -> anyhow::Result<Self> {
        Ok(Self {
            values: en.filled(value, len)?,
            gradients: en.zeroes(len)?,
        })
    }

    pub fn normal(en: &EN, len: usize, rng: &mut impl Rng) -> anyhow::Result<Self> {
        let distribution = Normal::<f64>::new(0.0, 0.02)
            .expect("weight initialization has a finite, positive standard deviation");
        let values: Vec<f64> = (0..len).map(|_| distribution.sample(rng)).collect();

        Ok(Self {
            values: en.buffer_from_slice(values)?,
            gradients: en.zeroes(len)?,
        })
    }
}

pub struct Linear<EN: Engine> {
    pub weight: Parameter<EN>,
    pub bias: Option<Parameter<EN>>,
    pub input_size: usize,
    pub output_size: usize,
}

impl<EN: Engine> Linear<EN> {
    pub fn new(
        en: &EN,
        input_size: usize,
        output_size: usize,
        bias: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Self> {
        let weight = Parameter::normal(en, input_size * output_size, rng)?;
        let bias = bias
            .then(|| Parameter::filled(en, output_size, 0.0))
            .transpose()?;

        Ok(Self {
            weight,
            bias,
            input_size,
            output_size,
        })
    }

    pub fn forward(&self, en: &EN, input: &EN::Buffer, rows: usize) -> anyhow::Result<EN::Buffer> {
        let len = rows.saturating_mul(self.output_size);
        let mut out = en.zeroes(len)?;

        en.matmul(
            input,
            &self.weight.values,
            rows,
            self.input_size,
            self.output_size,
            &mut out,
        )?;

        if let Some(bias) = &self.bias {
            en.add_bias_in_place(&mut out, &bias.values, rows, self.output_size)?;
        }

        Ok(out)
    }

    pub fn backward(
        &mut self,
        en: &EN,
        input: &EN::Buffer,
        d_output: &EN::Buffer,
        rows: usize,
    ) -> anyhow::Result<EN::Buffer> {
        let d_input_len = rows.saturating_mul(self.input_size);
        let d_weight_len = self.input_size.saturating_mul(self.output_size);
        let mut d_input = en.zeroes(d_input_len)?;
        let mut d_weight = en.zeroes(d_weight_len)?;

        en.matmul_backward(
            input,
            &self.weight.values,
            d_output,
            rows,
            self.input_size,
            self.output_size,
            &mut d_input,
            &mut d_weight,
        )?;

        self.weight.gradients = d_weight;

        if let Some(bias) = &mut self.bias {
            en.sum_rows(d_output, rows, self.output_size, &mut bias.gradients)?;
        }

        Ok(d_input)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        let mut result = vec![&self.weight];
        if let Some(bias) = &self.bias {
            result.push(bias);
        }
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        let mut result = vec![&mut self.weight];
        if let Some(bias) = &mut self.bias {
            result.push(bias);
        }
        result
    }
}

pub struct LayerNorm<EN: Engine> {
    pub gamma: Parameter<EN>,
    pub beta: Parameter<EN>,
}

impl<EN: Engine> LayerNorm<EN> {
    pub fn new(en: &EN, channels: usize) -> anyhow::Result<Self> {
        let gamma = Parameter::filled(en, channels, 1.0)?;
        let beta = Parameter::filled(en, channels, 0.0)?;

        Ok(Self { gamma, beta })
    }

    pub fn forward(&self, en: &EN, input: &EN::Buffer, rows: usize) -> anyhow::Result<EN::Buffer> {
        let input_len = en.buffer_len(input);
        let gamma_len = en.buffer_len(&self.gamma.values);
        let mut out = en.zeroes(input_len)?;

        en.layer_norm(
            input,
            &self.gamma.values,
            &self.beta.values,
            rows,
            gamma_len,
            1e-5,
            &mut out,
        )?;

        Ok(out)
    }

    pub fn backward(
        &mut self,
        en: &EN,
        input: &EN::Buffer,
        d_output: &EN::Buffer,
        rows: usize,
    ) -> anyhow::Result<EN::Buffer> {
        let input_len = en.buffer_len(input);
        let gamma_len = en.buffer_len(&self.gamma.values);

        let mut d_input = en.zeroes(input_len)?;
        let mut d_gamma = en.zeroes(gamma_len)?;
        let mut d_beta = en.zeroes(gamma_len)?;

        en.layer_norm_backward(
            input,
            &self.gamma.values,
            d_output,
            rows,
            gamma_len,
            1e-5,
            &mut d_input,
            &mut d_gamma,
            &mut d_beta,
        )?;

        self.gamma.gradients = d_gamma;
        self.beta.gradients = d_beta;

        Ok(d_input)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        vec![&self.gamma, &self.beta]
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        vec![&mut self.gamma, &mut self.beta]
    }
}

pub struct Head<EN: Engine> {
    pub query: Linear<EN>,
    pub key: Linear<EN>,
    pub value: Linear<EN>,
    pub size: usize,
    pub dropout: f64,
}

pub struct HeadCache<EN: Engine> {
    pub input: EN::Buffer,
    pub query: EN::Buffer,
    pub key: EN::Buffer,
    pub value: EN::Buffer,
    pub probabilities: EN::Buffer,
    pub mask: EN::Buffer,
    pub weights: EN::Buffer,
}

impl<EN: Engine> Head<EN> {
    pub fn new(
        en: &EN,
        channels: usize,
        size: usize,
        dropout: f64,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Self> {
        let query = Linear::new(en, channels, size, false, rng)?;
        let key = Linear::new(en, channels, size, false, rng)?;
        let value = Linear::new(en, channels, size, false, rng)?;

        Ok(Self {
            query,
            key,
            value,
            size,
            dropout,
        })
    }

    pub fn forward(
        &self,
        en: &EN,
        input: EN::Buffer,
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<(EN::Buffer, HeadCache<EN>)> {
        let batch_time = batch.saturating_mul(time);
        let batch_time_size = batch_time.saturating_mul(self.size);
        let batch_time2 = batch_time.saturating_mul(time);
        let time2 = time.saturating_mul(time);
        let time_size = time.saturating_mul(self.size);

        let query = self.query.forward(en, &input, batch_time)?;
        let key = self.key.forward(en, &input, batch_time)?;
        let value = self.value.forward(en, &input, batch_time)?;

        let mut ofs = 0;
        let mut probabilities = en.zeroes(batch_time2)?;
        let mut transpose = en.zeroes(time_size)?;
        let mut scores = en.zeroes(time2)?;

        for b in 0..batch {
            let offset = b.saturating_mul(time_size);
            let k = en.slice(&key, offset, time_size);
            let q = en.slice(&query, offset, time_size);

            en.transpose(&k, time, self.size, &mut transpose)?;
            en.matmul(&q, &transpose, time, self.size, time, &mut scores)?;
            en.scale_in_place(&mut scores, 1.0 / (self.size as f64).sqrt())?;
            en.causal_mask_in_place(&mut scores, time)?;
            en.softmax_rows_in_place(&mut scores, time, time)?;

            en.copy(&mut probabilities, &scores, ofs);
            ofs = ofs.saturating_add(time2);
        }

        let mask = dropout_mask(en, batch_time2, self.dropout, training, rng)?;
        let mut weights = en.clone(&probabilities)?;

        en.dropout_in_place(&mut weights, &mask)?;

        let mut ofs = 0;
        let mut output = en.zeroes(batch_time_size)?;
        let mut mul = en.zeroes(time_size)?;

        for b in 0..batch {
            let offset = b.saturating_mul(time2);
            let w = en.slice(&weights, offset, time2);

            let offset = b.saturating_mul(time_size);
            let v = en.slice(&value, offset, time_size);

            en.matmul(&w, &v, time, time, self.size, &mut mul)?;

            en.copy(&mut output, &mul, ofs);
            ofs = ofs.saturating_add(time_size);
        }

        let head = HeadCache {
            input,
            query,
            key,
            value,
            probabilities,
            mask,
            weights,
        };

        Ok((output, head))
    }

    pub fn backward(
        &mut self,
        en: &EN,
        cache: &HeadCache<EN>,
        d_output: &EN::Buffer,
        batch: usize,
        time: usize,
    ) -> anyhow::Result<EN::Buffer> {
        let time2 = time.saturating_mul(time);
        let time_size = time.saturating_mul(self.size);
        let batch_time = batch.saturating_mul(time);
        let batch_time_size = batch.saturating_mul(time_size);
        let batch_time_channels = batch_time.saturating_mul(self.query.input_size);

        let mut d_query = en.zeroes(batch_time_size)?;
        let mut d_key = en.zeroes(batch_time_size)?;
        let mut d_value = en.zeroes(batch_time_size)?;

        let mut d_scores = en.zeroes(time2)?;
        let mut dq = en.zeroes(time_size)?;
        let mut dk_key_transposed = en.zeroes(time_size)?;
        let mut dk_transposed = en.zeroes(time_size)?;

        let mut d_query_ofs = 0;
        let mut d_key_ofs = 0;
        let mut d_value_ofs = 0;
        let mut d_weights = en.zeroes(time2)?;
        let mut dv = en.zeroes(time_size)?;

        for b in 0..batch {
            let vectors_ofs = b.saturating_mul(time_size);
            let vectors_len = time_size;

            let matrix_ofs = b.saturating_mul(time2);
            let matrix_len = time2;

            let w = en.slice(&cache.weights, matrix_ofs, matrix_len);
            let v = en.slice(&cache.value, vectors_ofs, vectors_len);
            let o = en.slice(d_output, vectors_ofs, vectors_len);

            en.matmul_backward(&w, &v, &o, time, time, self.size, &mut d_weights, &mut dv)?;

            en.copy(&mut d_value, &dv, d_value_ofs);
            d_value_ofs = d_value_ofs.saturating_add(time_size);

            let m = en.slice(&cache.mask, matrix_ofs, matrix_len);

            en.dropout_in_place(&mut d_weights, &m)?;

            let p = en.slice(&cache.probabilities, matrix_ofs, matrix_len);

            en.softmax_rows_backward(&p, &d_weights, time, time, &mut d_scores)?;
            en.causal_mask_backward_in_place(&mut d_scores, time)?;
            en.scale_in_place(&mut d_scores, 1.0 / (self.size as f64).sqrt())?;

            let q = en.slice(&cache.query, vectors_ofs, vectors_len);
            let k = en.slice(&cache.key, vectors_ofs, vectors_len);

            en.transpose(&k, time, self.size, &mut dk_key_transposed)?;
            en.matmul_backward(
                &q,
                &dk_key_transposed,
                &d_scores,
                time,
                self.size,
                time,
                &mut dq,
                &mut dk_transposed,
            )?;

            en.copy(&mut d_query, &dq, d_query_ofs);
            d_query_ofs = d_query_ofs.saturating_add(time_size);

            en.transpose_in_place(&mut dk_transposed, self.size, time)?;

            en.copy(&mut d_key, &dk_transposed, d_key_ofs);
            d_key_ofs = d_key_ofs.saturating_add(time_size);
        }

        let mut output = en.zeroes(batch_time_channels)?;

        let dq = self
            .query
            .backward(en, &cache.input, &d_query, batch_time)?;
        let dk = self.key.backward(en, &cache.input, &d_key, batch_time)?;
        let dv = self
            .value
            .backward(en, &cache.input, &d_value, batch_time)?;

        en.add(&dq, &dk, &mut output)?;
        en.add_in_place(&mut output, &dv)?;

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        let mut result = self.query.parameters();
        result.extend(self.key.parameters());
        result.extend(self.value.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        let mut result = self.query.parameters_mut();
        result.extend(self.key.parameters_mut());
        result.extend(self.value.parameters_mut());
        result
    }
}

pub struct Attention<EN: Engine> {
    pub heads: Vec<Head<EN>>,
    pub projection: Linear<EN>,
    pub dropout: f64,
}

pub struct AttentionCache<EN: Engine> {
    pub heads: Vec<HeadCache<EN>>,
    pub concatenated: EN::Buffer,
    pub mask: EN::Buffer,
}

impl<EN: Engine> Attention<EN> {
    pub fn new(en: &EN, config: &ModelConfig, rng: &mut impl Rng) -> anyhow::Result<Self> {
        let projection = Linear::new(en, config.n_embd, config.n_embd, true, rng)?;
        let heads = (0..config.n_head)
            .map(|_| {
                Head::new(
                    en,
                    config.n_embd,
                    config.n_embd / config.n_head,
                    config.dropout,
                    rng,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            heads,
            projection,
            dropout: config.dropout,
        })
    }

    pub fn forward(
        &self,
        en: &EN,
        input: &EN::Buffer,
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<(EN::Buffer, AttentionCache<EN>)> {
        let mut outputs = Vec::with_capacity(self.heads.len());
        let mut caches = Vec::with_capacity(self.heads.len());

        for head in &self.heads {
            let input = en.clone(input)?;
            let (output, cache) = head.forward(en, input, batch, time, training, rng)?;

            outputs.push(output);
            caches.push(cache);
        }

        let head_size = self.heads[0].size;
        let outputs_len = en.buffer_len(&outputs[0]);
        let mut concatenated = en.clone(&outputs[0])?;

        for (index, output) in outputs.iter().enumerate().skip(1) {
            // TODO not efficient realloc; refactor
            let mut next = en.zeroes(outputs_len.saturating_mul(index + 1))?;
            en.concat_columns(
                &concatenated,
                output,
                batch * time,
                index * head_size,
                head_size,
                &mut next,
            )?;
            concatenated = next;
        }

        let mut projected = self.projection.forward(en, &concatenated, batch * time)?;
        let len = en.buffer_len(&projected);
        let mask = dropout_mask(en, len, self.dropout, training, rng)?;

        en.dropout_in_place(&mut projected, &mask)?;

        let attention = AttentionCache {
            heads: caches,
            concatenated,
            mask,
        };

        Ok((projected, attention))
    }

    pub fn backward(
        &mut self,
        en: &EN,
        cache: &AttentionCache<EN>,
        d_output: &EN::Buffer,
        batch: usize,
        time: usize,
    ) -> anyhow::Result<EN::Buffer> {
        let channels = self.projection.input_size;
        let batch_time = batch.saturating_mul(time);
        let batch_time_channels = batch_time.saturating_mul(channels);

        let d_output_len = en.buffer_len(d_output);
        let mut d_projection = en.zeroes(d_output_len)?;

        en.dropout(d_output, &cache.mask, &mut d_projection)?;

        let d_concat =
            self.projection
                .backward(en, &cache.concatenated, &d_projection, batch_time)?;

        let mut d_input = en.zeroes(batch_time_channels)?;

        let d_head_len = batch_time.saturating_mul(self.heads[0].size);
        let mut d_head = en.zeroes(d_head_len)?;

        for (index, head) in self.heads.iter_mut().enumerate() {
            en.slice_columns(
                &d_concat,
                batch * time,
                channels,
                index * head.size,
                head.size,
                &mut d_head,
            )?;

            let dx = head.backward(en, &cache.heads[index], &d_head, batch, time)?;

            en.add_in_place(&mut d_input, &dx)?;
        }

        Ok(d_input)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        let mut result = Vec::new();
        for head in &self.heads {
            result.extend(head.parameters());
        }
        result.extend(self.projection.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        let mut result = Vec::new();
        for head in &mut self.heads {
            result.extend(head.parameters_mut());
        }
        result.extend(self.projection.parameters_mut());
        result
    }
}

pub struct FeedForward<EN: Engine> {
    pub expand: Linear<EN>,
    pub project: Linear<EN>,
    pub dropout: f64,
}

pub struct FeedForwardCache<EN: Engine> {
    pub input: EN::Buffer,
    pub pre_relu: EN::Buffer,
    pub activated: EN::Buffer,
    pub mask: EN::Buffer,
}

impl<EN: Engine> FeedForward<EN> {
    pub fn new(en: &EN, channels: usize, dropout: f64, rng: &mut impl Rng) -> anyhow::Result<Self> {
        let expand = Linear::new(en, channels, 4 * channels, true, rng)?;
        let project = Linear::new(en, 4 * channels, channels, true, rng)?;

        Ok(Self {
            expand,
            project,
            dropout,
        })
    }

    pub fn forward(
        &self,
        en: &EN,
        input: EN::Buffer,
        rows: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<(EN::Buffer, FeedForwardCache<EN>)> {
        let pre_relu = self.expand.forward(en, &input, rows)?;
        let mut activated = en.clone(&pre_relu)?;

        en.relu(&pre_relu, &mut activated)?;

        let mut projected = self.project.forward(en, &activated, rows)?;
        let projected_len = en.buffer_len(&projected);
        let mask = dropout_mask(en, projected_len, self.dropout, training, rng)?;

        en.dropout_in_place(&mut projected, &mask)?;

        let cache = FeedForwardCache {
            input,
            pre_relu,
            activated,
            mask,
        };

        Ok((projected, cache))
    }

    pub fn backward(
        &mut self,
        en: &EN,
        cache: &FeedForwardCache<EN>,
        d_output: &EN::Buffer,
        rows: usize,
    ) -> anyhow::Result<EN::Buffer> {
        let projected_len = en.buffer_len(d_output);
        let mut d_projected = en.zeroes(projected_len)?;

        en.dropout(d_output, &cache.mask, &mut d_projected)?;

        let d_activated = self
            .project
            .backward(en, &cache.activated, &d_projected, rows)?;

        let mut d_pre_relu = en.clone(&d_activated)?;

        en.relu_backward(&cache.pre_relu, &d_activated, &mut d_pre_relu)?;

        self.expand.backward(en, &cache.input, &d_pre_relu, rows)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        let mut result = self.expand.parameters();
        result.extend(self.project.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        let mut result = self.expand.parameters_mut();
        result.extend(self.project.parameters_mut());
        result
    }
}

pub struct Block<EN: Engine> {
    pub norm1: LayerNorm<EN>,
    pub attention: Attention<EN>,
    pub norm2: LayerNorm<EN>,
    pub feed_forward: FeedForward<EN>,
}

pub struct BlockCache<EN: Engine> {
    pub input: EN::Buffer,
    pub after_attention: EN::Buffer,
    pub attention: AttentionCache<EN>,
    pub feed_forward: FeedForwardCache<EN>,
}

impl<EN: Engine> Block<EN> {
    pub fn new(en: &EN, config: &ModelConfig, rng: &mut impl Rng) -> anyhow::Result<Self> {
        let norm1 = LayerNorm::new(en, config.n_embd)?;
        let attention = Attention::new(en, config, rng)?;
        let norm2 = LayerNorm::new(en, config.n_embd)?;
        let feed_forward = FeedForward::new(en, config.n_embd, config.dropout, rng)?;

        Ok(Self {
            norm1,
            attention,
            norm2,
            feed_forward,
        })
    }

    pub fn forward(
        &self,
        en: &EN,
        input: EN::Buffer,
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<(EN::Buffer, BlockCache<EN>)> {
        let batch_time = batch.saturating_mul(time);

        let normalized = self.norm1.forward(en, &input, batch_time)?;
        let (mut after_attention, attention) =
            self.attention
                .forward(en, &normalized, batch, time, training, rng)?;

        en.add_in_place(&mut after_attention, &input)?;

        let normalized = self.norm2.forward(en, &after_attention, batch_time)?;
        let (fed, feed_forward) =
            self.feed_forward
                .forward(en, normalized, batch * time, training, rng)?;

        let mut output = en.clone(&after_attention)?;
        en.add_in_place(&mut output, &fed)?;

        let cache = BlockCache {
            input,
            after_attention,
            attention,
            feed_forward,
        };

        Ok((output, cache))
    }

    pub fn backward(
        &mut self,
        en: &EN,
        cache: &BlockCache<EN>,
        d_output: &EN::Buffer,
        batch: usize,
        time: usize,
    ) -> anyhow::Result<EN::Buffer> {
        let batch_time = batch.saturating_mul(time);

        let d_norm2 = self
            .feed_forward
            .backward(en, &cache.feed_forward, d_output, batch_time)?;

        let d_residual2 = self
            .norm2
            .backward(en, &cache.after_attention, &d_norm2, batch_time)?;

        let mut d_after_attention = en.clone(d_output)?;
        en.add_in_place(&mut d_after_attention, &d_residual2)?;

        let d_norm1 =
            self.attention
                .backward(en, &cache.attention, &d_after_attention, batch, time)?;

        let d_residual1 = self
            .norm1
            .backward(en, &cache.input, &d_norm1, batch_time)?;

        en.add_in_place(&mut d_after_attention, &d_residual1)?;

        Ok(d_after_attention)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        let mut result = self.norm1.parameters();
        result.extend(self.attention.parameters());
        result.extend(self.norm2.parameters());
        result.extend(self.feed_forward.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        let mut result = self.norm1.parameters_mut();
        result.extend(self.attention.parameters_mut());
        result.extend(self.norm2.parameters_mut());
        result.extend(self.feed_forward.parameters_mut());
        result
    }
}

/// Returns an inverted-dropout mask: zero for dropped entries, 1/(1-p) otherwise.
/// Evaluation, zero dropout, and empty masks do not consume randomness.
fn dropout_mask<EN: Engine>(
    en: &EN,
    len: usize,
    probability: f64,
    training: bool,
    rng: &mut impl Rng,
) -> anyhow::Result<EN::Buffer> {
    // TODO maybe should do distribution inside engine?
    debug_assert!((0.0..1.0).contains(&probability));

    if !training || probability == 0.0 {
        return en.filled(1.0, len);
    }

    let mask: Vec<_> = (0..len)
        .map(|_| {
            if rng.random_bool(probability) {
                0.0
            } else {
                1.0 / (1.0 - probability)
            }
        })
        .collect();

    en.buffer_from_slice(&mask)
}
