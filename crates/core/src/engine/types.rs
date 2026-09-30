use msgpacker::MsgPacker;
use rand::{Rng, RngExt as _};
use rand_distr::{Distribution as _, Normal};

use crate::{config::ModelConfig, engine::Engine};

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct Parameter {
    pub values: Vec<f64>,
    pub gradients: Vec<f64>,
}

impl Parameter {
    pub fn filled(len: usize, value: f64) -> Self {
        Self {
            values: vec![value; len],
            gradients: vec![0.0; len],
        }
    }

    pub fn normal(len: usize, rng: &mut impl Rng) -> Self {
        let distribution = Normal::<f64>::new(0.0, 0.02)
            .expect("weight initialization has a finite, positive standard deviation");

        Self {
            values: (0..len).map(|_| distribution.sample(rng)).collect(),
            gradients: vec![0.0; len],
        }
    }
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    input_size: usize,
    output_size: usize,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, bias: bool, rng: &mut impl Rng) -> Self {
        Self {
            weight: Parameter::normal(input_size * output_size, rng),
            bias: bias.then(|| Parameter::filled(output_size, 0.0)),
            input_size,
            output_size,
        }
    }

    pub fn forward<E: Engine>(&self, en: &E, input: &[f64], rows: usize) -> Vec<f64> {
        let output = en.matmul(
            input,
            &self.weight.values,
            rows,
            self.input_size,
            self.output_size,
        );
        match &self.bias {
            Some(bias) => en.add_bias(&output, &bias.values, rows, self.output_size),
            None => output,
        }
    }

    pub fn backward<E: Engine>(
        &mut self,
        en: &E,
        input: &[f64],
        d_output: &[f64],
        rows: usize,
    ) -> Vec<f64> {
        let (d_input, d_weight) = en.matmul_backward(
            input,
            &self.weight.values,
            d_output,
            rows,
            self.input_size,
            self.output_size,
        );
        self.weight.gradients = d_weight;
        if let Some(bias) = &mut self.bias {
            bias.gradients = en.sum_rows(d_output, rows, self.output_size);
        }
        d_input
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        let mut result = vec![&self.weight];
        if let Some(bias) = &self.bias {
            result.push(bias);
        }
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut result = vec![&mut self.weight];
        if let Some(bias) = &mut self.bias {
            result.push(bias);
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct LayerNorm {
    gamma: Parameter,
    beta: Parameter,
}

impl LayerNorm {
    pub fn new(channels: usize) -> Self {
        Self {
            gamma: Parameter::filled(channels, 1.0),
            beta: Parameter::filled(channels, 0.0),
        }
    }

    pub fn forward<E: Engine>(&self, en: &E, input: &[f64], rows: usize) -> Vec<f64> {
        en.layer_norm(
            input,
            &self.gamma.values,
            &self.beta.values,
            rows,
            self.gamma.values.len(),
            1e-5,
        )
    }

    pub fn backward<E: Engine>(
        &mut self,
        en: &E,
        input: &[f64],
        d_output: &[f64],
        rows: usize,
    ) -> Vec<f64> {
        let (d_input, d_gamma, d_beta) = en.layer_norm_backward(
            input,
            &self.gamma.values,
            d_output,
            rows,
            self.gamma.values.len(),
            1e-5,
        );
        self.gamma.gradients = d_gamma;
        self.beta.gradients = d_beta;
        d_input
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.gamma, &self.beta]
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.gamma, &mut self.beta]
    }
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct Head {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub size: usize,
    pub dropout: f64,
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct HeadCache {
    pub input: Vec<f64>,
    pub query: Vec<f64>,
    pub key: Vec<f64>,
    pub value: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub mask: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Head {
    pub fn new(channels: usize, size: usize, dropout: f64, rng: &mut impl Rng) -> Self {
        Self {
            query: Linear::new(channels, size, false, rng),
            key: Linear::new(channels, size, false, rng),
            value: Linear::new(channels, size, false, rng),
            size,
            dropout,
        }
    }

    pub fn forward<E: Engine>(
        &self,
        en: &E,
        input: &[f64],
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> (Vec<f64>, HeadCache) {
        let query = self.query.forward(en, input, batch * time);
        let key = self.key.forward(en, input, batch * time);
        let value = self.value.forward(en, input, batch * time);
        let mut probabilities = Vec::new();
        for b in 0..batch {
            let range = b * time * self.size..(b + 1) * time * self.size;
            let scores = en.matmul(
                &query[range.clone()],
                &en.transpose(&key[range], time, self.size),
                time,
                self.size,
                time,
            );
            let scaled = en.scale(&scores, 1.0 / (self.size as f64).sqrt());
            probabilities.extend(en.softmax_rows(&en.causal_mask(&scaled, time), time, time));
        }
        let mask = dropout_mask(probabilities.len(), self.dropout, training, rng);
        let weights = en.dropout(&probabilities, &mask);
        let mut output = Vec::new();
        for b in 0..batch {
            output.extend(en.matmul(
                &weights[b * time * time..(b + 1) * time * time],
                &value[b * time * self.size..(b + 1) * time * self.size],
                time,
                time,
                self.size,
            ));
        }
        (
            output,
            HeadCache {
                input: input.to_vec(),
                query,
                key,
                value,
                probabilities,
                mask,
                weights,
            },
        )
    }

    pub fn backward<E: Engine>(
        &mut self,
        en: &E,
        cache: &HeadCache,
        d_output: &[f64],
        batch: usize,
        time: usize,
    ) -> Vec<f64> {
        let mut d_query = Vec::new();
        let mut d_key = Vec::new();
        let mut d_value = Vec::new();
        for b in 0..batch {
            let vectors = b * time * self.size..(b + 1) * time * self.size;
            let matrix = b * time * time..(b + 1) * time * time;
            let (d_weights, dv) = en.matmul_backward(
                &cache.weights[matrix.clone()],
                &cache.value[vectors.clone()],
                &d_output[vectors.clone()],
                time,
                time,
                self.size,
            );
            d_value.extend(dv);
            let d_probabilities = en.dropout(&d_weights, &cache.mask[matrix.clone()]);
            let d_scores = en.softmax_rows_backward(
                &cache.probabilities[matrix],
                &d_probabilities,
                time,
                time,
            );
            let d_scaled = en.scale(
                &en.causal_mask_backward(&d_scores, time),
                1.0 / (self.size as f64).sqrt(),
            );
            let (dq, dk_transposed) = en.matmul_backward(
                &cache.query[vectors.clone()],
                &en.transpose(&cache.key[vectors], time, self.size),
                &d_scaled,
                time,
                self.size,
                time,
            );
            d_query.extend(dq);
            d_key.extend(en.transpose(&dk_transposed, self.size, time));
        }
        let dq = self
            .query
            .backward(en, &cache.input, &d_query, batch * time);
        let dk = self.key.backward(en, &cache.input, &d_key, batch * time);
        let dv = self
            .value
            .backward(en, &cache.input, &d_value, batch * time);
        en.add(&en.add(&dq, &dk), &dv)
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        let mut result = self.query.parameters();
        result.extend(self.key.parameters());
        result.extend(self.value.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut result = self.query.parameters_mut();
        result.extend(self.key.parameters_mut());
        result.extend(self.value.parameters_mut());
        result
    }
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct Attention {
    pub heads: Vec<Head>,
    pub projection: Linear,
    pub dropout: f64,
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct AttentionCache {
    pub heads: Vec<HeadCache>,
    pub concatenated: Vec<f64>,
    pub mask: Vec<f64>,
}

impl Attention {
    pub fn new(config: &ModelConfig, rng: &mut impl Rng) -> Self {
        Self {
            heads: (0..config.n_head)
                .map(|_| {
                    Head::new(
                        config.n_embd,
                        config.n_embd / config.n_head,
                        config.dropout,
                        rng,
                    )
                })
                .collect(),
            projection: Linear::new(config.n_embd, config.n_embd, true, rng),
            dropout: config.dropout,
        }
    }

    pub fn forward<E: Engine>(
        &self,
        en: &E,
        input: &[f64],
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> (Vec<f64>, AttentionCache) {
        let mut outputs = Vec::new();
        let mut caches = Vec::new();
        for head in &self.heads {
            let (output, cache) = head.forward(en, input, batch, time, training, rng);
            outputs.push(output);
            caches.push(cache);
        }
        let head_size = self.heads[0].size;
        let mut concatenated = outputs[0].clone();
        for (index, output) in outputs.iter().enumerate().skip(1) {
            concatenated = en.concat_columns(
                &concatenated,
                output,
                batch * time,
                index * head_size,
                head_size,
            );
        }
        let projected = self.projection.forward(en, &concatenated, batch * time);
        let mask = dropout_mask(projected.len(), self.dropout, training, rng);
        (
            en.dropout(&projected, &mask),
            AttentionCache {
                heads: caches,
                concatenated,
                mask,
            },
        )
    }

    pub fn backward<E: Engine>(
        &mut self,
        en: &E,
        cache: &AttentionCache,
        d_output: &[f64],
        batch: usize,
        time: usize,
    ) -> Vec<f64> {
        let d_projection = en.dropout(d_output, &cache.mask);
        let d_concat =
            self.projection
                .backward(en, &cache.concatenated, &d_projection, batch * time);
        let channels = self.projection.input_size;
        let mut d_input = vec![0.0; batch * time * channels];
        for (index, head) in self.heads.iter_mut().enumerate() {
            let d_head = en.slice_columns(
                &d_concat,
                batch * time,
                channels,
                index * head.size,
                head.size,
            );
            let dx = head.backward(en, &cache.heads[index], &d_head, batch, time);
            d_input = en.add(&d_input, &dx);
        }
        d_input
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        let mut result = Vec::new();
        for head in &self.heads {
            result.extend(head.parameters());
        }
        result.extend(self.projection.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut result = Vec::new();
        for head in &mut self.heads {
            result.extend(head.parameters_mut());
        }
        result.extend(self.projection.parameters_mut());
        result
    }
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct FeedForward {
    pub expand: Linear,
    pub project: Linear,
    pub dropout: f64,
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct FeedForwardCache {
    pub input: Vec<f64>,
    pub pre_relu: Vec<f64>,
    pub activated: Vec<f64>,
    pub mask: Vec<f64>,
}

impl FeedForward {
    pub fn new(channels: usize, dropout: f64, rng: &mut impl Rng) -> Self {
        Self {
            expand: Linear::new(channels, 4 * channels, true, rng),
            project: Linear::new(4 * channels, channels, true, rng),
            dropout,
        }
    }

    pub fn forward<E: Engine>(
        &self,
        en: &E,
        input: &[f64],
        rows: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> (Vec<f64>, FeedForwardCache) {
        let pre_relu = self.expand.forward(en, input, rows);
        let activated = en.relu(&pre_relu);
        let projected = self.project.forward(en, &activated, rows);
        let mask = dropout_mask(projected.len(), self.dropout, training, rng);
        (
            en.dropout(&projected, &mask),
            FeedForwardCache {
                input: input.to_vec(),
                pre_relu,
                activated,
                mask,
            },
        )
    }

    pub fn backward<E: Engine>(
        &mut self,
        en: &E,
        cache: &FeedForwardCache,
        d_output: &[f64],
        rows: usize,
    ) -> Vec<f64> {
        let d_projected = en.dropout(d_output, &cache.mask);
        let d_activated = self
            .project
            .backward(en, &cache.activated, &d_projected, rows);
        let d_pre_relu = en.relu_backward(&cache.pre_relu, &d_activated);
        self.expand.backward(en, &cache.input, &d_pre_relu, rows)
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        let mut result = self.expand.parameters();
        result.extend(self.project.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut result = self.expand.parameters_mut();
        result.extend(self.project.parameters_mut());
        result
    }
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct Block {
    pub norm1: LayerNorm,
    pub attention: Attention,
    pub norm2: LayerNorm,
    pub feed_forward: FeedForward,
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct BlockCache {
    pub input: Vec<f64>,
    pub after_attention: Vec<f64>,
    pub attention: AttentionCache,
    pub feed_forward: FeedForwardCache,
}

impl Block {
    pub fn new(config: &ModelConfig, rng: &mut impl Rng) -> Self {
        Self {
            norm1: LayerNorm::new(config.n_embd),
            attention: Attention::new(config, rng),
            norm2: LayerNorm::new(config.n_embd),
            feed_forward: FeedForward::new(config.n_embd, config.dropout, rng),
        }
    }

    pub fn forward<E: Engine>(
        &self,
        en: &E,
        input: &[f64],
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> (Vec<f64>, BlockCache) {
        let normalized = self.norm1.forward(en, input, batch * time);
        let (attended, attention) =
            self.attention
                .forward(en, &normalized, batch, time, training, rng);
        let after_attention = en.add(input, &attended);
        let normalized = self.norm2.forward(en, &after_attention, batch * time);
        let (fed, feed_forward) =
            self.feed_forward
                .forward(en, &normalized, batch * time, training, rng);
        (
            en.add(&after_attention, &fed),
            BlockCache {
                input: input.to_vec(),
                after_attention,
                attention,
                feed_forward,
            },
        )
    }

    pub fn backward<E: Engine>(
        &mut self,
        en: &E,
        cache: &BlockCache,
        d_output: &[f64],
        batch: usize,
        time: usize,
    ) -> Vec<f64> {
        let d_norm2 = self
            .feed_forward
            .backward(en, &cache.feed_forward, d_output, batch * time);
        let d_residual2 = self
            .norm2
            .backward(en, &cache.after_attention, &d_norm2, batch * time);
        let d_after_attention = en.add(d_output, &d_residual2);
        let d_norm1 =
            self.attention
                .backward(en, &cache.attention, &d_after_attention, batch, time);
        let d_residual1 = self
            .norm1
            .backward(en, &cache.input, &d_norm1, batch * time);
        en.add(&d_after_attention, &d_residual1)
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        let mut result = self.norm1.parameters();
        result.extend(self.attention.parameters());
        result.extend(self.norm2.parameters());
        result.extend(self.feed_forward.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut result = self.norm1.parameters_mut();
        result.extend(self.attention.parameters_mut());
        result.extend(self.norm2.parameters_mut());
        result.extend(self.feed_forward.parameters_mut());
        result
    }
}

/// Returns an inverted-dropout mask: zero for dropped entries, 1/(1-p) otherwise.
/// Evaluation, zero dropout, and empty masks do not consume randomness.
fn dropout_mask(len: usize, probability: f64, training: bool, rng: &mut impl Rng) -> Vec<f64> {
    assert!((0.0..1.0).contains(&probability));
    if !training || probability == 0.0 {
        return vec![1.0; len];
    }
    (0..len)
        .map(|_| {
            if rng.random_bool(probability) {
                0.0
            } else {
                1.0 / (1.0 - probability)
            }
        })
        .collect()
}
