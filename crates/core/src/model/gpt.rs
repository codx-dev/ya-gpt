use std::mem;

use anyhow::Context as _;
use rand::{Rng, RngExt as _};

use crate::{
    config::ModelConfig,
    engine::{BlockSpec, EmbeddingSpec, Engine, ForwardMode, NormSpec},
    model::{Block, LayerNorm, Linear, Parameter},
    tokenizer::Tokenizer,
};

pub struct Gpt<EN: Engine> {
    pub config: ModelConfig,
    pub vocab_size: usize,
    pub tokenizer: Tokenizer,
    pub token_embedding: Parameter<EN>,
    pub position_embedding: Parameter<EN>,
    pub blocks: Vec<Block<EN>>,
    pub final_norm: LayerNorm<EN>,
    pub language_head: Linear<EN>,
}

pub struct GptCache<EN: Engine> {
    tokens: EN::IndexBuffer,
    batch: usize,
    time: usize,
    blocks: Vec<EN::BlockCache>,
    final_norm: EN::LayerNormCache,
    normalized: EN::Buffer,
}

impl<EN: Engine> Gpt<EN> {
    pub fn new(
        en: &EN,
        config: ModelConfig,
        tokenizer: Tokenizer,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Self> {
        let vocab_size = tokenizer.chars().len();
        let config = config.validate(Some(vocab_size))?;
        let c = config.n_embd;

        let blocks = (0..config.n_layer)
            .map(|_| Block::new(en, &config, rng))
            .collect::<anyhow::Result<_>>()?;

        let token_embedding_len = vocab_size.saturating_mul(c);
        let token_embedding = Parameter::normal(en, token_embedding_len, rng)?;

        let position_embedding_len = config.block_size.saturating_mul(c);
        let position_embedding = Parameter::normal(en, position_embedding_len, rng)?;

        let final_norm = LayerNorm::new(en, c)?;
        let language_head = Linear::new(en, c, vocab_size, true, rng)?;

        Ok(Self {
            blocks,
            token_embedding,
            position_embedding,
            final_norm,
            language_head,
            config,
            vocab_size,
            tokenizer,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn block_spec(&self, batch: usize, time: usize) -> BlockSpec {
        BlockSpec {
            batch,
            time,
            channels: self.config.n_embd,
            heads: self.config.n_head,
            dropout: self.config.dropout,
        }
    }

    fn embedding_spec(&self, batch: usize, time: usize) -> EmbeddingSpec {
        EmbeddingSpec {
            batch,
            time,
            channels: self.config.n_embd,
            vocab: self.vocab_size,
            positions: self.config.block_size,
        }
    }

    fn norm_spec(&self, rows: usize) -> NormSpec {
        NormSpec {
            rows,
            channels: self.config.n_embd,
            epsilon: 1e-5,
        }
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn run_forward(
        &self,
        en: &EN,
        tokens: &[usize],
        batch: usize,
        time: usize,
        training: bool,
        save: bool,
        rng: &mut impl Rng,
        workspace: &mut EN::Workspace,
    ) -> anyhow::Result<(EN::Buffer, Option<GptCache<EN>>)> {
        let spec = self.block_spec(batch, time);
        let n = spec.elements();
        let rows = batch.saturating_mul(time);

        debug_assert!(
            time <= self.config.block_size && tokens.len() == rows,
            "invalid input shape"
        );
        debug_assert!(
            tokens.iter().all(|&t| t < self.vocab_size),
            "token outside vocabulary"
        );

        let tokens = en.indices_from_slice(tokens)?;
        let mut input = en.zeroes(n)?;
        let mut output = en.zeroes(n)?;

        en.token_position_embedding_forward(
            self.embedding_spec(batch, time),
            &tokens,
            &self.token_embedding.values,
            &self.position_embedding.values,
            &mut input,
        )?;

        let mut blocks = Vec::new();

        for block in &self.blocks {
            let mut cache = save.then(EN::BlockCache::default);
            let mode = ForwardMode {
                seed: (training && spec.dropout > 0.0).then(|| rng.random()),
            };

            en.block_forward(
                spec,
                block.weights(),
                &input,
                mode,
                workspace,
                &mut output,
                cache.as_mut(),
            )?;

            mem::swap(&mut input, &mut output);

            if let Some(cache) = cache {
                blocks.push(cache);
            }
        }

        let mut norm_cache = save.then(EN::LayerNormCache::default);

        en.layer_norm_forward(
            self.norm_spec(rows),
            self.final_norm.weights(),
            &input,
            &mut output,
            norm_cache.as_mut(),
        )?;

        let logits_len = rows.saturating_mul(self.vocab_size);
        let mut logits = en.zeroes(logits_len)?;

        en.linear_forward(
            self.language_head.spec(rows),
            self.language_head.weights(),
            &output,
            &mut logits,
        )?;

        let cache = norm_cache.map(|final_norm| GptCache {
            tokens,
            batch,
            time,
            blocks,
            final_norm,
            normalized: output,
        });

        Ok((logits, cache))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_workspace(
        &self,
        en: &EN,
        tokens: &[usize],
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
        workspace: &mut EN::Workspace,
    ) -> anyhow::Result<(EN::Buffer, GptCache<EN>)> {
        let (logits, cache) =
            self.run_forward(en, tokens, batch, time, training, true, rng, workspace)?;
        let cache = cache.context("backward cache requested")?;

        Ok((logits, cache))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache(
        &self,
        en: &EN,
        tokens: &[usize],
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<(EN::Buffer, GptCache<EN>)> {
        self.forward_with_workspace(
            en,
            tokens,
            batch,
            time,
            training,
            rng,
            &mut EN::Workspace::default(),
        )
    }

    pub fn forward_inference(
        &self,
        en: &EN,
        tokens: &[usize],
        batch: usize,
        time: usize,
        workspace: &mut EN::Workspace,
    ) -> anyhow::Result<EN::Buffer> {
        self.run_forward(
            en,
            tokens,
            batch,
            time,
            false,
            false,
            &mut rand::rng(),
            workspace,
        )
        .map(|(logits, _)| logits)
    }

    pub fn forward(
        &self,
        en: &EN,
        tokens: &[usize],
        batch: usize,
        time: usize,
    ) -> anyhow::Result<EN::Buffer> {
        self.forward_inference(en, tokens, batch, time, &mut EN::Workspace::default())
    }

    pub fn backward_with_workspace(
        &mut self,
        en: &EN,
        cache: GptCache<EN>,
        d_logits: &EN::Buffer,
        workspace: &mut EN::Workspace,
    ) -> anyhow::Result<()> {
        let spec = self.block_spec(cache.batch, cache.time);
        let n = spec.elements();
        let rows = cache.batch.saturating_mul(cache.time);
        let linear = self.language_head.spec(rows);
        let (weights, gradients) = self.language_head.parts();

        debug_assert_eq!(
            cache.blocks.len(),
            self.blocks.len(),
            "cache block count mismatch"
        );

        let mut dx = en.zeroes(n)?;
        let mut temp = en.zeroes(n)?;

        en.linear_backward(
            linear,
            weights,
            gradients,
            &cache.normalized,
            d_logits,
            &mut temp,
        )?;

        let norm = self.norm_spec(rows);
        let (weights, gradients) = self.final_norm.parts();

        en.layer_norm_backward(norm, weights, gradients, &cache.final_norm, &temp, &mut dx)?;

        for (block, saved) in self.blocks.iter_mut().zip(&cache.blocks).rev() {
            let (weights, gradients) = block.parts();

            en.block_backward(spec, weights, gradients, saved, &dx, workspace, &mut temp)?;

            mem::swap(&mut dx, &mut temp);
        }

        en.token_position_embedding_backward(
            self.embedding_spec(cache.batch, cache.time),
            &cache.tokens,
            &dx,
            &mut self.token_embedding.gradients,
            &mut self.position_embedding.gradients,
        )?;

        Ok(())
    }

    pub fn backward(
        &mut self,
        en: &EN,
        cache: GptCache<EN>,
        d_logits: &EN::Buffer,
    ) -> anyhow::Result<()> {
        self.backward_with_workspace(en, cache, d_logits, &mut EN::Workspace::default())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn loss_and_backward_with_workspace(
        &mut self,
        en: &EN,
        tokens: &[usize],
        targets: &[usize],
        batch: usize,
        time: usize,
        rng: &mut impl Rng,
        workspace: &mut EN::Workspace,
    ) -> anyhow::Result<f32> {
        debug_assert!(
            targets.len() == batch * time && targets.iter().all(|&t| t < self.vocab_size),
            "invalid targets"
        );

        let targets = en.indices_from_slice(targets)?;
        let (logits, cache) =
            self.forward_with_workspace(en, tokens, batch, time, true, rng, workspace)?;

        let gradient_len = en.buffer_len(&logits);
        let mut gradient = en.zeroes(gradient_len)?;

        let loss = en.cross_entropy_with_grad(
            &logits,
            &targets,
            batch * time,
            self.vocab_size,
            &mut gradient,
        )?;

        self.backward_with_workspace(en, cache, &gradient, workspace)?;

        Ok(loss)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn loss_and_backward(
        &mut self,
        en: &EN,
        tokens: &[usize],
        targets: &[usize],
        batch: usize,
        time: usize,
        rng: &mut impl Rng,
    ) -> anyhow::Result<f32> {
        self.loss_and_backward_with_workspace(
            en,
            tokens,
            targets,
            batch,
            time,
            rng,
            &mut EN::Workspace::default(),
        )
    }

    pub fn generate(
        &self,
        en: &EN,
        prompt: &[usize],
        new_tokens: usize,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Vec<usize>> {
        debug_assert!(
            prompt.iter().all(|&t| t < self.vocab_size),
            "token outside vocabulary"
        );

        let mut tokens = if prompt.is_empty() {
            vec![0]
        } else {
            prompt.to_vec()
        };

        let mut workspace = EN::Workspace::default();

        for _ in 0..new_tokens {
            let start = tokens.len().saturating_sub(self.config.block_size);
            let context = &tokens[start..];

            let logits = self.forward_inference(en, context, 1, context.len(), &mut workspace)?;
            let rnd = rng.random();
            let token = en.sample_last_token(&logits, context.len(), self.vocab_size, rnd)?;

            tokens.push(token);
        }
        Ok(tokens)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        let mut result = vec![&self.token_embedding, &self.position_embedding];
        for block in &self.blocks {
            result.extend(block.parameters());
        }
        result.extend(self.final_norm.parameters());
        result.extend(self.language_head.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        let mut result = vec![&mut self.token_embedding, &mut self.position_embedding];
        for block in &mut self.blocks {
            result.extend(block.parameters_mut());
        }
        result.extend(self.final_norm.parameters_mut());
        result.extend(self.language_head.parameters_mut());
        result
    }

    pub fn parameter_count(&self, en: &EN) -> usize {
        self.parameters()
            .iter()
            .map(|p| en.buffer_len(&p.values))
            .sum()
    }
}
