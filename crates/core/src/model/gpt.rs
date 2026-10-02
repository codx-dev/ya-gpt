use rand::Rng;
use rand_distr::{Distribution as _, weighted::WeightedIndex};

use crate::{
    config::ModelConfig,
    engine::Engine,
    model::types::{Block, BlockCache, LayerNorm, Linear, Parameter},
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
    pub tokens: Vec<usize>,
    pub batch: usize,
    pub time: usize,
    pub blocks: Vec<BlockCache<EN>>,
    pub before_final_norm: EN::Buffer,
    pub normalized: EN::Buffer,
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
        let n_embd = config.n_embd;
        let n_layer = config.n_layer;
        let block_size = config.block_size;

        let blocks = (0..n_layer)
            .map(|_| Block::new(en, &config, rng))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let token_embedding = Parameter::normal(en, vocab_size * n_embd, rng)?;
        let position_embedding = Parameter::normal(en, block_size * n_embd, rng)?;
        let final_norm = LayerNorm::new(en, n_embd)?;
        let language_head = Linear::new(en, n_embd, vocab_size, true, rng)?;

        Ok(Self {
            tokenizer,
            vocab_size,
            token_embedding,
            position_embedding,
            final_norm,
            language_head,
            blocks,
            config,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn forward_with_cache(
        &self,
        en: &EN,
        tokens: &[usize],
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<(EN::Buffer, GptCache<EN>)> {
        debug_assert!(batch > 0 && time > 0 && time <= self.config.block_size);
        debug_assert_eq!(
            tokens.len(),
            batch.checked_mul(time).expect("batch shape overflow")
        );

        let channels = self.config.n_embd;
        let tokens_channels = tokens.len().saturating_mul(channels);
        let batch_time = batch.saturating_mul(time);

        let mut token_values = en.zeroes(tokens_channels)?;
        en.embedding(
            &self.token_embedding.values,
            tokens,
            self.vocab_size,
            channels,
            &mut token_values,
        )?;

        let positions: Vec<usize> = (0..batch).flat_map(|_| 0..time).collect();
        let positions_channels = positions.len().saturating_mul(channels);

        let mut position_values = en.zeroes(positions_channels)?;
        en.embedding(
            &self.position_embedding.values,
            &positions,
            self.config.block_size,
            channels,
            &mut position_values,
        )?;
        en.add_in_place(&mut token_values, &position_values)?;

        let mut blocks = Vec::new();
        for block in &self.blocks {
            let t = en.clone(&token_values)?;
            let (output, cache) = block.forward(en, t, batch, time, training, rng)?;
            token_values = output;
            blocks.push(cache);
        }
        let normalized = self.final_norm.forward(en, &token_values, batch_time)?;
        let logits = self.language_head.forward(en, &normalized, batch_time)?;
        let cache = GptCache {
            tokens: tokens.to_vec(),
            batch,
            time,
            blocks,
            before_final_norm: token_values,
            normalized,
        };

        Ok((logits, cache))
    }

    /// Evaluation forward pass; returns flat [batch, time, vocabulary] logits.
    pub fn forward(
        &self,
        en: &EN,
        tokens: &[usize],
        batch: usize,
        time: usize,
    ) -> anyhow::Result<EN::Buffer> {
        self.forward_with_cache(en, tokens, batch, time, false, &mut rand::rng())
            .map(|(logits, _cache)| logits)
    }

    /// Overwrites every parameter gradient using this forward pass's activations.
    pub fn backward(
        &mut self,
        en: &EN,
        cache: GptCache<EN>,
        d_logits: &EN::Buffer,
    ) -> anyhow::Result<()> {
        let rows = cache.batch.saturating_mul(cache.time);
        let d_normalized = self
            .language_head
            .backward(en, &cache.normalized, d_logits, rows)?;

        let mut dx = self
            .final_norm
            .backward(en, &cache.before_final_norm, &d_normalized, rows)?;

        for (block, saved) in self.blocks.iter_mut().zip(&cache.blocks).rev() {
            dx = block.backward(en, saved, &dx, cache.batch, cache.time)?;
        }

        en.embedding_backward(
            &cache.tokens,
            &dx,
            self.vocab_size,
            self.config.n_embd,
            &mut self.token_embedding.gradients,
        )?;

        let positions: Vec<usize> = (0..cache.batch).flat_map(|_| 0..cache.time).collect();

        en.embedding_backward(
            &positions,
            &dx,
            self.config.block_size,
            self.config.n_embd,
            &mut self.position_embedding.gradients,
        )?;

        Ok(())
    }

    /// A training forward/backward pass; the optimizer is a separate explicit step.
    pub fn loss_and_backward(
        &mut self,
        en: &EN,
        tokens: &[usize],
        targets: &[usize],
        batch: usize,
        time: usize,
        rng: &mut impl Rng,
    ) -> anyhow::Result<f64> {
        let (logits, cache) = self.forward_with_cache(en, tokens, batch, time, true, rng)?;
        let logits_host = en.buffer_to_vec(&logits)?;

        anyhow::ensure!(
            logits_host.iter().all(|x| x.is_finite()),
            "non-finite logits; reduce the learning rate"
        );

        let loss = en.cross_entropy(&logits, targets, batch * time, self.vocab_size)?;

        anyhow::ensure!(
            loss.is_finite(),
            "non-finite training loss; reduce the learning rate"
        );

        let len = batch.saturating_mul(time).saturating_mul(self.vocab_size);
        let mut d_logits = en.zeroes(len)?;

        en.cross_entropy_backward(
            &logits,
            targets,
            batch * time,
            self.vocab_size,
            &mut d_logits,
        )?;
        self.backward(en, cache, &d_logits)?;

        Ok(loss)
    }

    pub fn generate(
        &self,
        en: &EN,
        prompt: &[usize],
        new_tokens: usize,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Vec<usize>> {
        debug_assert!(prompt.iter().all(|&token| token < self.vocab_size));

        let mut tokens = if prompt.is_empty() {
            vec![0]
        } else {
            prompt.to_vec()
        };

        for _ in 0..new_tokens {
            let start = tokens.len().saturating_sub(self.config.block_size);
            let context = &tokens[start..];
            let logits = self.forward(en, context, 1, context.len())?;

            let len = en.buffer_len(&logits);
            let ofs = len.saturating_sub(self.vocab_size);
            let len = len - ofs;
            let mut probabilities = en.slice(&logits, ofs, len);

            en.softmax_in_place(&mut probabilities)?;

            let probabilities = en.buffer_to_vec(&probabilities)?;
            let distribution = WeightedIndex::new(&probabilities)?;

            tokens.push(distribution.sample(rng));
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
            .map(|parameter| en.buffer_len(&parameter.values))
            .sum()
    }
}
