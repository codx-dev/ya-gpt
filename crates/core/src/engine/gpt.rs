use msgpacker::MsgPacker;
use rand::Rng;
use rand_distr::{Distribution as _, weighted::WeightedIndex};

use crate::{
    config::ModelConfig,
    data::tokenizer::Tokenizer,
    engine::{
        Engine,
        types::{Block, BlockCache, LayerNorm, Linear, Parameter},
    },
};

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct Gpt {
    pub config: ModelConfig,
    pub vocab_size: usize,
    pub tokenizer: Tokenizer,
    pub token_embedding: Parameter,
    pub position_embedding: Parameter,
    pub blocks: Vec<Block>,
    pub final_norm: LayerNorm,
    pub language_head: Linear,
}

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct GptCache {
    pub tokens: Vec<usize>,
    pub batch: usize,
    pub time: usize,
    pub blocks: Vec<BlockCache>,
    pub before_final_norm: Vec<f64>,
    pub normalized: Vec<f64>,
}

impl Gpt {
    pub fn new(
        config: ModelConfig,
        tokenizer: Tokenizer,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Self> {
        let vocab_size = tokenizer.chars().len();
        let config = config.validate(Some(vocab_size))?;
        let n_embd = config.n_embd;
        let n_layer = config.n_layer;
        let block_size = config.block_size;
        let blocks = (0..n_layer).map(|_| Block::new(&config, rng)).collect();

        Ok(Self {
            tokenizer,
            vocab_size,
            token_embedding: Parameter::normal(vocab_size * n_embd, rng),
            position_embedding: Parameter::normal(block_size * n_embd, rng),
            final_norm: LayerNorm::new(n_embd),
            language_head: Linear::new(n_embd, vocab_size, true, rng),
            blocks,
            config,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn forward_with_cache<E: Engine>(
        &self,
        en: &E,
        tokens: &[usize],
        batch: usize,
        time: usize,
        training: bool,
        rng: &mut impl Rng,
    ) -> (Vec<f64>, GptCache) {
        debug_assert!(batch > 0 && time > 0 && time <= self.config.block_size);
        debug_assert_eq!(
            tokens.len(),
            batch.checked_mul(time).expect("batch shape overflow")
        );

        let channels = self.config.n_embd;
        let token_values = en.embedding(
            &self.token_embedding.values,
            tokens,
            self.vocab_size,
            channels,
        );
        let positions: Vec<usize> = (0..batch).flat_map(|_| 0..time).collect();
        let position_values = en.embedding(
            &self.position_embedding.values,
            &positions,
            self.config.block_size,
            channels,
        );
        let mut x = en.add(&token_values, &position_values);
        let mut blocks = Vec::new();
        for block in &self.blocks {
            let (output, cache) = block.forward(en, &x, batch, time, training, rng);
            x = output;
            blocks.push(cache);
        }
        let normalized = self.final_norm.forward(en, &x, batch * time);
        let logits = self.language_head.forward(en, &normalized, batch * time);
        (
            logits,
            GptCache {
                tokens: tokens.to_vec(),
                batch,
                time,
                blocks,
                before_final_norm: x,
                normalized,
            },
        )
    }

    /// Evaluation forward pass; returns flat [batch, time, vocabulary] logits.
    pub fn forward<E: Engine>(
        &self,
        en: &E,
        tokens: &[usize],
        batch: usize,
        time: usize,
    ) -> Vec<f64> {
        self.forward_with_cache(en, tokens, batch, time, false, &mut rand::rng())
            .0
    }

    /// Overwrites every parameter gradient using this forward pass's activations.
    pub fn backward<E: Engine>(&mut self, en: &E, cache: GptCache, d_logits: &[f64]) {
        let rows = cache.batch * cache.time;
        let d_normalized = self
            .language_head
            .backward(en, &cache.normalized, d_logits, rows);
        let mut dx = self
            .final_norm
            .backward(en, &cache.before_final_norm, &d_normalized, rows);
        for (block, saved) in self.blocks.iter_mut().zip(&cache.blocks).rev() {
            dx = block.backward(en, saved, &dx, cache.batch, cache.time);
        }
        self.token_embedding.gradients =
            en.embedding_backward(&cache.tokens, &dx, self.vocab_size, self.config.n_embd);
        let positions: Vec<usize> = (0..cache.batch).flat_map(|_| 0..cache.time).collect();
        self.position_embedding.gradients =
            en.embedding_backward(&positions, &dx, self.config.block_size, self.config.n_embd);
    }

    /// A training forward/backward pass; the optimizer is a separate explicit step.
    pub fn loss_and_backward<E: Engine>(
        &mut self,
        en: &E,
        tokens: &[usize],
        targets: &[usize],
        batch: usize,
        time: usize,
        rng: &mut impl Rng,
    ) -> anyhow::Result<f64> {
        let (logits, cache) = self.forward_with_cache(en, tokens, batch, time, true, rng);
        anyhow::ensure!(
            logits.iter().all(|x| x.is_finite()),
            "non-finite logits; reduce the learning rate"
        );
        let loss = en.cross_entropy(&logits, targets, batch * time, self.vocab_size);
        anyhow::ensure!(
            loss.is_finite(),
            "non-finite training loss; reduce the learning rate"
        );
        let d_logits = en.cross_entropy_backward(&logits, targets, batch * time, self.vocab_size);
        self.backward(en, cache, &d_logits);
        Ok(loss)
    }

    pub fn generate<E: Engine>(
        &self,
        en: &E,
        prompt: &[usize],
        new_tokens: usize,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        debug_assert!(prompt.iter().all(|&token| token < self.vocab_size));
        let mut tokens = if prompt.is_empty() {
            vec![0]
        } else {
            prompt.to_vec()
        };
        for _ in 0..new_tokens {
            let start = tokens.len().saturating_sub(self.config.block_size);
            let context = &tokens[start..];
            let logits = self.forward(en, context, 1, context.len());
            let last = &logits[logits.len() - self.vocab_size..];
            let probabilities = en.softmax(last);
            let distribution = WeightedIndex::new(&probabilities)
                .expect("softmax must produce valid token probabilities");
            tokens.push(distribution.sample(rng));
        }
        tokens
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        let mut result = vec![&self.token_embedding, &self.position_embedding];
        for block in &self.blocks {
            result.extend(block.parameters());
        }
        result.extend(self.final_norm.parameters());
        result.extend(self.language_head.parameters());
        result
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut result = vec![&mut self.token_embedding, &mut self.position_embedding];
        for block in &mut self.blocks {
            result.extend(block.parameters_mut());
        }
        result.extend(self.final_norm.parameters_mut());
        result.extend(self.language_head.parameters_mut());
        result
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters()
            .iter()
            .map(|parameter| parameter.values.len())
            .sum()
    }
}
