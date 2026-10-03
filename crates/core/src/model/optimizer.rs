use alloc::{format, string::String, vec::Vec};
use rand::{Rng, RngExt as _};

use crate::{
    config::ModelConfig,
    engine::{AdamWConfig, AdamWGroup, Engine},
    model::{Parameter, gpt::Gpt},
    tokenizer::Tokenizer,
};

pub struct AdamW<EN: Engine, R: Reporter> {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub batch_size: usize,
    pub config: ModelConfig,
    pub iters: usize,
    pub report_loss_per_step: bool,
    pub step: i32,
    pub first_moment: Vec<EN::Buffer>,
    pub second_moment: Vec<EN::Buffer>,
    pub train: Vec<usize>,
    pub validation: Vec<usize>,
    pub reporter: R,
}

impl<EN: Engine> AdamW<EN, BasicReporter> {
    pub fn new(
        en: &EN,
        tokenizer: &Tokenizer,
        input: &str,
        model: &mut Gpt<EN>,
        config: ModelConfig,
        report_loss_per_step: bool,
    ) -> anyhow::Result<Self> {
        let tokens = tokenizer.encode(input);
        let split = tokens.len() / 10 * 9 + tokens.len() % 10 * 9 / 10;
        let train = tokens[..split].to_vec();
        let validation = tokens[split..].to_vec();
        let learning_rate = 0.0003;
        let weight_decay = 0.01;
        let batch_size = 4;
        let step = 0;
        let iters = 1000;
        let reporter = BasicReporter;

        let first_moment = model
            .parameters()
            .iter()
            .map(|p| en.zeroes(en.buffer_len(&p.values)))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let second_moment = model
            .parameters()
            .iter()
            .map(|p| en.zeroes(en.buffer_len(&p.values)))
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            learning_rate,
            weight_decay,
            config,
            batch_size,
            step,
            iters,
            report_loss_per_step,
            train,
            validation,
            first_moment,
            second_moment,
            reporter,
        })
    }
}

impl<EN: Engine, R: Reporter> AdamW<EN, R> {
    pub fn with_reporter<RR: Reporter>(self, reporter: RR) -> AdamW<EN, RR> {
        let Self {
            learning_rate,
            weight_decay,
            batch_size,
            config,
            iters,
            report_loss_per_step,
            step,
            first_moment,
            second_moment,
            train,
            validation,
            ..
        } = self;

        AdamW {
            learning_rate,
            weight_decay,
            batch_size,
            config,
            iters,
            report_loss_per_step,
            step,
            first_moment,
            second_moment,
            train,
            validation,
            reporter,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_max_iters(mut self, iters: usize) -> Self {
        self.iters = iters;
        self
    }

    pub fn with_tiny_preset(self) -> Self {
        self.with_batch_size(4).with_max_iters(1_000)
    }

    pub fn with_small_preset(self) -> Self {
        self.with_batch_size(8).with_max_iters(2_000)
    }

    pub fn with_medium_preset(self) -> Self {
        self.with_batch_size(8).with_max_iters(5_000)
    }

    pub fn with_large_preset(self) -> Self {
        self.with_batch_size(4).with_max_iters(5_000)
    }

    pub fn step(&mut self, en: &EN, parameters: &mut [&mut Parameter<EN>]) -> anyhow::Result<()> {
        anyhow::ensure!(
            parameters.len() == self.first_moment.len()
                && parameters.len() == self.second_moment.len(),
            "optimizer parameter count mismatch"
        );

        anyhow::ensure!(
            self.learning_rate.is_finite()
                && self.learning_rate >= 0.0
                && self.weight_decay.is_finite()
                && self.weight_decay >= 0.0,
            "invalid optimizer configuration"
        );

        for (index, parameter) in parameters.iter().enumerate() {
            let len = en.buffer_len(&parameter.values);

            anyhow::ensure!(
                en.buffer_len(&parameter.gradients) == len
                    && en.buffer_len(&self.first_moment[index]) == len
                    && en.buffer_len(&self.second_moment[index]) == len,
                "optimizer parameter shape mismatch"
            );
        }

        let buffers: Vec<_> = parameters
            .iter()
            .flat_map(|p| [&p.values, &p.gradients])
            .collect();

        anyhow::ensure!(
            en.buffers_all_finite(&buffers)?,
            "non-finite parameter or gradient; reduce the learning rate"
        );

        self.step = self.step.saturating_add(1);

        let mut groups: Vec<_> = parameters
            .iter_mut()
            .zip(&mut self.first_moment)
            .zip(&mut self.second_moment)
            .map(|((p, first_moment), second_moment)| AdamWGroup {
                values: &mut p.values,
                gradients: &p.gradients,
                first_moment,
                second_moment,
            })
            .collect();

        en.adamw_step(
            &mut groups,
            AdamWConfig {
                learning_rate: self.learning_rate,
                weight_decay: self.weight_decay,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                step: self.step,
            },
        )?;

        Ok(())
    }

    pub fn train(
        &mut self,
        en: &EN,
        model: &mut Gpt<EN>,
        rng: &mut impl Rng,
    ) -> anyhow::Result<()> {
        let mut ws = EN::Workspace::default();
        let step_loss = (self.iters / 4).max(1);
        let mut step_loss_count = 0;

        for step in 1..=self.iters {
            let (inputs, targets) = self.get_batch(&self.train, rng);

            model.loss_and_backward_with_workspace(
                en,
                &inputs,
                &targets,
                self.batch_size,
                self.config.block_size,
                rng,
                &mut ws,
            )?;

            self.step(en, &mut model.parameters_mut())?;

            if step < self.iters {
                if self.report_loss_per_step || step_loss_count == step_loss {
                    self.report_loss(en, model, step, rng)?;
                    step_loss_count = 0;
                } else {
                    self.reporter.report(format!("step {step}"));
                }
            } else {
                self.reporter.report(format!("step {step}"));
            }

            step_loss_count += 1;
        }

        self.report_loss(en, model, self.iters, rng)?;

        Ok(())
    }

    fn get_batch(&self, data: &[usize], rng: &mut impl Rng) -> (Vec<usize>, Vec<usize>) {
        debug_assert!(self.batch_size > 0 && data.len() > self.config.block_size);

        let size = self.batch_size.saturating_mul(self.config.block_size);

        let mut inputs = Vec::with_capacity(size);
        let mut targets = Vec::with_capacity(size);

        for _ in 0..self.batch_size {
            let start = rng.random_range(0..data.len() - self.config.block_size);

            inputs.extend_from_slice(&data[start..start + self.config.block_size]);
            targets.extend_from_slice(&data[start + 1..start + self.config.block_size + 1]);
        }

        (inputs, targets)
    }

    fn estimate_loss(
        &self,
        en: &EN,
        model: &Gpt<EN>,
        data: &[usize],
        rng: &mut impl Rng,
    ) -> anyhow::Result<f32> {
        let mut total = 0.0;
        let eval_iters = 10; // arbitrary
        let mut ws = EN::Workspace::default();

        for _ in 0..eval_iters {
            let (inputs, targets) = self.get_batch(data, rng);
            let logits = model.forward_inference(
                en,
                &inputs,
                self.batch_size,
                self.config.block_size,
                &mut ws,
                rng,
            )?;
            let targets = en.indices_from_slice(&targets)?;

            let loss = en.cross_entropy(
                &logits,
                &targets,
                self.batch_size * self.config.block_size,
                model.vocab_size,
            )?;

            debug_assert!(
                loss.is_finite(),
                "non-finite evaluation loss; reduce the learning rate"
            );

            total += loss / eval_iters as f32;
        }

        Ok(total)
    }

    fn report_loss(
        &self,
        en: &EN,
        model: &mut Gpt<EN>,
        step: usize,
        rng: &mut impl Rng,
    ) -> anyhow::Result<()> {
        let train = self.estimate_loss(en, model, &self.train, rng)?;
        let validation = self.estimate_loss(en, model, &self.validation, rng)?;
        let report = format!("step {step}: train loss {train:.4}, val loss {validation:.4}");

        self.reporter.report(report);

        model.train_loss = train;
        model.validation_loss = validation;

        Ok(())
    }
}

pub trait Reporter {
    fn report(&self, text: String);
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BasicReporter;

impl Reporter for BasicReporter {
    fn report(&self, _text: String) {
        #[cfg(feature = "std")]
        eprintln!("{_text}");
    }
}
