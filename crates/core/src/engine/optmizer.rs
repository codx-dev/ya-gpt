use msgpacker::MsgPacker;
use rand::{Rng, RngExt as _};

use crate::{
    config::ModelConfig,
    data::tokenizer::Tokenizer,
    engine::{Engine, gpt::Gpt, types::Parameter},
};

#[derive(Debug, Clone, PartialEq, MsgPacker)]
pub struct AdamW {
    learning_rate: f64,
    weight_decay: f64,
    batch_size: usize,
    config: ModelConfig,
    iters: usize,
    report_loss_per_step: bool,
    step: i32,
    first_moment: Vec<Vec<f64>>,
    second_moment: Vec<Vec<f64>>,
    train: Vec<usize>,
    validation: Vec<usize>,
}

impl AdamW {
    /// Parameter order must stay the same for every call to step.
    pub fn new(
        tokenizer: &Tokenizer,
        input: &str,
        model: &mut Gpt,
        config: ModelConfig,
        report_loss_per_step: bool,
    ) -> Self {
        let tokens = tokenizer.encode(input);
        let split = tokens.len() / 10 * 9 + tokens.len() % 10 * 9 / 10;
        let train = tokens[..split].to_vec();
        let validation = tokens[split..].to_vec();
        let first_moment = model
            .parameters()
            .iter()
            .map(|p| vec![0.0; p.values.len()])
            .collect();
        let second_moment = model
            .parameters()
            .iter()
            .map(|p| vec![0.0; p.values.len()])
            .collect();

        Self {
            learning_rate: 0.0003,
            weight_decay: 0.01,
            config,
            batch_size: 4,
            step: 0,
            iters: 1000,
            report_loss_per_step,
            train,
            validation,
            first_moment,
            second_moment,
        }
    }

    pub fn train<E: Engine>(
        &mut self,
        en: &E,
        model: &mut Gpt,
        rng: &mut impl Rng,
    ) -> anyhow::Result<()> {
        for step in 1..=self.iters {
            let (inputs, targets) = self.get_batch(&self.train, rng);

            model.loss_and_backward(
                en,
                &inputs,
                &targets,
                self.batch_size,
                self.config.block_size,
                rng,
            )?;

            self.step(&mut model.parameters_mut())?;

            if self.report_loss_per_step && step < self.iters {
                self.report_loss(en, model, step, rng);
            } else {
                eprintln!("step {step}");
            }
        }

        self.report_loss(en, model, self.iters, rng);

        Ok(())
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
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

    fn step(&mut self, parameters: &mut [&mut Parameter]) -> anyhow::Result<()> {
        debug_assert_eq!(parameters.len(), self.first_moment.len());
        for (index, parameter) in parameters.iter().enumerate() {
            debug_assert_eq!(parameter.values.len(), self.first_moment[index].len());
            debug_assert_eq!(parameter.values.len(), parameter.gradients.len());
            anyhow::ensure!(
                parameter
                    .values
                    .iter()
                    .chain(&parameter.gradients)
                    .all(|x| x.is_finite()),
                "non-finite parameter or gradient; reduce the learning rate"
            );
        }
        self.step = self
            .step
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("optimizer step overflow"))?;
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let correction1 = 1.0 - beta1.powi(self.step);
        let correction2 = 1.0 - beta2.powi(self.step);
        for (index, parameter) in parameters.iter_mut().enumerate() {
            for i in 0..parameter.values.len() {
                let gradient = parameter.gradients[i];
                let m = beta1 * self.first_moment[index][i] + (1.0 - beta1) * gradient;
                let v = beta2 * self.second_moment[index][i] + (1.0 - beta2) * gradient * gradient;
                self.first_moment[index][i] = m;
                self.second_moment[index][i] = v;
                parameter.values[i] *= 1.0 - self.learning_rate * self.weight_decay;
                parameter.values[i] -=
                    self.learning_rate * (m / correction1) / ((v / correction2).sqrt() + 1e-8);

                anyhow::ensure!(
                    parameter.values[i].is_finite() && m.is_finite() && v.is_finite(),
                    "non-finite optimizer update; reduce the learning rate"
                );
            }
        }
        Ok(())
    }

    fn get_batch(&self, data: &[usize], rng: &mut impl Rng) -> (Vec<usize>, Vec<usize>) {
        debug_assert!(self.batch_size > 0 && data.len() > self.config.block_size);
        let size = self
            .batch_size
            .checked_mul(self.config.block_size)
            .expect("batch shape overflow");
        let mut inputs = Vec::with_capacity(size);
        let mut targets = Vec::with_capacity(size);
        for _ in 0..self.batch_size {
            let start = rng.random_range(0..data.len() - self.config.block_size);
            inputs.extend_from_slice(&data[start..start + self.config.block_size]);
            targets.extend_from_slice(&data[start + 1..start + self.config.block_size + 1]);
        }
        (inputs, targets)
    }

    fn estimate_loss<E: Engine>(
        &self,
        en: &E,
        model: &Gpt,
        data: &[usize],
        rng: &mut impl Rng,
    ) -> f64 {
        let mut total = 0.0;
        let eval_iters = 10; // arbitrary

        for _ in 0..eval_iters {
            let (inputs, targets) = self.get_batch(data, rng);
            let logits = model.forward(en, &inputs, self.batch_size, self.config.block_size);

            debug_assert!(
                logits.iter().all(|x| x.is_finite()),
                "non-finite evaluation logits; reduce the learning rate"
            );

            let loss = en.cross_entropy(
                &logits,
                &targets,
                self.batch_size * self.config.block_size,
                model.vocab_size,
            );

            debug_assert!(
                loss.is_finite(),
                "non-finite evaluation loss; reduce the learning rate"
            );

            total += loss / eval_iters as f64;
        }

        total
    }

    fn report_loss<E: Engine>(&self, en: &E, model: &Gpt, step: usize, rng: &mut impl Rng) {
        let train = self.estimate_loss(en, model, &self.train, rng);
        let validation = self.estimate_loss(en, model, &self.validation, rng);

        eprintln!("step {step}: train loss {train:.4}, val loss {validation:.4}");
    }
}
