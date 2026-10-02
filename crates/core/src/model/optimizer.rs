use rand::{Rng, RngExt as _};

use crate::{
    config::ModelConfig,
    engine::Engine,
    model::{gpt::Gpt, types::Parameter},
    tokenizer::Tokenizer,
};

pub struct AdamW<EN: Engine> {
    learning_rate: f64,
    weight_decay: f64,
    batch_size: usize,
    config: ModelConfig,
    iters: usize,
    report_loss_per_step: bool,
    step: i32,
    first_moment: Vec<EN::Buffer>,
    second_moment: Vec<EN::Buffer>,
    train: Vec<usize>,
    validation: Vec<usize>,
}

impl<EN: Engine> AdamW<EN> {
    /// Parameter order must stay the same for every call to step.
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
        })
    }

    pub fn train(
        &mut self,
        en: &EN,
        model: &mut Gpt<EN>,
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

            self.step(en, &mut model.parameters_mut())?;

            if self.report_loss_per_step && step < self.iters {
                self.report_loss(en, model, step, rng)?;
            } else {
                eprintln!("step {step}");
            }
        }

        self.report_loss(en, model, self.iters, rng)?;

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

    // Validation failures leave all state unchanged. Once updates begin, an Engine
    // error must abort training; weights and moments are not rolled back.
    fn step(&mut self, en: &EN, parameters: &mut [&mut Parameter<EN>]) -> anyhow::Result<()> {
        debug_assert_eq!(parameters.len(), self.first_moment.len());
        debug_assert_eq!(parameters.len(), self.second_moment.len());

        for (index, parameter) in parameters.iter().enumerate() {
            let len = en.buffer_len(&parameter.values);
            debug_assert_eq!(len, en.buffer_len(&parameter.gradients));
            debug_assert_eq!(len, en.buffer_len(&self.first_moment[index]));
            debug_assert_eq!(len, en.buffer_len(&self.second_moment[index]));

            anyhow::ensure!(
                en.buffer_all_finite(&parameter.values)?
                    && en.buffer_all_finite(&parameter.gradients)?,
                "non-finite parameter or gradient; reduce the learning rate"
            );
        }

        self.step = self.step.saturating_add(1);

        for (index, parameter) in parameters.iter_mut().enumerate() {
            en.adamw_in_place(
                &mut parameter.values,
                &parameter.gradients,
                &mut self.first_moment[index],
                &mut self.second_moment[index],
                self.learning_rate,
                self.weight_decay,
                0.9,
                0.999,
                1e-8,
                self.step,
            )?;
        }

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
    ) -> anyhow::Result<f64> {
        let mut total = 0.0;
        let eval_iters = 10; // arbitrary

        for _ in 0..eval_iters {
            let (inputs, targets) = self.get_batch(data, rng);
            let logits = model.forward(en, &inputs, self.batch_size, self.config.block_size)?;
            let logits_host = en.buffer_to_vec(&logits)?;

            debug_assert!(
                logits_host.iter().all(|x| x.is_finite()),
                "non-finite evaluation logits; reduce the learning rate"
            );

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

            total += loss / eval_iters as f64;
        }

        Ok(total)
    }

    fn report_loss(
        &self,
        en: &EN,
        model: &Gpt<EN>,
        step: usize,
        rng: &mut impl Rng,
    ) -> anyhow::Result<()> {
        let train = self.estimate_loss(en, model, &self.train, rng)?;
        let validation = self.estimate_loss(en, model, &self.validation, rng)?;

        eprintln!("step {step}: train loss {train:.4}, val loss {validation:.4}");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::naive::Naive;
    use rand::{SeedableRng, rngs::StdRng};

    fn fixture() -> (Gpt<Naive>, AdamW<Naive>) {
        let config = ModelConfig::micro();
        let tokenizer = Tokenizer::new("ab");
        let mut model = Gpt::new(
            &Naive,
            config.clone(),
            tokenizer.clone(),
            &mut StdRng::seed_from_u64(42),
        )
        .unwrap();
        let optimizer = AdamW::new(
            &Naive,
            &tokenizer,
            &"ab".repeat(64),
            &mut model,
            config,
            false,
        )
        .unwrap()
        .with_learning_rate(0.1)
        .with_weight_decay(0.2);

        for parameter in model.parameters_mut() {
            parameter.values.fill(1.0);
            parameter.gradients.fill(2.0);
        }
        (model, optimizer)
    }

    #[test]
    fn invalid_last_gradient_leaves_all_weights_moments_and_step_unchanged() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let (mut model, mut optimizer) = fixture();
            model.parameters_mut().last_mut().unwrap().gradients[0] = invalid;

            assert!(optimizer.step(&Naive, &mut model.parameters_mut()).is_err());

            assert_eq!(optimizer.step, 0);
            for parameter in model.parameters() {
                assert!(parameter.values.iter().all(|value| *value == 1.0));
            }
            for moment in optimizer
                .first_moment
                .iter()
                .chain(&optimizer.second_moment)
            {
                assert!(moment.iter().all(|value| *value == 0.0));
            }
        }
    }

    #[test]
    fn step_updates_weights_and_moments_without_replacing_buffers() {
        let (mut model, mut optimizer) = fixture();
        let pointers: Vec<_> = model
            .parameters()
            .iter()
            .map(|parameter| parameter.values.as_ptr())
            .collect();
        let first_pointers: Vec<_> = optimizer.first_moment.iter().map(|m| m.as_ptr()).collect();
        let second_pointers: Vec<_> = optimizer.second_moment.iter().map(|m| m.as_ptr()).collect();

        optimizer.step(&Naive, &mut model.parameters_mut()).unwrap();

        assert_eq!(optimizer.step, 1);
        let expected = 0.98 - 0.1 * 2.0 / (2.0 + 1e-8);
        for (index, parameter) in model.parameters().iter().enumerate() {
            assert_eq!(parameter.values.as_ptr(), pointers[index]);
            assert_eq!(
                optimizer.first_moment[index].as_ptr(),
                first_pointers[index]
            );
            assert_eq!(
                optimizer.second_moment[index].as_ptr(),
                second_pointers[index]
            );
            assert_eq!(parameter.values.len(), optimizer.first_moment[index].len());
            assert_eq!(parameter.values.len(), optimizer.second_moment[index].len());
            assert!(
                parameter
                    .values
                    .iter()
                    .all(|value| (value - expected).abs() < 1e-12)
            );
            assert!(parameter.gradients.iter().all(|gradient| *gradient == 2.0));
            assert!(
                optimizer.first_moment[index]
                    .iter()
                    .all(|m| (m - 0.2).abs() < 1e-12)
            );
            assert!(
                optimizer.second_moment[index]
                    .iter()
                    .all(|v| (v - 0.004).abs() < 1e-12)
            );
        }
    }

    #[test]
    fn step_counter_still_saturates() {
        let (mut model, mut optimizer) = fixture();
        optimizer.step = i32::MAX;

        optimizer.step(&Naive, &mut model.parameters_mut()).unwrap();

        assert_eq!(optimizer.step, i32::MAX);
        let expected = 0.98 - 0.1 * 0.2 / (0.004_f64.sqrt() + 1e-8);
        for parameter in model.parameters() {
            assert!(
                parameter
                    .values
                    .iter()
                    .all(|value| (value - expected).abs() < 1e-12)
            );
        }
    }
}
