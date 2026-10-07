use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use ya_gpt::{
    config::ModelConfig,
    engine::{AdamWConfig, AdamWGroup, Engine},
    model::{
        gpt::Gpt,
        optimizer::{AdamW, BasicReporter},
    },
    tokenizer::Tokenizer,
};

use crate::{WebGpuBuffer as B, WebGpuEngine as E, WebGpuWorkspace, host::positive};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Phase {
    Step,
    Update,
    TrainLoss,
    ValidationLoss,
    Export,
    Done,
    Cancelled,
}

pub(crate) struct Training {
    pub model: Gpt<E>,
    optimizer: AdamW<E, BasicReporter>,
    rng: StdRng,
    workspace: WebGpuWorkspace,
    phase: Phase,
    pending: Option<B>,
    waiting: bool,
    stopped: bool,
    completed: u32,
    report_count: usize,
    eval_count: usize,
    eval_loss: f32,
    export_index: usize,
    export_offset: usize,
    export_count: usize,
    pub bytes: Vec<u8>,
}

impl Training {
    pub fn new(text: &str, args: &[u32; 12]) -> Result<Self> {
        let config = ModelConfig {
            block_size: args[0] as usize,
            n_embd: args[1] as usize,
            n_head: args[2] as usize,
            n_layer: args[3] as usize,
            dropout: f32::from_bits(args[4]),
        };
        let batch = args[5] as usize;
        let iterations = args[6] as usize;
        let lr = f32::from_bits(args[7]);
        let wd = f32::from_bits(args[8]);
        ensure!(
            batch > 0 && iterations > 0 && iterations <= i32::MAX as usize,
            "invalid batch size or iteration count"
        );
        ensure!(
            lr.is_finite()
                && (0.0..=1.0).contains(&lr)
                && wd.is_finite()
                && (0.0..=1.0).contains(&wd),
            "invalid optimizer settings"
        );
        ensure!(args[9] <= 1, "invalid loss reporting flag");
        let tokenizer = Tokenizer::char_stream(text);
        let config = config.validate(Some(tokenizer.vocab_size()))?;
        let len = text.chars().count();
        let split = len / 10 * 9 + len % 10 * 9 / 10;
        ensure!(
            split > config.block_size && len - split > config.block_size,
            "both dataset splits must exceed context length"
        );
        positive(&[batch, config.block_size, config.n_embd, 4])?;
        positive(&[batch, config.n_head, config.block_size, config.block_size])?;
        positive(&[batch, config.block_size, tokenizer.vocab_size()])?;
        let mut rng = StdRng::seed_from_u64(u64::from(args[10]) | u64::from(args[11]) << 32);
        let mut model = Gpt::new(&E, config.clone(), tokenizer.clone(), &mut rng)?;
        model.train_loss = f32::NAN;
        model.validation_loss = f32::NAN;
        let optimizer = AdamW::new(&E, &tokenizer, text, &mut model, config, args[9] == 1)?
            .with_batch_size(batch)
            .with_max_iters(iterations)
            .with_learning_rate(lr)
            .with_weight_decay(wd);
        Ok(Self {
            model,
            optimizer,
            rng,
            workspace: WebGpuWorkspace::default(),
            phase: Phase::Step,
            pending: None,
            waiting: false,
            stopped: false,
            completed: 0,
            report_count: 0,
            eval_count: 0,
            eval_loss: 0.0,
            export_index: 0,
            export_offset: 0,
            export_count: 0,
            bytes: Vec::new(),
        })
    }

    pub fn stop(&mut self) {
        self.stopped = true;
    }

    fn batch(&mut self, validation: bool) -> (Vec<usize>, Vec<usize>) {
        let data = if validation {
            &self.optimizer.validation
        } else {
            &self.optimizer.train
        };
        let time = self.model.config.block_size;
        let n = self.optimizer.batch_size * time;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..self.optimizer.batch_size {
            let start = self.rng.random_range(0..data.len() - time);
            x.extend_from_slice(&data[start..start + time]);
            y.extend_from_slice(&data[start + 1..start + time + 1]);
        }
        (x, y)
    }

    fn gradients(&mut self) -> Result<B> {
        let (x, y) = self.batch(false);
        let (logits, cache) = self.model.forward_with_workspace(
            &E,
            &x,
            self.optimizer.batch_size,
            self.model.config.block_size,
            true,
            &mut self.rng,
            &mut self.workspace,
        )?;
        let targets = E.indices_from_slice(&y)?;
        let (gradient, loss, invalid) =
            E.loss_gpu(&logits, &targets, y.len(), self.model.vocab_size, true)?;
        self.model
            .backward_with_workspace(&E, cache, &gradient, &mut self.workspace)?;
        let parameters = self.model.parameters();
        let mut buffers: Vec<_> = parameters
            .iter()
            .flat_map(|p| [&p.values, &p.gradients])
            .collect();
        buffers.extend(self.optimizer.first_moment.iter());
        buffers.extend(self.optimizer.second_moment.iter());
        let flags = E.finite_gpu(&buffers)?;
        E.summarize(&loss, &invalid, &flags)
    }

    fn update(&mut self) -> Result<B> {
        let mut parameters = self.model.parameters_mut();
        let mut groups: Vec<_> = parameters
            .iter_mut()
            .zip(&mut self.optimizer.first_moment)
            .zip(&mut self.optimizer.second_moment)
            .map(|((p, first_moment), second_moment)| AdamWGroup {
                values: &mut p.values,
                gradients: &p.gradients,
                first_moment,
                second_moment,
            })
            .collect();
        E.adamw_gpu(
            &mut groups,
            AdamWConfig {
                learning_rate: self.optimizer.learning_rate,
                weight_decay: self.optimizer.weight_decay,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                step: self.completed as i32 + 1,
            },
        )
    }

    fn evaluate(&mut self) -> Result<B> {
        let (x, y) = self.batch(self.phase == Phase::ValidationLoss);
        let logits = self.model.forward_inference(
            &E,
            &x,
            self.optimizer.batch_size,
            self.model.config.block_size,
            &mut self.workspace,
            &mut self.rng,
        )?;
        let targets = E.indices_from_slice(&y)?;
        let (_, loss, invalid) =
            E.loss_gpu(&logits, &targets, y.len(), self.model.vocab_size, false)?;
        let flags = E.zeroes(0)?;
        E.summarize(&loss, &invalid, &flags)
    }

    pub fn advance(&mut self) -> Result<[u32; 8]> {
        ensure!(!self.waiting, "training is awaiting a readback");
        if self.phase == Phase::Step && self.stopped {
            self.phase = if self.completed == 0 {
                Phase::Cancelled
            } else {
                Phase::TrainLoss
            };
        }
        let (kind, handle, offset, count) = match self.phase {
            Phase::Step | Phase::Update | Phase::TrainLoss | Phase::ValidationLoss => {
                let kind = match self.phase {
                    Phase::Step => 1,
                    Phase::Update => 2,
                    Phase::TrainLoss => 3,
                    _ => 4,
                };
                let buffer = match self.phase {
                    Phase::Step => self.gradients()?,
                    Phase::Update => self.update()?,
                    _ => self.evaluate()?,
                };
                let descriptor = (kind, buffer.handle, 0, buffer.len as u32);
                self.pending = Some(buffer);
                self.waiting = true;
                descriptor
            }
            Phase::Export => {
                let parameters = self.model.parameters();
                if self.export_index == parameters.len() * 2 {
                    self.bytes = self.model.to_bytes(&E)?;
                    self.phase = Phase::Done;
                    (6, 0, 0, 0)
                } else {
                    let p = parameters[self.export_index / 2];
                    let b = if self.export_index.is_multiple_of(2) {
                        &p.values
                    } else {
                        &p.gradients
                    };
                    self.export_count = (b.len - self.export_offset).min(2 * 1024 * 1024);
                    self.waiting = true;
                    (
                        5,
                        b.handle,
                        (self.export_offset * 4) as u32,
                        self.export_count as u32,
                    )
                }
            }
            Phase::Done => (6, 0, 0, 0),
            Phase::Cancelled => (7, 0, 0, 0),
        };
        Ok([
            kind,
            handle,
            offset,
            count,
            self.completed,
            self.optimizer.iters as u32,
            self.model.train_loss.to_bits(),
            self.model.validation_loss.to_bits(),
        ])
    }

    pub fn accept(&mut self, values: &[f32]) -> Result<()> {
        ensure!(self.waiting, "no training readback requested");
        if self.phase == Phase::Export {
            ensure!(
                values.len() == self.export_count && values.iter().all(|v| v.is_finite()),
                "invalid model export readback"
            );
            let mut parameters = self.model.parameters_mut();
            let p = &mut parameters[self.export_index / 2];
            let buffer = if self.export_index.is_multiple_of(2) {
                &mut p.values
            } else {
                &mut p.gradients
            };
            let snapshot = buffer.snapshot.get_or_insert_with(Vec::new);
            snapshot
                .try_reserve(values.len())
                .context("model export allocation failed")?;
            snapshot.extend_from_slice(values);
            self.export_offset += values.len();
            if self.export_offset == buffer.len {
                self.export_index += 1;
                self.export_offset = 0;
            }
        } else {
            ensure!(
                values.len() == self.pending.as_ref().context("missing GPU readback")?.len,
                "invalid training readback length"
            );
            ensure!(
                values.iter().all(|v| v.is_finite()) && *values.last().unwrap() == 0.0,
                "non-finite loss, parameter, gradient or optimizer update; reduce the learning rate"
            );
            match self.phase {
                Phase::Step => self.phase = Phase::Update,
                Phase::Update => {
                    self.completed += 1;
                    self.optimizer.step = self.completed as i32;
                    let interval = (self.optimizer.iters / 4).max(1);
                    let evaluate = self.stopped
                        || self.completed as usize == self.optimizer.iters
                        || self.optimizer.report_loss_per_step
                        || self.report_count == interval;
                    if evaluate {
                        self.report_count = 0;
                        self.eval_count = 0;
                        self.eval_loss = 0.0;
                        self.phase = Phase::TrainLoss;
                    } else {
                        self.phase = Phase::Step;
                    }
                    self.report_count += 1;
                }
                Phase::TrainLoss | Phase::ValidationLoss => {
                    self.eval_loss += values[0] / 10.0;
                    self.eval_count += 1;
                    if self.eval_count == 10 {
                        if self.phase == Phase::TrainLoss {
                            self.model.train_loss = self.eval_loss;
                            self.phase = Phase::ValidationLoss;
                        } else {
                            self.model.validation_loss = self.eval_loss;
                            self.phase = if self.stopped
                                || self.completed as usize == self.optimizer.iters
                            {
                                Phase::Export
                            } else {
                                Phase::Step
                            };
                        }
                        self.eval_count = 0;
                        self.eval_loss = 0.0;
                    }
                }
                _ => anyhow::bail!("invalid training phase"),
            }
        }
        self.pending = None;
        self.waiting = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::mock;
    use ya_gpt::engine::naive::Naive;

    fn options() -> [u32; 12] {
        [
            3,
            6,
            2,
            2,
            0.2f32.to_bits(),
            2,
            4,
            0.003f32.to_bits(),
            0.01f32.to_bits(),
            0,
            42,
            0,
        ]
    }
    fn text() -> String {
        "abcé🦀abc".repeat(20)
    }

    #[test]
    fn stopped_training_finishes_update_evaluates_exports_and_cleans_up() {
        mock::reset();
        let mut t = Training::new(&text(), &options()).unwrap();
        assert!(t.accept(&[]).is_err());
        assert_eq!(t.advance().unwrap()[0], 1);
        assert!(t.advance().is_err());
        assert!(t.accept(&[1.0]).is_err());
        t.stop(); // An in-flight backward pass must still commit its update.
        t.accept(&[1.0, 0.0]).unwrap();
        assert_eq!(t.advance().unwrap()[0], 2);
        t.accept(&[0.0]).unwrap();
        for kind in [3, 4] {
            for _ in 0..10 {
                let d = t.advance().unwrap();
                assert_eq!(d[0], kind);
                assert_eq!(d[4], 1);
                t.accept(&[kind as f32, 0.0]).unwrap();
            }
        }
        loop {
            let d = t.advance().unwrap();
            if d[0] == 6 {
                break;
            }
            assert_eq!(d[0], 5);
            let values = mock::read(d[1], d[2] as usize / 4, d[3] as usize);
            t.accept(&values).unwrap();
        }
        // Mock dispatches do not execute arithmetic; this tests ABI sequencing,
        // serialized tensor layouts, and ownership rather than GPU mathematics.
        let loaded = Gpt::try_from_bytes(&Naive, &t.bytes).unwrap();
        assert_eq!(loaded.config.n_embd, 6);
        assert!((loaded.train_loss - 3.0).abs() < 1e-6);
        assert!((loaded.validation_loss - 4.0).abs() < 1e-6);
        assert_eq!(loaded.tokenizer, t.model.tokenizer);
        assert_eq!(loaded.parameter_count(&Naive), t.model.parameter_count(&E));
        drop(t);
        assert_eq!(mock::count(), 0);
    }

    #[test]
    fn invalid_input_early_stop_and_non_finite_readbacks_do_not_commit_steps() {
        mock::reset();
        assert!(Training::new("abc", &options()).is_err());
        let mut bad = options();
        bad[6] = u32::MAX;
        assert!(Training::new(&text(), &bad).is_err());
        assert_eq!(mock::count(), 0);
        let mut t = Training::new(&text(), &options()).unwrap();
        t.stop();
        assert_eq!(t.advance().unwrap()[0], 7);
        assert!(t.bytes.is_empty());
        drop(t);
        assert_eq!(mock::count(), 0);
        let mut t = Training::new(&text(), &options()).unwrap();
        t.advance().unwrap();
        assert!(t.accept(&[f32::NAN, 0.0]).is_err());
        assert!(t.accept(&[1.0, 1.0]).is_err());
        assert_eq!(t.completed, 0);
        drop(t);
        assert_eq!(mock::count(), 0);
    }
}
