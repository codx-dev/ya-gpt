use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use ya_gpt::{
    engine::{Engine, naive::Naive},
    model::{Linear, Parameter, gpt::Gpt},
};

use crate::{WebGpuBuffer, WebGpuEngine, WebGpuWorkspace, host::elements};

pub(crate) struct Session {
    pub model: Gpt<WebGpuEngine>,
    tokens: Vec<usize>,
    remaining: u32,
    rng: StdRng,
    workspace: WebGpuWorkspace,
    pending: Option<WebGpuBuffer>,
    pub text: String,
}

fn parameter(p: &Parameter<WebGpuEngine>, n: usize) -> Result<()> {
    ensure!(
        p.values.len == n && p.gradients.len == n,
        "invalid model parameter shape"
    );
    Ok(())
}

fn linear(l: &Linear<WebGpuEngine>, inputs: usize, outputs: usize) -> Result<()> {
    ensure!(
        l.input_size == inputs && l.output_size == outputs,
        "invalid model linear dimensions"
    );
    parameter(&l.weight, elements(&[inputs, outputs])?)?;
    if let Some(bias) = &l.bias {
        parameter(bias, outputs)?;
    }
    Ok(())
}

fn validate(model: &Gpt<WebGpuEngine>) -> Result<()> {
    let c = model
        .config
        .clone()
        .validate(Some(model.vocab_size))?
        .n_embd;
    ensure!(
        model.blocks.len() == model.config.n_layer,
        "invalid model layer count"
    );
    ensure!(
        model.tokenizer.vocab_size() == model.vocab_size,
        "invalid model vocabulary"
    );
    parameter(&model.token_embedding, elements(&[model.vocab_size, c])?)?;
    parameter(
        &model.position_embedding,
        elements(&[model.config.block_size, c])?,
    )?;
    parameter(&model.final_norm.gamma, c)?;
    parameter(&model.final_norm.beta, c)?;
    linear(&model.language_head, c, model.vocab_size)?;
    for b in &model.blocks {
        for n in [&b.norm1, &b.norm2] {
            parameter(&n.gamma, c)?;
            parameter(&n.beta, c)?;
        }
        parameter(&b.qkv, elements(&[3, c, c])?)?;
        linear(&b.attention, c, c)?;
        linear(&b.expand, c, 4 * c)?;
        linear(&b.project, 4 * c, c)?;
    }
    Ok(())
}

impl Session {
    pub fn new(bytes: &[u8], prompt: &str, count: u32, seed: u64) -> Result<Self> {
        ensure!(count > 0, "new token count must be positive");
        let en = WebGpuEngine;
        let header = Gpt::header_from_bytes(bytes)?;
        let model = Gpt::try_from_bytes(&en, bytes).context("invalid YA-GPT model")?;
        validate(&model)?;
        ensure!(
            header.params == model.parameter_count(&en),
            "model parameter count mismatch"
        );
        let mut tokens = model.tokenizer.encode(prompt);
        if tokens.is_empty() {
            tokens.push(0);
        }
        Ok(Self {
            model,
            tokens,
            remaining: count,
            rng: StdRng::seed_from_u64(seed),
            workspace: WebGpuWorkspace::default(),
            pending: None,
            text: String::new(),
        })
    }

    pub fn forward(&mut self) -> Result<[u32; 3]> {
        ensure!(
            self.remaining > 0 && self.pending.is_none(),
            "generation is not ready for a forward pass"
        );
        let start = self
            .tokens
            .len()
            .saturating_sub(self.model.config.block_size);
        let context = &self.tokens[start..];
        let logits = self.model.forward_inference(
            &WebGpuEngine,
            context,
            1,
            context.len(),
            &mut self.workspace,
            &mut self.rng,
        )?;
        let offset = elements(&[context.len() - 1, self.model.vocab_size])? * 4;
        let result = [logits.handle, offset as u32, self.model.vocab_size as u32];
        self.pending = Some(logits);
        Ok(result)
    }

    pub fn accept(&mut self, logits: &[f32]) -> Result<bool> {
        ensure!(
            self.pending.is_some() && self.remaining > 0,
            "no logits requested"
        );
        ensure!(
            logits.len() == self.model.vocab_size,
            "invalid readback length"
        );
        let token =
            Naive.sample_last_token(&logits.to_vec(), 1, logits.len(), self.rng.random())?;
        self.tokens.push(token);
        self.pending = None;
        self.remaining -= 1;
        if self.remaining == 0 {
            self.text = self.model.tokenizer.decode(&self.tokens);
        }
        Ok(self.remaining == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::mock;
    const MODEL: &[u8] = include_bytes!("../../../frontend/tests/fixtures/micro.bin");

    #[test]
    fn stepped_sampling_matches_existing_generation_and_cleans_up() {
        for prompt in ["", "abc", "abcé🦀abcé🦀", "unknown"] {
            for seed in [0, 42, u64::MAX] {
                mock::reset();
                let naive = Gpt::try_from_bytes(&Naive, MODEL).unwrap();
                let expected = naive
                    .generate(
                        &Naive,
                        &naive.tokenizer.encode(prompt),
                        8,
                        &mut StdRng::seed_from_u64(seed),
                    )
                    .unwrap();
                let mut session = Session::new(MODEL, prompt, 8, seed).unwrap();
                assert!(session.accept(&[]).is_err());
                for i in 0..8 {
                    let context = &session.tokens
                        [session.tokens.len().saturating_sub(naive.config.block_size)..];
                    let logits = naive
                        .forward(
                            &Naive,
                            context,
                            1,
                            context.len(),
                            &mut StdRng::seed_from_u64(0),
                        )
                        .unwrap();
                    let last = logits[logits.len() - naive.vocab_size..].to_vec();
                    let descriptor = session.forward().unwrap();
                    assert_eq!(descriptor[2] as usize, naive.vocab_size);
                    assert_eq!(
                        descriptor[1] as usize,
                        (logits.len() - naive.vocab_size) * 4
                    );
                    assert!(session.forward().is_err());
                    assert_eq!(session.accept(&last).unwrap(), i == 7);
                }
                assert_eq!(session.text, naive.tokenizer.decode(&expected));
                assert!(session.forward().is_err());
                drop(session);
                assert_eq!(mock::count(), 0);
            }
        }
    }

    #[test]
    fn forward_records_the_expected_inference_operations() {
        mock::reset();
        let mut s = Session::new(MODEL, "abc", 1, 42).unwrap();
        s.forward().unwrap();
        let mut expected = vec![4]; // embeddings
        for _ in 0..2 {
            expected.extend([3, 2, 5, 6, 7, 2, 9, 3, 2, 8, 2, 9]);
        }
        expected.extend([3, 2]); // final norm and head
        assert_eq!(
            mock::commands()
                .iter()
                .map(|(op, _)| *op)
                .collect::<Vec<_>>(),
            expected
        );
        assert!(s.accept(&[f32::NAN; 6]).is_err());
        assert!(s.accept(&[0.0; 5]).is_err());
        drop(s);
        assert_eq!(mock::count(), 0);
    }

    #[test]
    fn malformed_models_are_rejected_and_partial_uploads_are_released() {
        mock::reset();
        for end in [0, 7, MODEL.len() / 2, MODEL.len() - 5] {
            assert!(Session::new(&MODEL[..end], "abc", 1, 0).is_err());
            assert_eq!(mock::count(), 0);
        }
        let mut model = Gpt::try_from_bytes(&Naive, MODEL).unwrap();
        model.blocks[0].expand.input_size += 1;
        assert!(Session::new(&model.to_bytes(&Naive).unwrap(), "abc", 1, 0).is_err());
        assert_eq!(mock::count(), 0);
    }
}
