use anyhow::{Context, Result, ensure};
use ya_gpt::engine::{AdamWConfig, BlockSpec, EmbeddingSpec, LinearSpec, NormSpec};

pub(crate) fn elements(dimensions: &[usize]) -> Result<usize> {
    let n = dimensions
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .context("tensor shape overflow")?;
    ensure!(
        n <= isize::MAX as usize / size_of::<f32>(),
        "tensor allocation is too large"
    );
    Ok(n)
}

pub(crate) fn dimension(n: usize) -> Result<i32> {
    ensure!(n > 0, "tensor dimensions must be positive");
    i32::try_from(n).context("dimension exceeds the cuBLAS i32 limit")
}

pub(crate) fn linear(s: LinearSpec) -> Result<()> {
    dimension(s.rows)?;
    dimension(s.inputs)?;
    dimension(s.outputs)?;
    elements(&[s.rows, s.inputs])?;
    elements(&[s.rows, s.outputs])?;
    elements(&[s.inputs, s.outputs])?;
    Ok(())
}

pub(crate) fn norm(s: NormSpec) -> Result<usize> {
    dimension(s.rows)?;
    dimension(s.channels)?;
    ensure!(
        s.epsilon.is_finite() && s.epsilon > 0.0,
        "invalid normalization epsilon"
    );
    elements(&[s.rows, s.channels])
}

pub(crate) fn block(s: BlockSpec) -> Result<(usize, usize)> {
    dimension(s.batch)?;
    dimension(s.time)?;
    dimension(s.channels)?;
    dimension(s.heads)?;
    ensure!(
        s.channels.is_multiple_of(s.heads),
        "channels must be divisible by heads"
    );
    ensure!(
        (0.0..1.0).contains(&s.dropout),
        "invalid dropout probability"
    );
    let rows = elements(&[s.batch, s.time])?;
    let hidden = elements(&[s.channels, 4])?;
    linear(LinearSpec {
        rows,
        inputs: s.channels,
        outputs: hidden,
    })?;
    let n = elements(&[rows, s.channels])?;
    elements(&[n, 4])?;
    let scores = elements(&[s.batch, s.heads, s.time, s.time])?;
    Ok((n, scores))
}

pub(crate) fn embedding(s: EmbeddingSpec) -> Result<usize> {
    for d in [s.batch, s.time, s.channels, s.vocab, s.positions] {
        dimension(d)?;
    }
    ensure!(s.time <= s.positions, "sequence exceeds position table");
    elements(&[s.vocab, s.channels])?;
    elements(&[s.positions, s.channels])?;
    elements(&[s.batch, s.time, s.channels])
}

pub(crate) fn adamw(c: AdamWConfig) -> Result<()> {
    ensure!(
        c.learning_rate.is_finite()
            && c.learning_rate >= 0.0
            && c.weight_decay.is_finite()
            && c.weight_decay >= 0.0
            && (0.0..1.0).contains(&c.beta1)
            && (0.0..1.0).contains(&c.beta2)
            && c.epsilon.is_finite()
            && c.epsilon > 0.0
            && c.step > 0,
        "invalid AdamW configuration"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_configuration_rejects_non_finite_and_boundary_values() {
        let valid = AdamWConfig {
            learning_rate: 0.03,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step: 1,
        };
        assert!(adamw(valid).is_ok());
        for invalid in [
            AdamWConfig {
                learning_rate: f32::NAN,
                ..valid
            },
            AdamWConfig {
                learning_rate: -1.0,
                ..valid
            },
            AdamWConfig {
                weight_decay: f32::INFINITY,
                ..valid
            },
            AdamWConfig {
                weight_decay: -1.0,
                ..valid
            },
            AdamWConfig {
                beta1: 1.0,
                ..valid
            },
            AdamWConfig {
                beta2: f32::NAN,
                ..valid
            },
            AdamWConfig {
                epsilon: 0.0,
                ..valid
            },
            AdamWConfig {
                epsilon: f32::INFINITY,
                ..valid
            },
            AdamWConfig { step: 0, ..valid },
        ] {
            assert!(adamw(invalid).is_err());
        }
    }

    #[test]
    fn shapes_reject_overflow_and_invalid_dimensions() {
        assert_eq!(elements(&[0]).unwrap(), 0);
        assert_eq!(elements(&[2, 3, 7]).unwrap(), 42);
        assert!(elements(&[usize::MAX, 2]).is_err());
        assert!(elements(&[isize::MAX as usize]).is_err());
        assert!(dimension(0).is_err());
        assert!(dimension(i32::MAX as usize + 1).is_err());
        let spec = BlockSpec {
            batch: 2,
            time: 3,
            channels: 6,
            heads: 2,
            dropout: 0.2,
        };
        assert_eq!(block(spec).unwrap(), (36, 36));
        assert!(block(BlockSpec { heads: 0, ..spec }).is_err());
        assert!(block(BlockSpec { heads: 4, ..spec }).is_err());
        assert!(
            block(BlockSpec {
                dropout: f32::NAN,
                ..spec
            })
            .is_err()
        );
        assert!(
            block(BlockSpec {
                dropout: 1.0,
                ..spec
            })
            .is_err()
        );
        assert!(
            block(BlockSpec {
                channels: i32::MAX as usize,
                heads: 1,
                ..spec
            })
            .is_err()
        );
        assert!(
            norm(NormSpec {
                rows: 1,
                channels: 2,
                epsilon: 0.0
            })
            .is_err()
        );
        assert!(
            embedding(EmbeddingSpec {
                batch: 1,
                time: 2,
                channels: 1,
                vocab: 3,
                positions: 1
            })
            .is_err()
        );
    }
}
