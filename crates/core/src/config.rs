use msgpacker::MsgPacker;

#[derive(Debug, Clone, PartialEq, MsgPacker)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct ModelConfig {
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub dropout: f64,
}

impl ModelConfig {
    pub fn micro() -> Self {
        Self {
            block_size: 4,
            n_embd: 4,
            n_head: 2,
            n_layer: 2,
            dropout: 0.0,
        }
    }

    pub fn tiny() -> Self {
        Self {
            block_size: 32,
            n_embd: 32,
            n_head: 4,
            n_layer: 2,
            dropout: 0.2,
        }
    }

    pub fn small() -> Self {
        Self {
            block_size: 64,
            n_embd: 64,
            n_head: 4,
            n_layer: 2,
            dropout: 0.2,
        }
    }

    pub fn medium() -> Self {
        Self {
            block_size: 128,
            n_embd: 128,
            n_head: 4,
            n_layer: 4,
            dropout: 0.2,
        }
    }

    pub fn large() -> Self {
        Self {
            block_size: 256,
            n_embd: 256,
            n_head: 8,
            n_layer: 6,
            dropout: 0.2,
        }
    }

    pub fn validate(self, vocab_size: Option<usize>) -> anyhow::Result<Self> {
        anyhow::ensure!(self.block_size != 0, "block size cannot be zero");
        anyhow::ensure!(self.n_embd != 0, "embedding width cannot be zero");
        anyhow::ensure!(self.n_head != 0, "number of heads cannot be zero");
        anyhow::ensure!(self.n_layer != 0, "number of layers cannot be zero");
        anyhow::ensure!(
            self.n_embd.is_multiple_of(self.n_head),
            "embedding width must be divisible by the number of heads"
        );
        anyhow::ensure!(
            (0.0..1.0).contains(&self.dropout),
            "dropout must be finite and in [0, 1)"
        );

        for dimensions in [
            [self.n_embd, self.n_embd, 4],
            [vocab_size.unwrap_or(1), self.n_embd, 1],
            [self.block_size, self.n_embd, 1],
            [self.block_size, self.block_size, 1],
        ] {
            let elements = dimensions
                .into_iter()
                .try_fold(1usize, |n, d| n.checked_mul(d));

            anyhow::ensure!(
                !elements.is_none_or(|n| n > isize::MAX as usize / size_of::<f64>()),
                "model dimensions are too large"
            );
        }

        Ok(self)
    }
}

#[test]
fn presets_are_valid() {
    ModelConfig::micro().validate(Some(65)).unwrap();
    ModelConfig::tiny().validate(Some(65)).unwrap();
    ModelConfig::small().validate(Some(65)).unwrap();
    ModelConfig::medium().validate(Some(65)).unwrap();
    ModelConfig::large().validate(Some(65)).unwrap();
}
