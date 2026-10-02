use std::path::PathBuf;

use anyhow::Context as _;
use clap::error::ErrorKind;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use rand::SeedableRng as _;
use rand::rngs::{StdRng, SysRng};
use ya_gpt::config::ModelConfig;

use crate::validation::{positive_usize, unit_interval};

#[derive(Parser, Debug)]
#[command(version, about, propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    /// Check relationships between arguments after clap has parsed their values.
    pub fn validate(&self) -> Result<(), clap::Error> {
        match &self.command {
            Command::Train(args) => args.resolved_model().map(|_| ()).map_err(|message| {
                Self::command().error(ErrorKind::ValueValidation, message.to_string())
            }),
            Command::Generate(_) => Ok(()),
            Command::Inspect(_) => Ok(()),
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Train a model and write its bytes to stdout or a file.
    Train(TrainArgs),
    /// Generate text from a prompt.
    Generate(GenerateArgs),
    /// Inspect the model contents.
    Inspect(InspectArgs),
}

#[derive(Args, Debug)]
pub struct CommonArgs {
    /// Optional random seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Model execution engine.
    #[arg(long, value_enum, default_value = "naive")]
    pub engine: Engine,
}

impl CommonArgs {
    pub fn rng(&self) -> StdRng {
        match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::try_from_rng(&mut SysRng).unwrap(),
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Naive,

    #[cfg(feature = "simd-wide")]
    SimdWide,
}

#[derive(Args, Debug)]
pub struct ModelArgs {
    /// Context length; otherwise taken from the preset.
    #[arg(required_unless_present = "preset", value_parser = positive_usize)]
    pub block_size: Option<usize>,

    /// Embedding width; otherwise taken from the preset.
    #[arg(required_unless_present = "preset", value_parser = positive_usize)]
    pub n_embd: Option<usize>,

    /// Number of attention heads; otherwise taken from the preset.
    #[arg(required_unless_present = "preset", value_parser = positive_usize)]
    pub n_head: Option<usize>,

    /// Number of transformer layers; otherwise taken from the preset.
    #[arg(required_unless_present = "preset", value_parser = positive_usize)]
    pub n_layer: Option<usize>,

    /// Dropout probability in [0, 1); otherwise taken from the preset.
    #[arg(required_unless_present = "preset", value_parser = unit_interval)]
    pub dropout: Option<f32>,
}

#[derive(Args, Debug)]
pub struct TrainArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    #[command(flatten)]
    pub model: ModelArgs,

    /// Defaults for model dimensions, dropout, batch size, and iterations;
    /// explicit values take precedence.
    #[arg(long, value_enum)]
    pub preset: Option<TrainingPreset>,

    /// Number of samples per batch; otherwise taken from the preset, if any.
    #[arg(long, value_parser = positive_usize)]
    pub batch_size: Option<usize>,

    /// Number of training iterations; otherwise taken from the preset, if any.
    #[arg(long, value_parser = positive_usize)]
    pub iterations: Option<usize>,

    /// Learning rate, between 0.0 and 1.0 inclusive.
    #[arg(long, value_parser = unit_interval)]
    pub learning_rate: Option<f32>,

    /// Weight decay, between 0.0 and 1.0 inclusive.
    #[arg(long, value_parser = unit_interval)]
    pub weight_decay: Option<f32>,

    /// Report cross-entropy loss to stderr after every training step (adds overhead).
    #[arg(long)]
    pub report_loss_per_step: bool,

    /// Read data input from this file; defaults to stdin.
    #[arg(short, long, value_name = "FILE")]
    pub input: Option<PathBuf>,

    /// Write model bytes to this file, overwriting it; defaults to stdout.
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

impl TrainArgs {
    pub fn resolved_model(&self) -> anyhow::Result<ModelConfig> {
        let defaults = self.preset.map(TrainingPreset::model_config);
        ModelConfig {
            block_size: self
                .model
                .block_size
                .or_else(|| defaults.as_ref().map(|model| model.block_size))
                .context("block size is required without a preset")?,
            n_embd: self
                .model
                .n_embd
                .or_else(|| defaults.as_ref().map(|model| model.n_embd))
                .context("embedding width is required without a preset")?,
            n_head: self
                .model
                .n_head
                .or_else(|| defaults.as_ref().map(|model| model.n_head))
                .context("number of heads is required without a preset")?,
            n_layer: self
                .model
                .n_layer
                .or_else(|| defaults.as_ref().map(|model| model.n_layer))
                .context("number of layers is required without a preset")?,
            dropout: self
                .model
                .dropout
                .or_else(|| defaults.as_ref().map(|model| model.dropout))
                .context("dropout is required without a preset")?,
        }
        .validate(None)
    }

    pub fn resolved_options(&self) -> TrainingOptions {
        let defaults = self.preset.map(TrainingPreset::defaults);
        TrainingOptions {
            batch_size: self
                .batch_size
                .or(defaults.map(|(batch_size, _)| batch_size)),
            iterations: self
                .iterations
                .or(defaults.map(|(_, iterations)| iterations)),
            learning_rate: self.learning_rate,
            weight_decay: self.weight_decay,
            input: self.input.clone(),
            report_loss_per_step: self.report_loss_per_step,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingOptions {
    pub batch_size: Option<usize>,
    pub iterations: Option<usize>,
    pub learning_rate: Option<f32>,
    pub weight_decay: Option<f32>,
    pub input: Option<PathBuf>,
    pub report_loss_per_step: bool,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPreset {
    /// Batch size 8, 1,000 iterations.
    Tiny,
    /// Batch size 16, 2,000 iterations.
    Small,
    /// Batch size 32, 5,000 iterations.
    Medium,
    /// Batch size 64, 10,000 iterations.
    Large,
}

impl TrainingPreset {
    fn model_config(self) -> ModelConfig {
        match self {
            Self::Tiny => ModelConfig::tiny(),
            Self::Small => ModelConfig::small(),
            Self::Medium => ModelConfig::medium(),
            Self::Large => ModelConfig::large(),
        }
    }

    fn defaults(self) -> (usize, usize) {
        match self {
            Self::Tiny => (8, 1_000),
            Self::Small => (16, 2_000),
            Self::Medium => (32, 5_000),
            Self::Large => (64, 10_000),
        }
    }
}

#[derive(Args, Debug)]
pub struct GenerateArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Read model from this file; defaults to stdin.
    #[arg(short, long, value_name = "FILE")]
    pub input: Option<PathBuf>,

    /// Text to start generation from. Quote prompts containing spaces.
    pub prompt: String,

    /// Number of new tokens to generate.
    #[arg(long, value_parser = positive_usize)]
    pub num_tokens: usize,
}

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Read model from this file; defaults to stdin.
    #[arg(short, long, value_name = "FILE")]
    pub input: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_train(args: &[&str]) -> TrainArgs {
        let cli = Cli::try_parse_from(
            ["ya-gpt-cli", "train"]
                .into_iter()
                .chain(args.iter().copied()),
        )
        .unwrap();
        cli.validate().unwrap();
        let Command::Train(args) = cli.command else {
            panic!("expected training arguments");
        };
        args
    }

    #[test]
    fn presets_supply_model_and_training_defaults() {
        for (preset, model, batch_size, iterations) in [
            ("tiny", ModelConfig::tiny(), 8, 1_000),
            ("small", ModelConfig::small(), 16, 2_000),
            ("medium", ModelConfig::medium(), 32, 5_000),
            ("large", ModelConfig::large(), 64, 10_000),
        ] {
            let args = parse_train(&["--preset", preset, "--input", "data/input.txt"]);
            assert_eq!(args.resolved_model().unwrap(), model);
            let options = args.resolved_options();
            assert_eq!(options.batch_size, Some(batch_size));
            assert_eq!(options.iterations, Some(iterations));
            assert_eq!(options.input, Some(PathBuf::from("data/input.txt")));
        }
    }

    #[test]
    fn explicit_values_override_preset_defaults() {
        let args = parse_train(&[
            "--preset",
            "tiny",
            "4",
            "4",
            "2",
            "2",
            "0",
            "--batch-size",
            "1",
            "--iterations",
            "2",
            "--learning-rate",
            "0.01",
            "--weight-decay",
            "0.1",
        ]);
        assert_eq!(args.resolved_model().unwrap(), ModelConfig::micro());
        let options = args.resolved_options();
        assert_eq!(options.batch_size, Some(1));
        assert_eq!(options.iterations, Some(2));
        assert_eq!(options.learning_rate, Some(0.01));
        assert_eq!(options.weight_decay, Some(0.1));
    }

    #[test]
    fn preset_fills_unspecified_model_values() {
        let args = parse_train(&["--preset", "tiny", "16"]);
        let mut expected = ModelConfig::tiny();
        expected.block_size = 16;
        assert_eq!(args.resolved_model().unwrap(), expected);
    }

    #[test]
    fn explicit_model_still_works_without_a_preset() {
        let args = parse_train(&["4", "4", "2", "2", "0"]);
        assert_eq!(args.resolved_model().unwrap(), ModelConfig::micro());
        assert_eq!(args.resolved_options().batch_size, None);
        assert_eq!(args.resolved_options().iterations, None);

        for args in [vec![], vec!["4", "4", "2", "2"]] {
            let error =
                Cli::try_parse_from(["ya-gpt-cli", "train"].into_iter().chain(args)).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
        }
    }

    #[test]
    fn resolved_model_is_validated() {
        for model in [["4", "5", "2", "2", "0"], ["4", "4", "2", "2", "1"]] {
            let cli = Cli::try_parse_from(
                ["ya-gpt-cli", "train", "--preset", "tiny"]
                    .into_iter()
                    .chain(model),
            )
            .unwrap();
            assert_eq!(
                cli.validate().unwrap_err().kind(),
                ErrorKind::ValueValidation
            );
        }
    }
}
