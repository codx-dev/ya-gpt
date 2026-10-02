use std::{
    fs::{self, File},
    io::{self, Read as _, Write as _},
    path::{Path, PathBuf},
};

use ya_gpt::{
    config::ModelConfig,
    engine::{self, Engine as GptEngine},
    model::{gpt::Gpt, optimizer::AdamW},
    tokenizer::Tokenizer,
};

use crate::cli::{Command, CommonArgs, Engine, GenerateArgs, InspectArgs, TrainingOptions};

pub fn execute(command: &Command, stdout: &mut impl io::Write) -> anyhow::Result<()> {
    match command {
        Command::Train(args) => {
            let bytes = train(
                &args.common,
                &args.resolved_model()?,
                &args.resolved_options(),
            )?;
            write_training_output(&bytes, args.output.as_deref(), stdout)
        }
        Command::Generate(args) => {
            let text = generate(args)?;
            stdout.write_all(text.as_bytes())?;
            Ok(stdout.flush()?)
        }
        Command::Inspect(args) => {
            let text = inspect(args)?;
            stdout.write_all(text.as_bytes())?;
            Ok(stdout.flush()?)
        }
    }
}

pub fn train(
    common: &CommonArgs,
    config: &ModelConfig,
    options: &TrainingOptions,
) -> anyhow::Result<Vec<u8>> {
    fn run<EN: GptEngine>(
        en: &EN,
        common: &CommonArgs,
        config: &ModelConfig,
        options: &TrainingOptions,
    ) -> anyhow::Result<Vec<u8>> {
        let mut input = options
            .input
            .as_ref()
            .map(fs::read_to_string)
            .transpose()?
            .unwrap_or_default();

        if options.input.is_none() {
            io::stdin().read_to_string(&mut input)?;
        }

        let tokenizer = Tokenizer::new(&input);
        let mut rng = common.rng();

        let mut model = Gpt::new(en, config.clone(), tokenizer.clone(), &mut rng)?;

        let mut optimizer = AdamW::new(
            en,
            &tokenizer,
            &input,
            &mut model,
            config.clone(),
            options.report_loss_per_step,
        )?;

        if let Some(batch_size) = options.batch_size {
            optimizer = optimizer.with_batch_size(batch_size);
        }

        if let Some(iterations) = options.iterations {
            optimizer = optimizer.with_max_iters(iterations);
        }

        if let Some(learning_rate) = options.learning_rate {
            optimizer = optimizer.with_learning_rate(learning_rate);
        }

        if let Some(weight_decay) = options.weight_decay {
            optimizer = optimizer.with_weight_decay(weight_decay);
        }

        optimizer.train(en, &mut model, &mut rng)?;
        model.to_bytes(en)
    }

    match common.engine {
        Engine::Naive => run(&engine::naive::Naive, common, config, options),

        #[cfg(feature = "simd-wide")]
        Engine::SimdWide => run(&engine::wide::SimdWide, common, config, options),
    }
}

pub fn generate(args: &GenerateArgs) -> anyhow::Result<String> {
    fn run<EN: GptEngine>(en: &EN, args: &GenerateArgs) -> anyhow::Result<String> {
        let mut rng = args.common.rng();

        let bytes = read_bytes_from_input(args.input.as_ref())?;
        let model = Gpt::try_from_bytes(en, &bytes)?;
        let prompt = model.tokenizer.encode(&args.prompt);
        let output = model.generate(en, &prompt, args.num_tokens, &mut rng)?;

        Ok(model.tokenizer.decode(&output))
    }

    match args.common.engine {
        Engine::Naive => run(&engine::naive::Naive, args),

        #[cfg(feature = "simd-wide")]
        Engine::SimdWide => run(&engine::wide::SimdWide, args),
    }
}

pub fn inspect(args: &InspectArgs) -> anyhow::Result<String> {
    let bytes = read_bytes_from_input(args.input.as_ref())?;
    // engine is irrelevant for header parse
    let header = Gpt::<engine::naive::Naive>::header_from_bytes(&bytes)?;

    Ok(serde_json::to_string(&header)?)
}

fn read_bytes_from_input(args: Option<&PathBuf>) -> anyhow::Result<Vec<u8>> {
    let mut input = args.as_ref().map(fs::read).transpose()?.unwrap_or_default();

    if args.is_none() {
        io::stdin().read_to_end(&mut input)?;
    }

    Ok(input)
}

fn write_training_output(
    bytes: &[u8],
    output: Option<&Path>,
    stdout: &mut impl io::Write,
) -> anyhow::Result<()> {
    if let Some(path) = output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(path)?;
        file.write_all(bytes)?;
        Ok(file.flush()?)
    } else {
        stdout.write_all(bytes)?;
        Ok(stdout.flush()?)
    }
}
