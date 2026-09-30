use std::{
    fs::{self, File},
    io::{self, Read as _, Write as _},
    path::Path,
};

use msgpacker::{Packable as _, Unpackable as _};
use ya_gpt::{
    config::ModelConfig,
    data::tokenizer::Tokenizer,
    engine::{gpt::Gpt, naive::Naive, optmizer::AdamW},
};

use crate::cli::{Command, CommonArgs, Engine, GenerateArgs, TrainingOptions};

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
    }
}

pub fn train(
    common: &CommonArgs,
    config: &ModelConfig,
    options: &TrainingOptions,
) -> anyhow::Result<Vec<u8>> {
    let engine = match common.engine {
        Engine::Naive => Naive,
    };

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
    let mut model = Gpt::new(config.clone(), tokenizer.clone(), &mut rng)?;

    let mut optimizer = AdamW::new(
        &tokenizer,
        &input,
        &mut model,
        config.clone(),
        options.report_loss_per_step,
    );
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
    optimizer.train(&engine, &mut model, &mut rng)?;

    Ok(model.pack_to_vec())
}

pub fn generate(args: &GenerateArgs) -> anyhow::Result<String> {
    let engine = match args.common.engine {
        Engine::Naive => Naive,
    };

    let mut input = args
        .input
        .as_ref()
        .map(fs::read)
        .transpose()?
        .unwrap_or_default();

    if args.input.is_none() {
        io::stdin().read_to_end(&mut input)?;
    }

    let mut rng = args.common.rng();
    let model = Gpt::unpack(&input)?;
    let prompt = model.tokenizer.encode(&args.prompt);
    let output = model.generate(&engine, &prompt, args.num_tokens, &mut rng);

    Ok(model.tokenizer.decode(&output))
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
