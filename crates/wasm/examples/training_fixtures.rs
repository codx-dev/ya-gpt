//! cargo run -p ya-gpt-wasm --example training_fixtures
use rand::{SeedableRng, rngs::StdRng};
use serde_json::json;
use ya_gpt::{
    config::ModelConfig,
    engine::{naive::Naive, *},
    model::{gpt::Gpt, optimizer::AdamW},
    tokenizer::Tokenizer,
};

fn main() -> anyhow::Result<()> {
    let directory = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "frontend/tests/fixtures".into());
    let text = "abcé🦀abcabcabc".repeat(24);
    let mut fixtures = Vec::new();
    for (name, dropout, seed, iterations) in [
        ("training-dropout", 0.2, 42, 4),
        ("training-learning", 0.0, u64::MAX, 8),
    ] {
        let config = ModelConfig {
            block_size: 5,
            n_embd: 6,
            n_head: 2,
            n_layer: 2,
            dropout,
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let tokenizer = Tokenizer::new(&text);
        let mut model = Gpt::new(&Naive, config.clone(), tokenizer, &mut rng)?;
        let tokens = model.tokenizer.encode("abcab");
        let targets = model.tokenizer.encode("bcabc");
        let initial_logits = model.forward(&Naive, &tokens, 1, 5, &mut StdRng::seed_from_u64(0))?;
        let initial_loss = Naive.cross_entropy(&initial_logits, &targets, 5, model.vocab_size)?;
        AdamW::new(
            &Naive,
            &model.tokenizer.clone(),
            &text,
            &mut model,
            config.clone(),
            false,
        )?
        .with_batch_size(2)
        .with_max_iters(iterations)
        .with_learning_rate(0.01)
        .with_weight_decay(0.01)
        .train(&Naive, &mut model, &mut rng)?;
        let logits = model.forward(&Naive, &tokens, 1, 5, &mut StdRng::seed_from_u64(0))?;
        let final_loss = Naive.cross_entropy(&logits, &targets, 5, model.vocab_size)?;
        let bytes = model.to_bytes(&Naive)?;
        std::fs::write(format!("{directory}/{name}.bin"), bytes)?;
        fixtures.push(json!({"name":name,"text":text,"options":{"backend":"webgpu","model":{"block_size":config.block_size,"n_embd":config.n_embd,"n_head":config.n_head,"n_layer":config.n_layer,"dropout":dropout},"batch_size":2,"iterations":iterations,"learning_rate":0.01,"weight_decay":0.01,"report_loss_per_step":false,"seed":seed.to_string()},"initial_loss":initial_loss,"final_loss":final_loss}));
    }
    std::fs::write(
        format!("{directory}/training.json"),
        serde_json::to_vec_pretty(&fixtures)?,
    )?;
    kernel_fixtures(&directory)?;
    Ok(())
}

fn kernel_fixtures(directory: &str) -> anyhow::Result<()> {
    let data = |n: usize| {
        (0..n)
            .map(|i| ((i * 17 % 23) as f32 - 11.0) / 13.0)
            .collect::<Vec<_>>()
    };
    let (x, w, dy) = (data(21), data(35), data(15));
    let (mut dx, mut dw, mut db) = (vec![99.0; 21], vec![99.0; 35], vec![99.0; 5]);
    Naive.linear_backward(
        LinearSpec {
            rows: 3,
            inputs: 7,
            outputs: 5,
        },
        LinearWeights {
            weight: &w,
            bias: Some(&vec![0.0; 5]),
        },
        LinearGradients {
            weight: &mut dw,
            bias: Some(&mut db),
        },
        &x,
        &dy,
        &mut dx,
    )?;
    let linear = json!({"x":x,"w":w,"dy":dy,"dx":dx,"dw":dw,"db":db});
    let (x, gamma, beta, dy) = (data(18), data(6), vec![0.1; 6], data(18));
    let (mut out, mut dx, mut dg, mut db) =
        (vec![0.0; 18], vec![99.0; 18], vec![99.0; 6], vec![99.0; 6]);
    let mut cache = Default::default();
    let norm = NormSpec {
        rows: 3,
        channels: 6,
        epsilon: 1e-5,
    };
    Naive.layer_norm_forward(
        norm,
        NormWeights {
            gamma: &gamma,
            beta: &beta,
        },
        &x,
        &mut out,
        Some(&mut cache),
    )?;
    Naive.layer_norm_backward(
        norm,
        NormWeights {
            gamma: &gamma,
            beta: &beta,
        },
        NormGradients {
            gamma: &mut dg,
            beta: &mut db,
        },
        &cache,
        &dy,
        &mut dx,
    )?;
    let norm = json!({"x":x,"gamma":gamma,"beta":beta,"dy":dy,"out":out,"dx":dx,"dg":dg,"db":db});
    let tokens = vec![1, 1, 2, 1, 3, 2];
    let dy = data(36);
    let (mut dt, mut dp) = (vec![99.0; 30], vec![99.0; 30]);
    Naive.token_position_embedding_backward(
        EmbeddingSpec {
            batch: 2,
            time: 3,
            channels: 6,
            vocab: 5,
            positions: 5,
        },
        &tokens,
        &dy,
        &mut dt,
        &mut dp,
    )?;
    let embedding = json!({"tokens":tokens,"dy":dy,"dt":dt,"dp":dp});
    let logits = vec![1000.0, 999.0, 998.0, -1000.0, -999.0, -998.0];
    let targets = vec![1, 2];
    let mut grad = vec![99.0; 6];
    let loss = Naive.cross_entropy_with_grad(&logits, &targets, 2, 3, &mut grad)?;
    let entropy = json!({"logits":logits,"targets":targets,"gradient":grad,"loss":loss});
    let initial = data(259); // Crosses the finite-scan and optimizer chunk boundary.
    let gradients = data(259);
    let mut values = initial.clone();
    let (mut first, mut second) = (vec![0.0; 259], vec![0.0; 259]);
    for step in 1..=3 {
        Naive.adamw_step(
            &mut [AdamWGroup {
                values: &mut values,
                gradients: &gradients,
                first_moment: &mut first,
                second_moment: &mut second,
            }],
            AdamWConfig {
                learning_rate: 0.01,
                weight_decay: 0.03,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                step,
            },
        )?;
    }
    let adamw = json!({"initial":initial,"gradients":gradients,"values":values,"first":first,"second":second});
    std::fs::write(
        format!("{directory}/gpu-training.json"),
        serde_json::to_vec_pretty(
            &json!({"linear":linear,"norm":norm,"embedding":embedding,"entropy":entropy,"adamw":adamw}),
        )?,
    )?;
    Ok(())
}
