//! Regenerate browser GPU reference data: cargo run -p ya-gpt-wasm --example fixtures
use rand::{SeedableRng, rngs::StdRng};
use serde_json::json;
use ya_gpt::{
    config::ModelConfig,
    engine::{naive::Naive, *},
    model::{Block, gpt::Gpt},
    tokenizer::Tokenizer,
};

fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 % 19) as f32 - 9.0) / 7.0).collect()
}

fn main() -> anyhow::Result<()> {
    let en = Naive;
    let (x, w, b) = (values(21), values(35), values(5));
    let mut biased = vec![0.0; 15];
    let mut unbiased = biased.clone();
    for (bias, output) in [(Some(&b), &mut biased), (None, &mut unbiased)] {
        Engine::linear_forward(
            &en,
            LinearSpec {
                rows: 3,
                inputs: 7,
                outputs: 5,
            },
            LinearWeights { weight: &w, bias },
            &x,
            output,
        )?;
    }
    let linear = json!({"x":x,"w":w,"b":b,"biased":biased,"unbiased":unbiased});
    let (x, gamma, beta) = (values(18), values(6), values(6));
    let mut output = vec![0.0; 18];
    en.layer_norm_forward(
        NormSpec {
            rows: 3,
            channels: 6,
            epsilon: 1e-5,
        },
        NormWeights {
            gamma: &gamma,
            beta: &beta,
        },
        &x,
        &mut output,
        None,
    )?;
    let norm = json!({"x":x,"gamma":gamma,"beta":beta,"expected":output});
    let (tokens, table, positions) = (vec![1, 2, 1, 0, 4, 2], values(30), values(30));
    let mut output = vec![0.0; 36];
    en.token_position_embedding_forward(
        EmbeddingSpec {
            batch: 2,
            time: 3,
            channels: 6,
            vocab: 5,
            positions: 5,
        },
        &tokens,
        &table,
        &positions,
        &mut output,
    )?;
    let embedding = json!({"tokens":tokens,"table":table,"positions":positions,"expected":output});
    let config = ModelConfig {
        block_size: 5,
        n_embd: 6,
        n_head: 2,
        n_layer: 2,
        dropout: 0.2,
    };
    let mut rng = StdRng::seed_from_u64(42);
    let block = Block::new(&en, &config, &mut rng)?;
    let input = values(36);
    let mut output = vec![0.0; 36];
    en.block_forward(
        BlockSpec {
            batch: 2,
            time: 3,
            channels: 6,
            heads: 2,
            dropout: 0.2,
        },
        block.weights(),
        &input,
        ForwardMode::default(),
        &mut Default::default(),
        &mut output,
        None,
    )?;
    let block = json!({"input":input,"expected":output,"norm1":[block.norm1.gamma.values,block.norm1.beta.values],"qkv":block.qkv.values,"attention":[block.attention.weight.values,block.attention.bias.unwrap().values],"norm2":[block.norm2.gamma.values,block.norm2.beta.values],"expand":[block.expand.weight.values,block.expand.bias.unwrap().values],"project":[block.project.weight.values,block.project.bias.unwrap().values]});
    let model = Gpt::new(&en, config, Tokenizer::char_stream("abcé🦀"), &mut rng)?;
    let tokens = model.tokenizer.encode("abc");
    let logits = model.forward(&en, &tokens, 1, tokens.len(), &mut rng)?;
    let generated = model.generate(&en, &tokens, 8, &mut StdRng::seed_from_u64(42))?;
    let output_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "frontend/tests/fixtures".into());
    std::fs::create_dir_all(&output_dir)?;
    std::fs::write(format!("{output_dir}/micro.bin"), model.to_bytes(&en)?)?;
    std::fs::write(
        format!("{output_dir}/gpu.json"),
        serde_json::to_vec_pretty(
            &json!({"linear":linear,"norm":norm,"embedding":embedding,"block":block,"model":{"logits":logits,"generated":model.tokenizer.decode(&generated)}}),
        )?,
    )?;
    Ok(())
}
