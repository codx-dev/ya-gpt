use super::{gpt::Gpt, optimizer::AdamW, types::Parameter};
use crate::{
    config::ModelConfig,
    engine::{naive::Naive, *},
    tokenizer::Tokenizer,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}
fn model<E: Engine>(en: &E) -> Gpt<E> {
    Gpt::new(en, ModelConfig::micro(), Tokenizer::new("abc"), &mut rng()).unwrap()
}
fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance * (1. + b.abs()),
            "{a} != {b}"
        );
    }
}

#[test]
fn initialization_and_disabled_dropout_are_reproducible() {
    let a = Parameter::normal(&Naive, 16, &mut rng()).unwrap();
    let b = Parameter::normal(&Naive, 16, &mut rng()).unwrap();
    assert_eq!(a.values, b.values);
    assert_eq!(a.gradients, vec![0.; 16]);
    let gpt = model(&Naive);
    let mut source = rng();
    let mut untouched = rng();
    let (cached, _) = gpt
        .forward_with_cache(&Naive, &[0, 1, 2, 0], 1, 4, true, &mut source)
        .unwrap();
    assert_eq!(source.random::<u64>(), untouched.random::<u64>());
    let inference = gpt.forward(&Naive, &[0, 1, 2, 0], 1, 4).unwrap();
    close(&cached, &inference, 0.);
    let mut config = ModelConfig::micro();
    config.dropout = 0.4;
    let gpt = Gpt::new(&Naive, config, Tokenizer::new("abc"), &mut rng()).unwrap();
    let mut source = rng();
    gpt.forward_with_cache(&Naive, &[0, 1, 2, 0], 1, 4, false, &mut source)
        .unwrap();
    assert_eq!(source.random::<u64>(), rng().random::<u64>());
}

#[test]
fn complete_model_gradients_match_finite_differences() {
    let mut gpt = model(&Naive);
    let tokens = [0, 1, 0, 2];
    let targets = [1, 0, 2, 1];
    gpt.loss_and_backward(&Naive, &tokens, &targets, 2, 2, &mut rng())
        .unwrap();
    let gradients: Vec<_> = gpt
        .parameters()
        .iter()
        .map(|p| p.gradients.clone())
        .collect();
    for (p, expected) in gradients.iter().enumerate() {
        let n = expected.len();
        // Block gradients are checked exhaustively by the engine conformance suite.
        for index in [0, n / 2, n - 1] {
            let original = gpt.parameters()[p].values[index];
            let h = 2e-3 * (1. + original.abs());
            gpt.parameters_mut()[p].values[index] = original + h;
            let plus = Naive
                .cross_entropy(
                    &gpt.forward(&Naive, &tokens, 2, 2).unwrap(),
                    &targets.to_vec(),
                    4,
                    gpt.vocab_size,
                )
                .unwrap();
            gpt.parameters_mut()[p].values[index] = original - h;
            let minus = Naive
                .cross_entropy(
                    &gpt.forward(&Naive, &tokens, 2, 2).unwrap(),
                    &targets.to_vec(),
                    4,
                    gpt.vocab_size,
                )
                .unwrap();
            gpt.parameters_mut()[p].values[index] = original;
            close(&[expected[index]], &[(plus - minus) / (2. * h)], 5e-3);
        }
    }
}

#[test]
fn gradients_reuse_allocations_and_training_reduces_loss() {
    let mut gpt = model(&Naive);
    let pointers: Vec<_> = gpt
        .parameters()
        .iter()
        .map(|p| (p.values.as_ptr(), p.gradients.as_ptr()))
        .collect();
    let mut optimizer = AdamW::new(
        &Naive,
        &gpt.tokenizer.clone(),
        &"abc".repeat(100),
        &mut gpt,
        ModelConfig::micro(),
        false,
    )
    .unwrap()
    .with_learning_rate(0.02);
    let input = [0, 1, 2, 0, 1, 2, 0, 1];
    let targets = [1, 2, 0, 1, 2, 0, 1, 2];
    let mut workspace = <Naive as Engine>::Workspace::default();
    let initial = gpt
        .loss_and_backward_with_workspace(
            &Naive,
            &input,
            &targets,
            2,
            4,
            &mut rng(),
            &mut workspace,
        )
        .unwrap();
    for _ in 0..40 {
        gpt.loss_and_backward_with_workspace(
            &Naive,
            &input,
            &targets,
            2,
            4,
            &mut rng(),
            &mut workspace,
        )
        .unwrap();
        optimizer.step(&Naive, &mut gpt.parameters_mut()).unwrap();
    }
    let loss = gpt
        .loss_and_backward_with_workspace(
            &Naive,
            &input,
            &targets,
            2,
            4,
            &mut rng(),
            &mut workspace,
        )
        .unwrap();
    assert!(loss < initial * 0.5, "loss {initial} -> {loss}");
    for (p, &(v, g)) in gpt.parameters().iter().zip(&pointers) {
        assert_eq!(p.values.as_ptr(), v);
        assert_eq!(p.gradients.as_ptr(), g);
    }
}

/// This engine makes accidental device-to-host bulk reads fail immediately.
#[derive(Default)]
struct Tracked {
    forward: std::cell::Cell<usize>,
    backward: std::cell::Cell<usize>,
    saved: std::cell::Cell<usize>,
    norm_saved: std::cell::Cell<usize>,
    uploads: std::cell::Cell<usize>,
    validations: std::cell::Cell<usize>,
    samples: std::cell::Cell<usize>,
}
macro_rules! delegate {
    ($(fn $name:ident(&self $(, $arg:ident : $ty:ty)*) -> $ret:ty; )*) => {
        $(fn $name(&self $(, $arg:$ty)*) -> $ret { Naive.$name($($arg),*) })*
    };
}
impl Engine for Tracked {
    type Buffer = Vec<f32>;
    type IndexBuffer = Vec<usize>;
    type BlockCache = <Naive as Engine>::BlockCache;
    type LayerNormCache = <Naive as Engine>::LayerNormCache;
    type Workspace = <Naive as Engine>::Workspace;
    delegate! {
        fn zeroes(&self,n:usize)->anyhow::Result<Vec<f32>>;
        fn filled(&self,v:f32,n:usize)->anyhow::Result<Vec<f32>>;
        fn buffer_len(&self,b:&Vec<f32>)->usize;
        fn buffer_from_slice(&self,v:&[f32])->anyhow::Result<Vec<f32>>;
        fn synchronize(&self)->anyhow::Result<()>;
        fn linear_forward(&self,s:LinearSpec,w:LinearWeights<'_,Vec<f32>>,x:&Vec<f32>,out:&mut Vec<f32>)->anyhow::Result<()>;
        fn linear_backward(&self,s:LinearSpec,w:LinearWeights<'_,Vec<f32>>,g:LinearGradients<'_,Vec<f32>>,x:&Vec<f32>,dy:&Vec<f32>,dx:&mut Vec<f32>)->anyhow::Result<()>;
        fn layer_norm_backward(&self,s:NormSpec,w:NormWeights<'_,Vec<f32>>,g:NormGradients<'_,Vec<f32>>,cache:&Self::LayerNormCache,dy:&Vec<f32>,dx:&mut Vec<f32>)->anyhow::Result<()>;
        fn token_position_embedding_forward(&self,s:EmbeddingSpec,t:&Vec<usize>,w:&Vec<f32>,p:&Vec<f32>,out:&mut Vec<f32>)->anyhow::Result<()>;
        fn token_position_embedding_backward(&self,s:EmbeddingSpec,t:&Vec<usize>,dy:&Vec<f32>,dw:&mut Vec<f32>,dp:&mut Vec<f32>)->anyhow::Result<()>;
        fn cross_entropy(&self,x:&Vec<f32>,t:&Vec<usize>,rows:usize,vocab:usize)->anyhow::Result<f32>;
        fn cross_entropy_with_grad(&self,x:&Vec<f32>,t:&Vec<usize>,rows:usize,vocab:usize,g:&mut Vec<f32>)->anyhow::Result<f32>;
        fn adamw_step(&self,g:&mut [AdamWGroup<'_,Vec<f32>>],c:AdamWConfig)->anyhow::Result<()>;
    }
    fn buffer_to_vec(&self, _: &Vec<f32>) -> anyhow::Result<Vec<f32>> {
        panic!("unexpected host download")
    }
    fn indices_from_slice(&self, v: &[usize]) -> anyhow::Result<Vec<usize>> {
        self.uploads.set(self.uploads.get() + 1);
        Naive.indices_from_slice(v)
    }
    fn buffers_all_finite(&self, v: &[&Vec<f32>]) -> anyhow::Result<bool> {
        self.validations.set(self.validations.get() + 1);
        Naive.buffers_all_finite(v)
    }
    fn sample_last_token(
        &self,
        x: &Vec<f32>,
        rows: usize,
        vocab: usize,
        u: f32,
    ) -> anyhow::Result<usize> {
        self.samples.set(self.samples.get() + 1);
        Naive.sample_last_token(x, rows, vocab, u)
    }
    fn block_forward(
        &self,
        s: BlockSpec,
        w: BlockWeights<'_, Vec<f32>>,
        x: &Vec<f32>,
        m: ForwardMode,
        ws: &mut Self::Workspace,
        out: &mut Vec<f32>,
        cache: Option<&mut Self::BlockCache>,
    ) -> anyhow::Result<()> {
        self.forward.set(self.forward.get() + 1);
        self.saved
            .set(self.saved.get() + usize::from(cache.is_some()));
        Naive.block_forward(s, w, x, m, ws, out, cache)
    }
    fn block_backward(
        &self,
        s: BlockSpec,
        w: BlockWeights<'_, Vec<f32>>,
        g: BlockGradients<'_, Vec<f32>>,
        cache: &Self::BlockCache,
        dy: &Vec<f32>,
        ws: &mut Self::Workspace,
        dx: &mut Vec<f32>,
    ) -> anyhow::Result<()> {
        self.backward.set(self.backward.get() + 1);
        Naive.block_backward(s, w, g, cache, dy, ws, dx)
    }
    fn layer_norm_forward(
        &self,
        s: NormSpec,
        w: NormWeights<'_, Vec<f32>>,
        x: &Vec<f32>,
        out: &mut Vec<f32>,
        cache: Option<&mut Self::LayerNormCache>,
    ) -> anyhow::Result<()> {
        self.norm_saved
            .set(self.norm_saved.get() + usize::from(cache.is_some()));
        Naive.layer_norm_forward(s, w, x, out, cache)
    }
}

#[test]
fn execution_boundaries_avoid_downloads_and_inference_caches() {
    let en = Tracked::default();
    let mut gpt = model(&en);
    let mut optimizer = AdamW::new(
        &en,
        &gpt.tokenizer.clone(),
        &"abc".repeat(100),
        &mut gpt,
        ModelConfig::micro(),
        false,
    )
    .unwrap();
    gpt.loss_and_backward(&en, &[0, 1], &[1, 2], 1, 2, &mut rng())
        .unwrap();
    assert_eq!(
        (
            en.forward.get(),
            en.backward.get(),
            en.saved.get(),
            en.norm_saved.get()
        ),
        (2, 2, 2, 1)
    );
    assert_eq!(en.uploads.get(), 2);
    optimizer.step(&en, &mut gpt.parameters_mut()).unwrap();
    assert_eq!(en.validations.get(), 1);
    let tokens = gpt.generate(&en, &[], 3, &mut rng()).unwrap();
    assert_eq!(tokens.len(), 4);
    assert_eq!(en.samples.get(), 3);
    assert_eq!(en.forward.get(), 8);
    assert_eq!((en.saved.get(), en.norm_saved.get()), (2, 1));
}
