#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockSpec {
    pub batch: usize,
    pub time: usize,
    pub channels: usize,
    pub heads: usize,
    pub dropout: f32,
}

impl BlockSpec {
    pub fn elements(&self) -> usize {
        let elements = self
            .batch
            .saturating_mul(self.time)
            .saturating_mul(self.channels);

        debug_assert!(
            elements <= (isize::MAX as usize / size_of::<f32>()),
            "tensor shape overflow"
        );

        elements
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardMode {
    pub seed: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct LinearSpec {
    pub rows: usize,
    pub inputs: usize,
    pub outputs: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormSpec {
    pub rows: usize,
    pub channels: usize,
    pub epsilon: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct EmbeddingSpec {
    pub batch: usize,
    pub time: usize,
    pub channels: usize,
    pub vocab: usize,
    pub positions: usize,
}

pub struct LinearWeights<'a, B> {
    pub weight: &'a B,
    pub bias: Option<&'a B>,
}

pub struct LinearGradients<'a, B> {
    pub weight: &'a mut B,
    pub bias: Option<&'a mut B>,
}

pub struct NormWeights<'a, B> {
    pub gamma: &'a B,
    pub beta: &'a B,
}

pub struct NormGradients<'a, B> {
    pub gamma: &'a mut B,
    pub beta: &'a mut B,
}

pub struct BlockWeights<'a, B> {
    pub norm1: NormWeights<'a, B>,
    pub qkv: &'a B,
    pub attention: LinearWeights<'a, B>,
    pub norm2: NormWeights<'a, B>,
    pub expand: LinearWeights<'a, B>,
    pub project: LinearWeights<'a, B>,
}

pub struct BlockGradients<'a, B> {
    pub norm1: NormGradients<'a, B>,
    pub qkv: &'a mut B,
    pub attention: LinearGradients<'a, B>,
    pub norm2: NormGradients<'a, B>,
    pub expand: LinearGradients<'a, B>,
    pub project: LinearGradients<'a, B>,
}

pub struct AdamWGroup<'a, B> {
    pub values: &'a mut B,
    pub gradients: &'a B,
    pub first_moment: &'a mut B,
    pub second_moment: &'a mut B,
}

#[derive(Clone, Copy, Debug)]
pub struct AdamWConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub step: i32,
}
