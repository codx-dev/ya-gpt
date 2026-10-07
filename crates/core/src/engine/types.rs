use alloc::vec::Vec;

use crate::engine::Engine;

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

impl Engine for () {
    type Buffer = ();
    type IndexBuffer = ();
    type BlockCache = ();
    type LayerNormCache = ();
    type Workspace = ();

    fn zeroes(&self, _len: usize) -> anyhow::Result<Self::Buffer> {
        anyhow::bail!("unit is ghost impl")
    }

    fn filled(&self, _value: f32, _len: usize) -> anyhow::Result<Self::Buffer> {
        anyhow::bail!("unit is ghost impl")
    }

    fn buffer_len(&self, _buffer: &Self::Buffer) -> usize {
        0
    }

    fn buffer_from_slice(&self, _values: &[f32]) -> anyhow::Result<Self::Buffer> {
        anyhow::bail!("unit is ghost impl")
    }

    fn buffer_to_vec(&self, _buffer: &Self::Buffer) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("unit is ghost impl")
    }

    fn indices_from_slice(&self, _values: &[usize]) -> anyhow::Result<Self::IndexBuffer> {
        anyhow::bail!("unit is ghost impl")
    }

    fn synchronize(&self) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn block_forward(
        &self,
        _spec: BlockSpec,
        _weights: BlockWeights<'_, Self::Buffer>,
        _input: &Self::Buffer,
        _mode: ForwardMode,
        _ws: &mut Self::Workspace,
        _output: &mut Self::Buffer,
        _cache: Option<&mut Self::BlockCache>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn block_backward(
        &self,
        _spec: BlockSpec,
        _weights: BlockWeights<'_, Self::Buffer>,
        _gradients: BlockGradients<'_, Self::Buffer>,
        _cache: &Self::BlockCache,
        _d_output: &Self::Buffer,
        _ws: &mut Self::Workspace,
        _d_input: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn linear_forward(
        &self,
        _spec: LinearSpec,
        _weights: LinearWeights<'_, Self::Buffer>,
        _input: &Self::Buffer,
        _output: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn linear_backward(
        &self,
        _spec: LinearSpec,
        _weights: LinearWeights<'_, Self::Buffer>,
        _gradients: LinearGradients<'_, Self::Buffer>,
        _input: &Self::Buffer,
        _d_output: &Self::Buffer,
        _d_input: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn layer_norm_forward(
        &self,
        _spec: NormSpec,
        _weights: NormWeights<'_, Self::Buffer>,
        _input: &Self::Buffer,
        _output: &mut Self::Buffer,
        _cache: Option<&mut Self::LayerNormCache>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn layer_norm_backward(
        &self,
        _spec: NormSpec,
        _weights: NormWeights<'_, Self::Buffer>,
        _gradients: NormGradients<'_, Self::Buffer>,
        _cache: &Self::LayerNormCache,
        _d_output: &Self::Buffer,
        _d_input: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn token_position_embedding_forward(
        &self,
        _spec: EmbeddingSpec,
        _tokens: &Self::IndexBuffer,
        _token_table: &Self::Buffer,
        _position_table: &Self::Buffer,
        _output: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn token_position_embedding_backward(
        &self,
        _spec: EmbeddingSpec,
        _tokens: &Self::IndexBuffer,
        _d_output: &Self::Buffer,
        _d_token_table: &mut Self::Buffer,
        _d_position_table: &mut Self::Buffer,
    ) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn cross_entropy(
        &self,
        _logits: &Self::Buffer,
        _targets: &Self::IndexBuffer,
        _rows: usize,
        _vocab: usize,
    ) -> anyhow::Result<f32> {
        anyhow::bail!("unit is ghost impl")
    }

    fn cross_entropy_with_grad(
        &self,
        _logits: &Self::Buffer,
        _targets: &Self::IndexBuffer,
        _rows: usize,
        _vocab: usize,
        _d_logits: &mut Self::Buffer,
    ) -> anyhow::Result<f32> {
        anyhow::bail!("unit is ghost impl")
    }

    fn buffers_all_finite(&self, _buffers: &[&Self::Buffer]) -> anyhow::Result<bool> {
        anyhow::bail!("unit is ghost impl")
    }

    fn adamw_step(
        &self,
        _groups: &mut [AdamWGroup<'_, Self::Buffer>],
        _config: AdamWConfig,
    ) -> anyhow::Result<()> {
        anyhow::bail!("unit is ghost impl")
    }

    fn sample_last_token(
        &self,
        _logits: &Self::Buffer,
        _rows: usize,
        _vocab: usize,
        _uniform: f32,
    ) -> anyhow::Result<usize> {
        anyhow::bail!("unit is ghost impl")
    }
}
