use alloc::vec::Vec;

mod types;

pub mod naive;
pub use types::*;

#[cfg(feature = "simd-wide")]
pub mod wide;

#[allow(clippy::too_many_arguments)]
pub trait Engine {
    type Buffer;
    type IndexBuffer;
    type BlockCache: Default;
    type LayerNormCache: Default;
    type Workspace: Default;

    fn zeroes(&self, len: usize) -> anyhow::Result<Self::Buffer>;
    fn filled(&self, value: f32, len: usize) -> anyhow::Result<Self::Buffer>;

    fn buffer_len(&self, buffer: &Self::Buffer) -> usize;
    fn buffer_from_slice(&self, values: &[f32]) -> anyhow::Result<Self::Buffer>;
    fn buffer_to_vec(&self, buffer: &Self::Buffer) -> anyhow::Result<Vec<f32>>;
    fn indices_from_slice(&self, values: &[usize]) -> anyhow::Result<Self::IndexBuffer>;
    fn synchronize(&self) -> anyhow::Result<()>;

    fn block_forward(
        &self,
        spec: BlockSpec,
        weights: BlockWeights<'_, Self::Buffer>,
        input: &Self::Buffer,
        mode: ForwardMode,
        ws: &mut Self::Workspace,
        output: &mut Self::Buffer,
        cache: Option<&mut Self::BlockCache>,
    ) -> anyhow::Result<()>;

    fn block_backward(
        &self,
        spec: BlockSpec,
        weights: BlockWeights<'_, Self::Buffer>,
        gradients: BlockGradients<'_, Self::Buffer>,
        cache: &Self::BlockCache,
        d_output: &Self::Buffer,
        ws: &mut Self::Workspace,
        d_input: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn linear_forward(
        &self,
        spec: LinearSpec,
        weights: LinearWeights<'_, Self::Buffer>,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn linear_backward(
        &self,
        spec: LinearSpec,
        weights: LinearWeights<'_, Self::Buffer>,
        gradients: LinearGradients<'_, Self::Buffer>,
        input: &Self::Buffer,
        d_output: &Self::Buffer,
        d_input: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn layer_norm_forward(
        &self,
        spec: NormSpec,
        weights: NormWeights<'_, Self::Buffer>,
        input: &Self::Buffer,
        output: &mut Self::Buffer,
        cache: Option<&mut Self::LayerNormCache>,
    ) -> anyhow::Result<()>;

    fn layer_norm_backward(
        &self,
        spec: NormSpec,
        weights: NormWeights<'_, Self::Buffer>,
        gradients: NormGradients<'_, Self::Buffer>,
        cache: &Self::LayerNormCache,
        d_output: &Self::Buffer,
        d_input: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn token_position_embedding_forward(
        &self,
        spec: EmbeddingSpec,
        tokens: &Self::IndexBuffer,
        token_table: &Self::Buffer,
        position_table: &Self::Buffer,
        output: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn token_position_embedding_backward(
        &self,
        spec: EmbeddingSpec,
        tokens: &Self::IndexBuffer,
        d_output: &Self::Buffer,
        d_token_table: &mut Self::Buffer,
        d_position_table: &mut Self::Buffer,
    ) -> anyhow::Result<()>;

    fn cross_entropy(
        &self,
        logits: &Self::Buffer,
        targets: &Self::IndexBuffer,
        rows: usize,
        vocab: usize,
    ) -> anyhow::Result<f32>;

    fn cross_entropy_with_grad(
        &self,
        logits: &Self::Buffer,
        targets: &Self::IndexBuffer,
        rows: usize,
        vocab: usize,
        d_logits: &mut Self::Buffer,
    ) -> anyhow::Result<f32>;

    fn buffers_all_finite(&self, buffers: &[&Self::Buffer]) -> anyhow::Result<bool>;

    fn adamw_step(
        &self,
        groups: &mut [AdamWGroup<'_, Self::Buffer>],
        config: AdamWConfig,
    ) -> anyhow::Result<()>;

    fn sample_last_token(
        &self,
        logits: &Self::Buffer,
        rows: usize,
        vocab: usize,
        uniform: f32,
    ) -> anyhow::Result<usize>;
}
