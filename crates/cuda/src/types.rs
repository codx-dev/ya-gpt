use cudarc::driver::CudaSlice;
use ya_gpt::engine::{BlockSpec, NormSpec};

/// An owned, contiguous FP32 device buffer. Transfer through the Engine methods.
pub struct CudaBuffer {
    pub(crate) data: CudaSlice<f32>,
    // Empty buffers own a one-element allocation, which is never exposed.
    pub(crate) len: usize,
}

/// Immutable device indices, with host metadata for bounds checks without downloads.
pub struct CudaIndexBuffer {
    pub(crate) data: CudaSlice<u32>,
    pub(crate) len: usize,
    pub(crate) max: Option<usize>,
}

/// Saved layer normalization values. Default does not allocate device memory.
#[derive(Default)]
pub struct CudaLayerNormCache {
    pub(crate) spec: Option<NormSpec>,
    pub(crate) normalized: Option<CudaBuffer>,
    pub(crate) inverse_std: Option<CudaBuffer>,
}

/// Saved block activations, independent of the scratch workspace.
#[derive(Default)]
pub struct CudaBlockCache {
    pub(crate) spec: Option<BlockSpec>,
    pub(crate) norm1: CudaLayerNormCache,
    pub(crate) norm2: CudaLayerNormCache,
    pub(crate) saved: Option<SavedBlock>,
}

pub(crate) struct SavedBlock {
    pub normalized1: CudaBuffer,
    pub qkv: CudaBuffer,
    pub context: CudaBuffer,
    pub normalized2: CudaBuffer,
    pub activated: CudaBuffer,
    pub probabilities: CudaBuffer,
    pub attention_mask: CudaBuffer,
    pub projection_mask: CudaBuffer,
    pub mlp_mask: CudaBuffer,
}

/// Reusable forward and backward scratch allocations, grown on demand.
#[derive(Default)]
pub struct CudaWorkspace {
    pub(crate) normalized1: Option<CudaBuffer>,
    pub(crate) qkv: Option<CudaBuffer>,
    pub(crate) context: Option<CudaBuffer>,
    pub(crate) after_attention: Option<CudaBuffer>,
    pub(crate) normalized2: Option<CudaBuffer>,
    pub(crate) activated: Option<CudaBuffer>,
    pub(crate) projected: Option<CudaBuffer>,
    pub(crate) probabilities: Option<CudaBuffer>,
    pub(crate) attention_mask: Option<CudaBuffer>,
    pub(crate) projection_mask: Option<CudaBuffer>,
    pub(crate) mlp_mask: Option<CudaBuffer>,
    pub(crate) d0: Option<CudaBuffer>,
    pub(crate) d1: Option<CudaBuffer>,
    pub(crate) d2: Option<CudaBuffer>,
    pub(crate) d_hidden: Option<CudaBuffer>,
    pub(crate) d_qkv: Option<CudaBuffer>,
    pub(crate) d_scores: Option<CudaBuffer>,
}
