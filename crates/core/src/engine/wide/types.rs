use alloc::vec::Vec;

use crate::engine::{BlockSpec, NormSpec};

/// Saved normalization values needed by the backward pass.
#[derive(Default)]
pub struct SimdWideLayerNormCache {
    pub(super) spec: Option<NormSpec>,
    pub(super) normalized: Vec<f32>,
    pub(super) inverse_std: Vec<f32>,
}

/// Saved activations and dropout masks, independent of the reusable workspace.
#[derive(Default)]
pub struct SimdWideBlockCache {
    pub(super) spec: Option<BlockSpec>,
    pub(super) norm1: SimdWideLayerNormCache,
    pub(super) norm2: SimdWideLayerNormCache,
    pub(super) normalized1: Vec<f32>,
    pub(super) qkv: Vec<f32>,
    pub(super) context: Vec<f32>,
    pub(super) normalized2: Vec<f32>,
    pub(super) activated: Vec<f32>,
    pub(super) probabilities: Vec<f32>,
    pub(super) attention_mask: Vec<f32>,
    pub(super) projection_mask: Vec<f32>,
    pub(super) mlp_mask: Vec<f32>,
}

/// Scratch buffers reused across blocks, shapes, and forward/backward calls.
#[derive(Default)]
pub struct SimdWideWorkspace {
    pub(super) normalized1: Vec<f32>,
    pub(super) qkv: Vec<f32>,
    pub(super) context: Vec<f32>,
    pub(super) after_attention: Vec<f32>,
    pub(super) normalized2: Vec<f32>,
    pub(super) activated: Vec<f32>,
    pub(super) projected: Vec<f32>,
    pub(super) scores: Vec<f32>,
    pub(super) d0: Vec<f32>,
    pub(super) d1: Vec<f32>,
    pub(super) d2: Vec<f32>,
    pub(super) d_hidden: Vec<f32>,
    pub(super) d_qkv: Vec<f32>,
}
