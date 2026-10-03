use alloc::vec::Vec;

use crate::engine::{BlockSpec, NormSpec};

#[derive(Default)]
pub struct NaiveLayerNormCache {
    pub spec: Option<NormSpec>,
    pub normalized: Vec<f32>,
    pub inverse_std: Vec<f32>,
}

#[derive(Default)]
pub struct NaiveBlockCache {
    pub spec: Option<BlockSpec>,
    pub norm1: NaiveLayerNormCache,
    pub norm2: NaiveLayerNormCache,
    pub normalized1: Vec<f32>,
    pub qkv: Vec<f32>,
    pub context: Vec<f32>,
    pub normalized2: Vec<f32>,
    pub activated: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub attention_mask: Vec<f32>,
    pub projection_mask: Vec<f32>,
    pub mlp_mask: Vec<f32>,
}

#[derive(Default)]
pub struct NaiveWorkspace {
    pub normalized1: Vec<f32>,
    pub qkv: Vec<f32>,
    pub context: Vec<f32>,
    pub after_attention: Vec<f32>,
    pub normalized2: Vec<f32>,
    pub activated: Vec<f32>,
    pub projected: Vec<f32>,
    pub scores: Vec<f32>,
    pub d0: Vec<f32>,
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub d_hidden: Vec<f32>,
    pub d_qkv: Vec<f32>,
}
