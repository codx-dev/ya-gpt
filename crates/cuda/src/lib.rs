#![doc = include_str!("../README.md")]

mod blocks;
pub mod engine;
pub mod kernels;
mod types;
mod validation;

#[cfg(not(target_pointer_width = "64"))]
compile_error!("ya-gpt-cuda requires a 64-bit target");

pub use engine::CudaEngine;
pub use types::{CudaBlockCache, CudaBuffer, CudaIndexBuffer, CudaLayerNormCache, CudaWorkspace};

#[cfg(test)]
mod tests;
