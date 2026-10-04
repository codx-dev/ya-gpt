//! Engine backed by the browser's TypeScript WebGPU host.
//!
//! GPU commands are synchronous imports. JavaScript submits and awaits their
//! results between generation steps; no Rust executor or generated JS is used.
#[cfg(any(target_arch = "wasm32", test))]
mod abi;
pub mod engine;
mod host;
#[cfg(any(target_arch = "wasm32", test))]
mod session;
#[cfg(any(target_arch = "wasm32", test))]
mod training;
mod training_engine;

pub use engine::{WebGpuBuffer, WebGpuEngine, WebGpuIndices, WebGpuWorkspace};
