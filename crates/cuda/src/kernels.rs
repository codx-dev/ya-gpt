//! CUDA kernel loading and launch configuration.
//!
//! Kernels are compiled by nvcc at build time; NVRTC is enabled in cudarc only
//! because its safe PTX loading API is behind that feature.

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

macro_rules! kernels {
    ($($name:ident),+ $(,)?) => {
        pub(crate) struct Kernels { $(pub $name: CudaFunction,)+ }

        impl Kernels {
            pub fn load(context: &Arc<CudaContext>) -> Result<Self> {
                let module = context.load_module(Ptx::from_src(
                    include_str!(concat!(env!("OUT_DIR"), "/ops.ptx"))
                )).context("loading compute_89 PTX; check GPU and driver compatibility")?;
                Ok(Self {
                    $($name: module.load_function(stringify!($name))
                        .with_context(|| concat!("loading kernel ", stringify!($name)))?,)+
                })
            }
        }
    };
}

kernels!(
    fill,
    copy,
    add,
    bias_add,
    bias_backward,
    relu,
    relu_backward,
    norm_forward,
    norm_backward_input,
    norm_backward_weights,
    embedding_forward,
    embedding_backward,
    dropout_add,
    masked,
    attention_scores,
    attention_softmax,
    attention_context,
    attention_dprob,
    attention_dsoftmax,
    attention_dqkv,
    cross_entropy,
    reduce_loss,
    all_finite,
    adamw,
    sample,
);

pub(crate) fn grid(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256).min(65535) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

// Every kernel uses a grid-stride loop, including reductions with one logical
// worker per row. Call sites validate all lengths and pass ABI-matched scalars.
macro_rules! launch {
    ($en:expr, $name:ident, $n:expr $(, $arg:expr)* $(,)?) => {{
        let count = $n;
        if count > 0 {
            use cudarc::driver::PushKernelArg as _;
            let mut args = $en.stream.launch_builder(&$en.kernels.$name);
            $(args.arg($arg);)*
            // SAFETY: private call sites verify lengths and contexts; argument
            // order and types match kernels/ops.cu. Mutable outputs are exclusive.
            unsafe { args.launch(crate::kernels::grid(count)) }?;
        }
    }};
}

pub(crate) use launch;
