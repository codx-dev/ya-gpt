use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=kernels/ops.cu");
    let output =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR")).join("ops.ptx");

    let status = Command::new("nvcc")
        .args([
            "-ptx",
            "-O3",
            "--gpu-architecture=compute_89", // Baseline: Ada Lovelace + newer
            "kernels/ops.cu",
            "-o",
        ])
        .arg(output)
        .output()
        .expect("CUDA kernel compilation requires nvcc on PATH (CUDA 11.8 or newer)");

    assert!(
        status.status.success(),
        "nvcc failed to compile CUDA kernels:\n{}\n{}",
        String::from_utf8_lossy(&status.stdout),
        String::from_utf8_lossy(&status.stderr),
    );
}
