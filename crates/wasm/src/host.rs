#[cfg(not(any(target_arch = "wasm32", test)))]
use anyhow::bail;
use anyhow::{Result, ensure};

#[derive(Clone, Copy)]
#[repr(u32)]
pub(crate) enum Kernel {
    Fill = 1,
    Linear,
    Norm,
    Embedding,
    Scores,
    Softmax,
    Context,
    Relu,
    Add,
    Training,
}

#[cfg(all(target_arch = "wasm32", not(test)))]
#[link(wasm_import_module = "ya_gpu")]
unsafe extern "C" {
    fn buffer_create(bytes: u32) -> u32;
    fn buffer_upload(handle: u32, ptr: *const u8, bytes: u32) -> i32;
    fn buffer_release(handle: u32);
    fn dispatch(op: u32, args: *const u32, len: u32) -> i32;
}

pub(crate) fn create(bytes: u32) -> Result<u32> {
    #[cfg(test)]
    {
        Ok(mock::create(bytes))
    }
    #[cfg(all(target_arch = "wasm32", not(test)))]
    {
        // The host returns an opaque nonzero handle, never a Rust pointer.
        let handle = unsafe { buffer_create(bytes) };
        ensure!(handle != 0, "WebGPU buffer allocation failed");
        Ok(handle)
    }
    #[cfg(not(any(target_arch = "wasm32", test)))]
    {
        let _ = bytes;
        bail!("WebGpuEngine requires the browser WASM host")
    }
}

pub(crate) fn upload(handle: u32, words: &[u32]) -> Result<()> {
    #[cfg(test)]
    {
        mock::upload(handle, words);
        Ok(())
    }
    #[cfg(all(target_arch = "wasm32", not(test)))]
    {
        // The host copies this borrowed slice before the import returns.
        let status =
            unsafe { buffer_upload(handle, words.as_ptr().cast(), size_of_val(words) as u32) };
        ensure!(status == 0, "WebGPU buffer upload failed");
        Ok(())
    }
    #[cfg(not(any(target_arch = "wasm32", test)))]
    {
        let _ = (handle, words);
        bail!("WebGpuEngine requires the browser WASM host")
    }
}

pub(crate) fn release(handle: u32) {
    #[cfg(test)]
    mock::release(handle);
    #[cfg(all(target_arch = "wasm32", not(test)))]
    // Removal is immediate; destruction is deferred until queued work finishes.
    unsafe {
        buffer_release(handle)
    }
    #[cfg(not(any(target_arch = "wasm32", test)))]
    let _ = handle;
}

pub(crate) fn run(kernel: Kernel, args: &[u32]) -> Result<()> {
    #[cfg(test)]
    {
        mock::record(kernel as u32, args);
        Ok(())
    }
    #[cfg(all(target_arch = "wasm32", not(test)))]
    {
        // Dispatch descriptors are consumed synchronously by the host.
        let status = unsafe { dispatch(kernel as u32, args.as_ptr(), args.len() as u32) };
        ensure!(status == 0, "WebGPU kernel dispatch failed");
        Ok(())
    }
    #[cfg(not(any(target_arch = "wasm32", test)))]
    {
        let _ = (kernel, args);
        bail!("WebGpuEngine requires the browser WASM host")
    }
}

pub(crate) fn elements(dims: &[usize]) -> Result<usize> {
    let n = dims.iter().try_fold(1usize, |a, &b| a.checked_mul(b));
    ensure!(
        n.is_some_and(|n| n <= (u32::MAX as usize / 4).min(isize::MAX as usize / 4)),
        "tensor shape overflow"
    );
    Ok(n.unwrap())
}

pub(crate) fn positive(dims: &[usize]) -> Result<usize> {
    ensure!(
        dims.iter().all(|&n| n > 0),
        "tensor dimensions must be positive"
    );
    elements(dims)
}

#[cfg(test)]
pub(crate) mod mock {
    use std::{cell::RefCell, collections::BTreeMap};
    #[derive(Default)]
    struct Host {
        next: u32,
        buffers: BTreeMap<u32, Vec<u32>>,
        commands: Vec<(u32, Vec<u32>)>,
    }
    thread_local! { static HOST: RefCell<Host> = RefCell::new(Host::default()); }
    pub fn create(bytes: u32) -> u32 {
        HOST.with_borrow_mut(|h| {
            h.next += 1;
            h.buffers.insert(h.next, vec![0; bytes as usize / 4]);
            h.next
        })
    }
    pub fn upload(handle: u32, words: &[u32]) {
        HOST.with_borrow_mut(|h| {
            h.buffers.get_mut(&handle).unwrap()[..words.len()].copy_from_slice(words)
        });
    }
    pub fn release(handle: u32) {
        HOST.with_borrow_mut(|h| assert!(h.buffers.remove(&handle).is_some()));
    }
    pub fn record(op: u32, args: &[u32]) {
        HOST.with_borrow_mut(|h| h.commands.push((op, args.to_vec())));
    }
    pub fn count() -> usize {
        HOST.with_borrow(|h| h.buffers.len())
    }
    pub fn read(handle: u32, offset: usize, count: usize) -> Vec<f32> {
        HOST.with_borrow(|h| {
            h.buffers[&handle][offset..offset + count]
                .iter()
                .map(|v| f32::from_bits(*v))
                .collect()
        })
    }
    pub fn commands() -> Vec<(u32, Vec<u32>)> {
        HOST.with_borrow(|h| h.commands.clone())
    }
    pub fn reset() {
        HOST.with_borrow_mut(|h| {
            assert!(h.buffers.is_empty());
            *h = Host::default();
        });
    }
}
