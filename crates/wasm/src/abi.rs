//! Private ABI v1. Allocation handles are linear-memory addresses of owned u32
//! arrays. All input ranges must belong to an active allocation. No pointer
//! passed by JS is dereferenced until it is found in the allocation registry.
use std::{cell::RefCell, collections::BTreeMap};

use anyhow::{Context, Result, ensure};
use ya_gpt::{engine::naive::Naive, model::gpt::Gpt};

use crate::session::Session;

#[derive(Default)]
struct State {
    allocations: BTreeMap<u32, Box<[u32]>>,
    session: Option<Session>,
    training: Option<crate::training::Training>,
    serialized: Vec<u8>,
    error: String,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

impl State {
    fn bytes(&self, ptr: u32, len: u32) -> Result<&[u8]> {
        let words = self
            .allocations
            .get(&ptr)
            .context("unknown WASM allocation")?;
        ensure!(
            len as usize <= size_of_val(words.as_ref()),
            "WASM allocation bounds exceeded"
        );
        // The allocation is initialized and alive, and every u8 bit pattern is valid.
        Ok(unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), len as usize) })
    }
}

fn status(f: impl FnOnce(&mut State) -> Result<i32>) -> i32 {
    STATE.with_borrow_mut(|state| match f(state) {
        Ok(value) => {
            state.error.clear();
            value
        }
        Err(error) => {
            state.error = format!("{error:#}");
            -1
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn version() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(bytes: u32) -> u32 {
    STATE.with_borrow_mut(|s| {
        let len = (bytes as usize).max(1).div_ceil(4);
        let mut words = Vec::new();
        if words.try_reserve_exact(len).is_err() {
            s.error = "WASM allocation failed".into();
            return 0;
        }
        words.resize(len, 0u32);
        let allocation = words.into_boxed_slice();
        let ptr = allocation.as_ptr() as u32;
        s.allocations.insert(ptr, allocation);
        ptr
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dealloc(ptr: u32) -> i32 {
    status(|s| {
        ensure!(
            s.allocations.remove(&ptr).is_some(),
            "unknown WASM allocation"
        );
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn generation_begin(
    model_ptr: u32,
    model_len: u32,
    prompt_ptr: u32,
    prompt_len: u32,
    count: u32,
    seed_lo: u32,
    seed_hi: u32,
) -> i32 {
    status(|s| {
        ensure!(
            s.session.is_none() && s.training.is_none(),
            "a compute session is already active"
        );
        let model = s.bytes(model_ptr, model_len)?;
        let prompt = std::str::from_utf8(s.bytes(prompt_ptr, prompt_len)?)?;
        s.session = Some(Session::new(
            model,
            prompt,
            count,
            u64::from(seed_lo) | (u64::from(seed_hi) << 32),
        )?);
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn generation_forward(descriptor_ptr: u32) -> i32 {
    status(|s| {
        s.bytes(descriptor_ptr, 12)?;
        let result = s
            .session
            .as_mut()
            .context("no generation session")?
            .forward()?;
        s.allocations.get_mut(&descriptor_ptr).unwrap()[..3].copy_from_slice(&result);
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn generation_accept_logits(ptr: u32, count: u32) -> i32 {
    status(|s| {
        let len = count.checked_mul(4).context("readback length overflow")?;
        let values: Vec<_> = s
            .bytes(ptr, len)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(i32::from(
            s.session
                .as_mut()
                .context("no generation session")?
                .accept(&values)?,
        ))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn generation_result_ptr() -> u32 {
    STATE.with_borrow(|s| s.session.as_ref().map_or(0, |s| s.text.as_ptr() as u32))
}

#[unsafe(no_mangle)]
pub extern "C" fn generation_result_len() -> u32 {
    STATE.with_borrow(|s| s.session.as_ref().map_or(0, |s| s.text.len() as u32))
}

#[unsafe(no_mangle)]
pub extern "C" fn generation_end() -> i32 {
    status(|s| {
        s.session = None;
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn error_ptr() -> u32 {
    STATE.with_borrow(|s| s.error.as_ptr() as u32)
}
#[unsafe(no_mangle)]
pub extern "C" fn error_len() -> u32 {
    STATE.with_borrow(|s| s.error.len() as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn training_version() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn training_begin(text_ptr: u32, text_len: u32, options_ptr: u32) -> i32 {
    status(|s| {
        ensure!(
            s.session.is_none() && s.training.is_none(),
            "a compute session is already active"
        );
        let options: Vec<u32> = s
            .bytes(options_ptr, 48)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let text = std::str::from_utf8(s.bytes(text_ptr, text_len)?)?;
        s.training = Some(crate::training::Training::new(
            text,
            options.as_slice().try_into()?,
        )?);
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn training_advance(ptr: u32) -> i32 {
    status(|s| {
        s.bytes(ptr, 32)?;
        let result = s
            .training
            .as_mut()
            .context("no training session")?
            .advance()?;
        s.allocations.get_mut(&ptr).unwrap()[..8].copy_from_slice(&result);
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn training_accept(ptr: u32, count: u32) -> i32 {
    status(|s| {
        let values: Vec<_> = s
            .bytes(ptr, count.checked_mul(4).context("readback overflow")?)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        s.training
            .as_mut()
            .context("no training session")?
            .accept(&values)?;
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn training_stop() -> i32 {
    status(|s| {
        s.training.as_mut().context("no training session")?.stop();
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn training_result_ptr() -> u32 {
    STATE.with_borrow(|s| s.training.as_ref().map_or(0, |t| t.bytes.as_ptr() as u32))
}

#[unsafe(no_mangle)]
pub extern "C" fn training_result_len() -> u32 {
    STATE.with_borrow(|s| s.training.as_ref().map_or(0, |t| t.bytes.len() as u32))
}

#[unsafe(no_mangle)]
pub extern "C" fn training_end() -> i32 {
    status(|s| {
        s.training = None;
        Ok(0)
    })
}

/// Reserialize a completed model without creating GPU buffers or a compute session.
#[unsafe(no_mangle)]
pub extern "C" fn model_serialize(
    model_ptr: u32,
    model_len: u32,
    author_ptr: u32,
    author_len: u32,
) -> i32 {
    status(|s| {
        s.serialized = Vec::new();
        ensure!(
            s.session.is_none() && s.training.is_none(),
            "a compute session is already active"
        );
        ensure!(
            model_len > 0 && model_len <= i32::MAX as u32,
            "invalid model file size"
        );
        // Keep exported metadata readable by the frontend's bounded inspector.
        ensure!(
            author_len <= 64 * 1024,
            "author exceeds the 64 KiB UTF-8 limit"
        );
        let author = std::str::from_utf8(s.bytes(author_ptr, author_len)?)?;
        let model = Gpt::try_from_bytes(&Naive, s.bytes(model_ptr, model_len)?)
            .context("invalid model file")?;
        s.serialized = if author.is_empty() {
            model.to_bytes(&Naive)?
        } else {
            model.to_bytes_with_author(&Naive, author)?
        };
        Ok(0)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn model_serialize_result_ptr() -> u32 {
    STATE.with_borrow(|s| {
        if s.serialized.is_empty() {
            0
        } else {
            s.serialized.as_ptr() as u32
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn model_serialize_result_len() -> u32 {
    STATE.with_borrow(|s| s.serialized.len() as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn model_serialize_end() -> i32 {
    status(|s| {
        s.serialized = Vec::new();
        Ok(0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_allocations_and_generation_order_return_errors() {
        let ptr = alloc(12);
        assert_ne!(ptr, 0);
        assert_eq!(ptr % 4, 0);
        assert_eq!(generation_forward(ptr), -1);
        assert!(error_len() > 0);
        assert_eq!(generation_accept_logits(ptr, u32::MAX), -1);
        assert_eq!(generation_begin(0, 3, ptr, 0, 1, 0, 0), -1);
        assert_eq!(dealloc(ptr), 0);
        assert_eq!(dealloc(ptr), -1);
        assert_eq!(generation_end(), 0);
        assert_eq!(generation_end(), 0);
    }

    fn write(bytes: &[u8]) -> u32 {
        let ptr = alloc(bytes.len() as u32);
        STATE.with_borrow_mut(|s| {
            for (word, chunk) in s
                .allocations
                .get_mut(&ptr)
                .unwrap()
                .iter_mut()
                .zip(bytes.chunks(4))
            {
                let mut padded = [0; 4];
                padded[..chunk.len()].copy_from_slice(chunk);
                *word = u32::from_le_bytes(padded);
            }
        });
        ptr
    }

    #[test]
    fn serialization_preserves_model_and_cleans_up_after_invalid_authors() {
        let original = include_bytes!("../../../frontend/tests/fixtures/training-dropout.bin");
        let model = Gpt::try_from_bytes(&Naive, original).unwrap();
        let defaults = model.to_bytes(&Naive).unwrap();
        let model_ptr = write(original);
        for author in ["Ada", "Zoë 🦀", "  ", ""] {
            let author_ptr = write(author.as_bytes());
            assert_eq!(
                model_serialize(
                    model_ptr,
                    original.len() as u32,
                    author_ptr,
                    author.len() as u32
                ),
                0
            );
            let bytes = STATE.with_borrow(|s| s.serialized.clone());
            let expected = if author.is_empty() {
                defaults.clone()
            } else {
                model.to_bytes_with_author(&Naive, author).unwrap()
            };
            assert_eq!(bytes, expected);
            assert_eq!(
                Gpt::try_from_bytes(&Naive, &bytes)
                    .unwrap()
                    .to_bytes(&Naive)
                    .unwrap(),
                defaults
            );
            assert_eq!(dealloc(author_ptr), 0);
        }
        let invalid = write(&[0xff]);
        assert_eq!(
            model_serialize(model_ptr, original.len() as u32, invalid, 1),
            -1
        );
        assert_eq!(model_serialize_result_len(), 0);
        assert_eq!(
            model_serialize(model_ptr, original.len() as u32, invalid, 65_537),
            -1
        );
        assert_eq!(model_serialize(model_ptr, original.len() as u32, 0, 0), -1);
        let empty = write(&[]);
        assert_eq!(
            model_serialize(model_ptr, original.len() as u32, empty, 0),
            0
        );
        assert!(model_serialize_result_len() > 0);
        assert_eq!(model_serialize_end(), 0);
        assert_eq!(model_serialize_end(), 0);
        assert_eq!(model_serialize_result_ptr(), 0);
        for ptr in [model_ptr, invalid, empty] {
            assert_eq!(dealloc(ptr), 0);
        }
        STATE.with_borrow(|s| assert!(s.allocations.is_empty()));
    }
}
