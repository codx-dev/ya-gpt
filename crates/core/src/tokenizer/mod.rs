use alloc::{string::String, vec::Vec};

use msgpacker::MsgPacker;

mod charstream;

pub use charstream::CharTokenizer;

#[derive(Debug, Clone, PartialEq, Eq, MsgPacker)]
pub enum Tokenizer {
    CharStream(CharTokenizer),
}

impl Tokenizer {
    pub fn char_stream(input: &str) -> Self {
        Self::CharStream(CharTokenizer::new(input))
    }

    pub fn as_char_stream(&self) -> Option<&CharTokenizer> {
        match self {
            Tokenizer::CharStream(t) => Some(t),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::CharStream(t) => t.vocab_size(),
        }
    }

    pub fn encode(&self, input: &str) -> Vec<usize> {
        match self {
            Tokenizer::CharStream(t) => t.encode(input),
        }
    }

    pub fn decode(&self, input: &[usize]) -> String {
        match self {
            Tokenizer::CharStream(t) => t.decode(input),
        }
    }
}
