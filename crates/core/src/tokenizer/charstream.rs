use core::iter;

use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::String,
    vec::Vec,
};
use msgpacker::MsgPacker;

#[derive(Debug, Clone, PartialEq, Eq, MsgPacker)]
pub struct CharTokenizer {
    pub chars: Vec<char>,
    pub char_to_idx: BTreeMap<char, usize>,
}

impl CharTokenizer {
    /// Unknown token.
    pub const UNK: char = char::MIN;

    pub fn new(input: &str) -> Self {
        let chars: BTreeSet<char> = iter::once(Self::UNK).chain(input.chars()).collect();
        let chars: Vec<_> = chars.into_iter().collect();
        let char_to_idx = chars
            .iter()
            .copied()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();

        Self { chars, char_to_idx }
    }

    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    pub fn encode(&self, input: &str) -> Vec<usize> {
        // TODO very naive/simple per char implementation
        input
            .chars()
            .map(|c| self.char_to_idx.get(&c).copied().unwrap_or(0))
            .collect()
    }

    pub fn decode(&self, input: &[usize]) -> String {
        input
            .iter()
            .map(|i| self.chars.get(*i).copied().unwrap_or(Self::UNK))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use crate::tokenizer::CharTokenizer;

    const INPUT: &str = include_str!("../../../../data/shakespeare.txt");

    #[test]
    fn alphabet_is_consistent() {
        let alphabet = CharTokenizer::new(INPUT).chars().to_vec();

        assert_eq!(66, alphabet.len());
        assert_eq!(
            "\0\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            alphabet.into_iter().collect::<String>()
        );
    }

    #[test]
    fn encode_decode_is_consistent() {
        let input = "Confess yourselves wondrous malicious,";
        let tokenizer = CharTokenizer::new(INPUT);
        let encoded = tokenizer.encode(input);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(input, decoded);
    }
}
