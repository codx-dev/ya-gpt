use alloc::vec::Vec;
use msgpacker::{Encoder, Packable as _};

use crate::{
    engine::Engine,
    model::{
        Block, LayerNorm, Linear, Parameter,
        gpt::Gpt,
        serialize::{ModelHeader, RichModelHeader},
    },
};

impl Gpt<()> {
    pub fn rich_header_from_bytes_v1(mut bytes: &[u8]) -> anyhow::Result<RichModelHeader> {
        let ModelHeader {
            name,
            authors,
            version,
            fp,
            params,
            train_loss,
            validation_loss,
        } = msgpacker::unpack_from_buf(&mut bytes)?;

        let config = msgpacker::unpack_from_buf(&mut bytes)?;
        let vocab_size = msgpacker::unpack_from_buf(&mut bytes)?;
        let train_tokens = msgpacker::unpack_from_buf(&mut bytes)?;

        Ok(RichModelHeader {
            name,
            authors,
            version,
            fp,
            params,
            train_loss,
            validation_loss,
            config,
            vocab_size,
            train_tokens,
        })
    }
}

impl<EN: Engine> Gpt<EN> {
    pub fn to_bytes_with_header_v1(
        &self,
        en: &EN,
        header: &ModelHeader,
    ) -> anyhow::Result<Vec<u8>> {
        let Self {
            config,
            vocab_size,
            tokenizer,
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            language_head,
            train_loss,
            validation_loss,
            train_tokens,
        } = self;

        let mut encoder_owned = Encoder::new();
        let encoder = &mut encoder_owned;

        header.pack(encoder);
        config.pack(encoder);
        vocab_size.pack(encoder);
        train_tokens.pack(encoder);
        train_loss.pack(encoder);
        validation_loss.pack(encoder);
        tokenizer.pack(encoder);
        token_embedding.pack_to_buffer(en, encoder)?;
        position_embedding.pack_to_buffer(en, encoder)?;

        (blocks.len() as u64).pack(encoder);

        for block in blocks {
            block.pack_to_buffer(en, encoder)?;
        }

        final_norm.pack_to_buffer(en, encoder)?;
        language_head.pack_to_buffer(en, encoder)?;

        Ok(encoder_owned.into_inner())
    }

    pub fn to_bytes_v1(&self, en: &EN) -> anyhow::Result<Vec<u8>> {
        let header = ModelHeader::from_model(en, self);

        self.to_bytes_with_header_v1(en, &header)
    }

    pub fn to_bytes_with_author_v1(&self, en: &EN, author: &str) -> anyhow::Result<Vec<u8>> {
        let header = ModelHeader::from_model(en, self).with_author(author);

        self.to_bytes_with_header_v1(en, &header)
    }

    pub fn try_from_bytes_v1(
        en: &EN,
        cursor: &mut &[u8],
        _header: &ModelHeader,
    ) -> anyhow::Result<Self> {
        let config = msgpacker::unpack_from_buf(cursor)?;
        let vocab_size = msgpacker::unpack_from_buf(cursor)?;
        let train_tokens = msgpacker::unpack_from_buf(cursor)?;
        let train_loss = msgpacker::unpack_from_buf(cursor)?;
        let validation_loss = msgpacker::unpack_from_buf(cursor)?;
        let tokenizer = msgpacker::unpack_from_buf(cursor)?;
        let token_embedding = Parameter::unpack_from_buf(en, cursor)?;
        let position_embedding = Parameter::unpack_from_buf(en, cursor)?;

        let blocks: u64 = msgpacker::unpack_from_buf(cursor)?;
        let blocks = (0..blocks)
            .map(|_| Block::unpack_from_buf(en, cursor))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let final_norm = LayerNorm::unpack_from_buf(en, cursor)?;
        let language_head = Linear::unpack_from_buf(en, cursor)?;

        Ok(Self {
            config,
            vocab_size,
            tokenizer,
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            language_head,
            train_loss,
            validation_loss,
            train_tokens,
        })
    }
}
