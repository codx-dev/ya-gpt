use std::io;

use msgpacker::{BufMut, Encoder, MsgPacker, Packable as _, Unpackable};

use crate::{
    engine::Engine,
    model::{Block, LayerNorm, Linear, Parameter, gpt::Gpt},
};

#[derive(Debug, Clone, PartialEq, Eq, MsgPacker)]
pub struct ModelHeader {
    pub name: String,
    pub authors: String,
    pub major: String,
    pub fp: String,
}

impl ModelHeader {
    pub fn current() -> Self {
        Self {
            name: env!("CARGO_PKG_NAME").to_owned(),
            authors: env!("CARGO_PKG_AUTHORS").to_owned(),
            major: env!("CARGO_PKG_VERSION_MAJOR").to_owned(),
            fp: "f32".to_owned(),
        }
    }
}

impl<EN: Engine> Gpt<EN> {
    pub fn to_bytes(&self, en: &EN) -> anyhow::Result<Vec<u8>> {
        let Self {
            config,
            vocab_size,
            tokenizer,
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            language_head,
        } = self;

        let mut encoder_owned = Encoder::new();
        let encoder = &mut encoder_owned;
        let header = ModelHeader::current();

        header.pack(encoder);
        config.pack(encoder);
        vocab_size.pack(encoder);
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

    pub fn try_from_bytes(en: &EN, bytes: &[u8]) -> anyhow::Result<Self> {
        let cursor = &mut io::Cursor::new(bytes);

        let header: ModelHeader = unpack_from_cursor(cursor)?;

        anyhow::ensure!(header == ModelHeader::current(), "model header mismatch");

        let config = unpack_from_cursor(cursor)?;
        let vocab_size = unpack_from_cursor(cursor)?;
        let tokenizer = unpack_from_cursor(cursor)?;
        let token_embedding = Parameter::unpack_from_cursor(en, cursor)?;
        let position_embedding = Parameter::unpack_from_cursor(en, cursor)?;

        let blocks: u64 = unpack_from_cursor(cursor)?;
        let blocks = (0..blocks)
            .map(|_| Block::unpack_from_cursor(en, cursor))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let final_norm = LayerNorm::unpack_from_cursor(en, cursor)?;
        let language_head = Linear::unpack_from_cursor(en, cursor)?;

        Ok(Self {
            config,
            vocab_size,
            tokenizer,
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            language_head,
        })
    }
}

impl<EN: Engine> Parameter<EN> {
    pub fn pack_to_buffer(&self, en: &EN, encoder: &mut impl BufMut) -> anyhow::Result<()> {
        let Self { values, gradients } = self;

        en.buffer_to_vec(values)?.pack(encoder);
        en.buffer_to_vec(gradients)?.pack(encoder);

        Ok(())
    }

    pub fn unpack_from_cursor(en: &EN, cursor: &mut io::Cursor<&[u8]>) -> anyhow::Result<Self> {
        let values: Vec<f32> = unpack_from_cursor(cursor)?;
        let gradients: Vec<f32> = unpack_from_cursor(cursor)?;

        let values = en.buffer_from_slice(&values)?;
        let gradients = en.buffer_from_slice(&gradients)?;

        Ok(Self { values, gradients })
    }
}

impl<EN: Engine> LayerNorm<EN> {
    pub fn pack_to_buffer(&self, en: &EN, encoder: &mut impl BufMut) -> anyhow::Result<()> {
        let Self { gamma, beta } = self;

        gamma.pack_to_buffer(en, encoder)?;
        beta.pack_to_buffer(en, encoder)?;

        Ok(())
    }

    pub fn unpack_from_cursor(en: &EN, cursor: &mut io::Cursor<&[u8]>) -> anyhow::Result<Self> {
        let gamma = Parameter::unpack_from_cursor(en, cursor)?;
        let beta = Parameter::unpack_from_cursor(en, cursor)?;

        Ok(Self { gamma, beta })
    }
}

impl<EN: Engine> Linear<EN> {
    pub fn pack_to_buffer(&self, en: &EN, encoder: &mut impl BufMut) -> anyhow::Result<()> {
        let Self {
            weight,
            bias,
            input_size,
            output_size,
        } = self;

        weight.pack_to_buffer(en, encoder)?;

        bias.is_some().pack(encoder);

        if let Some(bias) = bias {
            bias.pack_to_buffer(en, encoder)?;
        }

        input_size.pack(encoder);
        output_size.pack(encoder);

        Ok(())
    }

    pub fn unpack_from_cursor(en: &EN, cursor: &mut io::Cursor<&[u8]>) -> anyhow::Result<Self> {
        let weight = Parameter::unpack_from_cursor(en, cursor)?;
        let bias = unpack_from_cursor::<bool>(cursor)?
            .then(|| Parameter::unpack_from_cursor(en, cursor))
            .transpose()?;
        let input_size = unpack_from_cursor(cursor)?;
        let output_size = unpack_from_cursor(cursor)?;

        Ok(Self {
            weight,
            bias,
            input_size,
            output_size,
        })
    }
}

impl<EN: Engine> Block<EN> {
    pub fn pack_to_buffer(&self, en: &EN, encoder: &mut impl BufMut) -> anyhow::Result<()> {
        let Self {
            norm1,
            qkv,
            attention,
            norm2,
            expand,
            project,
        } = self;

        norm1.pack_to_buffer(en, encoder)?;
        qkv.pack_to_buffer(en, encoder)?;
        attention.pack_to_buffer(en, encoder)?;
        norm2.pack_to_buffer(en, encoder)?;
        expand.pack_to_buffer(en, encoder)?;
        project.pack_to_buffer(en, encoder)?;

        Ok(())
    }

    pub fn unpack_from_cursor(en: &EN, cursor: &mut io::Cursor<&[u8]>) -> anyhow::Result<Self> {
        let norm1 = LayerNorm::unpack_from_cursor(en, cursor)?;
        let qkv = Parameter::unpack_from_cursor(en, cursor)?;
        let attention = Linear::unpack_from_cursor(en, cursor)?;
        let norm2 = LayerNorm::unpack_from_cursor(en, cursor)?;
        let expand = Linear::unpack_from_cursor(en, cursor)?;
        let project = Linear::unpack_from_cursor(en, cursor)?;

        Ok(Self {
            norm1,
            qkv,
            attention,
            norm2,
            expand,
            project,
        })
    }
}

fn unpack_from_cursor<T: Unpackable<Error = msgpacker::Error>>(
    cursor: &mut io::Cursor<&[u8]>,
) -> anyhow::Result<T> {
    let pos = (cursor.position() as usize).min(cursor.get_ref().len());
    let remaining = &cursor.get_ref()[pos..];
    let (n, values): (_, T) = Unpackable::unpack_with_ofs(remaining)?;

    cursor.set_position((pos + n) as u64);

    Ok(values)
}
