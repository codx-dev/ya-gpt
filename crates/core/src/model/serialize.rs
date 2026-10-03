use alloc::{
    string::{String, ToString as _},
    vec::Vec,
};
use msgpacker::{BufMut, Encoder, MsgPacker, Packable as _};
use serde::{Deserialize, Serialize};

use crate::{
    engine::Engine,
    model::{Block, LayerNorm, Linear, Parameter, gpt::Gpt},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, MsgPacker)]
pub struct ModelHeader {
    pub name: String,
    pub authors: String,
    pub major: String,
    pub fp: String,
    pub params: usize,
    pub train_loss: f32,
    pub validation_loss: f32,
}

impl ModelHeader {
    const NAME: &str = env!("CARGO_PKG_NAME");
    const MAJOR: &str = env!("CARGO_PKG_VERSION_MAJOR");
    const FP: &str = "f32";

    pub fn from_model<EN: Engine>(en: &EN, model: &Gpt<EN>) -> Self {
        Self {
            name: Self::NAME.to_string(),
            authors: env!("CARGO_PKG_AUTHORS").to_string(),
            major: Self::MAJOR.to_string(),
            fp: Self::FP.to_string(),
            params: model.parameter_count(en),
            train_loss: model.train_loss,
            validation_loss: model.validation_loss,
        }
    }

    pub fn is_current(&self) -> bool {
        self.name == Self::NAME && self.major == Self::MAJOR && self.fp == Self::FP
    }
}

impl<EN: Engine> Gpt<EN> {
    pub fn to_bytes(&self, en: &EN) -> anyhow::Result<Vec<u8>> {
        let header = ModelHeader::from_model(en, self);

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
        } = self;

        let mut encoder_owned = Encoder::new();
        let encoder = &mut encoder_owned;

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
        train_loss.pack(encoder);
        validation_loss.pack(encoder);

        Ok(encoder_owned.into_inner())
    }

    pub fn header_from_bytes(mut bytes: &[u8]) -> anyhow::Result<ModelHeader> {
        Ok(msgpacker::unpack_from_buf(&mut bytes)?)
    }

    pub fn try_from_bytes(en: &EN, mut bytes: &[u8]) -> anyhow::Result<Self> {
        let cursor = &mut bytes;

        let header: ModelHeader = msgpacker::unpack_from_buf(cursor)?;

        anyhow::ensure!(header.is_current(), "model header mismatch");

        let config = msgpacker::unpack_from_buf(cursor)?;
        let vocab_size = msgpacker::unpack_from_buf(cursor)?;
        let tokenizer = msgpacker::unpack_from_buf(cursor)?;
        let token_embedding = Parameter::unpack_from_buf(en, cursor)?;
        let position_embedding = Parameter::unpack_from_buf(en, cursor)?;

        let blocks: u64 = msgpacker::unpack_from_buf(cursor)?;
        let blocks = (0..blocks)
            .map(|_| Block::unpack_from_buf(en, cursor))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let final_norm = LayerNorm::unpack_from_buf(en, cursor)?;
        let language_head = Linear::unpack_from_buf(en, cursor)?;
        let train_loss = msgpacker::unpack_from_buf(cursor)?;
        let validation_loss = msgpacker::unpack_from_buf(cursor)?;

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

    pub fn unpack_from_buf(en: &EN, cursor: &mut &[u8]) -> anyhow::Result<Self> {
        let values: Vec<f32> = msgpacker::unpack_from_buf(cursor)?;
        let gradients: Vec<f32> = msgpacker::unpack_from_buf(cursor)?;

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

    pub fn unpack_from_buf(en: &EN, cursor: &mut &[u8]) -> anyhow::Result<Self> {
        let gamma = Parameter::unpack_from_buf(en, cursor)?;
        let beta = Parameter::unpack_from_buf(en, cursor)?;

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

    pub fn unpack_from_buf(en: &EN, cursor: &mut &[u8]) -> anyhow::Result<Self> {
        let weight = Parameter::unpack_from_buf(en, cursor)?;
        let bias = msgpacker::unpack_from_buf::<_, bool>(cursor)?
            .then(|| Parameter::unpack_from_buf(en, cursor))
            .transpose()?;
        let input_size = msgpacker::unpack_from_buf(cursor)?;
        let output_size = msgpacker::unpack_from_buf(cursor)?;

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

    pub fn unpack_from_buf(en: &EN, cursor: &mut &[u8]) -> anyhow::Result<Self> {
        let norm1 = LayerNorm::unpack_from_buf(en, cursor)?;
        let qkv = Parameter::unpack_from_buf(en, cursor)?;
        let attention = Linear::unpack_from_buf(en, cursor)?;
        let norm2 = LayerNorm::unpack_from_buf(en, cursor)?;
        let expand = Linear::unpack_from_buf(en, cursor)?;
        let project = Linear::unpack_from_buf(en, cursor)?;

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
