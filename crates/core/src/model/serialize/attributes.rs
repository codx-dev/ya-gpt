use alloc::vec::Vec;
use msgpacker::{BufMut, Packable as _};

use crate::{
    engine::Engine,
    model::{Block, LayerNorm, Linear, Parameter},
};

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
