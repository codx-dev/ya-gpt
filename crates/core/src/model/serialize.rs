use msgpacker::{Encoder, Packable as _, Unpackable};

use crate::{
    engine::Engine,
    model::{
        gpt::Gpt,
        types::{Attention, Block, FeedForward, Head, LayerNorm, Linear, Parameter},
    },
};

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

        let mut encoder_owner = Encoder::new();
        let encoder = &mut encoder_owner;

        config.pack(encoder);
        vocab_size.pack(encoder);
        tokenizer.pack(encoder);

        token_embedding.pack_to_encoder(en, encoder)?;
        position_embedding.pack_to_encoder(en, encoder)?;

        (blocks.len() as u64).pack(encoder);

        for b in blocks {
            b.pack_to_encoder(en, encoder)?;
        }

        final_norm.pack_to_encoder(en, encoder)?;
        language_head.pack_to_encoder(en, encoder)?;

        Ok(encoder_owner.into_inner())
    }

    pub fn try_from_bytes(en: &EN, bytes: &[u8]) -> anyhow::Result<Self> {
        let decoder = &mut Decoder { buf: bytes };

        let config = decoder.unpack()?;
        let vocab_size = decoder.unpack()?;
        let tokenizer = decoder.unpack()?;

        let token_embedding = Parameter::unpack_from_decoder(en, decoder)?;
        let position_embedding = Parameter::unpack_from_decoder(en, decoder)?;

        let blocks: u64 = decoder.unpack()?;
        let blocks = (0..blocks)
            .map(|_| Block::unpack_from_decoder(en, decoder))
            .collect::<anyhow::Result<_>>()?;

        let final_norm = LayerNorm::unpack_from_decoder(en, decoder)?;
        let language_head = Linear::unpack_from_decoder(en, decoder)?;

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

struct Decoder<'a> {
    buf: &'a [u8],
}

impl<'a> Decoder<'a> {
    fn unpack<T: Unpackable<Error = msgpacker::Error>>(&mut self) -> anyhow::Result<T> {
        let (n, t) = T::unpack_with_ofs(self.buf).map_err(|e| anyhow::anyhow!(e))?;
        self.buf = &self.buf[n..];
        Ok(t)
    }
}

impl<EN: Engine> Parameter<EN> {
    fn pack_to_encoder(&self, en: &EN, encoder: &mut Encoder) -> anyhow::Result<()> {
        let Self { values, gradients } = self;

        en.buffer_to_vec(values)?.pack(encoder);
        en.buffer_to_vec(gradients)?.pack(encoder);

        Ok(())
    }

    fn unpack_from_decoder<'a>(en: &EN, decoder: &mut Decoder<'a>) -> anyhow::Result<Self> {
        let values: Vec<f64> = decoder.unpack()?;
        let gradients: Vec<f64> = decoder.unpack()?;

        let values = en.buffer_from_slice(&values)?;
        let gradients = en.buffer_from_slice(&gradients)?;

        Ok(Self { values, gradients })
    }
}

impl<EN: Engine> LayerNorm<EN> {
    fn pack_to_encoder(&self, en: &EN, encoder: &mut Encoder) -> anyhow::Result<()> {
        let Self { gamma, beta } = self;

        gamma.pack_to_encoder(en, encoder)?;
        beta.pack_to_encoder(en, encoder)?;

        Ok(())
    }

    fn unpack_from_decoder<'a>(en: &EN, decoder: &mut Decoder<'a>) -> anyhow::Result<Self> {
        let gamma = Parameter::unpack_from_decoder(en, decoder)?;
        let beta = Parameter::unpack_from_decoder(en, decoder)?;

        Ok(Self { gamma, beta })
    }
}

impl<EN: Engine> Linear<EN> {
    fn pack_to_encoder(&self, en: &EN, encoder: &mut Encoder) -> anyhow::Result<()> {
        let Self {
            weight,
            bias,
            input_size,
            output_size,
        } = self;

        weight.pack_to_encoder(en, encoder)?;
        bias.is_some().pack(encoder);

        if let Some(bias) = bias {
            bias.pack_to_encoder(en, encoder)?;
        }

        input_size.pack(encoder);
        output_size.pack(encoder);

        Ok(())
    }

    fn unpack_from_decoder<'a>(en: &EN, decoder: &mut Decoder<'a>) -> anyhow::Result<Self> {
        let weight = Parameter::unpack_from_decoder(en, decoder)?;

        let bias: bool = decoder.unpack()?;
        let bias = bias
            .then(|| Parameter::unpack_from_decoder(en, decoder))
            .transpose()?;

        let input_size = decoder.unpack()?;
        let output_size = decoder.unpack()?;

        Ok(Self {
            weight,
            bias,
            input_size,
            output_size,
        })
    }
}

impl<EN: Engine> Head<EN> {
    fn pack_to_encoder(&self, en: &EN, encoder: &mut Encoder) -> anyhow::Result<()> {
        let Self {
            query,
            key,
            value,
            size,
            dropout,
        } = self;

        query.pack_to_encoder(en, encoder)?;
        key.pack_to_encoder(en, encoder)?;
        value.pack_to_encoder(en, encoder)?;
        size.pack(encoder);
        dropout.pack(encoder);

        Ok(())
    }

    fn unpack_from_decoder<'a>(en: &EN, decoder: &mut Decoder<'a>) -> anyhow::Result<Self> {
        let query = Linear::unpack_from_decoder(en, decoder)?;
        let key = Linear::unpack_from_decoder(en, decoder)?;
        let value = Linear::unpack_from_decoder(en, decoder)?;
        let size = decoder.unpack()?;
        let dropout = decoder.unpack()?;

        Ok(Self {
            query,
            key,
            value,
            size,
            dropout,
        })
    }
}

impl<EN: Engine> Attention<EN> {
    fn pack_to_encoder(&self, en: &EN, encoder: &mut Encoder) -> anyhow::Result<()> {
        let Self {
            heads,
            projection,
            dropout,
        } = self;

        (heads.len() as u64).pack(encoder);

        for h in heads {
            h.pack_to_encoder(en, encoder)?;
        }

        projection.pack_to_encoder(en, encoder)?;
        dropout.pack(encoder);

        Ok(())
    }

    fn unpack_from_decoder<'a>(en: &EN, decoder: &mut Decoder<'a>) -> anyhow::Result<Self> {
        let heads: u64 = decoder.unpack()?;
        let heads = (0..heads)
            .map(|_| Head::unpack_from_decoder(en, decoder))
            .collect::<anyhow::Result<_>>()?;

        let projection = Linear::unpack_from_decoder(en, decoder)?;
        let dropout = decoder.unpack()?;

        Ok(Self {
            heads,
            projection,
            dropout,
        })
    }
}

impl<EN: Engine> FeedForward<EN> {
    fn pack_to_encoder(&self, en: &EN, encoder: &mut Encoder) -> anyhow::Result<()> {
        let Self {
            expand,
            project,
            dropout,
        } = self;

        expand.pack_to_encoder(en, encoder)?;
        project.pack_to_encoder(en, encoder)?;
        dropout.pack(encoder);

        Ok(())
    }

    fn unpack_from_decoder<'a>(en: &EN, decoder: &mut Decoder<'a>) -> anyhow::Result<Self> {
        let expand = Linear::unpack_from_decoder(en, decoder)?;
        let project = Linear::unpack_from_decoder(en, decoder)?;
        let dropout = decoder.unpack()?;

        Ok(Self {
            expand,
            project,
            dropout,
        })
    }
}

impl<EN: Engine> Block<EN> {
    fn pack_to_encoder(&self, en: &EN, encoder: &mut Encoder) -> anyhow::Result<()> {
        let Self {
            norm1,
            attention,
            norm2,
            feed_forward,
        } = self;

        norm1.pack_to_encoder(en, encoder)?;
        attention.pack_to_encoder(en, encoder)?;
        norm2.pack_to_encoder(en, encoder)?;
        feed_forward.pack_to_encoder(en, encoder)?;

        Ok(())
    }

    fn unpack_from_decoder<'a>(en: &EN, decoder: &mut Decoder<'a>) -> anyhow::Result<Self> {
        let norm1 = LayerNorm::unpack_from_decoder(en, decoder)?;
        let attention = Attention::unpack_from_decoder(en, decoder)?;
        let norm2 = LayerNorm::unpack_from_decoder(en, decoder)?;
        let feed_forward = FeedForward::unpack_from_decoder(en, decoder)?;

        Ok(Self {
            norm1,
            attention,
            norm2,
            feed_forward,
        })
    }
}
