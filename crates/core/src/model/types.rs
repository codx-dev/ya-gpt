use rand::Rng;
use rand_distr::{Distribution as _, Normal};

use crate::{
    config::ModelConfig,
    engine::{
        BlockGradients, BlockWeights, Engine, LinearGradients, LinearSpec, LinearWeights,
        NormGradients, NormWeights,
    },
};

pub struct Parameter<EN: Engine> {
    pub values: EN::Buffer,
    pub gradients: EN::Buffer,
}

impl<EN: Engine> Parameter<EN> {
    pub fn filled(en: &EN, len: usize, value: f32) -> anyhow::Result<Self> {
        let values = en.filled(value, len)?;
        let gradients = en.zeroes(len)?;

        Ok(Self { values, gradients })
    }

    pub fn normal(en: &EN, len: usize, rng: &mut impl Rng) -> anyhow::Result<Self> {
        let normal = Normal::<f32>::new(0.0, 0.02)?;
        let values: Vec<f32> = (0..len).map(|_| normal.sample(rng)).collect();
        let values = en.buffer_from_slice(&values)?;
        let gradients = en.zeroes(len)?;

        Ok(Self { values, gradients })
    }
}

pub struct Linear<EN: Engine> {
    pub weight: Parameter<EN>,
    pub bias: Option<Parameter<EN>>,
    pub input_size: usize,
    pub output_size: usize,
}

impl<EN: Engine> Linear<EN> {
    pub fn new(
        en: &EN,
        input_size: usize,
        output_size: usize,
        bias: bool,
        rng: &mut impl Rng,
    ) -> anyhow::Result<Self> {
        let len = input_size.saturating_mul(output_size);
        let weight = Parameter::normal(en, len, rng)?;
        let bias = bias
            .then(|| Parameter::filled(en, output_size, 0.0))
            .transpose()?;

        Ok(Self {
            weight,
            bias,
            input_size,
            output_size,
        })
    }

    pub fn spec(&self, rows: usize) -> LinearSpec {
        LinearSpec {
            rows,
            inputs: self.input_size,
            outputs: self.output_size,
        }
    }

    pub fn weights(&self) -> LinearWeights<'_, EN::Buffer> {
        let weight = &self.weight.values;
        let bias = self.bias.as_ref().map(|p| &p.values);

        LinearWeights { weight, bias }
    }

    pub fn parts(
        &mut self,
    ) -> (
        LinearWeights<'_, EN::Buffer>,
        LinearGradients<'_, EN::Buffer>,
    ) {
        let (bias, d_bias) = match &mut self.bias {
            Some(p) => (Some(&p.values), Some(&mut p.gradients)),
            None => (None, None),
        };

        let weights = LinearWeights {
            weight: &self.weight.values,
            bias,
        };

        let gradients = LinearGradients {
            weight: &mut self.weight.gradients,
            bias: d_bias,
        };

        (weights, gradients)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        self.bias
            .as_ref()
            .map(|bias| vec![&self.weight, bias])
            .unwrap_or_else(|| vec![&self.weight])
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        let mut p = vec![&mut self.weight];
        p.extend(self.bias.iter_mut());
        p
    }
}

pub struct LayerNorm<EN: Engine> {
    pub gamma: Parameter<EN>,
    pub beta: Parameter<EN>,
}

impl<EN: Engine> LayerNorm<EN> {
    pub fn new(en: &EN, channels: usize) -> anyhow::Result<Self> {
        let gamma = Parameter::filled(en, channels, 1.0)?;
        let beta = Parameter::filled(en, channels, 0.0)?;

        Ok(Self { gamma, beta })
    }

    pub fn weights(&self) -> NormWeights<'_, EN::Buffer> {
        NormWeights {
            gamma: &self.gamma.values,
            beta: &self.beta.values,
        }
    }

    pub fn parts(&mut self) -> (NormWeights<'_, EN::Buffer>, NormGradients<'_, EN::Buffer>) {
        let weights = NormWeights {
            gamma: &self.gamma.values,
            beta: &self.beta.values,
        };

        let gradients = NormGradients {
            gamma: &mut self.gamma.gradients,
            beta: &mut self.beta.gradients,
        };

        (weights, gradients)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        vec![&self.gamma, &self.beta]
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        vec![&mut self.gamma, &mut self.beta]
    }
}

pub struct Block<EN: Engine> {
    pub norm1: LayerNorm<EN>,
    pub qkv: Parameter<EN>,
    pub attention: Linear<EN>,
    pub norm2: LayerNorm<EN>,
    pub expand: Linear<EN>,
    pub project: Linear<EN>,
}

impl<EN: Engine> Block<EN> {
    pub fn new(en: &EN, config: &ModelConfig, rng: &mut impl Rng) -> anyhow::Result<Self> {
        let c = config.n_embd;
        let qkv_len = c.saturating_mul(c).saturating_mul(3);

        let norm1 = LayerNorm::new(en, c)?;
        let qkv = Parameter::normal(en, qkv_len, rng)?;
        let attention = Linear::new(en, c, c, true, rng)?;
        let norm2 = LayerNorm::new(en, c)?;
        let expand = Linear::new(en, c, 4 * c, true, rng)?;
        let project = Linear::new(en, 4 * c, c, true, rng)?;

        Ok(Self {
            norm1,
            qkv,
            attention,
            norm2,
            expand,
            project,
        })
    }

    pub fn weights(&self) -> BlockWeights<'_, EN::Buffer> {
        let norm1 = self.norm1.weights();
        let qkv = &self.qkv.values;
        let attention = self.attention.weights();
        let norm2 = self.norm2.weights();
        let expand = self.expand.weights();
        let project = self.project.weights();

        BlockWeights {
            norm1,
            qkv,
            attention,
            norm2,
            expand,
            project,
        }
    }

    pub fn parts(&mut self) -> (BlockWeights<'_, EN::Buffer>, BlockGradients<'_, EN::Buffer>) {
        let (norm1, d_norm1) = self.norm1.parts();
        let (attention, d_attention) = self.attention.parts();
        let (norm2, d_norm2) = self.norm2.parts();
        let (expand, d_expand) = self.expand.parts();
        let (project, d_project) = self.project.parts();

        let weights = BlockWeights {
            norm1,
            qkv: &self.qkv.values,
            attention,
            norm2,
            expand,
            project,
        };

        let gradients = BlockGradients {
            norm1: d_norm1,
            qkv: &mut self.qkv.gradients,
            attention: d_attention,
            norm2: d_norm2,
            expand: d_expand,
            project: d_project,
        };

        (weights, gradients)
    }

    pub fn parameters(&self) -> Vec<&Parameter<EN>> {
        let mut p = self.norm1.parameters();
        p.push(&self.qkv);
        p.extend(self.attention.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.expand.parameters());
        p.extend(self.project.parameters());
        p
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<EN>> {
        let mut p = self.norm1.parameters_mut();
        p.push(&mut self.qkv);
        p.extend(self.attention.parameters_mut());
        p.extend(self.norm2.parameters_mut());
        p.extend(self.expand.parameters_mut());
        p.extend(self.project.parameters_mut());
        p
    }
}
