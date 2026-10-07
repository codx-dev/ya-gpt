use alloc::string::{String, ToString as _};
use msgpacker::MsgPacker;
use serde::{Deserialize, Serialize};

use crate::{
    config::ModelConfig,
    engine::Engine,
    model::{gpt::Gpt, serialize::ModelVersion},
};

#[derive(Debug, Clone, PartialEq, MsgPacker, Serialize, Deserialize)]
pub struct ModelHeader {
    pub name: String,
    pub authors: String,
    pub version: String,
    pub fp: String,
    pub params: usize,
    pub train_loss: f32,
    pub validation_loss: f32,
}

#[derive(Debug, Clone, PartialEq, MsgPacker, Serialize, Deserialize)]
pub struct RichModelHeader {
    pub name: String,
    pub authors: String,
    pub version: String,
    pub fp: String,
    pub config: ModelConfig,
    pub params: usize,
    pub train_loss: f32,
    pub validation_loss: f32,
    pub vocab_size: usize,
    pub train_tokens: usize,
}

impl ModelHeader {
    const NAME: &str = env!("CARGO_PKG_NAME");
    const FP: &str = "f32";

    pub fn from_model<EN: Engine>(en: &EN, model: &Gpt<EN>) -> Self {
        Self {
            name: Self::NAME.to_string(),
            authors: env!("CARGO_PKG_AUTHORS").to_string(),
            version: ModelVersion::default().to_string(),
            fp: Self::FP.to_string(),
            params: model.parameter_count(en),
            train_loss: model.train_loss,
            validation_loss: model.validation_loss,
        }
    }

    pub fn with_author(mut self, author: &str) -> Self {
        self.authors = author.into();
        self
    }
}
