use core::fmt;

use alloc::{string::String, vec::Vec};

use crate::{engine::Engine, model::gpt::Gpt};

mod attributes;
mod header;
mod v0;
mod v1;

pub use header::{ModelHeader, RichModelHeader};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ModelVersion {
    V0,
    #[default]
    V1,
}

impl fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelVersion::V0 => write!(f, "0"),
            ModelVersion::V1 => write!(f, "1"),
        }
    }
}

impl Gpt<()> {
    pub fn header_from_bytes(mut bytes: &[u8]) -> anyhow::Result<ModelHeader> {
        Ok(msgpacker::unpack_from_buf(&mut bytes)?)
    }

    pub fn version_from_bytes(bytes: &[u8]) -> anyhow::Result<String> {
        Self::header_from_bytes(bytes).map(|h| h.version)
    }

    pub fn rich_header_from_bytes(bytes: &[u8]) -> anyhow::Result<RichModelHeader> {
        let version = Self::version_from_bytes(bytes)?;

        match version.as_str() {
            "0" => Self::rich_header_from_bytes_v0(bytes),
            "1" => Self::rich_header_from_bytes_v1(bytes),
            _ => anyhow::bail!("unknown version {version}"),
        }
    }
}

impl<EN: Engine> Gpt<EN> {
    pub fn to_bytes_with_header(&self, en: &EN, header: &ModelHeader) -> anyhow::Result<Vec<u8>> {
        self.to_bytes_with_header_v1(en, header)
    }

    pub fn to_bytes(&self, en: &EN) -> anyhow::Result<Vec<u8>> {
        self.to_bytes_v1(en)
    }

    pub fn to_bytes_with_author(&self, en: &EN, author: &str) -> anyhow::Result<Vec<u8>> {
        self.to_bytes_with_author_v1(en, author)
    }

    pub fn try_from_bytes(en: &EN, mut bytes: &[u8]) -> anyhow::Result<Self> {
        let cursor = &mut bytes;
        let header: ModelHeader = msgpacker::unpack_from_buf(cursor)?;

        match header.version.as_str() {
            "0" => Self::try_from_bytes_v0(en, cursor, &header),
            "1" => Self::try_from_bytes_v1(en, cursor, &header),
            _ => anyhow::bail!("unknown version {}", header.version),
        }
    }
}
