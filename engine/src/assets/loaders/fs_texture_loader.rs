use std::{
    io::{BufReader, Read},
    path::Path,
};

use anyhow::Context;
use mgpu::{Device, Extents2D};

use crate::{
    asset_map::AssetLoader,
    assets::texture::{SamplerConfiguration, Texture, TextureDescription, TextureUsageFlags},
    sampler_allocator::SamplerAllocator,
};

pub struct FsTextureLoader {
    device: Device,
    sampler_allocator: SamplerAllocator,
}

impl FsTextureLoader {
    pub fn new(device: Device, sampler_allocator: SamplerAllocator) -> Self {
        Self {
            device,
            sampler_allocator,
        }
    }
}

impl AssetLoader for FsTextureLoader {
    type LoadedAsset = Texture;

    fn accepts_identifier(&self, identifier: &str) -> bool {
        let extension = Path::new(identifier).extension().unwrap();
        let extension = extension.to_string_lossy();
        matches!(extension.as_ref(), "png" | "jpg" | "jpe&g")
    }

    fn load(&mut self, identifier: &str) -> anyhow::Result<Self::LoadedAsset> {
        let path = Path::new(identifier);
        let full_path = path.canonicalize().unwrap();
        let full_path = full_path.to_string_lossy().to_string();
        let mut reader = BufReader::new(std::fs::File::open(path).context(full_path)?);
        let mut content = vec![];
        reader.read_to_end(&mut content)?;
        let image = image::load_from_memory(&content)?;
        let image_rgba_bytes = image.to_rgba8();

        Texture::new(
            &self.device,
            &TextureDescription {
                label: Some(identifier),
                data: &[&image_rgba_bytes],
                ty: crate::assets::texture::TextureType::D2(Extents2D {
                    width: image.width(),
                    height: image.height(),
                }),
                format: mgpu::ImageFormat::Rgba8,
                usage_flags: TextureUsageFlags::default(),
                num_mips: 1.try_into().unwrap(),
                auto_generate_mips: false,
                sampler_configuration: SamplerConfiguration::default(),
            },
            &mut self.sampler_allocator,
        )
    }
}
