use std::io::{BufReader, Read};

use anyhow::Context;
use mgpu::Extents2D;
use serde::{Deserialize, Serialize};
use texture::{TextureDescription, TextureSamplerConfiguration, TextureUsageFlags};

use crate::asset_map::{Asset, LoadContext};

use self::{material::Material, mesh::Mesh, texture::Texture};

pub mod loaders;

pub mod material;
pub mod mesh;
pub mod shader;
pub mod texture;

impl Asset for Mesh {
    type Metadata = ();
    fn asset_type_name() -> &'static str {
        "Mesh"
    }

    fn dispose(&self, device: &mgpu::Device) {
        device.destroy_buffer(self.index_buffer).unwrap();
        device.destroy_buffer(self.position_component).unwrap();
        device.destroy_buffer(self.normal_component).unwrap();
        device.destroy_buffer(self.tangent_component).unwrap();
        device.destroy_buffer(self.uv_component).unwrap();
        device.destroy_buffer(self.color_component).unwrap();
    }

    fn load(_metadata: Self::Metadata, _ctx: &LoadContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}

#[derive(Serialize, Deserialize)]
pub struct TextureMetadata {
    pub source_path: String,
    pub sampler_configuration: TextureSamplerConfiguration,
}

impl Asset for Texture {
    type Metadata = TextureMetadata;
    fn asset_type_name() -> &'static str {
        "Texture"
    }

    fn dispose(&self, device: &mgpu::Device) {
        device.destroy_image_view(self.view).unwrap();
        device.destroy_image(self.image).unwrap();
    }

    fn load(metadata: Self::Metadata, ctx: &LoadContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let mut reader = BufReader::new(
            std::fs::File::open(&metadata.source_path)
                .with_context(|| metadata.source_path.clone())?,
        );
        let mut content = vec![];
        reader.read_to_end(&mut content)?;
        let image = image::load_from_memory(&content)?;
        let image_rgba_bytes = image.to_rgba8();

        let mips = Texture::compute_num_mips(image.width(), image.height());

        Texture::new(
            ctx.device,
            &TextureDescription {
                label: Some(&metadata.source_path),
                data: &[&image_rgba_bytes],
                ty: crate::assets::texture::TextureType::D2(Extents2D {
                    width: image.width(),
                    height: image.height(),
                }),
                format: mgpu::ImageFormat::Rgba8,
                usage_flags: TextureUsageFlags::default(),
                num_mips: mips.try_into().unwrap(),
                auto_generate_mips: true,
                sampler_configuration: TextureSamplerConfiguration::default(),
            },
            ctx.sampler_allocator,
        )
    }
}
impl Asset for Material {
    type Metadata = ();
    fn asset_type_name() -> &'static str {
        "Material"
    }

    fn dispose(&self, device: &mgpu::Device) {
        device
            .destroy_binding_set(self.binding_set.clone())
            .unwrap();
    }

    fn load(metadata: Self::Metadata, _ctx: &LoadContext) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}
